"""Runtime integration checks for the isolated solver worker.

The real-worker cases intentionally use only 24 cable elements.  They still
exercise the MuJoCo process boundary and persistent artifacts while keeping
the CI runtime short enough for a regular pull request.
"""

from __future__ import annotations

import queue
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import slinky_lab.manager as manager_module
from slinky_lab.api import create_app
from slinky_lab.manager import RunManager, _worker_run
from slinky_lab.schemas import Frame, RunConfig, SweepAxis, SweepConfig
from slinky_lab.storage import DataStore, TrajectoryWriter


def tiny_config(duration: float = 0.02, sample_hz: int = 60, max_wall_seconds: float = 5) -> RunConfig:
    """Return a stable three-turn, 24-element CPU test configuration."""

    return RunConfig(
        name="runtime test",
        material={"turns": 3, "young_modulus": 1.0e8},
        scene={"settle_time": 0.05},
        numerics={
            "profile": "fine",
            "duration": duration,
            "sample_hz": sample_hz,
            "segments_per_turn": 8,
            "timestep": 1.0e-4,
            "max_wall_seconds": max_wall_seconds,
        },
    )


def wait_for_status(manager: RunManager, run_id: str, statuses: set[str], timeout: float = 30) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run = manager.get_run(run_id)
        if run is not None and run["status"] in statuses:
            return run
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {run_id}: {manager.get_run(run_id)}")


def frame_at(time_value: float, metric: float | None = None) -> Frame:
    return Frame(
        time=time_value,
        positions=[[0.0, 0.0, time_value]],
        quaternions=[[1.0, 0.0, 0.0, 0.0]],
        metrics={} if metric is None else {"test_metric": metric},
    )


def write_frames(manager: RunManager, run_id: str, times: list[float], metric: bool = False) -> None:
    with TrajectoryWriter(manager.store.run_dir(run_id) / "trajectory.h5") as writer:
        for time_value in times:
            writer.append(frame_at(time_value, time_value if metric else None))


def test_real_worker_completes_and_writes_five_artifacts(tmp_path: Path):
    manager = RunManager(tmp_path, auto_start=True, recover=False)
    try:
        run = manager.create_run(tiny_config(duration=0.02))
        final = wait_for_status(manager, run["run_id"], {"completed", "failed", "cancelled"})
        assert final["status"] == "completed", final
        expected = {"config.json", "model.xml", "summary.json", "metrics.csv", "trajectory.h5"}
        assert expected <= set(final["artifacts"])
        frame_page = manager.frames(run["run_id"], 0, 1000)
        frames, total = frame_page["frames"], frame_page["total"]
        assert total >= 2
        assert len(frames) == total
        assert all(len(frame["positions"]) == 24 for frame in frames)
    finally:
        manager.shutdown()


def test_two_queued_runs_pause_step_resume_cancel(tmp_path: Path):
    manager = RunManager(tmp_path, auto_start=True, recover=False)
    try:
        first = manager.create_run(tiny_config(duration=0.02, sample_hz=30, max_wall_seconds=10))
        second = manager.create_run(tiny_config(duration=0.08, sample_hz=30, max_wall_seconds=10))

        # The second task is usually still queued while the first one settles.
        # These DB-only transitions must not be consumed by the first worker.
        manager.command(second["run_id"], "pause")
        manager.command(second["run_id"], "resume")
        manager.command(second["run_id"], "pause")

        first_final = wait_for_status(manager, first["run_id"], {"completed", "failed", "cancelled"}, timeout=45)
        assert first_final["status"] == "completed", first_final
        paused = wait_for_status(manager, second["run_id"], {"paused", "failed", "cancelled"}, timeout=45)
        assert paused["status"] == "paused", paused

        manager.command(second["run_id"], "step")
        deadline = time.monotonic() + 10
        stepped = paused
        while time.monotonic() < deadline:
            stepped = manager.get_run(second["run_id"]) or stepped
            if stepped["status"] == "paused" and stepped["time"] > 0:
                break
            time.sleep(0.02)
        assert stepped["status"] == "paused" and stepped["time"] > 0, stepped

        manager.command(second["run_id"], "resume")
        manager.command(second["run_id"], "cancel")
        second_final = wait_for_status(manager, second["run_id"], {"completed", "failed", "cancelled"}, timeout=30)
        assert second_final["status"] == "cancelled", second_final
    finally:
        manager.shutdown()


class BrokenSimulation:
    def __init__(self, _config: RunConfig):
        self.model_xml = "<mujoco/>"

    def prepare(self, progress: Any = None, should_cancel: Any = None) -> None:
        raise RuntimeError("intentional worker failure")


class TimeoutSimulation:
    def __init__(self, _config: RunConfig):
        self.model_xml = "<mujoco/>"

    def prepare(self, progress: Any = None, should_cancel: Any = None) -> None:
        assert should_cancel is not None
        assert should_cancel()


def run_direct_worker(manager: RunManager, run_id: str, config: RunConfig) -> list[dict[str, Any]]:
    controls: queue.Queue[dict[str, Any]] = queue.Queue()
    events: queue.Queue[dict[str, Any]] = queue.Queue()
    _worker_run(
        {"run_id": run_id, "config": config.model_dump(mode="json")},
        controls,
        events,
        str(manager.store.root),
    )
    output: list[dict[str, Any]] = []
    while True:
        try:
            output.append(events.get_nowait())
        except queue.Empty:
            return output


@pytest.mark.parametrize("simulation_class, expected", [(BrokenSimulation, "intentional worker failure"), (TimeoutSimulation, "max_wall_seconds")])
def test_worker_exception_and_timeout_are_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, simulation_class: type[Any], expected: str):
    manager = RunManager(tmp_path, auto_start=False, recover=False)
    manager.start = lambda: None
    config = tiny_config(duration=0.02)
    run = manager.create_run(config)
    monkeypatch.setattr(manager_module, "_simulation_class", lambda: simulation_class)
    if simulation_class is TimeoutSimulation:
        calls = [0]

        def fake_monotonic() -> float:
            calls[0] += 1
            return 0.0 if calls[0] == 1 else 100.0

        monkeypatch.setattr(manager_module.time, "monotonic", fake_monotonic)
    events = run_direct_worker(manager, run["run_id"], config)
    for event in events:
        manager._handle_event(event)
    failed = manager.get_run(run["run_id"])
    assert failed is not None and failed["status"] == "failed"
    assert expected in (failed["error"] or "")


def test_service_recovery_marks_active_run_interrupted(tmp_path: Path):
    store = DataStore(tmp_path, recover=False)
    run = store.create_run(tiny_config())
    store.update_run(run["run_id"], status="paused")
    recovered = DataStore(tmp_path, recover=True)
    interrupted = recovered.get_run(run["run_id"])
    assert interrupted is not None
    assert interrupted["status"] == "interrupted"
    assert "restarted" in (interrupted["error"] or "")


def test_sweep_preserves_failed_combination(tmp_path: Path):
    manager = RunManager(tmp_path, auto_start=False, recover=False)
    manager.start = lambda: None
    try:
        sweep = manager.create_sweep(
            SweepConfig(
                name="failure record",
                base=tiny_config(),
                axes=[SweepAxis(parameter="material.mass", values=[0.01, 0.02])],
            )
        )
        manager.store.update_run(sweep["runs"][0]["run_id"], status="completed")
        failed_id = sweep["runs"][1]["run_id"]
        manager.store.update_run(failed_id, status="failed", error="intentional combination failure")
        result = manager.sweep(sweep["sweep_id"])
        assert result is not None and result["status"] == "failed"
        assert result["runs"][1]["status"] == "failed"
        assert result["runs"][1]["error"] == "intentional combination failure"
        assert result["runs"][1]["parameters"] == {"material.mass": 0.02}
    finally:
        manager.shutdown()


def test_websocket_stream_sends_all_frames_over_one_page(tmp_path: Path):
    manager = RunManager(tmp_path, auto_start=False, recover=False)
    manager.start = lambda: None
    run = manager.create_run(tiny_config())
    write_frames(manager, run["run_id"], [index * 0.01 for index in range(121)])
    manager.store.update_run(
        run["run_id"],
        status="completed",
        progress=1.0,
        sim_time=1.2,
        metadata={"engine_version": "test"},
    )
    app = create_app(manager)
    received_frames = 0
    received_result = False
    with TestClient(app) as client:
        with client.websocket_connect(f"/api/v1/runs/{run['run_id']}/stream") as websocket:
            for _ in range(140):
                message = websocket.receive_json()
                if message["type"] == "frame":
                    received_frames += 1
                elif message["type"] == "result":
                    received_result = True
                    break
                elif message["type"] == "error":
                    pytest.fail(message)
    assert received_frames == 121
    assert received_result


def test_reference_comparison_uses_common_time_range_and_rejects_duplicates(tmp_path: Path):
    manager = RunManager(tmp_path, auto_start=False, recover=False)
    manager.start = lambda: None
    run = manager.create_run(tiny_config())
    write_frames(manager, run["run_id"], [0.0, 1.0], metric=True)
    content = "time,value\n-1,-1\n0,0\n0.5,0.5\n1,1\n2,2\n"
    comparison = manager.add_reference(run["run_id"], "test_metric", content.encode())
    assert comparison["common_time"] == [0.0, 1.0]
    assert comparison["coverage"] == pytest.approx(0.6)
    assert comparison["rmse"] == pytest.approx(0.0)
    assert comparison["summary"]["reference_rmse"] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="duplicate"):
        manager.add_reference(run["run_id"], "test_metric", b"time,value\n0,0\n0,1\n")
