from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import scripts.validate_stairs as validate_stairs
from slinky_lab.schemas import RunConfig


class ResetClockSimulation:
    """Advance several valid samples, then fail after a clock reset."""

    def __init__(self, config: RunConfig):
        self.config = config
        self.duration = config.numerics.duration
        self.time = 0.0
        self._successful_steps = 0
        self.model_xml = "<mujoco model='fake-validation'/>"
        self.data = SimpleNamespace(qpos=np.array([0.0]), qvel=np.array([0.0]))

    def prepare(self, should_cancel=None):
        assert not should_cancel or not should_cancel()

    def step(self):
        if self._successful_steps < 4:
            self._successful_steps += 1
            self.time = 0.02 * self._successful_steps
            self.data.qpos[:] = self.time
            self.data.qvel[:] = 1.0
            return
        self.time = 0.001
        raise RuntimeError("simulated solver failure after clock reset")

    def frame(self):
        return {
            "time": self.time,
            "positions": [[self.time, 0.0, 0.0]],
            "quaternions": [[1.0, 0.0, 0.0, 0.0]],
            "contacts": [],
            "contact_details": [],
            "metrics": {"com_x": self.time, "max_penetration": 0.0},
        }

    def summary(self):
        return {
            "status": "unstable" if self._successful_steps >= 4 else "running",
            "time": self.time,
            "duration": self.duration,
            "metrics": {"com_x": self.time, "max_penetration": 0.0},
            "support_diagnostics": {
                "candidate_flip_count": 0,
                "candidate_endpoint_sequence": [],
                "confirmed_flip_count": 0,
                "confirmed_endpoint_sequence": [],
                "verified_steps": [],
                "events": [],
            },
        }

    def metadata(self):
        return {"model_version": "fake", "coordinate_system": {"world_up": "+z"}}


def test_run_case_keeps_accepted_frames_separate_from_reset_clock(monkeypatch, tmp_path: Path):
    base = RunConfig(
        scenario="stairs",
        material={"turns": 3},
        scene={"step_count": 3},
        numerics={"duration": 0.1, "sample_hz": 50, "segments_per_turn": 8, "timestep": 0.0002, "max_wall_seconds": 5},
    )
    monkeypatch.setattr(validate_stairs, "Simulation", ResetClockSimulation)
    output = tmp_path / "reset-clock"
    record = validate_stairs.run_case(
        base,
        {"name": "reset-clock", "segments_per_turn": 8, "timestep": 0.0002, "wall_seconds": 5.0},
        output,
    )

    assert record["status"] == "failed"
    assert "simulated solver failure" in record["error"]
    assert record["simulation_time"] == pytest.approx(0.08)
    assert record["engine_time_after_failure_s"] == pytest.approx(0.001)
    assert record["simulation_time"] >= record["engine_time_after_failure_s"]
    assert record["post_failure_instantaneous_metrics_valid"] is False
    assert record["frame_count"] == 5

    frames = json.loads((output / "frames.json").read_text(encoding="utf-8"))
    frame_times = [frame["time"] for frame in frames]
    assert frame_times == pytest.approx([0.0, 0.02, 0.04, 0.06, 0.08])
    assert 0.001 not in frame_times

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["last_accepted_time_s"] == pytest.approx(0.08)
    assert summary["last_exported_frame_time_s"] == pytest.approx(0.08)
    assert summary["engine_time_after_failure_s"] == pytest.approx(0.001)
    assert summary["post_failure_instantaneous_metrics_valid"] is False
    assert summary["time"] == pytest.approx(0.001)

    with h5py.File(output / "trajectory.h5", "r", libver="latest", swmr=True) as trajectory:
        assert int(trajectory.attrs["committed_frames"]) == 5
        assert trajectory["time"][:].tolist() == pytest.approx(frame_times)
