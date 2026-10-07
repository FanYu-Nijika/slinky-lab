from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import scripts.validate_stairs as validate_stairs
import scripts.validate_literature_drop as validate_literature_drop
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
        self._prepared = True
        self._released = True

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
            "prepared": True,
            "released": True,
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


class HeldPreparationSimulation:
    """Expose a held-settle clock, then cancel before release."""

    def __init__(self, config: RunConfig):
        self.config = config
        self.duration = config.numerics.duration
        self.time = 0.0
        self.model_xml = "<mujoco model='fake-held-preparation'/>"
        self.data = SimpleNamespace(qpos=np.array([0.0]), qvel=np.array([0.0]))
        self._status = "settling"
        self.frame_called = False

    def prepare(self, should_cancel=None):
        self.time = 0.39658
        if should_cancel is not None and should_cancel():
            self._status = "cancelled"

    def frame(self):
        self.frame_called = True
        raise AssertionError("held preparation must not export a released frame")

    def summary(self):
        return {
            "status": self._status,
            "prepared": False,
            "released": False,
            "time": self.time,
            "duration": self.duration,
            "steps": 0,
            "static_equilibrium_length": 1.14,
            "metrics": {"max_penetration": 0.0},
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
        return {"model_version": "fake-held", "coordinate_system": {"world_up": "+z"}}


class StairPreparationFailureSimulation(HeldPreparationSimulation):
    """Mirror stairs' raw released flag before prepare completes."""

    def summary(self):
        result = super().summary()
        result["released"] = True
        return result


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
    assert record["phase"] == "failed"
    assert record["failure_phase"] == "released_integration"
    assert record["model_clock_s"] == pytest.approx(0.001)
    assert record["released_integration_time_s"] == pytest.approx(0.08)

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


def test_run_case_keeps_held_preparation_clock_out_of_released_timeline(monkeypatch, tmp_path: Path):
    base = RunConfig(
        scenario="drop",
        material={"turns": 3},
        numerics={"duration": 0.1, "sample_hz": 50, "segments_per_turn": 8, "timestep": 0.0002, "max_wall_seconds": 5},
    )
    monkeypatch.setattr(validate_stairs, "Simulation", HeldPreparationSimulation)
    clock = {"value": 0.0}

    def monotonic():
        clock["value"] += 6.0
        return clock["value"]

    monkeypatch.setattr(validate_stairs.time, "monotonic", monotonic)
    output = tmp_path / "held-preparation"
    record = validate_stairs.run_case(
        base,
        {"name": "held-preparation", "segments_per_turn": 8, "timestep": 0.0002, "wall_seconds": 5.0},
        output,
    )

    assert record["status"] == "timeout"
    assert record["phase"] == "failed"
    assert record["execution_phase"] == "preparation"
    assert record["failure_phase"] == "preparation"
    assert record["model_clock_s"] == pytest.approx(0.39658)
    assert record["released_integration_time_s"] == pytest.approx(0.0)
    assert record["simulation_time"] == pytest.approx(0.0)
    assert record["engine_time_after_failure_s"] == pytest.approx(0.39658)
    assert record["post_failure_instantaneous_metrics_valid"] is False
    assert record["frame_count"] == 0

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["prepared"] is False
    assert summary["released"] is False
    assert summary["time"] == pytest.approx(0.39658)
    assert summary["model_clock_s"] == pytest.approx(0.39658)
    assert summary["released_integration_time_s"] == pytest.approx(0.0)
    assert summary["engine_time_after_failure_s"] == pytest.approx(0.39658)
    assert summary["post_failure_instantaneous_metrics_valid"] is False
    assert json.loads((output / "frames.json").read_text(encoding="utf-8")) == []

    with h5py.File(output / "trajectory.h5", "r", libver="latest", swmr=True) as trajectory:
        assert int(trajectory.attrs["committed_frames"]) == 0


def test_stairs_raw_released_flag_does_not_skip_preparation_failure(monkeypatch, tmp_path: Path):
    base = RunConfig(
        scenario="stairs",
        material={"turns": 3},
        scene={"step_count": 3},
        numerics={"duration": 0.1, "sample_hz": 50, "segments_per_turn": 8, "timestep": 0.0002, "max_wall_seconds": 5},
    )
    monkeypatch.setattr(validate_stairs, "Simulation", StairPreparationFailureSimulation)
    clock = {"value": 0.0}

    def monotonic():
        clock["value"] += 6.0
        return clock["value"]

    monkeypatch.setattr(validate_stairs.time, "monotonic", monotonic)
    output = tmp_path / "stairs-preparation-failure"
    record = validate_stairs.run_case(
        base,
        {"name": "stairs-preparation-failure", "segments_per_turn": 8, "timestep": 0.0002, "wall_seconds": 5.0},
        output,
    )

    assert record["status"] == "timeout"
    assert record["released"] is True
    assert record["prepared"] is False
    assert record["release_ready"] is False
    assert record["phase"] == "failed"
    assert record["execution_phase"] == "preparation"
    assert record["failure_phase"] == "preparation"
    assert record["released_integration_time_s"] == pytest.approx(0.0)
    assert record["simulation_time"] == pytest.approx(0.0)
    assert record["frame_count"] == 0
    assert json.loads((output / "frames.json").read_text(encoding="utf-8")) == []


def test_literature_dynamic_comparison_rejects_held_preparation_clock():
    summary = {
        "prepared": False,
        "released": False,
        "time": 0.39658,
        "duration": 0.35,
        "collapse_time": 0.27,
    }
    comparison = validate_literature_drop.dynamic_collapse_comparison(summary)
    assert comparison["status"] == "preparation_not_completed"
    assert comparison["simulation_collapse_time_s"] is None
    assert comparison["relative_error"] is None
    assert comparison["reason"] == "preparation_not_completed"
    assert comparison["comparison_type"] == "dynamic prediction versus Table II best-fit model timing; no acceptance threshold applied"
    assert validate_literature_drop.event_diagnostic(summary, "collapse_time") == {
        "time_s": None,
        "reason": "preparation_not_completed",
    }
