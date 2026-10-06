import pytest

from slinky_lab.analysis import compare_reference, summarize_frames, summarize_sweep
from slinky_lab.schemas import RunConfig


def test_compare_reference_interpolates_and_reports_errors():
    result = compare_reference([0, 1, 2], [0, 2, 4], [0, 2], [0, 5])
    assert result["count"] == 3
    assert result["reference"] == pytest.approx([0, 2.5, 5])
    assert result["rmse"] == pytest.approx((0.0 + 0.25 + 1.0) ** 0.5 / 3**0.5)
    assert result["max_abs_error"] == pytest.approx(1.0)


def test_summarize_frames_tracks_energy_and_stair_progress():
    frames = [
        {"time": 0.0, "metrics": {"com_z": 1.0, "kinetic_energy": 0.0, "gravitational_energy": 1.0, "steps_descended": 0, "max_penetration": 0.0}},
        {"time": 0.1, "metrics": {"com_z": 0.9, "kinetic_energy": 0.2, "gravitational_energy": 0.8, "steps_descended": 2, "max_penetration": 0.001}},
    ]
    result = summarize_frames(frames, RunConfig(scenario="stairs"))
    assert result["frame_count"] == 2
    assert result["steps_descended"] == 2.0
    assert result["max_penetration"] == pytest.approx(0.001)
    assert result["reported_energy_change"] == pytest.approx(0.0)


def test_summarize_sweep_ranks_completed_runs():
    result = summarize_sweep([
        {"summary": {"scenario": "stairs", "steps_descended": 1, "status": "complete"}},
        {"summary": {"scenario": "stairs", "steps_descended": 3, "status": "complete"}},
        {"summary": {"scenario": "stairs", "status": "failed", "error": "unstable"}},
    ])
    assert result["count"] == 3
    assert result["successful_count"] == 2
    assert result["failed_count"] == 1
    assert result["best"]["objective"] == 3.0
    assert result["worst"]["objective"] == 1.0


def test_compare_reference_rejects_duplicate_reference_time_and_reports_overlap():
    result = compare_reference([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert result["time"] == pytest.approx([1.0, 2.0])
    assert result["common_time_range"] == pytest.approx([1.0, 2.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        compare_reference([0.0, 1.0], [0.0, 1.0], [0.0, 0.0], [0.0, 1.0])
