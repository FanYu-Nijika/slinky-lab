import importlib.util
from pathlib import Path
import sys

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
spec = importlib.util.spec_from_file_location("search_gait", SCRIPT_DIR / "search_gait.py")
search = importlib.util.module_from_spec(spec)
spec.loader.exec_module(search)


def test_search_refinement_retains_physical_and_scene_parameters():
    base = search.build_config("n12-g4-h06")
    half = search.build_config("n12-g4-h06", "half_dt")
    mesh = search.build_config("n12-g4-h06", "mesh")
    assert base.material == half.material == mesh.material
    assert base.scene == half.scene == mesh.scene
    assert half.numerics.timestep == base.numerics.timestep / 2
    assert mesh.numerics.timestep == base.numerics.timestep / 4
    assert mesh.numerics.segments_per_turn == 2 * base.numerics.segments_per_turn


def test_declared_search_cases_are_valid_and_perturbations_are_bounded():
    for case_id in search.CASES:
        assert search.build_config(case_id).material.rotational_viscosity > 0
    base = search.build_config("n12-g4-h06")
    shifted = search.build_config("n12-g4-h06", "offset_plus")
    assert shifted.material == base.material
    assert shifted.scene.launch_offset - base.scene.launch_offset == pytest.approx(0.002)
    with pytest.raises(ValueError, match="arched"):
        search.build_config("macro-tilt-control", "rise_plus")


def test_failed_or_penetrating_search_cannot_be_accepted_despite_three_supports():
    config = search.build_config("n12-g4-h06")
    record = {"status": "completed", "release_ready": True, "warning_detected": False,
              "timeout_detected": False, "frame_count": 61, "frame_errors": [], "hdf5_error": None,
              "max_penetration": 0.0001, "support": {"confirmed_flip_count": 4}}
    motion = {"passed": True, "completed_swing_count": 3}
    initial = {"preflight_valid": True, "initial_max_penetration_m": 0.0}
    frames = [{"metrics": {"com_y": 0.0}}, {"metrics": {"com_y": 0.01}}]
    result = search.candidate_check(record, config, motion, initial, frames)
    assert result["candidate_eligible_for_validation"]
    assert not result["numerical_convergence_passed"]
    assert not result["robustness_passed"]
    for changes in ({"status": "timeout"}, {"release_ready": False}, {"warning_detected": True},
                    {"max_penetration": config.material.strip_thickness}, {"movement_classification": "side_fall"}):
        assert not search.candidate_check({**record, **changes}, config, motion, initial, frames)["candidate_eligible_for_validation"]
    assert not search.candidate_check(record, config, None, initial, frames)["candidate_eligible_for_validation"]
    assert not search.candidate_check(record, config, {"passed": True, "completed_swing_count": 2}, initial, frames)["candidate_eligible_for_validation"]
    assert not search.candidate_check(record, config, motion, {**initial, "initial_max_penetration_m": 0.005}, frames)["candidate_eligible_for_validation"]
    assert not search.candidate_check(record, config, motion, initial, [{"metrics": {"com_y": 0.4}}])["candidate_eligible_for_validation"]


def test_new_macro_hypotheses_preserve_passive_release_and_mesh_viscosity():
    for case_id in search.MACRO_HYPOTHESES:
        base = search.build_config(case_id)
        mesh = search.build_config(case_id, "mesh")
        assert base.material == mesh.material
        assert base.scene == mesh.scene
        assert base.scene.step_width == pytest.approx(0.3)
        assert base.scene.initial_forward_velocity == 0
        assert base.scene.initial_angular_velocity == 0
        assert base.scene.initial_lateral_velocity == 0
