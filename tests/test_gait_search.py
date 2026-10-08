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
              "max_penetration": 0.0001, "support": {"confirmed_flip_count": 3}}
    result = search.candidate_check(record, config)
    assert result["candidate_eligible_for_validation"]
    assert not result["numerical_convergence_passed"]
    assert not result["robustness_passed"]
    for changes in ({"status": "timeout"}, {"release_ready": False}, {"warning_detected": True},
                    {"max_penetration": config.material.strip_thickness}, {"support": {"confirmed_flip_count": 2}}):
        assert not search.candidate_check({**record, **changes}, config)["candidate_eligible_for_validation"]
