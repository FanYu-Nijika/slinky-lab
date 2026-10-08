import numpy as np
import pytest
from pydantic import ValidationError


mujoco = pytest.importorskip("mujoco")

from slinky_lab.physics import Simulation
from slinky_lab.presets import get_preset
from slinky_lab.schemas import Numerics, RunConfig, Scene


def _config(scenario: str = "stairs", *, initial_pose: str = "tilted", turns: int = 8) -> RunConfig:
    return RunConfig(
        scenario=scenario,
        material={
            "turns": turns,
            "radius": 0.03,
            "strip_width": 0.006,
            "strip_thickness": 0.0045,
            "pitch": 0.0055,
            "mass": 0.0487,
            "young_modulus": 2.0e7,
            "shear_modulus": 2.0e7 / 2.7,
        },
        scene=Scene(step_count=3, step_height=0.04, step_depth=0.08, tilt_deg=65,
                    initial_pose=initial_pose, arch_end_turns=2),
        numerics=Numerics(profile="preview", duration=0.02, segments_per_turn=8, timestep=0.0005),
    )


def test_legacy_tilted_and_drop_keep_rest_qpos_and_zero_reference_force():
    stairs = Simulation(_config())
    drop = Simulation(_config("drop", turns=3))
    assert stairs.config.scene.initial_pose == "tilted"
    assert stairs.data.qpos == pytest.approx(stairs.model.qpos0)
    assert drop.data.qpos == pytest.approx(drop.model.qpos0)
    assert np.linalg.norm(stairs.data.qfrc_passive) == pytest.approx(0.0, abs=1e-12)
    assert np.linalg.norm(drop.data.qfrc_passive) == pytest.approx(0.0, abs=1e-12)
    assert stairs.summary()["initial_pose_diagnostics"]["qpos0_unchanged"] is True
    assert drop.summary()["initial_pose_diagnostics"]["resolved"] == "drop_reference"


def test_arched_pose_preserves_reference_and_fk_link_lengths():
    simulation = Simulation(_config(initial_pose="arched"))
    diagnostics = simulation.summary()["initial_pose_diagnostics"]
    nodes = np.concatenate((np.asarray(simulation.data.xpos, dtype=float)[simulation._body_ids],
                            np.asarray(simulation.data.site_xpos, dtype=float)[simulation._endpoint_ids[1]][None, :]))
    link_lengths = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
    assert diagnostics["resolved"] == "arched"
    assert diagnostics["qpos0_unchanged"] is True
    axis_centres = np.asarray(diagnostics["target_axis_centres"], dtype=float)
    assert axis_centres[:, 0] == pytest.approx([0.04, 0.12])
    assert axis_centres[:, 1] == pytest.approx([0.0, 0.0])
    assert diagnostics["max_link_error_m"] < 1e-7
    assert diagnostics["max_endpoint_error_m"] < 1e-7
    assert link_lengths == pytest.approx(simulation._rest_lengths, abs=1e-7)
    assert diagnostics["initial_elastic_energy_estimate"] > 0.0
    assert diagnostics["first_end_within_stair0"] is True
    assert diagnostics["last_end_within_stair1"] is True
    assert diagnostics["actual_first_clearance_m"] == pytest.approx(0.0002, abs=1e-7)
    assert diagnostics["actual_free_end_clearance_m"] == pytest.approx(0.025, abs=1e-7)
    assert diagnostics["equality_active_count"] == 0
    assert diagnostics["actuator_count"] == 0
    assert diagnostics["applied_force_max"] == pytest.approx(0.0, abs=1e-15)
    assert diagnostics["equilibrium_status"] == "manual_non_equilibrium"
    assert simulation.summary()["support_diagnostics"]["verified_steps"] == []
    # Fixed rod lengths alone do not imply collision-free geometry.
    assert diagnostics["preflight_valid"] is False


def test_arched_pose_can_release_for_one_short_real_step():
    simulation = Simulation(get_preset("stairs-arched"))
    assert simulation.summary()["initial_pose_diagnostics"]["preflight_valid"] is True
    simulation.prepare()
    simulation.step()
    assert simulation.time == pytest.approx(simulation.dt)
    assert np.all(np.isfinite(simulation.data.qpos))
    assert np.all(np.isfinite(simulation.data.qvel))


def test_arched_pose_rejects_invalid_scenario_turns_and_span():
    with pytest.raises(ValidationError, match="only valid"):
        _config("drop", initial_pose="arched", turns=8)
    with pytest.raises(ValidationError, match="turns"):
        _config(initial_pose="arched", turns=5)
    config = _config(initial_pose="arched")
    config.scene.step_depth = 0.015
    with pytest.raises(ValueError, match="too tight"):
        Simulation(config)
