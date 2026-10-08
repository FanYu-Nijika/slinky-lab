import numpy as np
import pytest


mujoco = pytest.importorskip("mujoco")

from slinky_lab import MODEL_VERSION
from slinky_lab.physics import Simulation
from slinky_lab.presets import get_preset
from slinky_lab.schemas import Numerics, RunConfig, Scene


def _config(scenario: str) -> RunConfig:
    return RunConfig(
        scenario=scenario,
        scene=Scene(settle_time=0.05, step_count=3),
        numerics=Numerics(profile="preview", duration=0.02, segments_per_turn=8, timestep=0.0005),
    )


def test_box_cable_model_exposes_material_order_and_metadata():
    simulation = Simulation(_config("drop"))
    metadata = simulation.metadata()
    assert "mujoco.elasticity.cable" in simulation.model_xml
    assert metadata["segment_count"] == 12 * 8
    assert len(metadata["half_sizes"]) == metadata["segment_count"]
    assert all(len(size) == 3 for size in metadata["half_sizes"])
    assert metadata["coordinate_system"]["quaternion"] == "wxyz"
    assert metadata["engine_version"] == "3.15.0"
    assert metadata["model_version"] == MODEL_VERSION
    assert "instance=\"rod_material\"" in simulation.model_xml


def test_drop_prepare_releases_and_steps_with_finite_metrics():
    simulation = Simulation(_config("drop"))
    simulation.prepare()
    assert simulation.time == pytest.approx(0.0)
    assert simulation.summary()["released"] is True
    simulation.step()
    frame = simulation.frame()
    assert frame["time"] > 0.0
    assert len(frame["positions"]) == simulation.summary()["segment_count"]
    assert all(len(contact) == 3 for contact in frame["contacts"])
    assert all(name in frame["metrics"] for name in ("com_vx", "com_vy", "com_vz"))
    assert all(value == value for value in frame["metrics"].values())


def test_stairs_model_contains_static_geometry_and_real_contacts_can_be_reported():
    simulation = Simulation(_config("stairs"))
    simulation.prepare()
    metadata = simulation.metadata()
    assert len(metadata["stairs"]) == 3
    for _ in range(4):
        simulation.step()
    assert simulation.frame()["contacts"] is not None
    assert simulation.summary()["max_penetration"] >= 0.0
    assert simulation.summary()["movement_classification"] in {"no_step_contact", "sliding", "flip"}


def test_reference_geometry_preserves_chords_radial_section_and_mass():
    simulation = Simulation(_config("drop"))
    vertices = simulation._rest_vertices
    frames = np.asarray(simulation._rest_frames)
    assert np.linalg.norm(vertices[:, :2], axis=1) == pytest.approx(simulation.config.material.radius)
    assert abs(vertices[0, 2] - vertices[-1, 2]) == pytest.approx(simulation._height)
    assert np.linalg.norm(np.diff(vertices, axis=0), axis=1) == pytest.approx(simulation._rest_lengths)
    assert frames.transpose(0, 2, 1) @ frames == pytest.approx(np.broadcast_to(np.eye(3), frames.shape), abs=1e-12)
    assert np.linalg.det(frames) == pytest.approx(np.ones(len(frames)), abs=1e-12)
    assert simulation._model_mass == pytest.approx(simulation.config.material.mass, rel=1e-10)
    assert simulation.data.ncon == 0
    assert np.linalg.norm(simulation.data.qfrc_passive) < 1e-12


def test_nonadjacent_turns_have_real_self_contact_after_bending():
    config = RunConfig.model_validate({"scenario": "drop", "material": {"turns": 3},
                                     "numerics": {"segments_per_turn": 8}})
    simulation = Simulation(config)
    joint = simulation.model.joint("J_1")
    qadr = int(joint.qposadr[0])
    angle = -0.08
    simulation.data.qpos[qadr:qadr + 4] = [np.cos(angle / 2), 0, np.sin(angle / 2), 0]
    mujoco.mj_forward(simulation.model, simulation.data)
    pairs = [(simulation._cable_geom_to_material.get(int(contact.geom[0]), -1),
              simulation._cable_geom_to_material.get(int(contact.geom[1]), -1))
             for contact in simulation.data.contact]
    assert any(a >= 0 and b >= 0 and abs(a - b) >= 6 for a, b in pairs)


def test_stair_angular_perturbation_uses_world_flip_axis():
    config = RunConfig.model_validate({"scenario": "stairs", "material": {"turns": 3},
                                     "scene": {"tilt_deg": 45, "initial_angular_velocity": 2},
                                     "numerics": {"segments_per_turn": 8}})
    simulation = Simulation(config)
    velocity = np.empty(6)
    mujoco.mj_objectVelocity(simulation.model, simulation.data, mujoco.mjtObj.mjOBJ_BODY,
                            simulation._body_ids[0], velocity, 0)
    assert velocity[:3] == pytest.approx([0, 2, 0], abs=1e-10)


def _viscous_config(segments_per_turn: int, viscosity: float | None) -> RunConfig:
    return RunConfig.model_validate({
        "scenario": "drop",
        "material": {"turns": 3, "damping": 0.003, "rotational_viscosity": viscosity},
        "numerics": {"segments_per_turn": segments_per_turn, "timestep": 0.0005, "duration": 0.02},
    })


def _ball_damping(simulation: Simulation) -> np.ndarray:
    return np.asarray(simulation.model.dof_damping[6:], dtype=float).reshape(-1, 3)[:, 0]


def test_rotational_viscosity_keeps_legacy_damping_and_free_root():
    legacy = Simulation(_viscous_config(8, None))
    assert np.allclose(_ball_damping(legacy), 0.003)
    assert np.allclose(legacy.model.dof_damping[:6], 0.0)
    assert legacy.metadata()["damping"]["definition"] == "legacy_per_joint"
    assert legacy.metadata()["damping"]["effective_range"]["free_root_damping_N_m_s"] == 0.0

    viscosity = 4.0e-5
    scaled = Simulation(_viscous_config(8, viscosity))
    assert np.allclose(scaled.model.dof_damping[:6], 0.0)
    assert np.asarray(scaled.model.dof_damping[6:], dtype=float).size == 3 * (scaled._segments - 1)
    assert scaled.metadata()["damping"]["input"]["rotational_viscosity"] == viscosity
    assert scaled.summary()["damping"]["effective_range"]["free_root_damping_N_m_s"] == 0.0


def test_rotational_viscosity_scales_each_mesh_link():
    viscosity = 4.0e-5
    for segments_per_turn in (8, 16):
        simulation = Simulation(_viscous_config(segments_per_turn, viscosity))
        links = np.asarray(simulation._rest_lengths[:-1], dtype=float)
        coefficients = _ball_damping(simulation)
        assert coefficients.size == links.size
        assert coefficients * links == pytest.approx(np.full(links.shape, viscosity), rel=1e-10, abs=1e-12)


def test_smooth_curvature_dissipation_is_mesh_consistent():
    viscosity = 4.0e-5
    normalized_powers = []
    for segments_per_turn in (8, 16):
        simulation = Simulation(_viscous_config(segments_per_turn, viscosity))
        links = np.asarray(simulation._rest_lengths[:-1], dtype=float)
        coefficients = _ball_damping(simulation)
        midpoint = (np.cumsum(links) - 0.5 * links) / np.sum(links)
        curvature_rate = np.sin(2.0 * np.pi * midpoint) + 0.3 * np.cos(4.0 * np.pi * midpoint)
        relative_rate = curvature_rate * links
        power = float(np.sum(coefficients * relative_rate**2))
        normalized_powers.append(power / float(np.sum(links)))
    assert normalized_powers[0] == pytest.approx(normalized_powers[1], rel=0.02)


def test_valid_stairs_arched_pose_passes_release_preflight():
    simulation = Simulation(get_preset("stairs-arched"))
    diagnostics = simulation.metadata()["initial_pose_diagnostics"]
    assert diagnostics["preflight_valid"] is True
    assert diagnostics["initial_max_penetration_m"] <= diagnostics["preflight_penetration_threshold_m"]
    simulation.prepare()
    summary = simulation.summary()
    assert summary["status"] == "ready"
    assert summary["prepared"] is True
    assert summary["released"] is True
    assert summary["steps"] == 0
    assert summary["time"] == pytest.approx(0.0)


def test_invalid_stairs_arched_pose_is_rejected_before_mj_step(monkeypatch):
    raw = get_preset("stairs-arched").model_dump(mode="python")
    raw["scene"].update({"step_depth": 0.08, "step_height": 0.06, "arch_rise": 0.04, "arch_end_turns": 2})
    config = RunConfig.model_validate(raw)
    simulation = Simulation(config)
    diagnostics = simulation.metadata()["initial_pose_diagnostics"]
    assert diagnostics["preflight_valid"] is False
    assert diagnostics["initial_max_penetration_m"] > diagnostics["preflight_penetration_threshold_m"]

    def unexpected_step(*_args, **_kwargs):
        raise AssertionError("invalid initial pose must not call mj_step")

    monkeypatch.setattr(mujoco, "mj_step", unexpected_step)
    with pytest.raises(ValueError, match="invalid arched initial pose"):
        simulation.prepare()
    summary = simulation.summary()
    assert summary["status"] == "invalid_initial_pose"
    assert summary["error"] is not None
    assert "preflight threshold" in summary["error"]
    assert summary["prepared"] is False
    assert summary["released"] is False
    assert summary["steps"] == 0
    assert summary["time"] == pytest.approx(0.0)
