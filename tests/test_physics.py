import pytest


mujoco = pytest.importorskip("mujoco")

from slinky_lab.physics import Simulation
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
    assert metadata["model_version"] == "helical-box-cable-v1"
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
