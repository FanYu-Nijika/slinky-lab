import numpy as np
import pytest


mujoco = pytest.importorskip("mujoco")

from slinky_lab.energy import cable_elastic_energy, rectangular_section_coefficients
from slinky_lab.physics import Simulation
from slinky_lab.schemas import Material, Numerics, RunConfig, Scene


def _small_simulation() -> Simulation:
    material = Material(turns=3, damping=0.0)
    config = RunConfig(
        scenario="stairs",
        material=material,
        scene=Scene(gravity=0.0, tilt_deg=0.0, step_count=3),
        numerics=Numerics(profile="preview", duration=0.02, segments_per_turn=8, timestep=0.0002),
    )
    simulation = Simulation(config)
    simulation.data.qpos[:] = simulation.model.qpos0
    simulation.data.qvel[:] = 0.0
    mujoco.mj_forward(simulation.model, simulation.data)
    return simulation


def test_rectangular_section_coefficients_match_cable_source():
    half_sizes = (0.04, 0.002, 0.001)
    h, w = half_sizes[1], half_sizes[2]
    a, b = max(h, w), min(h, w)
    expected = {
        "J": a * b**3 * (16.0 / 3.0 - 3.36 * b / a * (1.0 - b**4 / a**4 / 12.0)),
        "Iy": (2.0 * w)**3 * (2.0 * h) / 12.0,
        "Iz": (2.0 * h)**3 * (2.0 * w) / 12.0,
    }
    assert rectangular_section_coefficients(half_sizes) == pytest.approx(expected)


def test_no_load_rest_pose_has_zero_reconstructed_energy():
    simulation = _small_simulation()
    material = simulation.config.material
    energy = cable_elastic_energy(
        simulation.model,
        simulation.data,
        simulation._body_ids,
        material.young_modulus,
        material.shear_modulus,
    )
    assert energy == pytest.approx(0.0, abs=1e-18)
    assert np.linalg.norm(simulation.data.qfrc_passive) == pytest.approx(0.0, abs=1e-15)


def test_small_joint_rotation_reports_finite_plugin_torque_mismatch():
    simulation = _small_simulation()
    material = simulation.config.material
    body_id = simulation._body_ids[1]
    joint_id = int(simulation.model.body_jntadr[body_id])
    qadr = int(simulation.model.jnt_qposadr[joint_id])
    dofadr = int(simulation.model.jnt_dofadr[joint_id])

    def evaluate(angle: float) -> tuple[float, float]:
        simulation.data.qpos[:] = simulation.model.qpos0
        simulation.data.qpos[qadr:qadr + 4] = [np.cos(angle / 2.0), 0.0, np.sin(angle / 2.0), 0.0]
        mujoco.mj_forward(simulation.model, simulation.data)
        energy = cable_elastic_energy(
            simulation.model,
            simulation.data,
            simulation._body_ids,
            material.young_modulus,
            material.shear_modulus,
        )
        return energy, float(simulation.data.qfrc_passive[dofadr + 1])

    angle = 0.01
    delta = 1e-6
    energy_minus, _ = evaluate(angle - delta)
    energy_plus, _ = evaluate(angle + delta)
    _, passive_torque = evaluate(angle)
    finite_difference = (energy_plus - energy_minus) / (2.0 * delta)
    relative_error = abs(finite_difference + passive_torque) / max(abs(passive_torque), 1e-30)

    # The cable force law uses a rotation-vector residual directly.  Its
    # quaternion-to-torque map is not the exact gradient of the quadratic
    # reconstruction, so this is a bounded diagnostic comparison, not an
    # energy-conservation claim.
    assert np.isfinite(finite_difference)
    assert np.isfinite(passive_torque)
    assert relative_error < 0.25
