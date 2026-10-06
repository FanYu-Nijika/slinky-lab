import numpy as np
import pytest


mujoco = pytest.importorskip("mujoco")

from slinky_lab.equilibrium import solve_static_equilibrium
from slinky_lab.physics import Simulation
from slinky_lab.schemas import Numerics, RunConfig


def _small_drop() -> Simulation:
    config = RunConfig.model_validate({
        "scenario": "drop",
        "material": {"turns": 3, "young_modulus": 1.0e8, "shear_modulus": 1.0e8 / 2.7, "mass": 0.012},
        "numerics": {"segments_per_turn": 8, "timestep": 5.0e-5, "duration": 0.02},
    })
    simulation = Simulation(config)
    simulation._apply_hanging_guess()
    return simulation


def test_native_static_solver_keeps_reference_and_matches_forward_terms():
    simulation = _small_drop()
    model, data = simulation.model, simulation.data
    qpos0 = model.qpos0.copy()
    endpoint = model.site("S_first").id
    result = solve_static_equilibrium(model, data, simulation._anchor, endpoint_site_id=endpoint, time_limit_s=10.0)

    assert result["status"] == "converged"
    assert result["native_force_residual_max"] <= 1.0e-3
    assert result["full_mass_acceleration_residual_max"] <= 1.0e-8
    assert result["anchor_error_m"] <= 2.0e-4
    assert result["manual_forward_passive_max_error"] <= 1.0e-10
    assert result["manual_forward_bias_max_error"] <= 1.0e-10
    assert result["contacts"] == 0
    assert result["qpos_finite"] is True
    assert np.array_equal(model.qpos0, qpos0)


def test_cancelled_solver_restores_pose_and_velocity():
    simulation = _small_drop()
    model, data = simulation.model, simulation.data
    qpos = data.qpos.copy()
    qvel = data.qvel.copy()
    eq_active = data.eq_active.copy()
    result = solve_static_equilibrium(
        model,
        data,
        simulation._anchor,
        endpoint_site_id=model.site("S_first").id,
        should_cancel=lambda: True,
        time_limit_s=10.0,
    )

    assert result["status"] == "cancelled"
    assert result["restored"] is True
    assert np.array_equal(data.qpos, qpos)
    assert np.array_equal(data.qvel, qvel)
    assert np.array_equal(data.eq_active, eq_active)
