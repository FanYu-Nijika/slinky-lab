import numpy as np

from slinky_lab.initial_conditions import hanging_guess
from slinky_lab.schemas import Material, Numerics, RunConfig, Scene


def _reference_lengths(config: RunConfig, segments_per_turn: int = 8) -> np.ndarray:
    material = config.material
    count = material.turns * segments_per_turn
    theta = np.linspace(0.0, 2.0 * np.pi * material.turns, count + 1)
    vertices = np.column_stack((
        material.radius * np.cos(theta),
        material.radius * np.sin(theta),
        material.pitch * material.turns * np.linspace(0.0, 1.0, count + 1),
    ))
    return np.linalg.norm(np.diff(vertices, axis=0), axis=1)


def _feasible_config() -> RunConfig:
    material = Material(turns=3, mass=0.001, young_modulus=1.0e9, shear_modulus=1.0e9, damping=0.0)
    return RunConfig(
        scenario="drop",
        material=material,
        scene=Scene(settle_time=0.05),
        numerics=Numerics(profile="preview", duration=0.02, segments_per_turn=8, timestep=0.0002),
    )


def test_hanging_guess_preserves_chord_lengths_and_anchor():
    config = _feasible_config()
    rest_lengths = _reference_lengths(config)
    anchor = np.array([0.2, -0.1, 0.7])
    result = hanging_guess(config, rest_lengths, anchor)
    assert result is not None
    vertices, frames = result
    assert vertices.shape == (rest_lengths.size + 1, 3)
    assert frames.shape == (rest_lengths.size, 3, 3)
    assert np.array_equal(vertices[-1], anchor)
    lengths = np.linalg.norm(np.diff(vertices, axis=0), axis=1)
    assert np.max(np.abs(lengths - rest_lengths)) < 1e-9


def test_hanging_guess_frames_are_radial_orthonormal_and_tangent_aligned():
    config = _feasible_config()
    result = hanging_guess(config, _reference_lengths(config), [0.0, 0.0, 0.8])
    assert result is not None
    vertices, frames = result
    tangents = np.diff(vertices, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1)[:, None]
    for tangent, frame in zip(tangents, frames):
        assert np.allclose(frame.T @ frame, np.eye(3), atol=1e-12)
        assert np.dot(tangent, frame[:, 0]) > 1.0 - 1e-12


def test_infeasible_gravity_stretch_returns_none_instead_of_clipping():
    config = RunConfig(
        scenario="drop",
        material=Material(turns=3, mass=0.0487, shear_modulus=1.0e3),
        scene=Scene(settle_time=0.05),
        numerics=Numerics(profile="preview", duration=0.02, segments_per_turn=8, timestep=0.0002),
    )
    assert hanging_guess(config, _reference_lengths(config), [0.0, 0.0, 0.5]) is None


def test_documentation_marks_guess_as_non_equilibrium():
    assert hanging_guess.__doc__ is not None
    assert "not accepted as equilibrium" in hanging_guess.__doc__
