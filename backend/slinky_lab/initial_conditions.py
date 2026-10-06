"""Analytic initial guesses for a gravity-loaded helical cable.

The returned pose is only a starting point for a constrained numerical
relaxation.  It is not accepted as equilibrium until the solver checks force
and acceleration residuals.  The construction preserves the supplied
centerline chord lengths while using a Saint-Venant torsional stiffness to
estimate the gravity-induced local pitch.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _rotation_y(angle: float) -> np.ndarray:
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    return np.array([[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]], dtype=float)


def _section_torsion_constant(half_width: float, half_thickness: float) -> float:
    """Return the Saint-Venant rectangle constant used by cable.cc."""

    h = float(half_width)
    w = float(half_thickness)
    a = max(h, w)
    b = min(h, w)
    return a * b**3 * (16.0 / 3.0 - 3.36 * b / a * (1.0 - b**4 / a**4 / 12.0))


def _invalid_inputs(config: Any, rest_lengths: np.ndarray, anchor: np.ndarray) -> bool:
    material = config.material
    if rest_lengths.ndim != 1 or rest_lengths.size == 0:
        return True
    if not np.all(np.isfinite(rest_lengths)) or np.any(rest_lengths <= 0.0):
        return True
    if anchor.shape != (3,) or not np.all(np.isfinite(anchor)):
        return True
    values = (
        material.turns,
        material.radius,
        material.strip_width,
        material.strip_thickness,
        material.pitch,
        material.mass,
        material.shear_modulus,
    )
    gravity = float(config.scene.gravity)
    return any(not np.isfinite(float(value)) or float(value) <= 0.0 for value in values) or not np.isfinite(gravity) or gravity < 0.0


def hanging_guess(
    config: Any,
    rest_lengths: np.ndarray,
    anchor: tuple[float, float, float] | list[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Construct a chord-length-preserving gravity-stretched helix guess.

    The material order runs from the lower end to the anchored upper end, so
    ``vertices[-1]`` is exactly ``anchor``.  ``frames[:, :, 0:3]`` stores the
    tangent, radial-width and normal axes as columns.  For segment ``i`` the
    target vertical pitch is approximated by ``pitch + F_i / k_turn`` where
    ``F_i`` is the self-weight below the segment midpoint and
    ``k_turn = G*J/(2*pi*R**3)``.  The azimuth increment is then solved from
    the supplied chord length instead of changing the reference geometry.

    ``None`` means that the requested pitch cannot fit the supplied chord
    lengths at the configured radius.  Callers should fall back to the
    reference pose and still run numerical relaxation; this function never
    clips an infeasible geometry into a fabricated guess.  The returned pose
    is not accepted as equilibrium until force and acceleration residuals pass
    the numerical solver's tolerances.
    """

    lengths = np.asarray(rest_lengths, dtype=float).reshape(-1)
    final_anchor = np.asarray(anchor, dtype=float).reshape(-1)
    if _invalid_inputs(config, lengths, final_anchor):
        return None

    material = config.material
    segment_count = int(lengths.size)
    turns = int(material.turns)
    segments_per_turn = segment_count / turns
    if turns <= 0 or segments_per_turn <= 2.0:
        return None

    torsion_constant = _section_torsion_constant(material.strip_width * 0.5, material.strip_thickness * 0.5)
    k_turn = material.shear_modulus * torsion_constant / (2.0 * np.pi * material.radius**3)
    if not np.isfinite(k_turn) or k_turn <= 0.0:
        return None

    # Each segment carries the weight of the material below its midpoint.  A
    # midpoint load avoids a discontinuous pitch jump at every segment while
    # retaining the intended first-order gravity-stretch approximation.
    segment_mass = material.mass / segment_count
    below_force = config.scene.gravity * segment_mass * (np.arange(segment_count, dtype=float) + 0.5)
    local_pitch = material.pitch + below_force / k_turn
    vertical_increments = local_pitch / segments_per_turn
    horizontal_squared = lengths**2 - vertical_increments**2
    if np.any(horizontal_squared < 0.0):
        return None
    horizontal_chords = np.sqrt(horizontal_squared)
    sine_half_angle = horizontal_chords / (2.0 * material.radius)
    if np.any(sine_half_angle > 1.0) or np.any(sine_half_angle < 0.0):
        return None
    angle_increments = 2.0 * np.arcsin(sine_half_angle)
    theta = np.concatenate(([0.0], np.cumsum(angle_increments)))
    z = np.concatenate(([0.0], np.cumsum(vertical_increments)))
    local_vertices = np.column_stack((
        material.radius * np.cos(theta),
        material.radius * np.sin(theta),
        z,
    ))

    rotation = _rotation_y(np.deg2rad(config.scene.tilt_deg if config.scenario == "stairs" else 0.0))
    vertices = local_vertices @ rotation.T
    vertices += final_anchor - vertices[-1]
    vertices[-1] = final_anchor
    actual_lengths = np.linalg.norm(np.diff(vertices, axis=0), axis=1)
    if np.max(np.abs(actual_lengths - lengths)) >= 1e-9:
        return None

    frames = np.empty((segment_count, 3, 3), dtype=float)
    for index in range(segment_count):
        tangent = vertices[index + 1] - vertices[index]
        tangent /= np.linalg.norm(tangent)
        phase = 0.5 * (theta[index] + theta[index + 1])
        radial = rotation @ np.array([-np.cos(phase), -np.sin(phase), 0.0], dtype=float)
        radial -= tangent * float(np.dot(radial, tangent))
        radial_norm = np.linalg.norm(radial)
        if radial_norm <= 1e-14:
            return None
        radial /= radial_norm
        normal = np.cross(tangent, radial)
        normal_norm = np.linalg.norm(normal)
        if normal_norm <= 1e-14:
            return None
        normal /= normal_norm
        radial = np.cross(normal, tangent)
        frames[index] = np.column_stack((tangent, radial, normal))
    if not np.all(np.isfinite(frames)):
        return None
    return vertices, frames


__all__ = ["hanging_guess"]
