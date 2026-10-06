"""Independent elastic-energy diagnostic for MuJoCo's cable plugin.

The first-party ``elasticity.cable`` plugin contributes passive generalized
forces, but it does not expose its elastic potential through ``d.energy``.
This module mirrors the geometry and local curvature used by MuJoCo 3.15.0 so
the application can report a separate, explicitly approximate energy check.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

try:
    import mujoco
except ImportError:  # pragma: no cover - the project depends on MuJoCo
    mujoco = None  # type: ignore[assignment]


def rectangular_section_coefficients(half_sizes: Sequence[float]) -> dict[str, float]:
    """Return ``J``, ``Iy`` and ``Iz`` for a MuJoCo box geom.

    MuJoCo stores a box as half-sizes ``(x, y, z)``.  The cable plugin uses
    ``h = size[1]`` and ``w = size[2]`` and applies the exact expressions in
    ``plugin/elasticity/cable.cc``.  The first dimension is the segment length
    and is intentionally ignored here.
    """

    if len(half_sizes) != 3:
        raise ValueError("box half_sizes must contain exactly three values")
    h = float(half_sizes[1])
    w = float(half_sizes[2])
    if not np.isfinite(h) or not np.isfinite(w) or h <= 0.0 or w <= 0.0:
        raise ValueError("box cross-section half-sizes must be finite and positive")
    a = max(h, w)
    b = min(h, w)
    torsion = a * b**3 * (16.0 / 3.0 - 3.36 * b / a * (1.0 - b**4 / a**4 / 12.0))
    iy = (2.0 * w)**3 * (2.0 * h) / 12.0
    iz = (2.0 * h)**3 * (2.0 * w) / 12.0
    return {"J": float(torsion), "Iy": float(iy), "Iz": float(iz)}


def _require_mujoco() -> None:
    if mujoco is None:
        raise RuntimeError("MuJoCo 3.15.0 is required for cable energy diagnostics")


def _body_geom(model: object, body_id: int) -> int:
    body_id = int(body_id)
    body_geomnum = int(model.body_geomnum[body_id])  # type: ignore[attr-defined]
    if body_geomnum <= 0:
        raise ValueError(f"cable body {body_id} has no geom")
    return int(model.body_geomadr[body_id])  # type: ignore[attr-defined]


def _joint_quaternion(model: object, qpos: np.ndarray, body_id: int) -> np.ndarray:
    joint_id = int(model.body_jntadr[body_id])  # type: ignore[attr-defined]
    joint_count = int(model.body_jntnum[body_id])  # type: ignore[attr-defined]
    if joint_count <= 0:
        raise ValueError(f"cable body {body_id} has no joint")
    qadr = int(model.jnt_qposadr[joint_id]) + int(model.body_dofnum[body_id]) - 3  # type: ignore[attr-defined]
    if qadr < 0 or qadr + 4 > qpos.size:
        raise ValueError(f"cable body {body_id} has no valid ball-joint quaternion")
    return np.asarray(qpos[qadr:qadr + 4], dtype=float)


def _local_rotation_vector(model: object, data: object, body_id: int) -> np.ndarray:
    body_quat = np.asarray(model.body_quat[body_id], dtype=float).copy()  # type: ignore[attr-defined]
    joint_quat = _joint_quaternion(model, np.asarray(data.qpos, dtype=float), body_id)  # type: ignore[attr-defined]
    relative = np.empty(4, dtype=float)
    mujoco.mju_mulQuat(relative, body_quat, joint_quat)
    rotation = np.empty(3, dtype=float)
    mujoco.mju_quat2Vel(rotation, relative, 1.0)
    return rotation


def _reference_rotation_vector(model: object, body_id: int) -> np.ndarray:
    body_quat = np.asarray(model.body_quat[body_id], dtype=float).copy()  # type: ignore[attr-defined]
    joint_quat = _joint_quaternion(model, np.asarray(model.qpos0, dtype=float), body_id)  # type: ignore[attr-defined]
    rotation = np.empty(3, dtype=float)
    mujoco.mju_subQuat(rotation, body_quat, joint_quat)
    return rotation


def cable_elastic_energy(
    model: object,
    data: object,
    body_ids: Sequence[int],
    young: float,
    shear: float,
) -> float:
    """Return the cable plugin's rectangular-section elastic-energy estimate.

    ``body_ids`` must follow the plugin's material order.  For each edge after
    the first body, this evaluates the same relative quaternion log and
    reference ``omega0`` as ``cable.cc`` and integrates the quadratic local
    bending/twist law over the edge length.  The result is a diagnostic
    reconstruction, not a value supplied by ``data.energy`` and not a proof of
    exact potential-energy conservation for the plugin.
    """

    _require_mujoco()
    bodies = [int(body_id) for body_id in body_ids]
    if len(bodies) < 2:
        return 0.0
    young = float(young)
    shear = float(shear)
    if not np.isfinite(young) or not np.isfinite(shear) or young < 0.0 or shear < 0.0:
        raise ValueError("young and shear moduli must be finite and non-negative")
    xpos = np.asarray(data.xpos, dtype=float)  # type: ignore[attr-defined]
    total = 0.0
    for index in range(1, len(bodies)):
        body_id = bodies[index]
        previous_id = bodies[index - 1]
        geom_id = _body_geom(model, body_id)
        geom_type = int(model.geom_type[geom_id])  # type: ignore[attr-defined]
        if geom_type != int(mujoco.mjtGeom.mjGEOM_BOX):
            raise ValueError("cable energy reconstruction currently requires box geoms")
        coefficients = rectangular_section_coefficients(model.geom_size[geom_id])  # type: ignore[attr-defined]
        stiffness = np.array([
            shear * coefficients["J"],
            young * coefficients["Iy"],
            young * coefficients["Iz"],
        ], dtype=float)
        length = float(np.linalg.norm(xpos[body_id] - xpos[previous_id]))
        if not np.isfinite(length) or length <= 0.0:
            raise ValueError("cable edge length must be finite and positive")
        delta = _local_rotation_vector(model, data, body_id) - _reference_rotation_vector(model, body_id)
        total += 0.5 * float(np.dot(stiffness, delta * delta)) / length
    return float(max(0.0, total))


__all__ = ["cable_elastic_energy", "rectangular_section_coefficients"]
