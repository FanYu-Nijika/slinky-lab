"""Bounded native-force static relaxation for a held cable pose.

The solver in this module is deliberately independent of ``Simulation``.  It
uses MuJoCo's position, passive-force and recursive-Newton-Euler routines for
the residual and only changes ``data.qpos``/``data.qvel``.  A successful return
leaves the solved pose in ``data``; every failure path restores the caller's
original state.  The analytic hanging guess remains an initial condition, not
an equilibrium certificate.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy.optimize import root

try:
    import mujoco
except ImportError:  # pragma: no cover - the project depends on MuJoCo
    mujoco = None  # type: ignore[assignment]


CancelCallback = Callable[[], bool]


class _SolveAbort(RuntimeError):
    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status


def _require_mujoco() -> None:
    if mujoco is None:
        raise RuntimeError("MuJoCo 3.15.0 is required for equilibrium diagnostics")


def _as_anchor(anchor: Sequence[float]) -> np.ndarray:
    value = np.asarray(anchor, dtype=float).reshape(-1)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError("anchor must be a finite xyz vector")
    return value


def _site_id(model: Any, endpoint_site_id: int | None) -> int | None:
    if endpoint_site_id is not None:
        site_id = int(endpoint_site_id)
        if site_id < 0 or site_id >= int(model.nsite):
            raise ValueError("endpoint_site_id is outside the model")
        return site_id
    for name in ("S_first", "S_last"):
        site_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name))
        if site_id >= 0:
            return site_id
    return None


def _active_connect(model: Any, data: Any) -> bool:
    if int(model.neq) == 0:
        return False
    return bool(np.any(np.asarray(data.eq_active, dtype=int)))


def _restore(model: Any, data: Any, qpos: np.ndarray, qvel: np.ndarray, time_value: float) -> None:
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    data.time = time_value
    mujoco.mj_forward(model, data)


def _native_terms(model: Any, data: Any, bias: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate passive and RNE bias terms without invoking the constraint solver."""

    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_comVel(model, data)
    mujoco.mj_passive(model, data)
    mujoco.mj_rne(model, data, 0, bias)
    return np.asarray(data.qfrc_passive, dtype=float).copy(), bias.copy()


def _check_abort(cancel: CancelCallback | None, started: float, time_limit_s: float, callback_time: list[float]) -> None:
    if cancel is not None:
        before = time.monotonic()
        cancelled = cancel()
        callback_time[0] += time.monotonic() - before
        if cancelled:
            raise _SolveAbort("cancelled", "equilibrium solve cancelled")
    # The worker callback may wait while the user has paused preparation.
    # Paused time must not consume the numerical solve's computation budget.
    if time.monotonic() - started - callback_time[0] > time_limit_s:
        raise _SolveAbort("timeout", "equilibrium solve exceeded its wall-time budget")


def solve_static_equilibrium(
    model: Any,
    data: Any,
    anchor: Sequence[float],
    *,
    endpoint_site_id: int | None = None,
    should_cancel: CancelCallback | None = None,
    max_calls: int = 10000,
    time_limit_s: float = 45.0,
    force_residual_tolerance: float = 1.0e-3,
    anchor_tolerance: float = 2.0e-4,
    length_scale: float = 0.03,
) -> dict[str, Any]:
    """Relax rotational cable coordinates using native static force residuals.

    The first three generalized coordinates are the held root translation and
    are excluded from the LM unknowns.  The root rotational coordinates and
    all descendant joints are integrated with ``mj_integratePos``.  After that
    solve, a separate three-variable full-forward correction adjusts the soft
    connect translation.  Collision states are intentionally unsupported for
    this static shortcut: if any contact is found, the original state is
    restored so the ordinary contact-aware numerical relaxation can handle it.
    """

    _require_mujoco()
    if int(model.nv) < 6:
        raise ValueError("a free root with at least six velocity coordinates is required")
    if max_calls < 1 or time_limit_s <= 0.0:
        raise ValueError("max_calls must be positive and time_limit_s must be positive")
    if force_residual_tolerance <= 0.0 or anchor_tolerance <= 0.0 or length_scale <= 0.0:
        raise ValueError("residual tolerances must be positive")
    target_anchor = _as_anchor(anchor)
    endpoint = _site_id(model, endpoint_site_id)
    started = time.monotonic()
    callback_time = [0.0]
    initial_qpos = np.asarray(data.qpos, dtype=float).copy()
    initial_qvel = np.asarray(data.qvel, dtype=float).copy()
    initial_time = float(data.time)
    original_eq_active = np.asarray(data.eq_active, dtype=int).copy()
    calls = 0
    initial_contacts = 0
    support_calls = 0
    solver_result: Any = None
    support_result: Any = None
    final_raw = np.full(int(model.nv), np.nan, dtype=float)
    final_bias = np.full(int(model.nv), np.nan, dtype=float)
    final_passive = np.full(int(model.nv), np.nan, dtype=float)
    support_adjustment = np.zeros(3, dtype=float)
    status = "failed"
    message = "equilibrium solve did not run"

    def restore_with(status_value: str, message_value: str) -> dict[str, Any]:
        _restore(model, data, initial_qpos, initial_qvel, initial_time)
        data.eq_active[:] = original_eq_active
        mujoco.mj_forward(model, data)
        return _diagnostics(
            status_value,
            message_value,
            calls,
            support_calls,
            solver_result,
            support_result,
            initial_contacts,
            support_adjustment,
            final_raw,
            final_bias,
            final_passive,
            data,
            endpoint,
            target_anchor,
            started,
            restored=True,
        )

    if endpoint is None:
        return restore_with("invalid_input", "no endpoint site was supplied or named S_first/S_last")
    if not _active_connect(model, data):
        return restore_with("invalid_input", "an active connect equality is required for the held-root solve")

    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    initial_contacts = int(data.ncon)
    if initial_contacts:
        return restore_with("contact_inapplicable", "static native solve skipped because the initial pose has contacts")

    total_mass = float(np.sum(np.asarray(model.body_mass[1:], dtype=float)))
    gravity = float(np.linalg.norm(np.asarray(model.opt.gravity, dtype=float)))
    force_scale = max(total_mass * gravity, 1.0e-8)
    torque_scale = max(force_scale * length_scale, 1.0e-8)
    bias = np.empty(int(model.nv), dtype=float)
    base_qpos = np.asarray(data.qpos, dtype=float).copy()
    velocity = np.zeros(int(model.nv), dtype=float)

    def residual(unknown: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        _check_abort(should_cancel, started, time_limit_s, callback_time)
        if calls > max_calls:
            raise _SolveAbort("budget_exhausted", "equilibrium residual call budget exhausted")
        candidate = base_qpos.copy()
        velocity[:] = 0.0
        velocity[3:] = np.asarray(unknown, dtype=float)
        mujoco.mj_integratePos(model, candidate, velocity, 1.0)
        if not np.all(np.isfinite(candidate)):
            raise _SolveAbort("nonfinite", "equilibrium candidate qpos became non-finite")
        data.qpos[:] = candidate
        data.qvel[:] = 0.0
        passive, rne_bias = _native_terms(model, data, bias)
        raw = passive - rne_bias
        return raw[3:] / torque_scale

    try:
        initial_residual = residual(np.zeros(int(model.nv) - 3, dtype=float))
        options = {"maxiter": max_calls, "ftol": 1.0e-10, "xtol": 1.0e-10, "gtol": 1.0e-10}
        solver_result = root(residual, np.zeros(int(model.nv) - 3, dtype=float), method="lm", options=options)
        _check_abort(should_cancel, started, time_limit_s, callback_time)
        if not np.all(np.isfinite(np.asarray(solver_result.x, dtype=float))):
            raise _SolveAbort("nonfinite", "equilibrium optimizer returned non-finite coordinates")
        candidate = base_qpos.copy()
        velocity[:] = 0.0
        velocity[3:] = np.asarray(solver_result.x, dtype=float)
        mujoco.mj_integratePos(model, candidate, velocity, 1.0)
        data.qpos[:] = candidate
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        if int(data.ncon):
            return restore_with("contact_inapplicable", "rotational solve produced cable or external contacts")

        pose_after_rotation = data.qpos.copy()
        support_base = pose_after_rotation.copy()

        def translation_residual(offset: np.ndarray) -> np.ndarray:
            nonlocal support_calls
            support_calls += 1
            _check_abort(should_cancel, started, time_limit_s, callback_time)
            if calls + support_calls > max_calls:
                raise _SolveAbort("budget_exhausted", "support correction call budget exhausted")
            data.qpos[:] = support_base
            data.qpos[:3] += np.asarray(offset, dtype=float)
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            if not np.all(np.isfinite(np.asarray(data.qacc, dtype=float))):
                raise _SolveAbort("nonfinite", "support correction produced non-finite acceleration")
            mass_accel = np.empty(int(model.nv), dtype=float)
            mujoco.mj_mulM(model, data, mass_accel, data.qacc)
            return mass_accel[:3] / force_scale

        support_result = root(translation_residual, np.zeros(3, dtype=float), method="hybr", options={"maxfev": max(20, max_calls - calls), "xtol": 1.0e-10})
        _check_abort(should_cancel, started, time_limit_s, callback_time)
        if not np.all(np.isfinite(np.asarray(support_result.x, dtype=float))):
            raise _SolveAbort("nonfinite", "support correction returned non-finite translation")
        support_adjustment = np.asarray(support_result.x, dtype=float).copy()
        data.qpos[:] = support_base
        data.qpos[:3] += support_adjustment
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        if int(data.ncon):
            return restore_with("contact_inapplicable", "support correction produced cable or external contacts")

        final_passive, final_bias = _native_terms(model, data, bias)
        final_raw = final_passive - final_bias
        mujoco.mj_forward(model, data)
        final_passive_forward = np.asarray(data.qfrc_passive, dtype=float).copy()
        final_bias_forward = np.asarray(data.qfrc_bias, dtype=float).copy()
        passive_difference = float(np.max(np.abs(final_passive - final_passive_forward)))
        bias_difference = float(np.max(np.abs(final_bias - final_bias_forward)))
        mass_accel = np.empty(int(model.nv), dtype=float)
        mujoco.mj_mulM(model, data, mass_accel, data.qacc)
        full_mass_accel_residual = float(np.max(np.abs(mass_accel)))
        nontranslational_residual = float(np.max(np.abs(final_raw[3:])))
        normalized_torque_residual = nontranslational_residual / torque_scale
        full_force_residual = max(float(np.max(np.abs(mass_accel[:3]))) / force_scale,
                                  float(np.max(np.abs(mass_accel[3:]))) / torque_scale)
        anchor_error = float(np.linalg.norm(np.asarray(data.site_xpos[endpoint], dtype=float) - target_anchor))
        qpos_finite = bool(np.all(np.isfinite(np.asarray(data.qpos, dtype=float))))
        optimizer_ok = bool(getattr(solver_result, "success", False))
        support_ok = bool(getattr(support_result, "success", False))
        converged = (optimizer_ok and support_ok and qpos_finite and normalized_torque_residual <= force_residual_tolerance
                     and full_force_residual <= 0.01 and anchor_error <= anchor_tolerance
                     and passive_difference <= 1e-8 and bias_difference <= 1e-8
                     and not any(item.number for item in data.warning))
        if not converged:
            return restore_with("not_converged", "native force, support or anchor checks did not meet tolerances")
        status = "converged" if converged else "not_converged"
        message = str(getattr(solver_result, "message", "native solve completed"))
        result = _diagnostics(
            status,
            message,
            calls,
            support_calls,
            solver_result,
            support_result,
            initial_contacts,
            support_adjustment,
            final_raw,
            final_bias,
            final_passive,
            data,
            endpoint,
            target_anchor,
            started,
            restored=False,
        )
        result.update({
            "initial_residual_norm": float(np.linalg.norm(initial_residual)),
            "initial_residual_max_normalized": float(np.max(np.abs(initial_residual))),
            "native_force_residual_max": nontranslational_residual,
            "full_mass_acceleration_residual_max": full_mass_accel_residual,
            "normalized_native_torque_residual": normalized_torque_residual,
            "normalized_full_force_residual": full_force_residual,
            "manual_forward_passive_max_error": passive_difference,
            "manual_forward_bias_max_error": bias_difference,
            "anchor_error_m": anchor_error,
            "qpos_finite": qpos_finite,
            "force_residual_tolerance": force_residual_tolerance,
            "anchor_tolerance_m": anchor_tolerance,
            "callback_wait_seconds": callback_time[0],
            "active_wall_seconds": time.monotonic() - started - callback_time[0],
        })
        return result
    except _SolveAbort as error:
        return restore_with(error.status, str(error))
    except Exception as error:  # keep the caller's state safe on solver failures
        return restore_with("failed", repr(error))


def _diagnostics(
    status: str,
    message: str,
    calls: int,
    support_calls: int,
    solver_result: Any,
    support_result: Any,
    initial_contacts: int,
    support_adjustment: np.ndarray,
    raw: np.ndarray,
    bias: np.ndarray,
    passive: np.ndarray,
    data: Any,
    endpoint: int | None,
    anchor: np.ndarray,
    started: float,
    restored: bool,
) -> dict[str, Any]:
    anchor_error = None
    if endpoint is not None:
        anchor_error = float(np.linalg.norm(np.asarray(data.site_xpos[endpoint], dtype=float) - anchor))
    return {
        "status": status,
        "success": status == "converged",
        "message": message,
        "calls": calls,
        "support_calls": support_calls,
        "optimizer_success": None if solver_result is None else bool(getattr(solver_result, "success", False)),
        "support_success": None if support_result is None else bool(getattr(support_result, "success", False)),
        "initial_contacts": initial_contacts,
        "contacts": int(data.ncon),
        "support_adjustment": np.asarray(support_adjustment, dtype=float).tolist(),
        "native_force_residual_max": None if not np.all(np.isfinite(raw)) else float(np.max(np.abs(raw[3:]))),
        "native_global_residual_max": None if not np.all(np.isfinite(raw)) else float(np.max(np.abs(raw))),
        "anchor_error_m": anchor_error,
        "qpos_finite": bool(np.all(np.isfinite(np.asarray(data.qpos, dtype=float)))),
        "restored": restored,
        "wall_seconds": time.monotonic() - started,
    }


__all__ = ["solve_static_equilibrium"]
