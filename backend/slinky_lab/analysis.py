"""Reference comparison and frame-series diagnostics."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .schemas import RunConfig


def _series(values: Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(list(values), dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def compare_reference(
    time: Iterable[float],
    values: Iterable[float],
    reference_time: Iterable[float],
    reference_values: Iterable[float],
) -> dict[str, Any]:
    """Interpolate a reference trace and return reproducible error metrics."""

    sample_time = _series(time, "time")
    sample_values = _series(values, "values")
    ref_time = _series(reference_time, "reference_time")
    ref_values = _series(reference_values, "reference_values")
    if sample_time.size != sample_values.size:
        raise ValueError("time and values must have equal length")
    if ref_time.size != ref_values.size:
        raise ValueError("reference_time and reference_values must have equal length")
    if sample_time.size > 1 and np.any(np.diff(sample_time) <= 0.0):
        raise ValueError("time must be strictly increasing")
    order = np.argsort(ref_time, kind="stable")
    ref_time = ref_time[order]
    ref_values = ref_values[order]
    if ref_time.size > 1 and np.any(np.diff(ref_time) <= 0.0):
        raise ValueError("reference_time must be strictly increasing and contain no duplicates")
    lower = max(float(sample_time[0]), float(ref_time[0]))
    upper = min(float(sample_time[-1]), float(ref_time[-1]))
    common = (sample_time >= lower) & (sample_time <= upper)
    if not np.any(common):
        raise ValueError("sample and reference traces have no common time range")
    sample_time = sample_time[common]
    sample_values = sample_values[common]
    reference_at_sample = np.interp(sample_time, ref_time, ref_values)
    residual = sample_values - reference_at_sample
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    max_abs = float(np.max(np.abs(residual)))
    return {
        "count": int(sample_time.size),
        "time": sample_time.tolist(),
        "reference": reference_at_sample.tolist(),
        "values": sample_values.tolist(),
        "residual": residual.tolist(),
        "rmse": rmse,
        "mae": mae,
        "max_abs_error": max_abs,
        "max_error": max_abs,
        "bias": float(np.mean(residual)),
        "within_reference_range": True,
        "common_time_range": [lower, upper],
    }


def summarize_frames(frames: Iterable[dict[str, Any]], config: RunConfig | dict[str, Any] | None = None) -> dict[str, Any]:
    """Summarize a sequence of API frames without rerunning the solver."""

    frame_list = list(frames)
    if not frame_list:
        raise ValueError("frames must contain at least one frame")
    times = _series((frame["time"] for frame in frame_list), "frame times")
    metric_names = (
        "top_z", "bottom_z", "spatial_top_z", "spatial_bottom_z", "material_top_z", "material_bottom_z",
        "material_first_z", "material_last_z", "material_vertical_span", "com_x", "com_y", "com_z",
        "com_vx", "com_vy", "com_vz", "kinetic_energy", "gravitational_energy",
        "cable_work", "damping_work", "contact_work", "steps_descended", "max_penetration",
        "nonadjacent_self_contacts", "stair_contact_events", "max_nonadjacent_self_contacts", "max_stair_contacts",
    )
    metrics: dict[str, dict[str, float]] = {}
    for name in metric_names:
        values = np.asarray([float(frame.get("metrics", {}).get(name, float("nan"))) for frame in frame_list], dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size:
            metrics[name] = {"initial": float(finite[0]), "final": float(finite[-1]), "min": float(np.min(finite)), "max": float(np.max(finite))}
    result: dict[str, Any] = {
        "frame_count": len(frame_list),
        "time_start": float(times[0]),
        "time_end": float(times[-1]),
        "metrics": metrics,
        "energy_diagnostic_approximate": True,
    }
    if config is not None:
        if isinstance(config, RunConfig):
            result["scenario"] = config.scenario
            result["config"] = config.model_dump(mode="json")
        else:
            result["scenario"] = config.get("scenario")
            result["config"] = config
    if "steps_descended" in metrics:
        result["steps_descended"] = metrics["steps_descended"]["max"]
    if "max_penetration" in metrics:
        result["max_penetration"] = metrics["max_penetration"]["max"]
    if "kinetic_energy" in metrics and "gravitational_energy" in metrics:
        initial_energy = metrics["kinetic_energy"]["initial"] + metrics["gravitational_energy"]["initial"]
        final_energy = metrics["kinetic_energy"]["final"] + metrics["gravitational_energy"]["final"]
        result["reported_energy_change"] = final_energy - initial_energy
    return result


def summarize_sweep(
    results: Iterable[dict[str, Any]],
    objective: str = "steps_descended",
    maximize: bool = True,
) -> dict[str, Any]:
    """Rank completed solver summaries from a bounded parameter scan."""

    rows: list[dict[str, Any]] = []
    successful_rows: list[dict[str, Any]] = []
    for index, item in enumerate(results):
        summary = item.get("summary", item)
        metrics = summary.get("metrics", {})
        value = summary.get(objective, metrics.get(objective))
        config = item.get("config", summary.get("config", {}))
        status = summary.get("status", item.get("status", "unknown"))
        success = value is not None and np.isfinite(float(value)) and status not in {"failed", "error", "cancelled"}
        row = {
            "index": index,
            "objective": float(value) if success else None,
            "scenario": summary.get("scenario", config.get("scenario") if isinstance(config, dict) else None),
            "config": config,
            "status": status,
            "success": success,
            "error": item.get("error", summary.get("error")),
        }
        rows.append(row)
        if success:
            successful_rows.append(row)
    successful_rows.sort(key=lambda row: row["objective"], reverse=maximize)
    return {
        "count": len(rows),
        "successful_count": len(successful_rows),
        "failed_count": len(rows) - len(successful_rows),
        "objective": objective,
        "maximize": maximize,
        "rows": rows,
        "best": successful_rows[0] if successful_rows else None,
        "worst": successful_rows[-1] if successful_rows else None,
    }
