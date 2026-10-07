"""Validate the literature-sized drop preset without changing its configuration.

The solver run is delegated to ``validate_stairs.run_case`` so that this
single-case tool uses the same partial-frame, warning, timeout, and HDF5
export path as the staircase validation.  Static fitting and dynamic timing
are reported as separate evidence classes; the literature collapse time is a
comparison datum, not a newly invented acceptance threshold.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from slinky_lab.presets import get_preset  # noqa: E402
from validate_stairs import run_case  # noqa: E402


PRESET_ID = "literature-39-turn"
LITERATURE_COLLAPSE_TIME_S = 0.27
LITERATURE_SUSPENDED_LENGTH_M = 1.14
EXPECTED_ARTIFACTS = (
    "config.json",
    "model.xml",
    "metadata.json",
    "trajectory.h5",
    "metrics.csv",
    "frames.json",
    "summary.json",
    "warnings.json",
)


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def finite_tree(value: Any) -> bool:
    if value is None or isinstance(value, (bool, str, int)):
        return True
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    return True


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def read_json_file(path: Path) -> tuple[Any, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {}, f"{path.name}: {type(exc).__name__}: {exc}"


def literature_spec(config: Any) -> dict[str, Any]:
    """Return the exact numerical values already present in the preset."""

    return {
        "name": "literature-39-turn",
        "segments_per_turn": int(config.numerics.segments_per_turn),
        "timestep": float(config.numerics.timestep),
        "wall_seconds": float(config.numerics.max_wall_seconds),
    }


def freefall_diagnostic(frames: list[dict[str, Any]], gravity: float) -> dict[str, Any]:
    if len(frames) < 2:
        return {
            "status": "not_observed",
            "samples": len(frames),
            "max_absolute_error_m": None,
            "relative_error": None,
            "reason": "fewer than two finite sampled frames were exported",
        }
    try:
        times = np.asarray([float(frame["time"]) for frame in frames], dtype=float)
        heights = np.asarray([float(frame["metrics"]["com_z"]) for frame in frames], dtype=float)
        initial_velocity = float(frames[0]["metrics"]["com_vz"])
        if not (np.isfinite(times).all() and np.isfinite(heights).all() and math.isfinite(initial_velocity)):
            raise ValueError("freefall inputs contain non-finite values")
        expected = heights[0] + initial_velocity * times - 0.5 * float(gravity) * times**2
        error = float(np.max(np.abs(heights - expected)))
        scale = float(max(0.5 * float(gravity) * times[-1] ** 2, 1.0e-8))
        return {
            "status": "observed",
            "samples": int(times.size),
            "time_range_s": [float(times[0]), float(times[-1])],
            "max_absolute_error_m": error,
            "relative_error": error / scale,
            "initial_com_vz_mps": initial_velocity,
            "definition": "sampled COM z compared with release position plus release velocity minus 0.5*g*t^2; no experimental acceptance threshold applied",
        }
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        return {
            "status": "not_observed",
            "samples": len(frames),
            "max_absolute_error_m": None,
            "relative_error": None,
            "reason": f"invalid exported frame series: {type(exc).__name__}: {exc}",
        }


def equilibrium_diagnostic(summary: dict[str, Any]) -> dict[str, Any]:
    static_value = summary.get("static_solve")
    thresholds_value = summary.get("settle_thresholds")
    static_solve = static_value if isinstance(static_value, dict) else {}
    thresholds = thresholds_value if isinstance(thresholds_value, dict) else {}
    force_residual = summary.get("settle_force_residual")
    anchor_error = summary.get("settle_anchor_error_m")
    force_residual_value = finite_number(force_residual)
    anchor_error_value = finite_number(anchor_error)
    finite_values = force_residual_value is not None and anchor_error_value is not None
    numerically_converged = bool(
        summary.get("prepare_status") == "converged"
        and static_solve.get("status") in {"converged", "already_converged"}
        and finite_values
        and force_residual_value <= float(thresholds.get("relative_force_residual", 0.01))
        and anchor_error_value <= float(thresholds.get("anchor_error_m", 0.0002))
    )
    return {
        "status": "observed",
        "prepare_status": summary.get("prepare_status"),
        "static_solve_status": static_solve.get("status"),
        "settle_force_residual": force_residual,
        "settle_anchor_error_m": anchor_error,
        "thresholds": thresholds,
        "numerically_converged": numerically_converged,
        "note": "native held-equilibrium diagnostic; this is not an independent material validation",
    }


def static_fit_diagnostic(summary: dict[str, Any]) -> dict[str, Any]:
    measured = finite_number(summary.get("static_equilibrium_length"))
    if measured is None:
        return {
            "status": "not_observed",
            "measured_static_equilibrium_length_m": None,
            "target_suspended_length_m": LITERATURE_SUSPENDED_LENGTH_M,
            "absolute_error_m": None,
            "relative_error": None,
            "reason": "static equilibrium length was not finite in the solver summary",
            "evidence_boundary": "static fitting is absent; no dynamic conclusion follows",
        }
    absolute_error = abs(measured - LITERATURE_SUSPENDED_LENGTH_M)
    return {
        "status": "observed",
        "measured_static_equilibrium_length_m": measured,
        "target_suspended_length_m": LITERATURE_SUSPENDED_LENGTH_M,
        "absolute_error_m": absolute_error,
        "relative_error": absolute_error / LITERATURE_SUSPENDED_LENGTH_M,
        "evidence_boundary": "this preset participates in a static-length fit; the fit is not independent experimental validation",
    }


def dynamic_collapse_comparison(summary: dict[str, Any]) -> dict[str, Any]:
    prepared = bool(summary.get("prepared"))
    released = bool(summary.get("released"))
    observed = finite_number(summary.get("collapse_time")) if prepared and released else None
    base = {
        "literature_source": "Cross and Wheatland (2012), plastic B report",
        "literature_source_url": "https://arxiv.org/abs/1208.4629",
        "literature_collapse_time_s": LITERATURE_COLLAPSE_TIME_S,
        "reference_type": "Table II best-fit model total collapse time; not an independently measured event timestamp",
        "simulation_collapse_time_s": observed,
        "comparison_type": "dynamic prediction versus Table II best-fit model timing; no acceptance threshold applied",
        "experimental_support": False,
    }
    if not prepared or not released:
        base.update({
            "status": "preparation_not_completed",
            "relative_error": None,
            "reason": "preparation_not_completed",
        })
        return base
    if observed is None:
        base.update({
            "status": "event_not_observed",
            "relative_error": None,
            "reason": f"collapse event was not observed before the configured simulation duration ({summary.get('duration')} s)",
        })
        return base
    base.update({
        "status": "compared",
        "relative_error": abs(observed - LITERATURE_COLLAPSE_TIME_S) / LITERATURE_COLLAPSE_TIME_S,
    })
    return base


def event_diagnostic(summary: dict[str, Any], key: str) -> dict[str, Any]:
    """Keep held-preparation clocks and event times out of released dynamics."""

    if not bool(summary.get("prepared")) or not bool(summary.get("released")):
        return {"time_s": None, "reason": "preparation_not_completed"}
    value = finite_number(summary.get(key))
    if value is None:
        return {
            "time_s": None,
            "reason": f"event not observed before the configured simulation duration ({summary.get('duration')} s)",
        }
    return {"time_s": value, "reason": None}


def artifact_diagnostic(case_dir: Path) -> dict[str, Any]:
    present = [name for name in EXPECTED_ARTIFACTS if (case_dir / name).is_file()]
    missing = [name for name in EXPECTED_ARTIFACTS if name not in present]
    return {"expected": list(EXPECTED_ARTIFACTS), "present": present, "missing": missing, "complete": not missing}


def report_markdown(report: dict[str, Any]) -> str:
    case = report.get("case", {})
    equilibrium = report.get("native_held_equilibrium", {})
    static_fit = report.get("static_fit", {})
    freefall = report.get("com_freefall", {})
    collapse = report.get("dynamic_collapse_comparison", {})
    lines = [
        "# Literature 39-turn drop validation",
        "",
        "This is one real solver run using the existing `literature-39-turn` preset without changing material parameters, geometry, initial state, or numerical settings. Static fitting and dynamic timing are reported as separate evidence classes.",
        "",
        f"- status: `{case.get('status')}`; phase: `{case.get('phase')}` (execution phase: `{case.get('execution_phase')}`, failure phase: `{case.get('failure_phase')}`); wall time: `{case.get('wall_seconds')} s`; model clock: `{case.get('model_clock_s')} s`; released integration time: `{case.get('released_integration_time_s')} s`",
        f"- timeout/native warning/nonfinite/export error: `{case.get('timeout_detected')}` / `{case.get('native_warning_detected')}` / `{case.get('nonfinite_detected')}` / `{case.get('export_error')}`",
        f"- model version: `{case.get('model_version')}`; artifacts complete: `{case.get('artifacts_complete')}`",
        "",
        "## Native held equilibrium",
        "",
        f"- prepare status: `{equilibrium.get('prepare_status')}`; static solver status: `{equilibrium.get('static_solve_status')}`",
        f"- force residual: `{equilibrium.get('settle_force_residual')}`; anchor error: `{equilibrium.get('settle_anchor_error_m')}`",
        f"- numerical equilibrium diagnostic: `{equilibrium.get('numerically_converged')}`",
        "",
        "## Static length fit",
        "",
        f"- simulated static length: `{static_fit.get('measured_static_equilibrium_length_m')} m`; reported target: `{static_fit.get('target_suspended_length_m')} m`",
        f"- absolute error: `{static_fit.get('absolute_error_m')} m`; relative error: `{static_fit.get('relative_error')}`",
        f"- evidence boundary: {static_fit.get('evidence_boundary')}",
        "",
        "## Drop dynamics",
        "",
        f"- COM freefall status: `{freefall.get('status')}`; samples: `{freefall.get('samples')}`; relative error: `{freefall.get('relative_error')}`",
        f"- bottom onset: `{report.get('bottom_onset', {}).get('time_s')}` s; reason: `{report.get('bottom_onset', {}).get('reason')}`",
        f"- collapse: `{report.get('collapse', {}).get('time_s')}` s; reason: `{report.get('collapse', {}).get('reason')}`",
        f"- comparison with reported plastic-B collapse at `{collapse.get('literature_collapse_time_s')} s`: status `{collapse.get('status')}`, relative error `{collapse.get('relative_error')}`",
        f"- dynamic evidence boundary: {collapse.get('comparison_type')}; experimental support `{collapse.get('experimental_support')}`",
        "",
        "All event-not-observed values remain `null`; no terminal frame was substituted for a missing event.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the literature-39-turn drop validation.")
    parser.add_argument("--output", type=Path, default=Path("reports/literature-drop-validation"))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).isoformat()
    case_dir = output / PRESET_ID
    run_error: str | None = None
    record: dict[str, Any] = {}
    base = None
    try:
        base = get_preset(PRESET_ID)
        if base.scenario != "drop":
            raise ValueError(f"Preset {PRESET_ID!r} must use scenario='drop'")
        spec = literature_spec(base)
        record = run_case(base, spec, case_dir)
    except Exception as exc:
        run_error = f"{type(exc).__name__}: {exc}"
        record = {
            "name": PRESET_ID,
            "status": "failed",
            "error": run_error,
            "wall_seconds": None,
            "simulation_time": None,
            "phase": "failed",
            "execution_phase": "preparation",
            "failure_phase": "preparation",
            "model_clock_s": None,
            "released_integration_time_s": 0.0,
            "timeout_detected": False,
            "native_warning_detected": False,
            "nonfinite_detected": False,
        }
    summary: dict[str, Any] = {}
    summary_path = case_dir / "summary.json"
    if summary_path.is_file():
        summary_value, read_error = read_json_file(summary_path)
        if read_error:
            run_error = run_error or f"summary read failed: {read_error}"
        elif isinstance(summary_value, dict):
            summary = summary_value
        else:
            run_error = run_error or "summary.json did not contain an object"
    artifacts = artifact_diagnostic(case_dir)
    config_match = False
    if base is not None and (case_dir / "config.json").is_file():
        config_data, read_error = read_json_file(case_dir / "config.json")
        config_match = not read_error and config_data == base.model_dump(mode="json")
        if read_error:
            run_error = run_error or f"config read failed: {read_error}"
    warnings_value = summary.get("warnings", {})
    warnings_payload = warnings_value if isinstance(warnings_value, dict) else {}
    if warnings_value and not isinstance(warnings_value, dict):
        run_error = run_error or "summary.json warnings field was not an object"
    simulation_summary = summary
    metadata_data: dict[str, Any] = {}
    metadata_path = case_dir / "metadata.json"
    if metadata_path.is_file():
        metadata_value, read_error = read_json_file(metadata_path)
        if read_error:
            run_error = run_error or f"metadata read failed: {read_error}"
        elif isinstance(metadata_value, dict):
            metadata_data = metadata_value
    frames_data: list[dict[str, Any]] = []
    frames_path = case_dir / "frames.json"
    if frames_path.is_file():
        frames_value, read_error = read_json_file(frames_path)
        if read_error:
            run_error = run_error or f"frames read failed: {read_error}"
        elif isinstance(frames_value, list):
            frames_data = frames_value
    release_ready = bool(summary.get("prepared")) and bool(summary.get("released"))
    case = {
        **record,
        "model_version": metadata_data.get("model_version"),
        "simulation_time": record.get("simulation_time", 0.0),
        "phase": record.get("phase", "released" if release_ready else "preparation"),
        "execution_phase": record.get("execution_phase", "released" if release_ready else "preparation"),
        "failure_phase": record.get("failure_phase"),
        "release_ready": record.get("release_ready", release_ready),
        "model_clock_s": record.get("model_clock_s", finite_number(summary.get("time"))),
        "released_integration_time_s": record.get("released_integration_time_s", record.get("simulation_time", 0.0) if release_ready else 0.0),
        "duration": summary.get("duration", base.numerics.duration if base is not None else None),
        "warning_detected": bool(record.get("warning_detected") or warnings_payload.get("warning_detected")),
        "native_warning_detected": bool(record.get("native_warning_detected") or warnings_payload.get("native_warning_detected")),
        "nonfinite_detected": bool(record.get("nonfinite_detected") or warnings_payload.get("nonfinite_detected")),
        "timeout_detected": bool(record.get("timeout_detected") or warnings_payload.get("timeout_detected")),
        "artifacts_complete": artifacts["complete"],
        "artifact_diagnostic": artifacts,
        "config_matches_preset": config_match,
        "export_error": bool(record.get("hdf5_error") or record.get("frame_errors") or run_error or not artifacts["complete"]),
    }
    case["bottom_onset"] = event_diagnostic(simulation_summary, "bottom_onset_time")
    case["collapse"] = event_diagnostic(simulation_summary, "collapse_time")
    report = {
        "created_at": created_at,
        "preset": PRESET_ID,
        "case": case,
        "native_held_equilibrium": equilibrium_diagnostic(simulation_summary),
        "static_fit": static_fit_diagnostic(simulation_summary),
        "com_freefall": freefall_diagnostic(
            frames_data,
            float(base.scene.gravity) if base is not None else 9.81,
        ),
        "bottom_onset": case["bottom_onset"],
        "collapse": case["collapse"],
        "dynamic_collapse_comparison": dynamic_collapse_comparison(simulation_summary),
        "evidence_boundary": {
            "static": "The preset's static length participates in a fit to the reported 1.14 m suspended length; this is not independent experimental validation.",
            "dynamic": "The simulated endpoint-span collapse event is compared with the Table II fitted-model total collapse time of 0.27 s for plastic B. Event definitions and tied top turns differ; this is a literature-model comparison without an experimental pass threshold.",
        },
    }
    report["execution_passed"] = bool(
        case.get("status") == "completed"
        and not case.get("warning_detected")
        and not case.get("native_warning_detected")
        and not case.get("nonfinite_detected")
        and not case.get("timeout_detected")
        and not case.get("export_error")
        and case.get("config_matches_preset")
    )
    report["passed"] = report["execution_passed"] and report["native_held_equilibrium"]["numerically_converged"]
    report["acceptance_scope"] = "completed warning-free run with native held equilibrium and complete exports; not experimental validation or a timing-error threshold"
    write_json(output / "validation-summary.json", report)
    (output / "validation-report.md").write_text(report_markdown(report), encoding="utf-8")
    print(json.dumps({"report": str(output / "validation-summary.json"), "passed": report["passed"]}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
