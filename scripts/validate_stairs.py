"""Run the bounded staircase convergence validation.

The tool is deliberately an offline runner around the same ``Simulation``
class used by the service.  It writes each case before evaluating convergence,
so a timeout or solver failure leaves the frames that were actually produced.
The extended mode is intended for manual Linux dispatch; ``--quick`` runs only
the base case and never claims a convergence result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from slinky_lab import MODEL_VERSION
from slinky_lab.physics import Simulation
from slinky_lab.presets import get_preset
from slinky_lab.schemas import RunConfig
from slinky_lab.storage import TrajectoryWriter


CASE_SPECS = (
    {"name": "baseline", "segments_per_turn": 8, "timestep": 5.0e-5, "wall_seconds": 900.0},
    {"name": "half_dt", "segments_per_turn": 8, "timestep": 2.5e-5, "wall_seconds": 1800.0},
    {"name": "mesh", "segments_per_turn": 16, "timestep": 1.25e-5, "wall_seconds": 3600.0},
)
PRESET_ID = "stairs-walking"
NATIVE_LOG_NAMES = ("MUJOCO_LOG.TXT", "mujoco.log", "mujoco_log.txt")
WARNING_TOKENS = ("warning", "nan", "inf", "reset", "unstable", "qacc", "qpos", "qvel")


def json_safe(value: Any) -> Any:
    """Convert numerical containers and non-finite values for honest JSON output."""

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


def native_log_baseline() -> dict[str, bytes]:
    return {str(path): path.read_bytes() for name in NATIVE_LOG_NAMES if (path := Path.cwd() / name).is_file()}


def native_log_delta(baseline: dict[str, bytes]) -> tuple[str, list[str]]:
    texts: list[str] = []
    paths: list[str] = []
    for name in NATIVE_LOG_NAMES:
        path = Path.cwd() / name
        if not path.is_file():
            continue
        raw = path.read_bytes()
        previous = baseline.get(str(path), b"")
        if raw.startswith(previous):
            raw = raw[len(previous):]
        text = raw.decode("utf-8", errors="replace")
        if text:
            texts.append(text)
            paths.append(str(path))
    return "\n".join(texts), paths


def warning_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if any(token in line.lower() for token in WARNING_TOKENS)]


def config_for_case(base: RunConfig, spec: dict[str, Any]) -> RunConfig:
    payload = base.model_dump(mode="json")
    payload["numerics"].update({
        "segments_per_turn": spec["segments_per_turn"],
        "timestep": spec["timestep"],
        "max_wall_seconds": spec["wall_seconds"],
    })
    return RunConfig.model_validate(payload)


def compact_support_event(event: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "step", "first_touch_time", "first_touch_label", "first_touch_force_n",
        "first_sustained_time", "first_sustained_label", "middle_support_time",
        "middle_support_peak_force_n", "middle_preceded", "endpoint_label",
        "endpoint_supports", "ambiguous", "ambiguity_reasons", "verified",
        "verified_time", "verification_reason", "transitions",
    )
    return {key: event.get(key) for key in keys}


def support_evidence(summary: dict[str, Any]) -> dict[str, Any]:
    diagnostics = summary.get("support_diagnostics") or {}
    events = [compact_support_event(event) for event in diagnostics.get("events", [])]
    first_three = events[:3]
    middle_first = [
        {
            key: event.get(key)
            for key in (
                "step", "first_touch_time", "first_touch_label", "first_touch_force_n",
                "first_sustained_time", "first_sustained_label", "middle_support_time",
                "middle_support_peak_force_n", "middle_preceded", "endpoint_label",
                "ambiguous", "ambiguity_reasons", "transitions",
            )
        }
        for event in events
        if event.get("first_touch_label") == "middle" or event.get("first_sustained_label") == "middle" or event.get("middle_preceded")
    ]
    return {
        "candidate_flip_count": diagnostics.get("candidate_flip_count", 0),
        "candidate_endpoint_sequence": diagnostics.get("candidate_endpoint_sequence", []),
        "confirmed_flip_count": diagnostics.get("confirmed_flip_count", 0),
        "confirmed_endpoint_sequence": diagnostics.get("confirmed_endpoint_sequence", []),
        "verified_steps": diagnostics.get("verified_steps", []),
        "first_three_endpoint_support_events": first_three,
        "middle_first_touch_evidence": middle_first,
        "all_event_count": len(events),
    }


def append_frame(simulation: Simulation, frames: list[dict[str, Any]], frame_errors: list[str]) -> dict[str, Any] | None:
    try:
        frame = simulation.frame()
        frames.append(frame)
        return frame
    except Exception as exc:  # preserve earlier real frames if a sampled frame cannot be encoded
        frame_errors.append(f"{type(exc).__name__}: {exc}")
        return None


def write_metrics_csv(path: Path, frames: list[dict[str, Any]]) -> None:
    keys: set[str] = set()
    for frame in frames:
        keys.update(str(key) for key in (frame.get("metrics") or {}))
    columns = ["time", *sorted(keys)]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for frame in frames:
            values = {"time": frame.get("time", 0.0)}
            values.update(frame.get("metrics") or {})
            writer.writerow({column: values.get(column, "") for column in columns})


def write_trajectory(path: Path, frames: list[dict[str, Any]], metadata: dict[str, Any], config: RunConfig, case_name: str) -> str | None:
    try:
        with TrajectoryWriter(path) as writer:
            for frame in frames:
                writer.append(frame)
        with h5py.File(path, "r+") as handle:
            handle.attrs["model_version"] = str(metadata.get("model_version", MODEL_VERSION))
            handle.attrs["engine_version"] = str(metadata.get("engine_version", ""))
            handle.attrs["coordinate_system"] = json.dumps(metadata.get("coordinate_system", {}), ensure_ascii=False)
            handle.attrs["config_json"] = json.dumps(config.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
            handle.attrs["validation_case"] = case_name
        return None
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def hash_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_case(base: RunConfig, spec: dict[str, Any], output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    config = config_for_case(base, spec)
    config_data = config.model_dump(mode="json")
    write_json(output / "config.json", config_data)
    started = time.monotonic()
    print(f"[{spec['name']}] start: {spec['segments_per_turn']} segments/turn, dt={spec['timestep']:g} s, duration={config.numerics.duration:g} s", flush=True)
    simulation: Simulation | None = None
    metadata: dict[str, Any] = {}
    simulation_summary: dict[str, Any] = {}
    frames: list[dict[str, Any]] = []
    frame_errors: list[str] = []
    python_warnings: list[dict[str, str]] = []
    error: str | None = None
    timed_out = False
    last_accepted_time = 0.0
    native_baseline = native_log_baseline()
    next_heartbeat = 30.0

    def should_cancel() -> bool:
        nonlocal next_heartbeat
        elapsed = time.monotonic() - started
        if elapsed >= next_heartbeat:
            model_clock = simulation.time if simulation is not None else 0.0
            print(f"[{spec['name']}] model clock {model_clock:.5f} s; wall {elapsed:.1f}/{spec['wall_seconds']:g} s", flush=True)
            next_heartbeat = elapsed + 30.0
        return elapsed > float(spec["wall_seconds"])

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            simulation = Simulation(config)
            (output / "model.xml").write_text(str(simulation.model_xml), encoding="utf-8")
            simulation.prepare(should_cancel=should_cancel)
            if should_cancel():
                timed_out = True
            if simulation is not None:
                try:
                    metadata = simulation.metadata()
                    write_json(output / "metadata.json", metadata)
                except Exception as exc:
                    frame_errors.append(f"metadata: {type(exc).__name__}: {exc}")
            if not timed_out and simulation.summary().get("status") != "cancelled":
                initial = append_frame(simulation, frames, frame_errors)
                sample_period = 1.0 / float(config.numerics.sample_hz)
                next_sample = sample_period
                next_progress = 0.2
                while simulation.time < simulation.duration - 1.0e-12:
                    if should_cancel():
                        timed_out = True
                        break
                    simulation.step()
                    last_accepted_time = simulation.time
                    if simulation.time + 1.0e-12 >= next_progress:
                        print(f"[{spec['name']}] model clock {simulation.time:.3f}/{simulation.duration:g} s; wall {time.monotonic() - started:.1f} s", flush=True)
                        next_progress += 0.2
                    if simulation.time + 1.0e-12 >= next_sample:
                        append_frame(simulation, frames, frame_errors)
                        next_sample += sample_period
                        while next_sample <= simulation.time + 1.0e-12:
                            next_sample += sample_period
                if not timed_out and simulation.time >= simulation.duration - 1.0e-12:
                    if not frames or abs(float(frames[-1]["time"]) - simulation.time) > 1.0e-12:
                        append_frame(simulation, frames, frame_errors)
            python_warnings = [{"category": item.category.__name__, "message": str(item.message)} for item in caught]
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        timed_out = timed_out or should_cancel()
    if simulation is not None:
        try:
            simulation_summary = simulation.summary()
        except Exception as exc:
            simulation_summary = {"summary_error": f"{type(exc).__name__}: {exc}"}
        if not metadata:
            try:
                metadata = simulation.metadata()
                write_json(output / "metadata.json", metadata)
            except Exception as exc:
                frame_errors.append(f"metadata: {type(exc).__name__}: {exc}")

    native_text, native_paths = native_log_delta(native_baseline)
    native_warnings = warning_lines(native_text)
    nonfinite_state = False
    if simulation is not None:
        try:
            nonfinite_state = not (np.isfinite(np.asarray(simulation.data.qpos)).all() and np.isfinite(np.asarray(simulation.data.qvel)).all())
        except Exception:
            nonfinite_state = True
    nonfinite_detected = nonfinite_state or not finite_tree(simulation_summary) or not finite_tree(frames)
    warning_detected = bool(native_warnings or python_warnings or nonfinite_detected)
    if native_text:
        (output / "native-log.txt").write_text(native_text, encoding="utf-8")
    warnings_payload = {
        "warning_detected": warning_detected,
        "native_warning_detected": bool(native_warnings),
        "native_log_paths": native_paths,
        "native_warning_lines": native_warnings,
        "python_warnings": python_warnings,
        "nonfinite_detected": nonfinite_detected,
        "timeout_detected": timed_out,
        "frame_errors": frame_errors,
        "rule": "Any MuJoCo/native warning, Python warning, non-finite state/value, or timeout fails convergence acceptance.",
    }
    write_json(output / "warnings.json", warnings_payload)

    if timed_out:
        status = "timeout"
    elif error is not None:
        status = "failed"
    elif simulation_summary.get("status") == "cancelled":
        status = "cancelled"
    elif simulation is None or simulation.time < config.numerics.duration - 1.0e-9:
        status = "failed"
        error = error or "Simulation ended before its configured duration"
    else:
        status = "completed"

    summary_time = simulation_summary.get("time")
    try:
        model_clock_s = float(summary_time)
        if not math.isfinite(model_clock_s):
            model_clock_s = None
    except (TypeError, ValueError):
        model_clock_s = None
    if model_clock_s is None and simulation is not None:
        try:
            candidate_clock = float(simulation.time)
            model_clock_s = candidate_clock if math.isfinite(candidate_clock) else None
        except (TypeError, ValueError):
            model_clock_s = None
    prepared = bool(simulation_summary.get("prepared", getattr(simulation, "_prepared", False)))
    released = bool(simulation_summary.get("released", getattr(simulation, "_released", False)))
    release_ready = prepared and released
    terminal_failure = bool(error or timed_out or status in {"failed", "timeout", "cancelled", "unstable"})
    execution_phase = "released" if release_ready else "preparation"
    if terminal_failure:
        phase = "failed"
        failure_phase = "released_integration" if release_ready else "preparation"
    else:
        phase = execution_phase
        failure_phase = None
    released_integration_time_s = last_accepted_time if release_ready else 0.0
    simulation_time_s = released_integration_time_s
    engine_time_after_failure_s = simulation.time if simulation is not None and terminal_failure else None
    post_failure_metrics_valid = not bool(terminal_failure or warning_detected)

    hdf5_error = write_trajectory(output / "trajectory.h5", frames, metadata, config, spec["name"])
    write_metrics_csv(output / "metrics.csv", frames)
    if simulation_summary.get("support_diagnostics") is not None:
        write_json(output / "support-diagnostics.json", simulation_summary["support_diagnostics"])
    support = support_evidence(simulation_summary)
    write_json(output / "support-events.json", support)
    write_json(output / "frames.json", frames)
    wall_seconds = time.monotonic() - started
    summary_payload = dict(simulation_summary)
    summary_payload.update({
        "validation_case": spec["name"],
        "validation_status": status,
        "validation_error": error,
        "wall_seconds": wall_seconds,
        "wall_budget_seconds": spec["wall_seconds"],
        "frame_count": len(frames),
        "hdf5_error": hdf5_error,
        "warnings": warnings_payload,
        "support_evidence": support,
        "last_accepted_time_s": last_accepted_time,
        "last_exported_frame_time_s": frames[-1]["time"] if frames else None,
        "engine_time_after_failure_s": engine_time_after_failure_s,
        "post_failure_instantaneous_metrics_valid": post_failure_metrics_valid,
        "phase": phase,
        "execution_phase": execution_phase,
        "failure_phase": failure_phase,
        "model_clock_s": model_clock_s,
        "released_integration_time_s": released_integration_time_s,
        "release_ready": release_ready,
    })
    write_json(output / "summary.json", summary_payload)
    record = {
        "name": spec["name"],
        "status": status,
        "error": error,
        "wall_seconds": wall_seconds,
        "wall_budget_seconds": spec["wall_seconds"],
        # ``model_clock_s`` preserves data.time, including held preparation and
        # BADQACC reset clocks; ``simulation_time`` is released accepted time.
        "simulation_time": simulation_time_s,
        "phase": phase,
        "execution_phase": execution_phase,
        "failure_phase": failure_phase,
        "prepared": prepared,
        "released": released,
        "release_ready": release_ready,
        "model_clock_s": model_clock_s,
        "released_integration_time_s": released_integration_time_s,
        "engine_time_after_failure_s": engine_time_after_failure_s,
        "post_failure_instantaneous_metrics_valid": post_failure_metrics_valid,
        "duration": config.numerics.duration,
        "config": config_data,
        "warning_detected": warning_detected,
        "native_warning_detected": bool(native_warnings),
        "nonfinite_detected": nonfinite_detected,
        "timeout_detected": timed_out,
        "frame_count": len(frames),
        "frame_errors": frame_errors,
        "hdf5_error": hdf5_error,
        "support": support,
        "movement_classification": simulation_summary.get("movement_classification"),
        "max_penetration": simulation_summary.get("metrics", {}).get("max_penetration"),
        "final_com_velocity": [
            simulation_summary.get("metrics", {}).get("com_vx"),
            simulation_summary.get("metrics", {}).get("com_vy"),
            simulation_summary.get("metrics", {}).get("com_vz"),
        ],
        "artifacts": sorted(item.name for item in output.iterdir() if item.is_file()),
        "artifact_sha256": {item.name: hash_file(item) for item in output.iterdir() if item.is_file()},
        "summary_path": str(output / "summary.json"),
    }
    print(f"[{spec['name']}] {status}: confirmed={support['confirmed_flip_count']}, sequence={support['confirmed_endpoint_sequence']}, warnings={warning_detected}, wall={wall_seconds:.1f} s", flush=True)
    return record


def case_quality(record: dict[str, Any]) -> dict[str, Any]:
    support = record.get("support", {})
    first_three = support.get("first_three_endpoint_support_events", [])
    first_three_steps = [event.get("step") for event in first_three[:3]]
    passed = bool(
        record.get("status") == "completed"
        and not record.get("warning_detected")
        and not record.get("native_warning_detected")
        and not record.get("nonfinite_detected")
        and not record.get("timeout_detected")
        and not record.get("frame_errors")
        and not record.get("hdf5_error")
        and float(record.get("simulation_time") or 0.0) >= float(record.get("duration") or 0.0) - 1.0e-9
        and int(support.get("confirmed_flip_count", 0) or 0) >= 3
        and first_three_steps == [1, 2, 3]
        and all(event.get("verified") and not event.get("ambiguous") and endpoint_support_time(event) is not None for event in first_three[:3])
    )
    return {"passed": passed, "reason": "completed, warning-free, finite, full duration, complete frame/export artifacts, and three confirmed supports on contiguous steps 1-3"}


def sequence_for(record: dict[str, Any]) -> list[str | None]:
    return [event.get("endpoint_label") for event in record.get("support", {}).get("first_three_endpoint_support_events", [])[:3]]


def endpoint_support_time(event: dict[str, Any]) -> float | None:
    label = event.get("endpoint_label")
    supports = event.get("endpoint_supports") or []
    if label not in {"first", "last"} or not supports or supports[0].get("label") != label:
        return None
    value = supports[0].get("time")
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def times_for(record: dict[str, Any]) -> list[float | None]:
    return [endpoint_support_time(event) for event in record.get("support", {}).get("first_three_endpoint_support_events", [])[:3]]


def physical_signature(record: dict[str, Any]) -> dict[str, Any]:
    config = record.get("config", {})
    numerics = config.get("numerics", {})
    fixed_numerics = {key: value for key, value in numerics.items() if key not in {"segments_per_turn", "timestep", "max_wall_seconds"}}
    return {"scenario": config.get("scenario"), "material": config.get("material", {}), "scene": config.get("scene", {}), "fixed_numerics": fixed_numerics}


def evaluate_convergence(records: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    if mode == "quick":
        return {
            "status": "not_evaluated",
            "passed": case_quality(records[0])["passed"] if records else False,
            "rule": "--quick runs baseline only and does not evaluate discretization convergence.",
        }
    if [record.get("name") for record in records] != [spec["name"] for spec in CASE_SPECS]:
        return {"status": "incomplete", "passed": False, "rule": "extended mode requires baseline, half_dt, and mesh"}
    quality = {record["name"]: case_quality(record) for record in records}
    baseline = records[0]
    baseline_sequence = sequence_for(baseline)
    baseline_times = times_for(baseline)
    physical_checks = [
        {
            "case": record["name"],
            "matches_baseline": physical_signature(record) == physical_signature(baseline),
        }
        for record in records[1:]
    ]
    sequence_checks: list[dict[str, Any]] = []
    time_checks: list[dict[str, Any]] = []
    for record in records[1:]:
        sequence = sequence_for(record)
        sequence_checks.append({
            "case": record["name"],
            "baseline": baseline_sequence,
            "case_sequence": sequence,
            "steps_are_1_to_3": [event.get("step") for event in record.get("support", {}).get("first_three_endpoint_support_events", [])[:3]] == [1, 2, 3],
            "passed": sequence == baseline_sequence and len(sequence) == 3,
        })
        case_times = times_for(record)
        comparisons = []
        for index in range(3):
            base_time = baseline_times[index] if index < len(baseline_times) else None
            case_time = case_times[index] if index < len(case_times) else None
            if base_time is None or case_time is None:
                comparisons.append({"step_index": index + 1, "baseline_s": base_time, "case_s": case_time, "passed": False, "status": "event_not_observed"})
                continue
            relative = abs(float(case_time) - float(base_time)) / max(abs(float(base_time)), 1.0e-12)
            comparisons.append({"step_index": index + 1, "baseline_s": base_time, "case_s": case_time, "relative_change": relative, "passed": relative <= 0.05})
        time_checks.append({"case": record["name"], "events": comparisons, "passed": all(item["passed"] for item in comparisons)})
    passed = (
        all(item["passed"] for item in quality.values())
        and all(item["matches_baseline"] for item in physical_checks)
        and all(item["passed"] for item in sequence_checks)
        and all(item["passed"] for item in time_checks)
    )
    return {
        "status": "evaluated",
        "passed": passed,
        "per_case_quality": quality,
        "physical_parameter_checks": physical_checks,
        "baseline_sequence": baseline_sequence,
        "baseline_support_times_s": baseline_times,
        "sequence_checks": sequence_checks,
        "support_time_checks": time_checks,
        "rule": "All three cases must be completed, warning-free, finite, non-timeout, have complete artifacts, >=3 confirmed flips, match the first three endpoint sequence, and keep each support-event time within 5% of baseline. The 5% stair-event threshold is an additional conservative numerical criterion, not experimental accuracy.",
    }


def write_report(output: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Stair walking numerical validation",
        "",
        f"- mode: `{report['mode']}`; base preset: `{PRESET_ID}`; model: `{report.get('model_version', MODEL_VERSION)}`",
        f"- created: `{report['created_at']}`",
        "- physical material and scene parameters are copied from the preset; only numerical segments, timestep, and wall budget vary by case.",
        "",
        "| case | status | wall s | simulated s | confirmed | endpoint sequence | warning/nonfinite/timeout |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for record in report.get("cases", []):
        support = record.get("support", {})
        sequence = sequence_for(record)
        bad = (record.get("warning_detected"), record.get("nonfinite_detected"), record.get("timeout_detected"))
        lines.append(f"| {record['name']} | {record['status']} | {record['wall_seconds']:.3f} | {record.get('simulation_time')} | {support.get('confirmed_flip_count', 0)} | `{sequence}` | `{bad}` |")
    lines.extend(["", "## Support evidence", ""])
    for record in report.get("cases", []):
        support = record.get("support", {})
        lines.append(f"### {record['name']}")
        lines.append("")
        lines.append(f"- first three endpoint support events: `{support.get('first_three_endpoint_support_events', [])}`")
        lines.append(f"- middle-first-touch evidence: `{support.get('middle_first_touch_evidence', [])}`")
        lines.append(f"- artifacts: `{record.get('artifacts', [])}`")
        lines.append("")
    convergence = report.get("convergence", {})
    lines.extend(["## Acceptance", "", f"- status: `{convergence.get('status')}`; passed: `{convergence.get('passed')}`", f"- rule: {convergence.get('rule')}", ""])
    if convergence.get("sequence_checks"):
        lines.append(f"- sequence checks: `{convergence['sequence_checks']}`")
    if convergence.get("physical_parameter_checks"):
        lines.append(f"- physical parameter checks: `{convergence['physical_parameter_checks']}`")
    if convergence.get("support_time_checks"):
        lines.append(f"- support time checks: `{convergence['support_time_checks']}`")
    (output / "validation-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_base_preset() -> RunConfig:
    try:
        config = get_preset(PRESET_ID)
    except KeyError as exc:
        raise RuntimeError(f"Required preset {PRESET_ID!r} is unavailable") from exc
    if config.scenario != "stairs":
        raise ValueError(f"Preset {PRESET_ID!r} must use scenario='stairs'")
    return config


def recheck_report(output: Path) -> dict[str, Any]:
    report = json.loads((output / "validation-summary.json").read_text(encoding="utf-8"))
    records = report.get("cases", [])
    for record in records:
        errors = list(record.get("frame_errors") or [])
        if record.get("name") not in {spec["name"] for spec in CASE_SPECS}:
            record["frame_errors"] = ["Unknown validation case"]
            continue
        for name, expected in record.get("artifact_sha256", {}).items():
            if Path(name).name != name or hash_file(output / record["name"] / name) != expected:
                errors.append(f"Missing or changed evidence artifact: {name}")
        if not record.get("artifact_sha256"):
            errors.append("No artifact integrity manifest")
        record["frame_errors"] = errors
    report["original_convergence"] = report.get("convergence")
    report["convergence"] = evaluate_convergence(records, report.get("mode", "extended"))
    report["passed"] = bool(report["convergence"].get("passed"))
    report["acceptance_recheck"] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "criterion": "verified unambiguous endpoint support times and matching artifact hashes",
        "checker_sha256": hash_file(Path(__file__)),
    }
    write_json(output / "validation-summary.json", report)
    write_report(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the bounded stairs-walking numerical validation.")
    parser.add_argument("--output", type=Path, default=Path("reports/stairs-validation"))
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--quick", action="store_true", help="Run only baseline; convergence is not evaluated.")
    mode_group.add_argument("--recheck", action="store_true", help="Verify existing artifact hashes and reevaluate endpoint support convergence; run no simulation.")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.recheck:
        report = recheck_report(output)
        print(json.dumps({"report": str(output / "validation-summary.json"), "mode": "recheck", "passed": report["passed"]}, ensure_ascii=False))
        return 0 if report["passed"] else 1
    mode = "quick" if args.quick else "extended"
    created_at = datetime.now(timezone.utc).isoformat()
    try:
        base = load_base_preset()
    except Exception as exc:
        report = {"mode": mode, "base_preset": PRESET_ID, "created_at": created_at, "cases": [], "convergence": {"status": "error", "passed": False}, "error": f"{type(exc).__name__}: {exc}", "passed": False}
        write_json(output / "validation-summary.json", report)
        write_report(output, report)
        print(json.dumps({"report": str(output / 'validation-summary.json'), "passed": False, "error": report["error"]}, ensure_ascii=False))
        return 1
    records: list[dict[str, Any]] = []
    specs = CASE_SPECS[:1] if args.quick else CASE_SPECS
    for spec in specs:
        records.append(run_case(base, spec, output / spec["name"]))
    report = {
        "mode": mode,
        "base_preset": PRESET_ID,
        "created_at": created_at,
        "model_version": MODEL_VERSION,
        "cases": records,
        "convergence": evaluate_convergence(records, mode),
    }
    report["passed"] = bool(report["convergence"].get("passed"))
    write_json(output / "validation-summary.json", report)
    write_report(output, report)
    print(json.dumps({"report": str(output / "validation-summary.json"), "mode": mode, "passed": report["passed"]}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
