"""Write actual solver diagnostics; never replace unavailable events with fabricated values."""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from slinky_lab import __version__
from slinky_lab.physics import Simulation
from slinky_lab.schemas import RunConfig


def run_case(config):
    started = time.monotonic()
    simulation = Simulation(config)
    simulation.prepare(should_cancel=lambda: time.monotonic() - started > config.numerics.max_wall_seconds)
    if time.monotonic() - started > config.numerics.max_wall_seconds:
        raise TimeoutError("Validation preparation exceeded its wall-time budget")
    frames = [simulation.frame()]
    next_sample = 1 / config.numerics.sample_hz
    while simulation.time < simulation.duration - simulation.dt / 2:
        simulation.step()
        if simulation.time + simulation.dt / 2 >= next_sample:
            frames.append(simulation.frame())
            next_sample += 1 / config.numerics.sample_hz
        if time.monotonic() - started > config.numerics.max_wall_seconds:
            raise TimeoutError("Validation case exceeded its wall-time budget")
    summary = simulation.summary()
    return {"config": config.model_dump(), "metadata": simulation.metadata(), "summary": summary,
            "wall_seconds": time.monotonic() - started}, frames


def freefall_check(frames, gravity):
    times = np.array([frame["time"] for frame in frames])
    positions = np.array([frame["metrics"]["com_z"] for frame in frames])
    # The release velocity is measured from the solver, independently of the later trajectory.
    velocity = frames[0]["metrics"]["com_vz"]
    expected = positions[0] + velocity * times - 0.5 * gravity * times ** 2
    error = float(np.max(np.abs(positions - expected)))
    scale = float(max(0.5 * gravity * times[-1] ** 2, 1e-8))
    return {"max_absolute_error_m": error, "relative_error": error / scale, "passed": error / scale <= 0.01}


def work_balance(frames):
    first, last = frames[0]["metrics"], frames[-1]["metrics"]
    mechanical_change = sum(last[key] - first[key] for key in ("kinetic_energy", "gravitational_energy"))
    work = sum(last[key] - first[key] for key in ("cable_work", "damping_work", "contact_work"))
    return {"kinetic_and_gravity_change_J": mechanical_change, "total_work_J": work,
            "residual_J": mechanical_change - work,
            "note": "Power integral diagnostic, not an assertion that contact/friction conserves mechanical energy."}


def compare_events(baseline, refinement):
    output = {}
    for name in ("bottom_onset_time", "collapse_time"):
        a, b = baseline.get(name), refinement.get(name)
        if a is None or b is None:
            output[name] = {"passed": False, "status": "event_not_observed", "baseline_s": a, "refined_s": b}
        else:
            relative = abs(a - b) / max(abs(b), 1e-12)
            output[name] = {"passed": relative <= 0.05, "relative_change": relative, "baseline_s": a, "refined_s": b}
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="docs/validation-results.json")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    config = RunConfig.model_validate({
        "name": "质心和时间步长检查", "scenario": "drop",
        "material": {"turns": 3, "mass": 0.012, "pitch": 0.002, "young_modulus": 1e8, "shear_modulus": 1e8 / 2.7},
        "scene": {"settle_time": 2},
        "numerics": {"duration": 0.18, "segments_per_turn": 8 if args.quick else 16,
                     "timestep": 0.000025 if args.quick else 0.0000125, "max_wall_seconds": 600},
    })
    report = {"application_version": __version__, "created_at": datetime.now(timezone.utc).isoformat(),
              "mode": "quick" if args.quick else "extended", "checks": {}}
    try:
        case, frames = run_case(config)
        report["baseline"] = case
        report["checks"]["held_native_equilibrium"] = {
            "passed": case["summary"]["prepare_status"] == "converged",
            "relative_force_residual": case["summary"]["settle_force_residual"],
            "anchor_error_m": case["summary"]["settle_anchor_error_m"],
        }
        report["checks"]["center_of_mass_freefall"] = freefall_check(frames, config.scene.gravity)
        report["checks"]["work_balance"] = work_balance(frames)
        refined = config.model_copy(deep=True)
        refined.numerics.timestep = config.numerics.timestep / 2
        second, fine_frames = run_case(refined)
        report["half_timestep"] = second
        t = [frame["time"] for frame in frames]
        z = np.array([frame["metrics"]["bottom_z"] for frame in frames])
        fine_z = np.interp(t, [f["time"] for f in fine_frames], [f["metrics"]["bottom_z"] for f in fine_frames])
        discrepancy = float(np.max(np.abs(z - fine_z)))
        report["checks"]["half_timestep_trajectory"] = {
            "max_absolute_error_m": discrepancy,
            "note": "Trajectory diagnostic; this does not certify collapse-time or experimental agreement.",
        }
        report["checks"]["half_timestep_events"] = compare_events(case["summary"], second["summary"])
        if not args.quick:
            refined_mesh = config.model_copy(deep=True)
            refined_mesh.numerics.segments_per_turn *= 2
            refined_mesh.numerics.timestep /= 4
            mesh_case, mesh_frames = run_case(refined_mesh)
            report["double_mesh"] = mesh_case
            report["checks"]["double_mesh_events"] = compare_events(second["summary"], mesh_case["summary"])
            report["checks"]["double_mesh_freefall"] = freefall_check(mesh_frames, config.scene.gravity)
        report["checks"]["experimental_validation"] = {"status": "not_established_by_this_check"}
        report["scope"] = "Held drop integration and event checks; staircase and experimental acceptance are reported separately."
        required = [report["checks"]["center_of_mass_freefall"]["passed"],
                    report["checks"]["held_native_equilibrium"]["passed"]]
        required.extend(value["passed"] for value in report["checks"]["half_timestep_events"].values())
        if not args.quick:
            required.extend(value["passed"] for value in report["checks"]["double_mesh_events"].values())
            required.append(report["checks"]["double_mesh_freefall"]["passed"])
        report["passed"] = all(required)
    except Exception as exc:
        report["passed"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"report": str(target), "passed": report["passed"]}))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
