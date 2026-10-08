"""Bounded passive-gait experiments using the production MuJoCo adapter.

Each invocation evaluates one declared case and one declared variant. Failure
is a research result, so every run retains its original configuration and
trajectory. A successful search is only a candidate for independent numerical
and initial-condition checks; it does not promote a preset or image.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from slinky_lab.presets import get_preset
from slinky_lab.schemas import RunConfig
from validate_stairs import run_case, write_json


# These are engineering hypotheses, not dimensions measured from the video.
# G factors scale both elastic moduli together, retaining the assumed Poisson
# ratio. No case applies forces or changes parameters after release.
CASES: dict[str, dict[str, Any]] = {
    "arched-control": {"preset": "stairs-arched"},
    "n12-g1-h06": {"scene": {"step_depth": 0.08, "step_height": 0.06}},
    "n12-g4-h06": {"modulus_factor": 4, "scene": {"step_depth": 0.08, "step_height": 0.06}, "dt": 1.25e-5},
    "n12-g16-h06": {"modulus_factor": 16, "scene": {"step_depth": 0.08, "step_height": 0.06}, "dt": 6.25e-6},
    "n12-g4-h04": {"modulus_factor": 4, "scene": {"step_depth": 0.08, "step_height": 0.04}, "dt": 1.25e-5},
    "n12-g4-h08": {"modulus_factor": 4, "scene": {"step_depth": 0.08, "step_height": 0.08}, "dt": 1.25e-5},
    "n12-g4-grip": {"modulus_factor": 4, "material": {"friction": 1.4}, "scene": {"step_depth": 0.08, "step_height": 0.06}, "dt": 1.25e-5},
    "n24-g4-h06": {"modulus_factor": 4, "material": {"turns": 24}, "scene": {"step_depth": 0.08, "step_height": 0.06, "arch_end_turns": 4}, "dt": 1e-5, "wall_seconds": 3600},
    "n36-g6-h06": {"modulus_factor": 6, "material": {"turns": 36}, "scene": {"step_depth": 0.08, "step_height": 0.06, "arch_end_turns": 8}, "dt": 6.25e-6, "wall_seconds": 4800},
    "macro-arch": {"preset": "stairs-walking", "scene": {"initial_pose": "arched", "step_height": 0.06, "initial_forward_velocity": 0, "initial_angular_velocity": 0}},
    "macro-tilt-control": {"preset": "stairs-walking"},
    "macro-tilt-h06": {"preset": "stairs-walking", "scene": {"step_height": 0.06}},
    "safe-g2-h06": {"modulus_factor": 2, "scene": {"step_depth": 0.12, "step_height": 0.06, "arch_rise": 0.06}, "dt": 1.25e-5},
    "safe-g4-h06": {"modulus_factor": 4, "scene": {"step_depth": 0.12, "step_height": 0.06, "arch_rise": 0.06}, "dt": 1.25e-5},
    "safe-g4-h04": {"modulus_factor": 4, "scene": {"step_depth": 0.12, "step_height": 0.04, "arch_rise": 0.06}, "dt": 1.25e-5},
    "safe-g4-rise": {"modulus_factor": 4, "scene": {"step_depth": 0.12, "step_height": 0.06, "arch_rise": 0.08}, "dt": 1.25e-5},
    "safe-g4-launch": {"modulus_factor": 4, "scene": {"step_depth": 0.12, "step_height": 0.06, "arch_rise": 0.06, "initial_angular_velocity": 1}, "dt": 1.25e-5},
    "safe-g4-grip": {"modulus_factor": 4, "material": {"friction": 1.4}, "scene": {"step_depth": 0.12, "step_height": 0.06, "arch_rise": 0.06}, "dt": 1.25e-5},
    "safe-g4-low-damp": {"modulus_factor": 4, "material": {"damping": 5e-6}, "scene": {"step_depth": 0.12, "step_height": 0.06, "arch_rise": 0.06}, "dt": 6.25e-6},
    "safe-g4-short": {"modulus_factor": 4, "scene": {"step_depth": 0.10, "step_height": 0.06, "arch_rise": 0.06, "arch_end_turns": 1}, "dt": 1.25e-5},
    "safe-n24-g8": {"modulus_factor": 8, "material": {"turns": 24}, "scene": {"step_depth": 0.12, "step_height": 0.06, "arch_rise": 0.06, "arch_end_turns": 4}, "dt": 6.25e-6, "wall_seconds": 4800},
}
VARIANTS = ("base", "half_dt", "mesh", "offset_minus", "offset_plus", "rise_minus", "rise_plus")


def build_config(case_id: str, variant: str = "base", duration: float | None = None, wall_seconds: float | None = None) -> RunConfig:
    if case_id not in CASES or variant not in VARIANTS:
        raise ValueError("Unknown declared gait case or variant")
    case = CASES[case_id]
    payload = get_preset(case.get("preset", "stairs-arched")).model_dump(mode="json")
    payload["name"] = f"被动三阶研究 {case_id} / {variant}"
    payload["material"].update(case.get("material", {}))
    payload["scene"].update(case.get("scene", {}))
    material = payload["material"]
    factor = float(case.get("modulus_factor", 1))
    material["young_modulus"] *= factor
    material["shear_modulus"] *= factor
    # Match the legacy 8-segment joint damping at the reference chord length,
    # then retain this SI viscosity for every mesh of the same physical case.
    reference_chord = math.hypot(2 * material["radius"] * math.sin(math.pi / 8), material["pitch"] / 8)
    material["rotational_viscosity"] = material["damping"] * reference_chord
    numerical = payload["numerics"]
    numerical.update({
        "profile": "fine", "segments_per_turn": 8,
        "timestep": case.get("dt", numerical["timestep"]),
        "duration": duration if duration is not None else 2.4,
        "sample_hz": 25,
        "max_wall_seconds": wall_seconds if wall_seconds is not None else case.get("wall_seconds", 1800),
    })
    if variant == "half_dt":
        numerical["timestep"] *= 0.5
        numerical["max_wall_seconds"] = min(14400, 2 * numerical["max_wall_seconds"])
    elif variant == "mesh":
        numerical["segments_per_turn"] *= 2
        numerical["timestep"] *= 0.25
        numerical["max_wall_seconds"] = min(14400, 4 * numerical["max_wall_seconds"])
    elif variant.startswith("offset_"):
        payload["scene"]["launch_offset"] += -0.002 if variant == "offset_minus" else 0.002
    elif variant.startswith("rise_"):
        if payload["scene"]["initial_pose"] != "arched":
            raise ValueError("Arch-rise perturbations require an arched initial pose")
        rise = payload["scene"]["arch_rise"]
        if rise is None:
            rise = payload["scene"]["step_depth"] * 0.5
        payload["scene"]["arch_rise"] = rise * (0.98 if variant == "rise_minus" else 1.02)
    payload["provenance"].update({
        "research_case": case_id, "research_variant": variant,
        "calibration": "engineering hypotheses; no experimental parameter calibration",
        "release": "initial pose/velocity only; no actuator or post-release driving",
        "damping": "isotropic rotational viscosity in N m^2 s, fixed under mesh refinement",
        "acceptance": "three sustained alternating endpoint supports, warnings/timeout rejected; then independent dt/mesh/perturbation checks",
    })
    return RunConfig.model_validate(payload)


def candidate_check(record: dict[str, Any], config: RunConfig) -> dict[str, Any]:
    support = record.get("support", {})
    penetration = record.get("max_penetration")
    checks = {
        "completed_release": record.get("status") == "completed" and record.get("release_ready") is True,
        "no_warning": not record.get("warning_detected", True),
        "no_timeout": not record.get("timeout_detected", True),
        "no_side_fall": record.get("movement_classification") != "side_fall",
        "trajectory_saved": not record.get("frame_errors") and not record.get("hdf5_error") and record.get("frame_count", 0) > 1,
        "three_alternating_supports": support.get("confirmed_flip_count", 0) >= 3,
        "penetration_bounded": penetration is not None and 0 <= penetration <= 0.5 * config.material.strip_thickness,
    }
    return {
        "checks": checks, "candidate_eligible_for_validation": all(checks.values()),
        "numerical_convergence_passed": False, "robustness_passed": False, "experimental_support": False,
        "note": "A search candidate is not a validated stable gait. Geometry/lift review and independent dt, mesh and perturbation runs remain required.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(CASES), default="arched-control")
    parser.add_argument("--variant", choices=VARIANTS, default="base")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--wall-seconds", type=float)
    parser.add_argument("--output", type=Path, default=Path("reports/gait-search"))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--preflight-only", action="store_true", help="Record initial geometry without preparing or integrating the model")
    args = parser.parse_args()
    if args.list:
        for case_id in CASES:
            config = build_config(case_id)
            print(f"{case_id}: {config.material.turns} turns, E={config.material.young_modulus:g}, h={config.scene.step_height:g}, d={config.scene.step_depth:g}, dt={config.numerics.timestep:g}")
        return
    config = build_config(args.case, args.variant, args.duration, args.wall_seconds)
    if args.preflight_only:
        from slinky_lab.physics import Simulation
        simulation = Simulation(config)
        output = args.output / f"{args.case}-{args.variant}"
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "preflight.json", {
            "config": config.model_dump(mode="json"), "phase": "preflight",
            "integration_steps": 0, "initial_geometry": simulation.summary()["initial_pose_diagnostics"],
        })
        initial = simulation.summary()["initial_pose_diagnostics"]
        print(f"{args.case}: preflight={initial.get('preflight_valid')}, penetration={initial['initial_max_penetration_m']:.9g} m, qacc={initial['initial_acceleration_max']:.9g}")
        return
    spec = {
        "name": f"{args.case}-{args.variant}",
        "segments_per_turn": config.numerics.resolved()["segments_per_turn"],
        "timestep": config.numerics.resolved()["timestep"],
        "wall_seconds": config.numerics.max_wall_seconds,
    }
    record = run_case(config, spec, args.output / spec["name"])
    assessment = candidate_check(record, config)
    write_json(args.output / "search-result.json", {
        "schema": "slinky-lab.passive-gait-search.v1", "case": args.case,
        "variant": args.variant, "record": record, "assessment": assessment,
    })
    print(f"{spec['name']}: {record['status']}; confirmed={record.get('support', {}).get('confirmed_flip_count', 0)}; candidate={assessment['candidate_eligible_for_validation']}", flush=True)


if __name__ == "__main__":
    main()
