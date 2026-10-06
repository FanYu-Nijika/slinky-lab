"""Reproduce the stated 39-turn static-length fit, not independent validation."""

import argparse
import json
from pathlib import Path

from slinky_lab.calibration import moduli_from_coil_stiffness
from slinky_lab.equilibrium import solve_static_equilibrium
from slinky_lab.physics import Simulation
from slinky_lab.schemas import Material, RunConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", type=int, default=8)
    parser.add_argument("--target-length", type=float, default=1.14)
    parser.add_argument("--stiffness", type=float, default=0.22)
    parser.add_argument("--budget", type=float, default=120)
    parser.add_argument("--output", default="reports/static-calibration.json")
    args = parser.parse_args()
    if args.target_length <= 0.066:
        parser.error("target length must exceed the compressed length")
    material = Material(turns=39, pitch=0.066 / 39, mass=0.0487)
    estimate = moduli_from_coil_stiffness(material, args.stiffness)
    factor = 1.0
    records = []
    for _ in range(3):
        config = RunConfig.model_validate({
            "material": {**material.model_dump(), "young_modulus": estimate["young_modulus_pa"] * factor,
                         "shear_modulus": estimate["shear_modulus_pa"] * factor},
            "numerics": {"segments_per_turn": args.segments, "timestep": 0.000025},
            "provenance": {"source": "https://arxiv.org/abs/1208.4629", "calibration": "static length fitted",
                           "assumptions": "R=.03m, width=.003m, thickness=.0015m, nu=.35"},
        })
        simulation = Simulation(config)
        simulation._apply_hanging_guess()
        diagnostics = solve_static_equilibrium(simulation.model, simulation.data, simulation._anchor,
                                               endpoint_site_id=simulation._endpoint_ids[0], max_calls=60000,
                                               time_limit_s=args.budget, length_scale=material.radius)
        length = simulation._material_vertical_span()
        error = abs(length - args.target_length) / args.target_length
        records.append({"config": config.model_dump(), "initial_mapping": estimate, "moduli_scale": factor,
                        "solver": diagnostics, "length_m": length, "relative_length_error": error,
                        "target_length_m": args.target_length, "independent_experimental_support": False})
        print(json.dumps({"length_m": length, "relative_error": error, "status": diagnostics["status"]}), flush=True)
        if diagnostics["status"] != "converged" or error <= 0.01:
            break
        factor *= (length - 0.066) / (args.target_length - 0.066)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf8")
    return 0 if records[-1]["solver"]["status"] == "converged" and error <= 0.05 else 1


if __name__ == "__main__":
    raise SystemExit(main())
