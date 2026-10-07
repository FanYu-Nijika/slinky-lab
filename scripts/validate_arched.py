"""Check manual arched initial geometry and a bounded, passive release.

This is an initial-condition check, not a three-step gait or experimental
validation. Long staircase convergence is evaluated separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from slinky_lab import MODEL_VERSION
from slinky_lab.physics import Simulation
from slinky_lab.presets import get_preset
from validate_stairs import run_case, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/arched-validation"))
    parser.add_argument("--duration", type=float, default=0.12)
    parser.add_argument("--wall-seconds", type=float, default=180)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    geometry = []
    for preset_id in ("stairs-arched", "stairs-arched-dense"):
        config = get_preset(preset_id)
        case_dir = output / preset_id
        case_dir.mkdir(parents=True, exist_ok=True)
        simulation = Simulation(config)
        diagnostics = simulation.summary()["initial_pose_diagnostics"]
        write_json(case_dir / "config.json", config.model_dump(mode="json"))
        write_json(case_dir / "metadata.json", simulation.metadata())
        write_json(case_dir / "initial-frame.json", simulation.frame())
        write_json(case_dir / "initial-diagnostics.json", diagnostics)
        (case_dir / "model.xml").write_text(simulation.model_xml, encoding="utf-8")
        passed = (
            diagnostics["qpos0_unchanged"]
            and diagnostics["max_link_error_m"] < 1e-7
            and diagnostics["max_endpoint_error_m"] < 1e-7
            and diagnostics["first_end_within_stair0"]
            and diagnostics["last_end_within_stair1"]
            and abs(diagnostics["actual_first_clearance_m"] - 0.0002) < 1e-7
            and abs(diagnostics["actual_free_end_clearance_m"] - config.scene.arch_free_clearance) < 1e-7
            and diagnostics["initial_self_contact_count"] == 0
            and diagnostics["initial_max_penetration_m"] == 0
            and diagnostics["equality_active_count"] == 0
            and diagnostics["actuator_count"] == 0
            and diagnostics["applied_force_max"] == 0
            and diagnostics["initial_elastic_energy_estimate"] > 0
        )
        geometry.append({"preset": preset_id, "passed": bool(passed), "diagnostics": diagnostics})
        print(json.dumps({"preset": preset_id, "geometry_passed": bool(passed), "self_contacts": diagnostics["initial_self_contact_count"]}), flush=True)
    base = get_preset("stairs-arched")
    base.numerics.duration = args.duration
    base.numerics.sample_hz = 100
    spec = {"name": "short-release", "segments_per_turn": base.numerics.resolved()["segments_per_turn"],
            "timestep": base.numerics.resolved()["timestep"], "wall_seconds": args.wall_seconds}
    release = run_case(base, spec, output / "short-release")
    release_passed = (release["status"] == "completed" and not release["warning_detected"]
                      and not release["timeout_detected"] and not release["frame_errors"] and not release["hdf5_error"])
    root = Path(__file__).resolve().parents[1]
    source_files = ["backend/slinky_lab/physics.py", "backend/slinky_lab/schemas.py", "backend/slinky_lab/presets.py", "backend/slinky_lab/__init__.py"]
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "model_version": MODEL_VERSION,
              "source_file_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in source_files},
              "geometry": geometry, "short_release": release, "short_release_passed": bool(release_passed),
              "passed": bool(all(case["passed"] for case in geometry) and release_passed),
              "acceptance_scope": "initial geometry and bounded short passive release only",
              "three_step_gait_verified": False, "experimental_support": False}
    write_json(output / "validation-summary.json", report)
    print(json.dumps({"report": str(output / "validation-summary.json"), "passed": report["passed"]}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
