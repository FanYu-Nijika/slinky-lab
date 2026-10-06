"""Command-line entry points for local serving and batch runs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from .manager import RunManager
from .calibration import moduli_from_coil_stiffness
from .presets import get_preset
from .schemas import RunConfig, SweepConfig


def _config_from_args(path: str | None, preset: str | None) -> RunConfig:
    if path and preset:
        raise ValueError("Use either --config or --preset")
    if preset:
        return get_preset(preset)
    if not path:
        return RunConfig()
    source = Path(path).read_text(encoding="utf-8")
    return RunConfig.model_validate(json.loads(source))


def _wait(manager: RunManager, run_id: str) -> dict[str, Any]:
    while True:
        run = manager.get_run(run_id)
        if run is None:
            raise RuntimeError(f"Run {run_id} disappeared")
        print(json.dumps({"run_id": run_id, "status": run["status"], "progress": run["progress"]}, ensure_ascii=False), flush=True)
        if run["status"] in {"completed", "failed", "cancelled", "interrupted"}:
            return run
        time.sleep(0.2)


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("slinky_lab.api:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def _run(args: argparse.Namespace) -> int:
    manager = RunManager()
    try:
        result = manager.create_run(_config_from_args(args.config, args.preset))
        final = _wait(manager, result["run_id"])
        print(json.dumps(final, ensure_ascii=False, indent=2, default=str))
        return 0 if final["status"] == "completed" else 1
    finally:
        manager.shutdown()


def _sweep(args: argparse.Namespace) -> int:
    manager = RunManager()
    try:
        config = SweepConfig.model_validate(json.loads(Path(args.config).read_text(encoding="utf-8")))
        result = manager.create_sweep(config)
        sweep_id = result["sweep_id"]
        while True:
            current = manager.sweep(sweep_id)
            if current is None:
                raise RuntimeError(f"Sweep {sweep_id} disappeared")
            print(json.dumps(current, ensure_ascii=False), flush=True)
            if current["status"] in {"completed", "failed"}:
                print(json.dumps(current, ensure_ascii=False, indent=2, default=str))
                return 0 if current["status"] == "completed" else 1
            time.sleep(0.2)
    finally:
        manager.shutdown()


def _validate(args: argparse.Namespace) -> int:
    config = _config_from_args(args.config, args.preset)
    print(json.dumps({"valid": True, "config": config.model_dump(mode="json")}, ensure_ascii=False, indent=2))
    return 0


def _calibrate(args: argparse.Namespace) -> int:
    config = _config_from_args(args.config, args.preset)
    result = moduli_from_coil_stiffness(config.material, args.stiffness, args.poisson_ratio)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="slinky-lab", description="Slinky Lab dynamics workbench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="start the FastAPI service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(handler=_serve)

    run = subparsers.add_parser("run", help="run one simulation")
    run.add_argument("--config")
    run.add_argument("--preset")
    run.set_defaults(handler=_run)

    sweep = subparsers.add_parser("sweep", help="run a sweep config JSON")
    sweep.add_argument("config")
    sweep.set_defaults(handler=_sweep)

    validate = subparsers.add_parser("validate", help="validate one RunConfig JSON or preset")
    validate.add_argument("--config")
    validate.add_argument("--preset")
    validate.set_defaults(handler=_validate)
    calibrate = subparsers.add_parser("calibrate", help="estimate rod moduli from coil stiffness; requires subsequent native static fitting")
    calibrate.add_argument("--config")
    calibrate.add_argument("--preset")
    calibrate.add_argument("--stiffness", type=float, required=True, help="measured bulk stiffness in N/m")
    calibrate.add_argument("--poisson-ratio", type=float, default=0.35, help="assumed isotropic Poisson ratio")
    calibrate.set_defaults(handler=_calibrate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except Exception as exc:
        print(f"slinky-lab: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
