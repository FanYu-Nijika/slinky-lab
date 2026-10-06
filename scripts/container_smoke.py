"""Exercise the packaged app through its public HTTP API, without host dependencies."""

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def request(base, path, payload=None):
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as response:
        data = response.read()
        return json.loads(data) if "application/json" in response.headers.get("Content-Type", "") else data


def healthy(base):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            return request(base, "/api/v1/health")
        except (OSError, urllib.error.URLError):
            time.sleep(0.5)
    raise RuntimeError("Container did not become healthy within 60 seconds")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", default="reports/container.json")
    parser.add_argument("--check-persisted")
    args = parser.parse_args()
    health = healthy(args.url)
    if args.check_persisted:
        previous = json.loads(Path(args.check_persisted).read_text())
        run = request(args.url, f"/api/v1/runs/{previous['run_id']}")
        assert run["status"] == "completed", run
        frames = request(args.url, f"/api/v1/runs/{previous['run_id']}/frames?limit=2")
        assert frames["total"] == previous["frame_count"], frames
        print("Container restart: persisted configuration, result and trajectory verified")
        return

    page = request(args.url, "/")
    assert b"<html" in page.lower(), "Packaged frontend is missing"
    presets = request(args.url, "/api/v1/presets")
    assert presets, "No presets supplied"
    config = {
        "name": "容器真实动力学检查", "scenario": "drop",
        "material": {"turns": 3, "mass": 0.006, "pitch": 0.002},
        "scene": {"settle_time": 0.1},
        "numerics": {"duration": 0.08, "segments_per_turn": 8, "timestep": 0.0001, "max_wall_seconds": 60},
    }
    created = request(args.url, "/api/v1/runs", config)
    run_id = created["run_id"]
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        run = request(args.url, f"/api/v1/runs/{run_id}")
        if run["status"] in {"completed", "failed", "cancelled", "interrupted"}:
            break
        time.sleep(0.2)
    assert run["status"] == "completed", run
    frames = request(args.url, f"/api/v1/runs/{run_id}/frames?limit=100")
    assert frames["total"] >= 2, frames
    assert frames["frames"][-1]["time"] > frames["frames"][0]["time"]
    assert len(frames["frames"][0]["positions"]) >= 20
    csv = request(args.url, f"/api/v1/runs/{run_id}/artifacts/metrics.csv")
    assert b"time" in csv and b"com_z" in csv, "Metrics export incomplete"
    config_file = request(args.url, f"/api/v1/runs/{run_id}/artifacts/config.json")
    assert config_file["material"]["turns"] == 3
    result = {"health": health, "run_id": run_id, "frame_count": frames["total"], "summary": run["summary"]}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run_id": run_id, "status": "passed", "frames": frames["total"]}))


if __name__ == "__main__":
    main()
