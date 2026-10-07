"""Run two real drop-validation jobs through HTTP and consume their WebSocket streams.

This is an operational smoke check for the packaged API.  It deliberately uses
the public endpoints and the locked ``websockets`` package instead of importing
the solver or fabricating frames, so a native worker failure remains visible in
the saved report.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import platform
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import h5py
import websockets


TERMINAL_STATUSES = {"completed", "failed", "cancelled", "interrupted"}
POLL_SECONDS = 0.25
HTTP_TIMEOUT = 30.0
ARTIFACT_NAMES = ("summary.json", "metrics.csv", "trajectory.h5")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _http_url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + path


def _ws_url(base_url: str, path: str) -> str:
    parts = urlsplit(base_url.rstrip("/"))
    scheme = "wss" if parts.scheme == "https" else "ws"
    prefix = parts.path.rstrip("/")
    return urlunsplit((scheme, parts.netloc, prefix + path, "", ""))


def _request_bytes(base_url: str, path: str, payload: dict[str, Any] | None = None) -> bytes:
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(_http_url(base_url, path), data=body, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {exc.reason}: {detail[:1000]}") from exc
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError(f"HTTP request failed: {exc}") from exc


def _request_json(base_url: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any] | list[Any]:
    raw = _request_bytes(base_url, path, payload)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Expected JSON from {path}: {exc}") from exc
    if not isinstance(value, (dict, list)):
        raise RuntimeError(f"Expected an object or array from {path}")
    return value


def _stream_summary() -> dict[str, Any]:
    return {
        "connected": False,
        "message_count": 0,
        "type_counts": {},
        "frame_count": 0,
        "first_frame_time": None,
        "last_frame_time": None,
        "last_status": None,
        "terminal_type": None,
        "terminal_status": None,
        "result": None,
        "errors": [],
    }


def _record_stream_message(summary: dict[str, Any], raw: str | bytes) -> None:
    try:
        message = json.loads(raw)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        summary["errors"].append(f"invalid websocket JSON: {exc}")
        return
    if not isinstance(message, dict):
        summary["errors"].append("websocket message was not an object")
        return
    kind = str(message.get("type", "unknown"))
    counts = Counter(summary["type_counts"])
    counts[kind] += 1
    summary["type_counts"] = dict(counts)
    summary["message_count"] += 1
    data = message.get("data")
    if kind == "status" and isinstance(data, dict):
        summary["last_status"] = {
            "status": data.get("status"),
            "time": data.get("time"),
            "progress": data.get("progress"),
            "error": data.get("error"),
        }
    elif kind == "frame" and isinstance(data, dict):
        summary["frame_count"] += 1
        frame_time = data.get("time")
        if summary["first_frame_time"] is None:
            summary["first_frame_time"] = frame_time
        summary["last_frame_time"] = frame_time
    elif kind == "result" and isinstance(data, dict):
        summary["terminal_type"] = "result"
        summary["terminal_status"] = data.get("status")
        summary["result"] = {
            "status": data.get("status"),
            "time": data.get("time"),
            "progress": data.get("progress"),
            "summary": data.get("summary") or {},
            "error": data.get("error"),
        }
    elif kind == "error" and isinstance(data, dict):
        summary["terminal_type"] = "error"
        run = data.get("run")
        summary["terminal_status"] = run.get("status") if isinstance(run, dict) else "failed"
        summary["errors"].append(str(data.get("message") or data.get("error") or "websocket error"))
        summary["result"] = {
            "status": summary["terminal_status"],
            "summary": run.get("summary", {}) if isinstance(run, dict) else {},
            "error": data.get("message") or data.get("error"),
        }


async def _consume_stream(stream_url: str, summary: dict[str, Any]) -> None:
    try:
        async with websockets.connect(stream_url, max_size=None, open_timeout=HTTP_TIMEOUT) as websocket:
            summary["connected"] = True
            async for raw in websocket:
                _record_stream_message(summary, raw)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        summary["errors"].append(f"websocket failed: {type(exc).__name__}: {exc}")


async def _request_json_async(base_url: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any] | list[Any]:
    return await asyncio.to_thread(_request_json, base_url, path, payload)


async def _request_bytes_async(base_url: str, path: str) -> bytes:
    return await asyncio.to_thread(_request_bytes, base_url, path)


async def _fetch_artifacts(base_url: str, run_id: str) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for name in ARTIFACT_NAMES:
        check: dict[str, Any] = {"available": False}
        try:
            raw = await _request_bytes_async(base_url, f"/api/v1/runs/{run_id}/artifacts/{name}")
            check["available"] = True
            check["bytes"] = len(raw)
            if name == "summary.json":
                try:
                    check["summary"] = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    check["error"] = f"invalid summary.json: {exc}"
            elif name == "metrics.csv":
                try:
                    rows = list(csv.reader(io.StringIO(raw.decode("utf-8"))))
                    header = rows[0] if rows else []
                    check["header"] = header
                    check["row_count"] = max(0, len(rows) - 1)
                    check["has_time"] = "time" in header
                    check["has_com_z"] = "com_z" in header
                except (UnicodeDecodeError, csv.Error) as exc:
                    check["error"] = f"invalid metrics.csv: {exc}"
            else:
                try:
                    with h5py.File(io.BytesIO(raw), "r") as trajectory:
                        committed = int(trajectory.attrs.get("committed_frames", -1))
                        datasets = {name: int(trajectory[name].shape[0]) for name in ("time", "positions", "quaternions") if name in trajectory}
                        check["format"] = str(trajectory.attrs.get("format", ""))
                        check["committed_frames"] = committed
                        check["dataset_frames"] = datasets
                        check["complete"] = committed > 0 and len(datasets) == 3 and all(value == committed for value in datasets.values())
                except (OSError, ValueError) as exc:
                    check["error"] = f"invalid trajectory.h5: {exc}"
        except Exception as exc:
            check["error"] = str(exc)
        checks[name] = check
    return checks


def _artifact_checks_pass(checks: dict[str, Any], frame_total: int) -> bool:
    summary = checks.get("summary.json", {})
    metrics = checks.get("metrics.csv", {})
    trajectory = checks.get("trajectory.h5", {})
    return bool(
        summary.get("available") and "summary" in summary
        and metrics.get("available") and metrics.get("has_time") and metrics.get("has_com_z") and metrics.get("row_count", 0) >= 2
        and trajectory.get("available") and trajectory.get("complete") and trajectory.get("committed_frames") == frame_total
    )


async def _run_one(base_url: str, index: int, deadline: float) -> dict[str, Any]:
    result: dict[str, Any] = {
        "index": index,
        "preset_id": "drop-validation",
        "started_at": _now(),
        "run_id": None,
        "create": None,
        "status_history": [],
        "health_samples": [],
        "stream": _stream_summary(),
        "frames": {},
        "artifacts": {},
        "final_run": None,
        "errors": [],
        "passed": False,
    }
    try:
        created = await _request_json_async(base_url, "/api/v1/runs", {"preset_id": "drop-validation"})
        if not isinstance(created, dict):
            raise RuntimeError("create run response was not an object")
        result["create"] = created
        run_id = str(created.get("run_id") or "")
        if not run_id:
            raise RuntimeError("create run response did not contain run_id")
        result["run_id"] = run_id
    except Exception as exc:
        result["errors"].append(f"create failed: {type(exc).__name__}: {exc}")
        return result

    run_id = result["run_id"]
    stream_task = asyncio.create_task(_consume_stream(_ws_url(base_url, f"/api/v1/runs/{run_id}/stream"), result["stream"]))
    started = time.monotonic()
    next_health = started
    final_run: dict[str, Any] | None = None
    try:
        while True:
            now = time.monotonic()
            if now >= next_health:
                try:
                    health = await _request_json_async(base_url, "/api/v1/health")
                    result["health_samples"].append(health)
                except Exception as exc:
                    result["health_samples"].append({"error": str(exc)})
                next_health = now + 5.0
            try:
                current = await _request_json_async(base_url, f"/api/v1/runs/{run_id}")
            except Exception as exc:
                result["errors"].append(f"status poll failed: {type(exc).__name__}: {exc}")
                current = None
            if isinstance(current, dict):
                compact = {key: current.get(key) for key in ("status", "time", "progress", "error")}
                if not result["status_history"] or compact != result["status_history"][-1]:
                    result["status_history"].append(compact)
                if current.get("status") in TERMINAL_STATUSES:
                    final_run = current
                    break
            if time.monotonic() - started >= deadline:
                result["errors"].append(f"deadline exceeded after {deadline:.1f}s")
                try:
                    cancelled = await _request_json_async(base_url, f"/api/v1/runs/{run_id}/commands", {"action": "cancel"})
                    result["cancel_after_deadline"] = cancelled
                except Exception as exc:
                    result["cancel_after_deadline"] = {"error": str(exc)}
                try:
                    after_cancel = await _request_json_async(base_url, f"/api/v1/runs/{run_id}")
                    if isinstance(after_cancel, dict):
                        final_run = after_cancel if after_cancel.get("status") in TERMINAL_STATUSES else None
                except Exception as exc:
                    result["errors"].append(f"final status poll failed: {exc}")
                break
            await asyncio.sleep(POLL_SECONDS)
    finally:
        try:
            await asyncio.wait_for(stream_task, timeout=10.0)
        except asyncio.TimeoutError:
            stream_task.cancel()
            await asyncio.gather(stream_task, return_exceptions=True)
        except Exception as exc:
            result["errors"].append(f"stream task failed: {type(exc).__name__}: {exc}")

    if final_run is None:
        try:
            polled = await _request_json_async(base_url, f"/api/v1/runs/{run_id}")
            if isinstance(polled, dict):
                final_run = polled
        except Exception as exc:
            result["errors"].append(f"terminal status unavailable: {type(exc).__name__}: {exc}")
    result["final_run"] = final_run
    if final_run and final_run.get("error"):
        result["errors"].append(f"run error: {final_run['error']}")

    try:
        frames = await _request_json_async(base_url, f"/api/v1/runs/{run_id}/frames?start=0&limit=1000")
        if isinstance(frames, dict):
            frame_list = frames.get("frames") or []
            result["frames"] = {
                "total": frames.get("total"),
                "returned": len(frame_list),
                "first_time": frame_list[0].get("time") if frame_list else None,
                "last_time": frame_list[-1].get("time") if frame_list else None,
                "complete": bool(frame_list) and len(frame_list) == frames.get("total") and len(frame_list) >= 2,
            }
    except Exception as exc:
        result["frames"] = {"error": str(exc), "complete": False}

    result["artifacts"] = await _fetch_artifacts(base_url, run_id)
    final_status = final_run.get("status") if final_run else None
    frame_total = result["frames"].get("total") if isinstance(result["frames"], dict) else None
    stream = result["stream"]
    frame_complete = bool(result["frames"].get("complete")) and result["frames"].get("last_time", 0) > result["frames"].get("first_time", 0)
    stream_complete = bool(stream.get("connected") and stream.get("terminal_type") == "result" and stream.get("terminal_status") == "completed")
    stream_frame_match = frame_total is not None and stream.get("frame_count") == frame_total
    result["checks"] = {
        "terminal_completed": final_status == "completed",
        "http_frames_complete": frame_complete,
        "stream_terminal_result": stream_complete,
        "stream_frame_count_matches_http": stream_frame_match,
        "exports_complete": _artifact_checks_pass(result["artifacts"], int(frame_total or -1)),
    }
    result["passed"] = all(result["checks"].values())
    result["finished_at"] = _now()
    return result


async def _run_report(base_url: str, output: Path, deadline: float) -> dict[str, Any]:
    report: dict[str, Any] = {
        "created_at": _now(),
        "base_url": base_url,
        "preset_id": "drop-validation",
        "deadline_seconds_per_run": deadline,
        "python": platform.python_version(),
        "websockets": getattr(websockets, "__version__", "unknown"),
        "health_before": None,
        "runs": [],
        "passed": False,
    }
    try:
        report["health_before"] = await _request_json_async(base_url, "/api/v1/health")
    except Exception as exc:
        report["health_before"] = {"error": str(exc)}
        report["error"] = f"initial health check failed: {exc}"
        return report

    for index in (1, 2):
        report["runs"].append(await _run_one(base_url, index, deadline))
    try:
        report["health_after"] = await _request_json_async(base_url, "/api/v1/health")
    except Exception as exc:
        report["health_after"] = {"error": str(exc)}
    report["passed"] = len(report["runs"]) == 2 and all(run.get("passed") for run in report["runs"])
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", default="reports/runtime-stream.json")
    parser.add_argument("--deadline", type=float, default=180.0)
    args = parser.parse_args()
    if args.deadline <= 0:
        parser.error("--deadline must be positive")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = asyncio.run(_run_report(args.url.rstrip("/"), output, args.deadline))
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "passed": report.get("passed", False), "runs": len(report.get("runs", []))}, ensure_ascii=False))
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
