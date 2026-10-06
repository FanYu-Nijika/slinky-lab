"""FastAPI application for interactive and batch Slinky experiments."""

from __future__ import annotations

import asyncio
import importlib
import os
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, File, HTTPException, Query, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from .manager import RunManager
from .schemas import Command, RunConfig, SweepConfig


def _presets() -> list[dict[str, Any]]:
    try:
        module = importlib.import_module(".presets", package="slinky_lab")
    except ImportError:
        return []
    getter = getattr(module, "get_presets", None)
    if not callable(getter):
        return []
    return list(getter())


def _preset_config(preset_id: str) -> RunConfig:
    module = importlib.import_module(".presets", package="slinky_lab")
    getter = getattr(module, "get_preset", None)
    if not callable(getter):
        raise KeyError(preset_id)
    return getter(preset_id)


def _run_response(run: dict[str, Any]) -> dict[str, Any]:
    """Keep API responses JSON-compatible while exposing progress fields."""

    return {
        "run_id": run["run_id"],
        "status": run["status"],
        "config": run.get("config", {}),
        "summary": run.get("summary", {}),
        "metadata": run.get("metadata", {}),
        "artifacts": run.get("artifacts", []),
        "error": run.get("error"),
        "time": run.get("time", 0),
        "progress": run.get("progress", 0),
        "created_at": run.get("created_at"),
        "updated_at": run.get("updated_at"),
    }


def _manager_error(exc: Exception) -> HTTPException:
    if isinstance(exc, ValidationError):
        details = []
        for error in exc.errors():
            details.append({"loc": list(error.get("loc", ())), "msg": str(error.get("msg", "Validation error")), "type": error.get("type", "value_error")})
        return HTTPException(status_code=422, detail=details)
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail="Run or sweep not found")
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


def create_app(manager: RunManager | None = None) -> FastAPI:
    service = manager or RunManager(auto_start=False, recover=False)
    app = FastAPI(title="Slinky Lab", version="0.1.0")
    app.state.manager = service

    @app.on_event("startup")
    async def startup() -> None:
        service.store.recover_active_runs()
        if any(run.get("status") == "queued" for run in service.store.list_runs(1000)):
            service.start()

    @app.on_event("shutdown")
    async def shutdown() -> None:
        service.shutdown()

    @app.get("/api/v1/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "version": "0.1.0", "worker": service.worker_alive}

    @app.get("/api/v1/presets")
    async def presets() -> list[dict[str, Any]]:
        return _presets()

    @app.get("/api/v1/runs")
    async def list_runs(limit: int = Query(100, ge=1, le=1000)) -> list[dict[str, Any]]:
        return [_run_response(run) for run in service.list_runs(limit)]

    @app.post("/api/v1/runs")
    async def create_run(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        try:
            # Accept a raw RunConfig (the browser's default) and {config: ...}
            # so command-line and API clients can share a wrapper envelope.
            config_data = payload.get("config", payload)
            if "preset_id" in payload:
                config = _preset_config(str(payload["preset_id"]))
            else:
                config = RunConfig.model_validate(config_data)
            return _run_response(service.create_run(config))
        except Exception as exc:
            raise _manager_error(exc) from exc

    @app.get("/api/v1/runs/{run_id}")
    async def get_run(run_id: str) -> dict[str, Any]:
        run = service.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return _run_response(run)

    @app.get("/api/v1/runs/{run_id}/frames")
    async def frames(
        run_id: str,
        start: int = Query(0, ge=0),
        limit: int = Query(120, ge=0, le=1000),
    ) -> dict[str, Any]:
        try:
            return service.frames(run_id, start, limit)
        except Exception as exc:
            raise _manager_error(exc) from exc

    @app.post("/api/v1/runs/{run_id}/commands")
    async def command(run_id: str, command_data: Command) -> dict[str, Any]:
        try:
            return _run_response(service.command(run_id, command_data.action))
        except Exception as exc:
            raise _manager_error(exc) from exc

    @app.get("/api/v1/runs/{run_id}/artifacts/{name}")
    async def artifact(run_id: str, name: str) -> FileResponse:
        try:
            path = service.store.artifact_path(run_id, name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Artifact not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return FileResponse(path, filename=path.name)

    @app.post("/api/v1/sweeps")
    async def create_sweep(config: SweepConfig) -> dict[str, Any]:
        try:
            return service.create_sweep(config)
        except Exception as exc:
            raise _manager_error(exc) from exc

    @app.get("/api/v1/sweeps/{sweep_id}")
    async def get_sweep(sweep_id: str) -> dict[str, Any]:
        sweep = service.sweep(sweep_id)
        if sweep is None:
            raise HTTPException(status_code=404, detail="Sweep not found")
        return sweep

    @app.post("/api/v1/runs/{run_id}/reference")
    async def upload_reference(
        run_id: str,
        request: Request,
        file: UploadFile = File(...),
        metric: str | None = None,
    ) -> dict[str, Any]:
        # For multipart clients metric is commonly a form field; for simple
        # clients it is a query parameter.  Query parameters take precedence.
        metric = metric or request.query_params.get("metric")
        if not metric:
            form = await request.form()
            metric = str(form.get("metric") or "")
        if not metric:
            raise HTTPException(status_code=400, detail="metric is required")
        try:
            return service.add_reference(run_id, metric, await file.read())
        except Exception as exc:
            raise _manager_error(exc) from exc

    @app.get("/api/v1/runs/{run_id}/reference")
    async def get_reference(run_id: str) -> dict[str, Any]:
        run = service.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        reference = service.store.get_reference(run_id)
        if reference is None:
            raise HTTPException(status_code=404, detail="Reference data not found")
        return reference

    @app.websocket("/api/v1/runs/{run_id}/stream")
    async def stream(websocket: WebSocket, run_id: str) -> None:
        await websocket.accept()
        try:
            run = service.get_run(run_id)
            if run is None:
                await websocket.send_json({"type": "error", "data": {"message": "Run not found"}})
                await websocket.close(code=1008)
                return
            cursor = 0
            metadata_sent = False
            last_status: tuple[Any, ...] | None = None
            while True:
                run = service.get_run(run_id)
                if run is None:
                    await websocket.send_json({"type": "error", "data": {"message": "Run not found"}})
                    return
                status_key = (run.get("status"), run.get("progress"), run.get("time"), run.get("error"))
                if status_key != last_status:
                    await websocket.send_json({"type": "status", "data": _run_response(run)})
                    last_status = status_key
                metadata = run.get("metadata") or {}
                if metadata and not metadata_sent:
                    await websocket.send_json({"type": "metadata", "data": metadata})
                    metadata_sent = True
                batch = service.frames(run_id, cursor, 120)
                for frame in batch["frames"]:
                    await websocket.send_json({"type": "frame", "data": frame})
                    cursor += 1
                if run["status"] in {"completed", "failed", "cancelled", "interrupted"}:
                    if cursor < batch["total"]:
                        await asyncio.sleep(0)
                        continue
                    if run["status"] == "completed":
                        await websocket.send_json({"type": "result", "data": _run_response(run)})
                    elif run.get("error"):
                        await websocket.send_json({"type": "error", "data": {"message": run["error"], "run": _run_response(run)}})
                    return
                await asyncio.sleep(0.1)
        except WebSocketDisconnect:
            return

    @app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
    async def unknown_api(path: str) -> JSONResponse:
        return JSONResponse({"detail": "API route not found"}, status_code=404)

    static_env = os.environ.get("SLINKY_STATIC_DIR")
    static_dir = Path(static_env) if static_env else Path(__file__).resolve().parents[2] / "frontend" / "dist"
    if static_dir.is_dir():
        # This mount is deliberately registered after /api routes so an
        # absent frontend file can never swallow an API 404.
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")
    else:
        @app.get("/")
        async def no_frontend() -> JSONResponse:
            return JSONResponse({"service": "slinky-lab", "message": "frontend build not installed"})

    return app


app = create_app()
