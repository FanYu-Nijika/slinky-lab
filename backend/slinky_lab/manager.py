"""Run queue and isolated simulation worker.

The worker process owns the solver object and receives only serialisable
configuration/command messages.  This prevents a MuJoCo object from crossing
the process boundary and keeps a slow simulation from blocking FastAPI.
"""

from __future__ import annotations

import csv
import importlib
import itertools
import multiprocessing as mp
import queue
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from .analysis import compare_reference
from .schemas import Frame, RunConfig, SweepConfig
from .storage import (
    DataStore,
    TrajectoryWriter,
    dump_json,
    read_frames,
)


def _simulation_class() -> type[Any]:
    """Load the physics module lazily so API health works during development."""

    candidates = (".physics", ".simulation", "slinky_lab.physics", "slinky_lab.simulation")
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name, package="slinky_lab")
        except ImportError:
            continue
        simulation = getattr(module, "Simulation", None)
        if simulation is not None:
            return simulation
    raise RuntimeError("Physics module with Simulation(config) is not available")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _simulation_xml(simulation: Any) -> str | bytes | None:
    value = getattr(simulation, "model_xml", None)
    if value is not None:
        return value
    model = getattr(simulation, "model", None)
    if model is not None:
        value = getattr(model, "xml", None)
        if value is not None:
            return value
    return None


def _call_frame(simulation: Any) -> dict[str, Any]:
    frame = simulation.frame()
    if isinstance(frame, Frame):
        return frame.model_dump(mode="json")
    return Frame.model_validate(frame).model_dump(mode="json")


def _call_metadata(simulation: Any) -> dict[str, Any]:
    value = simulation.metadata()
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("Simulation.metadata() must return a dictionary")
    # A round trip catches numpy objects before an event reaches the API.
    metadata = __import__("json").loads(dump_json(value))
    # Keep one small renderer contract even when a solver exposes the legacy
    # top-level names used by the physics module.
    if "render" not in metadata:
        metadata["render"] = {
            "half_sizes": metadata.get("half_sizes", []),
            "static_geoms": metadata.get("static_geoms", metadata.get("stairs", [])),
        }
    return metadata


def _call_summary(simulation: Any) -> dict[str, Any]:
    value = simulation.summary()
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("Simulation.summary() must return a dictionary")
    return __import__("json").loads(dump_json(value))


def _worker_emit(events: Any, event: dict[str, Any]) -> None:
    try:
        events.put(event)
    except (BrokenPipeError, EOFError, OSError):
        pass


def _worker_run(task: dict[str, Any], controls: Any, events: Any, data_dir: str) -> None:
    """Execute one task inside the worker process."""

    run_id = str(task["run_id"])
    config = RunConfig.model_validate(task["config"])
    store = DataStore(data_dir, recover=False)
    run_dir = store.run_dir(run_id)
    started = time.monotonic()
    paused = False
    cancel_requested = False
    pause_requested = False
    step_requested = False
    timeout_requested = False
    paused_since: float | None = None
    paused_total = 0.0
    # ``preparing`` remains true until the first frame is published.  This
    # gives a step command received at the prepare/frame boundary the explicit
    # meaning "finish preparation, then stop at the first frame".
    preparing = True
    preparation_progress = 0.0
    preparation_step_requested = False
    pause_after_prepare = False
    resume_seen = False
    simulation: Any = None
    frame_count = 0
    sample_period = 1.0 / config.numerics.sample_hz
    next_sample = 0.0
    last_time = 0.0

    def event_status(status: str, progress: float = 0.0, sim_time: float = 0.0, **extra: Any) -> None:
        _worker_emit(
            events,
            {
                "kind": "status",
                "run_id": run_id,
                "status": status,
                "progress": max(0.0, min(1.0, float(progress))),
                "time": float(sim_time),
                **extra,
            },
        )

    def active_wall_seconds() -> float:
        if paused_since is None:
            return time.monotonic() - started - paused_total
        return time.monotonic() - started - paused_total - (time.monotonic() - paused_since)

    def resume_from_pause() -> None:
        nonlocal paused_since, paused_total
        if paused_since is not None:
            paused_total += time.monotonic() - paused_since
            paused_since = None

    def inspect_controls() -> None:
        """Consume commands without blocking while physics is integrating."""

        nonlocal paused, cancel_requested, pause_requested, step_requested, preparation_step_requested, resume_seen, pause_after_prepare
        while True:
            try:
                command = controls.get_nowait()
            except queue.Empty:
                return
            if command.get("run_id") != run_id:
                # The manager sends controls for the active job only.  A stale
                # command is ignored rather than allowing it to affect a later run.
                continue
            action = command.get("action")
            if action == "cancel":
                cancel_requested = True
                return
            if action == "pause":
                pause_requested = True
                resume_seen = False
            elif action == "resume":
                paused = False
                pause_requested = False
                resume_seen = True
                if preparing:
                    # A newer explicit resume cancels a preparation-step
                    # request, including one already converted to the
                    # pause-after-first-frame boundary state.
                    preparation_step_requested = False
                    pause_after_prepare = False
                resume_from_pause()
            elif action == "step" and (paused or pause_requested):
                # During preparation, step resumes only the preparation phase;
                # it never silently advances the released simulation.  The
                # first frame is published paused after prepare returns.
                if preparing:
                    preparation_step_requested = True
                else:
                    # A normal step is handled by the paused loop, where it
                    # integrates through the next sample boundary.
                    step_requested = True

    def wait_for_preparation_pause() -> None:
        """Hold inside ``prepare`` while a user pause is active.

        ``Simulation.prepare`` exposes a cancellation callback but no pause
        callback.  Blocking here keeps its static solve/relaxation state
        intact, lets cancel interrupt promptly, and excludes the wait from
        the wall-clock budget through ``paused_since``.
        """

        nonlocal paused, paused_since, pause_after_prepare, preparation_step_requested, pause_requested
        if not pause_requested:
            return
        paused = True
        if paused_since is None:
            paused_since = time.monotonic()
        event_status("paused", progress=min(0.25, preparation_progress), sim_time=0.0)
        while True:
            inspect_controls()
            if cancel_requested:
                return
            if preparation_step_requested:
                preparation_step_requested = False
                pause_after_prepare = True
                paused = False
                pause_requested = False
                resume_from_pause()
                event_status("preparing", progress=min(0.25, preparation_progress), sim_time=0.0)
                return
            if not pause_requested:
                paused = False
                resume_from_pause()
                event_status("preparing", progress=min(0.25, preparation_progress), sim_time=0.0)
                return
            time.sleep(0.05)

    def should_cancel() -> bool:
        nonlocal timeout_requested
        inspect_controls()
        if preparing and pause_requested:
            wait_for_preparation_pause()
        if active_wall_seconds() > config.numerics.max_wall_seconds:
            timeout_requested = True
            return True
        return cancel_requested

    def refresh_queued_state() -> None:
        """Close the DB/queue race before a queued task becomes active.

        Commands issued while the event thread still reports ``queued`` are
        persisted in SQLite instead of being put on the shared control queue.
        Re-reading that state after preparation lets this worker honour a
        pause or cancellation even when it loaded the task just before the
        API transaction committed.  A queued resume also clears the initial
        pause flag; ``resume_seen`` prevents a stale paused row from undoing a
        resume command that was already consumed from the control queue.
        """

        nonlocal cancel_requested, pause_requested, resume_seen
        latest = store.get_run(run_id)
        if latest is None:
            return
        status = latest.get("status")
        if status == "cancelled":
            cancel_requested = True
        elif status == "paused":
            if not resume_seen:
                pause_requested = True
        elif status == "queued":
            pause_requested = False
            resume_seen = False
        else:
            resume_seen = False

    def prepare_progress(value: Any) -> None:
        nonlocal preparation_progress
        if isinstance(value, dict):
            value = value.get("progress", 0.0)
        progress = _safe_float(value)
        preparation_progress = progress
        event_status("preparing", progress=min(0.25, progress), sim_time=0.0)

    def apply_preparation_boundary() -> None:
        """Convert controls seen at the prepare/frame boundary into state."""

        nonlocal paused, paused_since, pause_requested, pause_after_prepare, preparation_step_requested
        if preparation_step_requested:
            preparation_step_requested = False
            pause_after_prepare = True
            paused = False
            pause_requested = False
            resume_from_pause()
        if pause_requested:
            was_paused = paused
            paused = True
            if paused_since is None:
                paused_since = time.monotonic()
            if not was_paused:
                event_status("paused", progress=min(0.25, preparation_progress), sim_time=0.0)

    def append_frame(writer: TrajectoryWriter, emit: bool = True) -> dict[str, Any]:
        nonlocal frame_count, last_time
        frame = _call_frame(simulation)
        Frame.model_validate(frame)
        writer.append(frame)
        writer.flush()
        frame_count += 1
        last_time = _safe_float(frame.get("time"))
        if emit:
            _worker_emit(events, {"kind": "frame", "run_id": run_id, "frame": frame, "index": frame_count - 1})
        return frame

    try:
        initial = store.get_run(run_id)
        if initial is not None and initial.get("status") == "cancelled":
            return
        pause_requested = bool(initial and initial.get("status") == "paused")
        event_status("preparing")
        Simulation = _simulation_class()
        simulation = Simulation(config)
        prepare = getattr(simulation, "prepare", None)
        if not callable(prepare):
            raise RuntimeError("Simulation.prepare() is required")
        prepare(progress=prepare_progress, should_cancel=should_cancel)
        inspect_controls()
        refresh_queued_state()
        apply_preparation_boundary()
        if active_wall_seconds() > config.numerics.max_wall_seconds:
            timeout_requested = True
        if timeout_requested:
            raise TimeoutError("Simulation exceeded max_wall_seconds during preparation")
        if cancel_requested:
            event_status("cancelled", sim_time=last_time)
            return
        metadata = _call_metadata(simulation)
        inspect_controls()
        refresh_queued_state()
        apply_preparation_boundary()
        if active_wall_seconds() > config.numerics.max_wall_seconds:
            timeout_requested = True
        if timeout_requested:
            raise TimeoutError("Simulation exceeded max_wall_seconds during preparation")
        if cancel_requested:
            event_status("cancelled", sim_time=last_time)
            return
        _worker_emit(events, {"kind": "metadata", "run_id": run_id, "metadata": metadata})
        store.write_model_xml(run_id, _simulation_xml(simulation))
        with TrajectoryWriter(run_dir / "trajectory.h5") as writer:
            frame = append_frame(writer)
            preparing = False
            next_sample = max(sample_period, _safe_float(frame.get("time")) + sample_period)
            if pause_after_prepare or paused or pause_requested:
                paused = True
                pause_requested = False
                if paused_since is None:
                    paused_since = time.monotonic()
                event_status("paused", progress=min(1.0, last_time / config.numerics.duration), sim_time=last_time)
            else:
                event_status("running", sim_time=last_time)

            while True:
                inspect_controls()
                if frame_count == 1:
                    refresh_queued_state()
                if cancel_requested:
                    event_status("cancelled", progress=last_time / config.numerics.duration, sim_time=last_time)
                    return
                if pause_requested:
                    paused = True
                    pause_requested = False
                    if paused_since is None:
                        paused_since = time.monotonic()
                    event_status("paused", progress=last_time / config.numerics.duration, sim_time=last_time)
                if paused:
                    if step_requested:
                        step_requested = False
                        target = min(config.numerics.duration, max(next_sample, last_time + sample_period))
                        while _safe_float(getattr(simulation, "time", last_time)) < target:
                            simulation.step()
                            inspect_controls()
                            if cancel_requested:
                                break
                        if cancel_requested:
                            continue
                        frame = append_frame(writer)
                        last_time = _safe_float(frame.get("time"))
                        next_sample = max(next_sample + sample_period, last_time + sample_period)
                        if last_time >= config.numerics.duration:
                            paused = False
                            resume_from_pause()
                            break
                        event_status("paused", progress=min(1.0, last_time / config.numerics.duration), sim_time=last_time)
                        continue
                    try:
                        command = controls.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if command.get("run_id") != run_id:
                        continue
                    action = command.get("action")
                    if action == "cancel":
                        cancel_requested = True
                        continue
                    if action == "resume":
                        paused = False
                        resume_from_pause()
                        event_status("running", progress=last_time / config.numerics.duration, sim_time=last_time)
                        continue
                    if action == "pause":
                        continue
                    if action == "step":
                        step_requested = True
                        continue
                    continue

                if active_wall_seconds() > config.numerics.max_wall_seconds:
                    raise TimeoutError("Simulation exceeded max_wall_seconds")
                current_time = _safe_float(getattr(simulation, "time", last_time), last_time)
                if current_time >= config.numerics.duration:
                    break
                simulation.step()
                inspect_controls()
                current_time = _safe_float(getattr(simulation, "time", last_time), last_time)
                if current_time + 1e-12 >= next_sample or current_time >= config.numerics.duration:
                    frame = append_frame(writer)
                    last_time = _safe_float(frame.get("time"))
                    next_sample += sample_period
                    while next_sample <= last_time:
                        next_sample += sample_period
                    event_status("running", progress=min(1.0, last_time / config.numerics.duration), sim_time=last_time)
                if cancel_requested:
                    event_status("cancelled", progress=last_time / config.numerics.duration, sim_time=last_time)
                    return

        summary = _call_summary(simulation)
        summary.setdefault("frame_count", frame_count)
        summary.setdefault("duration", last_time)
        store.write_summary(run_id, summary)
        frames, _ = read_frames(run_dir / "trajectory.h5", limit=None)
        store.write_metrics_csv(run_id, frames)
        _worker_emit(
            events,
            {
                "kind": "result",
                "run_id": run_id,
                "status": "completed",
                "summary": summary,
                "metadata": metadata,
                "time": last_time,
                "progress": 1.0,
                "artifacts": store.artifacts(run_id),
            },
        )
    except Exception as exc:  # worker must report failures instead of dying silently
        message = f"{type(exc).__name__}: {exc}"
        _worker_emit(
            events,
            {
                "kind": "error",
                "run_id": run_id,
                "status": "failed",
                "error": message,
                "traceback": traceback.format_exc(limit=8),
                "time": last_time,
                "progress": min(1.0, last_time / config.numerics.duration),
            },
        )


def _worker_loop(tasks: Any, controls: Any, events: Any, data_dir: str) -> None:
    while True:
        task = tasks.get()
        if task is None:
            return
        _worker_run(task, controls, events, data_dir)


class RunManager:
    """Single-worker run manager used by both the API and CLI."""

    def __init__(self, data_dir: str | Path | None = None, auto_start: bool = True, recover: bool = True):
        self.store = DataStore(data_dir, recover=recover)
        self.context = mp.get_context("spawn")
        self.tasks: Any = None
        self.controls: Any = None
        self.events: Any = None
        self.process: mp.Process | None = None
        self._event_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._enqueued_ids: set[str] = set()
        self._active_run_id: str | None = None
        if auto_start:
            self.start()

    @property
    def worker_alive(self) -> bool:
        return bool(self.process and self.process.is_alive())

    def start(self) -> None:
        with self._lock:
            if self.worker_alive:
                return
            self._stop.clear()
            if self.tasks is None:
                self.tasks = self.context.Queue()
                self.controls = self.context.Queue()
                self.events = self.context.Queue()
            self.process = self.context.Process(
                target=_worker_loop,
                args=(self.tasks, self.controls, self.events, str(self.store.root)),
                name="slinky-worker",
                daemon=True,
            )
            self.process.start()
            self._enqueue_existing()
            if self._event_thread is None or not self._event_thread.is_alive():
                self._event_thread = threading.Thread(target=self._event_loop, name="slinky-events", daemon=True)
                self._event_thread.start()

    def shutdown(self) -> None:
        force_interrupted = False
        with self._lock:
            self._stop.set()
            if self.process and self.process.is_alive():
                if self.tasks is not None:
                    self.tasks.put(None)
                self.process.join(timeout=2)
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(timeout=1)
                    force_interrupted = True
            self.process = None
            if self._event_thread and self._event_thread.is_alive():
                self._event_thread.join(timeout=1)
        if self._event_thread is None or not self._event_thread.is_alive():
            self._drain_events()
        if force_interrupted:
            self._active_run_id = None
            for run in self.store.list_runs(1000):
                if run.get("status") in {"preparing", "running", "paused"}:
                    self.store.update_run(run["run_id"], status="interrupted", error="Service stopped while the run was active")
            # A terminated worker cannot consume the sentinel that was queued
            # for graceful shutdown.  Fresh queues make a later start safe.
            self.tasks = self.context.Queue()
            self.controls = self.context.Queue()
            self.events = self.context.Queue()
            self._enqueued_ids.clear()

    close = shutdown

    def _event_loop(self) -> None:
        if self.events is None:
            return
        while not self._stop.is_set():
            if self.process and not self.process.is_alive() and self.process.exitcode is not None:
                self._mark_worker_lost()
            try:
                event = self.events.get(timeout=0.2)
            except queue.Empty:
                continue
            self._handle_event(event)

    def _mark_worker_lost(self) -> None:
        process = self.process
        if self._stop.is_set() or process is None or process.exitcode is None:
            return
        reason = f"Simulation worker exited unexpectedly (exit code {process.exitcode})"
        for run in self.store.list_runs(1000):
            if run.get("status") in {"preparing", "running", "paused"}:
                self.store.update_run(run["run_id"], status="failed", error=reason)
        self._active_run_id = None
        self.process = None
        if any(run.get("status") == "queued" for run in self.store.list_runs(1000)):
            # A dead process may have left tasks in its queue.  Replacing the
            # task/control queues prevents stale messages from running twice,
            # while the existing event queue keeps this sole consumer alive.
            self.tasks = self.context.Queue()
            self.controls = self.context.Queue()
            self._enqueued_ids.clear()
            self.start()

    def _drain_events(self) -> None:
        if self.events is None:
            return
        while True:
            try:
                self._handle_event(self.events.get_nowait())
            except queue.Empty:
                return

    def _handle_event(self, event: dict[str, Any]) -> None:
        run_id = event.get("run_id")
        if not run_id:
            return
        kind = event.get("kind")
        if kind == "status":
            current = self.store.get_run(run_id)
            if current is not None and current.get("status") == "cancelled" and event.get("status") not in {"cancelled", "failed"}:
                return
            if event.get("status") in {"cancelled", "failed", "completed", "interrupted"} and self._active_run_id == run_id:
                self._active_run_id = None
            if event.get("status") in {"preparing", "running", "paused"}:
                self._active_run_id = run_id
            self.store.update_run(
                run_id,
                status=event.get("status"),
                progress=event.get("progress"),
                sim_time=event.get("time"),
            )
        elif kind == "metadata":
            self.store.update_run(run_id, metadata=event.get("metadata") or {})
        elif kind == "frame":
            frame_time = event.get("frame", {}).get("time", 0)
            run = self.store.get_run(run_id)
            duration = 0.0
            if run:
                duration = float(run.get("config", {}).get("numerics", {}).get("duration", 0) or 0)
            self.store.update_run(
                run_id,
                progress=(float(frame_time) / duration if duration > 0 else 0),
                sim_time=frame_time,
            )
        elif kind == "result":
            self._active_run_id = None
            self.store.update_run(
                run_id,
                status="completed",
                progress=1.0,
                sim_time=event.get("time", 0),
                summary=event.get("summary") or {},
                metadata=event.get("metadata") or {},
            )
        elif kind == "error":
            self._active_run_id = None
            self.store.update_run(
                run_id,
                status="failed",
                progress=event.get("progress", 0),
                sim_time=event.get("time", 0),
                error=event.get("error", "Simulation failed"),
            )

    def create_run(self, config: RunConfig) -> dict[str, Any]:
        self.start()
        run = self.store.create_run(config)
        self._enqueue_run(run)
        return run

    def _enqueue_run(self, run: dict[str, Any]) -> None:
        run_id = run["run_id"]
        if self.tasks is None or run_id in self._enqueued_ids or run.get("status") != "queued":
            return
        self.tasks.put({"run_id": run_id, "config": run["config"]})
        self._enqueued_ids.add(run_id)

    def _enqueue_existing(self) -> None:
        for run in self.store.list_runs(1000):
            self._enqueue_run(run)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        return self.store.get_run(run_id)

    def list_runs(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.store.list_runs(limit)

    def frames(self, run_id: str, start: int = 0, limit: int = 120) -> dict[str, Any]:
        run = self.get_run(run_id)
        if run is None:
            raise KeyError(run_id)
        frames, total = read_frames(self.store.run_dir(run_id) / "trajectory.h5", start, limit)
        return {"frames": frames, "total": total, "metadata": run.get("metadata") or {}}

    def command(self, run_id: str, action: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        if run is None:
            raise KeyError(run_id)
        status = run["status"]
        if status in {"completed", "failed", "cancelled", "interrupted"}:
            raise ValueError(f"Cannot {action} a {status} run")
        if status in {"queued", "paused"} and self._active_run_id != run_id:
            if action == "cancel":
                self.store.update_run(run_id, status="cancelled")
                return self.get_run(run_id)  # type: ignore[return-value]
            if action == "pause":
                self.store.update_run(run_id, status="paused")
                return self.get_run(run_id)  # type: ignore[return-value]
            if action == "resume":
                self.store.update_run(run_id, status="queued")
                return self.get_run(run_id)  # type: ignore[return-value]
        self.controls.put({"run_id": run_id, "action": action})
        return self.get_run(run_id)  # type: ignore[return-value]

    def create_sweep(self, config: SweepConfig) -> dict[str, Any]:
        sweep_id = uuid.uuid4().hex
        config_data = config.model_dump(mode="json")
        axes = [axis.model_dump(mode="json") for axis in config.axes]
        combinations = itertools.product(*(axis["values"] for axis in axes))
        prepared: list[tuple[dict[str, Any], RunConfig]] = []
        # Validate every combination before creating the sweep row or enqueueing
        # a single run.  A typo in an axis must not leave a half-started scan.
        for index, values in enumerate(combinations):
            parameters = {axis["parameter"]: value for axis, value in zip(axes, values)}
            config_dict = config.base.model_dump(mode="json")
            for parameter, value in parameters.items():
                self._set_parameter(config_dict, parameter, value)
            run_config = RunConfig.model_validate(config_dict)
            prepared.append((parameters, run_config))
        self.store.create_sweep(sweep_id, config.name, config_data)
        for index, (parameters, run_config) in enumerate(prepared):
            run = self.create_run(run_config)
            self.store.add_sweep_item(sweep_id, index, parameters, run["run_id"])
        self.store.update_sweep_status(sweep_id, "running")
        return self.store.get_sweep(sweep_id)  # type: ignore[return-value]

    @staticmethod
    def _set_parameter(config: dict[str, Any], parameter: str, value: float) -> None:
        path = parameter.split(".")
        if len(path) == 1:
            # A dotted path is preferred, but accepting a unique leaf keeps
            # sweep CSVs concise for common parameters such as ``mass``.
            matches: list[dict[str, Any]] = []
            for section in ("material", "scene", "numerics"):
                candidate = config.get(section)
                if isinstance(candidate, dict) and path[0] in candidate:
                    matches.append(candidate)
            if len(matches) != 1:
                raise ValueError(f"Sweep parameter must identify one field: {parameter}")
            matches[0][path[0]] = value
            return
        target: Any = config
        for key in path[:-1]:
            if not isinstance(target, dict) or key not in target:
                raise ValueError(f"Unknown sweep parameter: {parameter}")
            target = target[key]
        if not isinstance(target, dict) or path[-1] not in target:
            raise ValueError(f"Unknown sweep parameter: {parameter}")
        target[path[-1]] = value

    def sweep(self, sweep_id: str) -> dict[str, Any] | None:
        return self.store.get_sweep(sweep_id)

    def add_reference(self, run_id: str, metric: str, content: bytes | str) -> dict[str, Any]:
        run = self.get_run(run_id)
        if run is None:
            raise KeyError(run_id)
        text = content.decode("utf-8-sig") if isinstance(content, bytes) else content
        reader = csv.DictReader(text.splitlines())
        if not reader.fieldnames or "time" not in reader.fieldnames or "value" not in reader.fieldnames:
            raise ValueError("Reference CSV must contain time and value columns")
        times: list[float] = []
        values: list[float] = []
        for row in reader:
            try:
                t = float(row["time"])
                v = float(row["value"])
            except (TypeError, ValueError):
                raise ValueError("Reference CSV contains non-numeric time or value") from None
            if not np.isfinite(t) or not np.isfinite(v):
                raise ValueError("Reference CSV contains non-finite values")
            times.append(t)
            values.append(v)
        if len(times) < 2:
            raise ValueError("Reference CSV must contain at least two samples")
        order = np.argsort(np.asarray(times), kind="stable")
        ref_times = np.asarray(times, dtype=float)[order]
        ref_values = np.asarray(values, dtype=float)[order]
        if np.unique(ref_times).size != ref_times.size:
            raise ValueError("Reference CSV contains duplicate time values")
        frames, _ = read_frames(self.store.run_dir(run_id) / "trajectory.h5", 0, None)
        sim_times: list[float] = []
        sim_values: list[float] = []
        for frame in frames:
            if metric not in frame.get("metrics", {}):
                continue
            value = float(frame["metrics"][metric])
            if np.isfinite(value):
                sim_times.append(float(frame["time"]))
                sim_values.append(value)
        if len(sim_times) < 2:
            raise ValueError(f"Metric {metric!r} is not available in trajectory frames")
        sample_times = np.asarray(sim_times, dtype=float)
        sample_values = np.asarray(sim_values, dtype=float)
        if np.unique(sample_times).size != sample_times.size:
            unique, indices = np.unique(sample_times, return_index=True)
            sample_times = unique
            sample_values = sample_values[indices]
        common_start = max(float(sample_times.min()), float(ref_times.min()))
        common_end = min(float(sample_times.max()), float(ref_times.max()))
        if common_end <= common_start:
            raise ValueError("Reference and simulation time ranges do not overlap")
        covered = (ref_times >= common_start) & (ref_times <= common_end)
        if int(np.count_nonzero(covered)) < 2:
            raise ValueError("Reference has fewer than two samples in the common time range")
        comparison = compare_reference(
            sample_times[(sample_times >= common_start) & (sample_times <= common_end)],
            sample_values[(sample_times >= common_start) & (sample_times <= common_end)],
            ref_times,
            ref_values,
        )
        coverage = float(np.count_nonzero(covered) / ref_times.size)
        comparison = {
            "metric": metric,
            "rmse": comparison["rmse"],
            "mae": comparison["mae"],
            "max_abs_error": comparison["max_abs_error"],
            "samples": int(np.count_nonzero(covered)),
            "reference_samples": int(ref_times.size),
            "coverage": coverage,
            "common_time": [common_start, common_end],
            "reference_time": ref_times.tolist(),
            "reference_value": ref_values.tolist(),
            "interpolated_value": comparison["reference"],
            "comparison_time": comparison["time"],
            "comparison_reference": comparison["reference"],
            "comparison_value": comparison["values"],
        }
        self.store.write_reference(run_id, metric, ref_times.tolist(), ref_values.tolist(), comparison)
        summary = dict(run.get("summary") or {})
        summary.update(
            {
                "reference_metric": metric,
                "reference_rmse": comparison["rmse"],
                "reference_coverage": coverage,
            }
        )
        self.store.write_summary(run_id, summary)
        updated = self.store.get_run(run_id) or run
        return {
            **updated,
            "comparison": comparison,
            **comparison,
            "reference_metric": metric,
            "reference_rmse": comparison["rmse"],
            "reference_coverage": coverage,
        }
