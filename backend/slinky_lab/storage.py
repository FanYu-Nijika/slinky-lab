"""Persistent run and trajectory storage for Slinky Lab.

The service keeps a small SQLite index and stores the potentially large frame
arrays in one HDF5 file per run.  Keeping those two concerns separate makes it
possible to list and resume jobs without loading a complete trajectory into
memory.
"""

from __future__ import annotations

import csv
import errno
import json
import os
import re
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

from .schemas import Frame, RunConfig


TERMINAL_STATUSES = {"completed", "failed", "cancelled", "interrupted"}
ACTIVE_STATUSES = {"queued", "preparing", "running", "paused"}
ARTIFACT_NAMES = {
    "config.json",
    "model.xml",
    "summary.json",
    "metrics.csv",
    "trajectory.h5",
    "reference.csv",
    "comparison.json",
}
RUN_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
JSON_FIELD_BYTES = 65536
CONTACT_DETAILS_BYTES = 524288
CONTACT_COLUMNS = (
    "x",
    "y",
    "z",
    "geom_a",
    "geom_b",
    "normal_force",
    "kind_code",
    "stair_step",
    "material_index",
    "surface_code",
)
CONTACT_KIND_CODES = {"self": 0, "stair": 1, "external": 2}
CONTACT_SURFACE_CODES = {"other": 0, "tread": 1}
CONTACT_KIND_NAMES = {value: key for key, value in CONTACT_KIND_CODES.items()}
CONTACT_SURFACE_NAMES = {value: key for key, value in CONTACT_SURFACE_CODES.items()}


class _TrajectoryFileLock:
    """Serialize HDF5 open, mutation, flush, and close operations per path.

    SWMR protects HDF5 readers from committed-frame visibility races, but it
    does not make a Windows writer close safe while another process opens the
    same file.  A tiny sidecar lock keeps those OS-level operations ordered
    without holding a lock across the simulation; each append owns it only for
    one frame.
    """

    def __init__(self, path: str | Path):
        self.lock_path = Path(f"{Path(path)}.lock")
        self._handle: Any = None
        self._windows = os.name == "nt"

    def __enter__(self) -> "_TrajectoryFileLock":
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path.touch(exist_ok=True)
        self._handle = self.lock_path.open("r+b")
        if self._windows:
            import msvcrt

            self._handle.seek(0, os.SEEK_END)
            if self._handle.tell() == 0:
                self._handle.write(b"\0")
                self._handle.flush()
            self._handle.seek(0)
            while True:
                try:
                    msvcrt.locking(self._handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as exc:
                    if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                        self._handle.close()
                        self._handle = None
                        raise
                    time.sleep(0.01)
                    self._handle.seek(0)
        else:
            import fcntl

            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            if self._windows:
                import msvcrt

                handle.seek(0)
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_default(value: Any) -> Any:
    """Convert common numerical values before serialising API/artifact JSON."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def dump_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=json_default, allow_nan=False, separators=(",", ":"))


def load_json(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return fallback


class DataStore:
    """SQLite index plus per-run files.

    A connection is opened for each operation.  This is deliberate: the API
    thread, event thread, and worker process can all touch the same database,
    while SQLite's WAL mode keeps those short transactions independent.
    """

    def __init__(self, data_dir: str | Path | None = None, recover: bool = True):
        configured = data_dir or os.environ.get("SLINKY_DATA_DIR", "./data")
        self.root = Path(configured).expanduser().resolve()
        self.runs_dir = self.root / "runs"
        self.root.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "slinky.sqlite3"
        self._lock = threading.RLock()
        self._initialise()
        if recover:
            self.recover_active_runs()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path), timeout=30, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _initialise(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    summary_json TEXT NOT NULL DEFAULT '{}',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    error TEXT,
                    progress REAL NOT NULL DEFAULT 0,
                    sim_time REAL NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sweeps (
                    sweep_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sweep_items (
                    sweep_id TEXT NOT NULL,
                    item_index INTEGER NOT NULL,
                    parameters_json TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    summary_json TEXT NOT NULL DEFAULT '{}',
                    error TEXT,
                    PRIMARY KEY (sweep_id, item_index),
                    UNIQUE (sweep_id, run_id)
                );
                CREATE TABLE IF NOT EXISTS references_table (
                    run_id TEXT PRIMARY KEY,
                    metric TEXT NOT NULL,
                    times_json TEXT NOT NULL,
                    values_json TEXT NOT NULL,
                    comparison_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
                CREATE INDEX IF NOT EXISTS idx_sweep_items_run ON sweep_items(run_id);
                """
            )

    def recover_active_runs(self) -> int:
        """Mark jobs from a previous service instance as interrupted."""

        now = utc_now()
        with self._lock, self._connect() as connection:
            result = connection.execute(
                "UPDATE runs SET status='interrupted', error=?, updated_at=? "
                "WHERE status IN ('running', 'paused', 'preparing')",
                ("Service restarted while the run was active", now),
            )
            return result.rowcount

    def run_dir(self, run_id: str) -> Path:
        if not RUN_ID_PATTERN.fullmatch(str(run_id)):
            raise ValueError("Invalid run id")
        with self._lock, self._connect() as connection:
            exists = connection.execute("SELECT 1 FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if exists is None:
            raise FileNotFoundError(run_id)
        path = self.runs_dir / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def create_run(self, config: RunConfig, run_id: str | None = None) -> dict[str, Any]:
        run_id = run_id or uuid.uuid4().hex
        run_id = str(run_id)
        if not RUN_ID_PATTERN.fullmatch(str(run_id)):
            raise ValueError("Invalid run id")
        now = utc_now()
        config_data = config.model_dump(mode="json")
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO runs(run_id,status,config_json,created_at,updated_at) VALUES(?,?,?,?,?)",
                (run_id, "queued", dump_json(config_data), now, now),
            )
        run_path = self.run_dir(run_id)
        (run_path / "config.json").write_text(
            json.dumps(config_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return self.get_run(run_id)  # type: ignore[return-value]

    def update_run(
        self,
        run_id: str,
        status: str | None = None,
        progress: float | None = None,
        sim_time: float | None = None,
        summary: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        values: list[Any] = []
        clauses: list[str] = []
        if status is not None:
            clauses.append("status=?")
            values.append(status)
        if progress is not None:
            clauses.append("progress=?")
            values.append(float(max(0.0, min(1.0, progress))))
        if sim_time is not None:
            clauses.append("sim_time=?")
            values.append(float(sim_time))
        if summary is not None:
            clauses.append("summary_json=?")
            values.append(dump_json(summary))
        if metadata is not None:
            clauses.append("metadata_json=?")
            values.append(dump_json(metadata))
        if error is not None:
            clauses.append("error=?")
            values.append(str(error))
        if not clauses:
            return
        clauses.append("updated_at=?")
        values.append(utc_now())
        values.append(run_id)
        with self._lock, self._connect() as connection:
            connection.execute(f"UPDATE runs SET {', '.join(clauses)} WHERE run_id=?", values)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            return None
        return self._run_row(row)

    def list_runs(self, limit: int = 100) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 1000))
        with self._lock, self._connect() as connection:
            rows = connection.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._run_row(row) for row in rows]

    def _run_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "run_id": row["run_id"],
            "status": row["status"],
            "config": load_json(row["config_json"], {}),
            "summary": load_json(row["summary_json"], {}),
            "metadata": load_json(row["metadata_json"], {}),
            "error": row["error"],
            "progress": float(row["progress"] or 0),
            "time": float(row["sim_time"] or 0),
            "artifacts": self.artifacts(row["run_id"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def artifacts(self, run_id: str) -> list[str]:
        path = self.run_dir(run_id)
        return sorted(item.name for item in path.iterdir() if item.is_file() and item.name in ARTIFACT_NAMES)

    def artifact_path(self, run_id: str, name: str) -> Path:
        if name not in ARTIFACT_NAMES:
            raise ValueError("Unknown artifact")
        if not RUN_ID_PATTERN.fullmatch(str(run_id)):
            raise ValueError("Invalid run id")
        with self._lock, self._connect() as connection:
            exists = connection.execute("SELECT 1 FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if exists is None:
            raise FileNotFoundError(run_id)
        path = (self.run_dir(run_id) / name).resolve()
        if path.parent != self.run_dir(run_id).resolve():
            raise ValueError("Invalid artifact path")
        if not path.is_file():
            raise FileNotFoundError(name)
        return path

    def write_model_xml(self, run_id: str, model_xml: str | bytes | None) -> None:
        if model_xml is None:
            return
        data = model_xml.encode("utf-8") if isinstance(model_xml, str) else bytes(model_xml)
        self.run_dir(run_id).joinpath("model.xml").write_bytes(data)

    def write_summary(self, run_id: str, summary: dict[str, Any]) -> None:
        self.run_dir(run_id).joinpath("summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8"
        )
        self.update_run(run_id, summary=summary)

    def write_metrics_csv(self, run_id: str, frames: Iterable[dict[str, Any]]) -> None:
        rows = list(frames)
        keys: set[str] = set()
        for frame in rows:
            keys.update(str(key) for key in (frame.get("metrics") or {}))
        columns = ["time", *sorted(keys)]
        with self.run_dir(run_id).joinpath("metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for frame in rows:
                values = {"time": frame.get("time", 0)}
                values.update((frame.get("metrics") or {}))
                writer.writerow({column: values.get(column, "") for column in columns})

    def write_reference(
        self,
        run_id: str,
        metric: str,
        times: list[float],
        values: list[float],
        comparison: dict[str, Any],
    ) -> None:
        path = self.run_dir(run_id) / "reference.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time", "value"])
            writer.writerows(zip(times, values))
        self.run_dir(run_id).joinpath("comparison.json").write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8"
        )
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO references_table(run_id,metric,times_json,values_json,comparison_json,created_at) "
                "VALUES(?,?,?,?,?,?) ON CONFLICT(run_id) DO UPDATE SET metric=excluded.metric, "
                "times_json=excluded.times_json, values_json=excluded.values_json, "
                "comparison_json=excluded.comparison_json, created_at=excluded.created_at",
                (run_id, metric, dump_json(times), dump_json(values), dump_json(comparison), utc_now()),
            )

    def get_reference(self, run_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT * FROM references_table WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            return None
        return {
            "run_id": run_id,
            "metric": row["metric"],
            "times": load_json(row["times_json"], []),
            "values": load_json(row["values_json"], []),
            "comparison": load_json(row["comparison_json"], {}),
        }

    def create_sweep(self, sweep_id: str, name: str, config: dict[str, Any]) -> None:
        now = utc_now()
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO sweeps(sweep_id,name,config_json,status,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                (sweep_id, name, dump_json(config), "queued", now, now),
            )

    def add_sweep_item(
        self, sweep_id: str, item_index: int, parameters: dict[str, Any], run_id: str, status: str = "queued"
    ) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO sweep_items(sweep_id,item_index,parameters_json,run_id,status) VALUES(?,?,?,?,?)",
                (sweep_id, item_index, dump_json(parameters), run_id, status),
            )

    def get_sweep(self, sweep_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            sweep = connection.execute("SELECT * FROM sweeps WHERE sweep_id=?", (sweep_id,)).fetchone()
            items = connection.execute(
                "SELECT * FROM sweep_items WHERE sweep_id=? ORDER BY item_index", (sweep_id,)
            ).fetchall()
        if sweep is None:
            return None
        output_items = []
        for item in items:
            run = self.get_run(item["run_id"])
            output_items.append(
                {
                    "index": item["item_index"],
                    "parameters": load_json(item["parameters_json"], {}),
                    "run_id": item["run_id"],
                    "status": run["status"] if run else item["status"],
                    "summary": run["summary"] if run else load_json(item["summary_json"], {}),
                    "error": run["error"] if run else item["error"],
                }
            )
        statuses = [item["status"] for item in output_items]
        if statuses and all(status in TERMINAL_STATUSES for status in statuses):
            status = "failed" if any(item["status"] == "failed" for item in output_items) else "completed"
        elif any(status in {"preparing", "running", "paused"} for status in statuses):
            status = "running"
        else:
            status = sweep["status"]
        completed_count = sum(status in TERMINAL_STATUSES for status in statuses)
        progress = float(completed_count / len(statuses)) if statuses else 0.0
        return {
            "sweep_id": sweep_id,
            "name": sweep["name"],
            "status": status,
            "config": load_json(sweep["config_json"], {}),
            "items": output_items,
            # ``runs`` is the browser-facing alias; ``items`` remains useful
            # to batch clients that want the explicit grid index.
            "runs": output_items,
            "progress": progress,
            "summary": {"results": output_items},
            "created_at": sweep["created_at"],
            "updated_at": sweep["updated_at"],
        }

    def update_sweep_status(self, sweep_id: str, status: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("UPDATE sweeps SET status=?,updated_at=? WHERE sweep_id=?", (status, utc_now(), sweep_id))


class TrajectoryWriter:
    """Append-only v3 HDF5 writer with numeric variable-contact arrays.

    Contact counts are stored beside the growable arrays.  The reader uses the
    committed frame attribute as the publication barrier, so a reader never
    observes a frame while its arrays are being resized or filled.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file: h5py.File | None = None
        self._count = 0
        self._shape: tuple[int, int] | None = None
        with _TrajectoryFileLock(self.path):
            self.file = h5py.File(self.path, "w", libver="latest")
            self.file.attrs["format"] = "slinky-lab-trajectory-v3"
            self.file.attrs["contact_columns"] = json.dumps(CONTACT_COLUMNS, separators=(",", ":"))
            self.file.attrs["contact_kind_codes"] = json.dumps(CONTACT_KIND_CODES, separators=(",", ":"))
            self.file.attrs["contact_surface_codes"] = json.dumps(CONTACT_SURFACE_CODES, separators=(",", ":"))
            self.file.create_dataset("time", shape=(0,), maxshape=(None,), dtype="f8", compression="gzip", compression_opts=4)
            self.file.create_dataset("positions", shape=(0, 0, 3), maxshape=(None, None, 3), dtype="f8", compression="gzip", compression_opts=4)
            self.file.create_dataset("quaternions", shape=(0, 0, 4), maxshape=(None, None, 4), dtype="f8", compression="gzip", compression_opts=4)
            self.file.create_dataset("contacts", shape=(0, 0, 3), maxshape=(None, None, 3), dtype="f8", compression="gzip", compression_opts=4, shuffle=True)
            self.file.create_dataset("contact_counts", shape=(0,), maxshape=(None,), dtype="u8", compression="gzip", compression_opts=4, shuffle=True)
            self.file.create_dataset("contact_details", shape=(0, 0, len(CONTACT_COLUMNS)), maxshape=(None, None, len(CONTACT_COLUMNS)), dtype="f8", compression="gzip", compression_opts=4, shuffle=True)
            self.file.create_dataset("contact_detail_counts", shape=(0,), maxshape=(None,), dtype="u8", compression="gzip", compression_opts=4, shuffle=True)
            self.file.create_dataset("metrics_json", shape=(0,), maxshape=(None,), dtype=f"S{JSON_FIELD_BYTES}", compression="gzip", compression_opts=4, shuffle=True)
            self.file.attrs["committed_frames"] = 0
            self.file.flush()
            self.file.swmr_mode = True

    def _open_file(self) -> h5py.File:
        if self.file is None:
            raise RuntimeError("TrajectoryWriter is closed")
        return self.file

    def _resize_first_axis(self, name: str, count: int) -> None:
        dataset = self._open_file()[name]
        shape = list(dataset.shape)
        shape[0] = count
        dataset.resize(tuple(shape))

    def _resize_contact_axis(self, name: str, frame_count: int, width: int) -> None:
        dataset = self._open_file()[name]
        shape = list(dataset.shape)
        shape[0] = frame_count
        shape[1] = max(shape[1], width)
        dataset.resize(tuple(shape))

    @staticmethod
    def _contact_detail_array(details: list[Any]) -> np.ndarray:
        result = np.zeros((len(details), len(CONTACT_COLUMNS)), dtype=np.float64)
        for index, detail in enumerate(details):
            # ``Frame`` validation makes each entry a Contact model.  Keeping
            # this explicit also supports callers that pass a plain mapping.
            item = detail.model_dump() if hasattr(detail, "model_dump") else detail
            result[index, 0:3] = np.asarray(item["position"], dtype=np.float64)
            result[index, 3] = int(item["geom_a"])
            result[index, 4] = int(item["geom_b"])
            result[index, 5] = float(item["normal_force"])
            result[index, 6] = CONTACT_KIND_CODES[item["kind"]]
            result[index, 7] = -1 if item["stair_step"] is None else int(item["stair_step"])
            result[index, 8] = -1 if item["material_index"] is None else int(item["material_index"])
            result[index, 9] = CONTACT_SURFACE_CODES[item["surface"]]
        return result

    def append(self, frame: Frame | dict[str, Any]) -> None:
        model = frame if isinstance(frame, Frame) else Frame.model_validate(frame)
        with _TrajectoryFileLock(self.path):
            positions = np.asarray(model.positions, dtype=np.float64)
            quaternions = np.asarray(model.quaternions, dtype=np.float64)
            contacts = np.asarray(model.contacts, dtype=np.float64)
            if contacts.size == 0:
                contacts = np.empty((0, 3), dtype=np.float64)
            details = self._contact_detail_array(model.contact_details)
            if positions.ndim != 2 or positions.shape[-1] != 3:
                raise ValueError("Frame positions must have shape (n, 3)")
            if quaternions.ndim != 2 or quaternions.shape[-1] != 4 or len(quaternions) != len(positions):
                raise ValueError("Frame quaternions must have shape (n, 4)")
            if contacts.ndim != 2 or contacts.shape[-1] != 3:
                raise ValueError("Frame contacts must have shape (n, 3)")
            if details.ndim != 2 or details.shape[-1] != len(CONTACT_COLUMNS):
                raise ValueError("Frame contact details have an invalid numeric shape")
            if (not np.isfinite(positions).all() or not np.isfinite(quaternions).all()
                    or not np.isfinite(contacts).all() or not np.isfinite(details).all()
                    or not np.isfinite(model.time)):
                raise ValueError("Frame contains non-finite values")
            metrics_encoded = dump_json(model.metrics).encode("utf-8")
            if len(metrics_encoded) > JSON_FIELD_BYTES:
                raise ValueError("metrics_json exceeds the trajectory JSON field limit")
            file = self._open_file()
            if self._shape is None:
                self._shape = (len(positions), len(quaternions))
                file["positions"].resize((0, len(positions), 3))
                file["quaternions"].resize((0, len(quaternions), 4))
            if len(positions) != self._shape[0] or len(quaternions) != self._shape[1]:
                raise ValueError("Frame geometry count changed during a run")
            new_count = self._count + 1
            for name, value in (("time", model.time), ("positions", positions), ("quaternions", quaternions)):
                self._resize_first_axis(name, new_count)
                file[name][self._count] = value
            self._resize_contact_axis("contacts", new_count, len(contacts))
            file["contacts"][self._count, :len(contacts)] = contacts
            file["contact_counts"].resize((new_count,))
            file["contact_counts"][self._count] = len(contacts)
            self._resize_contact_axis("contact_details", new_count, len(details))
            file["contact_details"][self._count, :len(details)] = details
            file["contact_detail_counts"].resize((new_count,))
            file["contact_detail_counts"][self._count] = len(details)
            self._resize_first_axis("metrics_json", new_count)
            file["metrics_json"][self._count] = metrics_encoded
            self._count = new_count
            self._flush_unlocked()

    def _flush_unlocked(self) -> None:
        file = self._open_file()
        file.flush()
        file.attrs["committed_frames"] = self._count
        file.flush()

    def flush(self) -> None:
        with _TrajectoryFileLock(self.path):
            self._flush_unlocked()

    def close(self) -> None:
        with _TrajectoryFileLock(self.path):
            if self.file is not None:
                self._flush_unlocked()
                file = self.file
                self.file = None
                file.close()

    def __enter__(self) -> "TrajectoryWriter":
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()


def _decode_fixed_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.rstrip(b"\x00").decode("utf-8")
    return str(value)


def _decode_contact_details(
    rows: np.ndarray,
    count: int,
    kind_names: dict[int, str] | None = None,
    surface_names: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    kind_names = kind_names or CONTACT_KIND_NAMES
    surface_names = surface_names or CONTACT_SURFACE_NAMES
    for row in np.asarray(rows[:max(0, count)], dtype=np.float64):
        kind_code = int(round(float(row[6])))
        surface_code = int(round(float(row[9])))
        stair_step = int(round(float(row[7])))
        material_index = int(round(float(row[8])))
        details.append({
            "position": [float(value) for value in row[:3]],
            "geom_a": int(round(float(row[3]))),
            "geom_b": int(round(float(row[4]))),
            "normal_force": float(row[5]),
            "kind": kind_names.get(kind_code, "external"),
            "stair_step": None if stair_step < 0 else stair_step,
            "material_index": None if material_index < 0 else material_index,
            "surface": surface_names.get(surface_code, "other"),
        })
    return details


def _read_code_names(file: h5py.File, attribute: str, fallback: dict[str, int]) -> dict[int, str]:
    raw = file.attrs.get(attribute)
    try:
        encoded = _decode_fixed_string(raw)
        mapping = json.loads(encoded)
        return {int(value): str(key) for key, value in mapping.items()}
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
        return {int(value): key for key, value in fallback.items()}


def read_frames(path: str | Path, start: int = 0, limit: int | None = 120) -> tuple[list[dict[str, Any]], int]:
    path = Path(path)
    if not path.is_file():
        return [], 0
    with _TrajectoryFileLock(path):
        if not path.is_file():
            return [], 0
        start = max(0, int(start))
        try:
            file = h5py.File(path, "r", libver="latest", swmr=True)
        except (OSError, ValueError):
            # A writer may have created the path but not committed its first
            # SWMR metadata flush yet.  The next poll will see the frame.
            return [], 0
        with file:
            base_fields = [name for name in ("time", "positions", "quaternions", "metrics_json") if name in file]
            if len(base_fields) != 4:
                return [], 0
            v3_contacts = "contacts" in file and "contact_counts" in file
            v3_details = v3_contacts and "contact_details" in file and "contact_detail_counts" in file
            kind_names = _read_code_names(file, "contact_kind_codes", CONTACT_KIND_CODES)
            surface_names = _read_code_names(file, "contact_surface_codes", CONTACT_SURFACE_CODES)
            old_contacts = not v3_contacts and "contacts_json" in file
            old_details = old_contacts and "contact_details_json" in file
            fields = list(base_fields)
            if v3_contacts:
                fields.extend(("contacts", "contact_counts"))
            elif old_contacts:
                fields.append("contacts_json")
            if v3_details:
                fields.extend(("contact_details", "contact_detail_counts"))
            elif old_details:
                fields.append("contact_details_json")
            for name in fields:
                refresh = getattr(file[name], "refresh", None)
                if callable(refresh):
                    refresh()
            available = min(int(file[name].shape[0]) for name in fields)
            committed = int(file.attrs.get("committed_frames", available))
            total = min(available, max(0, committed))
            if limit is None:
                limit = total - start
            else:
                limit = max(0, min(int(limit), 1_000_000))
            end = min(total, start + limit)
            frames = []
            for index in range(start, end):
                metrics = load_json(_decode_fixed_string(file["metrics_json"][index]), {})
                if v3_contacts:
                    contact_count = int(file["contact_counts"][index])
                    contacts = file["contacts"][index, :max(0, contact_count)].tolist()
                elif old_contacts:
                    contacts = load_json(_decode_fixed_string(file["contacts_json"][index]), [])
                else:
                    contacts = []
                if v3_details:
                    detail_count = int(file["contact_detail_counts"][index])
                    details = _decode_contact_details(file["contact_details"][index], detail_count, kind_names, surface_names)
                elif old_details:
                    details = load_json(_decode_fixed_string(file["contact_details_json"][index]), [])
                else:
                    details = []
                frames.append({
                    "time": float(file["time"][index]),
                    "positions": file["positions"][index].tolist(),
                    "quaternions": file["quaternions"][index].tolist(),
                    "contacts": contacts,
                    "contact_details": details,
                    "metrics": metrics,
                })
        return frames, total
