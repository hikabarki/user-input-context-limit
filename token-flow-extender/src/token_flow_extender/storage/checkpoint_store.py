"""Checkpoint persistence: SQLite default, Redis optional, memory for tests.

Rules enforced here:
  - A checkpoint is written BEFORE and AFTER every external LLM call.
  - Serialization is JSON only (no pickle).
  - Writes are transactional (SQLite WAL + explicit commits).
  - Resume never reprocesses completed windows and never loses results.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

from ..exceptions import CheckpointError
from ..utils.logger import get_logger

logger = get_logger(__name__)

JOB_STATUSES = ("pending", "running", "paused", "completed", "failed")


@dataclass
class JobRecord:
    """Persistent job metadata."""

    job_id: str
    status: str
    model: str
    task: str
    strategy: str
    input_sha256: str
    total_windows: int
    config_json: str
    result: Optional[str] = None
    error: Optional[str] = None
    created_at: float = 0.0
    updated_at: float = 0.0


class CheckpointStore(ABC):
    """Abstract checkpoint backend."""

    @abstractmethod
    def create_job(self, record: JobRecord) -> None:
        """Insert a job if absent (idempotent on job_id)."""

    @abstractmethod
    def get_job(self, job_id: str) -> Optional[JobRecord]:
        """Fetch a job record or None."""

    @abstractmethod
    def set_status(self, job_id: str, status: str, error: Optional[str] = None) -> None:
        """Transition a job's status."""

    @abstractmethod
    def save_window_result(self, job_id: str, window_index: int, result: str,
                           token_count: int, char_start: int, char_end: int) -> None:
        """Persist one completed window's result (transactional)."""

    @abstractmethod
    def get_window_results(self, job_id: str) -> Dict[int, str]:
        """Return {window_index: result} for all completed windows."""

    @abstractmethod
    def save_final(self, job_id: str, result: str) -> None:
        """Persist the merged final result and mark the job completed."""

    def completed_indices(self, job_id: str) -> List[int]:
        """Sorted indices of completed windows (used by resume)."""
        return sorted(self.get_window_results(job_id).keys())


class SQLiteCheckpointStore(CheckpointStore):
    """Default durable backend. One file, WAL mode, thread-safe."""

    def __init__(self, db_path: str = "token_flow.db") -> None:
        self._path = db_path
        self._lock = threading.Lock()
        try:
            self._init_schema()
        except sqlite3.Error as exc:
            raise CheckpointError(f"Could not initialize SQLite at {db_path}: {exc}") from exc

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    model TEXT NOT NULL,
                    task TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    input_sha256 TEXT NOT NULL,
                    total_windows INTEGER NOT NULL,
                    config_json TEXT NOT NULL,
                    result TEXT,
                    error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )""")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS windows (
                    job_id TEXT NOT NULL,
                    window_index INTEGER NOT NULL,
                    result TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    char_start INTEGER NOT NULL,
                    char_end INTEGER NOT NULL,
                    completed_at REAL NOT NULL,
                    PRIMARY KEY (job_id, window_index)
                )""")
            conn.commit()

    def create_job(self, record: JobRecord) -> None:
        now = time.time()
        try:
            with self._lock, self._connect() as conn:
                conn.execute(
                    """INSERT OR IGNORE INTO jobs
                       (job_id, status, model, task, strategy, input_sha256,
                        total_windows, config_json, result, error, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?)""",
                    (record.job_id, record.status, record.model, record.task,
                     record.strategy, record.input_sha256, record.total_windows,
                     record.config_json, now, now),
                )
                conn.commit()
        except sqlite3.Error as exc:
            raise CheckpointError(f"create_job failed: {exc}") from exc

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        try:
            with self._lock, self._connect() as conn:
                row = conn.execute(
                    "SELECT job_id, status, model, task, strategy, input_sha256, "
                    "total_windows, config_json, result, error, created_at, updated_at "
                    "FROM jobs WHERE job_id = ?", (job_id,)
                ).fetchone()
        except sqlite3.Error as exc:
            raise CheckpointError(f"get_job failed: {exc}") from exc
        if row is None:
            return None
        return JobRecord(*row)

    def set_status(self, job_id: str, status: str, error: Optional[str] = None) -> None:
        if status not in JOB_STATUSES:
            raise CheckpointError(f"Invalid job status {status!r}")
        try:
            with self._lock, self._connect() as conn:
                conn.execute(
                    "UPDATE jobs SET status = ?, error = ?, updated_at = ? WHERE job_id = ?",
                    (status, error, time.time(), job_id),
                )
                conn.commit()
        except sqlite3.Error as exc:
            raise CheckpointError(f"set_status failed: {exc}") from exc

    def save_window_result(self, job_id: str, window_index: int, result: str,
                           token_count: int, char_start: int, char_end: int) -> None:
        try:
            with self._lock, self._connect() as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO windows
                       (job_id, window_index, result, token_count, char_start, char_end, completed_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (job_id, window_index, result, token_count, char_start, char_end, time.time()),
                )
                conn.execute("UPDATE jobs SET updated_at = ? WHERE job_id = ?",
                             (time.time(), job_id))
                conn.commit()
        except sqlite3.Error as exc:
            raise CheckpointError(f"save_window_result failed: {exc}") from exc

    def get_window_results(self, job_id: str) -> Dict[int, str]:
        try:
            with self._lock, self._connect() as conn:
                rows = conn.execute(
                    "SELECT window_index, result FROM windows WHERE job_id = ? "
                    "ORDER BY window_index", (job_id,)
                ).fetchall()
        except sqlite3.Error as exc:
            raise CheckpointError(f"get_window_results failed: {exc}") from exc
        return {int(index): result for index, result in rows}

    def save_final(self, job_id: str, result: str) -> None:
        try:
            with self._lock, self._connect() as conn:
                conn.execute(
                    "UPDATE jobs SET result = ?, status = 'completed', updated_at = ? "
                    "WHERE job_id = ?",
                    (result, time.time(), job_id),
                )
                conn.commit()
        except sqlite3.Error as exc:
            raise CheckpointError(f"save_final failed: {exc}") from exc


class MemoryCheckpointStore(CheckpointStore):
    """Non-durable backend for tests and ephemeral runs."""

    def __init__(self) -> None:
        self._jobs: Dict[str, JobRecord] = {}
        self._windows: Dict[str, Dict[int, str]] = {}
        self._lock = threading.Lock()

    def create_job(self, record: JobRecord) -> None:
        with self._lock:
            self._jobs.setdefault(record.job_id, record)
            self._windows.setdefault(record.job_id, {})

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            return self._jobs.get(job_id)

    def set_status(self, job_id: str, status: str, error: Optional[str] = None) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = status
                job.error = error
                job.updated_at = time.time()

    def save_window_result(self, job_id: str, window_index: int, result: str,
                           token_count: int, char_start: int, char_end: int) -> None:
        with self._lock:
            self._windows.setdefault(job_id, {})[window_index] = result

    def get_window_results(self, job_id: str) -> Dict[int, str]:
        with self._lock:
            return dict(self._windows.get(job_id, {}))

    def save_final(self, job_id: str, result: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.result = result
                job.status = "completed"


class RedisCheckpointStore(CheckpointStore):
    """Optional Redis backend (extra: 'redis'). JSON values only."""

    def __init__(self, redis_url: str) -> None:
        try:
            import redis  # type: ignore[import-not-found]
        except ImportError as exc:
            raise CheckpointError(
                "redis is not installed; install token-flow-extender[redis]"
            ) from exc
        self._redis = redis.Redis.from_url(redis_url, decode_responses=True)

    @staticmethod
    def _job_key(job_id: str) -> str:
        return f"tfe:job:{job_id}"

    @staticmethod
    def _windows_key(job_id: str) -> str:
        return f"tfe:windows:{job_id}"

    def create_job(self, record: JobRecord) -> None:
        key = self._job_key(record.job_id)
        if not self._redis.exists(key):
            record.created_at = record.updated_at = time.time()
            self._redis.set(key, json.dumps(record.__dict__))

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        raw = self._redis.get(self._job_key(job_id))
        return JobRecord(**json.loads(raw)) if raw else None

    def set_status(self, job_id: str, status: str, error: Optional[str] = None) -> None:
        job = self.get_job(job_id)
        if job:
            job.status, job.error, job.updated_at = status, error, time.time()
            self._redis.set(self._job_key(job_id), json.dumps(job.__dict__))

    def save_window_result(self, job_id: str, window_index: int, result: str,
                           token_count: int, char_start: int, char_end: int) -> None:
        self._redis.hset(self._windows_key(job_id), str(window_index), result)

    def get_window_results(self, job_id: str) -> Dict[int, str]:
        raw = self._redis.hgetall(self._windows_key(job_id))
        return {int(k): v for k, v in raw.items()}

    def save_final(self, job_id: str, result: str) -> None:
        job = self.get_job(job_id)
        if job:
            job.result, job.status, job.updated_at = result, "completed", time.time()
            self._redis.set(self._job_key(job_id), json.dumps(job.__dict__))


def build_store(backend: str, db_path: str = "token_flow.db",
                redis_url: str = "redis://localhost:6379/0") -> CheckpointStore:
    """Factory used by the SDK: sqlite (default), redis, or memory."""
    if backend == "sqlite":
        return SQLiteCheckpointStore(db_path)
    if backend == "redis":
        return RedisCheckpointStore(redis_url)
    if backend == "memory":
        return MemoryCheckpointStore()
    raise CheckpointError(f"Unknown storage backend {backend!r}")
