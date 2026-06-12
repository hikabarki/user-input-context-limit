"""Job lifecycle management on top of a CheckpointStore.

Status flow: pending -> running -> completed
                       \\-> paused (awaiting capacity: rate limit, budget)
                       \\-> failed (after retries; resumable)
"""

from __future__ import annotations

import hashlib
import json
from typing import Optional

from ..config import TokenFlowConfig
from ..exceptions import JobStateError
from ..utils.logger import get_logger
from .checkpoint_store import CheckpointStore, JobRecord

logger = get_logger(__name__)


def deterministic_job_id(input_text: str, model: str, task: str,
                         config: TokenFlowConfig) -> str:
    """Same input + same config => same job id (idempotent resume)."""
    material = json.dumps({
        "input": hashlib.sha256(input_text.encode("utf-8")).hexdigest(),
        "model": model,
        "task": task,
        "margin": config.safety_margin,
        "carry": config.carry_state_ratio,
        "overlap": config.overlap_ratio,
        "strategy": config.strategy,
        "mode": config.mode,
    }, sort_keys=True)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


class JobQueue:
    """Thin lifecycle wrapper around the checkpoint store."""

    def __init__(self, store: CheckpointStore) -> None:
        self._store = store

    def open(self, job_id: str, model: str, task: str, strategy: str,
             input_sha256: str, total_windows: int,
             config: TokenFlowConfig) -> JobRecord:
        """Create the job if new; return the (possibly pre-existing) record."""
        self._store.create_job(JobRecord(
            job_id=job_id,
            status="pending",
            model=model,
            task=task,
            strategy=strategy,
            input_sha256=input_sha256,
            total_windows=total_windows,
            config_json=config.model_dump_json(),
        ))
        record = self._store.get_job(job_id)
        if record is None:
            raise JobStateError(f"Job {job_id} could not be created")
        return record

    def start(self, job_id: str) -> None:
        """Move a job into running state (valid from pending/paused/failed)."""
        record = self._require(job_id)
        if record.status == "completed":
            raise JobStateError(f"Job {job_id} is already completed")
        self._store.set_status(job_id, "running")

    def pause(self, job_id: str, reason: str = "awaiting_capacity") -> None:
        """Pause a running job (rate limit hit, token budget exhausted)."""
        self._store.set_status(job_id, "paused", error=reason)
        logger.warning("Job %s paused: %s", job_id, reason)

    def fail(self, job_id: str, error: str) -> None:
        """Mark a job failed; completed windows remain and resume works."""
        self._store.set_status(job_id, "failed", error=error[:2000])
        logger.error("Job %s failed: %s", job_id, error)

    def complete(self, job_id: str, result: str) -> None:
        """Persist final result and mark completed."""
        self._store.save_final(job_id, result)

    def status(self, job_id: str) -> Optional[str]:
        """Current status or None if unknown."""
        record = self._store.get_job(job_id)
        return record.status if record else None

    def resume_point(self, job_id: str) -> int:
        """First window index that still needs processing."""
        done = self._store.completed_indices(job_id)
        cursor = 0
        for index in done:
            if index == cursor:
                cursor += 1
            else:
                break
        return cursor

    def _require(self, job_id: str) -> JobRecord:
        record = self._store.get_job(job_id)
        if record is None:
            raise JobStateError(f"Unknown job {job_id}")
        return record
