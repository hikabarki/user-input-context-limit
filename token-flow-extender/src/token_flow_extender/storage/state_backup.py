"""Export/import of full job state as portable JSON files.

Lets operators move an in-flight job between hosts or back up progress
before maintenance. JSON only; no pickle anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..exceptions import CheckpointError
from .checkpoint_store import CheckpointStore, JobRecord

BACKUP_VERSION = 1


def export_job(store: CheckpointStore, job_id: str, path: str) -> str:
    """Write a job's full state (record + window results) to a JSON file."""
    record = store.get_job(job_id)
    if record is None:
        raise CheckpointError(f"Cannot export unknown job {job_id}")
    payload = {
        "version": BACKUP_VERSION,
        "job": record.__dict__,
        "windows": {str(k): v for k, v in store.get_window_results(job_id).items()},
    }
    target = Path(path)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(target.resolve())


def import_job(store: CheckpointStore, path: str) -> Optional[str]:
    """Restore a job exported by export_job. Returns the job_id."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("version") != BACKUP_VERSION:
        raise CheckpointError(f"Unsupported backup version: {raw.get('version')!r}")
    record = JobRecord(**raw["job"])
    store.create_job(record)
    store.set_status(record.job_id, record.status, record.error)
    for index, result in (raw.get("windows") or {}).items():
        store.save_window_result(record.job_id, int(index), result, 0, 0, 0)
    if record.result:
        store.save_final(record.job_id, record.result)
    return record.job_id
