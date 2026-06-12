"""Progress broadcasting for streaming/WebSocket clients.

ProgressBroker fans window-completion events out to any number of
subscribers per job. Events look like:

    {"event": "window_completed", "job_id": "...", "window": 17,
     "total_windows": 212, "percent": 8.0, "tokens_lost": 0, "coverage": {...}}
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator, Dict, List

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ProgressBroker:
    """In-process pub/sub for job progress events."""

    def __init__(self) -> None:
        self._subscribers: Dict[str, List[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()

    def callback(self, event: Dict[str, object]) -> None:
        """Synchronous callback suitable for FlowEngine(progress=...)."""
        job_id = str(event.get("job_id", ""))
        for queue in self._subscribers.get(job_id, []):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass  # slow consumer: drop rather than block processing

    async def subscribe(self, job_id: str) -> AsyncIterator[Dict[str, object]]:
        """Async iterator of progress events for one job."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=1024)
        async with self._lock:
            self._subscribers.setdefault(job_id, []).append(queue)
        try:
            while True:
                event = await queue.get()
                yield event
                if event.get("event") in ("job_completed", "job_failed"):
                    return
        finally:
            async with self._lock:
                subscribers = self._subscribers.get(job_id, [])
                if queue in subscribers:
                    subscribers.remove(queue)

    @staticmethod
    def format_human(event: Dict[str, object]) -> str:
        """Render an event as a human-readable progress line, for example:
        'window 17/212 complete (8.0%) - 0 tokens lost'."""
        if event.get("event") == "window_completed":
            return (f"window {event.get('window')}/{event.get('total_windows')} "
                    f"complete ({event.get('percent')}%) - "
                    f"{event.get('tokens_lost', 0)} tokens lost")
        return json.dumps(event, ensure_ascii=False)


broker = ProgressBroker()
"""Default module-level broker shared by routes.py."""
