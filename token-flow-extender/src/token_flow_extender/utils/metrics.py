"""Lightweight in-process metrics: counters and timers.

No external dependencies; integrators can scrape Metrics.snapshot() or wire
the values into Prometheus/OpenTelemetry themselves.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Dict, Iterator


class Metrics:
    """Thread-safe counters and cumulative timers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Dict[str, int] = {}
        self._timers: Dict[str, float] = {}

    def incr(self, name: str, value: int = 1) -> None:
        """Increment a named counter."""
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + value

    @contextmanager
    def timer(self, name: str) -> Iterator[None]:
        """Accumulate wall-clock seconds under a named timer."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            with self._lock:
                self._timers[name] = self._timers.get(name, 0.0) + elapsed

    def snapshot(self) -> Dict[str, object]:
        """Return a copy of all metrics."""
        with self._lock:
            return {"counters": dict(self._counters), "timers_seconds": dict(self._timers)}


metrics = Metrics()
"""Module-level default metrics instance."""
