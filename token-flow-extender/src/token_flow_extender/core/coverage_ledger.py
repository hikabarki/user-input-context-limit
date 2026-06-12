"""Coverage ledger: the 100-percent-processing proof.

The ledger records every window's exact char span and token count, plus its
processing status. verify() asserts that the union of PROCESSED spans equals
the full input span — the machine-checkable form of the Golden Rule that no
token may be dropped, skipped, or silently truncated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from ..exceptions import CoverageError
from .window_planner import WindowPlan


@dataclass
class LedgerEntry:
    """One window's accounting record."""

    window_index: int
    char_start: int
    char_end: int
    token_count: int
    status: str = "pending"  # pending | processed


@dataclass
class CoverageReport:
    """Summary returned to callers and surfaced in API responses."""

    input_chars: int
    input_tokens: int
    processed_chars: int
    processed_tokens: int
    total_windows: int
    processed_windows: int
    coverage_ratio: float
    complete: bool

    def as_dict(self) -> Dict[str, object]:
        """Serializable form for JSON responses and progress events."""
        return {
            "input_chars": self.input_chars,
            "input_tokens": self.input_tokens,
            "processed_chars": self.processed_chars,
            "processed_tokens": self.processed_tokens,
            "total_windows": self.total_windows,
            "processed_windows": self.processed_windows,
            "coverage_ratio": round(self.coverage_ratio, 6),
            "tokens_lost": self.input_tokens - self.processed_tokens,
            "complete": self.complete,
        }


@dataclass
class CoverageLedger:
    """Tracks per-window processing against the plan's full span."""

    input_chars: int
    input_tokens: int
    entries: List[LedgerEntry] = field(default_factory=list)

    @classmethod
    def from_plan(cls, plan: WindowPlan) -> "CoverageLedger":
        """Initialize a ledger with one pending entry per planned window."""
        ledger = cls(input_chars=plan.input_chars, input_tokens=plan.input_tokens)
        for window in plan.windows:
            ledger.entries.append(LedgerEntry(
                window_index=window.index,
                char_start=window.char_start,
                char_end=window.char_end,
                token_count=window.token_count,
            ))
        return ledger

    def mark_processed(self, window_index: int) -> None:
        """Record that a window completed processing."""
        for entry in self.entries:
            if entry.window_index == window_index:
                entry.status = "processed"
                return
        raise CoverageError(f"Unknown window index {window_index} in ledger")

    def report(self) -> CoverageReport:
        """Compute current coverage accounting."""
        processed = [e for e in self.entries if e.status == "processed"]
        processed_chars = sum(e.char_end - e.char_start for e in processed)
        processed_tokens = sum(e.token_count for e in processed)
        ratio = processed_tokens / self.input_tokens if self.input_tokens else 0.0
        return CoverageReport(
            input_chars=self.input_chars,
            input_tokens=self.input_tokens,
            processed_chars=processed_chars,
            processed_tokens=processed_tokens,
            total_windows=len(self.entries),
            processed_windows=len(processed),
            coverage_ratio=ratio,
            complete=self._is_complete(processed),
        )

    def verify(self) -> CoverageReport:
        """Assert 100 percent coverage; raise CoverageError otherwise.

        Checks both accounting identities:
          processed_tokens == input_tokens, and
          processed char spans tile [0, input_chars) with no gap."""
        report = self.report()
        if not report.complete:
            raise CoverageError(
                f"Coverage incomplete: {report.processed_windows}/{report.total_windows} "
                f"windows, {report.processed_tokens}/{report.input_tokens} tokens "
                f"({report.coverage_ratio:.2%})."
            )
        if report.processed_tokens != report.input_tokens:
            raise CoverageError(
                f"Token accounting mismatch: processed {report.processed_tokens} "
                f"!= input {report.input_tokens}."
            )
        return report

    def _is_complete(self, processed: List[LedgerEntry]) -> bool:
        spans = sorted((e.char_start, e.char_end) for e in processed)
        if not spans or spans[0][0] != 0:
            return False
        cursor = 0
        for start, end in spans:
            if start != cursor:
                return False
            cursor = end
        return cursor == self.input_chars
