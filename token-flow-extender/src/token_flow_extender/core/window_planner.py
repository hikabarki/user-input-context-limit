"""Structure-aware lossless window planning.

Splitting hierarchy: protected blocks (code fences, tables, JSON/base64
blobs) are never split; outside them, boundaries prefer headers, then
paragraph breaks, then sentence ends, then raw characters (last resort,
emitted as explicit CONTINUED parts).

Lossless invariant: the primary char spans of all windows tile the input
exactly — start at 0, no gaps, no overlaps of primary spans, end at
len(text). Overlap context is carried separately and is duplicate context
only, never a substitute for coverage. The plan is deterministic: the same
input and configuration always hash to the same plan_id.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..config import WindowBudget
from ..exceptions import WindowPlanError
from ..utils.logger import get_logger
from ..utils.validators import require_text
from .token_counter import TokenCounter

logger = get_logger(__name__)

_FENCE_RE = re.compile(r"```.*?(?:```|\Z)", re.DOTALL)
_TABLE_RE = re.compile(r"(?:^\|[^\n]*\n?){2,}", re.MULTILINE)
_BASE64_RE = re.compile(r"[A-Za-z0-9+/]{512,}={0,2}")
_HEADER_RE = re.compile(r"^#{1,6}\s", re.MULTILINE)
_PARA_RE = re.compile(r"\n\s*\n")
_SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+")


@dataclass(frozen=True)
class Window:
    """One processing window with full provenance metadata."""

    index: int
    total: int
    text: str
    char_start: int
    char_end: int
    token_count: int
    token_start: int
    token_end: int
    overlap_text: str = ""
    is_continuation: bool = False
    part: int = 1
    parts: int = 1
    doc_id: str = ""


@dataclass
class WindowPlan:
    """The complete deterministic plan for one input."""

    plan_id: str
    doc_id: str
    input_chars: int
    input_tokens: int
    windows: List[Window] = field(default_factory=list)

    @property
    def total_windows(self) -> int:
        return len(self.windows)


class WindowPlanner:
    """Builds lossless, structure-aware, deterministic window plans."""

    def __init__(self, counter: TokenCounter, budget: WindowBudget) -> None:
        self._counter = counter
        self._budget = budget

    def plan(self, text: str, doc_id: Optional[str] = None) -> WindowPlan:
        """Produce a WindowPlan whose primary spans tile text exactly."""
        text = require_text(text)
        if not text:
            raise WindowPlanError("Cannot plan windows for empty input")

        doc_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        doc_id = doc_id or doc_hash
        usable = self._budget.usable_window_tokens

        blocks = self._segment(text)
        raw_windows = self._pack(text, blocks, usable)
        windows = self._finalize(text, raw_windows, doc_id)

        plan_id = hashlib.sha256(json.dumps({
            "doc": doc_hash,
            "usable": usable,
            "overlap": self._budget.overlap_tokens,
            "spans": [(w.char_start, w.char_end) for w in windows],
        }, sort_keys=True).encode("utf-8")).hexdigest()[:16]

        plan = WindowPlan(
            plan_id=plan_id,
            doc_id=doc_id,
            input_chars=len(text),
            input_tokens=sum(w.token_count for w in windows),
            windows=windows,
        )
        self._assert_tiling(plan, len(text))
        logger.info(
            "Window plan %s: %d window(s), %d chars, ~%d tokens, usable=%d",
            plan_id, plan.total_windows, plan.input_chars, plan.input_tokens, usable,
        )
        return plan

    # -- segmentation -------------------------------------------------------

    def _segment(self, text: str) -> List[Tuple[int, int, bool]]:
        """Tile the text into (start, end, protected) blocks.

        Protected blocks (code fences, tables, base64 blobs) must never be
        split across windows unless a single block alone exceeds the window.
        The returned blocks tile the input exactly."""
        protected: List[Tuple[int, int]] = []
        for regex in (_FENCE_RE, _TABLE_RE, _BASE64_RE):
            for match in regex.finditer(text):
                protected.append((match.start(), match.end()))
        protected.sort()

        merged: List[Tuple[int, int]] = []
        for start, end in protected:
            if merged and start < merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        blocks: List[Tuple[int, int, bool]] = []
        cursor = 0
        for start, end in merged:
            if cursor < start:
                blocks.extend((s, e, False) for s, e in self._split_plain(text, cursor, start))
            blocks.append((start, end, True))
            cursor = end
        if cursor < len(text):
            blocks.extend((s, e, False) for s, e in self._split_plain(text, cursor, len(text)))
        return blocks

    @staticmethod
    def _split_plain(text: str, start: int, end: int) -> List[Tuple[int, int]]:
        """Split a plain-text region on headers, then paragraph breaks, then
        sentence ends. Resulting spans tile [start, end) exactly."""
        region = text[start:end]
        cuts = {0, len(region)}
        for match in _HEADER_RE.finditer(region):
            cuts.add(match.start())
        for match in _PARA_RE.finditer(region):
            cuts.add(match.end())
        for match in _SENTENCE_RE.finditer(region):
            cuts.add(match.end())
        ordered = sorted(cuts)
        return [(start + a, start + b) for a, b in zip(ordered, ordered[1:]) if b > a]

    # -- packing ------------------------------------------------------------

    def _pack(self, text: str, blocks: List[Tuple[int, int, bool]],
              usable: int) -> List[Tuple[int, int, bool, int, int]]:
        """Greedily pack blocks into windows of at most `usable` tokens.

        Returns (char_start, char_end, is_continuation, part, parts) tuples.
        A protected or atomic block larger than the window is hard-split
        with explicit CONTINUED part numbering."""
        out: List[Tuple[int, int, bool, int, int]] = []
        cur_start: Optional[int] = None
        cur_end = 0
        cur_tokens = 0

        def flush() -> None:
            nonlocal cur_start, cur_tokens
            if cur_start is not None and cur_end > cur_start:
                out.append((cur_start, cur_end, False, 1, 1))
            cur_start, cur_tokens = None, 0

        for b_start, b_end, _protected in blocks:
            b_tokens = self._counter.count(text[b_start:b_end])
            if b_tokens > usable:
                flush()
                out.extend(self._hard_split(text, b_start, b_end, usable))
                continue
            if cur_start is None:
                cur_start, cur_end, cur_tokens = b_start, b_end, b_tokens
            elif cur_tokens + b_tokens <= usable:
                cur_end, cur_tokens = b_end, cur_tokens + b_tokens
            else:
                flush()
                cur_start, cur_end, cur_tokens = b_start, b_end, b_tokens
        flush()

        if not out:
            raise WindowPlanError("Packing produced no windows")
        return out

    def _hard_split(self, text: str, start: int, end: int,
                    usable: int) -> List[Tuple[int, int, bool, int, int]]:
        """Last-resort split of one oversized atomic unit into CONTINUED
        parts, sized by real token counts via bisection on char length."""
        pieces: List[Tuple[int, int]] = []
        cursor = start
        while cursor < end:
            lo, hi = cursor + 1, end
            best = cursor + 1
            while lo <= hi:
                mid = (lo + hi) // 2
                if self._counter.count(text[cursor:mid]) <= usable:
                    best = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            pieces.append((cursor, best))
            cursor = best
        parts = len(pieces)
        return [(s, e, True, i + 1, parts) for i, (s, e) in enumerate(pieces)]

    # -- finalization ---------------------------------------------------------

    def _finalize(self, text: str,
                  raw: List[Tuple[int, int, bool, int, int]],
                  doc_id: str) -> List[Window]:
        """Attach token ranges and boundary overlap context to each window."""
        windows: List[Window] = []
        total = len(raw)
        token_cursor = 0
        overlap_budget = self._budget.overlap_tokens
        for index, (start, end, cont, part, parts) in enumerate(raw):
            body = text[start:end]
            tokens = self._counter.count(body)
            overlap_text = ""
            if index > 0 and overlap_budget > 0:
                prev_start, prev_end = raw[index - 1][0], raw[index - 1][1]
                overlap_text = self._tail_by_tokens(text[prev_start:prev_end], overlap_budget)
            windows.append(Window(
                index=index,
                total=total,
                text=body,
                char_start=start,
                char_end=end,
                token_count=tokens,
                token_start=token_cursor,
                token_end=token_cursor + tokens,
                overlap_text=overlap_text,
                is_continuation=cont,
                part=part,
                parts=parts,
                doc_id=doc_id,
            ))
            token_cursor += tokens
        return windows

    def _tail_by_tokens(self, text: str, budget: int) -> str:
        """Return the suffix of text that fits within `budget` tokens."""
        if self._counter.count(text) <= budget:
            return text
        lo, hi = 0, len(text)
        best = len(text)
        while lo <= hi:
            mid = (lo + hi) // 2
            if self._counter.count(text[mid:]) <= budget:
                best = mid
                hi = mid - 1
            else:
                lo = mid + 1
        return text[best:]

    @staticmethod
    def _assert_tiling(plan: WindowPlan, input_chars: int) -> None:
        """Assert the lossless invariant: primary spans tile [0, len)."""
        spans = sorted((w.char_start, w.char_end) for w in plan.windows)
        if not spans or spans[0][0] != 0 or spans[-1][1] != input_chars:
            raise WindowPlanError("Window spans do not cover the full input")
        for (_, prev_end), (next_start, _) in zip(spans, spans[1:]):
            if next_start != prev_end:
                raise WindowPlanError(
                    f"Gap or overlap in window spans at char {prev_end} -> {next_start}"
                )
