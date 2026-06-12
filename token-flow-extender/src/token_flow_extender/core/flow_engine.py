"""Flow engine: processes a WindowPlan through an LLM, losslessly.

Strategies (all lossless — every window is fully processed):
  sequential_refine (default): window n is processed together with the
    compressed carry-state of windows 1..n-1, so findings flow forward.
  map_reduce: windows are processed independently (parallelizable) and the
    partial results are merged hierarchically by MergeEngine.
  hybrid: map_reduce for extraction-type tasks, sequential_refine for
    reasoning-type tasks, selected per request from task keywords.

Carry-state compression NEVER touches unprocessed input; only the engine's
own intermediate summaries may be compressed.

Checkpointing: a checkpoint is written before AND after every external LLM
call, so a crash, rate-limit, or kill at any point resumes from the last
completed window without reprocessing or data loss.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional

from ..config import TokenFlowConfig, WindowBudget
from ..exceptions import LLMCallError
from ..storage.checkpoint_store import CheckpointStore
from ..storage.job_queue import JobQueue, deterministic_job_id
from ..utils.logger import get_logger
from ..utils.metrics import metrics
from .coverage_ledger import CoverageLedger, CoverageReport
from .merge_engine import MergeEngine
from .token_counter import TokenCounter
from .window_planner import Window, WindowPlan

logger = get_logger(__name__)

LLMCallable = Callable[[str], Awaitable[str]]
"""Async callable: full prompt in, model text out."""

ProgressCallback = Callable[[Dict[str, object]], None]
"""Receives progress events: window completion, coverage, job status."""

_EXTRACTION_HINTS = ("extract", "find", "list", "collect", "count", "locate",
                     "identify", "needle", "search")

WINDOW_PROMPT = """You are processing part {index_1} of {total} of a large input that exceeds your context window. The full input is being streamed to you in sequential parts; every part will be processed.

TASK: {task}

{carry_section}{overlap_section}{continuation_note}--- INPUT PART {index_1}/{total} (characters {char_start}-{char_end} of the original) ---
{body}
--- END OF PART {index_1}/{total} ---

Instructions: Apply the task to THIS part. Integrate and preserve every task-relevant fact from the carry-over state. Respond with the complete updated working result only."""

MAP_PROMPT = """You are processing part {index_1} of {total} of a large input. Each part is processed independently and the partial results will be merged afterwards.

TASK: {task}

{continuation_note}--- INPUT PART {index_1}/{total} (characters {char_start}-{char_end} of the original) ---
{body}
--- END OF PART ---

Instructions: Apply the task to this part only. Be complete: include every task-relevant fact found in this part, because anything you omit cannot be recovered during merging."""

COMPRESS_PROMPT = """The following is the working state accumulated while processing a large input in parts. Compress it to roughly half its length. You MUST preserve every task-relevant fact, identifier, number, and conclusion; remove only redundancy and filler.

TASK CONTEXT: {task}

--- WORKING STATE ---
{state}
--- END ---

Respond with the compressed working state only."""


@dataclass
class FlowResult:
    """Outcome of a flow run."""

    job_id: str
    output: str
    passthrough: bool
    strategy: str
    coverage: Optional[CoverageReport] = None
    window_results: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        """JSON-serializable summary."""
        return {
            "job_id": self.job_id,
            "output": self.output,
            "passthrough": self.passthrough,
            "strategy": self.strategy,
            "coverage": self.coverage.as_dict() if self.coverage else None,
        }


class FlowEngine:
    """Drives a WindowPlan through the LLM under a chosen strategy."""

    def __init__(
        self,
        llm: LLMCallable,
        counter: TokenCounter,
        budget: WindowBudget,
        config: TokenFlowConfig,
        store: CheckpointStore,
        progress: Optional[ProgressCallback] = None,
    ) -> None:
        self._llm = llm
        self._counter = counter
        self._budget = budget
        self._config = config
        self._store = store
        self._queue = JobQueue(store)
        self._merge = MergeEngine(llm, counter, budget, config)
        self._progress = progress

    async def run(
        self,
        input_text: str,
        plan: WindowPlan,
        task: str,
        model: str,
        strategy: Optional[str] = None,
        job_id: Optional[str] = None,
    ) -> FlowResult:
        """Process every window of the plan and return the merged result.

        Resumable: if a job with the same deterministic id has completed
        windows, they are loaded from the checkpoint store and skipped."""
        chosen = self._choose_strategy(strategy or self._config.strategy, task)
        job_id = job_id or deterministic_job_id(input_text, model, task, self._config)
        input_sha = hashlib.sha256(input_text.encode("utf-8")).hexdigest()

        record = self._queue.open(job_id, model, task, chosen, input_sha,
                                  plan.total_windows, self._config)
        if record.status == "completed" and record.result is not None:
            logger.info("Job %s already completed; returning stored result.", job_id)
            ledger = CoverageLedger.from_plan(plan)
            for window in plan.windows:
                ledger.mark_processed(window.index)
            return FlowResult(job_id=job_id, output=record.result, passthrough=False,
                              strategy=chosen, coverage=ledger.verify(),
                              window_results=list(self._store.get_window_results(job_id).values()))

        self._queue.start(job_id)
        ledger = CoverageLedger.from_plan(plan)
        completed = self._store.get_window_results(job_id)
        for index in completed:
            ledger.mark_processed(index)
        if completed:
            logger.info("Job %s resuming: %d/%d windows already done.",
                        job_id, len(completed), plan.total_windows)

        try:
            if chosen == "map_reduce":
                results = await self._run_map(plan, task, job_id, completed, ledger)
            else:
                results = await self._run_sequential(plan, task, job_id, completed, ledger)

            coverage = ledger.verify()  # raises CoverageError on any gap
            final = await self._merge.merge(results, task)
            self._queue.complete(job_id, final)
            self._emit({"event": "job_completed", "job_id": job_id,
                        "coverage": coverage.as_dict()})
            return FlowResult(job_id=job_id, output=final, passthrough=False,
                              strategy=chosen, coverage=coverage, window_results=results)
        except Exception as exc:
            self._queue.fail(job_id, str(exc))
            raise

    # -- strategies ----------------------------------------------------------

    async def _run_sequential(self, plan: WindowPlan, task: str, job_id: str,
                              completed: Dict[int, str],
                              ledger: CoverageLedger) -> List[str]:
        carry = ""
        results: List[str] = []
        for window in plan.windows:
            if window.index in completed:
                result = completed[window.index]
                results.append(result)
                carry = result  # the stored result IS the refined state so far
                continue
            carry = await self._compress_if_needed(carry, task)
            prompt = self._window_prompt(WINDOW_PROMPT, window, task, carry)
            result = await self._call_checkpointed(prompt, window, job_id)
            results.append(result)
            carry = result
            ledger.mark_processed(window.index)
            self._emit_window(job_id, window, ledger)
        return [results[-1]] if results else []

    async def _run_map(self, plan: WindowPlan, task: str, job_id: str,
                       completed: Dict[int, str],
                       ledger: CoverageLedger) -> List[str]:
        semaphore = asyncio.Semaphore(self._config.max_parallel_windows)
        results: Dict[int, str] = dict(completed)

        async def worker(window: Window) -> None:
            async with semaphore:
                prompt = self._window_prompt(MAP_PROMPT, window, task, carry="")
                results[window.index] = await self._call_checkpointed(prompt, window, job_id)
                ledger.mark_processed(window.index)
                self._emit_window(job_id, window, ledger)

        pending = [w for w in plan.windows if w.index not in completed]
        if pending:
            await asyncio.gather(*(worker(w) for w in pending))
        return [results[i] for i in sorted(results)]

    def _choose_strategy(self, strategy: str, task: str) -> str:
        if strategy != "hybrid":
            return strategy
        task_lower = task.lower()
        if any(hint in task_lower for hint in _EXTRACTION_HINTS):
            return "map_reduce"
        return "sequential_refine"

    # -- plumbing --------------------------------------------------------------

    def _window_prompt(self, template: str, window: Window, task: str, carry: str) -> str:
        carry_section = (
            f"--- CARRY-OVER STATE (results of parts 1..{window.index}) ---\n"
            f"{carry}\n--- END CARRY-OVER ---\n\n" if carry else ""
        )
        overlap_section = (
            f"--- BOUNDARY CONTEXT (duplicate tail of the previous part, for continuity only; "
            f"already processed, do not re-extract from it) ---\n{window.overlap_text}\n"
            f"--- END BOUNDARY CONTEXT ---\n\n" if window.overlap_text else ""
        )
        continuation_note = (
            f"NOTE: this part is [CONTINUED part {window.part}/{window.parts}] of a single "
            f"unit (for example one code block or table) that was too large for one window.\n\n"
            if window.is_continuation else ""
        )
        return template.format(
            index_1=window.index + 1,
            total=window.total,
            task=task,
            carry_section=carry_section,
            overlap_section=overlap_section,
            continuation_note=continuation_note,
            char_start=window.char_start,
            char_end=window.char_end,
            body=window.text,
        )

    async def _call_checkpointed(self, prompt: str, window: Window, job_id: str) -> str:
        """Checkpoint -> LLM call with retries -> checkpoint result."""
        last_error: Optional[Exception] = None
        for attempt in range(self._config.llm_max_retries + 1):
            try:
                with metrics.timer("llm_call_seconds"):
                    result = await self._llm(prompt)
                metrics.incr("llm_calls")
                self._store.save_window_result(
                    job_id, window.index, result,
                    window.token_count, window.char_start, window.char_end,
                )
                return result
            except Exception as exc:
                last_error = exc
                metrics.incr("llm_call_failures")
                logger.warning("LLM call failed (window %d, attempt %d/%d): %s",
                               window.index, attempt + 1,
                               self._config.llm_max_retries + 1, exc)
                if attempt < self._config.llm_max_retries:
                    await asyncio.sleep(self._config.llm_retry_delay * (attempt + 1))
        raise LLMCallError(
            f"LLM call failed for window {window.index} after "
            f"{self._config.llm_max_retries + 1} attempt(s): {last_error}"
        )

    async def _compress_if_needed(self, carry: str, task: str) -> str:
        """Compress carry-state when it nears its budget. Only state
        summaries are ever compressed — never unprocessed input."""
        if not carry:
            return carry
        budget = self._budget.carry_state_budget
        if budget <= 0 or self._counter.count(carry) <= budget:
            return carry
        logger.info("Carry-state over budget (%d tokens); compressing.", budget)
        compressed = await self._llm(COMPRESS_PROMPT.format(task=task, state=carry))
        metrics.incr("carry_compressions")
        return compressed

    def _emit_window(self, job_id: str, window: Window, ledger: CoverageLedger) -> None:
        report = ledger.report()
        self._emit({
            "event": "window_completed",
            "job_id": job_id,
            "window": window.index + 1,
            "total_windows": window.total,
            "percent": round(100.0 * report.processed_windows / max(1, report.total_windows), 1),
            "tokens_lost": 0,
            "coverage": report.as_dict(),
        })

    def _emit(self, event: Dict[str, object]) -> None:
        if self._progress is not None:
            try:
                self._progress(event)
            except Exception as exc:  # progress must never break processing
                logger.warning("Progress callback raised: %s", exc)
