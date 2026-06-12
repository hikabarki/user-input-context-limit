"""Hierarchical recursive merge of partial window results.

If the concatenated partial results exceed the usable window, they are
merged in batches that each fit, producing a smaller layer of results;
the process recurses to unlimited depth until a single result remains.
Only intermediate RESULTS are merged/condensed here; raw input is never
touched by this engine.
"""

from __future__ import annotations

from typing import Awaitable, Callable, List

from ..config import TokenFlowConfig, WindowBudget
from ..utils.logger import get_logger
from ..utils.metrics import metrics
from .token_counter import TokenCounter

logger = get_logger(__name__)

MERGE_PROMPT = """You are merging partial results produced from sequential parts of one large input.

TASK: {task}

Below are {count} partial results, in original input order. Merge them into ONE coherent result for the task. Preserve every task-relevant fact, identifier, number, and conclusion from every partial result; deduplicate only exact redundancy.

{sections}

Respond with the single merged result only."""


class MergeEngine:
    """Recursive hierarchical reducer."""

    def __init__(self, llm: Callable[[str], Awaitable[str]],
                 counter: TokenCounter, budget: WindowBudget,
                 config: TokenFlowConfig) -> None:
        self._llm = llm
        self._counter = counter
        self._budget = budget
        self._config = config

    async def merge(self, results: List[str], task: str, depth: int = 0) -> str:
        """Reduce partial results to a single result, recursing as needed."""
        results = [r for r in results if r]
        if not results:
            return ""
        if len(results) == 1:
            return results[0]
        if depth > 50:  # safety against pathological non-convergence
            logger.error("Merge depth exceeded 50; concatenating remaining results.")
            return "\n\n".join(results)

        batches = self._batch(results)
        logger.info("Merge depth %d: %d result(s) -> %d batch(es).",
                    depth, len(results), len(batches))
        merged_layer: List[str] = []
        for batch in batches:
            if len(batch) == 1:
                merged_layer.append(batch[0])
                continue
            sections = "\n\n".join(
                f"--- PARTIAL RESULT {i + 1}/{len(batch)} ---\n{part}"
                for i, part in enumerate(batch)
            )
            prompt = MERGE_PROMPT.format(task=task, count=len(batch), sections=sections)
            merged_layer.append(await self._llm(prompt))
            metrics.incr("merge_calls")
        return await self.merge(merged_layer, task, depth + 1)

    def _batch(self, results: List[str]) -> List[List[str]]:
        """Group results so each batch's token total fits the usable window."""
        budget = max(1, self._budget.usable_window_tokens)
        batches: List[List[str]] = []
        current: List[str] = []
        current_tokens = 0
        for result in results:
            tokens = self._counter.count(result)
            if current and current_tokens + tokens > budget:
                batches.append(current)
                current, current_tokens = [], 0
            current.append(result)
            current_tokens += tokens
        if current:
            batches.append(current)
        # Guarantee convergence: if every result landed in its own batch and
        # nothing can pair up, force pairs so recursion strictly shrinks.
        if len(batches) == len(results) and len(results) > 1:
            batches = [results[i:i + 2] for i in range(0, len(results), 2)]
        return batches
