"""Shared test fixtures: deterministic counter, mock LLM, helpers."""

from __future__ import annotations

import asyncio
from typing import List

import pytest

from token_flow_extender.config import ResolvedLimits, TokenFlowConfig, WindowBudget
from token_flow_extender.storage.checkpoint_store import MemoryCheckpointStore


class CharTokenCounter:
    """Deterministic counter for tests: 1 token per 4 characters (min 1)."""

    def count(self, text: str) -> int:
        if not text:
            return 0
        return max(1, len(text) // 4)


class MockLLM:
    """Deterministic mock LLM that echoes needle markers found in prompts.

    It scans the prompt for NEEDLE-<n> markers and accumulates them, which
    lets tests prove that every part of the input reached the model."""

    def __init__(self, fail_on_calls: List[int] | None = None) -> None:
        self.calls = 0
        self.prompts: List[str] = []
        self._fail_on = set(fail_on_calls or [])

    async def __call__(self, prompt: str) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        if self.calls in self._fail_on:
            raise RuntimeError(f"Injected failure on call {self.calls}")
        import re
        needles = sorted(set(re.findall(r"NEEDLE-\d+", prompt)),
                         key=lambda s: int(s.split("-")[1]))
        return "FOUND: " + " ".join(needles) if needles else f"RESULT({self.calls})"


@pytest.fixture
def counter() -> CharTokenCounter:
    return CharTokenCounter()


@pytest.fixture
def config(tmp_path) -> TokenFlowConfig:
    return TokenFlowConfig(
        storage_backend="memory",
        registry_path=str(tmp_path / "missing_registry.yaml"),
        llm_retry_delay=0.0,
    )


@pytest.fixture
def store() -> MemoryCheckpointStore:
    return MemoryCheckpointStore()


def make_budget(max_input: int, config: TokenFlowConfig,
                max_output: int = 64) -> WindowBudget:
    limits = ResolvedLimits(model="mock", max_input_tokens=max_input,
                            max_output_tokens=max_output, source="explicit")
    return WindowBudget.derive(limits, config)


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)
