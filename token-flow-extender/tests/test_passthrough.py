"""Tests for pass-through purity and the lossless needle guarantee."""

from __future__ import annotations

import time

from token_flow_extender.api.sdk import TokenFlowExtender
from token_flow_extender.config import TokenFlowConfig

from conftest import CharTokenCounter, MockLLM


def _tfe(tmp_path) -> TokenFlowExtender:
    config = TokenFlowConfig(storage_backend="memory",
                             registry_path=str(tmp_path / "none.yaml"),
                             llm_retry_delay=0.0)
    return TokenFlowExtender(config=config)


async def test_passthrough_untouched(tmp_path):
    """Under-limit input must reach the LLM byte-identical, in one call."""
    tfe = _tfe(tmp_path)
    llm = MockLLM()
    original = "Short prompt with special chars: 한국어, ```code```, | table |."
    result = await tfe.process(original, task="t", model="mock", llm=llm,
                               max_input_tokens=100000,
                               counter=CharTokenCounter())
    assert result.passthrough is True
    assert llm.calls == 1
    assert llm.prompts[0] == original  # byte-identical, zero added tokens


async def test_passthrough_overhead_under_5ms(tmp_path):
    """Acceptance criterion 3: pass-through decision overhead < 5 ms."""
    tfe = _tfe(tmp_path)
    llm = MockLLM()
    counter = CharTokenCounter()
    prompt = "hello " * 200
    # Warm-up (limit resolution path)
    await tfe.process(prompt, task="t", model="mock", llm=llm,
                      max_input_tokens=100000, counter=counter)
    start = time.perf_counter()
    runs = 20
    for _ in range(runs):
        await tfe.process(prompt, task="t", model="mock", llm=llm,
                          max_input_tokens=100000, counter=counter)
    per_call_ms = (time.perf_counter() - start) * 1000 / runs
    assert per_call_ms < 5.0, f"pass-through overhead {per_call_ms:.2f} ms"


async def test_oversized_input_processes_all_needles(tmp_path):
    """Scaled-down acceptance criterion 1: every needle accounted for."""
    tfe = _tfe(tmp_path)
    llm = MockLLM()
    needles = 100
    text = "\n\n".join(
        f"Paragraph {i}. " + "filler content " * 25 + f"NEEDLE-{i}."
        for i in range(needles)
    )
    result = await tfe.process(text, task="find all NEEDLE markers",
                               model="mock", llm=llm,
                               max_input_tokens=400,
                               counter=CharTokenCounter(),
                               strategy="map_reduce")
    assert result.passthrough is False
    assert result.coverage is not None and result.coverage.complete

    # Every needle must have reached the model in some window prompt.
    seen = set()
    for prompt in llm.prompts:
        import re
        seen.update(re.findall(r"NEEDLE-\d+", prompt))
    assert {f"NEEDLE-{i}" for i in range(needles)} <= seen


async def test_sequential_refine_carries_state(tmp_path):
    tfe = _tfe(tmp_path)
    llm = MockLLM()
    text = "\n\n".join(f"Part {i}. " + "filler " * 30 + f"NEEDLE-{i}."
                       for i in range(20))
    result = await tfe.process(text, task="find needles", model="mock", llm=llm,
                               max_input_tokens=400, counter=CharTokenCounter(),
                               strategy="sequential_refine")
    assert result.coverage is not None and result.coverage.complete
    # Later prompts must include carry-over state sections.
    carry_prompts = [p for p in llm.prompts if "CARRY-OVER STATE" in p]
    assert carry_prompts


async def test_hybrid_selects_map_reduce_for_extraction(tmp_path):
    tfe = _tfe(tmp_path)
    llm = MockLLM()
    text = "\n\n".join("filler " * 40 for _ in range(20))
    result = await tfe.process(text, task="extract all dates", model="mock",
                               llm=llm, max_input_tokens=400,
                               counter=CharTokenCounter(), strategy="hybrid")
    assert result.strategy == "map_reduce"


async def test_hybrid_selects_refine_for_reasoning(tmp_path):
    tfe = _tfe(tmp_path)
    llm = MockLLM()
    text = "\n\n".join("filler " * 40 for _ in range(20))
    result = await tfe.process(text, task="explain the overall argument",
                               model="mock", llm=llm, max_input_tokens=400,
                               counter=CharTokenCounter(), strategy="hybrid")
    assert result.strategy == "sequential_refine"
