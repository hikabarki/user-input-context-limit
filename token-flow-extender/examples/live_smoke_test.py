"""Live vLLM smoke test.

Run:  python examples/live_smoke_test.py

Tests passthrough (small input) and windowed flow (oversized input) against
a real vLLM endpoint. Expects the model to be reachable at BASE_URL.
"""

from __future__ import annotations

import asyncio

from token_flow_extender import TokenFlowConfig, TokenFlowExtender
from token_flow_extender.api.sdk import OpenAICompatLLM

BASE_URL = "http://110.8.41.182:11434"
MODEL = "google/gemma-4-26B-A4B-it"


async def main() -> None:
    llm = OpenAICompatLLM(BASE_URL, MODEL, max_tokens=512)
    tfe = TokenFlowExtender(
        config=TokenFlowConfig(storage_backend="memory"),
        progress=lambda e: print(f"  [{e.get('event')}] window {e.get('window')}/{e.get('total_windows')}"),
    )

    # 1) Passthrough: small input, should hit the model once untouched
    print("\n--- TEST 1: passthrough (small input) ---")
    result = await tfe.process(
        "In one sentence, what is the capital of South Korea?",
        task="answer",
        model=MODEL,
        llm=llm,
        base_url=BASE_URL,
    )
    print(f"passthrough : {result.passthrough}")
    print(f"output      : {result.output[:200]}")
    assert result.passthrough, "Expected passthrough for small input"

    # 2) Windowed: input larger than a small forced limit to trigger windowing
    print("\n--- TEST 2: windowed flow (oversized input) ---")
    big = "\n\n".join(
        f"Paragraph {i}. " + "This is filler text for the smoke test. " * 20 + f" NEEDLE-{i}."
        for i in range(30)
    )
    result = await tfe.process(
        big,
        task="List every NEEDLE marker you find (e.g. NEEDLE-0, NEEDLE-1, ...).",
        model=MODEL,
        llm=llm,
        max_input_tokens=800,   # artificially low to force windowing
        max_output_tokens=256,
        strategy="map_reduce",
    )
    print(f"passthrough : {result.passthrough}")
    cov = result.coverage
    if cov:
        d = cov.as_dict()
        print(f"coverage    : {d['coverage_ratio']:.0%}  windows={d['total_windows']}  tokens_lost={d['tokens_lost']}")
        assert d["complete"], "Coverage ledger reports incomplete processing"
        assert d["tokens_lost"] == 0, f"Tokens lost: {d['tokens_lost']}"
    print(f"output      : {result.output[:300]}")

    print("\nPASS: live smoke test complete.")


if __name__ == "__main__":
    asyncio.run(main())
