"""Million-token lossless demo — acceptance criterion 1 at full scale.

Run:  python examples/million_token_demo.py

Builds a ~1.2M-token document containing 1,000 numbered NEEDLE markers
spread evenly, processes it through a mock model with a small context
window, and proves that ALL 1,000 needles were seen and the coverage
ledger accounts for 100 percent of input tokens.

Uses a mock LLM so the demo runs in seconds with no GPU. Swap in
OpenAICompatLLM for a real endpoint; the flow is identical.
"""

from __future__ import annotations

import asyncio
import re
import time

from token_flow_extender import TokenFlowConfig, TokenFlowExtender


NEEDLES = 1000
TARGET_TOKENS = 1_200_000
MOCK_MODEL_MAX_INPUT = 8000  # small window -> hundreds of windows


class SimpleCounter:
    """Deterministic counter: 1 token per 4 characters."""

    def count(self, text: str) -> int:
        return max(1, len(text) // 4) if text else 0


class NeedleCollectorLLM:
    """Mock model that reports every NEEDLE marker present in its prompt."""

    def __init__(self) -> None:
        self.calls = 0
        self.seen: set[str] = set()

    async def __call__(self, prompt: str) -> str:
        self.calls += 1
        found = set(re.findall(r"NEEDLE-\d{4}", prompt))
        self.seen |= found
        return "MARKERS: " + " ".join(sorted(found)) if found else "MARKERS: none"


def build_document() -> str:
    """~1.2M tokens (~4.8M chars) with 1,000 evenly spread needles."""
    chars_per_section = (TARGET_TOKENS * 4) // NEEDLES
    filler_unit = "The platform processes documents of arbitrary size. "
    repeats = max(1, chars_per_section // len(filler_unit))
    sections = []
    for i in range(NEEDLES):
        sections.append(
            f"Section {i:04d}.\n" + filler_unit * repeats + f"NEEDLE-{i:04d}.\n"
        )
    return "\n\n".join(sections)


async def main() -> None:
    print("Building ~1.2M-token document with 1,000 needles...")
    document = build_document()
    counter = SimpleCounter()
    print(f"  chars={len(document):,}  est_tokens={counter.count(document):,}")

    llm = NeedleCollectorLLM()
    tfe = TokenFlowExtender(config=TokenFlowConfig(storage_backend="memory"))

    done_windows = {"n": 0}

    def progress(event: dict) -> None:
        if event.get("event") == "window_completed":
            done_windows["n"] += 1
            if done_windows["n"] % 50 == 0:
                print(f"  window {event['window']}/{event['total_windows']} "
                      f"complete ({event['percent']}%) - "
                      f"{event['tokens_lost']} tokens lost")

    tfe._progress = progress  # demo-only direct wiring

    start = time.perf_counter()
    result = await tfe.process(
        document,
        task="Find and list every NEEDLE-#### marker.",
        model="mock-2m-model",
        llm=llm,
        max_input_tokens=MOCK_MODEL_MAX_INPUT,
        counter=counter,
        strategy="map_reduce",
    )
    elapsed = time.perf_counter() - start

    coverage = result.coverage.as_dict() if result.coverage else {}
    missing = {f"NEEDLE-{i:04d}" for i in range(NEEDLES)} - llm.seen

    print()
    print(f"windows processed : {coverage.get('total_windows')}")
    print(f"llm calls         : {llm.calls}")
    print(f"coverage complete : {coverage.get('complete')}")
    print(f"tokens lost       : {coverage.get('tokens_lost')}")
    print(f"needles seen      : {len(llm.seen)}/{NEEDLES}")
    print(f"needles missing   : {len(missing)}  {sorted(missing)[:5] if missing else ''}")
    print(f"elapsed           : {elapsed:.1f}s")

    assert coverage.get("complete") is True
    assert coverage.get("tokens_lost") == 0
    assert not missing, "LOSSLESS GUARANTEE VIOLATED"
    print("\nPASS: all 1,000 needles accounted for; 100 percent token coverage.")


if __name__ == "__main__":
    asyncio.run(main())
