"""Basic usage of the token-flow-extender SDK.

Run:  python examples/basic_usage.py

Uses a mock LLM so it runs anywhere with no GPU or network. Replace MockLLM
with OpenAICompatLLM(base_url=..., model=...) for a real vLLM endpoint.
"""

from __future__ import annotations

import asyncio

from token_flow_extender import TokenFlowConfig, TokenFlowExtender


class MockLLM:
    """Stand-in model: extracts NEEDLE markers it sees in each prompt."""

    async def __call__(self, prompt: str) -> str:
        import re
        needles = sorted(set(re.findall(r"NEEDLE-\d+", prompt)),
                         key=lambda s: int(s.split("-")[1]))
        return "Found markers: " + ", ".join(needles) if needles else "No markers."


class SimpleCounter:
    """Deterministic demo counter: roughly 1 token per 4 characters."""

    def count(self, text: str) -> int:
        return max(1, len(text) // 4) if text else 0


async def main() -> None:
    config = TokenFlowConfig(storage_backend="memory")
    tfe = TokenFlowExtender(config=config,
                            progress=lambda e: print("  progress:", e.get("event"),
                                                     e.get("window"), "/",
                                                     e.get("total_windows")))

    # 1) Small input: passes through untouched.
    small = "Summarize: the quick brown fox jumps over the lazy dog."
    result = await tfe.process(small, task="summarize", model="demo-model",
                               llm=MockLLM(), max_input_tokens=100000,
                               counter=SimpleCounter())
    print("passthrough:", result.passthrough)

    # 2) Big input: windowed losslessly.
    big = "\n\n".join(f"Paragraph {i}. " + "lorem ipsum " * 30 + f"NEEDLE-{i}."
                      for i in range(60))
    result = await tfe.process(big, task="find all NEEDLE markers",
                               model="demo-model", llm=MockLLM(),
                               max_input_tokens=400, counter=SimpleCounter(),
                               strategy="map_reduce")
    print("passthrough:", result.passthrough)
    print("coverage:", result.coverage.as_dict() if result.coverage else None)
    print("output:", result.output[:300])


if __name__ == "__main__":
    asyncio.run(main())
