"""Direct SDK — Pattern 3.

    tfe = TokenFlowExtender.auto()
    result = await tfe.process(huge_input, task="summarize", model="gemma",
                               llm=my_async_llm, base_url="http://vllm:8000")

Guarantees:
  - Inputs that fit the model's resolved limit pass through 100 percent
    untouched (single direct LLM call, zero added tokens).
  - Oversized inputs are windowed, processed losslessly, merged, and
    verified by the coverage ledger.
  - Idempotent: same input + same config resumes rather than restarts.
"""

from __future__ import annotations

from typing import Dict, Optional

import httpx

from ..config import ResolvedLimits, TokenFlowConfig, WindowBudget
from ..core.flow_engine import FlowEngine, FlowResult, LLMCallable, ProgressCallback
from ..core.limit_resolver import LimitResolver
from ..core.token_counter import TokenCounter, get_counter
from ..core.window_planner import WindowPlanner
from ..storage.checkpoint_store import CheckpointStore, build_store
from ..utils.logger import get_logger
from ..utils.metrics import metrics

logger = get_logger(__name__)


class OpenAICompatLLM:
    """Minimal async client for any OpenAI-compatible /v1/chat/completions
    endpoint (vLLM, OpenAI, and others). Provided for convenience; any
    async callable str -> str works as an LLM for the SDK."""

    def __init__(self, base_url: str, model: str,
                 api_key: str = "EMPTY", timeout: float = 600.0,
                 max_tokens: Optional[int] = None,
                 system_prompt: Optional[str] = None) -> None:
        self._url = f"{base_url.rstrip('/')}/v1/chat/completions"
        self._model = model
        self._api_key = api_key
        self._timeout = timeout
        self._max_tokens = max_tokens
        self._system_prompt = system_prompt

    async def __call__(self, prompt: str) -> str:
        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload: Dict[str, object] = {"model": self._model, "messages": messages}
        if self._max_tokens is not None:
            payload["max_tokens"] = self._max_tokens
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                self._url, json=payload,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            response.raise_for_status()
            data = response.json()
        return data["choices"][0]["message"]["content"]


class TokenFlowExtender:
    """Main SDK entry point."""

    def __init__(self, config: Optional[TokenFlowConfig] = None,
                 store: Optional[CheckpointStore] = None,
                 progress: Optional[ProgressCallback] = None) -> None:
        self.config = config or TokenFlowConfig()
        self._store = store or build_store(
            self.config.storage_backend, self.config.db_path, self.config.redis_url
        )
        self._progress = progress

    @classmethod
    def auto(cls, **overrides: object) -> "TokenFlowExtender":
        """Build from environment variables plus keyword overrides."""
        return cls(config=TokenFlowConfig.auto(**overrides))

    async def process(
        self,
        input_text: str,
        task: str,
        model: str,
        llm: Optional[LLMCallable] = None,
        base_url: Optional[str] = None,
        max_input_tokens: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        counter: Optional[TokenCounter] = None,
        strategy: Optional[str] = None,
        system_prompt: str = "",
        doc_id: Optional[str] = None,
    ) -> FlowResult:
        """Process input of ANY size against the given model.

        llm: async callable(prompt) -> response text. If omitted, base_url
        must be given and an OpenAI-compatible client is built automatically.
        """
        if llm is None:
            if base_url is None:
                raise ValueError("Provide either llm= or base_url=")
            llm = OpenAICompatLLM(base_url, model, timeout=self.config.request_timeout)

        resolver = LimitResolver(self.config, base_url=base_url)
        limits = resolver.resolve(model, max_input_tokens, max_output_tokens)
        active_counter = counter or get_counter(model, self.config, base_url=base_url)

        system_tokens = active_counter.count(system_prompt) if system_prompt else 0
        budget = WindowBudget.derive(limits, self.config,
                                     system_prompt_tokens=system_tokens)

        input_tokens = active_counter.count(input_text)
        threshold = int(limits.max_input_tokens * self.config.safety_margin) \
            - budget.reserved_output_tokens - system_tokens

        if input_tokens <= max(1, threshold):
            # PASS-THROUGH: untouched, zero added tokens, single direct call.
            metrics.incr("passthrough")
            logger.info("Pass-through: %d tokens fit %s (limit %d, source %s).",
                        input_tokens, model, limits.max_input_tokens, limits.source)
            output = await llm(input_text)
            return FlowResult(job_id="passthrough", output=output,
                              passthrough=True, strategy="passthrough")

        metrics.incr("windowed_jobs")
        logger.info("Windowed flow: %d tokens > threshold %d for %s "
                    "(limit %d, source %s).",
                    input_tokens, threshold, model,
                    limits.max_input_tokens, limits.source)
        planner = WindowPlanner(active_counter, budget)
        plan = planner.plan(input_text, doc_id=doc_id)
        engine = FlowEngine(llm, active_counter, budget, self.config,
                            self._store, progress=self._progress)
        return await engine.run(input_text, plan, task, model, strategy=strategy)

    def resolve_limits(self, model: str, base_url: Optional[str] = None,
                       max_input_tokens: Optional[int] = None) -> ResolvedLimits:
        """Expose the limit chain for diagnostics and host integration."""
        return LimitResolver(self.config, base_url=base_url).resolve(
            model, max_input_tokens=max_input_tokens
        )

    @property
    def store(self) -> CheckpointStore:
        """The active checkpoint store (for job inspection/backup)."""
        return self._store
