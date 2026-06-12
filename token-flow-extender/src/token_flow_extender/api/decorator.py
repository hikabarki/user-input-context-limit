"""@token_flow decorator — Pattern 2.

    @token_flow(model="qwen", base_url="http://vllm:8000")
    async def my_llm_call(prompt: str) -> str: ...

The wrapped function IS the LLM transport. If the prompt fits the model's
resolved limit, the function is called exactly once with the original,
untouched prompt. If it does not fit, the prompt is windowed and the
function is called once per window, then results are merged.

Fail-open: if the extender itself errors for any reason, the original
function is called with the original prompt and a warning is logged.
"""

from __future__ import annotations

import functools
from typing import Any, Awaitable, Callable, Optional, TypeVar

from ..config import TokenFlowConfig
from ..utils.logger import get_logger
from .sdk import TokenFlowExtender

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Awaitable[str]])


def token_flow(
    model: str,
    task: str = "Process the input and respond as instructed.",
    base_url: Optional[str] = None,
    max_input_tokens: Optional[int] = None,
    config: Optional[TokenFlowConfig] = None,
    extender: Optional[TokenFlowExtender] = None,
) -> Callable[[F], F]:
    """Wrap an async LLM-calling function with transparent windowing."""

    def decorate(func: F) -> F:
        tfe = extender or TokenFlowExtender(config=config)

        @functools.wraps(func)
        async def wrapper(prompt: str, *args: Any, **kwargs: Any) -> str:
            async def transport(window_prompt: str) -> str:
                return await func(window_prompt, *args, **kwargs)

            try:
                result = await tfe.process(
                    prompt,
                    task=task,
                    model=model,
                    llm=transport,
                    base_url=base_url,
                    max_input_tokens=max_input_tokens,
                )
                return result.output
            except Exception as exc:
                # FAIL-OPEN: never block the host call path.
                logger.error(
                    "token_flow failed (%s); forwarding original prompt unchanged: %s",
                    type(exc).__name__, exc,
                )
                return await func(prompt, *args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorate
