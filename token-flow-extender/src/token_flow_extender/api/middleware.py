"""TokenFlowMiddleware — Pattern 1 (pure ASGI, framework-agnostic).

    app.add_middleware(TokenFlowMiddleware, settings=TokenFlowMiddlewareSettings(
        watch_paths=("/api/chat", "/api/message"),
        model_field="model", text_fields=("message", "prompt"),
        llm_factory=my_llm_factory,           # model name -> async llm
        base_url_factory=my_base_url_factory, # model name -> vLLM base url
    ))

Behavior:
  - Requests outside watch_paths, non-POST, or non-JSON: forwarded
    byte-identically (the middleware never consumes them destructively).
  - Watched JSON requests whose text fits the model's limit: forwarded with
    the ORIGINAL body bytes, untouched.
  - Oversized text: processed through the windowed flow; the middleware
    responds itself with the merged result and a coverage report.
  - ANY middleware/extender error: FAIL-OPEN — original request forwarded
    unchanged and a warning logged. The host app keeps working identically
    whether this middleware is installed or not.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional, Sequence, Tuple

from ..config import TokenFlowConfig
from ..utils.logger import get_logger
from .sdk import TokenFlowExtender

logger = get_logger(__name__)

LLMFactory = Callable[[str], Callable[[str], Awaitable[str]]]
BaseURLFactory = Callable[[str], Optional[str]]


@dataclass
class TokenFlowMiddlewareSettings:
    """Integration-time settings for the middleware."""

    watch_paths: Sequence[str] = ("/api/chat", "/api/message")
    model_field: str = "model"
    text_fields: Tuple[str, ...] = ("message", "prompt", "input")
    task_field: str = "task"
    default_task: str = "Answer the user's request using the full input."
    default_model: str = ""
    llm_factory: Optional[LLMFactory] = None
    base_url_factory: Optional[BaseURLFactory] = None
    config: TokenFlowConfig = field(default_factory=TokenFlowConfig)


class TokenFlowMiddleware:
    """Pure ASGI middleware. No host-framework imports required."""

    def __init__(self, app: Callable, settings: Optional[TokenFlowMiddlewareSettings] = None,
                 **kwargs: object) -> None:
        self.app = app
        self.settings = settings or TokenFlowMiddlewareSettings(**kwargs)  # type: ignore[arg-type]
        self._tfe = TokenFlowExtender(config=self.settings.config)

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope.get("type") != "http" or scope.get("method") != "POST" \
                or scope.get("path") not in self.settings.watch_paths:
            await self.app(scope, receive, send)
            return

        # Buffer the body so we can either act on it or replay it untouched.
        body = b""
        more = True
        while more:
            message = await receive()
            body += message.get("body", b"")
            more = message.get("more_body", False)

        async def replay() -> dict:
            return {"type": "http.request", "body": body, "more_body": False}

        try:
            handled = await self._maybe_handle(scope, body, send)
            if handled:
                return
        except Exception as exc:
            # FAIL-OPEN: any extender problem forwards the original request.
            logger.error("TokenFlowMiddleware fail-open (%s): %s",
                         type(exc).__name__, exc)

        await self.app(scope, replay, send)

    async def _maybe_handle(self, scope: dict, body: bytes, send: Callable) -> bool:
        """Return True if the middleware produced the response itself."""
        settings = self.settings
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            return False  # not JSON: pass through untouched
        if not isinstance(payload, dict):
            return False

        text_field = next((f for f in settings.text_fields
                           if isinstance(payload.get(f), str)), None)
        if text_field is None:
            return False
        text: str = payload[text_field]
        model = str(payload.get(settings.model_field) or settings.default_model)
        if not model:
            return False

        base_url = settings.base_url_factory(model) if settings.base_url_factory else None
        limits = self._tfe.resolve_limits(model, base_url=base_url)
        # Cheap pre-check; the SDK re-checks exactly with the model tokenizer.
        from ..core.token_counter import HeuristicTokenCounter
        estimate = HeuristicTokenCounter().count(text)
        if estimate <= int(limits.max_input_tokens * settings.config.safety_margin):
            return False  # fits: forward ORIGINAL bytes untouched

        if settings.llm_factory is None and base_url is None:
            logger.warning(
                "Oversized input (%d est. tokens > %s limit %d) but no llm_factory/"
                "base_url_factory configured; passing through unchanged.",
                estimate, model, limits.max_input_tokens,
            )
            return False

        llm = settings.llm_factory(model) if settings.llm_factory else None
        task = str(payload.get(settings.task_field) or settings.default_task)
        result = await self._tfe.process(
            text, task=task, model=model, llm=llm, base_url=base_url,
        )
        response = json.dumps({
            "object": "token_flow.result",
            "model": model,
            "output": result.output,
            "job_id": result.job_id,
            "strategy": result.strategy,
            "passthrough": result.passthrough,
            "coverage": result.coverage.as_dict() if result.coverage else None,
        }, ensure_ascii=False).encode("utf-8")

        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"application/json"),
                        (b"x-token-flow-extender", b"windowed")],
        })
        await send({"type": "http.response.body", "body": response})
        return True
