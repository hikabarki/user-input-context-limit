"""FastAPI integration example — the one-line attach pattern.

Run:  uvicorn examples.fastapi_integration:app --port 8800

This simulates a host backend (like the Keural FastAPI service). The host's
own endpoint code is completely unaware of the middleware. Small requests hit
the host endpoint untouched; oversized requests are windowed by the module.
"""

from __future__ import annotations

from fastapi import FastAPI, Request

from token_flow_extender import TokenFlowConfig
from token_flow_extender.api.middleware import (
    TokenFlowMiddleware,
    TokenFlowMiddlewareSettings,
)


# --- the "host application" (unchanged, knows nothing about the module) -----

app = FastAPI(title="Host backend with token-flow-extender attached")


@app.post("/api/chat")
async def chat(request: Request):
    payload = await request.json()
    # In the real host this would call its LLM; here we echo.
    return {"answer": f"host handled {len(payload.get('message', ''))} chars"}


@app.get("/health")
def health():
    return {"ok": True}


# --- the ONE integration line(s) --------------------------------------------

class DemoLLM:
    """Replace with OpenAICompatLLM('http://your-vllm:8000', 'your-model')."""

    async def __call__(self, prompt: str) -> str:
        return f"[processed {len(prompt)} chars]"


app.add_middleware(
    TokenFlowMiddleware,
    settings=TokenFlowMiddlewareSettings(
        watch_paths=("/api/chat",),
        default_model="demo-model",
        llm_factory=lambda model: DemoLLM(),
        config=TokenFlowConfig.auto(storage_backend="memory"),
    ),
)
