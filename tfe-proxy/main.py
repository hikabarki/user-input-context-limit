"""TFE Proxy — standalone reverse proxy in front of mkd-keural-be-fastapi.

Intercepts POST /fastapi/v1/message:
  - Input fits the model's context  → forwards full request to host unchanged
    (host still runs RAG, file loading, conversation assembly, etc.)
  - Input too large                  → windows through vLLM via TFE, returns answer
  - Any error in TFE                 → fails open (forwards to host, logs warning)

Every other endpoint is forwarded to the host transparently.

Run:
    uvicorn main:app --host 0.0.0.0 --port 8900

Environment variables (see .env.example):
    HOST_BE_URL          URL of mkd-keural-be-fastapi   default: http://127.0.0.1:8000
    VLLM_URL             vLLM serving endpoint           default: http://110.8.41.182:11434
    TFE_MODEL            model name                      default: google/gemma-4-26B-A4B-it
    TFE_MAX_OUTPUT_TOKENS  max tokens per window         default: 2048
    TFE_DB_PATH          SQLite checkpoint file          default: tfe_proxy.db
    TFE_FORCE_MAX_INPUT_TOKENS  (testing only, unset in production)
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from token_flow_extender import (
    LimitResolver,
    OpenAICompatLLM,
    TokenFlowConfig,
    TokenFlowExtender,
    get_counter,
)

logger = logging.getLogger("tfe_proxy")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

HOST_BE_URL      = os.getenv("HOST_BE_URL", "http://127.0.0.1:8000").rstrip("/")
VLLM_URL         = os.getenv("VLLM_URL", "http://110.8.41.182:11434")
MODEL            = os.getenv("TFE_MODEL", "google/gemma-4-26B-A4B-it")
MAX_TOKENS       = int(os.getenv("TFE_MAX_OUTPUT_TOKENS", "2048"))
DB_PATH          = os.getenv("TFE_DB_PATH", "tfe_proxy.db")
_FORCE_MAX_INPUT = int(os.getenv("TFE_FORCE_MAX_INPUT_TOKENS", "0")) or None

_config = TokenFlowConfig(storage_backend="sqlite", db_path=DB_PATH)
_tfe    = TokenFlowExtender(config=_config)

# Cache resolved limit so we don't probe vLLM on every request
_cached_threshold: Optional[int] = None


def _get_threshold() -> int:
    global _cached_threshold
    if _cached_threshold is None:
        limits = LimitResolver(_config, base_url=VLLM_URL).resolve(
            MODEL, max_input_tokens=_FORCE_MAX_INPUT
        )
        _cached_threshold = int(limits.max_input_tokens * _config.safety_margin)
        logger.info("Resolved token threshold: %d for model %s", _cached_threshold, MODEL)
    return _cached_threshold


def _is_oversized(combined: str) -> bool:
    counter = get_counter(MODEL, _config, base_url=VLLM_URL)
    return counter.count(combined) > _get_threshold()


def _build_task(body: Dict[str, Any]) -> str:
    parts = []
    if body.get("user_name"):
        parts.append(f"User: {body['user_name']}")
    if body.get("datetime"):
        parts.append(f"Date: {body['datetime']}")
    if body.get("locale"):
        parts.append(f"Language: {body['locale']}")
    parts.append(f"\nQuestion: {body.get('message', '')}")
    return "\n".join(parts)


# Shared httpx client — reuses connections across requests
_http: httpx.AsyncClient


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http
    _http = httpx.AsyncClient(timeout=120)
    _get_threshold()  # warm up on startup
    yield
    await _http.aclose()


class MessageRequest(BaseModel):
    locale: str
    datetime: str
    user_id: str
    user_name: str
    message: str
    model: Optional[str] = None
    mode: Optional[str] = "standard"
    drive: Optional[bool] = False
    token: Optional[str] = None
    files: Optional[List[Any]] = None
    prompt_title: Optional[str] = None
    prompt: Optional[str] = None
    context: Optional[str] = None
    service_identity: Optional[str] = None


class MessageResponse(BaseModel):
    response: str


app = FastAPI(
    title="TFE Proxy",
    description="Reverse proxy for mkd-keural-be-fastapi with automatic context-limit handling.",
    version="1.0.0",
    docs_url="/docs",
    lifespan=lifespan,
)


@app.get("/health", tags=["Health"])
def health():
    return {"ok": True, "model": MODEL, "host": HOST_BE_URL, "threshold": _cached_threshold}


async def _forward_message(request: Request, body: dict) -> Response:
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in ("host", "content-length", "transfer-encoding")}
    resp = await _http.post(f"{HOST_BE_URL}/fastapi/v1/message", json=body, headers=headers)
    return Response(content=resp.content, status_code=resp.status_code,
                    media_type=resp.headers.get("content-type", "application/json"))


@app.post("/fastapi/v1/message", response_model=None, summary="Chat Message",
          description="Forwards to the host BE. If message+context exceeds the model's context window, TFE windows it automatically. Fails open on any TFE error.")
async def proxy_message(payload: MessageRequest, request: Request) -> Response:
    body     = payload.model_dump()
    context  = payload.context or ""
    combined = f"{context}\n\n{payload.message}".strip() if context else payload.message

    try:
        if not _is_oversized(combined):
            logger.info("passthrough user=%s tokens<=threshold", payload.user_id)
            return await _forward_message(request, body)

        logger.info("windowing user=%s input oversized", payload.user_id)
        llm    = OpenAICompatLLM(VLLM_URL, MODEL, max_tokens=MAX_TOKENS)
        result = await _tfe.process(
            input_text=combined,
            task=_build_task(body),
            model=MODEL,
            llm=llm,
            base_url=VLLM_URL,
            system_prompt=payload.prompt or "",
            strategy="map_reduce",
        )
        return JSONResponse({"response": result.output})

    except Exception as exc:
        logger.warning("TFE fail-open: %s — forwarding original request to host", exc)
        return await _forward_message(request, body)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"])
async def catch_all(request: Request, path: str) -> Response:
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in ("host", "content-length", "transfer-encoding")}
    resp = await _http.request(
        method=request.method,
        url=f"{HOST_BE_URL}/{path}",
        content=await request.body(),
        headers=headers,
        params=dict(request.query_params),
    )
    return Response(content=resp.content, status_code=resp.status_code,
                    media_type=resp.headers.get("content-type", "application/json"))
