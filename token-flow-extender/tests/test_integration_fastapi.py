"""Integration tests: ASGI middleware pass-through, windowing, fail-open."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from token_flow_extender.api.middleware import (
    TokenFlowMiddleware,
    TokenFlowMiddlewareSettings,
)
from token_flow_extender.config import TokenFlowConfig

from conftest import MockLLM


REGISTRY = """
models:
  "mock-model":
    max_input_tokens: 200
    max_output_tokens: 50
"""


def _build_app(tmp_path, llm=None, registry: str = REGISTRY):
    registry_path = tmp_path / "model_limits.yaml"
    registry_path.write_text(registry, encoding="utf-8")
    config = TokenFlowConfig(storage_backend="memory",
                             registry_path=str(registry_path),
                             llm_retry_delay=0.0)

    app = FastAPI()

    @app.post("/api/chat")
    async def chat(request: Request):
        payload = await request.json()
        return {"echo": payload.get("message", ""), "handled_by": "host"}

    settings = TokenFlowMiddlewareSettings(
        watch_paths=("/api/chat",),
        default_model="mock-model",
        llm_factory=(lambda model: llm) if llm else None,
        config=config,
    )
    app.add_middleware(TokenFlowMiddleware, settings=settings)
    return app


def test_small_request_passes_through_to_host(tmp_path):
    app = _build_app(tmp_path, llm=MockLLM())
    client = TestClient(app)
    response = client.post("/api/chat", json={"message": "hello"})
    assert response.status_code == 200
    assert response.json()["handled_by"] == "host"
    assert response.json()["echo"] == "hello"


def test_unwatched_path_untouched(tmp_path):
    app = _build_app(tmp_path, llm=MockLLM())

    @app.get("/health")
    def health():
        return {"ok": True}

    client = TestClient(app)
    assert client.get("/health").json() == {"ok": True}


def test_non_json_body_replayed_byte_identical(tmp_path):
    """A non-JSON body must reach the host byte-identical (untouched)."""
    app = _build_app(tmp_path, llm=MockLLM())
    raw = b"\x00\x01 raw bytes \xff payload"

    @app.post("/api/raw-check")
    async def raw_check(request: Request):
        body = await request.body()
        return {"identical": body == raw, "length": len(body)}

    # Watch the raw path too, so the middleware actually buffers/replays it.
    app.user_middleware[0].kwargs["settings"].watch_paths = (
        "/api/chat", "/api/raw-check",
    )
    client = TestClient(app)
    response = client.post("/api/raw-check", content=raw,
                           headers={"content-type": "application/octet-stream"})
    assert response.status_code == 200
    assert response.json()["identical"] is True
    assert response.json()["length"] == len(raw)


def test_oversized_request_gets_windowed_response(tmp_path):
    llm = MockLLM()
    app = _build_app(tmp_path, llm=llm)
    client = TestClient(app)
    big_message = "word " * 5000  # far over the 200-token mock limit
    response = client.post("/api/chat", json={"message": big_message,
                                              "task": "summarize"})
    assert response.status_code == 200
    body = response.json()
    assert body.get("object") == "token_flow.result"
    assert body["coverage"]["complete"] is True
    assert body["coverage"]["tokens_lost"] == 0
    assert llm.calls > 1  # genuinely windowed


def test_fail_open_when_extender_breaks(tmp_path):
    """Acceptance criterion 5: corrupt module internals -> host still works."""

    class ExplodingLLM:
        async def __call__(self, prompt: str) -> str:
            raise RuntimeError("module is broken")

    app = _build_app(tmp_path, llm=ExplodingLLM())
    client = TestClient(app)
    big_message = "word " * 5000
    response = client.post("/api/chat", json={"message": big_message})
    # Fail-open: the ORIGINAL request reaches the host endpoint unchanged.
    assert response.status_code == 200
    assert response.json()["handled_by"] == "host"
    assert response.json()["echo"] == big_message


def test_oversized_without_llm_factory_passes_through(tmp_path):
    app = _build_app(tmp_path, llm=None)
    client = TestClient(app)
    big_message = "word " * 5000
    response = client.post("/api/chat", json={"message": big_message})
    assert response.status_code == 200
    assert response.json()["handled_by"] == "host"


def test_zero_coupling_grep():
    """Acceptance criterion 6: no reference to the host repos in src/."""
    import pathlib
    src = pathlib.Path(__file__).resolve().parents[1] / "src"
    offenders = []
    for path in src.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "mkd_keural" in text or "mkd-keural" in text:
            offenders.append(str(path))
    assert offenders == []


def test_no_hardcoded_limits_grep():
    """Acceptance criterion 7: no hardcoded model limits in src/."""
    import pathlib
    import re
    src = pathlib.Path(__file__).resolve().parents[1] / "src"
    banned = re.compile(r"120832|496768|245760|270000|131072|100_000")
    offenders = []
    for path in src.rglob("*.py"):
        if banned.search(path.read_text(encoding="utf-8")):
            offenders.append(str(path))
    assert offenders == []
