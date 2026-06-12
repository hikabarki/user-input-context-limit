"""Tests for core/token_counter.py."""

from __future__ import annotations

import pytest

from token_flow_extender.config import TokenFlowConfig
from token_flow_extender.core.token_counter import (
    EndpointTokenCounter,
    FallbackTokenCounter,
    HeuristicTokenCounter,
    get_counter,
)
from token_flow_extender.exceptions import TokenCountError


class _Failing:
    def count(self, text: str) -> int:
        raise RuntimeError("boom")


class _Fixed:
    def count(self, text: str) -> int:
        return 42


def test_heuristic_empty_is_zero():
    assert HeuristicTokenCounter().count("") == 0


def test_heuristic_latin_text_reasonable():
    text = "hello world " * 100  # 1200 chars
    count = HeuristicTokenCounter().count(text)
    assert 200 <= count <= 600  # conservative over-estimate of ~300 real


def test_heuristic_korean_counts_higher_per_char():
    korean = "안녕하세요 한국어 문서입니다 " * 50
    latin = "a" * len(korean)
    heuristic = HeuristicTokenCounter()
    assert heuristic.count(korean) > heuristic.count(latin)


def test_heuristic_korean_near_one_token_per_char():
    korean = "토큰한도확장모듈" * 100  # 800 CJK chars
    count = HeuristicTokenCounter().count(korean)
    assert count >= 800  # conservative: at least 1 token per CJK char


def test_fallback_uses_first_success():
    counter = FallbackTokenCounter([_Failing(), _Fixed()])
    assert counter.count("anything") == 42


def test_fallback_all_fail_raises():
    counter = FallbackTokenCounter([_Failing(), _Failing()])
    with pytest.raises(TokenCountError):
        counter.count("anything")


def test_fallback_requires_counters():
    with pytest.raises(TokenCountError):
        FallbackTokenCounter([])


def test_get_counter_always_has_heuristic_tail():
    config = TokenFlowConfig(storage_backend="memory")
    counter = get_counter("any-model", config)  # no endpoint, no HF
    assert counter.count("hello world") >= 1


def test_get_counter_explicit_first():
    config = TokenFlowConfig(storage_backend="memory")
    counter = get_counter("any-model", config, explicit=_Fixed())
    assert counter.count("hello") == 42


class _FakeResponse:
    def __init__(self, payload, status_error=False):
        self._payload = payload
        self._status_error = status_error

    def raise_for_status(self):
        if self._status_error:
            raise RuntimeError("HTTP 500")

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, payload, status_error=False):
        self.payload = payload
        self.status_error = status_error
        self.requests = []

    def post(self, url, json=None):
        self.requests.append((url, json))
        return _FakeResponse(self.payload, self.status_error)


def test_endpoint_counter_returns_count():
    client = _FakeClient({"count": 123, "tokens": []})
    counter = EndpointTokenCounter("http://vllm:8000/", "my-model", client=client)
    assert counter.count("hello") == 123
    url, payload = client.requests[0]
    assert url == "http://vllm:8000/tokenize"
    assert payload["model"] == "my-model"
    assert payload["add_special_tokens"] is True


def test_endpoint_counter_bad_shape_raises():
    client = _FakeClient({"unexpected": True})
    counter = EndpointTokenCounter("http://vllm:8000", "m", client=client)
    with pytest.raises(TokenCountError):
        counter.count("hello")


def test_endpoint_counter_http_error_raises():
    client = _FakeClient({"count": 1}, status_error=True)
    counter = EndpointTokenCounter("http://vllm:8000", "m", client=client)
    with pytest.raises(TokenCountError):
        counter.count("hello")


def test_get_counter_endpoint_falls_back_to_heuristic():
    """Endpoint down -> heuristic fallback keeps counting (fail-open)."""
    config = TokenFlowConfig(storage_backend="memory", request_timeout=0.2)
    counter = get_counter("m", config, base_url="http://127.0.0.1:1")  # dead port
    assert counter.count("hello world this is a test") >= 1


def test_hf_counter_missing_dependency_handled():
    from token_flow_extender.core.token_counter import HFTokenCounter
    try:
        import transformers  # noqa: F401
        pytest.skip("transformers installed in this environment")
    except ImportError:
        with pytest.raises(TokenCountError):
            HFTokenCounter("any/model")
