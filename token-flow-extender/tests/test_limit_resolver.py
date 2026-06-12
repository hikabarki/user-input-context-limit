"""Tests for core/limit_resolver.py — the 5-step chain."""

from __future__ import annotations

from token_flow_extender.config import TokenFlowConfig
from token_flow_extender.core.limit_resolver import LimitResolver


def _config(tmp_path, registry_text: str | None = None) -> TokenFlowConfig:
    registry = tmp_path / "model_limits.yaml"
    if registry_text is not None:
        registry.write_text(registry_text, encoding="utf-8")
    return TokenFlowConfig(storage_backend="memory", registry_path=str(registry))


REGISTRY = """
models:
  "mock-model":
    max_input_tokens: 50000
    max_output_tokens: 2000
    aliases: ["mocky"]
"""


def test_explicit_wins(tmp_path):
    resolver = LimitResolver(_config(tmp_path, REGISTRY))
    limits = resolver.resolve("mock-model", max_input_tokens=12345)
    assert limits.max_input_tokens == 12345
    assert limits.source == "explicit"


def test_registry_resolution(tmp_path):
    resolver = LimitResolver(_config(tmp_path, REGISTRY))
    limits = resolver.resolve("mock-model")
    assert limits.max_input_tokens == 50000
    assert limits.max_output_tokens == 2000
    assert limits.source == "registry"


def test_registry_alias_and_case_insensitive(tmp_path):
    resolver = LimitResolver(_config(tmp_path, REGISTRY))
    assert resolver.resolve("MOCKY").max_input_tokens == 50000
    assert resolver.resolve("Mock-Model").max_input_tokens == 50000


def test_fallback_when_unknown(tmp_path):
    config = _config(tmp_path, REGISTRY)
    resolver = LimitResolver(config)
    limits = resolver.resolve("completely-unknown-model")
    assert limits.max_input_tokens == config.fallback_max_input_tokens
    assert limits.source == "fallback"
    assert limits.is_fallback


def test_fallback_when_registry_missing(tmp_path):
    resolver = LimitResolver(_config(tmp_path, None))
    limits = resolver.resolve("anything")
    assert limits.source == "fallback"


def test_hot_reload_on_mtime_change(tmp_path):
    import os
    import time
    config = _config(tmp_path, REGISTRY)
    resolver = LimitResolver(config)
    assert resolver.resolve("mock-model").max_input_tokens == 50000

    new_registry = REGISTRY.replace("50000", "2000000")
    registry_path = tmp_path / "model_limits.yaml"
    registry_path.write_text(new_registry, encoding="utf-8")
    os.utime(registry_path, (time.time() + 5, time.time() + 5))

    assert resolver.resolve("mock-model").max_input_tokens == 2000000


def test_probe_resolution(tmp_path, monkeypatch):
    """Step 2: live /v1/models probe wins over registry."""
    import token_flow_extender.core.limit_resolver as lr

    class _FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [{"id": "mock-model", "max_model_len": 777000}]}

    monkeypatch.setattr(lr.httpx, "get", lambda url, timeout: _FakeResponse())
    resolver = LimitResolver(_config(tmp_path, REGISTRY), base_url="http://vllm:8000")
    limits = resolver.resolve("mock-model")
    assert limits.max_input_tokens == 777000
    assert limits.source == "probe"
    # Probe result is cached for the TTL window.
    assert resolver.resolve("mock-model").max_input_tokens == 777000


def test_probe_failure_falls_through_to_registry(tmp_path, monkeypatch):
    import token_flow_extender.core.limit_resolver as lr

    def _boom(url, timeout):
        raise RuntimeError("endpoint down")

    monkeypatch.setattr(lr.httpx, "get", _boom)
    resolver = LimitResolver(_config(tmp_path, REGISTRY), base_url="http://vllm:8000")
    limits = resolver.resolve("mock-model")
    assert limits.source == "registry"
    assert limits.max_input_tokens == 50000


def test_invalid_registry_yaml_falls_back(tmp_path):
    resolver = LimitResolver(_config(tmp_path, ":: not yaml ["))
    assert resolver.resolve("anything").source == "fallback"


def test_explicit_output_tokens_respected(tmp_path):
    resolver = LimitResolver(_config(tmp_path, REGISTRY))
    limits = resolver.resolve("mock-model", max_input_tokens=1000,
                              max_output_tokens=111)
    assert limits.max_output_tokens == 111


def test_dynamic_adaptation_across_sizes(tmp_path):
    """Acceptance criterion 2: same code adapts to any limit, no edits."""
    from token_flow_extender.config import ResolvedLimits, TokenFlowConfig, WindowBudget
    cfg = TokenFlowConfig(storage_backend="memory")
    for max_input in (8192, 120000, 245000, 497000, 2000000):
        limits = ResolvedLimits(model="m", max_input_tokens=max_input,
                                max_output_tokens=1000, source="explicit")
        budget = WindowBudget.derive(limits, cfg)
        assert 0 < budget.usable_window_tokens < max_input
        assert budget.usable_window_tokens > max_input * 0.5  # sane derivation
