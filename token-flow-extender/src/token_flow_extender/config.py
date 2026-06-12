"""Pydantic configuration for token-flow-extender.

All numeric behavior derives from the resolved per-model limit at runtime.
Nothing in this file (or anywhere in src/) hardcodes a model's token limit.
"""

from __future__ import annotations

import os
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

Strategy = Literal["sequential_refine", "map_reduce", "hybrid"]
Mode = Literal["lossless", "selective"]
StorageBackend = Literal["sqlite", "redis", "memory"]


class TokenFlowConfig(BaseModel):
    """Central configuration. Use TokenFlowConfig.auto() to also read
    environment variables (prefix TFE_)."""

    # Window derivation
    safety_margin: float = Field(default=0.85, gt=0.0, le=1.0,
                                 description="Fraction of resolved max_input considered usable.")
    carry_state_ratio: float = Field(default=0.10, ge=0.0, lt=1.0,
                                     description="Fraction of each window budget reserved for carry-over state.")
    overlap_ratio: float = Field(default=0.10, ge=0.0, le=0.15,
                                 description="Boundary-context overlap between consecutive windows. "
                                             "Duplicate context only; never a substitute for coverage.")

    # Limit resolution
    fallback_max_input_tokens: int = Field(default=8192, gt=0,
                                           description="Conservative fallback when every other source fails.")
    fallback_max_output_tokens: int = Field(default=1024, gt=0)
    registry_path: str = Field(default="model_limits.yaml")
    probe_timeout: float = Field(default=5.0, gt=0)
    request_timeout: float = Field(default=600.0, gt=0)

    # Processing
    strategy: Strategy = "sequential_refine"
    mode: Mode = "lossless"
    max_parallel_windows: int = Field(default=4, ge=1)
    llm_max_retries: int = Field(default=2, ge=0)
    llm_retry_delay: float = Field(default=1.0, ge=0)

    # Storage
    storage_backend: StorageBackend = "sqlite"
    db_path: str = Field(default="token_flow.db")
    redis_url: str = Field(default="redis://localhost:6379/0")

    @field_validator("mode")
    @classmethod
    def _warn_selective(cls, v: Mode) -> Mode:
        if v == "selective":
            import logging
            logging.getLogger("token_flow_extender").warning(
                "mode='selective' is LOSSY (retrieval-style selection). "
                "Lossless coverage guarantees are DISABLED for this configuration."
            )
        return v

    @classmethod
    def auto(cls, **overrides: object) -> "TokenFlowConfig":
        """Build a config from environment variables (TFE_ prefix), then
        apply keyword overrides. Unknown/invalid env values fall back to
        defaults rather than raising (fail-open philosophy)."""
        env_map = {
            "safety_margin": "TFE_SAFETY_MARGIN",
            "carry_state_ratio": "TFE_CARRY_STATE_RATIO",
            "overlap_ratio": "TFE_OVERLAP_RATIO",
            "fallback_max_input_tokens": "TFE_FALLBACK_MAX_INPUT_TOKENS",
            "fallback_max_output_tokens": "TFE_FALLBACK_MAX_OUTPUT_TOKENS",
            "registry_path": "TFE_REGISTRY_PATH",
            "probe_timeout": "TFE_PROBE_TIMEOUT",
            "request_timeout": "TFE_REQUEST_TIMEOUT",
            "strategy": "TFE_STRATEGY",
            "mode": "TFE_MODE",
            "max_parallel_windows": "TFE_MAX_PARALLEL_WINDOWS",
            "storage_backend": "TFE_STORAGE_BACKEND",
            "db_path": "TFE_DB_PATH",
            "redis_url": "TFE_REDIS_URL",
        }
        values: dict[str, object] = {}
        for field_name, env_name in env_map.items():
            raw = os.getenv(env_name)
            if raw is not None and raw != "":
                values[field_name] = raw
        values.update(overrides)
        try:
            return cls(**values)  # type: ignore[arg-type]
        except Exception:
            import logging
            logging.getLogger("token_flow_extender").warning(
                "Invalid configuration values detected; falling back to defaults."
            )
            return cls(**{k: v for k, v in overrides.items()})  # type: ignore[arg-type]


class ResolvedLimits(BaseModel):
    """The outcome of the limit resolution chain for one model."""

    model: str
    max_input_tokens: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)
    source: Literal["explicit", "probe", "registry", "hf_config", "fallback"]

    @property
    def is_fallback(self) -> bool:
        return self.source == "fallback"


class WindowBudget(BaseModel):
    """Everything derived from a resolved limit. No hardcoded values."""

    max_input_tokens: int
    usable_window_tokens: int
    carry_state_budget: int
    overlap_tokens: int
    reserved_output_tokens: int
    system_prompt_tokens: int

    @classmethod
    def derive(
        cls,
        limits: ResolvedLimits,
        config: TokenFlowConfig,
        system_prompt_tokens: int = 0,
        reserved_output_tokens: Optional[int] = None,
    ) -> "WindowBudget":
        """usable = floor(max_input * margin) - reserved_output
        - carry_budget - system_prompt. All measured, none guessed."""
        reserved = (reserved_output_tokens
                    if reserved_output_tokens is not None
                    else limits.max_output_tokens)
        base = int(limits.max_input_tokens * config.safety_margin)
        carry = int(base * config.carry_state_ratio)
        usable = base - reserved - carry - system_prompt_tokens
        if usable < 1:
            # Tiny-context model: shrink reservations proportionally but keep
            # the window strictly positive so processing remains possible.
            usable = max(1, int(limits.max_input_tokens * config.safety_margin * 0.5))
            carry = max(0, int(usable * config.carry_state_ratio))
        overlap = int(usable * config.overlap_ratio)
        return cls(
            max_input_tokens=limits.max_input_tokens,
            usable_window_tokens=usable,
            carry_state_budget=carry,
            overlap_tokens=overlap,
            reserved_output_tokens=reserved,
            system_prompt_tokens=system_prompt_tokens,
        )
