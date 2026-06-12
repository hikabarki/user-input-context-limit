"""Input validation helpers."""

from __future__ import annotations

from ..exceptions import WindowPlanError


def require_text(value: object, name: str = "input") -> str:
    """Validate that a value is a non-None string and return it."""
    if value is None:
        raise WindowPlanError(f"{name} must not be None")
    if not isinstance(value, str):
        raise WindowPlanError(f"{name} must be str, got {type(value).__name__}")
    return value


def require_positive(value: int, name: str) -> int:
    """Validate that an integer is strictly positive."""
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise WindowPlanError(f"{name} must be a positive integer, got {value!r}")
    return value


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value into [lo, hi]."""
    return max(lo, min(hi, value))
