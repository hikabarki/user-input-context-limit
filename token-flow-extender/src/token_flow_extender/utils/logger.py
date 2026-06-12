"""Structured logging for token-flow-extender.

Emits key=value structured records so log aggregators can parse them.
Never configures the root logger; only the package's own namespace.
"""

from __future__ import annotations

import logging
from typing import Any

_PACKAGE = "token_flow_extender"


class _StructuredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = getattr(record, "tfe_fields", None)
        if extras:
            kv = " ".join(f"{k}={v}" for k, v in extras.items())
            return f"{base} | {kv}"
        return base


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger with a structured formatter attached once."""
    logger = logging.getLogger(name if name.startswith(_PACKAGE) else f"{_PACKAGE}.{name}")
    pkg_logger = logging.getLogger(_PACKAGE)
    if not pkg_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(_StructuredFormatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s"
        ))
        pkg_logger.addHandler(handler)
        pkg_logger.setLevel(logging.INFO)
        pkg_logger.propagate = False
    return logger


def log_event(logger: logging.Logger, level: int, message: str, **fields: Any) -> None:
    """Log a message with structured key=value fields."""
    logger.log(level, message, extra={"tfe_fields": fields})
