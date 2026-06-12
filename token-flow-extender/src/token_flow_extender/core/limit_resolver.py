"""Dynamic model-limit resolution: the 5-step chain.

  1. Explicit value passed by the integrator
  2. Live probe of the serving endpoint (/v1/models -> max_model_len)
  3. Hot-reloadable model_limits.yaml registry
  4. HF config introspection (config.json -> max_position_embeddings)
  5. Conservative fallback from TokenFlowConfig + loud warning

ZERO hardcoded model limits exist in this module or anywhere in src/.
A new model with any context size works with no code changes: it is either
probed live or added to the registry file.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, Optional, Tuple

import httpx
import yaml

from ..config import ResolvedLimits, TokenFlowConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class _RegistryCache:
    """mtime-based hot reload of the YAML registry."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._path: Optional[str] = None
        self._mtime: float = -1.0
        self._data: Dict[str, Dict[str, Any]] = {}

    def load(self, path: str) -> Dict[str, Dict[str, Any]]:
        """Return the registry's model table, re-reading the file only when
        its mtime changes. Returns {} when the file is missing/invalid."""
        with self._lock:
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                return {}
            if path == self._path and mtime == self._mtime:
                return self._data
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    raw = yaml.safe_load(fh) or {}
                models = raw.get("models") or {}
                table: Dict[str, Dict[str, Any]] = {}
                for name, entry in models.items():
                    if not isinstance(entry, dict):
                        continue
                    table[str(name).lower()] = entry
                    for alias in entry.get("aliases") or []:
                        table[str(alias).lower()] = entry
                self._path, self._mtime, self._data = path, mtime, table
                logger.info("Model registry loaded: %d name(s) from %s", len(table), path)
                return table
            except Exception as exc:
                logger.warning("Failed to parse model registry %s: %s", path, exc)
                return {}


_registry_cache = _RegistryCache()


class LimitResolver:
    """Resolves a model's max input/output tokens via the 5-step chain."""

    def __init__(self, config: TokenFlowConfig,
                 base_url: Optional[str] = None,
                 hf_name: Optional[str] = None) -> None:
        self._config = config
        self._base_url = base_url
        self._hf_name = hf_name
        self._probe_cache: Dict[str, Tuple[float, Optional[int]]] = {}
        self._probe_ttl = 300.0  # seconds

    def resolve(
        self,
        model: str,
        max_input_tokens: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
    ) -> ResolvedLimits:
        """Run the chain and return ResolvedLimits with its source recorded."""

        # Step 1: explicit
        if max_input_tokens is not None and max_input_tokens > 0:
            return ResolvedLimits(
                model=model,
                max_input_tokens=max_input_tokens,
                max_output_tokens=max_output_tokens or self._config.fallback_max_output_tokens,
                source="explicit",
            )

        # Step 2: live endpoint probe
        probed = self._probe_endpoint(model)
        if probed is not None:
            registry_entry = self._registry_entry(model)
            out_tokens = (max_output_tokens
                          or (registry_entry or {}).get("max_output_tokens")
                          or self._config.fallback_max_output_tokens)
            return ResolvedLimits(
                model=model,
                max_input_tokens=probed,
                max_output_tokens=int(out_tokens),
                source="probe",
            )

        # Step 3: registry
        entry = self._registry_entry(model)
        if entry and isinstance(entry.get("max_input_tokens"), int):
            return ResolvedLimits(
                model=model,
                max_input_tokens=int(entry["max_input_tokens"]),
                max_output_tokens=int(
                    max_output_tokens
                    or entry.get("max_output_tokens")
                    or self._config.fallback_max_output_tokens
                ),
                source="registry",
            )

        # Step 4: HF config introspection
        hf_limit = self._hf_max_position_embeddings()
        if hf_limit is not None:
            return ResolvedLimits(
                model=model,
                max_input_tokens=hf_limit,
                max_output_tokens=max_output_tokens or self._config.fallback_max_output_tokens,
                source="hf_config",
            )

        # Step 5: conservative fallback + loud warning
        logger.error(
            "LIMIT RESOLUTION FALLBACK for model=%r: no explicit value, probe, "
            "registry entry, or HF config found. Using conservative %d input tokens. "
            "Add the model to model_limits.yaml to silence this warning.",
            model, self._config.fallback_max_input_tokens,
        )
        return ResolvedLimits(
            model=model,
            max_input_tokens=self._config.fallback_max_input_tokens,
            max_output_tokens=max_output_tokens or self._config.fallback_max_output_tokens,
            source="fallback",
        )

    # -- chain steps -------------------------------------------------------

    def _registry_entry(self, model: str) -> Optional[Dict[str, Any]]:
        table = _registry_cache.load(self._config.registry_path)
        return table.get(model.lower())

    def _probe_endpoint(self, model: str) -> Optional[int]:
        """GET {base_url}/v1/models and read max_model_len for this model.
        Results are cached for a short TTL; failures return None."""
        if not self._base_url:
            return None
        cached = self._probe_cache.get(model)
        if cached and (time.monotonic() - cached[0]) < self._probe_ttl:
            return cached[1]
        result: Optional[int] = None
        try:
            url = f"{self._base_url.rstrip('/')}/v1/models"
            response = httpx.get(url, timeout=self._config.probe_timeout)
            response.raise_for_status()
            for item in response.json().get("data", []):
                if item.get("id") == model or len(response.json().get("data", [])) == 1:
                    raw = item.get("max_model_len")
                    if isinstance(raw, int) and raw > 0:
                        result = raw
                        break
        except Exception as exc:
            logger.warning("Endpoint probe failed for %s at %s: %s", model, self._base_url, exc)
        self._probe_cache[model] = (time.monotonic(), result)
        if result:
            logger.info("Probed max_model_len=%d for model=%s", result, model)
        return result

    def _hf_max_position_embeddings(self) -> Optional[int]:
        if not self._hf_name:
            return None
        try:
            from transformers import AutoConfig  # type: ignore[import-not-found]
            hf_config = AutoConfig.from_pretrained(self._hf_name)
            value = getattr(hf_config, "max_position_embeddings", None)
            if isinstance(value, int) and value > 0:
                return value
        except Exception as exc:
            logger.warning("HF config introspection failed for %s: %s", self._hf_name, exc)
        return None
