"""Exact per-model token counting.

Priority chain per model:
  1. EndpointTokenCounter — the serving endpoint's own /tokenize (vLLM).
     This is the ground truth: counts always match what the model will see.
  2. HFTokenCounter — local HuggingFace tokenizer (optional dependency).
  3. HeuristicTokenCounter — CJK-aware conservative estimate, last resort.

The factory get_counter() builds a FallbackTokenCounter that tries each in
order at call time, so a transient endpoint outage degrades gracefully
instead of failing the request (see AUDIT_REPORT.md finding B14 for why).
"""

from __future__ import annotations

import re
from typing import List, Optional, Protocol, Sequence

import httpx

from ..config import TokenFlowConfig
from ..exceptions import TokenCountError
from ..utils.logger import get_logger

logger = get_logger(__name__)

_CJK_RE = re.compile(
    "[ᄀ-ᇿ㄰-㆏가-힯"   # Korean jamo + hangul
    "぀-ヿㇰ-ㇿ"                  # Japanese kana
    "一-鿿㐀-䶿]"                 # CJK ideographs
)


class TokenCounter(Protocol):
    """Anything that can count tokens for a fixed model."""

    def count(self, text: str) -> int:
        """Return the exact (or best-available) token count of text."""
        ...


class EndpointTokenCounter:
    """Counts via the serving endpoint's POST {base_url}/tokenize.

    This is the same tokenizer the model itself uses, so counts are exact
    by construction (acceptance criterion 9)."""

    def __init__(self, base_url: str, model: str,
                 timeout: float = 600.0,
                 client: Optional[httpx.Client] = None) -> None:
        self._url = f"{base_url.rstrip('/')}/tokenize"
        self._model = model
        self._timeout = timeout
        self._client = client or httpx.Client(timeout=timeout)

    def count(self, text: str) -> int:
        """POST the text to the endpoint tokenizer and return its count."""
        payload = {
            "model": self._model,
            "prompt": text,
            "add_special_tokens": True,
            "return_token_strs": False,
        }
        try:
            response = self._client.post(self._url, json=payload)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            raise TokenCountError(
                f"Endpoint tokenize failed for model={self._model!r} at {self._url}: {exc}"
            ) from exc
        count = data.get("count")
        if not isinstance(count, int):
            raise TokenCountError(f"Unexpected tokenize response shape: {data!r}")
        return count


class HFTokenCounter:
    """Counts with a locally loaded HuggingFace tokenizer (extra: 'hf')."""

    def __init__(self, model_name_or_path: str) -> None:
        try:
            from transformers import AutoTokenizer  # type: ignore[import-not-found]
        except ImportError as exc:
            raise TokenCountError(
                "transformers is not installed; install token-flow-extender[hf]"
            ) from exc
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        except Exception as exc:
            raise TokenCountError(
                f"Could not load HF tokenizer for {model_name_or_path!r}: {exc}"
            ) from exc

    def count(self, text: str) -> int:
        """Count tokens with the loaded tokenizer (special tokens included)."""
        return len(self._tokenizer.encode(text, add_special_tokens=True))


class HeuristicTokenCounter:
    """Conservative CJK-aware estimate; deliberately OVER-estimates so the
    derived windows are never too large for the real model.

    Rationale: Korean/CJK text averages close to 1 token per character on
    modern BPE tokenizers, while Latin text averages roughly 4 characters
    per token. The host platform's own 4-chars-per-token assumption
    over-truncates Korean (AUDIT_REPORT.md finding B3); this counter exists
    only as a last-resort fallback and is biased to be safe, never exact."""

    def __init__(self, cjk_tokens_per_char: float = 1.0,
                 latin_chars_per_token: float = 3.5) -> None:
        self._cjk_rate = cjk_tokens_per_char
        self._latin_rate = latin_chars_per_token

    def count(self, text: str) -> int:
        """Estimate the token count of text (intentionally conservative)."""
        if not text:
            return 0
        cjk_chars = len(_CJK_RE.findall(text))
        other_chars = len(text) - cjk_chars
        estimate = cjk_chars * self._cjk_rate + other_chars / self._latin_rate
        return max(1, int(estimate) + 1)


class FallbackTokenCounter:
    """Tries an ordered sequence of counters; first success wins.

    A failure of an earlier counter is logged loudly but never propagated
    unless every counter fails."""

    def __init__(self, counters: Sequence[TokenCounter]) -> None:
        if not counters:
            raise TokenCountError("FallbackTokenCounter requires at least one counter")
        self._counters: List[TokenCounter] = list(counters)

    def count(self, text: str) -> int:
        """Return the first successful count in priority order."""
        last_error: Optional[Exception] = None
        for index, counter in enumerate(self._counters):
            try:
                return counter.count(text)
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Token counter %d/%d (%s) failed, falling back: %s",
                    index + 1, len(self._counters), type(counter).__name__, exc,
                )
        raise TokenCountError(f"All token counters failed; last error: {last_error}")


def get_counter(
    model: str,
    config: TokenFlowConfig,
    base_url: Optional[str] = None,
    hf_name: Optional[str] = None,
    explicit: Optional[TokenCounter] = None,
) -> TokenCounter:
    """Build the counter chain for a model.

    Order: explicit counter -> endpoint /tokenize (if base_url given)
    -> HF tokenizer (if hf_name given and transformers installed)
    -> heuristic fallback. The heuristic is ALWAYS appended so counting
    can never hard-fail an entire request."""
    chain: List[TokenCounter] = []
    if explicit is not None:
        chain.append(explicit)
    if base_url:
        try:
            chain.append(EndpointTokenCounter(base_url, model,
                                              timeout=config.request_timeout))
        except Exception as exc:  # client construction must never hard-fail
            logger.warning("Endpoint counter unavailable for %s: %s", model, exc)
    if hf_name:
        try:
            chain.append(HFTokenCounter(hf_name))
        except TokenCountError as exc:
            logger.warning("HF tokenizer unavailable for %s: %s", model, exc)
    chain.append(HeuristicTokenCounter())
    return FallbackTokenCounter(chain)
