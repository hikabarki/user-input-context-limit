# AUDIT_REPORT.md — Input-Token Handling Audit

Phase 0 read-only audit of `mkd-keural-be-fastapi-main` (BE) and `mkd-keural-api-rag-main` (RAG).
Date: 2026-06-11. No files in either repo were modified.

All file paths are relative to each repo's inner `app/` root.
Severity scale: CRITICAL (data loss or broken request path in production), HIGH (silent data loss or wrong limit), MEDIUM (inconsistency, latent failure), LOW (smell, perf, maintainability).

---

## 1. Verified ground truth

### 1.1 Per-model token limits (BE `config/constants.py`, lines 33-54)

| Model | MAX_TOKEN | MAX_INPUT | MAX_OUTPUT | Input+Output vs MAX |
|---|---|---|---|---|
| GPT-OSS (`openai/gpt-oss-120b`) | 131,072 | 120,832 | 10,240 | 131,072 = MAX. Consistent |
| Qwen (`Qwen/Qwen3.5-35B-A3B`) | 504,960 | 496,768 | 8,192 | 504,960 = MAX. Consistent |
| Gemma (`google/gemma-4-26B-A4B-it`) | 262,144 | 245,760 | 16,384 | 262,144 = MAX. Consistent |
| GPT-5 Nano (`gpt-5-nano`) | 400,000 | 270,000 | 32,768 | 302,768 != 400,000. Inconsistent (see B12) |
| Keural (`keural-alpha`) | 1,024 | none defined | none defined | No input/output split exists (see B1, B13) |

### 1.2 Limit resolver (BE `service/llm/util.py`, `get_llm_max_input_token`, lines 140-150)

If/elif chain on model-name string sets. Final `else` returns `settings.KEURAL_MAX_TOKEN` (1,024) — this is the fallback for keural AND for any unrecognized model name.

### 1.3 Overflow handling (BE `service/message/process.py`, lines 119-156)

Token check runs only for models in `{"gpt-oss-120", "openai/gpt-oss-120b", "Qwen/Qwen3.5-35B-A3B", "Qwen/Qwen3.5-122B-A10B-FP8"}` (note the typo, B2). Over 100% of limit returns `TOKEN_OVER`. Over 75% silently drops conversation history using a 4-chars-per-token estimate (B3).

### 1.4 Agent summarization triggers

- `service/agent/orchestrator.py` line 415: `max_input_tokens * 0.90`
- `service/agent/experiment_orchestrator.py` line 173: `max_input_tokens * 0.75` (imports the same `_MODEL_CONFIG` from orchestrator, line 57)

### 1.5 RAG hard cap (RAG `service/rag/process.py`)

Line 30: `MAX_LLM_CONTEXT_TOKENS = 100_000`. Line 93: `trim_hits_to_token_limit(hits)` applied to every retrieval regardless of target model.

### 1.6 LLM call entry points and output caps

`service/llm/generate.py` (GPT-OSS), `generate_qwen.py`, `generate_gemma.py`, `generate_keural.py`, `generate_gptnano.py`. Each hardcodes `max_tokens` to its `*_MAX_OUTPUT_TOKEN` constant; keural uses `KEURAL_MAX_TOKEN` (1,024) as `max_tokens`.

### 1.7 Tokenizers

- BE `service/llm/tokenize.py`: POST `{model_url}/tokenize` — counts with the target model's own serving endpoint. Correct pairing.
- RAG `service/llm/tokenize.py` line 27: POST to `VLLM_GEMMA_URL/tokenize` with payload model defaulting to `QWEN_MODEL`. Mismatched (B5).

### 1.8 Hardcoded model selection

- `service/message/process.py` line 35: `model = settings.QWEN_MODEL` (request model ignored)
- `service/chat/process.py` line 45: `model = settings.GEMMA_MODEL` (two alternatives commented out)

---

## 2. Findings

### B1 — CRITICAL — Resolver returns 1,024 (a TOTAL cap) as the INPUT limit for keural and ALL unknown models
- File: BE `service/llm/util.py`, lines 149-150
- Evidence: `else: return settings.KEURAL_MAX_TOKEN`. `KEURAL_MAX_TOKEN = 1024` is a total-context-style cap (constants.py line 54; no `KEURAL_MAX_INPUT_TOKEN` exists). The `else` branch also catches every future or misspelled model name, silently assigning them a 1,024-token input limit.
- Impact: Any caller trusting this value treats a model as having a 1,024-token input window; unknown models inherit the same wrong limit instead of failing loudly.
- Suggested fix (owning team): add explicit `KEURAL_MAX_INPUT_TOKEN`/`KEURAL_MAX_OUTPUT_TOKEN`; make the unknown-model path raise or log an error rather than silently returning the smallest limit in the system.

### B2 — HIGH — Token overflow check skipped for Gemma, GPT-5 Nano, Keural, and (via typo) bare "gpt-oss-120b"
- File: BE `service/message/process.py`, line 121
- Evidence: check-set contains the typo `"gpt-oss-120"` (missing `b`); `format_generate_response` (util.py line 50) accepts `"gpt-oss-120b"`, so that alias skips the check. Gemma, nano, and keural are absent from the set entirely; the `else` branch (line 151-152) just logs "Skip token size check".
- Impact: Oversized prompts for unchecked models go straight to the serving endpoint, producing provider-side errors or provider-side silent truncation. Note `chat/process.py` currently pins `model = GEMMA_MODEL` — the production streaming model is one of the unchecked ones.
- Suggested fix: drive the check from `get_llm_max_input_token()` for every model instead of a hardcoded name set.

### B3 — HIGH — Silent truncation of conversation history at 75%, with a chars-per-token heuristic that over-drops Korean text
- File: BE `service/message/process.py`, lines 131-149
- Evidence: `chars_to_drop = int((prompt_token_count - threshold) * 4)`; `conversation_context = conversation_context[chars_to_drop:]`. Only a server-side `logger.warning` — the API response carries no indication that history was discarded.
- Impact, four distinct defects:
  1. Caller is never informed (silent data loss — violates lossless expectations).
  2. The 4-chars/token estimate is tuned for English. Korean runs roughly 1-2.5 chars per token, so the same token overshoot spans fewer chars; dropping `tokens * 4` chars removes roughly 2x more Korean history than needed.
  3. Truncation targets only `conversation_context`. If the overflow comes from `file_context` or `rag_context`, dropping history may not bring the prompt under the limit — and the rebuilt prompt is never re-counted before sending.
  4. `chars_to_drop` can exceed `len(conversation_context)`, wiping all history and still not resolving the overflow.
- Suggested fix: re-tokenize after truncation; loop or fail loudly; return a flag/warning in the response; use measured token counts, not char heuristics.

### B4 — HIGH — Full-file "analyze" mode is unbounded and meets no token check
- Files: BE `service/message/process.py` lines 85-87 to RAG `service/rag/target_file.py` lines 34-37
- Evidence: `mode="analyze"` calls `retrieve_full_sources(target_ids)`, which reconstructs entire files from Qdrant with no token counting or trimming anywhere in that path; the resulting context then re-enters `process.py`, where the token check is skipped for most models (B2).
- Impact: A large drive file in analyze mode produces a prompt of arbitrary size sent directly to the model.
- Suggested fix: count tokens on the full-source path and surface overflow to the caller.

### B5 — HIGH — RAG service counts tokens with the wrong tokenizer endpoint
- File: RAG `service/llm/tokenize.py`, lines 26-29
- Evidence: URL is `VLLM_GEMMA_URL/tokenize` (the Qwen line is commented out) while the payload `model` defaults to `settings.QWEN_MODEL`. A vLLM server tokenizes with the model it serves; the payload/model and endpoint disagree, and neither necessarily matches the model that will consume the context.
- Impact: All RAG trimming decisions (B6) are made with Gemma token counts. Gemma and Qwen tokenizers differ materially on Korean text, so the 100K cap is enforced against the wrong measure — either over-trimming (extra silent loss) or under-trimming (downstream overflow).
- Suggested fix: resolve the tokenize endpoint from the target model per request.

### B6 — HIGH — RAG hard cap of 100,000 tokens ignores per-model capability and discards hits silently
- File: RAG `service/rag/process.py`, lines 30, 48-68, 93
- Evidence: `MAX_LLM_CONTEXT_TOKENS = 100_000` module constant; `trim_hits_to_token_limit()` pops hits until under the cap; removals are logged at DEBUG level only (lines 64-66); the API response does not indicate that retrieved content was dropped.
- Impact: For Qwen (496K input) and Gemma (245K input), 60-80% of usable context capacity is forfeited. Retrieved, relevant content is silently discarded with no caller-visible signal.
- Suggested fix: accept the target model's budget as a request parameter; report dropped-hit count in the response; log removals at WARNING.

### B7 — CRITICAL — Keural generation parameters cannot succeed on non-trivial input
- File: BE `service/llm/generate_keural.py`, lines 36 and 71
- Evidence: `"max_tokens": settings.KEURAL_MAX_TOKEN` (1,024) — but 1,024 is the model's TOTAL context (constants.py line 54). Requesting 1,024 completion tokens leaves zero budget for the prompt; any non-empty input makes `input + max_tokens > total`, which vLLM rejects.
- Impact: The keural path fails (or is server-side clamped) for effectively all real prompts.
- Suggested fix: define separate input/output budgets for keural and set `max_tokens = total - measured_input` or a fixed output budget well below total.

### B8 — MEDIUM — Inconsistent summarization thresholds across orchestrators (0.90 vs 0.75)
- Files: BE `service/agent/orchestrator.py` line 415 (0.90); `service/agent/experiment_orchestrator.py` line 173 (0.75)
- Evidence: same `_MODEL_CONFIG`, same middleware, different magic numbers, no shared constant, no comment explaining the divergence.
- Impact: The same conversation summarizes (lossy compression of history) at very different points depending on entry path; behavior drift between prod and experiment makes experiment results non-representative.
- Suggested fix: single named constant (or per-model config field) with documented rationale.

### B9 — MEDIUM — Agent `_MODEL_CONFIG` covers only Qwen and Gemma; other models raise at runtime
- File: BE `service/agent/orchestrator.py`, lines 159-188, 415
- Evidence: dict keys are only `settings.QWEN_MODEL` and `settings.GEMMA_MODEL`. Line 415 does a direct `_MODEL_CONFIG[model_name]` index — a KeyError (not the friendlier ValueError in `_build_llm`, line 191) if any other model is routed in.
- Impact: Latent crash path; the limit knowledge here also duplicates `get_llm_max_input_token()` (two sources of truth that can drift).
- Suggested fix: derive `_MODEL_CONFIG` token fields from the single resolver; guard the line-415 lookup.

### B10 — MEDIUM — Streaming chat path has no pre-flight token check at all
- File: BE `service/chat/process.py` (entire flow), relying on `orchestrator.py` SummarizationMiddleware
- Evidence: no call to `get_tokenize`/`get_llm_max_input_token` anywhere in the streaming path; protection is only the 0.90-trigger summarization middleware, which compresses history lossily and invisibly to the user.
- Impact: A single oversized turn (huge file context in one message, between 90% and 100%+ of the window) can pass the trigger gap and hit the provider limit raw; summarization losses are never disclosed.
- Suggested fix: pre-flight count against the resolver limit before dispatching the agent.

### B11 — MEDIUM — Trim loop is O(n^2) network tokenize calls over the full context
- File: RAG `service/rag/process.py`, lines 55-66
- Evidence: each loop iteration re-formats ALL remaining hits and re-tokenizes the full concatenated context over HTTP, removing one hit at a time.
- Impact: For large retrievals, dozens of sequential tokenize round-trips, each carrying up to ~100K tokens of text; latency and tokenize-server load scale quadratically.
- Suggested fix: tokenize per-hit once and trim by cumulative sum, or binary-search the cut point.

### B12 — LOW — GPT-5 Nano constant arithmetic inconsistent
- File: BE `config/constants.py`, lines 49-51
- Evidence: `270,000 + 32,768 = 302,768 != NANO_MAX_TOKEN (400,000)`. Every other model satisfies input + output = total.
- Impact: 97K tokens of context capacity unaccounted for; unclear which number is authoritative; anyone computing input limit as `MAX - OUTPUT` gets 367,232, not 270,000.
- Suggested fix: document or correct so the three values are arithmetically consistent.

### B13 — LOW — Unbounded-memory token-count cache keyed on full prompt strings
- File: RAG `service/llm/tokenize.py`, line 23
- Evidence: `@lru_cache(maxsize=4096)` on `tokenize_count_cached(prompt, ...)` — each cache entry retains the entire prompt string (up to ~100K tokens, hundreds of KB) as its key.
- Impact: Up to 4,096 large contexts pinned in memory (potentially GBs); cache hits are also near-useless since large contexts rarely repeat exactly.
- Suggested fix: cache on a content hash, or drop caching for large inputs.

### B14 — LOW — Tokenize failure fails closed and blocks the request
- Files: BE `service/llm/tokenize.py` lines 37-43 with `service/message/process.py` lines 154-156
- Evidence: `raise_for_status()` re-raises; the caller catches it and returns `SERVER_ERROR` for the whole request.
- Impact: A transient tokenize-endpoint failure turns a servable request into a 5xx — counting infrastructure is a hard dependency of the happy path.
- Suggested fix: on count failure, fall back to a conservative estimate plus warning instead of failing the request.

### B15 — INFO — Hardcoded model selection ignores request parameters
- Files: BE `service/message/process.py` line 35 (`QWEN_MODEL`), `service/chat/process.py` lines 42-45 (`GEMMA_MODEL`, alternatives commented out)
- Impact: Not a token bug per se, but it means each path's overflow behavior is whatever the pinned model gets (Qwen: checked; Gemma: unchecked, B2). Any model switch silently changes overflow semantics.

---

## 3. Consequences for token-flow-extender

The module must not trust or depend on any of the following host behaviors:

1. `get_llm_max_input_token()` — wrong for keural/unknown models (B1). The module resolves limits via its own 5-step chain.
2. Host token counts — the RAG service counts with a mismatched tokenizer (B5). The module counts with the target model's own tokenizer and validates endpoint/model pairing.
3. Host overflow handling — silent truncation (B3), silent trimming (B6), skipped checks (B2, B10). The module intercepts BEFORE these fire, so oversized inputs are windowed losslessly and never reach the host's truncation code paths.
4. `input + max_output <= total` arithmetic — not reliable for nano/keural (B7, B12). The module derives usable windows from `max_input` and measured reservations, never from `MAX_TOKEN` subtraction.
5. Threshold values — 0.75 / 0.90 are inconsistent (B8). The module uses its own configurable safety margin (default 0.85) and does not mirror host thresholds.

End of report.
