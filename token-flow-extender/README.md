# token-flow-extender

Lossless unlimited-input processing layer for any LLM on the Keural platform
(or any OpenAI-compatible serving endpoint). Feed 1M, 10M, or more tokens to
a model with a 120K window: the module streams the input through the model in
dynamically sized windows, carries state between windows, merges the partial
results, and proves with a coverage ledger that 100 percent of the input was
processed. Zero tokens dropped, ever.

It is a fully independent package: it never imports, modifies, or
monkey-patches the host system. Install it, add one line, done. If the module
itself fails for any reason, it fails OPEN: the original request is forwarded
unchanged and the host keeps working exactly as before.

## The Golden Rule

LOSSLESS PROCESSING. Every token of the input is processed. This is not a
RAG/retrieval module that picks relevant chunks and discards the rest.
Default mode is 100 percent coverage, verified by accounting
(`coverage.complete == true`, `tokens_lost == 0` in every response).
A retrieval-style `mode="selective"` exists only as an explicit opt-in and is
clearly lossy.

## Quickstart (under 10 minutes)

### Install

    pip install ./token-flow-extender            # or from your index
    pip install "token-flow-extender[fastapi]"   # with FastAPI extras

### Pattern 3 — direct SDK

    import asyncio
    from token_flow_extender import TokenFlowExtender

    async def main():
        tfe = TokenFlowExtender.auto()
        result = await tfe.process(
            huge_input,                       # any size
            task="summarize",
            model="Qwen/Qwen3.5-35B-A3B",
            base_url="http://your-vllm-host:8090",   # used for /tokenize,
        )                                             # /v1/models probe, and calls
        print(result.passthrough)             # False for oversized inputs
        print(result.coverage.as_dict())      # {"complete": true, "tokens_lost": 0, ...}
        print(result.output)

    asyncio.run(main())

### Pattern 2 — decorator around any LLM-calling function

    from token_flow_extender import token_flow

    @token_flow(model="Qwen/Qwen3.5-35B-A3B", base_url="http://your-vllm-host:8090")
    async def my_llm_call(prompt: str) -> str:
        ...  # your existing transport, completely unchanged

Small prompts: the function is called once with the original prompt.
Oversized prompts: the function is called once per window, results merged.

### Pattern 1 — ASGI middleware (zero changes to host endpoints)

    from token_flow_extender import TokenFlowMiddleware, TokenFlowMiddlewareSettings

    app.add_middleware(TokenFlowMiddleware, settings=TokenFlowMiddlewareSettings(
        watch_paths=("/api/chat", "/api/message"),
        model_field="model",
        text_fields=("message", "prompt"),
        base_url_factory=lambda model: resolve_vllm_url(model),
    ))

Requests that fit the model's limit are forwarded byte-identical. Oversized
requests are processed by the module and answered with output plus coverage
proof. Any middleware error forwards the original request (fail-open).

## How it works

    incoming input (any size)
            |
            v
    [1] COUNT   exact token count with the target model's own tokenizer
            |
            v
    [2] DECIDE  fits within (max_input x safety_margin - reservations)?
            |
       +----+---------------+
       v YES                v NO
    PASS-THROUGH       [3] WINDOWED FLOW
    (untouched,             split into sequential windows sized to the
     zero overhead)         model's real max input (structure-aware)
                            |
                            v
                       [4] PROCESS  sequential_refine / map_reduce / hybrid
                            window 1 -> result 1
                            window 2 + carry(result 1) -> result 2
                            ...any number of windows...
                            checkpoint after EVERY window
                            |
                            v
                       [5] MERGE    hierarchical recursive reduce
                            |
                            v
                       [6] VERIFY   coverage ledger proves
                            processed_tokens == input_tokens (100 percent)

## Dynamic limit resolution (no hardcoded limits anywhere)

Per model, in priority order:

1. Explicit `max_input_tokens=` parameter
2. Live probe of the serving endpoint: `GET /v1/models` -> `max_model_len`
3. `model_limits.yaml` registry (hot-reloadable on file change)
4. HF config introspection: `max_position_embeddings`
5. Conservative fallback (default 8192) with a loud warning

A new model with a 2M context tomorrow needs zero code changes: it is probed
live, or you add three lines to `model_limits.yaml`.

From the resolved limit everything is derived:

    usable_window = floor(max_input x safety_margin)   # default 0.85
                  - reserved_output_tokens             # model's max_output
                  - carry_state_budget                 # default 10% of window
                  - system_prompt_tokens               # measured, not guessed

## Splitting rules

Semantic boundaries first: headers, then paragraph breaks, then sentence
ends, then raw tokens as a last resort. Code fences, markdown tables, and
base64 blobs are never split; a single unit larger than a window becomes
explicit `CONTINUED part i/n` windows. Every window carries
`{window_index, total_windows, char_range, token_range, doc_id, carry_state}`.
The union of window spans must equal the full input span; this is asserted
at planning time and verified again by the ledger after processing.

## Strategies

- `sequential_refine` (default): window n is processed with the compressed
  carry-state of windows 1..n-1. Best for reasoning/summary tasks.
- `map_reduce`: windows processed independently (parallel, bounded by
  `max_parallel_windows`), then merged hierarchically. Best for extraction.
- `hybrid`: picks map_reduce for extraction-type tasks (keywords: extract,
  find, list, collect...) and sequential_refine otherwise.

Carry-state compression only ever compresses the module's own intermediate
summaries, never unprocessed input.

## Checkpoints and recovery

A checkpoint is written after every window (SQLite default, Redis optional
via `storage_backend`). Kill the process at window 50/200, restart, and the
job resumes at 51: completed windows are never reprocessed and partial
results are never lost. Job ids are deterministic (same input + same config
= same job), so retries are idempotent. Job states:
`pending / running / paused / completed / failed`.

## Configuration

Environment variables (prefix `TFE_`) or `TokenFlowConfig(...)`:

| Option | Default | Meaning |
|---|---|---|
| safety_margin | 0.85 | usable fraction of resolved max input |
| carry_state_ratio | 0.10 | window fraction reserved for carry-over state |
| overlap_ratio | 0.10 | boundary-context overlap (duplicate context only) |
| fallback_max_input_tokens | 8192 | last-resort limit when resolution fails |
| fallback_max_output_tokens | 1024 | last-resort output reservation |
| registry_path | model_limits.yaml | hot-reloadable model registry |
| strategy | sequential_refine | sequential_refine, map_reduce, hybrid |
| mode | lossless | lossless (default) or selective (opt-in, lossy) |
| storage_backend | sqlite | sqlite, redis, memory |
| db_path | token_flow.db | SQLite checkpoint file |
| redis_url | redis://localhost:6379/0 | Redis backend URL |
| max_parallel_windows | 4 | map_reduce concurrency |
| llm_max_retries | 2 | retries per window call |
| probe_timeout | 5 | seconds for /v1/models probe |
| request_timeout | 600 | seconds for LLM/tokenize calls |

## Progress events

Wire `TokenFlowExtender(progress=callback)` or use the bundled
`ProgressBroker` and FastAPI router (`/token-flow/jobs/{id}/progress`
WebSocket) to stream events such as:

    window 17/212 complete (8.0%) - 0 tokens lost

## Running the proof

    pip install -e ".[dev]"
    pytest                                   # full suite
    python examples/million_token_demo.py    # 1.2M tokens, 1,000 needles, 0 lost
    python examples/basic_usage.py

## Repository layout

    src/token_flow_extender/
      config.py exceptions.py
      core/      token_counter, limit_resolver, window_planner,
                 flow_engine, merge_engine, coverage_ledger
      storage/   checkpoint_store, job_queue, state_backup
      api/       middleware, decorator, sdk, websocket, routes
      utils/     logger, validators, metrics
    tests/       full test suite (90%+ coverage on core/)
    examples/    basic_usage, fastapi_integration, million_token_demo
    model_limits.yaml   hot-reloadable registry
    AUDIT_REPORT.md     findings on the host system this module works around

## Hard guarantees

1. Inputs under the limit pass through untouched: identical request, zero
   added tokens.
2. Every failure path ends in fail-open pass-through plus a logged warning;
   the host app never crashes because of this module.
3. No imports from or writes to any host codebase
   (`grep -r "mkd_keural" src/` returns nothing).
4. No hardcoded model limits
   (`grep -rE "120832|496768|245760|270000|131072|100_000" src/` returns nothing).
5. JSON-only persistence; no pickle. Python 3.10+, typed, Pydantic config.
