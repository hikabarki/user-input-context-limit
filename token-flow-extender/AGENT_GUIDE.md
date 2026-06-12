# AGENT_GUIDE.md — running this project with an automated agent

This file is the operational manual for an AI agent (or a new developer)
asked to install, run, verify, or integrate token-flow-extender.

## 1. What this project is

A standalone pip package that lets any model behind an OpenAI-compatible
endpoint process inputs larger than its context window, losslessly, by
windowed streaming with carry-state, checkpointed resume, and a coverage
ledger that proves 100 percent of input tokens were processed.
It must NEVER import from or modify the host repositories
(mkd-keural-be-fastapi, mkd-keural-api-rag). Integration is install + one line.

## 2. Environment setup

Requirements: Python 3.10+.

    cd token-flow-extender
    python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
    pip install -e ".[dev]"

## 3. Verify the build (run these in order)

    pytest -q
        Expected: all tests pass.

    pytest --cov=src/token_flow_extender/core --cov-report=term
        Expected: 90%+ total coverage on core/.

    python examples/million_token_demo.py
        Expected final line: "PASS: all 1,000 needles accounted for;
        100 percent token coverage."

    python examples/basic_usage.py
        Expected: passthrough True for the small input, coverage complete
        True for the big input.

    grep -r "mkd_keural\|mkd-keural" src/
        Expected: no output (zero coupling).

    grep -rE "120832|496768|245760|270000|131072|100_000" src/
        Expected: no output (no hardcoded limits).

## 4. Run the demo HTTP service

    pip install -e ".[fastapi]"
    uvicorn examples.fastapi_integration:app --port 8800

    # small request -> handled by host endpoint untouched:
    curl -s -X POST localhost:8800/api/chat -H 'content-type: application/json' \
         -d '{"message": "hello"}'

    # oversized request -> windowed by the module, coverage report included.

Or with Docker:

    docker compose up --build

## 5. Integrating with a real platform

1. Find each model's serving base URL (vLLM). The module uses it for
   `/tokenize` (exact counts), `/v1/models` (limit probe), and
   `/v1/chat/completions` (window calls).
2. Choose a pattern:
   - Own code calling LLMs: wrap the call with `@token_flow(...)`.
   - Own FastAPI app in front of the platform: `app.add_middleware(TokenFlowMiddleware, ...)`.
   - Anything else: use `TokenFlowExtender.process(...)` directly.
3. Add models to `model_limits.yaml` if their endpoints cannot be probed.
4. For durable jobs across restarts set `TFE_STORAGE_BACKEND=sqlite`
   (default) and a persistent `TFE_DB_PATH`, or `redis`.

## 6. Operational notes

- Pass-through is the hot path: under-limit traffic is never modified.
- All failures fail open. If you see the warning
  "TokenFlowMiddleware fail-open", the original request was forwarded and
  served by the host; investigate the logged exception at leisure.
- Resume: rerun the same `process()` call (same input, task, model, config);
  the deterministic job id resumes from the last completed window.
- Korean text: token counting must use the endpoint tokenizer (it does by
  default when base_url is set). The heuristic fallback intentionally
  over-estimates CJK so windows are never oversized.
- Never enable `mode="selective"` unless lossy retrieval is explicitly
  wanted; it disables the lossless guarantee.

## 7. Where things are

- AUDIT_REPORT.md — bugs found in the host platform that this module is
  designed to work around (do not rely on host limit handling).
- src/token_flow_extender/core/ — counting, limits, planning, flow, merge,
  ledger. Start reading at api/sdk.py: process() shows the whole pipeline.
- tests/ — each acceptance criterion has a corresponding test; the
  kill-and-resume proof is tests/test_checkpoint_resume.py.
