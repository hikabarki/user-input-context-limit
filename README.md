# user-input-context-limit

Context limit solution for the Keural AI chatbot platform.

## Problem

The Keural platform (mkd-keural-be-fastapi + mkd-keural-api-rag) has 15 bugs in its token/context limit handling — silent truncation of conversation history, wrong tokenizers, skipped overflow checks, and broken limits for some models. See [`token-flow-extender/AUDIT_REPORT.md`](token-flow-extender/AUDIT_REPORT.md) for the full findings.

## Solution

Two components, both standalone — zero changes to the host repos.

### `token-flow-extender/`
A pip package that processes inputs of any size against any OpenAI-compatible vLLM endpoint. Oversized inputs are split into windows, processed losslessly, and merged back into a single answer. A coverage ledger proves 100% of input tokens were processed.

### `tfe-proxy/`
A reverse proxy that sits in front of mkd-keural-be-fastapi. Small requests are forwarded to the host unchanged. Oversized requests are windowed by token-flow-extender before the host's broken truncation logic can fire.

```
Client → tfe-proxy:8900 → mkd-keural-be-fastapi:8000 → vLLM
                ↑
         auto-probes model limit
         windows anything oversized
         forwards everything else untouched
```

## Quick Start

```bash
# 1. Install the package
cd token-flow-extender
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q  # 64 tests, all pass

# 2. Run the proxy
cd ../tfe-proxy
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # edit HOST_BE_URL and VLLM_URL
uvicorn main:app --host 0.0.0.0 --port 8900
```

Swagger UI: `http://<server>:8900/docs`  
Health check: `http://<server>:8900/health`
