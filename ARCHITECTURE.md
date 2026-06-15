# Architecture — User Input Context Limit Solution

## 1. System Overview

```mermaid
graph TD
    FE[Frontend / Client]
    PROXY[tfe-proxy :8900\nyou own this]
    BE[mkd-keural-be-fastapi :8000\nunchanged]
    RAG[mkd-keural-api-rag\nunchanged]
    GEMMA[vLLM — Gemma\n110.8.41.182:11434]
    QWEN[vLLM — Qwen\n110.8.41.182:8090]
    GPTOSS[vLLM — GPT-OSS\n110.8.41.182:41434]

    FE -->|POST /fastapi/v1/message| PROXY
    PROXY -->|small input: forward unchanged| BE
    PROXY -->|oversized input: windowed answer| FE
    BE -->|RAG context| RAG
    BE --> QWEN
    BE --> GEMMA
    BE --> GPTOSS
    PROXY -->|windowing calls| GEMMA

    style PROXY fill:#2563eb,color:#fff
    style BE fill:#374151,color:#fff
    style RAG fill:#374151,color:#fff
    style GEMMA fill:#065f46,color:#fff
    style QWEN fill:#374151,color:#fff
    style GPTOSS fill:#374151,color:#fff
```

---

## 2. tfe-proxy Request Flow

```mermaid
flowchart TD
    REQ[Incoming Request\nPOST /fastapi/v1/message]
    CHECK{message + context\n> 111K tokens?}
    FORWARD[Forward to BE unchanged\nhost runs RAG, files, LLM]
    TFE[token-flow-extender\nwindowed processing]
    FAILOPEN{TFE error?}
    RESP_HOST[Return BE response]
    RESP_TFE[Return windowed answer\n'response': '...']
    CATCHALL[All other endpoints\ntransparent forward to BE]

    REQ --> CHECK
    CHECK -->|No — fits| FORWARD
    CHECK -->|Yes — oversized| TFE
    TFE --> FAILOPEN
    FAILOPEN -->|No error| RESP_TFE
    FAILOPEN -->|Error — fail open| FORWARD
    FORWARD --> RESP_HOST

    style TFE fill:#2563eb,color:#fff
    style FAILOPEN fill:#92400e,color:#fff
    style CHECK fill:#1e3a5f,color:#fff
```

---

## 3. token-flow-extender Pipeline

```mermaid
flowchart LR
    INPUT[Large Input Text]

    subgraph RESOLVE[Step 1 — Limit Resolution]
        direction TB
        R1[explicit max_input_tokens]
        R2[vLLM probe /v1/models]
        R3[model_limits.yaml]
        R4[HuggingFace config]
        R5[conservative fallback]
        R1 --> R2 --> R3 --> R4 --> R5
    end

    subgraph COUNT[Step 2 — Token Count]
        TC[TokenCounter\nvLLM /tokenize endpoint\nor heuristic fallback]
    end

    subgraph PLAN[Step 3 — Window Planning]
        WP[WindowPlanner\nsplit into N windows\neach fits model limit]
    end

    subgraph MAP[Step 4 — Map]
        W1[Window 1 → LLM]
        W2[Window 2 → LLM]
        W3[Window N → LLM]
    end

    subgraph REDUCE[Step 5 — Reduce]
        M1[Merge batch 1]
        M2[Merge batch 2]
        MF[Final merge]
    end

    subgraph VERIFY[Step 6 — Coverage Ledger]
        CL[Verify 100% tokens processed\ntokens_lost == 0]
    end

    CP[(SQLite\nCheckpoints)]

    INPUT --> RESOLVE --> COUNT --> PLAN --> MAP
    MAP --> W1 & W2 & W3
    W1 & W2 & W3 --> REDUCE
    M1 & M2 --> MF --> VERIFY
    MAP <-->|resume on crash| CP

    style MAP fill:#1e3a5f,color:#fff
    style REDUCE fill:#065f46,color:#fff
    style VERIFY fill:#2563eb,color:#fff
```

---

## 4. What Was Built vs What Was Untouched

```mermaid
graph LR
    subgraph BUILT[Built — you own these]
        TFE[token-flow-extender\npip package]
        PROXY2[tfe-proxy\nFastAPI service]
    end

    subgraph UNTOUCHED[Untouched — other teams]
        BE2[mkd-keural-be-fastapi]
        RAG2[mkd-keural-api-rag]
    end

    PROXY2 -->|pip install| TFE
    PROXY2 -->|forwards to| BE2
    BE2 -->|calls| RAG2

    style BUILT fill:#1e3a5f,color:#fff
    style UNTOUCHED fill:#374151,color:#fff
```

---

## 5. Key Numbers

| Component | Metric |
|---|---|
| token-flow-extender tests | 64 passing |
| Core coverage | 93% |
| Bugs found in host platform | 15 (see AUDIT_REPORT.md) |
| Models supported | Gemma 131K · Qwen 496K · GPT-OSS 120K · Nano 270K · Keural 768 |
| Proxy token threshold (Gemma) | 111,411 tokens |
| Host repos modified | 0 |
