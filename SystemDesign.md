# System Design & Architecture

## High-level architecture

Two services communicate over HTTP. All persistent state lives in SQLite (structured data) and ChromaDB (vector embeddings, two separate collections).

```
┌──────────────────────┐        HTTP         ┌──────────────────────────────────────────┐
│   Streamlit UI        │ ──────────────────▶ │          FastAPI Backend                  │
│   Port 8501           │                     │          Port 8000                        │
│                       │                     │                                           │
│  - Workspace login    │                     │  POST /brief  ← primary endpoint          │
│  - Brief viewer       │                     │  POST /chat   ← backwards-compat stub     │
│  - File upload        │                     │  POST /chat/stream                        │
│  - Bulk questionnaire │                     │  POST /upload-doc (doc_type param)        │
│  - Analytics          │                     │  GET  /list-docs                          │
│  - Audit log          │                     │  POST /delete-doc                         │
│  - System logs        │                     │  GET  /analytics  /audit-log  /logs       │
└──────────────────────┘                     │  POST /answer-questionnaire               │
                                             │  GET  /health                             │
                                             └──────────────┬────────────────────────────┘
                                                            │
                   ┌────────────────────────────────────────┼──────────────────────────────┐
                   │                                        │                              │
        ┌──────────▼──────────┐                  ┌─────────▼──────┐            ┌──────────▼──────┐
        │  ChromaDB (disk)     │                  │   SQLite        │            │   LLM Provider   │
        │                      │                  │                 │            │  (Gemini 2.5-    │
        │  child_chunks        │                  │ application_logs│            │   flash today)   │
        │  (embedded, queried) │                  │ document_store  │            │                  │
        │                      │                  │ brief_logs      │            │  Single source   │
        │  parent_chunks       │                  │ schema_         │            │  of truth:       │
        │  (stored, fetched    │                  │ migrations      │            │  LLM_MODEL env   │
        │   by ID for context) │                  └─────────────────┘            └──────────────────┘
        └──────────────────────┘
```

---

## Startup sequence

`lifespan()` in `main.py` runs three steps before `/health` reports ready:

1. **Migrations** — `run_migrations()` scans `api/migrations/` for `.sql` files in alphabetical order, applies any not yet recorded in `schema_migrations`, and records them.
2. **Env-var validation** — In `ENVIRONMENT=production`, missing `API_KEY` or `JWT_SECRET` aborts startup. In dev, missing values produce warnings.
3. **Model warmup** — `warmup_models()` (in `chroma_utils`) eagerly loads the 440MB nomic-embed model and the cross-encoder reranker. Without this, the embedding model lazy-loads on the first upload request, which can hang for 3-5 minutes on a cold disk cache and trip request timeouts. Warmup failures (no internet, HF Hub down, disk full) are caught and logged as a warning — the API becomes ready in degraded mode and falls back to lazy loading on first request, rather than the entire process becoming unhealthy.

Migration files applied in order:
- `001_initial_schema.sql` — creates `application_logs` and `document_store`
- `002_add_user_id.sql` — adds `user_id` column to both tables
- `003_add_indexes.sql` — adds indexes on `(user_id, session_id)`, `(user_id, created_at DESC)`, `(user_id)`
- `004_add_brief_logs.sql` — creates `brief_logs(customer_id, query, brief_json, faithfulness_score, loop_count)`

---

## Authentication design

Two independent auth layers:

### Layer 1 — API key (service-level)
`X-API-Key` header checked against `API_KEY` env var. If unset, all requests pass through (development mode). Applied on all write endpoints and sensitive reads via `dependencies=[Depends(verify_api_key)]`.

### Layer 2 — JWT workspace tokens (user-level)
`POST /auth/token` takes `{workspace, passkey}`, computes `user_id = sha256(workspace:passkey)[:32]`, signs a 24-hour HS256 JWT using `JWT_SECRET`. The `get_current_user` dependency decodes the Bearer token to extract `user_id`. **Tenant identity is taken only from the signed token** — there is no `?user_id=` query-param fallback. (An earlier version had one for "backwards compatibility" but it allowed any caller to spoof another tenant; it has been removed.) When no Authorization header is present, `get_current_user` resolves to a single shared `"default"` tenant; production deployments rely on `API_KEY` enforcement on write endpoints to keep that path closed in practice.

The deterministic hash means the same workspace + passkey always produces the same `user_id` — sessions survive restarts without an account system. The passkey is the secret.

> Dev caveat: when `JWT_SECRET` is unset, the API generates a random secret at import time. Under `uvicorn --workers > 1` each worker has its own secret, so a token minted by one worker won't validate on another. Always set `JWT_SECRET` for any non-single-process run.

---

## Document ingestion pipeline

### Format-aware chunking + parent-child split

```
File → format loader → raw documents
          ↓
    _PARENT_SPLITTER (1600 chars, 200 overlap)
    → parent_docs → stored in parent_chunks Chroma collection (by deterministic ID)
          ↓
    _CHILD_SPLITTER (500 chars, 50 overlap) per parent_doc
    → child_docs (each carries parent_chunk_id in metadata)
    → stored in child_chunks Chroma collection (embedded)
```

**Why parent-child?**
Child chunks are small enough for precise embedding-based retrieval. Parent chunks provide a larger surrounding context window so the LLM reasons over complete thoughts, not fragment sentences. Retrieval targets children; the LLM reads parents.

### Format-specific loaders

| Format | Loader | Strategy |
|--------|--------|----------|
| PDF | pymupdf (`fitz`) | `page.get_text("text")` → `noise_filter()` removes headers/footers/page numbers → `sentence_chunk(size=256 words, overlap=64 words)` using spaCy sentence boundaries |
| HTML | `HTMLHeaderTextSplitter` | Splits on h1/h2/h3, each section becomes a chunk with header in metadata |
| DOCX | `python-docx` | Groups body paragraphs under the preceding Heading-style paragraph |
| TXT | `transcript_parser` + `transcript_chunker` | Parses speaker-labelled turns, groups into ~300-word chunks with 2-turn overlap |
| JSON | `ticket_parser` + `ticket_chunker` | Parses ticket fields into TicketStructure, one Document per section (description / comment / resolution) |

---

## Brief generation pipeline (LangGraph)

The core query pipeline is a `StateGraph` with four nodes and a conditional loop edge.

### State schema (`graph/state.py`)

```
customer_id       — workspace identifier
original_query    — raw user question
sub_queries       — decomposed sub-queries (set by query_rewrite_node)
retrieved_chunks  — child chunks from HybridRetriever
parent_chunks     — expanded context from fetch_parents()
reasoning_output  — structured JSON from LLM analyst
iteration_count   — loop counter (hard cap: 3)
is_sufficient     — set by completeness_node
brief             — final formatted output
information_gaps  — gaps found by completeness_node
audit_trail       — per-node event log
```

### Node descriptions

**query_rewrite_node** — Calls the LLM with a decomposition prompt. Returns 2–4 targeted sub-queries. Increments `iteration_count`. On a loop iteration, the prompt is augmented with `information_gaps` from the previous pass so sub-queries are *refined* rather than just regenerated. Falls back to the original query if `llm_breaker` is open.

**retrieve_node** — For each sub-query, runs `HybridRetriever` (dense + BM25 + RRF + cross-encoder top-4). Deduplicates across sub-queries. Tags each child chunk with a stable `chunk_id`. Calls `fetch_parents()` to expand to parent chunks. If contextual retrieval was enabled at ingest time, the retrieved chunks already include the LLM-generated context prefix in their `page_content` — the reasoning LLM benefits from seeing that context, but it's stripped at the citation-display layer (see brief generator below).

**reason_node** — Calls the LLM with the analyst prompt. Context is the parent chunks (or child chunks as fallback), each prefixed with its `chunk_id`. Returns structured JSON: `{issues, risks, open_questions, talking_points}` where every claim carries a `chunk_id` citation.

**completeness_node** — The check is grounding-quality, not section-presence. It counts findings that include a `chunk_id` citation and computes a `citation_rate`. The loop fires when:
- `findings_total == 0` (LLM produced no useful output) on iteration ≤ 1, OR
- `citation_rate < _CITATION_RATE_THRESHOLD` (default `0.5` — more than half of findings are uncited / probably hallucinated) on iteration ≤ 1.
- On iteration > 1 with the same weak signal, the loop is capped and `loop_capped_at_iteration_limit` is recorded in `audit_trail` so a downstream consumer can tell "judged sufficient" from "we ran out of retries."

The threshold is a named constant in `graph/nodes.py` so it can be retuned without re-reading the loop logic. This replaces an earlier `total_findings < 2` heuristic that never fired in practice (LLMs reliably return ≥ 2 findings even on garbage input). Citation rate is the real grounding signal.

### Conditional loop

```
check_completeness
      │
      ├── is_sufficient OR iteration_count >= 3 → generate_brief → END
      │
      └── else → query_rewrite_node (loop with refined sub-queries
                 driven by information_gaps from this pass)
```

The hard cap of `iteration_count >= 3` in `workflow.py` is a belt-and-braces safety net. The completeness node only *adds* a gap (which is what triggers another loop) when `iteration_count <= 1`, so in practice the loop fires at most once and the workflow runs **at most two passes** through retrieve+reason. The third-iteration guard exists only to bound a misbehaving variant of the completeness logic and is not reached on the current rule set.

### Brief generator (`output/brief_generator.py`)

Resolves each `chunk_id` → source doc name + passage text (first 200 chars). Strips the contextual-retrieval `[Context: ...]` prefix from displayed passages so users see the real source text, not the LLM-generated context sentence (the prefix is preserved in `page_content` because the reasoning LLM benefits from it).

Runs three hallucination layers (see "Hallucination detection" below). Computes the faithfulness score. Returns the complete brief dict with:

- `summary`, `issues[]`, `risks[]`, `open_questions[]`, `talking_points[]` — analyst output with chunk citations
- `sources[]` — files referenced by the brief
- `faithfulness_score` — claim grounding ratio (0–1)
- `suspicious_facts[]` — Layer-1 regex catches
- `suspicious_claims[]` — Layer-2/3 catches with `caught_by: "regex" | "llm_judge"`
- `judge_status` — `ok` / `parse_error` / `error` / `skipped_breaker_open` / `no_claims` / `no_context_all_unsupported` / `disabled` — distinguishes "judge ran cleanly" from silent skips
- `verification_stats` — counts of verified/flagged/judged claims for transparency
- `loop_count` — iterations the workflow ran

---

## Retrieval pipeline

HybridRetriever runs four stages on every query:

### Stage 1 — Dense vector search
Chroma cosine similarity on `nomic-embed-text-v1.5` embeddings, filtered by `user_id`. Top-10 by semantic similarity. Fails on exact keyword queries where wording doesn't match.

### Stage 2 — BM25 keyword search
Fetches all child chunks for the user from Chroma's raw collection API, builds `BM25Okapi` in memory, scores and returns top-10. Excels where dense search fails: exact model numbers, priority codes, version strings.

### Stage 3 — Reciprocal Rank Fusion
Merges two ranked lists: `score = Σ 1/(k + rank)` with k=60. Documents in both lists receive a combined boost.

### Stage 4 — Cross-encoder reranking
`cross-encoder/ms-marco-MiniLM-L-6-v2` scores every `(query, chunk)` pair jointly. Returns top-4 by actual relevance. Too slow for initial retrieval; applied only to the merged short list.

After reranking, `fetch_parents()` exchanges the top-4 child chunks for their parent chunks (larger context windows), which are what the LLM sees.

### Optional: contextual retrieval (Anthropic, Sep 2024)

Behind `CONTEXTUAL_RETRIEVAL=1`, ingestion calls the LLM once per child chunk to generate a 1-2 sentence context describing where the chunk sits in its document. The context is prepended to the chunk's `page_content` before embedding, so cross-chunk references ("the fix", "this customer") survive into the vector. The chunk's `metadata` carries `has_context_prefix: True` and `context_text: "<the LLM output>"`.

At query time:
- The retrieved chunk's contextualized `page_content` is what gets passed to the reasoning LLM (it sees the context, which improves disambiguation)
- The brief generator's `_strip_context_prefix` removes the `[Context: ...]\n\n` header before showing passages to end users — the user-facing citation is the raw source text

A `_warn_if_mixed_contextual` check runs at ingest: if the workspace already contains chunks with a different `ingest_contextual_retrieval` flag value, a WARN is logged. Mixed-flag workspaces have inconsistent vector spaces, so the safest recovery is to wipe and re-ingest the whole workspace.

The cost-of-ingest tradeoff (~5s per chunk per LLM call) is documented in `ImprovementsForProd.md` (#14 — "Contextual retrieval batching") as future work.

---

## Embedding model design

### Lazy singleton + eager warmup
`get_embedder()` loads `nomic-embed-text-v1.5` (440 MB) on first call only. GPU detection via `torch.cuda.is_available()`. `_LazyEmbedder(Embeddings)` wraps it so the Chroma vectorstore can be initialized at import time without loading the model.

To avoid the cold-load penalty hitting the first user request, FastAPI's lifespan calls `warmup_models()` before reporting the API ready. This runs one no-op `embed_query("warmup")` and one `predict([("warmup", "warmup")])` on the cross-encoder, forcing both models into memory. Failures are caught and degrade to lazy loading rather than crashing the API (so a network blip during model download doesn't make the service permanently unhealthy).

### LRU cache
`@functools.lru_cache(maxsize=512)` on `embed_cached(text: str) -> tuple`. Repeated questions (or confidence scoring on the same chunks) skip model compute. Returns `tuple` because lists are not hashable.

---

## Confidence scoring (faithfulness metric)

```
1. Split answer on [.!?] into claim sentences (ignore <= 20 chars)
2. Embed each sentence via embed_cached()
3. Embed first 300 chars of each retrieved chunk via embed_cached()
4. Sentence is "grounded" if cosine similarity to any chunk >= _FAITHFULNESS_THRESHOLD
5. score = grounded / total_sentences
```

`_FAITHFULNESS_THRESHOLD` defaults to `0.65` (calibrated against the eval set for <5% false-positive grounding rate) and is overridable via the `FAITHFULNESS_THRESHOLD` env var.

Edge cases: "don't have enough information" → 0.0. No docs → 0.0. No sentences → 0.0.
Responses with score < `CONFIDENCE_THRESHOLD` (default `0.4`) are marked `escalated=True`.

---

## Hallucination detection (three layers)

Each layer catches a different kind of unfaithfulness. Layers are independent — running all three is cheap because Layer 1 is regex, Layer 2 is a fast classifier, and Layer 3 is one batched LLM call.

### Layer 1 — Atomic-fact regex (`detect_hallucination`)

Ten regex patterns applied to all generated text (see `_FACT_PATTERNS` in `langchain_utils.py`):

- Dates — `\b\d{1,4}[-/]\d{1,2}[-/]\d{2,4}\b` (e.g. `2024-09-25`, `9/25/2024`)
- Currency — `\$[\d,]+\.?\d*[KkMm]?` (e.g. `$28,000`, `$4.5M`)
- Percentages — `\b\d+\.?\d*\s*%`
- Quantities with time/size units — hours, days, weeks, months, years, minutes, seconds, ms, KB/MB/GB/TB
- Version numbers — `v?\d+\.\d+(?:\.\d+)*` (e.g. `v2.4.1`)
- Priority labels — `P0`–`P4`
- Severity labels — `Sev 2`, `SEV-1`, etc.
- Ticket / case IDs — `[A-Z]{2,5}-\d{3,6}` (e.g. `TICK-4521`, `INC-0892`)
- Cloud regions — `us-east-2`, `eu-west-1`, etc.
- Large quantity + unit — `50,000 records`, `12,000 users`, etc.

Matches not found verbatim in concatenated context text are returned as `suspicious_facts`. Catches "the fix is in v2.4" when the docs only mention v2.3. Cannot catch named-entity swaps ("Sarah Park" vs "Sarah Chen") or relational inversions ("X blocked Y" vs "X enabled Y").

### Layer 2 — Claim classifier (`classify_claims`)

Routes whole claims into one of three buckets:
- **`verified_by_regex`** — every fact in the claim was already verified by Layer 1
- **`flagged_by_regex`** — Layer 1 found at least one ungrounded fact in the claim → automatically flagged as suspicious without sending to Layer 3
- **`needs_judge`** — claim has no atomic facts to regex-check (relational, narrative, named-entity); needs Layer 3

This is the dispatcher that prevents Layer 3 from doing redundant work on regex-verifiable claims.

### Layer 3 — LLM-as-judge (`llm_judge_claims`)

One batched LLM call sends all `needs_judge` claims plus the retrieved context, asks the LLM to verdict each as `supported` or `unsupported` with a reason. Catches:
- Named-entity swaps ("Sarah Park" not in context; context mentions "Sarah Chen") → unsupported
- Relational inversions ("X blocked Y" when context says "X enabled Y") → unsupported
- Faithful paraphrases (no new entities) → supported

Returns a structured dict: `{"unsupported": [{claim, reason, judge_verdict}], "status": ok | parse_error | error | skipped_breaker_open | no_claims | no_context_all_unsupported}`. The status is propagated to `BriefResponse.judge_status` so callers can detect when Layer 3 silently failed (rate limit, breaker open, JSON parse error) instead of treating an empty unsupported-list as "verified clean."

### Composition

```
Brief generated → list of all claim texts
        ↓
Layer 1 detect_hallucination → suspicious_facts[]
        ↓
Layer 2 classify_claims → verified / flagged / needs_judge
        ↓
Layer 3 llm_judge_claims (only on needs_judge) → unsupported claims + status
        ↓
suspicious_claims[] = Layer 2 flagged + Layer 3 unsupported
   (each carries caught_by: "regex" | "llm_judge")
judge_status = Layer 3 status (or "disabled" if ENABLE_LLM_JUDGE=0)
```

Caveat: Layer 3 uses an LLM to judge another LLM's output, so correlated errors are possible. A production-grade system would use cross-model consensus (e.g., judge with a different vendor's model) or human-in-the-loop approval for high-stakes briefs.

---

## Circuit breaker

Three-state machine in `langchain_utils.py`, protected by `threading.Lock`:

```
CLOSED (normal)
    │  5 failures
    ▼
OPEN (fail-fast, 503 returned immediately)
    │  30 seconds elapsed
    ▼
HALF_OPEN (one probe allowed)
    │  success → CLOSED    │  failure → OPEN
```

Every LLM node in the graph checks `llm_breaker.is_open()` before calling the LLM. State exposed in `/health`. The breaker is provider-agnostic — `_llm_invoke_with_retry` (single shared implementation in `langchain_utils.py`) recognizes 429 / quota / `retry_delay` shapes from Groq, Gemini, and OpenAI-compatible providers.

---

## Rate limiting and concurrency control

Three independent mechanisms protect the system from runaway load and abuse:

### 1. Per-IP request rate limits (slowapi)

`slowapi.Limiter(key_func=get_remote_address)` is wired to FastAPI in `main.py`. The `@_limit(rate)` decorator applies to write endpoints and the auth flow:

| Endpoint | Limit | Reason |
|----------|-------|--------|
| `POST /auth/token` | 5/minute | Slows brute-force passkey guessing |
| `POST /brief` | 10/minute | Caps LLM cost per IP |
| `POST /upload-doc` | 10/minute | Caps embedding work and disk pressure |
| `POST /answer-questionnaire` | 2/minute | Bulk endpoint — limit is per-call (each call up to 200 rows) |

When the limit is exceeded, slowapi raises `RateLimitExceeded`, handled by `_rate_limit_exceeded_handler` which returns 429 with a `Retry-After` header. The limiter is keyed by the remote IP address, which means a single user can't share their quota across IPs but also can't be limited per user — that gap is documented in `ImprovementsForProd.md` (#7).

The `_limit()` wrapper degrades to a no-op decorator if `slowapi` isn't installed, so the API runs in dev environments without the dependency. In production, slowapi is required.

### 2. In-process concurrency cap on the bulk endpoint

`POST /answer-questionnaire` accepts CSV uploads up to `MAX_QUESTIONNAIRE_ROWS = 200`. Each row runs the full LangGraph workflow concurrently via `asyncio.gather()`, but `QUESTIONNAIRE_SEMAPHORE = asyncio.Semaphore(5)` caps active workflows at 5. This prevents a 200-row CSV from spawning 200 simultaneous LLM calls and tripping the LLM provider's rate limit.

Per-row failures return a fallback result `{question, answer: "[error: ...]", confidence: 0.0}` rather than aborting the whole batch.

### 3. Upload-payload validation

`POST /upload-doc` validates two things before the bytes hit Chroma:
- **File extension** — must be `.pdf`, `.docx`, `.html`, `.txt`, or `.json`. Rejects with 400 otherwise.
- **File size** — `MAX_FILE_SIZE_MB = 10`. Rejects with 400 otherwise. (HTTP-purist note: 413 would be more semantically correct but the existing handler returns 400 for both validation errors so clients have one error path.)

These are server-side guards against accidental gigabyte uploads exhausting the embedding pipeline. The 10MB cap is conservative — `ImprovementsForProd.md` (#2) notes that real enterprise documents (annual reports, technical specs) exceed this.

### 4. Circuit breaker (LLM-side)

Detailed in the "Circuit breaker" section above. Different from rate limiting — protects against *downstream* failure rather than upstream abuse.

---

## Token usage logging and cost telemetry

When `TOKEN_LOGGING=1` is set, every LLM call writes one JSON line to `api/data/token_usage.jsonl`:

```json
{"call": "reason", "timestamp": 1777..., "prompt_tokens": 1537,
 "completion_tokens": 3159, "tiktoken_estimate": 1383, "prompt_length_chars": 6841}
```

Field semantics:
- `call` — one of `query_rewrite` / `reason` / `llm_judge`
- `prompt_tokens` / `completion_tokens` — read from the LLM response's standardized `usage_metadata` (Gemini, Anthropic) or fallback `response_metadata.token_usage` (Groq/OpenAI). Provider-agnostic.
- `tiktoken_estimate` — `cl100k_base` tokenizer applied to the same prompt text. Lets `exp6b` compute estimator error against real reported usage.
- `prompt_length_chars` — raw character count, useful for distinguishing context-heavy from query-heavy calls.

Off by default (production cost is zero). Enabled per-experiment by `start_api({"TOKEN_LOGGING": "1"})` in `exp6b_real_cost.py`, which clears the log first so it captures only that run's calls.

`exp6b_real_cost.py` reads this file post-run and computes:
- per-call-type averages
- total cost at configurable input/output rates (`LLM_INPUT_PRICE_PER_M` / `LLM_OUTPUT_PRICE_PER_M`)
- monthly projections at 10/100/1k/10k queries per day
- side-by-side comparison against `exp6_cost.py`'s pre-call tiktoken estimate

The 7× gap between `exp6` (estimate) and `exp6b` (real) on this corpus is documented in the README "Results" section as evidence that Gemini 2.5-flash's thinking-token output dominates billing.

---

## Bulk questionnaire

`POST /answer-questionnaire` constraints (covered above in "Rate limiting and concurrency control"):
- `MAX_QUESTIONNAIRE_ROWS = 200` — hard cap before any LLM calls
- `QUESTIONNAIRE_SEMAPHORE = asyncio.Semaphore(5)` — at most 5 concurrent workflow invocations
- `@_limit("2/minute")` — per-IP rate limit
- `dependencies=[Depends(verify_api_key)]` — API-key auth required

Each row runs the full LangGraph workflow via `run_workflow()`. Failures in individual rows return a fallback result and do not block other rows.

---

## Error handling and graceful degradation

The system has three categories of failure that each have explicit handling rather than relying on caller heuristics:

### LLM rate-limit / quota failures

`_llm_invoke_with_retry` (in `langchain_utils.py`) wraps every LLM call. It recognizes 429 / quota / `retry_delay` shapes from Groq (`"try again in 6.5s"`), Gemini (`ResourceExhausted`, `retry_delay { seconds: 20 }`), and OpenAI-compatible providers. Sleeps for the provider-suggested delay (parsed from the error message) or 10s default, then retries up to 3 times. After exhaustion, re-raises — and the outer `llm_breaker` catches the failure to update the circuit.

### LLM output failures (silent corruption)

When the LLM returns a malformed JSON brief, the reason node catches the parse error and returns a sentinel: `{"_parse_error": True, "open_questions": ["Could not parse analyst output: ..."], ...}`. The completeness node detects `_parse_error` and skips the loop (a parse error is a system failure, not an information gap — looping won't help). The brief generator passes the parse error through, and `judge_status` on `BriefResponse` is `"parse_error"`.

The eval scripts (`eval_simple.py`, `faithfulness_eval.py`) detect this exact shape and convert it to an explicit error rather than counting it as `faithfulness=0.0` — so `error_rate` in result JSONs reflects real failure count, not silently-degraded scores.

### Verification step skipped (judge unavailable)

If `llm_breaker.is_open()` when the LLM judge is about to run, `llm_judge_claims` short-circuits and returns `{"unsupported": [], "status": "skipped_breaker_open"}`. The `BriefResponse.judge_status` field surfaces this so the Streamlit UI can render an amber banner (verification didn't run) instead of a green "Verified" chip (verification ran cleanly). Same for `parse_error`, `error`, `no_claims`, `no_context_all_unsupported`. Each status corresponds to a distinct UX outcome.

### Warmup failure

If `warmup_models()` fails during lifespan startup (no internet, HF Hub unreachable, disk full), the failure is caught and logged as a warning. The API still starts; the embedding model lazy-loads on first request instead. Without this guard, a transient network issue would leave `/health` permanently returning 503, making the service appear broken when it's just slow to warm.

### Retrieval-layer degradations (logged loudly, never silent)

Two retrieval paths previously caught their own exceptions and returned empty results without telling anyone. Both now log explicitly so operators can distinguish "no data" from "broken data path":

- **BM25 unavailable** — `bm25_search` returns `[]` and logs `bm25_unavailable` if `rank_bm25` isn't installed (graceful degradation to dense-only retrieval). Returns `[]` and logs `bm25_search_chroma_failed` with the user_id and exception if Chroma's raw collection API throws (system failure — operators should investigate).
- **Parent fetch fallback** — `fetch_parents` returns the original child chunks if `parent_store` doesn't have the requested IDs or throws an exception. Three log lines distinguish the cases: `fetch_parents_skipped` (no user_id provided — safety guard), `fetch_parents_empty_result` (parent IDs didn't resolve — workspace may have been wiped between child upload and query), `fetch_parents_failed` (parent_store exception — health-check signal).

In all of these, the brief still completes (with smaller-than-ideal context). The point of the logs is that the quality regression is now diagnosable instead of being misattributed to "the question was just hard."

### Auth: missing dependency vs. invalid token

`_create_token` does NOT swallow `ImportError`. If `jose` is missing from the environment, the import error propagates and `/auth/token` returns 500 — surfacing the dependency issue immediately. The previous behavior (returning an empty string) made `/auth/token` look like it succeeded; the next request would fail with a confusing 401 from `_decode_token`. Letting the import fail loudly keeps "missing dependency" and "invalid token" as distinguishable failure modes.

---

## Multi-tenancy and data isolation

Every piece of data is tagged with `user_id = sha256(workspace:passkey)[:32]`:
- ChromaDB: every chunk has `metadata["user_id"]`; all queries filter with `where={"user_id": ...}`
- SQLite: all queries include `WHERE user_id = ?`
- Delete: filters on both `file_id` and `user_id` to prevent cross-workspace deletion

`db_utils.get_all_documents` and `get_query_stats` raise `ValueError` if `user_id` is `None` or empty — closes a footgun where a missing argument silently fell back to the `"default"` tenant and returned another workspace's data.

---

## Document lifecycle

```
Upload (POST /upload-doc)
    ↓
Validate extension (.pdf .docx .html .txt .json) and size (≤ 10MB)
    ↓
Insert SQLite row in document_store → returns file_id
    ↓
Write file to a temp path on disk
    ↓
_warn_if_mixed_contextual(user_id) — warn if existing chunks were ingested
    with a different CONTEXTUAL_RETRIEVAL flag value
    ↓
load_and_split_document(temp_path) — format-aware loader produces raw chunks
    ↓
Parent-child split (full mode) or flat (baseline/sentence)
    ↓
[optional] _contextualize_chunks (if CONTEXTUAL_RETRIEVAL=1)
    ↓
Stamp ingest_contextual_retrieval flag onto every chunk's metadata
    ↓
Embed and store in ChromaDB (child_chunks + parent_chunks collections)
    ↓
Delete temp file in finally block
```

Delete (`POST /delete-doc`):
1. Resolve `user_id` from the JWT via `Depends(get_current_user)` — the `user_id` field in the request body is **not** trusted (and is no longer accepted in `DeleteFileRequest`). Earlier this endpoint took `user_id` from the request body, which let any API-key holder delete another tenant's documents by guessing `file_id` and supplying a target `user_id`.
2. Look up document by `(file_id, user_id)` — 404 if not found OR not owned by the JWT-derived workspace
3. Delete chunks from `vectorstore._collection` (child_chunks) with `where={$and: [{file_id}, {user_id}]}`
4. Delete chunks from `parent_store._collection` with the same filter
5. Delete the row from `document_store`

The compound filter on `(file_id, user_id)` enforces the same isolation in storage that the JWT enforces at the endpoint boundary.

---

## Streaming endpoint (`POST /chat/stream`)

Legacy stub kept for clients that want progressive output. Internally:

1. Same auth and breaker check as `/brief`
2. Runs the full LangGraph workflow synchronously (no token-level streaming yet — workflow doesn't expose intermediate state)
3. Streams the brief sections as plain-text lines: `summary`, `Issue: ...`, `Risk: ...`, `Talking point: ...`
4. Final line is a sentinel `---META---` followed by JSON metadata: `session_id`, `confidence`, `sources`, `escalated`

This is *not* token-level streaming — the workflow still blocks for ~30s. Real streaming (where the user sees the brief assemble incrementally) is a planned feature documented in `ImprovementsForProd.md` (#13), and would require replumbing the graph nodes to emit progress events.

---

## Schema migrations

`api/migrations/` contains numbered SQL files (`001_*.sql`, `002_*.sql`, ...). On startup, `db_utils.run_migrations()`:

1. Creates a `schema_migrations(name TEXT PRIMARY KEY, applied_at TIMESTAMP)` table if absent
2. Globs `*.sql` from the migrations directory, sorted alphabetically
3. For each file not yet recorded, runs `executescript()` (handles multi-statement SQL files) and inserts the filename into `schema_migrations`

The runner is idempotent — re-running on an already-migrated database is a no-op. Adding a new migration is just dropping a new numbered file in the directory; the next startup applies it.

Currently applied:
- `001_initial_schema.sql` — creates `application_logs` and `document_store`
- `002_add_user_id.sql` — adds `user_id TEXT DEFAULT 'default'` to both
- `003_add_indexes.sql` — composite indexes for the read-heavy queries
- `004_add_brief_logs.sql` — adds `brief_logs` table for `/brief` endpoint history

---

## Observability

| Endpoint | Data | Use case |
|----------|------|----------|
| `/audit-log` | `application_logs` rows: query, answer, confidence, escalated, sources | Review what was told to a customer |
| `/analytics` | Aggregated: total queries, escalation count, avg confidence, recent questions, low-confidence questions | Identify knowledge gaps |
| `/logs` | Last N JSON lines from `app.log`, filterable by level | Diagnose operational errors |
| `/health` | SQLite + Chroma connectivity, `GOOGLE_API_KEY` presence, `llm_breaker` state. Returns 200 only after `warmup_models()` completes | Service health check |

---

## Measured behavior (Gemini 2.5-flash, 6 sample docs, 20 golden queries)

The numbers below come from running the experiment kit end-to-end. Full JSONs in `experiment_kit/eval_results/`; concise narrative in `README.md` "Results" section.

### Where time goes per query

A median brief takes ~31.5 seconds. The graph nodes break down as:

| Node | p50 | Share |
|------|-----|-------|
| reason | 13.8s | 46% |
| llm_judge | 6.7s | 22% |
| faithfulness | 4.0s | 13% |
| retrieve | 2.9s | 10% |
| query_rewrite | 1.8s | 9% |
| completeness | <1ms | 0% |

LLM calls dominate (77% of total). The hybrid retrieval stack — dense + BM25 + RRF + cross-encoder + parent fetch — runs in ~3 seconds, well within budget for the value it provides (perfect recall@5 in `full` mode).

### What chunking strategy actually moves

Across the three chunking modes, the quality metrics (faithfulness, semantic similarity, coverage) are within ~4% of each other. What differs is *retrieval depth*: `full` (parent-child) gets recall@5 = 1.00 and MRR = 0.704, vs 0.95 / 0.633 for `baseline` and 0.947 / 0.658 for `sentence`. `full` is also the fastest (p50 = 34s vs 41s) because the LLM ingests fewer, larger chunks.

### What retrieval strategy actually moves

Holding chunking constant (`full`), retrieval mode trades off three things:
- **dense alone** — best MRR (0.767) but lower coverage (0.457). Precise, narrow.
- **dense + BM25** — best faithfulness (0.823) and tightest tail latency (p95 = 52s). BM25 surfaces verbatim keyword matches the LLM can't dispute.
- **full (with reranker)** — perfect recall@5 (1.00), best coverage (0.517). Pays p95 latency (277s on outlier queries) for breadth.

For a brief generator that prefers over-citing to missing material, `full` is the right tradeoff.

### What the loop actually does

`avg_loops = 1.0` in both `single_pass` and `loop` modes on the 20-query eval set. The citation-rate trigger never fires because Gemini's first-pass output reliably produces ≥ 50% chunk_id-cited findings. The loop is wired correctly (the audit_trail confirms the completeness check ran and judged the output sufficient); it's simply not needed for this LLM on this corpus. On a weaker model or a harder dataset, it would activate.

### Where the cost actually goes

- Tiktoken estimate (input + schema-based output): **$0.0025 / query** at Gemini 2.5-flash pricing
- Real reported usage: **$0.0170 / query** — 7× higher

The gap is **Gemini 2.5-flash's thinking tokens**. By default the model emits internal reasoning tokens (uncapped) before its visible output. They count toward billing but never appear in the response. Per-call output:

| Call | Visible JSON tokens (estimated) | Real reported output |
|------|--------------------------------|----------------------|
| query_rewrite | 40 | 920 |
| reason | 500 | 3,159 |
| llm_judge | 200 | 2,143 |

Setting `thinking_config={"thinking_budget": 0}` on `ChatGoogleGenerativeAI` would cap or eliminate this overhead. Documented as `ImprovementsForProd.md` #21 with a projected 5–7× cost reduction; quality impact unmeasured.

### Hallucination layers in practice

exp4 covers five contrived cases (A–E) targeting: atomic-fact lies (regex catches A and B), named-entity swap (judge catches C), relational inversion (judge catches D), and faithful paraphrase (judge correctly approves E). The script asserts both that the right layer triggered AND that no other layer false-positived.

In real briefs, the most common path is Layer 1 catching nothing (the regex finds no atomic facts), Layer 2 routing most claims to "needs_judge", and Layer 3 verdicting them as supported. The `judge_status` field on `BriefResponse` distinguishes `ok` (judge ran cleanly), `no_claims` (Layer 1+2 verified everything; nothing reached Layer 3), and the various failure modes (`parse_error`, `error`, `skipped_breaker_open`) that previously looked identical to a clean bill of health.

---

## Data model

### `application_logs`
```sql
id           INTEGER PK AUTOINCREMENT
session_id   TEXT
user_id      TEXT
user_query   TEXT
gpt_response TEXT
model        TEXT
confidence   REAL    DEFAULT 0.0
escalated    INTEGER DEFAULT 0
sources      TEXT    DEFAULT ''
created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
```

### `document_store`
```sql
id               INTEGER PK AUTOINCREMENT
filename         TEXT
user_id          TEXT
upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
```

### `brief_logs`
```sql
id                 INTEGER PK AUTOINCREMENT
customer_id        TEXT
query              TEXT
brief_json         TEXT
faithfulness_score REAL    DEFAULT 0.0
loop_count         INTEGER DEFAULT 0
created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
```

### ChromaDB collections
**`child_chunks`** (embedded, queried by vector search):
- `page_content` — small chunk text (500 chars). When `CONTEXTUAL_RETRIEVAL=1` was set at ingest, this is prefixed with `[Context: <LLM-generated summary>]\n\n` before the original text
- `metadata.file_id`, `metadata.user_id`, `metadata.source`
- `metadata.parent_chunk_id` — ID of the corresponding parent chunk
- `metadata.chunk_id` — stable citation ID assigned at retrieval time
- `metadata.doc_type` — transcript / ticket / pdf / html / docx
- `metadata.has_context_prefix` — boolean, set when contextual retrieval was used at ingest. Triggers prefix-stripping at citation-display time
- `metadata.context_text` — the LLM-generated context sentence (kept for transparency / debugging)
- `metadata.ingest_contextual_retrieval` — boolean stamped on every chunk, used by `_warn_if_mixed_contextual` to detect mixed-flag workspaces

**`parent_chunks`** (stored, fetched by ID only):
- `page_content` — larger context window (1600 chars)
- `metadata.file_id`, `metadata.user_id`, `metadata.source`
- `metadata.parent_chunk_id` — matches ID used by children

---

## Technology stack

| Component | Library / Service | Version |
|-----------|------------------|---------|
| API framework | FastAPI | 0.115.0 |
| Workflow orchestration | LangGraph | ≥ 0.2.0 |
| LLM | Google Gemini 2.5-flash (default; set via `LLM_MODEL` env var) | — |
| LLM client | langchain-google-genai | ≥ 2.0.0 |
| Embedding model | nomic-ai/nomic-embed-text-v1.5 | — |
| Embedding library | langchain-huggingface + sentence-transformers | 3.0.1 |
| Vector store | ChromaDB (two collections) | ≥ 0.5.5 |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | via sentence-transformers |
| Keyword search | rank-bm25 | 0.2.2 |
| Chain framework | LangChain | 0.3.x |
| PDF parsing | pymupdf | 1.24.9 |
| Sentence splitting | spaCy en_core_web_sm | ≥ 3.7.0 |
| DOCX parsing | python-docx | 1.1.2 |
| JWT | python-jose[cryptography] | 3.3.0 |
| Rate limiting | slowapi | 0.1.9 |
| Database | SQLite (via stdlib sqlite3) | — |
| Frontend | Streamlit | — |
| HTTP client | httpx (Slack), requests (Streamlit) | — |
