# Production Readiness

## Current state

The system is a working single-node deployment with a LangGraph-based brief generation pipeline. It handles document ingestion (PDF/DOCX/HTML/TXT/JSON), parent-child hybrid retrieval, structured brief generation, JWT auth, rate limiting, circuit breaking, hallucination detection, and structured observability. It works well for a small team or a single FDE on one machine.

The gaps below are what separate it from a multi-tenant, high-availability production service.

---

## What was already hardened

| Was | Is now |
|-----|--------|
| Confidence = response length × doc count | `calculate_faithfulness()`: grounded claims / total claims via cosine similarity |
| No hallucination awareness | `detect_hallucination()` flags ungrounded facts, logged as WARNING and included in brief |
| Cosine similarity retrieval only | BM25 + dense merged with RRF + cross-encoder reranking |
| Single retrieval query | `query_rewrite_node` decomposes queries into 2–4 sub-queries |
| Fixed-size chunking for all document types | Format-aware: pymupdf + spaCy sentences (PDF), header-split (HTML), heading-grouped (DOCX), speaker-turn (TXT), section-split (JSON tickets) |
| Single flat Chroma collection | Parent-child collections: children are retrieved by embedding; parents provide larger context to LLM |
| Embedding model loaded at import (every worker) | Lazy singleton `get_embedder()`, loads once on first call |
| No query embedding cache | `@lru_cache(maxsize=512)` on `embed_cached()` |
| Fixed 2s sleep retry on LLM failures | `CircuitBreaker` (CLOSED/OPEN/HALF_OPEN), fail-fast when open. Single shared `_llm_invoke_with_retry` recognizes 429 / quota / `retry_delay` shapes from Groq, Gemini, OpenAI |
| Linear RAG chain: retrieve → answer | LangGraph `StateGraph`: rewrite → retrieve → reason → completeness check → loop or brief |
| Free-text Q&A answer | Structured analyst brief: issues[], risks[], open_questions[], talking_points[] with chunk_id citations |
| `user_id` claimed by client (honour system) | JWT signed by server, `get_current_user` dependency |
| No auth on upload/delete/questionnaire | `dependencies=[Depends(verify_api_key)]` on all three |
| Startup migration inside a function body (never ran) | `lifespan()` context manager at module level |
| Bulk endpoint: unlimited rows, no concurrency cap | `MAX_QUESTIONNAIRE_ROWS=200`, `Semaphore(5)`, `@limit("2/minute")` |
| Eval measured only generation quality | Recall@5, MRR, chunk-level precision/recall, faithfulness eval script |

### Most-recent hardening pass

The session that produced the current state addressed a number of correctness and operational issues that aren't reflected in the older row pairs above:

| Was | Is now |
|-----|--------|
| LLM provider locked to Groq + llama-3.1-8b | Provider-agnostic with `LLM_MODEL` env var as single source of truth (default `gemini-2.5-flash`). All three call types (query_rewrite, reason, llm_judge) share the same model |
| `_llm_invoke_with_retry` duplicated across `nodes.py` and `langchain_utils.py` | Consolidated into `langchain_utils._llm_invoke_with_retry`, imported from one location |
| `groq_breaker` variable name (misleading after LLM swap) | Renamed to `llm_breaker` across `main.py`, `nodes.py`, `langchain_utils.py`, `tests/` |
| Token log fields named `groq_prompt_tokens` regardless of actual provider | Generic `prompt_tokens` / `completion_tokens`; reader normalizes both schemas for backward compat |
| `llm_judge_claims` returns empty list on failure → caller sees pristine brief | Returns `{"unsupported": [...], "status": "ok"/"parse_error"/"error"/"skipped_breaker_open"/"no_claims"/"no_context_all_unsupported"}`; `judge_status` surfaced as explicit field on `BriefResponse` |
| Streamlit silently shows incomplete briefs when verification fails | Amber/red banner when `judge_status` is degraded; green "Verified" chip when judge ran cleanly |
| Embedding model loads lazily on first upload (5+ min hang behind silent timeout) | `warmup_models()` called from FastAPI lifespan; `/health` only returns ready after warmup. Failures degrade to lazy-load with a logged warning rather than crashing the API |
| Completeness loop checked only that all sections exist (always passed → loop never fired, avg_loops=1.0) | Citation-rate check: loops back when more than half the LLM's findings lack chunk_id citations on the first pass — a real grounding signal. Loop cap surfaces in audit_trail |
| `eval_simple.py` and `faithfulness_eval.py` silently counted rate-limit failures as faithfulness=0.0 | Detect the `"Could not analyze: ..."` / `"Could not parse analyst output"` shape and treat as explicit errors, surfacing in `error_rate` |
| `db_utils.get_all_documents` / `get_query_stats` defaulted to `user_id="default"` when called with `None` | Both now raise `ValueError` for missing user_id — closes a multi-tenant footgun |
| `BriefResponse` had `brief: Dict[str, Any]` only — `judge_status` could silently drop | Explicit `judge_status: str` field on `BriefResponse`; documented values in the schema |
| Mixed-embedding workspaces silently degrade retrieval quality | `_warn_if_mixed_contextual` checks the contextual-retrieval flag on existing chunks at ingest and warns when it differs; every chunk now carries `ingest_contextual_retrieval` metadata |
| Contextual retrieval (Anthropic, Sep 2024) absent | `_contextualize_chunks` implemented and unit-tested (3 tests covering happy path, LLM failure, empty response). Gated by `CONTEXTUAL_RETRIEVAL=1` env var. See "Remaining gaps" below for the cost-of-ingest issue still pending |
| `assert_workspace_ready` absent — exp2/exp3 silently ran on partial workspaces | Health check fails fast if `eval_full` doesn't have all expected docs |
| `_create_token` swallowed `ImportError` and returned `""` — clients got a "200 OK" with an empty token, then `_decode_token` raised on the next request | The ImportError now propagates so missing-`jose` surfaces at first call instead of as a confusing auth-flow failure later |
| `bm25_search` had a bare `except Exception: return []` on Chroma failure — caller couldn't distinguish "no BM25 results" (legitimate) from "vector store is broken" (system failure) | Two distinct exception paths log explicitly: `bm25_unavailable` (rank_bm25 missing) and `bm25_search_chroma_failed` (Chroma exception). Empty list is preserved for graceful degradation, but operators see the cause |
| `fetch_parents` had a bare `except Exception: pass` and silently fell back to child chunks on parent_store failure | Three failure paths now log with context: missing user_id, empty parent result, parent_store exception. Falling back to children is still the safe behavior, but the logs surface the degraded state instead of pretending it's normal |
| `notification_utils` had hardcoded confidence labels (`0.7`, `0.4`) that drifted from `main.CONFIDENCE_THRESHOLD` | Lifted to `_CONFIDENCE_HIGH` and `_CONFIDENCE_ESCALATION` module constants with a comment linking to the corresponding escalation threshold |
| `completeness_node` used a magic `0.5` for the citation-rate loop trigger | Lifted to `_CITATION_RATE_THRESHOLD = 0.5` at module scope with a docstring explaining the calibration rationale and pointer to `ImprovementsForProd.md` #23 for tuning context |
| `db_utils.run_migrations` and `chroma_utils` used bare `print()` for status output — bypassing the structured logger | Replaced with `_log.info(...)` so output appears in `app.log` alongside the rest of the structured events. Module-level loggers now declared in both files |

---

## Remaining gaps for production

### 1. PostgreSQL instead of SQLite

SQLite uses file-level locking. Under concurrent writes from multiple workers or API replicas it serializes all writes. For a single FDE it is fine; for a team with concurrent queries it becomes a bottleneck.

**What to change:**
- Replace `sqlite3` in `db_utils.py` with `asyncpg` or SQLAlchemy with async PostgreSQL driver
- Move connection pooling to SQLAlchemy connection pool
- Migrations stay the same structure
- Add `DATABASE_URL` env var

**Why now:** Any horizontal scaling (multiple uvicorn workers, Docker replicas) requires this.

---

### 2. Async document processing with a job queue

Document uploads are synchronous: the HTTP request blocks until the entire ingestion pipeline (parse → noise filter → sentence chunk → embed → Chroma insert) completes. For large PDFs this can take 30–60 seconds.

**What to change:**
```
POST /upload-doc
      ↓
Write file to object storage (S3 / GCS)
Insert document_store row with status="pending"
Push job_id to queue (Redis / SQS)
Return 202 Accepted with job_id
      ↓
Worker process consumes queue
      ↓
Parse + chunk + embed + Chroma insert
Update document_store status="indexed"
```
Add `GET /upload-status/{job_id}` so the UI can poll. Streamlit shows a spinner until `"indexed"`.

---

### 3. Distributed vector store

ChromaDB persists to a local directory. Two API replicas cannot share the same index, and the index is lost if the container's disk is wiped.

**What to change:**
- Replace `Chroma` with Pinecone, Qdrant, or Weaviate (all have LangChain integrations)
- Or run ChromaDB in standalone server mode with a persistent volume (simpler migration path)
- Update `get_retriever_for_user` and the parent store lookup to use the new store

**Why now:** Any deployment beyond a single container requires this.

---

### 4. Persistent BM25 index

`bm25_search()` re-fetches all user chunks from Chroma and rebuilds the BM25 index on every query. For a user with 500 chunks this is ~10ms; for 50,000 chunks it becomes 100ms+ and CPU-intensive.

**What to change:**
- Maintain a per-user BM25 index in memory, rebuilt only when documents are added or deleted
- Or serialize with `pickle` to disk and load on startup
- Invalidate (set a dirty flag) in `index_document_to_chroma()` and `delete_doc_from_chroma()`

**Why later:** Acceptable for small document sets. Becomes a problem at scale.

---

### 5. Proper secret management

`JWT_SECRET` falls back to `secrets.token_hex(32)` if unset — all tokens are invalidated on every process restart. Users are silently logged out after a deploy.

**What to change:**
- Require `JWT_SECRET` and `API_KEY` in production (fail startup if absent)
- Rotate secrets via a secrets manager (AWS Secrets Manager, Vault, Kubernetes Secrets)
- Consider short-lived tokens (1 hour) with a refresh token flow for long-running sessions

---

### 6. Token refresh and session management

JWTs expire in 24 hours. The Streamlit app has no refresh logic — when the token expires, the user gets silent 401 errors until they switch workspaces to re-login.

**What to change:**
- Add `POST /auth/refresh` that accepts a still-valid token and returns a new one
- In Streamlit's `_headers()`, check token expiry and call refresh before it expires

---

### 7. Per-user rate limiting + monthly cost budgets

**Current state:** slowapi rate limits are wired on `/auth/token` (5/min), `/brief` (10/min), `/upload-doc` (10/min), and `/answer-questionnaire` (2/min). All are keyed by `get_remote_address` (the client IP). This is enough to slow brute-force attacks and accidental burst load. It's *not* enough to:

- Stop a determined client from rotating IPs (cloud functions, residential proxies)
- Cap *cumulative* spend per workspace — a workspace can hit 10 briefs/min × 60 × 24 × 30 = 432,000 briefs/month and burn ~$7,000 at current Gemini cost
- Distinguish a heavy-but-legitimate workspace from a runaway loop

**What to change:**
- Switch the rate-limiter key from `get_remote_address` to a function that derives `user_id` from the JWT (or falls back to IP for unauthenticated calls). Per-user limits replace per-IP limits where a JWT is present.
- Add per-user **monthly token budget** in SQLite (`user_budgets` table with columns `user_id`, `tokens_used_month_to_date`, `monthly_cap_tokens`, `reset_at`). Check before each `/brief` and decrement after. Reject with 429 + `Retry-After` once exhausted.
- Surface budget state in `/health` and a new `/me/quota` endpoint so the Streamlit UI can show a "X of Y queries used this month" indicator.

---

### 8. LangGraph persistence / checkpointing

The `StateGraph` currently runs statelessly — each `/brief` call starts a fresh graph with no memory of prior runs for the same customer.

**What to change:**
- Add a LangGraph checkpointer (SQLite or PostgreSQL) so partial graph states survive crashes
- Enable `thread_id`-based resumability for long-running brief generations
- This also enables graph replay for debugging

---

### 9. Replace honour-based Delete with ownership check via JWT — **DONE**

`POST /delete-doc` previously took `file_id` and `user_id` from the request body, allowing any API-key holder to delete another workspace's documents.

**What changed:**
- Removed `user_id` from `DeleteFileRequest`
- `delete_document()` now takes `user_id: str = Depends(get_current_user)` and uses the JWT-derived id for both the existence check and the SQL/Chroma deletes
- The same audit also hardened `get_current_user` to drop its `?user_id=` query-param fallback, which was the same class of IDOR
- Tests + eval scripts + Streamlit updated to use Bearer headers

---

### 10. Structured eval with gold chunk IDs

The current eval measures Recall@5 by filename. New chunk-level metrics (`chunk_precision_at_k`, `chunk_recall_at_k`) are wired in but require `gold_chunks` column in the eval CSV.

**What to change:**
- Build a golden dataset with `gold_chunks` column (exact Chroma document IDs that should be retrieved)
- Modify `/brief` to return chunk IDs in the response for eval tooling to consume
- This tells you whether reranking is selecting the right chunk within a document, not just the right document

---

### 11. Multi-worker embedding model

`get_embedder()` is a module-level singleton in a single process. When uvicorn spawns multiple workers (`--workers 4`), each process independently loads its own copy of the 440MB model.

**What to change:**
- Run the embedding model as a separate microservice (sentence-transformers REST server or Triton)
- All API workers call the embedding service over HTTP
- Single model instance, shared by all workers

**Why later:** Only matters when running multiple workers. Single-worker deployment is fine.

---

### 12. Slack notification improvements

Slack is currently notified on every query when the user enables it. In production this creates noise.

**What to change:**
- Only notify on specific triggers: faithfulness < threshold, hallucination detected, document indexing failure
- Batch low-confidence notifications into a digest rather than one message per query

---

### 13. Streaming brief responses

`/brief` blocks the client for 40-90 seconds while three sequential LLM calls (query_rewrite + reason + llm_judge) complete. From the user's perspective this looks frozen.

**What to change:**
- Convert `/brief` to a streaming endpoint that emits structured events: `query_rewrite_done`, `retrieved_n_chunks`, `reasoning_token`, `judge_started`, `brief_complete`
- Streamlit consumes the stream and renders progressively (sub-queries first, then chunks, then claims as they arrive)
- Doesn't reduce wall-clock time, but the perceived latency drops dramatically because the user sees activity within ~5s

**Why now:** This is the biggest user-facing UX gap. Anyone clicking "Generate brief" expects feedback within seconds.

---

### 14. Contextual retrieval — batch the per-chunk LLM calls

Anthropic's contextual retrieval method (implemented behind `CONTEXTUAL_RETRIEVAL=1` in `chroma_utils._contextualize_chunks`) makes one LLM call per child chunk at ingest time. For a 100-child document at ~5s/call this is ~10 minutes per document. The 6-doc sample corpus needs ~3 hours of ingest, which made same-day full-ablation evaluation infeasible during the most recent hardening pass.

**What to change:**
- Batch 5-10 chunks per LLM call: send the full document plus N chunks, ask for a JSON map of `{index: context_string}`
- Add JSON-parse fallback that falls through to the original chunk on per-batch failure (same robustness pattern as the current per-chunk implementation)
- Expected throughput improvement: ~5×, bringing full-corpus ingest from ~3 hours to ~30 minutes
- Then run an `exp1_contextual` ablation for a real before/after measurement of the technique's lift

**Why now:** Without batching, the feature is implemented but not measurable on this codebase's evaluation budget.

---

### 15. Observability — Sentry + metrics export + tracing

Current observability is structured JSON logs to a file. There is no error aggregation, no metrics export, no distributed tracing.

**What to change:**
- **Sentry** for error tracking — wraps every endpoint, captures stacktrace + request context, dedupes
- **Prometheus metrics** — `request_duration_seconds`, `llm_tokens_used_total`, `circuit_breaker_state`, `judge_status_total{status}` — exported on `/metrics` endpoint
- **OpenTelemetry tracing** for `/brief` — one span per graph node, propagated through the LLM client so you can see "judge took 30s of the 60s total"
- Once these exist, define SLOs: p95 brief latency < 90s, error_rate < 1%, faithfulness ≥ 0.7 on 95% of briefs

---

### 16. Backup and disaster recovery

`api/data/rag_app.db` (SQLite) and `api/data/chroma_db/` (vector store) are persisted to a single disk. There's no backup, no replication, no recovery procedure.

**What to change:**
- Daily snapshot of `api/data/` to S3 / GCS with 30-day retention
- Test restore: spin up a fresh container with the snapshot, verify briefs still work
- For PostgreSQL migration (gap #1), use managed snapshot/PITR features
- For distributed vector store (gap #3), the provider's snapshot tooling

---

### 17. Cost monitoring and alerting

Real LLM cost is captured in `experiment_kit/eval_results/exp6b_real_cost.json` for the eval set. There's no mechanism to track spending per workspace in production or alert on cost anomalies.

**What to change:**
- Persist token usage per `/brief` call to `application_logs` (already partially done — extend with provider + model + input/output token counts)
- Daily aggregation job → email digest "$X spent today, top 3 workspaces by cost"
- Alert when daily cost > 2x 7-day rolling average (flags a runaway loop or query bomb)

---

### 18. Concurrency safety on shared singletons

`get_embedder()`, `_get_cross_encoder()`, and `llm_breaker` are module-level singletons. `llm_breaker` is thread-safe (uses a `threading.Lock`), but the embedder and cross-encoder rely on the underlying libraries being thread-safe. Untested under concurrent /brief load.

**What to change:**
- Load test with 10 concurrent /brief requests, measure error rate and latency
- If race conditions surface, wrap the model calls in a process-level semaphore
- For the multi-worker scaling case (gap #11), externalize models to a separate service

---

### 19. Tokenizer mismatch for cost projections

`exp6_cost.py` uses `tiktoken` with `cl100k_base` encoding (an OpenAI/Llama tokenizer) to estimate token counts before calling Gemini. Gemini's tokenizer differs by ~10-20%, so cost projections from `exp6` are slightly inaccurate.

**What to change:**
- Use `google.generativeai`'s tokenizer for Gemini-specific counts
- Or accept the mismatch and document the bias in `exp6_cost.py`
- `exp6b_real_cost.py` is unaffected — it uses the LLM's reported token counts directly

---

### 20. Compliance and PII handling

- No DPA, privacy policy, or terms of service
- No PII detection / redaction in uploaded documents
- No GDPR cascade-delete: `DELETE /delete-doc` removes one document, but there's no `DELETE /workspace` that wipes a tenant's full data
- No data residency guarantees

**What to change:**
- Add `DELETE /workspace?user_id=X` that cascades through SQLite (`document_store`, `application_logs`, `brief_logs`) and Chroma (filter by `user_id`)
- Run uploads through a PII detector (Presidio or similar) and store a redacted copy
- Add legal artifacts (DPA template, privacy policy)

---

### 21. Cap Gemini thinking tokens (5–7× cost reduction)

`exp6b` measured real Gemini token usage at **$0.017/query** — 7× higher than `exp6`'s tiktoken estimate of $0.0025/query. The gap is Gemini 2.5-flash's "thinking tokens": internal reasoning tokens emitted before the visible output, billed at output rate but never appearing in the response.

Per-call breakdown from `exp6b`:

| Call | Visible JSON output | Real reported output | Thinking overhead |
|------|--------------------|----------------------|-------------------|
| query_rewrite | 40 tokens (the JSON array) | 920 tokens | 23× |
| reason | 500 tokens (the structured brief) | 3,159 tokens | 6× |
| llm_judge | 200 tokens (the verdicts) | 2,143 tokens | 11× |

**What to change:**
- Pass `thinking_config={"thinking_budget": 0}` (or a small cap like 200) when constructing `ChatGoogleGenerativeAI` in `langchain_utils._llm_invoke_with_retry`'s caller paths
- Re-run exp6b to measure the new cost
- Re-run exp1/exp3 to measure quality impact — disabling thinking may reduce faithfulness on hard queries

**Expected outcome:** real cost drops from $0.017 to ~$0.003 per query (close to the tiktoken estimate). At 1000 queries/day, that's $510 → $90 per month. The faithfulness delta is unknown; that's why this needs a measurement, not a code change in isolation.

---

### 22. Cache claim-sentence embeddings in faithfulness scorer

`exp5` shows the faithfulness scorer is 13% of total query latency (4 seconds, p50). It embeds every claim sentence from scratch on every brief, even when the same sentence appears across multiple queries (e.g., recurring talking points like "Meridian's enterprise tier is $28,000/month").

**What to change:**
- Wrap `calculate_faithfulness`'s claim-sentence embedding call with `embed_cached()` (the existing `lru_cache(maxsize=512)` helper)
- Or: cache claim embeddings in SQLite keyed by sha256(claim_text)

**Expected outcome:** faithfulness latency drops to <1s on repeat-claim briefs, no impact on first-time claims. Cumulative effect on a busy workspace: 3-second p50 reduction per query.

---

### 23. Tune the citation-rate completeness threshold

`exp3` shows `avg_loops = 1.0` in both `single_pass` and `loop` modes — the citation-rate trigger never fires on the eval set. Gemini's first-pass output reliably has citation_rate ≥ 0.5, so the loop is currently vestigial.

**What to change:**
- Bump the citation-rate threshold from 0.5 to 0.7 in `completeness_node` and re-run exp3 — see if the loop fires more often AND whether faithfulness improves
- Or: switch the trigger entirely to `faithfulness_score < 0.7` (computed inside `generate_brief_node` and surfaced into completeness on the next iteration). Faithfulness is a more direct quality signal than citation rate; the price is moving the score computation earlier in the graph

**Expected outcome:** if the loop genuinely helps, mean_faithfulness in `loop` mode pulls clearly ahead of `single_pass`. If it doesn't, the loop should be deleted from the graph rather than kept as cosmetic complexity.

---

## Priority order

| Priority | Change | Reason |
|----------|--------|--------|
| High | Cap Gemini thinking tokens (#21) | Measured 7× cost gap. Single config change projected to drop $510/mo → $90/mo at 1k queries/day |
| High | Streaming /brief responses (#13) | 30-40s blocking call is the biggest user-facing UX gap; streaming changes the perception even if wall time is the same |
| High | Per-user cost budgets (#7) | Without this a single bad client can run up an unbounded LLM bill |
| High | Sentry + metrics export (#15) | You can't operate what you can't see; today's only signal is `tail -f api/data/app.log` |
| High | PostgreSQL (#1) | Unblocks any horizontal scaling — SQLite serializes all writes |
| High | Backup + DR (#16) | One disk failure today = all customer data gone |
| High | Async document processing (#2) | Prevents request timeouts on large files / contextual retrieval |
| High | Require JWT_SECRET in prod (#5) | Security: tokens must survive restarts |
| Medium | Contextual retrieval batching (#14) | The feature exists; without batching it can't be measured on the current eval set |
| Medium | Distributed vector store (#3) | Required for multi-replica deployment |
| Medium | Cost monitoring + anomaly alerts (#17) | Catches runaway-loop incidents before the bill arrives |
| Medium | Compliance / GDPR cascade (#20) | Required for any enterprise customer |
| Medium | Token refresh (#6) | UX: users should not be silently logged out |
| Medium | Delete endpoint JWT-only user_id (#9) | Close remaining ownership bypass |
| Medium | Rate limiting on /brief and /upload-doc (#7) | Quota protection complement to cost budgets |
| Medium | LangGraph checkpointing (#8) | Crash recovery + debuggability |
| Medium | Cache claim-sentence embeddings (#22) | Measured 13% of query latency in faithfulness scoring; lru_cache wraps it cheaply |
| Medium | Tune citation-rate threshold (#23) | exp3 shows loop never fires; either tune or delete the loop edge |
| Low | Concurrency load test (#18) | Validate shared-singleton safety before multi-worker |
| Low | Persistent BM25 index (#4) | Performance at large scale |
| Low | Tokenizer accuracy (#19) | Cost projections off — measured but largely subsumed by #21 (the bigger gap) |
| Low | Slack trigger logic (#12) | Noise reduction |
| Low | Multi-worker embedding service (#11) | Only matters at high worker counts |
| Low | Gold chunk IDs in eval (#10) | Evaluation depth |
