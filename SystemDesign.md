# System Design

This system ingests raw customer documents and makes them queryable in a way that traces every claim to a specific source chunk. The architectural principle is that structured metadata decisions — is_overdue, is_stale, health_score — are computed deterministically from field values and never delegated to an LLM.

---

## Document Ingestion

**Format-specific strategy registry** — Every supported file extension maps to an `IngestStrategy` (parser + chunker + parent/child flag) in `_FULL_STRATEGY` in `chroma_utils.py`. The failure mode this prevents: a generic "split at 800 chars" approach loses speaker boundaries in transcripts and merges unrelated tickets into a single chunk, making retrieval noisy.

**Parent-child chunking for PDF/DOCX/HTML** — Each document produces two chunk levels. Parent chunks (~2400 chars, 200-char overlap) are stored in ChromaDB but not embedded. Child chunks (~800 chars, 150-char overlap) are embedded and used for retrieval. The parent_chunk_id is stored as metadata on each child. At query time, children are retrieved by similarity, then parents are fetched by ID to give the LLM more surrounding context. Short chunks recover retrieval precision; returning the parent recovers context.

**Flat chunking for transcripts and tickets** — Transcripts chunk at speaker-turn boundaries, not character counts. Tickets produce one chunk per section (summary, description, comments). Applying parent-child to these would create parents that span multiple speakers or multiple tickets, which degrades both retrieval precision and LLM coherence.

**Commitment fields stamped at ingest** — `is_overdue`, `is_slipped`, `is_open` are computed in Python from `target_date`, `status`, and `actual_completion_date` before the chunk is written to ChromaDB. They are embedded into the chunk text as human-readable tokens ("OVERDUE by 14 days") so both semantic and keyword search can surface them. The failure mode this prevents: an LLM asked at query time "is this overdue?" will sometimes agree when it shouldn't, and sometimes miss it when it should.

**Version management** — `set_latest_version_flag()` runs after each successful upload and marks previous docs with the same (customer_id, doc_type, filename) combination as `is_latest_version=0` in both ChromaDB metadata and SQLite. Old chunks remain in the collection — version-aware queries filter with `where={"is_latest_version": 1}` — but they remain available for explicit historical queries. Scope is intentionally narrow: superseding is per (customer, doc_type, filename), not per doc_type globally, so uploading a new QBR deck doesn't suppress a different QBR deck with a different filename.

---

## Retrieval

**Hybrid retriever** — Every query runs BM25 (rank_bm25) and dense cosine (ChromaDB, hnsw:space=cosine) independently with k=10 each, then merges results with Reciprocal Rank Fusion (k=60). RRF rewards consistent high placement across both rankers rather than raw score magnitude, which means a result that appears at rank 3 in both lists beats one that appears at rank 1 in only one. The RRF identity key is `(file_id, parent_chunk_id)`, not a content prefix, because contextual retrieval mode prepends LLM-generated context that changes chunk text without changing identity.

**Cross-encoder reranker** — ms-marco-MiniLM-L-6-v2 rescores the top RRF candidates and selects top_k=6. Running it on the full candidate set is expensive; running it only on the merged top-20 candidates gives a reasonable recall/latency tradeoff. The reranker loads on first use and runs on CPU.

**Diversity cap** — Maximum 2 chunks per source document after reranking. Without this, a single verbose document dominates the context window and blocks out other sources that might contradict or qualify the answer.

**Structured metadata path** — Commitment and ticket queries call `structured_metadata_retrieve()`, which issues direct ChromaDB `where` filter queries instead of embedding lookups. For "show all overdue commitments for customer X", this is faster and more precise than semantic search, which would miss commitments whose text doesn't happen to contain the word "overdue."

**Recency boost** — Optional per-query scoring blend: `score = 0.4 × semantic_score + 0.6 × recency_weight`. Applied only when the query has temporal intent signals. Disabled by default.

**Query rewrite** — A query classifier decides whether the input is FOCUSED (single fact) or BROAD (synthesis across topics). Broad queries are decomposed into 2–4 sub-queries and retrieved independently before merging. Queries with overdue/past-due/slipped intent get an additional sub-query targeting the "OVERDUE by N days" tokens baked into commitment chunks at ingest.

**BM25 cache** — Keyed by user_id, invalidated when doc_count changes. The BM25 index is rebuilt from all chunks for a user on the first query after invalidation. At large scale (~10,000+ chunks), rebuild adds noticeable latency on the first post-upload query.

---

## LangGraph Workflows

**Three compiled graph singletons** — `_pre_meeting_workflow`, `_exec_1on1_workflow`, and `_query_workflow` are built once at module import and reused for every request. Rebuilding the graph per request adds ~50–100ms and gains nothing; the graphs are stateless between invocations.

**Fan-out / fan-in in the pre-meeting brief** — Seven section nodes (overdue_commitments, account_summary, open_items, recent_changes, outstanding_commitments, anticipated_questions, fetch_corpus_health) run in parallel. They converge into `posture_node`, which reads all their outputs before generating directives. `GraphState` uses `Annotated[List, operator.add]` accumulators for `stale_warnings`, `conflicts_raw`, and `audit_trail` so parallel writes don't race.

**`@_safe` decorator** — Every section node is wrapped so any exception writes `section_status[node_name] = "unavailable"` to state and returns a partial result. The brief assembler generates output from whatever sections succeeded. The failure mode this prevents: one bad LLM response or one missing doc type silently killing the entire brief and returning nothing.

**Posture node constraints** — The posture node is given a constrained vocabulary: verbs must be Lead, Acknowledge, Defer, or Push. Each directive must name the specific ticket or commitment that drives it. Pydantic validates the output; any verb outside the set is rejected. This prevents vague directives ("be careful," "stay flexible") that give no actionable guidance.

---

## Output Generation

**Hallucination detection pipeline** — `detect_hallucination()` runs regex over nine atomic fact patterns (dates, percentages, dollar amounts, version strings, SLA values, ticket IDs). `classify_claims()` routes each detected claim to one of three buckets: `verified_by_regex` (the value appears verbatim in a retrieved chunk), `flagged_by_regex` (detected but not found in source), or `needs_judge` (requires LLM evaluation). `llm_judge_claims()` runs only on the `needs_judge` bucket in a single batched call. Answer status starts at `ok` and downgrades to `partial` on any flagged or unresolved claim.

**Citation validation** — The answer generator receives both the LLM's response and the set of chunk_ids that were actually retrieved. Any chunk_id cited in the answer that was not in the retrieved set is logged as a forged citation and treated as a hallucination signal. The LLM cannot successfully cite a document it was not given.

**Exec brief evidence gate** — `exec_brief_generator.py` drops any section item where `source.document == "unknown"`. This handles cases where the LLM produces a statement with no traceable source document — the section is left empty rather than showing a fabricated claim.

**Corpus warning** — If the corpus is empty (no documents uploaded) or entirely stale (all doc dates past their per-type threshold), `brief_generator.py` emits a `corpus_warning` string that appears at the top of the brief. The brief is still generated from whatever is available; the warning surfaces the limitation rather than suppressing the output.

**Faithfulness scoring** — `calculate_faithfulness()` computes cosine similarity between answer sentences and source chunk embeddings. Thresholds differ by embedding model: 0.45 for OpenAI embeddings, 0.65 for HuggingFace (the HuggingFace model has a smaller representation space, so scores are naturally higher for unrelated content and the threshold compensates).

---

## Safety and Tenant Isolation

Every write and read in the system passes through `_require_user_id()` in `db_utils.py`. The function raises `ValueError` before any query executes if the user_id parameter is missing. ChromaDB embedding and retrieval functions take `user_id` as a required parameter and include it in every `where` clause.

There is no admin-level query that returns cross-tenant data. The `/customers` list endpoint returns only customers created by the requesting user_id. Brief and query endpoints infer customer ownership from the JWT before invoking any workflow.

ChromaDB collections are shared across all tenants — one collection per chunk layer, not one collection per tenant — with user_id as a metadata field. A misconfigured `where` clause could leak data across tenants. The safeguard is `_require_user_id()` raising before any query reaches ChromaDB if the parameter is absent.

Auth uses HS256 JWT with configurable `JWT_SECRET`. The UI mints a token on first load using `DEMO_WORKSPACE` and `DEMO_PASSKEY` env vars (both default to "demo") and caches it in session state. For non-demo deployments, override these env vars. The API also accepts `X-API-Key` header as an alternative to JWT for service-to-service calls.

---

## Operational

**SQLite configuration** — WAL mode with NORMAL synchronous. Each API process opens its own connection via `get_db()`. Suitable for ~50 concurrent readers with a single writer. Higher write concurrency produces lock contention, not data corruption. The fix is replacing `get_db()` with an async Postgres pool.

**Circuit breaker state** — Stored in a module-level object (`llm_breaker` in `langchain_utils.py`). The breaker is per-process, not shared across workers. Under multi-process deployment, each process independently tracks failure state. This is acceptable: the goal is to stop hammering a rate-limited API, and each process will independently detect the condition within its own request stream.

**APScheduler** — Runs in-process within the FastAPI lifespan handler. Fires the overdue commitment digest daily at 08:00 UTC via Slack webhook. If the process restarts before 08:00, the next scheduled firing runs from the new start time. There is no persistent job store; a restart within the same day means the digest fires again at the next 08:00.

**Brief cache** — In-process dict, 30-minute TTL, thread-safe. Invalidated by `invalidate_customer()` on upload or when corpus changes. Under multi-process deployment, each process caches independently — a brief cached by process A is not visible to process B.

---

## Limitations

**SQLite write concurrency** — Single writer. More than ~50 concurrent writes will produce lock contention. Postgres with an async pool is the path forward for production scale.

**File size** — 10MB per upload, enforced in `main.py`. Large PDF slide decks that exceed this limit are rejected at the API boundary.

**Rate limiting** — 20 uploads per minute per user, enforced by slowapi. Brief and query endpoints are not rate-limited separately.

**BM25 memory** — The BM25 index for a user is held in memory until invalidated. At 10,000 chunks × ~50 bytes average, this is ~500KB per user. Under many concurrent users on a single process, memory pressure accumulates.

**LLM dependency** — Defaults to `gemini-2.5-flash`. No offline fallback. If `GOOGLE_API_KEY` is missing or the Gemini API is down, all LLM-dependent section nodes return their `@_safe` fallback (empty result). The circuit breaker prevents prolonged hammering, not the absence of a response.

**Embedding model switching** — If `OPENAI_API_KEY` is set, the system uses `text-embedding-3-small` (1536 dims). Otherwise it uses `sentence-transformers/all-MiniLM-L6-v2` (384 dims) loaded locally. Switching the embedder after initial ingestion produces nonsensical similarity scores because the existing vectors were generated with a different model. Re-index from scratch after switching.

**Reranker cold start** — ms-marco-MiniLM-L-6-v2 loads on first use and runs on CPU. The first query that triggers the reranker adds ~2–3 seconds.

**In-process caches** — Brief cache, BM25 cache, and embedder cache are not shared across processes or containers. Horizontal scaling requires externalizing state (Redis or equivalent).

**Contextual retrieval cost** — `CONTEXTUAL_RETRIEVAL=1` prepends an LLM-generated context prefix to each child chunk before embedding. This adds one LLM call per chunk at ingest, increasing ingest cost by roughly 10x for large documents. Off by default.
