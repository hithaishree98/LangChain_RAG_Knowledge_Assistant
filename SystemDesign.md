# System Design

This system ingests raw customer documents and makes every claim in every output traceable to a source chunk. Metadata decisions — is_overdue, health_score, is_stale — are computed from field values, never delegated to the LLM.

---

## Document Ingestion

**Format-specific strategy registry** — Every supported file extension maps to an `IngestStrategy` (parser + chunker + parent/child flag) in `_FULL_STRATEGY` in `chroma_utils.py`. The registry exists because document types genuinely need different treatment: a transcript chunked at 800 chars mid-sentence loses the speaker attribution on that fragment. A ticket CSV split at a character boundary mid-comment ends up with a chunk that has no status or priority field.

**Parent-child chunking for PDF/DOCX/HTML** — Each document produces two chunk levels. Parent chunks (~2400 chars, 200-char overlap) are stored in ChromaDB but not embedded. Child chunks (~800 chars, 150-char overlap) are embedded and used for retrieval. The parent_chunk_id is stored as metadata on each child. Consider a child chunk that says "we committed to fixing the login issue by Q2" — that's 12 words. It retrieves fine against "what did we commit to?" but gives the LLM no context. The parent includes the full conversation turn that led to that commitment, which is what actually answers the question.

**Flat chunking for transcripts and tickets** — Transcripts chunk at speaker-turn boundaries. Tickets produce one chunk per section (summary, description, comments). If you applied parent-child to these, the parents would span multiple speakers or multiple tickets — that makes retrieval noise, not signal.

**Commitment fields stamped at ingest** — `is_overdue`, `is_slipped`, `is_open` are computed in Python from `target_date`, `status`, and `actual_completion_date` before the chunk is written. They're also embedded as text into the chunk: "OVERDUE by 14 days." Without this, if you ask "what commitments are overdue?", the LLM sees 30 commitment chunks and has to judge each one. It will get some wrong. With the text already there, the answer is in the source — the LLM doesn't have to infer it.

**Version management** — `set_latest_version_flag()` runs after each upload and marks previous docs with the same (customer_id, doc_type, filename) as `is_latest_version=0` in both ChromaDB and SQLite. Old chunks stay in the collection — `where={"is_latest_version": 1}` filters them in fresh queries — but they remain available for historical lookups. Scope is intentionally per (customer, doc_type, filename), not per doc_type globally, so uploading a new QBR deck doesn't accidentally suppress a different QBR deck with a different filename.

---

## Retrieval

**Hybrid retriever** — Every query runs BM25 (rank_bm25) and dense cosine (ChromaDB, hnsw:space=cosine) independently with k=10 each, then merges with Reciprocal Rank Fusion (k=60). RRF rewards consistent high placement across both rankers — a result at rank 3 in both BM25 and dense beats one at rank 1 in only one. The identity key for deduplication is `(file_id, parent_chunk_id)`, not a content prefix. If you use `CONTEXTUAL_RETRIEVAL=1`, each chunk gets an LLM-generated context prefix prepended before embedding, which changes the text but not the identity — content-prefix dedup would treat those as different documents.

**Cross-encoder reranker** — ms-marco-MiniLM-L-6-v2 rescores the top RRF candidates and selects top_k=6. The reranker runs only on the merged top-20; top-20 was fine in practice and increasing it adds latency with minimal recall gain. It loads on first use and runs on CPU, so the first query adds ~2–3 seconds.

**Diversity cap** — After reranking, a hard cap of 2 chunks per source document is applied. Without this, one verbose document (a long transcript, a dense QBR deck) tends to fill the context window and crowd out other sources. That's a problem when you want the LLM to see both what the customer said in the call and what's in the ticket history.

**Structured metadata path** — Commitment and ticket queries bypass embeddings entirely. `structured_metadata_retrieve()` runs direct ChromaDB `where` filter queries on fields like `status`, `priority`, and `is_overdue`. For "show all overdue commitments for customer X", this is faster and more precise than semantic search — a commitment chunk that says "we're working on improving performance" won't surface just because "performance" appeared in both the query and the chunk.

**Query rewrite** — A classifier decides whether the query is FOCUSED (single fact) or BROAD (synthesis across topics). Broad queries get decomposed into 2–4 sub-queries retrieved independently before merging. Queries with overdue/past-due/slipped wording get an extra sub-query targeting the "OVERDUE by N days" tokens baked in at ingest.

---

## LangGraph Workflows

**Three compiled graph singletons** — `_pre_meeting_workflow`, `_exec_1on1_workflow`, and `_query_workflow` are built once at module import. Rebuilding per request adds ~50–100ms with no benefit since the graphs are stateless between invocations.

**Fan-out / fan-in in the pre-meeting brief** — Seven section nodes (overdue_commitments, account_summary, open_items, recent_changes, outstanding_commitments, anticipated_questions, fetch_corpus_health) run in parallel. They converge into `posture_node`, which reads all their outputs. `GraphState` uses `Annotated[List, operator.add]` accumulators for `stale_warnings`, `conflicts_raw`, and `audit_trail` so parallel writes don't race.

**`@_safe` decorator** — Every section node is wrapped with a decorator that catches any exception and writes `section_status[node_name] = "unavailable"` to state before returning a partial result. If the `open_items` LLM call fails — bad JSON from Gemini, a rate limit, anything — without `@_safe` that exception propagates and LangGraph stops. The whole brief returns nothing. With it, open_items shows as unavailable and the remaining six sections still generate normally.

**Posture node constraints** — The posture node is constrained to four verbs: Lead, Acknowledge, Defer, Push. Each directive must name the specific ticket or commitment that grounds it. Pydantic validates the output; any verb outside the set is rejected. This matters because a posture directive like "stay flexible" or "be cautious" is useless going into a customer call. The format forces specificity: "Acknowledge — we missed the Q1 SLA commitment (COMMIT-042)."

---

## Output Generation

**Hallucination detection pipeline** — `detect_hallucination()` runs regex over nine atomic fact patterns (dates, percentages, dollar amounts, version strings, SLA values, ticket IDs). `classify_claims()` routes each detected claim to `verified_by_regex` (the value appears verbatim in a source chunk), `flagged_by_regex` (detected but not found in source), or `needs_judge` (requires LLM evaluation). `llm_judge_claims()` runs on only the `needs_judge` bucket in a single batched call. Sending everything to the judge would be expensive and slow; the classifier routes only what can't be resolved by pattern matching.

**Citation validation** — The answer generator receives both the LLM's response and the actual set of chunk_ids that were retrieved. Any chunk_id in the answer that wasn't in the retrieved set is logged as a forged citation. The LLM cannot successfully cite a document it wasn't given context from.

**Exec brief evidence gate** — `exec_brief_generator.py` drops any item where `source.document == "unknown"`. If the LLM produces a statement with no traceable source document, the section is left empty. Better to show nothing than a fabricated claim.

**Corpus warning** — If the corpus is empty, the brief generator emits: "No documents uploaded for this customer. Upload a transcript, ticket export, and commitment tracker to generate a useful brief." If the corpus exists but is entirely stale, it emits a stale warning string listing the affected doc types. The brief still generates from whatever is available; the warning tells you what to trust and what to refresh.

---

## Safety and Tenant Isolation

Every write and read passes through `_require_user_id()` in `db_utils.py`. It raises `ValueError` before any query executes if the user_id is missing. ChromaDB retrieval functions take `user_id` as a required parameter and include it in every `where` clause.

ChromaDB collections are shared across all tenants — one collection per chunk layer, not one per tenant — with user_id as a metadata field. A misconfigured `where` clause could return cross-tenant results. The safeguard is that `_require_user_id()` prevents the query from running at all if the parameter is absent.

There is no admin endpoint that returns cross-tenant data. The `/customers` list returns only customers created by the requesting user_id. Brief and query endpoints verify customer ownership from the JWT before invoking any workflow.

---

## Operational

**SQLite** runs in WAL mode with NORMAL synchronous, which supports ~50 concurrent readers with a single writer. At higher write concurrency, you'll get lock contention, not data corruption.

**Circuit breaker** (`llm_breaker` in `langchain_utils.py`) is per-process, not shared across workers. Under multi-process deployment, each process tracks failure state independently. That's acceptable for the use case — the goal is to stop hammering a rate-limited API, and each process will detect the condition from its own request stream.

**APScheduler** runs in-process within the FastAPI lifespan. The overdue commitment digest fires daily at 08:00 UTC. If the process restarts after midnight and before 08:00, the digest fires again at the next 08:00 — there's no persistent job store, so a restart doesn't skip the day's run.

**Brief cache** is an in-process dict with 30-minute TTL, invalidated by `invalidate_customer()` on upload. Under multi-container deployment, each container caches independently. The same customer's brief might be regenerated across containers — not a correctness issue, just redundant LLM calls.

---

## Limitations

**SQLite write concurrency** — single writer, ~50 concurrent limit before lock contention. Worth noting for any deployment that expects concurrent uploads or brief generation.

**File size** — 10MB per upload, enforced in `main.py`. Large PDF slide decks will hit this.

**Upload rate limiting** — 20 uploads per minute per user via slowapi.

**Embedding model switching** — this is a genuine gotcha. If you indexed 1000 chunks with `sentence-transformers/all-MiniLM-L6-v2` and then added `OPENAI_API_KEY`, new uploads generate OpenAI vectors in the same ChromaDB collection. Querying with OpenAI embeddings against sentence-transformer vectors returns meaningless similarity scores. Re-index everything from scratch after switching.

**Reranker cold start** — ms-marco-MiniLM-L-6-v2 loads on first use and runs on CPU. First query adds ~2–3 seconds.

**BM25 memory** — index is per user_id, held in memory until invalidated. At 10,000 chunks × ~50 bytes average, that's ~500KB per user. Rebuilds on the first query after an upload.

**In-process caches** — brief cache, BM25 cache, and embedder cache are not shared across processes or containers. Horizontal scaling needs these externalized (Redis or equivalent).

**No offline LLM fallback** — if `GOOGLE_API_KEY` is missing or the Gemini API is unavailable, all LLM-dependent nodes return their `@_safe` empty result. The circuit breaker stops repeated retries, but the output will be sparse.
