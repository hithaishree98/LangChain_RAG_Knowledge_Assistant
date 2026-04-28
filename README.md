# FDE Knowledge Assistant

A RAG-powered pre-call intelligence tool for Forward Deployed Engineers. Upload PDFs, call transcripts, support tickets, DOCX, or HTML pages. Ask a question before a customer call and get back a structured brief — issues, risks, talking points, and open questions — with inline citations pointing to the exact source passage.

In customer-facing work, a confident wrong answer is worse than admitting uncertainty. This system is built around that constraint: every claim in a brief traces to a retrieved chunk, and suspicious facts are flagged before the brief is returned.

---

## Problem context

Before every customer call you need to know what was last discussed, what was promised, what is broken right now, and what their tech stack looks like. That information exists but it is scattered across Slack, email, Notion, Git, and personal notes.

This project started as a way to deeply learn RAG architecture, but it is built around a real FDE use case: upload your customer documents in whatever format they come in, ask what you need to know before the call, and get a structured brief you can read in 60 seconds.

---

## What it does

- Upload and index PDF, DOCX, HTML, TXT (transcripts), and JSON (support tickets) per workspace
- Format-aware chunking: PDFs use pymupdf + spaCy sentence boundaries, HTML splits on headers, DOCX chunks by heading structure, transcripts chunk by speaker turn groups, tickets chunk by section
- **Parent-child retrieval**: child chunks are embedded and retrieved; parent chunks (larger context windows) are fetched for the LLM to reason over
- **Contextual retrieval** available behind `CONTEXTUAL_RETRIEVAL=1`: an LLM generates a 1-2 sentence document-level context prepended to each chunk before embedding, so cross-chunk references ("the fix", "this customer") aren't lost. Implemented and unit-tested; gated behind a flag because per-chunk LLM calls add 5-10 minutes per document at ingest time
- **Hybrid retrieval**: dense vector search + BM25 keyword search merged with Reciprocal Rank Fusion, then cross-encoder reranked
- **LangGraph stateful workflow**: query decomposition → retrieval → structured reasoning → grounding-quality check → loop or generate brief (hard cap: 3 iterations)
- **Citation-rate-based completeness loop**: re-queries with refined sub-queries when more than half the LLM's findings lack chunk_id citations on the first pass — a real grounding signal, not just section-presence checks
- **Three-layer hallucination detection**: regex catches atomic fact lies (dates, amounts, version numbers); a classifier routes claims to the right verifier; an LLM judge verifies relational/named-entity claims that regex can't. Every brief carries a `judge_status` field so callers can distinguish "judge ran cleanly" from "judge silently failed"
- **Structured briefs**: LLM acts as analyst, returns JSON with issues, risks, open questions, talking points — each claim cited to a chunk ID
- Faithfulness-based confidence score on every brief (grounded claims / total claims)
- Streamlit UI surfaces verification state: a green "Verified" chip when the judge ran cleanly, an amber banner when verification was skipped or failed, a red banner when no supporting docs were found
- Bulk mode: upload a CSV of questions, get all answers back at once (useful for RFPs and security questionnaires)
- JWT-signed workspace tokens — user identity is server-verified, not client-claimed
- Rate limiting on bulk endpoints
- Full audit log of every query, answer, confidence score, and source
- Usage analytics per workspace: query counts, escalation rate, average confidence, knowledge gaps
- Circuit breaker on LLM calls — fail fast when the service is degraded
- Mixed-embedding workspace guard — warns if you ingest some docs with contextual retrieval and others without, since the resulting vector space is silently inconsistent
- Embedding-model warmup at API startup — `/health` only returns ready after the 440MB nomic-embed model is loaded, so first-request latency is bounded
- Optional Slack notifications
- Health check endpoint reporting database, vector store, LLM key, and circuit breaker state

---

## How it works

### Upload pipeline

```
User uploads PDF / DOCX / HTML / TXT / JSON
        ↓
Format-aware loader + doc_type routing:
  PDF        → pymupdf → noise_filter() → sentence_chunk(256 tokens, 64 overlap)
  HTML       → HTMLHeaderTextSplitter (h1/h2/h3) → sectioned text
  DOCX       → python-docx (heading structure) → semantic sections
  TXT        → transcript_parser → speaker-labelled Turn objects → transcript_chunker
  JSON       → ticket_parser → TicketStructure → ticket_chunker (one chunk per section)
        ↓
Parent chunks written to parent_chunks Chroma collection (not embedded, fetched by ID)
        ↓
Child chunks written to child_chunks Chroma collection (embedded, retrieved by vector search)
Each child chunk carries parent_chunk_id in metadata
        ↓
nomic-ai/nomic-embed-text-v1.5 embeds each child chunk
        ↓
Stored in ChromaDB (on disk) + SQLite document_store table
```

### Brief pipeline (LangGraph workflow)

```
User submits query → POST /brief
        ↓
JWT verified → customer_id extracted
        ↓
Circuit breaker (llm_breaker) checked → 503 if LLM is known-down
        ↓
LangGraph StateGraph starts:

  [query_rewrite_node]
  Gemini decomposes query into 2-4 targeted sub-queries.
  On a loop iteration, the prompt includes information_gaps from the
  previous pass so sub-queries are refined, not just regenerated.
        ↓
  [retrieve_node]
  For each sub-query, HybridRetriever runs:
    ├── Dense: child_chunks cosine similarity top-10 (filtered by customer_id)
    └── Sparse: BM25 top-10 (same user's chunks)
  Reciprocal Rank Fusion merges both ranked lists
  Cross-encoder (ms-marco-MiniLM-L-6-v2) reranks → top-4 child chunks
  fetch_parents() expands child chunks to parent chunks for richer context
        ↓
  [reason_node]
  Gemini 2.5-flash with analyst prompt: detect issues, risks, open
  questions, talking points. Returns structured JSON with chunk_id
  citation required on every claim.
        ↓
  [completeness_node]
  Counts findings that actually include a chunk_id citation. Loops back
  to query_rewrite if (citation_rate < 0.5) on the first pass — a real
  grounding signal, not just section-presence.
  → is_sufficient=True OR iteration_count >= 2 → generate_brief
  → else → back to query_rewrite_node (refinement loop, hard cap: 3 iterations)
  Audit trail records "loop_capped_at_iteration_limit" if the cap fires.
        ↓
  [generate_brief_node]
  Resolves chunk_id → source doc name + passage text. Strips any
  contextual-retrieval prefix from displayed passages. Runs the three
  hallucination layers:
    Layer 1 — regex on atomic facts (dates, amounts, version numbers)
    Layer 2 — classify_claims routes claims to the right verifier
    Layer 3 — Gemini llm_judge_claims verifies relational claims
  Computes faithfulness score (claim sentences vs. retrieved chunks).
        ↓
Brief returned with: summary, issues[], risks[], open_questions[],
  talking_points[], sources[], faithfulness_score, suspicious_facts[],
  suspicious_claims[], judge_status, verification_stats, loop_count,
  audit_trail
        ↓
Logged to SQLite brief_logs + application_logs
```

---

## Project structure

```
LangChain_RAG_Knowledge_Assistant-main/
├── api/
│   ├── main.py                 FastAPI app — all endpoints, auth, lifespan
│   ├── chroma_utils.py         Embeddings, hybrid retrieval, parent/child collections, chunking
│   ├── langchain_utils.py      CircuitBreaker, faithfulness scoring, hallucination detection
│   ├── db_utils.py             SQLite CRUD + migration runner
│   ├── pydantic_models.py      Request/response schemas (incl. BriefRequest, BriefResponse)
│   ├── notification_utils.py   Slack webhook
│   ├── graph/
│   │   ├── state.py            GraphState TypedDict — shared state between all nodes
│   │   ├── nodes.py            query_rewrite, retrieve, reason, completeness nodes
│   │   └── workflow.py         StateGraph wiring + conditional loop edge
│   ├── output/
│   │   └── brief_generator.py  Formats reasoning output into cited brief
│   ├── ingestion/
│   │   ├── transcript_parser.py  Parses Otter JSON + plain-text transcripts → List[Turn]
│   │   ├── transcript_chunker.py Groups turns into overlapping word-count chunks
│   │   ├── ticket_parser.py    Parses support ticket JSON → TicketStructure
│   │   └── ticket_chunker.py   One Document per ticket section
│   ├── migrations/
│   │   ├── 001_initial_schema.sql
│   │   ├── 002_add_user_id.sql
│   │   ├── 003_add_indexes.sql
│   │   └── 004_add_brief_logs.sql
│   ├── data/                   Created at runtime
│   │   ├── rag_app.db          SQLite database
│   │   ├── chroma_db/          Chroma vector store on disk (child_chunks + parent_chunks)
│   │   └── app.log             Structured JSON logs
│   └── requirements.txt
├── app/
│   ├── streamlit_app.py        Streamlit frontend — brief viewer, upload, bulk, analytics
│   ├── styles.py               HTML/CSS component helpers
│   └── requirements.txt
├── eval/
│   ├── eval_simple.py          Generation + retrieval quality eval (Recall@5, MRR, chunk P/R)
│   └── faithfulness_eval.py    Calls /brief on golden set, measures claim grounding
└── experiment_kit/
    ├── eval_set.csv             20-question golden eval set
    ├── eval_results/            JSON outputs from exp1-exp6b runs
    └── experiments/
        ├── api_utils.py         start_api / stop_api helpers + assert_workspace_ready guard
        ├── bootstrap_upload.py  Re-uploads sample docs to a given user_id
        ├── repopulate_eval_full.py  Quick eval_full reset for partial-state recovery
        ├── exp1_chunking.py     Chunking ablation (baseline / sentence / full)
        ├── exp2_retrieval.py    Retrieval ablation (dense / dense+BM25 / +reranker)
        ├── exp3_workflow_loop.py  Loop vs single_pass — measures lift from completeness loop
        ├── exp4_hallucination.py  9 assertions across the 3 hallucination layers
        ├── exp5_latency.py / exp5_analyze.py  Per-node latency breakdown
        ├── exp6_cost.py         Tiktoken-based cost estimate at Gemini 2.5-flash pricing
        └── exp6b_real_cost.py   Real cost from the LLM's reported token usage
```

---

## Setup

### Prerequisites

- Docker and Docker Compose

### Environment variables

Create a `.env` file in the project root:

```
# LLM key — get one free at https://aistudio.google.com/apikey
GOOGLE_API_KEY=your_google_api_key

# Optional — override the default LLM model (gemini-2.5-flash)
# LLM_MODEL=gemini-2.5-flash

# Optional — enable Anthropic-style contextual retrieval at ingest time.
# Adds 1 LLM call per chunk during upload (~5s/chunk on Gemini), so plan
# accordingly. See ImprovementsForProd.md for the cost / latency tradeoff.
# CONTEXTUAL_RETRIEVAL=1

# Optional — if not set, all endpoints are open (dev mode)
API_KEY=your_api_key

# Optional — if not set, a random secret is generated per process start
# (tokens will be invalidated on restart — set this in production)
JWT_SECRET=a_long_random_string

# Optional
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
ALLOWED_ORIGINS=http://localhost:8501
```

### Start with Docker

```bash
docker-compose up --build
```

- API: http://localhost:8000
- UI:  http://localhost:8501
- API docs: http://localhost:8000/docs

### Start without Docker

```bash
# Install spaCy sentence model (one-time)
pip install spacy
python -m spacy download en_core_web_sm

# API
cd api
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Streamlit (separate terminal)
cd app
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## Using the app

1. Open http://localhost:8501
2. Enter a workspace name and passkey (first time creates the workspace)
3. Upload documents from the sidebar — select the document type (auto / pdf / transcript / ticket)
4. In the **Brief** tab, type a pre-call query and click **Generate brief**
5. Expand issues, risks, and talking points to see the source passage that supports each claim
6. Use the **Bulk query** tab to upload a CSV with a `question` column for batch answering (max 200 rows)
7. Check the **Analytics** tab for usage stats and knowledge gaps
8. Check the **Audit log** tab to review every interaction

---

## API endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | /health | none | Service health and circuit breaker state |
| POST | /auth/token | none | Issue JWT for a workspace |
| POST | /brief | API key + JWT | Run LangGraph workflow, return structured brief |
| POST | /chat | API key | Legacy: prose answer (delegates to brief internally) |
| POST | /chat/stream | API key | Streaming tokens + metadata sentinel |
| POST | /upload-doc | API key | Upload and index a document (supports doc_type param) |
| GET | /list-docs | JWT | List documents in workspace |
| POST | /delete-doc | API key | Delete a document |
| GET | /analytics | API key | Query stats |
| GET | /audit-log | API key | Interaction history |
| GET | /logs | API key | Structured operational logs |
| POST | /answer-questionnaire | API key, 2/min rate limit | Bulk CSV answering |

---

## Running the evaluation harness

### Generation + retrieval eval

```bash
cd eval
python eval_simple.py
# With a chunking config label for comparison:
python eval_simple.py --config sentence_256
```

Required CSV columns: `question`, `reference_answer`, `key_facts` (semicolon-separated)
Optional: `gold_source` (filename — enables Recall@5 and MRR), `gold_chunks` (semicolon-separated chunk IDs — enables chunk-level precision/recall)

### Faithfulness eval (headline metric)

```bash
cd eval
python faithfulness_eval.py --user_id your_workspace_id
```

Calls `/brief` for each golden query. Target: mean faithfulness ≥ 0.90.

---

## Results (Gemini 2.5-flash, 6 sample docs, 20 golden questions)

All numbers below are from real runs of the experiment kit (`experiment_kit/experiments/`) with raw JSONs in `experiment_kit/eval_results/`. They reflect the post-hardening state of the codebase: provider-agnostic LLM client, citation-rate completeness loop, three-layer hallucination detection, model warmup at startup, and silent-failure detection in eval scripts.

### exp1 — Chunking ablation

Three chunking strategies indexed under separate workspaces (`eval_baseline`, `eval_sentence`, `eval_full`), evaluated with the full retrieval stack (dense + BM25 + reranker).

| Mode | semantic_sim | coverage | faithfulness | recall@5 | MRR | p50 | p95 | err_rate |
|------|--------------|----------|--------------|----------|-----|-----|-----|----------|
| baseline | 0.534 | 0.487 | 0.773 | 0.95 | 0.633 | 41s | 80s | 0.00 |
| sentence | 0.514 | 0.478 | 0.794 | 0.947 | 0.658 | 40s | 58s | 0.05 |
| **full** | 0.508 | **0.509** | 0.756 | **1.00** | **0.704** | **34s** | **52s** | 0.00 |

**Reading this:** The three strategies are within ~4% on quality metrics (chunking strategy alone doesn't move generation faithfulness much for this corpus), but `full` (parent-child) wins clearly on retrieval depth — perfect recall@5 and the highest MRR. It's also the *fastest* at p50 (34s) and p95 (52s), since 1600-char parent chunks are cheaper for the LLM to ingest than many 500-char children.

`sentence` mode hit one silent_llm_failure (5% error rate). The detection caught it instead of letting it slip through as a 0.0 faithfulness — exactly what the eval-script silent-failure guard exists for.

**Decision:** `full` is the production default. `baseline` and `sentence` remain in the codebase as ablation comparators.

### exp2 — Retrieval ablation (eval_full workspace)

Three retrieval configurations against the same indexed workspace:

| Mode | semantic_sim | coverage | faithfulness | recall@5 | MRR | p50 | p95 |
|------|--------------|----------|--------------|----------|-----|-----|-----|
| dense | 0.560 | 0.457 | 0.816 | 0.90 | **0.767** | 32s | 205s |
| dense + BM25 | 0.548 | 0.439 | **0.823** | 0.90 | 0.679 | 31s | **52s** |
| **full (dense + BM25 + cross-encoder rerank)** | 0.543 | **0.517** | 0.793 | **1.00** | 0.683 | 31s | 277s |

**Reading this:** Each mode optimizes for a different goal.
- **dense** is best when you want the *single* most-relevant doc fast (best MRR).
- **dense + BM25** has the highest faithfulness (BM25 finds verbatim keyword matches the LLM can't dispute) and the cleanest tail latency.
- **full** is best when *coverage* matters — perfect recall@5 means the right doc is always in the top-5, and the highest coverage (0.517) means the brief includes more facts. Pays a tail-latency cost (p95=277s) because the cross-encoder occasionally amplifies one slow query.

For a brief generator where you'd rather over-cite than miss material, `full` is correct. For a Q&A system where exact-match latency matters, `dense + BM25` is.

### exp3 — Workflow loop ablation

Compares single-pass vs the citation-rate-triggered loop (re-query if more than half of LLM findings lack chunk_id citations on iteration 1).

| Mode | mean faithfulness | pass_rate ≥ 0.90 | p50 | avg_loops |
|------|-------------------|------------------|-----|-----------|
| single_pass | 0.758 | 0.15 | 33s | 1.00 |
| loop | 0.732 | 0.20 | 34s | 1.00 |

**Reading this:** `avg_loops = 1.0` in *both* modes — the loop never fired on this eval set. That means Gemini's first-pass output reliably cleared the 50% citation-rate threshold, so the completeness check accepted it without re-querying. The loop is functioning correctly (the wiring is verified — see `audit_trail`); it's just that with this LLM and this eval set, the first pass is grounded enough.

The slight difference between modes (single_pass `mean=0.758`, loop `mean=0.732`) is within noise on a 20-query set. Loop mode's higher pass_rate (0.20 vs 0.15) is a difference of one query.

**Honest take:** the loop is implemented and observable (it would fire on a worse LLM or harder corpus), but on this configuration it's vestigial. Two follow-ups documented in `ImprovementsForProd.md`: tune the citation-rate threshold higher (0.7+) to make the loop more aggressive, or evaluate on a corpus where Gemini's first pass is weaker.

### exp4 — Hallucination detection

Tests the three verification layers (regex → classifier → LLM judge) against five contrived cases A-E that target different failure modes:

- **Case A** — atomic-fact lie (wrong dollar amount): regex catches.
- **Case B** — atomic-fact lie (wrong version number): regex catches.
- **Case C** — named-entity swap ("Sarah Park" instead of "Sarah Chen"): LLM judge catches.
- **Case D** — relational inversion ("X blocked Y" instead of "X enabled Y"): LLM judge catches.
- **Case E** — faithful paraphrase: LLM judge correctly approves.

Each case asserts both that the right layer triggered AND that no other layer false-positived. Run via `python experiment_kit/experiments/exp4_hallucination.py`. The script prints a PASS/FAIL counter to stdout (no JSON output today — exp4 is a unit-style assertion runner, not a metrics producer).

### exp5 — Per-node latency breakdown

20 queries against `eval_full` with `NODE_TIMING=1`. Aggregated from `api/data/node_timings.jsonl`:

| Node | Mean | p50 | p95 | Share |
|------|------|-----|-----|-------|
| reason | 14.4s | 13.8s | 19.6s | **46%** |
| llm_judge | 7.0s | 6.7s | 19.5s | 22% |
| faithfulness | 4.0s | 4.0s | 6.8s | 13% |
| retrieve | 3.2s | 2.9s | 5.9s | 10% |
| query_rewrite | 2.8s | 1.8s | 9.4s | 9% |
| completeness | <1ms | <1ms | <1ms | 0% |
| **Total per query** | **~31.5s** | | | |

**Reading this:** LLM time (reason + judge + query_rewrite) is **77% of total latency**. The custom retrieval stack — dense + BM25 + RRF + cross-encoder + parent fetch — is only **10%**. Critics who think hybrid retrieval is "overengineered for the latency cost" are looking at the wrong column.

The faithfulness scorer is unexpectedly heavy at 13% — it embeds every claim sentence from scratch, so the same claim across multiple queries pays the cost again. A small `lru_cache` on `embed_cached(claim_sentence)` could trim this — flagged in `ImprovementsForProd.md`.

### exp6 / exp6b — Cost (Gemini 2.5-flash pricing: $0.30/M input, $2.50/M output)

Two methodologies, deliberately compared.

**exp6 — tiktoken-based estimate.** Reconstructs the exact prompts the LLM would see and counts input tokens with `cl100k_base`. Estimates output from the JSON schema size:

| Metric | Value |
|--------|-------|
| avg_input_tokens / query | 2,124 |
| avg_output_tokens / query | 740 (estimated from schema) |
| avg_cost / query | **$0.0025** |
| Projected at 1,000 queries/day | $74.61/month |

**exp6b — real cost from Gemini's reported `usage_metadata`.** Same 10 queries, but token counts come from the LLM response instead of estimation:

| Metric | Value |
|--------|-------|
| total_input_tokens (10 queries × 3 calls) | 48,926 |
| total_output_tokens | 62,215 |
| **avg_cost / query** | **$0.0170** |
| Projected at 1,000 queries/day | $510.65/month |

**The gap is ~7×, and it's the most interesting finding from the cost study.**

The tiktoken estimate is correct on input (~2,200 tokens) but radically off on output. exp6b's per-call breakdown shows why:

| Call | exp6 estimated output | exp6b real output |
|------|----------------------|-------------------|
| query_rewrite | 40 (the JSON array of sub-queries) | 920 |
| reason | 500 (the structured analyst JSON) | 3,159 |
| llm_judge | 200 (the verdicts JSON) | 2,143 |

The visible output is what we estimated. The extra tokens are **internal "thinking" tokens** that Gemini 2.5-flash emits before its final answer. They're billed at output rate but never appear in the response content. By default the thinking budget is uncapped, so the model spends 80–95% of its output cost reasoning silently.

This is a real finding with a concrete fix: pass `thinking_config={"thinking_budget": 0}` (or a small cap) to `ChatGoogleGenerativeAI`. We project this would cut the real cost from $0.017/query to ~$0.003/query — close to the tiktoken estimate, with quality impact to be measured. Documented as #21 in `ImprovementsForProd.md`.

### What this means in aggregate

The codebase produces grounded, cited briefs at ~$0.017/query and ~30s wall time. Latency is dominated by LLM time, not retrieval. The three-layer hallucination guard runs on every brief and exposes its status to the caller via `judge_status`. The completeness loop is wired and observable but currently doesn't fire on this eval set — that's a calibration question, not a correctness bug. The biggest cheap win available is disabling Gemini's thinking-token budget; the second biggest is batching the contextual-retrieval LLM calls so an ablation against vanilla retrieval becomes feasible on this corpus.

---

## Technology choices

| Component | Choice | Reason |
|-----------|--------|--------|
| LLM | Google Gemini 2.5-flash | Generous free tier (1M TPM), cheap paid tier ($0.30/M in, $2.50/M out), competitive on structured-output tasks. Single source of truth (`langchain_utils.LLM_MODEL`) — overridable via `LLM_MODEL` env var |
| Embeddings | nomic-ai/nomic-embed-text-v1.5 | Strong semantic embedding, open weights, 8192-token context (fits parent chunks) |
| Vector store | ChromaDB (two collections) | Local persistence, metadata filtering. `child_chunks` collection for retrieval, `parent_chunks` for context lookup |
| Keyword search | rank-bm25 | Exact match retrieval for keywords dense embeddings miss |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Accurate relevance scoring after retrieval |
| Workflow orchestration | LangGraph 0.2+ | Stateful graph with conditional loops; supports the citation-rate completeness loop |
| Chain framework | LangChain 0.3.x | Embeddings, retrievers, document loaders, provider-agnostic `usage_metadata` |
| PDF parsing | pymupdf (fitz) | Fast, layout-aware, supports noise filtering |
| Sentence splitting | spaCy en_core_web_sm | Accurate sentence boundaries for chunking |
| API | FastAPI | Async support, Pydantic validation, lifespan startup for model warmup |
| UI | Streamlit | Rapid prototyping, brief viewer layout |
| Database | SQLite | Zero-dependency, sufficient for single-node |
| Auth | python-jose (JWT) | Signed workspace tokens, server-verified identity |
