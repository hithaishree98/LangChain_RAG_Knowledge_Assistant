# LangChain RAG Knowledge Assistant

Account managers go into customer calls without the right context. They rely on whoever took notes last quarter, stale slide decks, or a ten-minute skim that misses the thing the customer actually raised. The information exists — transcripts, ticket exports, commitment trackers, QBR decks — it's just scattered and nobody reads all of it.

I wanted to fix that but not by just handing everything to an LLM. "Summarize these documents" produces confident hallucinations. You can't walk into a customer meeting and cite "the model said so." I needed structured extraction, verifiable citations, and an explicit not-found signal when the evidence isn't there instead of a fabricated answer.

## Architecture

```
Documents (PDF · DOCX · HTML · TXT · JSON · CSV)
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  Ingestion                                           │
│  doc_type detection → format-specific parser         │
│  → chunker (parent/child or flat)                    │
│  → ChromaDB (child_chunks + parent_chunks)           │
│  → SQLite  (metadata, versions, corpus health)       │
└─────────────────────────────────────────────────────┘
                │
       ┌────────┴───────────────────────┐
       │                                │
    SQLite                        FastAPI :8000 ◄── Streamlit :8501
    (shared store)                      │
                                        └── LangGraph Workflows
                                              │
                                         ├─ Pre-Meeting Brief
                                         │   7 parallel section nodes
                                         │   → posture → generate_brief
                                         │
                                         ├─ Exec 1:1 Brief
                                         │   4 parallel section nodes
                                         │   → approach → generate_exec
                                         │
                                         └─ Query
                                             rewrite → retrieve
                                             → answer → generate_answer
                                                      │
                                               HybridRetriever
                                           BM25 + dense + RRF
                                           + cross-encoder rerank
```

## Features

**Pre-meeting brief** — one click generates overdue commitments, open tickets sorted by P0/P1/P2, recent changes, outstanding commitments, anticipated questions with verbatim source quotes, and a posture directive (Lead/Acknowledge/Defer/Push) tied to a specific ticket or commitment ID.

**Exec 1:1 brief** — for a named stakeholder: role and tenure, what they've said on record with sentiment tags, recent signals, open asks; sections with no evidence are dropped rather than fabricated.

**Query** — free-form question, rewritten into sub-queries if broad, runs hybrid retrieval, validates every cited chunk ID against what was actually retrieved, runs three-layer hallucination detection, returns `ok`/`partial`/`not_found` with an explanation.

**Account health** — 0–100 score from P0 ticket count, overdue commitments, commitment slip rate, and days since last call; no LLM; computed from metadata alone.

**Multi-tenant isolation** — SHA256(workspace:passkey)[:32] is the user_id; enforced at storage layer in every ChromaDB and SQLite query with no bypass path.

**Format-aware ingestion** — PDF/DOCX/HTML get parent-child chunking (2400-char parents, 800-char children); transcripts get turn-boundary chunks; tickets get section chunks; commitment trackers produce one chunk per commitment with `is_overdue`/`is_slipped`/`is_open` stamped from field values at ingest, not at query time.

**Hybrid retrieval** — BM25 + dense cosine + RRF (k=60) + cross-encoder reranker (top_k=6) + max-2-per-source diversity cap; commitment and ticket queries bypass embeddings via ChromaDB `where` filters.

**Hallucination detection** — regex over nine atomic fact patterns → claim classifier (routes to verified/flagged/needs_judge) → batched LLM judge for only the ambiguous claims; answer downgrades from `ok` to `partial` on any unresolved flag.

**Staleness thresholds** — commitment_tracker: 14d; transcripts/tickets/account_notes: 30d; QBR decks: 90d; solution architecture: 180d; stale sources are flagged in every citation.

**Circuit breaker** — 10 LLM failures open the breaker; 30-second recovery window; thread-safe; failures return structured partial results rather than 500s.

**Deterministic commitment fields** — `is_overdue`, `is_slipped`, `is_open` are computed at ingest from actual field values; the LLM cannot mark something overdue that the data doesn't support.

**Version tracking** — uploading a replacement marks the previous as superseded, scoped by (customer_id, doc_type, filename); old chunks stay indexed but are excluded from fresh queries via `is_latest_version` filter.

## Quick start

```bash
cp .env.example .env
# set GOOGLE_API_KEY and JWT_SECRET at minimum

docker compose up --build
# UI:  http://localhost:8501
# API: http://localhost:8000/docs
```

Seed demo data:

```bash
docker compose exec api python scripts/demo_seed.py
```

Optional env vars: `LLM_MODEL` (default: `gemini-2.5-flash`), `OPENAI_API_KEY` (switches embedder to `text-embedding-3-small`), `SLACK_WEBHOOK_URL` (daily overdue digest at 08:00 UTC), `CONTEXTUAL_RETRIEVAL=1` (prepends LLM-generated context to each chunk before embedding).

FastAPI · LangGraph · ChromaDB · SQLite · Gemini · sentence-transformers · rank-bm25 · Streamlit · APScheduler
