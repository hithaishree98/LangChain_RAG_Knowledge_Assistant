# LangChain RAG Knowledge Assistant

Account managers walk into customer calls without context. The notes are scattered — a ticket export, a transcript, a commitment tracker. I built this to pull it together without just handing everything to an LLM. If the answer isn't in the uploaded documents, it says so instead of making something up.

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

## What it generates

**Pre-meeting brief** — one click before a customer call gives you overdue commitments, open tickets sorted by P0/P1/P2, recent changes, outstanding commitments, anticipated questions with verbatim source quotes, and a posture directive (Lead/Acknowledge/Defer/Push) tied to a specific ticket or commitment ID. 

**Exec 1:1 brief** — same idea for a named stakeholder: what they've said on record (with sentiment tags), their open asks, recent signals, and recommended approach. Sections with no source evidence are dropped rather than filled with guesses.

**Query** — ask a free-form question about a customer, get an answer that cites which document it came from. Broad questions get decomposed into sub-queries automatically. Returns `ok`, `partial`, or `not_found` — `partial` means the answer was found but something didn't verify cleanly.

**Account health** — a 0–100 score from P0/P1 ticket counts, overdue commitments, commitment slip rate, and days since last call. Pure metadata arithmetic, no LLM call.

## How it works

**Multi-tenant isolation** — every workspace credential hashes to a user_id (SHA256(workspace:passkey)[:32]) stored on every record in ChromaDB and SQLite. Two customers can both have a ticket called "login issue." Without user_id in every `where` clause, a query would return both. The filter is at the storage layer, so a bug in the API layer can't accidentally bypass it.

**Format-aware ingestion** — documents chunk differently depending on their type. PDFs and Word docs get parent-child splits: 800-char children are embedded for retrieval, 2400-char parents are fetched by ID at query time to give the LLM the surrounding context. Transcripts chunk at speaker-turn boundaries — splitting mid-speaker loses who said what. Commitment trackers produce one chunk per commitment with `is_overdue`, `is_slipped`, and `is_open` computed from the actual field values and baked into the chunk text at ingest. The LLM never has to decide if something is overdue; it reads "OVERDUE by 14 days" directly in the source. Uploading a replacement file marks the old one as superseded, scoped by (customer_id, doc_type, filename).

**Hybrid retrieval** — queries run BM25 and dense cosine independently (k=10 each), merge with Reciprocal Rank Fusion, then rerank the top candidates with ms-marco-MiniLM-L-6-v2 (top_k=6). For commitment and ticket queries, embeddings are skipped entirely — ChromaDB `where` filters on status and priority fields are more precise than semantic search for that kind of lookup.

**Hallucination detection** — answers go through three layers: regex checks for atomic facts (dates, dollar amounts, version strings, SLA values), a classifier routes each claim to verified/flagged/needs_judge, then only the ambiguous ones go to the LLM judge in a single batched call. Any unresolved flag downgrades the answer from `ok` to `partial`.

**Staleness awareness** — each doc type has a freshness threshold: commitment trackers 14d, transcripts/tickets/account notes 30d, QBR decks 90d, solution architecture 180d. Past-threshold sources get a stale warning in citations rather than being silently treated as current.

**Circuit breaker** — after 10 LLM failures the breaker opens, and subsequent calls return immediately rather than queuing retries. Recovery window is 30 seconds. During a rate limit event, without this a single brief would fire off 30+ retrying API calls and block for minutes. With it, failed section nodes return their `@_safe` fallback (empty result, status "unavailable") and the rest of the brief still generates.

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

Optional env vars: `LLM_MODEL` (default: `gemini-2.5-flash`), `OPENAI_API_KEY` (switches embedder to `text-embedding-3-small`), `SLACK_WEBHOOK_URL` (daily overdue digest at 08:00 UTC), `CONTEXTUAL_RETRIEVAL=1` (prepends LLM-generated context to each chunk before embedding — adds one LLM call per chunk at ingest).

FastAPI · LangGraph · ChromaDB · SQLite · Gemini · sentence-transformers · rank-bm25 · Streamlit · APScheduler
