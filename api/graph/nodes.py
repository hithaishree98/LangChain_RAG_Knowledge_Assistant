import hashlib
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

_log = logging.getLogger(__name__)

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

from .state import GraphState
from langchain_utils import llm_breaker, LLM_MODEL, _llm_invoke_with_retry
from chroma_utils import get_retriever_for_user, fetch_parents

# ── Experiment flags ──────────────────────────────────────────────────────────
_DATA_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
NODE_TIMING    = os.getenv("NODE_TIMING", "").lower() in ("1", "true", "yes")
TOKEN_LOGGING  = os.getenv("TOKEN_LOGGING", "").lower() in ("1", "true", "yes")
TIMING_FILE    = os.path.join(_DATA_DIR, "node_timings.jsonl")
TOKEN_LOG_FILE = os.path.join(_DATA_DIR, "token_usage.jsonl")

if NODE_TIMING or TOKEN_LOGGING:
    _log.info("[experiment] NODE_TIMING=%s TOKEN_LOGGING=%s", NODE_TIMING, TOKEN_LOGGING)


def _log_timing(node_name, elapsed_ms, extra=None):
    if not NODE_TIMING:
        return
    entry = {"node": node_name, "elapsed_ms": round(elapsed_ms, 2),
             "timestamp": time.time()}
    if extra:
        entry.update(extra)
    try:
        with open(TIMING_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def _timed(node_name):
    def wrap(fn):
        def inner(state, *args, **kwargs):
            t0 = time.perf_counter()
            result = fn(state, *args, **kwargs)
            elapsed = (time.perf_counter() - t0) * 1000
            _log_timing(node_name, elapsed)
            return result
        return inner
    return wrap


# ── Token usage logging ──────────────────────────────────────────────────────
# When TOKEN_LOGGING=1, every LLM call records prompt/completion tokens from
# the API response. Used by exp6b_real_cost.py to compare against tiktoken
# estimates from exp6_cost.py. Silently no-ops in production.


def _log_token_usage(call_name, response, prompt_text):
    """Extract real token usage from an LLM response and append to the log file.

    Provider-agnostic: handles LangChain's standardized ``usage_metadata`` shape
    (used by Gemini and newer wrappers) as well as the legacy
    ``response_metadata.token_usage`` shape (Groq/OpenAI). Field names in the
    log are vendor-neutral (``prompt_tokens``, ``completion_tokens``).

    Silently no-ops if TOKEN_LOGGING isn't set, so production cost is zero.
    """
    if not TOKEN_LOGGING:
        return
    try:
        prompt_tokens = 0
        completion_tokens = 0
        # Preferred: LangChain's standardized usage_metadata attribute (Gemini,
        # Anthropic, newer OpenAI). Keys are input_tokens / output_tokens.
        um = getattr(response, "usage_metadata", None)
        if um:
            prompt_tokens     = um.get("input_tokens", 0) or 0
            completion_tokens = um.get("output_tokens", 0) or 0
        # Fallback: legacy response_metadata.token_usage (Groq/older OpenAI).
        if not prompt_tokens and hasattr(response, "response_metadata"):
            meta = response.response_metadata or {}
            tu = meta.get("token_usage", {}) or {}
            prompt_tokens     = tu.get("prompt_tokens", 0) or 0
            completion_tokens = tu.get("completion_tokens", 0) or 0

        # Compute tiktoken estimate for direct comparison with exp6a.
        # Note: cl100k_base is OpenAI's tokenizer; it's a reasonable rough
        # estimate for most models but will be off ~10-20% for Gemini.
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            tiktoken_estimate = len(enc.encode(prompt_text))
        except Exception:
            tiktoken_estimate = 0

        entry = {
            "call": call_name,
            "timestamp": time.time(),
            "prompt_tokens":     prompt_tokens,
            "completion_tokens": completion_tokens,
            "tiktoken_estimate": tiktoken_estimate,
            "prompt_length_chars": len(prompt_text),
        }
        with open(TOKEN_LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def _build_context_str(docs):
    """Format retrieved chunks for the LLM with chunk_id, content date, and source.

    Header shape: ``[chunk_id | YYYY-MM-DD | filename]``

    Including dates and source filenames in the context lets the LLM reason
    about recency ("the most recent agreement") and provenance ("which doc
    said this") without needing extra tooling. The fields are best-effort —
    chunks without a doc_date display ``?``.
    """
    parts = []
    for i, doc in enumerate(docs):
        cid = doc.metadata.get("chunk_id", f"chunk_{i}")
        date = doc.metadata.get("doc_date") or "?"
        src = doc.metadata.get("filename") or doc.metadata.get("source") or "?"
        # Strip directory prefixes so the header stays compact
        src = os.path.basename(src) if src and src != "?" else src
        parts.append(f"[{cid} | {date} | {src}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _strip_json(content: str) -> str:
    """Strip markdown code fences from LLM JSON output.

    Handles ``` and ```json fences robustly. Uses a non-greedy regex so that
    backticks inside the JSON body (rare but possible) don't truncate the match.
    Falls back to the raw content if no fence is found.
    """
    import re
    content = content.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
    if m:
        return m.group(1).strip()
    return content


def _get_llm():
    return ChatGoogleGenerativeAI(model=LLM_MODEL,
                                  google_api_key=os.getenv("GOOGLE_API_KEY"),
                                  temperature=0)


# _llm_invoke_with_retry is imported from langchain_utils — single source of truth.
# Handles 429 / rate-limit retries across Groq- and Gemini-style error shapes.


# ── _safe decorator ───────────────────────────────────────────────────────────

def _safe(section_name: str):
    """Decorator: catches any exception from a section node, marks it 'unavailable'."""
    def wrap(fn):
        def inner(state, *args, **kwargs):
            try:
                return fn(state, *args, **kwargs)
            except Exception as e:
                _log.error("section_node_failed node=%s error=%s", section_name, str(e))
                return {
                    "section_status": {**state.get("section_status", {}),
                                       section_name: "unavailable"},
                    "audit_trail": [{"node": section_name, "error": str(e), "status": "unavailable"}],
                }
        return inner
    return wrap


# ── Prompts ───────────────────────────────────────────────────────────────────

_ACCOUNT_SUMMARY_PROMPT = """You are writing the account status summary for a pre-meeting brief as of {as_of_date}.

Retrieved context:
{context}

Write 2-3 sentences (MAX 80 words) covering:
1. The customer's current posture — use exactly one of: positive / neutral / at-risk / critical
2. The single most important situation or opportunity right now
3. The nearest upcoming deadline or decision point

Rules:
- Prose only, no bullets, no headers
- State posture explicitly: e.g. "This account is at-risk because..."
- If posture cannot be determined, write: "Insufficient data to assess account status."
- No hedging ("it appears", "it seems"), no preamble

Return ONLY the paragraph text."""


_OPEN_ITEMS_PROMPT = """You are extracting open action items from these support ticket chunks.

{context}

For each open ticket or unresolved issue, output a JSON object:
{{
  "title": "brief title of the issue",
  "status": "open|in_progress|pending",
  "last_update": "YYYY-MM-DD or empty",
  "owner": "assignee name or empty",
  "priority": "P0|P1|P2|normal",
  "source_doc": "filename from chunk header",
  "doc_date": "YYYY-MM-DD from chunk header"
}}

Rules:
- Only include actively open items (not closed, resolved, done)
- Use the chunk header [chunk_id | date | filename] for source_doc and doc_date
- Max 10 items, prioritize P0/P1 first
- Return ONLY a JSON array"""


_RECENT_CHANGES_PROMPT = """Summarize what changed from {since_date} to {as_of_date} based on these document chunks.

{context}

For each significant change, output:
{{
  "what": "what changed (one sentence)",
  "date": "YYYY-MM-DD from chunk header or unknown",
  "source_doc": "filename from chunk header",
  "customer_aware": true if this is NOT from an internal document
}}

Rules:
- Exclude routine ticket updates with no new information
- Exclude entries with chunk dates before {since_date}
- If nothing significant changed, return []
- Return ONLY a JSON array"""


_ANTICIPATED_QUESTIONS_PROMPT = """You are identifying topics the CUSTOMER is likely to raise in the next call.

Customer-side evidence (tickets filed by customer, transcript turns attributed to customer):
{context}

Rules:
- Only include topics where you can quote specific evidence from the context
- If you cannot provide a verbatim or near-verbatim quote, do NOT include the topic
- No speculation — if no customer-side signal, return []
- 3-5 topics maximum

Return ONLY a JSON array:
[{{
  "topic": "brief topic label",
  "evidence": "one sentence explaining why this will come up",
  "source_quote": "verbatim or near-verbatim phrase from the context that signals this concern",
  "source_doc": "filename from chunk header",
  "urgency": "high|medium|low"
}}]"""


_POSTURE_PROMPT = """You are writing the Recommended Posture section of a pre-call brief.

Account summary: {account_summary}
Overdue commitments: {overdue_commitments}
Open items (top 5): {open_items}
Recent changes: {recent_changes}
Outstanding commitments: {commitments}
Anticipated customer questions: {anticipated_questions}

Generate 2-4 directives. Each must:
- Use exactly one verb: Lead / Acknowledge / Defer / Push
- Name the SPECIFIC ticket ID, commitment description, or event that drives it
- Explain the consequence or intent in one sentence

BAD (too generic):
{{"verb": "Acknowledge", "directive": "Address the deployment issues", "basis": "There are open tickets", "grounding_item": ""}}

GOOD (specific and grounded):
{{"verb": "Acknowledge", "directive": "Open with TICK-4521 us-east-2 outage — 45 days open, still P0", "basis": "Customer flagged this in Sep 15 call as blocking their production deploy", "grounding_item": "TICK-4521: us-east-2 deployment failing"}}

Rules:
- Overdue commitments always take precedence over open items
- If there are no overdue commitments and no P0 tickets, you may use Lead
- Do not produce generic directives — if you cannot name a specific item, omit the directive

Return ONLY a JSON array:
[{{"verb": "Lead|Acknowledge|Defer|Push", "directive": "...", "basis": "...", "grounding_item": "specific item name"}}]"""


_EXEC_ROLE_TENURE_PROMPT = """What is the role and tenure of {person_name} at this customer?

Context:
{context}

Write 1-2 sentences about their role and how long they've been in it. If unknown, write "No information available about this person's role."

Return ONLY the prose text."""

_EXEC_STATED_POSITION_PROMPT = """What has {person_name} said about our product, relationship, or upcoming decisions?

Context (transcript excerpts and account notes):
{context}

For each clear statement attributed to this person, output:
{{
  "content": "what they said (paraphrase is fine, stay faithful)",
  "said_by": "person",
  "stated_date": "YYYY-MM-DD from the chunk header date",
  "sentiment": "positive|neutral|concern|request",
  "source_doc": "filename from chunk header",
  "doc_date": "YYYY-MM-DD from chunk header"
}}

Rules:
- Only include statements clearly attributed to {person_name} — not statements about them
- sentiment: positive = expressing satisfaction; concern = raising a problem; request = asking for something; neutral = factual statement
- stated_date must come from the chunk header — do not infer or guess
- Max 5 statements, most recent first
- Return ONLY a JSON array"""

_EXEC_RECENT_SIGNALS_PROMPT = """What recent signals or events involve {person_name}?

Context (events since {since_date}):
{context}

For each signal, output:
{{"event": "what happened", "date": "YYYY-MM-DD or unknown", "source_doc": "filename"}}

Return ONLY a JSON array, or [] if no signals found."""

_EXEC_OPEN_ASKS_PROMPT = """What open requests or asks has {person_name} made that are still unresolved?

Context:
{context}

For each open ask, output:
{{"ask": "what they asked for", "date": "YYYY-MM-DD when asked", "status": "open", "source_doc": "filename"}}

Return ONLY a JSON array, or [] if no open asks found."""

_EXEC_APPROACH_PROMPT = """Write the recommended approach for a meeting with {person_name} at this customer.

Their stated positions: {stated_positions}
Open asks: {open_asks}
Recent signals: {recent_signals}

Write 2-3 sentences on how to engage with this person specifically. Be concrete.

Return ONLY the prose text."""


# ── Q&A prompt (used by /query workflow) ──────────────────────────────────────

_QA_PROMPT = """ROLE:
You answer factual questions about a customer using ONLY the retrieved chunks below.
You are not generating a brief — produce a direct, dated answer with citations.

CONTEXT:
Each chunk header has the format: [chunk_id | YYYY-MM-DD | source_filename]
Use the dates to reason about recency. When the question asks "what's the current
status", prefer the chunk with the most recent date.

{context}

QUESTION:
{query}

TASK:
1. Find the most recent chunk that answers the question.
2. Begin your answer with "As of [date from that chunk]:" so the reader knows the timeframe.
3. Cite chunk_id inline next to every factual claim.
4. If the chunks do NOT contain a clear answer, return answer_status "not_found"
   and answer "" — do NOT guess, infer, or use general knowledge.

Return ONLY valid JSON (no prose, no markdown):

{{
  "answer": "As of YYYY-MM-DD: Direct answer with [chunk_id] inline citations.",
  "answer_status": "ok" | "not_found" | "partial",
  "answer_date": "YYYY-MM-DD of the most recent chunk used",
  "citations": [
    {{"claim": "the specific fact this chunk supports", "chunk_id": "...", "date": "YYYY-MM-DD"}}
  ]
}}

CONSTRAINTS:
- Start answer with "As of [date]:" — never omit this
- 1-3 sentences for simple questions; longer only if the question genuinely requires it
- Use exact dates and numbers from the chunks; never paraphrase numerics
- "not_found" with answer "" is the correct output when the answer is not in the chunks
- If the question has multiple parts and you can only answer some, use "partial"
"""


# ── Helper: fallback ticket items from metadata (no LLM) ─────────────────────

def _tickets_to_items_from_metadata(docs):
    items = []
    for doc in docs:
        md = doc.metadata
        items.append({
            "title": md.get("title") or md.get("ticket_id") or doc.page_content[:80],
            "status": md.get("status") or "open",
            "last_update": md.get("updated_date") or md.get("doc_date") or "",
            "owner": md.get("assignee") or "",
            "priority": md.get("priority") or "normal",
            "source_doc": md.get("filename") or os.path.basename(md.get("source", "")),
            "doc_date": md.get("doc_date") or "",
        })
    return items


# ── /query workflow nodes ─────────────────────────────────────────────────────

@_timed("retrieve")
def retrieve_node(state):
    retriever = get_retriever_for_user(state["customer_id"])
    seen = set(); child_chunks = []; errors = []
    for sq in state["sub_queries"]:
        try:
            try:
                docs = retriever.invoke(sq)
            except AttributeError:
                docs = retriever.get_relevant_documents(sq)
            for doc in docs:
                key = hashlib.md5(doc.page_content.encode(), usedforsecurity=False).hexdigest()
                if key not in seen:
                    seen.add(key); child_chunks.append(doc)
        except Exception as e:
            _log.error("retrieve_subquery_failed", extra={"sub_query": sq[:120], "error": str(e)})
            errors.append(str(e))

    if errors and not child_chunks:
        # Every sub-query failed — surface the error rather than silently returning
        # an empty brief. The /query endpoint catches this and returns 503.
        raise RuntimeError(f"Retrieval failed for all sub-queries: {errors[0]}")
    for i, doc in enumerate(child_chunks):
        if "chunk_id" not in doc.metadata:
            doc.metadata["chunk_id"] = f"C{i+1}"
    parent_chunks = fetch_parents(child_chunks, state["customer_id"])
    for i, doc in enumerate(parent_chunks):
        doc.metadata["chunk_id"] = f"P{i+1}"
    return {"retrieved_chunks": child_chunks, "parent_chunks": parent_chunks,
            "audit_trail": [
                {"node": "retrieve",
                 "child_count": len(child_chunks),
                 "parent_count": len(parent_chunks)}]}


@_timed("answer")
def answer_node(state):
    """Single-pass Q&A node — produces a focused answer instead of a brief."""
    if llm_breaker.is_open():
        return {"answer_output": {
                    "answer": "AI service temporarily unavailable.",
                    "answer_status": "error",
                    "citations": [],
                    "_breaker_open": True,
                },
                "audit_trail": [{"node": "answer", "error": "circuit_open"}]}

    docs = state.get("parent_chunks") or state.get("retrieved_chunks") or []
    if not docs:
        return {"answer_output": {
                    "answer": "I don't have any indexed documents that answer this question.",
                    "answer_status": "not_found",
                    "citations": [],
                },
                "audit_trail": [{"node": "answer", "note": "no_docs"}]}

    ctx = _build_context_str(docs)
    prompt = _QA_PROMPT.format(context=ctx, query=state["original_query"])

    content = ""
    try:
        llm = _get_llm()
        resp = _llm_invoke_with_retry(llm, prompt)
        _log_token_usage("answer", resp, prompt)
        content = (resp.content or "").strip()
        content = _strip_json(content)
        result = json.loads(content)
        # Defensive normalization — LLM occasionally omits keys
        result.setdefault("answer", "")
        result.setdefault("answer_status", "ok")
        result.setdefault("citations", [])
        llm_breaker.on_success()
    except json.JSONDecodeError as e:
        llm_breaker.on_failure()
        _log.error("answer_json_parse_failed",
                   extra={"error": str(e), "raw": content[:200]})
        result = {"answer": f"Could not parse answer output: {e}",
                  "answer_status": "error",
                  "citations": [],
                  "_parse_error": True}
    except Exception as e:
        llm_breaker.on_failure()
        _log.warning("answer_node_failed", extra={"error": str(e)})
        result = {"answer": f"Could not generate answer: {e}",
                  "answer_status": "error",
                  "citations": [],
                  "_parse_error": True}

    return {"answer_output": result,
            "loop_count": 1,
            "audit_trail": [{"node": "answer", "status": result.get("answer_status", "ok")}]}


# ── Pre-meeting section nodes ─────────────────────────────────────────────────

@_timed("overdue_commitments")
@_safe("overdue_commitments")
def overdue_commitments_node(state):
    from chroma_utils import get_latest_chunks_by_doctype
    as_of = state.get("as_of_date") or state.get("today_date") or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    docs = get_latest_chunks_by_doctype(state["customer_id"], "commitment_tracker")
    items = []
    for doc in docs:
        md = doc.metadata
        status = (md.get("status") or md.get("commitment_status") or "").lower()
        if status in ("closed", "resolved", "done", "complete"):
            continue
        target = md.get("current_target_date") or md.get("promised_date") or ""
        is_overdue = md.get("is_overdue") in ("true", True, "1")
        if not is_overdue and target:
            try:
                is_overdue = target < as_of
            except Exception:
                pass
        if not is_overdue:
            continue
        items.append({
            "description": doc.page_content.split("\n")[0].replace("Commitment: ", ""),
            "promised_date": md.get("promised_date") or "",
            "target_date": target,
            "status": md.get("status") or md.get("commitment_status") or "open",
            "owner": md.get("owner") or "",
            "is_slipped": md.get("is_slipped") in ("true", True, "1"),
            "is_overdue": True,
            "customer_aware": md.get("customer_aware") in ("true", True, "1"),
            "source_doc": md.get("filename") or os.path.basename(md.get("source", "")),
            "doc_date": md.get("doc_date") or "",
        })
    sources = list({doc.metadata.get("filename", "") for doc in docs if doc.metadata.get("filename")})
    as_of_dates = [doc.metadata.get("doc_date", "") for doc in docs if doc.metadata.get("doc_date")]
    return {
        "overdue_commitments_data": items,
        "overdue_sources": sources,
        "overdue_as_of": max(as_of_dates) if as_of_dates else "",
        "section_status": {**state.get("section_status", {}),
                           "overdue_commitments": "ok" if items else "empty"},
        "audit_trail": [{"node": "overdue_commitments", "count": len(items)}],
    }


@_timed("open_items")
@_safe("open_items")
def open_items_node(state):
    from chroma_utils import vectorstore
    from langchain_core.documents import Document as _Doc
    user_id = state["customer_id"]
    try:
        result = vectorstore._collection.get(
            where={"$and": [
                {"user_id": {"$eq": user_id}},
                {"doc_type": {"$eq": "ticket"}},
                {"is_latest_version": {"$eq": 1}},
            ]},
            include=["documents", "metadatas"],
        )
        all_docs = [
            _Doc(page_content=result["documents"][i], metadata=result["metadatas"][i])
            for i in range(len(result.get("documents") or []))
        ]
    except Exception as e:
        _log.warning("open_items_chroma_failed error=%s", str(e))
        all_docs = []

    # Filter to open/in-progress only
    docs = [
        d for d in all_docs
        if (d.metadata.get("status") or "").lower() in ("open", "in_progress", "in progress", "")
    ]

    if not docs:
        return {
            "open_items_data": [],
            "section_status": {**state.get("section_status", {}), "open_items": "empty"},
            "audit_trail": [{"node": "open_items", "note": "no_open_tickets"}],
        }

    ctx = _build_context_str(docs[:10])
    prompt = _OPEN_ITEMS_PROMPT.format(context=ctx)
    if llm_breaker.is_open():
        # Degrade to metadata-only items when LLM is down
        items = _tickets_to_items_from_metadata(docs[:10])
        return {
            "open_items_data": items,
            "section_status": {**state.get("section_status", {}), "open_items": "ok"},
            "audit_trail": [{"node": "open_items", "fallback": "circuit_open", "count": len(items)}],
        }
    try:
        llm = _get_llm()
        resp = _llm_invoke_with_retry(llm, prompt)
        _log_token_usage("open_items", resp, prompt)
        content = _strip_json(resp.content)
        items = json.loads(content)
        if not isinstance(items, list):
            items = []
        llm_breaker.on_success()
    except Exception as e:
        llm_breaker.on_failure()
        _log.warning("open_items_llm_failed error=%s", str(e))
        items = _tickets_to_items_from_metadata(docs[:10])
    sources = list({d.metadata.get("filename", "") for d in docs[:10] if d.metadata.get("filename")})
    as_of_dates = [d.metadata.get("doc_date", "") for d in docs[:10] if d.metadata.get("doc_date")]
    return {
        "open_items_data": items,
        "open_items_sources": sources,
        "open_items_as_of": max(as_of_dates) if as_of_dates else "",
        "section_status": {**state.get("section_status", {}), "open_items": "ok" if items else "empty"},
        "audit_trail": [{"node": "open_items", "count": len(items)}],
    }


@_timed("account_summary")
@_safe("account_summary")
def account_summary_node(state):
    if llm_breaker.is_open():
        return {
            "account_summary_text": "AI service temporarily unavailable.",
            "section_status": {**state.get("section_status", {}), "account_summary": "unavailable"},
            "audit_trail": [{"node": "account_summary", "error": "circuit_open"}],
        }
    from chroma_utils import get_retriever_for_user
    # Broad retrieval for overall account context
    retriever = get_retriever_for_user(state["customer_id"])
    try:
        docs = retriever.invoke("account status health relationship summary")
    except Exception:
        docs = []
    ctx = _build_context_str(docs[:6]) if docs else "No documents retrieved."
    as_of = state.get("as_of_date") or state.get("today_date") or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    prompt = _ACCOUNT_SUMMARY_PROMPT.format(context=ctx, as_of_date=as_of)
    sources = list({d.metadata.get("filename", "") for d in docs[:6] if d.metadata.get("filename")})
    as_of_dates = [d.metadata.get("doc_date", "") for d in docs[:6] if d.metadata.get("doc_date")]
    try:
        llm = _get_llm()
        resp = _llm_invoke_with_retry(llm, prompt)
        _log_token_usage("account_summary", resp, prompt)
        summary = resp.content.strip()
        words = summary.split()
        if len(words) > 100:
            summary = " ".join(words[:100]) + "…"
        llm_breaker.on_success()
        return {
            "account_summary_text": summary,
            "account_summary_sources": sources,
            "account_summary_as_of": max(as_of_dates) if as_of_dates else "",
            "section_status": {**state.get("section_status", {}), "account_summary": "ok"},
            "audit_trail": [{"node": "account_summary", "status": "ok"}],
        }
    except Exception as e:
        llm_breaker.on_failure()
        _log.warning("account_summary_node_failed error=%s", str(e))
        return {
            "account_summary_text": "Insufficient data to assess account status.",
            "account_summary_sources": [],
            "account_summary_as_of": "",
            "section_status": {**state.get("section_status", {}), "account_summary": "unavailable"},
            "audit_trail": [{"node": "account_summary", "error": str(e)}],
        }


@_timed("recent_changes")
@_safe("recent_changes")
def recent_changes_node(state):
    from chroma_utils import get_chunks_since_date
    as_of = state.get("as_of_date") or state.get("today_date") or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    # Use "since last call" window — if no last_call_date, fall back to 14 days
    last_call = state.get("last_call_date")
    if last_call:
        since = last_call
    else:
        try:
            cutoff = (datetime.strptime(as_of, "%Y-%m-%d") - timedelta(days=14))
            since = cutoff.strftime("%Y-%m-%d")
        except Exception:
            since = "1970-01-01"

    docs = get_chunks_since_date(state["customer_id"], since, exclude_doc_types=("ticket",))
    if not docs:
        return {
            "recent_changes_data": [],
            "section_status": {**state.get("section_status", {}), "recent_changes": "empty"},
            "audit_trail": [{"node": "recent_changes", "since": since, "note": "no_recent_docs"}],
        }
    ctx = _build_context_str(docs[:8])
    prompt = _RECENT_CHANGES_PROMPT.format(context=ctx, since_date=since, as_of_date=as_of)
    if llm_breaker.is_open():
        return {
            "recent_changes_data": [],
            "section_status": {**state.get("section_status", {}), "recent_changes": "unavailable"},
            "audit_trail": [{"node": "recent_changes", "error": "circuit_open"}],
        }
    try:
        llm = _get_llm()
        resp = _llm_invoke_with_retry(llm, prompt)
        _log_token_usage("recent_changes", resp, prompt)
        content = _strip_json(resp.content)
        changes = json.loads(content)
        if not isinstance(changes, list):
            changes = []
        llm_breaker.on_success()
    except Exception as e:
        llm_breaker.on_failure()
        _log.warning("recent_changes_node_failed error=%s", str(e))
        changes = []
    sources = list({d.metadata.get("filename", "") for d in docs[:8] if d.metadata.get("filename")})
    as_of_dates = [d.metadata.get("doc_date", "") for d in docs[:8] if d.metadata.get("doc_date")]
    return {
        "recent_changes_data": changes,
        "recent_changes_sources": sources,
        "recent_changes_as_of": max(as_of_dates) if as_of_dates else "",
        "section_status": {**state.get("section_status", {}), "recent_changes": "ok" if changes else "empty"},
        "audit_trail": [{"node": "recent_changes", "since": since, "count": len(changes)}],
    }


@_timed("outstanding_commitments")
@_safe("outstanding_commitments")
def outstanding_commitments_node(state):
    from chroma_utils import get_latest_chunks_by_doctype
    docs = get_latest_chunks_by_doctype(state["customer_id"], "commitment_tracker")
    items = []
    for doc in docs:
        md = doc.metadata
        status = (md.get("status") or md.get("commitment_status") or "").lower()
        if status in ("closed", "resolved", "done", "complete"):
            continue
        items.append({
            "description": doc.page_content.split("\n")[0].replace("Commitment: ", ""),
            "promised_date": md.get("promised_date") or "",
            "target_date": md.get("current_target_date") or md.get("promised_date") or "",
            "status": md.get("status") or md.get("commitment_status") or "open",
            "owner": md.get("owner") or "",
            "is_slipped": md.get("is_slipped") in ("true", True, "1"),
            "is_overdue": md.get("is_overdue") in ("true", True, "1"),
            "customer_aware": md.get("customer_aware") in ("true", True, "1"),
            "source_doc": md.get("filename") or os.path.basename(md.get("source", "")),
            "doc_date": md.get("doc_date") or "",
        })
    sources = list({doc.metadata.get("filename", "") for doc in docs if doc.metadata.get("filename")})
    as_of_dates = [doc.metadata.get("doc_date", "") for doc in docs if doc.metadata.get("doc_date")]
    return {
        "outstanding_commitments_data": items,
        "outstanding_sources": sources,
        "outstanding_as_of": max(as_of_dates) if as_of_dates else "",
        "section_status": {**state.get("section_status", {}), "outstanding_commitments": "ok" if items else "empty"},
        "audit_trail": [{"node": "outstanding_commitments", "count": len(items)}],
    }


@_timed("anticipated_questions")
@_safe("anticipated_questions")
def anticipated_questions_node(state):
    from chroma_utils import vectorstore
    from langchain_core.documents import Document as _Doc

    user_id = state["customer_id"]
    try:
        result = vectorstore._collection.get(
            where={
                "$and": [
                    {"user_id": {"$eq": user_id}},
                    {"doc_type": {"$in": ["ticket", "transcript"]}},
                    {"is_latest_version": {"$eq": 1}},
                ]
            },
            include=["documents", "metadatas"],
        )
        all_docs = [
            _Doc(page_content=result["documents"][i], metadata=result["metadatas"][i])
            for i in range(len(result.get("documents") or []))
        ]
    except Exception as e:
        _log.warning("anticipated_questions_fetch_failed error=%s", str(e))
        all_docs = []

    # Filter to customer-side evidence:
    # - Open tickets (customer filed the issue)
    # - Transcript chunks (customer voice is present)
    customer_docs = []
    for doc in all_docs:
        md = doc.metadata
        dt = md.get("doc_type", "")
        if dt == "ticket" and md.get("status", "").lower() in ("open", "in_progress", ""):
            customer_docs.append(doc)
        elif dt == "transcript":
            customer_docs.append(doc)

    if not customer_docs:
        return {
            "anticipated_questions_data": [],
            "section_status": {**state.get("section_status", {}), "anticipated_questions": "empty"},
            "audit_trail": [{"node": "anticipated_questions", "note": "no_customer_signals"}],
        }

    ctx = _build_context_str(customer_docs[:8])
    prompt = _ANTICIPATED_QUESTIONS_PROMPT.format(context=ctx)
    if llm_breaker.is_open():
        return {
            "anticipated_questions_data": [],
            "section_status": {**state.get("section_status", {}), "anticipated_questions": "unavailable"},
            "audit_trail": [{"node": "anticipated_questions", "error": "circuit_open"}],
        }
    try:
        llm = _get_llm()
        resp = _llm_invoke_with_retry(llm, prompt)
        _log_token_usage("anticipated_questions", resp, prompt)
        content = _strip_json(resp.content)
        topics = json.loads(content)
        if not isinstance(topics, list):
            topics = []
        # Filter to valid urgency values
        _VALID_URGENCY = {"high", "medium", "low"}
        for t in topics:
            if isinstance(t, dict) and t.get("urgency") not in _VALID_URGENCY:
                t["urgency"] = "medium"
        llm_breaker.on_success()
        sources = list({d.metadata.get("filename", "") for d in customer_docs[:8] if d.metadata.get("filename")})
        return {
            "anticipated_questions_data": topics[:5],  # cap at 5 per prompt spec
            "anticipated_questions_sources": sources,
            "section_status": {**state.get("section_status", {}), "anticipated_questions": "ok" if topics else "empty"},
            "audit_trail": [{"node": "anticipated_questions", "count": len(topics)}],
        }
    except Exception as e:
        llm_breaker.on_failure()
        _log.warning("anticipated_questions_node_failed error=%s", str(e))
        return {
            "anticipated_questions_data": [],
            "section_status": {**state.get("section_status", {}), "anticipated_questions": "unavailable"},
            "audit_trail": [{"node": "anticipated_questions", "error": str(e)}],
        }


@_timed("posture")
@_safe("posture")
def posture_node(state):
    if llm_breaker.is_open():
        return {
            "posture_directives_data": [],
            "section_status": {**state.get("section_status", {}), "recommended_posture": "unavailable"},
            "audit_trail": [{"node": "posture", "error": "circuit_open"}],
        }
    prompt = _POSTURE_PROMPT.format(
        account_summary=state.get("account_summary_text") or "Not available",
        overdue_commitments=json.dumps((state.get("overdue_commitments_data") or [])[:5]),
        open_items=json.dumps((state.get("open_items_data") or [])[:5]),
        recent_changes=json.dumps((state.get("recent_changes_data") or [])[:5]),
        commitments=json.dumps((state.get("outstanding_commitments_data") or [])[:5]),
        anticipated_questions=json.dumps((state.get("anticipated_questions_data") or [])[:5]),
    )
    _VALID_VERBS = {"Lead", "Acknowledge", "Defer", "Push"}
    try:
        llm = _get_llm()
        resp = _llm_invoke_with_retry(llm, prompt)
        _log_token_usage("posture", resp, prompt)
        content = _strip_json(resp.content)
        directives = json.loads(content)
        if not isinstance(directives, list):
            directives = []
        validated = []
        for d in directives:
            if not isinstance(d, dict):
                continue
            verb = (d.get("verb") or "").strip().capitalize()
            if verb not in _VALID_VERBS:
                continue
            d["verb"] = verb
            d.setdefault("grounding_item", "")
            validated.append(d)
        llm_breaker.on_success()
        return {
            "posture_directives_data": validated,
            "section_status": {**state.get("section_status", {}), "recommended_posture": "ok" if validated else "empty"},
            "audit_trail": [{"node": "posture", "count": len(validated)}],
        }
    except Exception as e:
        llm_breaker.on_failure()
        _log.warning("posture_node_failed error=%s", str(e))
        return {
            "posture_directives_data": [],
            "section_status": {**state.get("section_status", {}), "recommended_posture": "unavailable"},
            "audit_trail": [{"node": "posture", "error": str(e)}],
        }


# ── Exec 1:1 section nodes ────────────────────────────────────────────────────

@_timed("exec_role_tenure")
@_safe("exec_role_tenure")
def exec_role_tenure_node(state):
    if not state.get("person_id"):
        return {"exec_role_tenure": "No person specified.",
                "section_status": {**state.get("section_status", {}), "exec_role_tenure": "empty"},
                "audit_trail": [{"node": "exec_role_tenure", "note": "no_person_id"}]}
    from chroma_utils import get_retriever_for_user
    retriever = get_retriever_for_user(state["customer_id"])
    person_name = state.get("person_name") or state.get("person_id") or "this person"
    try:
        docs = retriever.invoke(f"{person_name} role title position tenure")
    except Exception:
        docs = []
    ctx = _build_context_str(docs[:4]) if docs else "No documents retrieved."
    prompt = _EXEC_ROLE_TENURE_PROMPT.format(person_name=person_name, context=ctx)
    if llm_breaker.is_open():
        return {"exec_role_tenure": "AI service unavailable.",
                "section_status": {**state.get("section_status", {}), "exec_role_tenure": "unavailable"},
                "audit_trail": [{"node": "exec_role_tenure", "error": "circuit_open"}]}
    try:
        llm = _get_llm()
        resp = _llm_invoke_with_retry(llm, prompt)
        role_text = resp.content.strip()
        llm_breaker.on_success()
        return {"exec_role_tenure": role_text,
                "section_status": {**state.get("section_status", {}), "exec_role_tenure": "ok"},
                "audit_trail": [{"node": "exec_role_tenure", "status": "ok"}]}
    except Exception as e:
        llm_breaker.on_failure()
        return {"exec_role_tenure": "Could not determine role.",
                "section_status": {**state.get("section_status", {}), "exec_role_tenure": "unavailable"},
                "audit_trail": [{"node": "exec_role_tenure", "error": str(e)}]}


@_timed("exec_stated_position")
@_safe("exec_stated_position")
def exec_stated_position_node(state):
    if not state.get("person_id"):
        return {"exec_stated_position": [],
                "section_status": {**state.get("section_status", {}), "exec_stated_position": "empty"},
                "audit_trail": [{"node": "exec_stated_position", "note": "no_person_id"}]}
    from chroma_utils import get_retriever_for_user
    person_name = state.get("person_name") or state.get("person_id") or "this person"
    retriever = get_retriever_for_user(state["customer_id"])
    try:
        docs = retriever.invoke(f"{person_name} said stated believes thinks concerned")
    except Exception:
        docs = []
    if not docs:
        return {"exec_stated_position": [],
                "section_status": {**state.get("section_status", {}), "exec_stated_position": "empty"},
                "audit_trail": [{"node": "exec_stated_position", "note": "no_docs"}]}
    ctx = _build_context_str(docs[:6])
    prompt = _EXEC_STATED_POSITION_PROMPT.format(person_name=person_name, context=ctx)
    if llm_breaker.is_open():
        return {"exec_stated_position": [],
                "section_status": {**state.get("section_status", {}), "exec_stated_position": "unavailable"},
                "audit_trail": [{"node": "exec_stated_position", "error": "circuit_open"}]}
    try:
        llm = _get_llm()
        resp = _llm_invoke_with_retry(llm, prompt)
        content = _strip_json(resp.content)
        statements = json.loads(content)
        if not isinstance(statements, list):
            statements = []
        llm_breaker.on_success()
        return {"exec_stated_position": statements[:5],
                "section_status": {**state.get("section_status", {}), "exec_stated_position": "ok" if statements else "empty"},
                "audit_trail": [{"node": "exec_stated_position", "count": len(statements)}]}
    except Exception as e:
        llm_breaker.on_failure()
        return {"exec_stated_position": [],
                "section_status": {**state.get("section_status", {}), "exec_stated_position": "unavailable"},
                "audit_trail": [{"node": "exec_stated_position", "error": str(e)}]}


@_timed("exec_recent_signals")
@_safe("exec_recent_signals")
def exec_recent_signals_node(state):
    if not state.get("person_id"):
        return {"exec_recent_signals": [],
                "section_status": {**state.get("section_status", {}), "exec_recent_signals": "empty"},
                "audit_trail": [{"node": "exec_recent_signals", "note": "no_person_id"}]}
    from chroma_utils import get_chunks_since_date
    person_name = state.get("person_name") or state.get("person_id") or "this person"
    since = state.get("last_call_date") or "1970-01-01"
    docs = get_chunks_since_date(state["customer_id"], since)
    person_docs = [d for d in docs if person_name.lower() in d.page_content.lower()]
    if not person_docs:
        return {"exec_recent_signals": [],
                "section_status": {**state.get("section_status", {}), "exec_recent_signals": "empty"},
                "audit_trail": [{"node": "exec_recent_signals", "note": "no_person_mentions"}]}
    ctx = _build_context_str(person_docs[:6])
    prompt = _EXEC_RECENT_SIGNALS_PROMPT.format(person_name=person_name, context=ctx, since_date=since)
    if llm_breaker.is_open():
        return {"exec_recent_signals": [],
                "section_status": {**state.get("section_status", {}), "exec_recent_signals": "unavailable"},
                "audit_trail": [{"node": "exec_recent_signals", "error": "circuit_open"}]}
    try:
        llm = _get_llm()
        resp = _llm_invoke_with_retry(llm, prompt)
        content = _strip_json(resp.content)
        signals = json.loads(content)
        if not isinstance(signals, list):
            signals = []
        llm_breaker.on_success()
        return {"exec_recent_signals": signals,
                "section_status": {**state.get("section_status", {}), "exec_recent_signals": "ok" if signals else "empty"},
                "audit_trail": [{"node": "exec_recent_signals", "count": len(signals)}]}
    except Exception as e:
        llm_breaker.on_failure()
        return {"exec_recent_signals": [],
                "section_status": {**state.get("section_status", {}), "exec_recent_signals": "unavailable"},
                "audit_trail": [{"node": "exec_recent_signals", "error": str(e)}]}


@_timed("exec_open_asks")
@_safe("exec_open_asks")
def exec_open_asks_node(state):
    if not state.get("person_id"):
        return {"exec_open_asks": [],
                "section_status": {**state.get("section_status", {}), "exec_open_asks": "empty"},
                "audit_trail": [{"node": "exec_open_asks", "note": "no_person_id"}]}
    from chroma_utils import get_retriever_for_user
    person_name = state.get("person_name") or state.get("person_id") or "this person"
    retriever = get_retriever_for_user(state["customer_id"])
    try:
        docs = retriever.invoke(f"{person_name} asked requested wants needs feature")
    except Exception:
        docs = []
    if not docs:
        return {"exec_open_asks": [],
                "section_status": {**state.get("section_status", {}), "exec_open_asks": "empty"},
                "audit_trail": [{"node": "exec_open_asks", "note": "no_docs"}]}
    ctx = _build_context_str(docs[:6])
    prompt = _EXEC_OPEN_ASKS_PROMPT.format(person_name=person_name, context=ctx)
    if llm_breaker.is_open():
        return {"exec_open_asks": [],
                "section_status": {**state.get("section_status", {}), "exec_open_asks": "unavailable"},
                "audit_trail": [{"node": "exec_open_asks", "error": "circuit_open"}]}
    try:
        llm = _get_llm()
        resp = _llm_invoke_with_retry(llm, prompt)
        content = _strip_json(resp.content)
        asks = json.loads(content)
        if not isinstance(asks, list):
            asks = []
        llm_breaker.on_success()
        return {"exec_open_asks": asks,
                "section_status": {**state.get("section_status", {}), "exec_open_asks": "ok" if asks else "empty"},
                "audit_trail": [{"node": "exec_open_asks", "count": len(asks)}]}
    except Exception as e:
        llm_breaker.on_failure()
        return {"exec_open_asks": [],
                "section_status": {**state.get("section_status", {}), "exec_open_asks": "unavailable"},
                "audit_trail": [{"node": "exec_open_asks", "error": str(e)}]}


@_timed("exec_recommended_approach")
@_safe("exec_recommended_approach")
def exec_recommended_approach_node(state):
    if llm_breaker.is_open():
        return {"exec_recommended_approach": "AI service unavailable.",
                "section_status": {**state.get("section_status", {}), "exec_recommended_approach": "unavailable"},
                "audit_trail": [{"node": "exec_recommended_approach", "error": "circuit_open"}]}
    person_name = state.get("person_name") or state.get("person_id") or "this person"
    prompt = _EXEC_APPROACH_PROMPT.format(
        person_name=person_name,
        stated_positions=json.dumps((state.get("exec_stated_position") or [])[:3]),
        open_asks=json.dumps((state.get("exec_open_asks") or [])[:3]),
        recent_signals=json.dumps((state.get("exec_recent_signals") or [])[:3]),
    )
    try:
        llm = _get_llm()
        resp = _llm_invoke_with_retry(llm, prompt)
        approach = resp.content.strip()
        llm_breaker.on_success()
        return {"exec_recommended_approach": approach,
                "section_status": {**state.get("section_status", {}), "exec_recommended_approach": "ok"},
                "audit_trail": [{"node": "exec_recommended_approach", "status": "ok"}]}
    except Exception as e:
        llm_breaker.on_failure()
        return {"exec_recommended_approach": "Could not generate approach.",
                "section_status": {**state.get("section_status", {}), "exec_recommended_approach": "unavailable"},
                "audit_trail": [{"node": "exec_recommended_approach", "error": str(e)}]}
