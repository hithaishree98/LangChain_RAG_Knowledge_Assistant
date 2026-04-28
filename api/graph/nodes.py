import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, List

_log = logging.getLogger(__name__)

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

from .state import GraphState
from langchain_utils import llm_breaker, LLM_MODEL, _llm_invoke_with_retry
from chroma_utils import get_retriever_for_user, fetch_parents

# ── Experiment flags ──────────────────────────────────────────────────────────
_DATA_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
WORKFLOW_MODE  = os.getenv("WORKFLOW_MODE", "loop").lower()
NODE_TIMING    = os.getenv("NODE_TIMING", "").lower() in ("1", "true", "yes")
TOKEN_LOGGING  = os.getenv("TOKEN_LOGGING", "").lower() in ("1", "true", "yes")
TIMING_FILE    = os.path.join(_DATA_DIR, "node_timings.jsonl")
TOKEN_LOG_FILE = os.path.join(_DATA_DIR, "token_usage.jsonl")

if WORKFLOW_MODE != "loop" or NODE_TIMING or TOKEN_LOGGING:
    _log.info("[experiment] WORKFLOW_MODE=%s NODE_TIMING=%s TOKEN_LOGGING=%s",
              WORKFLOW_MODE, NODE_TIMING, TOKEN_LOGGING)


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
        def inner(state):
            t0 = time.perf_counter()
            result = fn(state)
            elapsed = (time.perf_counter() - t0) * 1000
            _log_timing(node_name, elapsed)
            return result
        return inner
    return wrap

# ── Token usage logging ──────────────────────────────────────────────────────
# When TOKEN_LOGGING=1, every Groq call records prompt/completion tokens from
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


_ANALYST_SYSTEM = """ROLE:
You are a pre-call intelligence analyst for a Forward Deployed Engineer.
Your job is NOT to answer questions — it is to surface issues, risks, open questions,
and talking points from customer documents so the FDE walks into the call prepared.

CONTEXT:
Retrieved document chunks (each prefixed with its chunk_id, e.g. [P1], [P2]):
{context}

QUERY:
{query}

TASK:
Analyze the chunks carefully. For every claim you make, cite the chunk_id it came from.
Return ONLY valid JSON matching this exact schema (no prose, no markdown):

{{
  "issues": [{{"claim": "...", "chunk_id": "..."}}],
  "risks": [{{"claim": "...", "chunk_id": "..."}}],
  "open_questions": ["..."],
  "talking_points": [{{"point": "...", "chunk_id": "..."}}]
}}

CONSTRAINTS:
- Every issue and talking_point MUST include a chunk_id
- Never fabricate facts not present in the chunks
- If a section has no relevant content, return an empty list for that key
- open_questions should be unanswered questions the FDE should raise on the call"""


def _build_context_str(docs):
    parts = []
    for i, doc in enumerate(docs):
        cid = doc.metadata.get("chunk_id", f"chunk_{i}")
        parts.append(f"[{cid}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _get_llm():
    return ChatGoogleGenerativeAI(model=LLM_MODEL,
                                  google_api_key=os.getenv("GOOGLE_API_KEY"),
                                  temperature=0)


# _llm_invoke_with_retry is imported from langchain_utils — single source of truth.
# Handles 429 / rate-limit retries across Groq- and Gemini-style error shapes.


_DECOMPOSE_PROMPT = """You are a query decomposition assistant.
Given a user query about a customer, break it into 2-4 specific sub-queries
that together cover the full intent. Return ONLY a JSON array of strings, e.g.:
["sub-query 1", "sub-query 2", "sub-query 3"]
Query: {query}"""


@_timed("query_rewrite")
def query_rewrite_node(state):
    if llm_breaker.is_open():
        return {"sub_queries": [state["original_query"]],
                "iteration_count": state["iteration_count"] + 1,
                "audit_trail": state["audit_trail"] + [
                    {"node": "query_rewrite", "fallback": "circuit_open"}]}
    try:
        llm = _get_llm()
        info_gaps = state.get("information_gaps", [])
        if state.get("iteration_count", 0) > 0 and info_gaps:
            gaps_str = "; ".join(info_gaps[:3])
            _prompt = (
                f"You are a query decomposition assistant. "
                f"A previous retrieval pass was insufficient: {gaps_str}. "
                f"Generate 2-4 more targeted sub-queries to find better information for: "
                f"{state['original_query']} "
                f"Return ONLY a JSON array of strings."
            )
        else:
            _prompt = _DECOMPOSE_PROMPT.format(query=state["original_query"])
        resp = _llm_invoke_with_retry(llm, _prompt)
        _log_token_usage("query_rewrite", resp, _prompt)
        content = resp.content.strip()
        if "```" in content:
            content = content.split("```")[1].strip()
            if content.startswith("json"): content = content[4:].strip()
        sub_queries = json.loads(content)
        if not isinstance(sub_queries, list) or not sub_queries:
            raise ValueError(f"expected list, got: {type(sub_queries).__name__}")
        llm_breaker.on_success()
    except json.JSONDecodeError as e:
        llm_breaker.on_failure()
        _log.error("query_rewrite_json_parse_failed", extra={"error": str(e), "raw": content[:200]})
        sub_queries = [state["original_query"]]
    except Exception as e:
        llm_breaker.on_failure()
        _log.warning("query_rewrite_failed", extra={"error": str(e)})
        sub_queries = [state["original_query"]]
    return {"sub_queries": sub_queries,
            "iteration_count": state["iteration_count"] + 1,
            "audit_trail": state["audit_trail"] + [
                {"node": "query_rewrite", "sub_queries": sub_queries}]}


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
        # an empty brief. The /brief endpoint catches this and returns 503.
        raise RuntimeError(f"Retrieval failed for all sub-queries: {errors[0]}")
    for i, doc in enumerate(child_chunks):
        if "chunk_id" not in doc.metadata:
            doc.metadata["chunk_id"] = f"C{i+1}"
    parent_chunks = fetch_parents(child_chunks, state["customer_id"])
    for i, doc in enumerate(parent_chunks):
        doc.metadata["chunk_id"] = f"P{i+1}"
    return {"retrieved_chunks": child_chunks, "parent_chunks": parent_chunks,
            "audit_trail": state["audit_trail"] + [
                {"node": "retrieve",
                 "child_count": len(child_chunks),
                 "parent_count": len(parent_chunks)}]}


@_timed("reason")
def reason_node(state):
    if llm_breaker.is_open():
        return {"reasoning_output": None,
                "audit_trail": state["audit_trail"] + [
                    {"node": "reason", "error": "circuit_open"}]}
    docs = state["parent_chunks"] or state["retrieved_chunks"]
    if not docs:
        return {"reasoning_output": {"issues": [], "risks": [],
                                      "open_questions": [], "talking_points": []},
                "audit_trail": state["audit_trail"] + [{"node": "reason", "note": "no_docs"}]}
    ctx = _build_context_str(docs)
    prompt = _ANALYST_SYSTEM.format(context=ctx, query=state["original_query"])
    try:
        llm = _get_llm()
        resp = _llm_invoke_with_retry(llm, prompt)
        _log_token_usage("reason", resp, prompt)
        content = resp.content.strip()
        if "```" in content:
            content = content.split("```")[1].strip()
            if content.startswith("json"): content = content[4:].strip()
        reasoning = json.loads(content)
        llm_breaker.on_success()
    except json.JSONDecodeError as e:
        llm_breaker.on_failure()
        _log.error("reason_json_parse_failed", extra={"error": str(e), "raw": content[:200]})
        reasoning = {"issues": [], "risks": [],
                     "open_questions": [f"Could not parse analyst output: {e}"],
                     "talking_points": [], "_parse_error": True}
    except Exception as e:
        llm_breaker.on_failure()
        _log.warning("reason_node_failed", extra={"error": str(e)})
        reasoning = {"issues": [], "risks": [],
                     "open_questions": [f"Could not analyze: {e}"],
                     "talking_points": [], "_parse_error": True}
    return {"reasoning_output": reasoning,
            "audit_trail": state["audit_trail"] + [{"node": "reason", "status": "ok"}]}


_REQUIRED_KEYS = {"issues", "risks", "open_questions", "talking_points"}

# Citation-rate threshold for the completeness loop. The first-pass brief is
# considered "weak" if fewer than this fraction of findings carry a chunk_id
# citation — interpretation: the LLM produced claims it cannot ground in the
# retrieved chunks. On the eval set this fires roughly never with Gemini
# 2.5-flash (it reliably cites ≥ 50% of findings on the first pass), so the
# loop is currently vestigial. Tuning options live in ImprovementsForProd.md
# #23. Loop hard cap is 3 iterations regardless of this value.
_CITATION_RATE_THRESHOLD = 0.5


@_timed("completeness")
def completeness_node(state):
    # ── SINGLE_PASS MODE: short-circuit the loop ────────────────────────────
    if WORKFLOW_MODE == "single_pass":
        return {"is_sufficient": True, "information_gaps": [],
                "audit_trail": state["audit_trail"] + [
                    {"node": "completeness", "mode": "single_pass_forced_sufficient"}]}

    ro = state["reasoning_output"] or {}
    # A parse/LLM error is a system failure, not an information gap — don't loop.
    if ro.get("_parse_error"):
        return {"is_sufficient": True, "information_gaps": ["reasoning_output_parse_error"],
                "audit_trail": state["audit_trail"] + [
                    {"node": "completeness", "note": "parse_error_skip_loop"}]}
    gaps = [f"missing '{k}' section" for k in _REQUIRED_KEYS if k not in ro]
    loop_capped = False
    if not gaps:
        # Grounding check: count findings that actually include a chunk_id
        # citation. The analyst prompt requires citations on every finding, so a
        # low citation rate means the LLM either hallucinated or the retrieved
        # chunks didn't cover the question well — in either case, re-querying
        # with refined sub-queries is the right move. Citation rate is the real
        # quality signal; raw finding count is not (LLMs return 2+ findings on
        # garbage input just as reliably as on good input).
        findings_total = 0
        findings_with_citations = 0
        for section in ("issues", "risks", "talking_points"):
            for item in ro.get(section, []) or []:
                if not isinstance(item, dict):
                    continue
                findings_total += 1
                if item.get("chunk_id"):
                    findings_with_citations += 1
        citation_rate = (findings_with_citations / findings_total) if findings_total else 0.0
        iter_no = state.get("iteration_count", 1)

        weak_pass = (findings_total == 0) or (citation_rate < _CITATION_RATE_THRESHOLD)
        if weak_pass and iter_no <= 1:
            if findings_total == 0:
                gaps.append("no findings extracted on first pass — retrying with refined queries")
            else:
                gaps.append(
                    f"only {findings_with_citations}/{findings_total} findings have chunk "
                    f"citations (citation_rate={citation_rate:.2f}) — retrying for better grounding"
                )
        elif weak_pass and iter_no > 1:
            # Second pass also looks weak but we cap the loop to prevent runaway
            # iterations. Surface this in the audit trail so callers can tell
            # "judged sufficient" from "loop capped at iteration limit".
            loop_capped = True
    is_sufficient = len(gaps) == 0
    audit_entry = {"node": "completeness", "is_sufficient": is_sufficient, "gaps": gaps}
    if loop_capped:
        audit_entry["loop_capped_at_iteration_limit"] = True
    return {"is_sufficient": is_sufficient, "information_gaps": gaps,
            "audit_trail": state["audit_trail"] + [audit_entry]}
