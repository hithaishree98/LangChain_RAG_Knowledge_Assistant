"""
graph/workflow.py — Three compiled LangGraph workflows.

Pre-meeting brief:  START → section nodes (parallel) → posture → generate_brief
Exec 1:1 brief:     START → exec nodes (parallel) → recommended_approach → generate_exec
Query:              query_rewrite → retrieve → answer → generate_answer
"""

import re as _re
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# Detects temporal queries about overdue/past-due commitments so the rewrite node
# can inject a sub-query tuned to surface "OVERDUE by N days" chunk text.
_OVERDUE_INTENT_RE = _re.compile(
    r"\b(overdue|past[\s_-]?due|behind[\s_-]?schedule|missed[\s_-]?deadline|slipped|late[\s_-]?commitment)\b",
    _re.IGNORECASE,
)

from langgraph.graph import StateGraph, END, START

from .state import GraphState

# Module-level imports for patching in tests.
# _query_rewrite_node uses these via local import, but having them here lets
# unittest.mock.patch("graph.workflow._llm_invoke_with_retry", ...) work.
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None  # type: ignore[assignment,misc]

from langchain_utils import _llm_invoke_with_retry  # noqa: F401  (re-export for patch targets)
from .nodes import (
    retrieve_node,
    answer_node,
    overdue_commitments_node,
    account_summary_node,
    open_items_node,
    recent_changes_node,
    outstanding_commitments_node,
    anticipated_questions_node,
    posture_node,
    exec_role_tenure_node,
    exec_stated_position_node,
    exec_recent_signals_node,
    exec_open_asks_node,
    exec_recommended_approach_node,
)


# ── Pre-meeting brief assembly ────────────────────────────────────────────────

def _generate_pre_meeting_node(state: GraphState) -> dict:
    from output.brief_generator import generate_brief
    brief = generate_brief(state)
    return {"brief": brief}


# ── Exec 1:1 brief assembly ───────────────────────────────────────────────────

def _generate_exec_node(state: GraphState) -> dict:
    from output.exec_brief_generator import generate_exec_brief
    brief = generate_exec_brief(state)
    return {"exec_brief_result": brief.model_dump()}


# ── Query answer assembly ─────────────────────────────────────────────────────

def _generate_answer_node(state: GraphState) -> dict:
    from output.answer_generator import generate_answer
    answer = generate_answer(state)
    answer["loop_count"] = state.get("loop_count", 1)
    return {"lookup_response": answer}


# ── Adaptive query rewrite (for /query workflow only) ─────────────────────────

_ADAPTIVE_PROMPT = """You are a query analyzer for a customer-knowledge retrieval system.
Decide if the query is FOCUSED (single fact/topic) or BROAD (synthesis across topics).

FOCUSED → return ["original query"] (1-element array)
BROAD   → return 2-4 focused sub-queries

Query: {query}

Return ONLY a JSON array of 1-4 strings. No prose. No markdown."""


def _query_rewrite_node(state: GraphState) -> dict:
    """Decompose broad queries; pass focused queries through unchanged."""
    import json
    import logging
    import os
    from langchain_utils import llm_breaker

    _log = logging.getLogger(__name__)
    question = state.get("original_query") or ""

    if llm_breaker.is_open():
        return {"sub_queries": [question],
                "audit_trail": [{"node": "query_rewrite", "fallback": "circuit_open"}]}

    try:
        from langchain_utils import LLM_MODEL
        from .nodes import _strip_json
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL,
                                     google_api_key=os.getenv("GOOGLE_API_KEY"),
                                     temperature=0)
        resp = _llm_invoke_with_retry(llm, _ADAPTIVE_PROMPT.format(query=question))
        content = _strip_json(resp.content.strip())
        sub_queries = json.loads(content)
        if not isinstance(sub_queries, list) or not sub_queries:
            raise ValueError("empty result")
        sub_queries = [s for s in sub_queries if isinstance(s, str) and s.strip()][:4]
        if not sub_queries:
            raise ValueError("no string entries")
        llm_breaker.on_success()
    except Exception as e:
        llm_breaker.on_failure()
        _log.warning("query_rewrite_failed error=%s", str(e))
        sub_queries = [question]

    # For overdue/temporal queries, prepend a sub-query that targets the
    # "OVERDUE by N days" text baked into commitment chunks at ingest.
    if _OVERDUE_INTENT_RE.search(question):
        overdue_sq = "OVERDUE commitment overdue by days tracker"
        if overdue_sq not in sub_queries:
            sub_queries = [overdue_sq] + sub_queries[:3]

    return {"sub_queries": sub_queries,
            "audit_trail": [{"node": "query_rewrite", "sub_queries": sub_queries}]}


# ── Corpus health node ────────────────────────────────────────────────────────

def _corpus_health_node(state: GraphState) -> dict:
    """Fetch corpus health from DB and store in state for the brief generator."""
    import logging
    _log = logging.getLogger(__name__)
    customer_id = state.get("customer_id") or ""
    try:
        from db_utils import get_corpus_health
        health = get_corpus_health(customer_id)
    except Exception as e:
        _log.warning("corpus_health_node_failed customer=%s error=%s", customer_id, e)
        health = {}
    return {"corpus_health": health}


# ── Build pre-meeting brief graph ─────────────────────────────────────────────

def _build_pre_meeting() -> Any:
    graph = StateGraph(GraphState)

    graph.add_node("fetch_corpus_health", _corpus_health_node)
    graph.add_node("overdue_commitments", overdue_commitments_node)
    graph.add_node("account_summary", account_summary_node)
    graph.add_node("open_items", open_items_node)
    graph.add_node("recent_changes", recent_changes_node)
    graph.add_node("outstanding_commitments", outstanding_commitments_node)
    graph.add_node("anticipated_questions", anticipated_questions_node)
    graph.add_node("posture", posture_node)
    graph.add_node("generate_brief", _generate_pre_meeting_node)

    # Fan-out from START: all section nodes run in parallel
    for node in ("fetch_corpus_health", "overdue_commitments", "account_summary", "open_items",
                 "recent_changes", "outstanding_commitments", "anticipated_questions"):
        graph.add_edge(START, node)

    # Fan-in: all section nodes → posture
    for node in ("fetch_corpus_health", "overdue_commitments", "account_summary", "open_items",
                 "recent_changes", "outstanding_commitments", "anticipated_questions"):
        graph.add_edge(node, "posture")

    graph.add_edge("posture", "generate_brief")
    graph.add_edge("generate_brief", END)
    return graph.compile()


# ── Build exec 1:1 brief graph ────────────────────────────────────────────────

def _build_exec_1on1() -> Any:
    graph = StateGraph(GraphState)

    graph.add_node("exec_role_tenure_section", exec_role_tenure_node)
    graph.add_node("exec_stated_position_section", exec_stated_position_node)
    graph.add_node("exec_recent_signals_section", exec_recent_signals_node)
    graph.add_node("exec_open_asks_section", exec_open_asks_node)
    graph.add_node("exec_recommended_approach_section", exec_recommended_approach_node)
    graph.add_node("generate_exec", _generate_exec_node)

    # Fan-out from START: all exec section nodes run in parallel
    for node in ("exec_role_tenure_section", "exec_stated_position_section",
                 "exec_recent_signals_section", "exec_open_asks_section"):
        graph.add_edge(START, node)

    # Fan-in: all exec section nodes → recommended_approach_section
    for node in ("exec_role_tenure_section", "exec_stated_position_section",
                 "exec_recent_signals_section", "exec_open_asks_section"):
        graph.add_edge(node, "exec_recommended_approach_section")

    graph.add_edge("exec_recommended_approach_section", "generate_exec")
    graph.add_edge("generate_exec", END)
    return graph.compile()


# ── Build query graph ─────────────────────────────────────────────────────────

def _build_query() -> Any:
    graph = StateGraph(GraphState)
    graph.add_node("query_rewrite", _query_rewrite_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)
    graph.add_node("generate_answer", _generate_answer_node)

    graph.add_edge(START, "query_rewrite")
    graph.add_edge("query_rewrite", "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", "generate_answer")
    graph.add_edge("generate_answer", END)
    return graph.compile()


# ── Compiled graph singletons ─────────────────────────────────────────────────
_pre_meeting_workflow = _build_pre_meeting()
_exec_1on1_workflow = _build_exec_1on1()
_query_workflow = _build_query()


# ── Public runner functions ───────────────────────────────────────────────────

async def run_pre_meeting_workflow(
    customer_id: str,
    as_of_date: Optional[str] = None,
    last_call_date: Optional[str] = None,
) -> Dict[str, Any]:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    initial_state: GraphState = {
        "customer_id": customer_id,
        "as_of_date": as_of_date or today,
        "last_call_date": last_call_date,
        "today_date": today,
        "person_id": None,
        "original_query": None,
        "sub_queries": [],
        "retrieved_chunks": [],
        "parent_chunks": [],
        "overdue_commitments_data": None,
        "account_summary_text": None,
        "open_items_data": None,
        "recent_changes_data": None,
        "outstanding_commitments_data": None,
        "anticipated_questions_data": None,
        "posture_directives_data": None,
        "corpus_health": None,
        "stale_warnings": [],
        "conflicts_raw": [],
        "section_status": {},
        "audit_trail": [],
    }
    return await _pre_meeting_workflow.ainvoke(initial_state)


async def run_exec_1on1_workflow(
    customer_id: str,
    person_id: str,
    as_of_date: Optional[str] = None,
    last_call_date: Optional[str] = None,
) -> Dict[str, Any]:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    person_name: str = person_id  # fallback to raw id if lookup fails
    person_role: "str | None" = None
    try:
        from db_utils import get_person_by_id
        p = get_person_by_id(int(person_id), customer_id)
        if p:
            person_name = p["name"]
            person_role = p.get("role")
    except Exception:
        pass

    initial_state: GraphState = {
        "customer_id": customer_id,
        "as_of_date": as_of_date or today,
        "last_call_date": last_call_date,
        "today_date": today,
        "person_id": person_id,
        "person_name": person_name,
        "person_role": person_role,
        "original_query": None,
        "sub_queries": [],
        "retrieved_chunks": [],
        "parent_chunks": [],
        "exec_role_tenure": None,
        "exec_stated_position": None,
        "exec_recent_signals": None,
        "exec_open_asks": None,
        "exec_recommended_approach": None,
        "stale_warnings": [],
        "conflicts_raw": [],
        "section_status": {},
        "audit_trail": [],
    }
    return await _exec_1on1_workflow.ainvoke(initial_state)


async def run_query_workflow(
    customer_id: Optional[str],
    question: str,
) -> Dict[str, Any]:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    initial_state: GraphState = {
        "customer_id": customer_id or "default",
        "as_of_date": today,
        "today_date": today,
        "last_call_date": None,
        "person_id": None,
        "original_query": question,
        "sub_queries": [],
        "retrieved_chunks": [],
        "parent_chunks": [],
        "answer_output": None,
        "lookup_response": None,
        "stale_warnings": [],
        "conflicts_raw": [],
        "section_status": {},
        "audit_trail": [],
        "loop_count": 0,
    }
    return await _query_workflow.ainvoke(initial_state)



async def run_workflow(
    customer_id: str,
    query: str,
    brief_type: str = "status",
    focus: str = None,
) -> Dict[str, Any]:
    """Deprecated wrapper — use run_pre_meeting_workflow."""
    return await run_pre_meeting_workflow(customer_id=customer_id)


async def run_lookup_workflow(customer_id: str, question: str) -> Dict[str, Any]:
    """Deprecated wrapper — use run_query_workflow."""
    return await run_query_workflow(customer_id=customer_id, question=question)
