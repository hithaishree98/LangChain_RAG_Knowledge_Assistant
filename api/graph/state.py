"""
graph/state.py — LangGraph state schema for three distinct workflows.

Pre-meeting brief:  customer_id + as_of_date → PreMeetingBrief sections
Exec 1:1 brief:     customer_id + person_id  → ExecBrief sections
Query:              question + customer_id   → answer + citations

All workflows share this TypedDict. Each workflow only writes to its own slots;
unused slots stay None/empty without causing errors.
"""

import operator
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict
from langchain_core.documents import Document


def _merge_dicts(a: Dict, b: Dict) -> Dict:
    """Merge two dicts; used as reducer for section_status in parallel fan-out."""
    return {**a, **b}


class GraphState(TypedDict, total=False):
    # ── Shared identity ────────────────────────────────────────────────────────
    customer_id: str
    as_of_date: str             # YYYY-MM-DD; defaults to today
    last_call_date: Optional[str]   # from customers table, drives "since last call" window
    person_id: Optional[str]     # exec 1:1 only — raw ID from request
    person_name: Optional[str]   # resolved from person_id; used in prompts
    person_role: Optional[str]   # resolved from person_id; used in prompts

    # ── Query workflow ─────────────────────────────────────────────────────────
    original_query: Optional[str]
    sub_queries: List[str]
    retrieved_chunks: List[Document]
    parent_chunks: List[Document]
    answer_output: Optional[Dict[str, Any]]  # raw from answer_node
    lookup_response: Optional[Dict[str, Any]]  # assembled QueryResult dict

    # ── Brief assembly output (written by generate_brief / generate_exec nodes) ─
    brief: Optional[Dict[str, Any]]             # pre-meeting brief result dict
    exec_brief_result: Optional[Dict[str, Any]]  # exec 1:1 brief result dict

    # ── Pre-meeting brief section data (raw dicts from nodes) ─────────────────
    overdue_commitments_data: Optional[List[Dict[str, Any]]]
    account_summary_text: Optional[str]
    open_items_data: Optional[List[Dict[str, Any]]]
    recent_changes_data: Optional[List[Dict[str, Any]]]
    outstanding_commitments_data: Optional[List[Dict[str, Any]]]
    anticipated_questions_data: Optional[List[Dict[str, Any]]]
    posture_directives_data: Optional[List[Dict[str, Any]]]

    # ── Exec 1:1 brief section data ────────────────────────────────────────────
    exec_role_tenure: Optional[str]
    exec_stated_position: Optional[List[Dict[str, Any]]]
    exec_recent_signals: Optional[List[Dict[str, Any]]]
    exec_open_asks: Optional[List[Dict[str, Any]]]
    exec_recommended_approach: Optional[str]

    # ── Cross-section accumulators (fan-out safe with Annotated reducer) ───────
    stale_warnings: Annotated[List[str], operator.add]
    conflicts_raw: Annotated[List[Dict[str, Any]], operator.add]

    # ── Corpus health (pre-meeting only) ──────────────────────────────────────
    corpus_health: Optional[Dict[str, Any]]  # from db_utils.get_corpus_health()

    # ── Section source provenance (populated by each section node) ───────────
    open_items_sources: Optional[List[str]]
    open_items_as_of: Optional[str]
    recent_changes_sources: Optional[List[str]]
    recent_changes_as_of: Optional[str]
    account_summary_sources: Optional[List[str]]
    account_summary_as_of: Optional[str]
    anticipated_questions_sources: Optional[List[str]]
    overdue_sources: Optional[List[str]]
    overdue_as_of: Optional[str]
    outstanding_sources: Optional[List[str]]
    outstanding_as_of: Optional[str]

    # ── Tracking ───────────────────────────────────────────────────────────────
    section_status: Annotated[Dict[str, str], _merge_dicts]   # section → "ok" | "empty" | "unavailable"
    today_date: Optional[str]        # YYYY-MM-DD; injected at workflow start for testability
    audit_trail: Annotated[List[Dict[str, Any]], operator.add]
    loop_count: Annotated[int, operator.add]  # incremented by answer_node each invocation
