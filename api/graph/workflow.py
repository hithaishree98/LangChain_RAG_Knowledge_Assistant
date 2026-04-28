"""
graph/workflow.py — StateGraph wiring + conditional loop edge.

Flow:
  query_rewrite → retrieve → reason → check_completeness
                                             │
                          ┌──────────────────┤
                          │ is_sufficient OR iteration >= 3
                          ↓                  │ else
                    generate_brief      query_rewrite (loop)

Hard cap: 3 iterations prevents infinite cycles.
"""

from typing import Any, Dict

from langgraph.graph import StateGraph, END

from .state import GraphState
from .nodes import (
    query_rewrite_node,
    retrieve_node,
    reason_node,
    completeness_node,
)


# ── Brief generation node (thin wrapper around brief_generator) ───────────────

def generate_brief_node(state: GraphState) -> dict:
    from output.brief_generator import generate_brief
    brief = generate_brief(state)
    return {"brief": brief}


# ── Routing function ──────────────────────────────────────────────────────────

def _route(state: GraphState) -> str:
    if state["is_sufficient"] or state["iteration_count"] >= 3:
        return "sufficient"
    return "refine"


# ── Build and compile the graph ───────────────────────────────────────────────

def _build() -> Any:
    graph = StateGraph(GraphState)

    graph.add_node("query_rewrite", query_rewrite_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("reason", reason_node)
    graph.add_node("check_completeness", completeness_node)
    graph.add_node("generate_brief", generate_brief_node)

    graph.set_entry_point("query_rewrite")
    graph.add_edge("query_rewrite", "retrieve")
    graph.add_edge("retrieve", "reason")
    graph.add_edge("reason", "check_completeness")

    graph.add_conditional_edges(
        "check_completeness",
        _route,
        {"sufficient": "generate_brief", "refine": "query_rewrite"},
    )

    graph.add_edge("generate_brief", END)
    return graph.compile()


_workflow = _build()


async def run_workflow(customer_id: str, query: str) -> Dict[str, Any]:
    """Invoke the full LangGraph workflow and return the final state."""
    initial_state: GraphState = {
        "customer_id": customer_id,
        "original_query": query,
        "sub_queries": [],
        "retrieved_chunks": [],
        "parent_chunks": [],
        "reasoning_output": None,
        "iteration_count": 0,
        "is_sufficient": False,
        "brief": None,
        "information_gaps": [],
        "audit_trail": [],
    }
    result = await _workflow.ainvoke(initial_state)
    return result
