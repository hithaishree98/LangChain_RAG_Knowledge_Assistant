"""
graph/state.py — LangGraph state schema.

This TypedDict is the single contract between all nodes in the workflow.
Every node reads from this state and returns a partial update dict.
Define it first — getting this wrong means refactoring all nodes.
"""

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict
from langchain_core.documents import Document


class GraphState(TypedDict):
    # Identity
    customer_id: str
    original_query: str

    # Retrieval
    sub_queries: List[str]
    retrieved_chunks: List[Document]   # child chunks from HybridRetriever
    parent_chunks: List[Document]      # expanded context fetched by parent_chunk_id

    # Reasoning
    reasoning_output: Optional[Dict[str, Any]]  # structured JSON from LLM analyst

    # Loop control
    iteration_count: int
    is_sufficient: bool

    # Output
    brief: Optional[Dict[str, Any]]
    information_gaps: List[str]

    # Audit
    audit_trail: List[Dict[str, Any]]
