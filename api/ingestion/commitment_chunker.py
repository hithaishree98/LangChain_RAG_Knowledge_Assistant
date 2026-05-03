"""
commitment_chunker.py — Turn a list of Commitment objects into LangChain Documents.

Strategy: one Document per commitment. All structured fields are stored as both
chunk metadata (for filter-based retrieval) and in page_content (for LLM reading).
No parent-child split is applied — each commitment is its own natural unit.
"""

from typing import List
from langchain_core.documents import Document
from .commitment_parser import Commitment


def chunk(commitments: List[Commitment], source: str = "") -> List[Document]:
    """Convert a list of Commitment objects into LangChain Documents."""
    docs: List[Document] = []
    for c in commitments:
        is_slipped = c.is_slipped
        content_lines = [
            f"Commitment {c.commitment_id}: {c.description}",
            f"Promised: {c.promised_date} | Current target: {c.current_target_date} | Status: {c.status}",
            f"Owner: {c.owner}",
        ]
        if c.source_doc:
            content_lines.append(f"Source: {c.source_doc}" + (f" — {c.source_section}" if c.source_section else ""))
        content_lines.append(f"Customer aware: {c.customer_aware} | Last updated: {c.last_updated}")

        docs.append(Document(
            page_content="\n".join(content_lines),
            metadata={
                "source": source,
                "doc_type": "commitment_tracker",
                "commitment_id": c.commitment_id,
                "commitment_status": c.status,
                "promised_date": c.promised_date,
                "current_target_date": c.current_target_date,
                "owner": c.owner,
                "customer_aware": str(c.customer_aware).lower(),
                "is_slipped": str(is_slipped).lower(),
                # doc_date points to the last_updated field so recency ranking
                # treats it as "how recent is this commitment record"
                "doc_date": c.last_updated[:10] if len(c.last_updated) >= 10 else c.last_updated,
            },
        ))
    return docs
