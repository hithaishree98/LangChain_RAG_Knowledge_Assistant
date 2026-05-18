"""
commitment_chunker.py — Turn a list of Commitment objects into LangChain Documents.

Strategy: one Document per commitment. All structured fields are stored as both
chunk metadata (for filter-based retrieval) and in page_content (for LLM reading).
No parent-child split is applied — each commitment is its own natural unit.
"""

from datetime import datetime
from typing import List
from langchain_core.documents import Document
from .commitment_parser import Commitment


def chunk(commitments: List[Commitment], source: str = "", today: str = "") -> List[Document]:
    """Convert a list of Commitment objects into LangChain Documents.

    today: YYYY-MM-DD reference date for computing is_overdue/days_overdue.
           Defaults to UTC today when omitted.
    """
    if not today:
        today = datetime.utcnow().strftime("%Y-%m-%d")

    docs: List[Document] = []
    for c in commitments:
        is_slipped = c.is_slipped
        target = c.current_target_date or c.promised_date or ""

        # Deterministic overdue check — never delegated to the LLM
        is_overdue = bool(target and c.is_open and target < today)
        days_overdue = 0
        if is_overdue:
            try:
                days_overdue = (
                    datetime.strptime(today, "%Y-%m-%d") - datetime.strptime(target, "%Y-%m-%d")
                ).days
            except Exception:
                pass

        content_lines = [
            f"Commitment {c.commitment_id}: {c.description}",
        ]
        if is_overdue:
            content_lines.append(f"OVERDUE by {days_overdue} days (target was {target})")
        content_lines.extend([
            f"Promised: {c.promised_date} | Current target: {target} | Status: {c.status}",
            f"Owner: {c.owner}",
        ])
        if c.source_doc:
            content_lines.append(f"Source: {c.source_doc}" + (f" — {c.source_section}" if c.source_section else ""))
        content_lines.append(f"Customer aware: {c.customer_aware} | Last updated: {c.last_updated}")
        # For long descriptions the embedding model down-weights distant text, so
        # repeat the OVERDUE signal at the end to ensure it has retrieval weight.
        if is_overdue and len(c.description.split()) > 40:
            content_lines.append(f"STATUS: OVERDUE by {days_overdue} days")

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
                "is_open": str(c.is_open).lower(),
                "is_overdue": str(is_overdue).lower(),
                "days_overdue": days_overdue,
                # doc_date points to the last_updated field so recency ranking
                # treats it as "how recent is this commitment record"
                "doc_date": c.last_updated[:10] if len(c.last_updated) >= 10 else c.last_updated,
            },
        ))
    return docs
