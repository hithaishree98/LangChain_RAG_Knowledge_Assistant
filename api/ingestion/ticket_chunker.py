"""
ticket_chunker.py — Turn a TicketStructure into LangChain Documents.

Strategy: one Document per logical section so retrieval can pinpoint
which part of a ticket is relevant.
  - Chunk 1: subject + description
  - Chunk N: each comment (if non-trivial)
  - Last chunk: resolution (if present)
"""

from typing import List
from langchain_core.documents import Document
from .ticket_parser import TicketStructure

MIN_BODY_LEN = 30  # skip comments shorter than this


def chunk(ticket: TicketStructure, source: str = "") -> List[Document]:
    """Convert a TicketStructure into chunked LangChain Documents."""
    docs: List[Document] = []
    base_meta = {
        "source": source,
        "doc_type": "ticket",
        "ticket_id": ticket.ticket_id,
        "priority": ticket.priority,
        "status": ticket.status,
        "reporter": ticket.reporter,
        "assignee": ticket.assignee,
        "updated_at": ticket.updated_at,
    }

    # ── Subject + description ─────────────────────────────────────────────────
    header = f"Ticket {ticket.ticket_id}"
    if ticket.priority:
        header += f" [{ticket.priority}]"
    if ticket.subject:
        header += f": {ticket.subject}"

    body_parts = [header]
    if ticket.description:
        body_parts.append(ticket.description)
    if ticket.created_at:
        body_parts.append(f"Created: {ticket.created_at}")

    docs.append(Document(
        page_content="\n".join(body_parts),
        metadata={**base_meta, "section": "description"},
    ))

    # ── Comments ──────────────────────────────────────────────────────────────
    for i, comment in enumerate(ticket.comments):
        if len(comment.body) < MIN_BODY_LEN:
            continue
        text = f"Comment by {comment.author}"
        if comment.created_at:
            text += f" ({comment.created_at})"
        text += f":\n{comment.body}"
        docs.append(Document(
            page_content=text,
            metadata={**base_meta, "section": "comment", "comment_index": i},
        ))

    # ── Resolution ────────────────────────────────────────────────────────────
    if ticket.resolution and len(ticket.resolution) >= MIN_BODY_LEN:
        docs.append(Document(
            page_content=f"Resolution:\n{ticket.resolution}",
            metadata={**base_meta, "section": "resolution"},
        ))

    return docs
