"""
ticket_chunker.py — Turn a TicketStructure into LangChain Documents.

Strategy: one Document per logical section so retrieval can pinpoint
which part of a ticket is relevant.
  - Chunk 1: subject + description
  - Chunk N: each comment (if non-trivial)
  - Last chunk: resolution (if present)

Long sections (> MAX_SECTION_WORDS) are split with sentence-aware overlap
so the embedding model never receives a truncated context window.
"""

import re
from typing import List
from langchain_core.documents import Document
from .ticket_parser import TicketStructure

MIN_BODY_LEN = 15        # skip comments shorter than this (chars)
MAX_SECTION_WORDS = 350  # split sections larger than this to avoid embedding truncation


def _split_long_text(text: str, max_words: int = MAX_SECTION_WORDS, overlap_words: int = 50) -> List[str]:
    """Split text into word-capped segments at sentence boundaries with word overlap.

    Used to prevent embedding model truncation on long ticket descriptions/comments.
    Splits at sentence endings first; falls back to whitespace if no sentence boundary found.
    """
    if overlap_words >= max_words:
        raise ValueError(
            f"overlap_words ({overlap_words}) must be less than max_words ({max_words})"
        )
    words = text.split()
    if len(words) <= max_words:
        return [text]

    # Split into sentences using a simple regex (avoids heavy NLP dependency here)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) <= 1:
        # No sentence boundaries — split by word count with overlap
        segments = []
        i = 0
        while i < len(words):
            segment = " ".join(words[i: i + max_words])
            segments.append(segment)
            i += max_words - overlap_words
        return segments

    segments: List[str] = []
    current: List[str] = []
    current_words = 0

    i = 0
    while i < len(sentences):
        sent = sentences[i]
        sent_words = len(sent.split())
        if current_words + sent_words > max_words and current:
            segments.append(" ".join(current))
            # Rewind by overlap_words worth of sentences
            overlap_buf: List[str] = []
            overlap_count = 0
            for prev in reversed(current):
                pw = len(prev.split())
                if overlap_count + pw > overlap_words:
                    break
                overlap_buf.insert(0, prev)
                overlap_count += pw
            current = overlap_buf
            current_words = overlap_count
        current.append(sent)
        current_words += sent_words
        i += 1

    if current:
        segments.append(" ".join(current))

    return segments if segments else [text]


def chunk(ticket: TicketStructure, source: str = "") -> List[Document]:
    """Convert a TicketStructure into chunked LangChain Documents."""
    docs: List[Document] = []
    base_meta = {
        "source": source,
        "doc_type": "ticket",
        "ticket_id": ticket.ticket_id,
        "priority": ticket.priority,
        "status": ticket.status,
        "is_open": str(ticket.is_open).lower(),
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

    # Prepend status explicitly so the LLM never infers open/closed from description
    # language (e.g. "mitigated" or "workaround applied" looks resolved without this).
    desc_text = f"Status: {ticket.status}\n{ticket.description}" if ticket.description else f"Status: {ticket.status}"
    date_parts = []
    if ticket.created_at:
        date_parts.append(f"Created: {ticket.created_at}")
    if ticket.updated_at:
        date_parts.append(f"Updated: {ticket.updated_at}")
    if date_parts:
        suffix = " | ".join(date_parts)
        desc_text = (desc_text + f"\n{suffix}") if desc_text else suffix

    desc_segments = _split_long_text(desc_text) if desc_text else [""]
    for seg_i, seg in enumerate(desc_segments):
        page = f"{header}\n{seg}".strip() if seg else header
        meta = {**base_meta, "section": "description"}
        if len(desc_segments) > 1:
            meta["section_part"] = seg_i
        docs.append(Document(page_content=page, metadata=meta))

    # ── Comments ──────────────────────────────────────────────────────────────
    for i, comment in enumerate(ticket.comments):
        if len(comment.body) < MIN_BODY_LEN:
            continue
        prefix = f"Comment by {comment.author}"
        if comment.created_at:
            prefix += f" ({comment.created_at})"
        prefix += ":"

        for seg_i, seg in enumerate(_split_long_text(comment.body)):
            meta = {**base_meta, "section": "comment", "comment_index": i}
            if seg_i > 0:
                meta["section_part"] = seg_i
            docs.append(Document(
                page_content=f"{header}\n{prefix}\n{seg}",
                metadata=meta,
            ))

    # ── Resolution ────────────────────────────────────────────────────────────
    if ticket.resolution and len(ticket.resolution) >= MIN_BODY_LEN:
        for seg_i, seg in enumerate(_split_long_text(ticket.resolution)):
            meta = {**base_meta, "section": "resolution"}
            if seg_i > 0:
                meta["section_part"] = seg_i
            docs.append(Document(
                page_content=f"{header}\nResolution:\n{seg}",
                metadata=meta,
            ))

    return docs
