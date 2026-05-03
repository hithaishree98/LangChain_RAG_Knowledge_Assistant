"""
transcript_chunker.py — Turn a list of Turn objects into LangChain Documents.

Strategy: group consecutive turns into chunks of ~TARGET_WORDS words with
OVERLAP_TURNS turn overlap so context is not lost at boundaries.
Each chunk includes speaker attribution so the LLM can reason about
who said what.
"""

import os
import re
from typing import List
from langchain_core.documents import Document
from .transcript_parser import Turn

TARGET_WORDS = 300
OVERLAP_TURNS = 2

_MEETING_TYPE_RE = re.compile(
    r"(qbr|status[-_]call|status|incident[-_]review|incident|kickoff|renewal)",
    re.I,
)


def _extract_meeting_type(source: str) -> str:
    """Extract meeting type keyword from filename. Returns empty string if not found."""
    name = os.path.basename(source or "").lower()
    m = _MEETING_TYPE_RE.search(name)
    return m.group(1).replace("_", "-") if m else ""


def chunk(turns: List[Turn], source: str = "") -> List[Document]:
    """Convert transcript turns into chunked LangChain Documents."""
    if not turns:
        return []

    meeting_type = _extract_meeting_type(source)
    all_speakers = sorted({t.speaker for t in turns})

    chunks: List[Document] = []
    i = 0
    while i < len(turns):
        # Accumulate turns until we reach TARGET_WORDS
        group: List[Turn] = []
        word_count = 0
        j = i
        while j < len(turns) and word_count < TARGET_WORDS:
            group.append(turns[j])
            word_count += len(turns[j].text.split())
            j += 1

        text = _format_group(group)
        meta = {
            "source": source,
            "doc_type": "transcript",
            "speakers": ", ".join(sorted({t.speaker for t in group})),
            "all_speakers": ", ".join(all_speakers),
            "start_ms": group[0].start_ms,
        }
        if meeting_type:
            meta["meeting_type"] = meeting_type
        chunks.append(Document(page_content=text, metadata=meta))

        # Advance, leaving OVERLAP_TURNS for context continuity
        i = max(i + 1, j - OVERLAP_TURNS)

    return chunks


def _format_group(turns: List[Turn]) -> str:
    lines = []
    for t in turns:
        lines.append(f"{t.speaker}: {t.text}")
    return "\n".join(lines)
