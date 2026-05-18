"""
transcript_chunker.py — Turn a list of Turn objects into LangChain Documents.

Strategy: one Document per speaker turn. Consecutive same-speaker turns are merged
when their combined word count stays within MAX_WORDS_PER_CHUNK. Each chunk carries
the primary speaker in metadata so person-based exec node filtering works cleanly —
one chunk embeds one person's meaning, not a blend of all participants.

A leading [Prior: Speaker: text] context line in each chunk gives the LLM enough
surrounding context to understand what the speaker is responding to without mixing
multiple speakers' semantics into a single embedding.
"""

import os
import re
from typing import List
from langchain_core.documents import Document
from .transcript_parser import Turn

MIN_WORDS_PER_TURN = 12    # turns shorter than this are merged with adjacent same-speaker turns
MAX_WORDS_PER_CHUNK = 120  # cap merged same-speaker runs to prevent monologue-length chunks

_MEETING_TYPE_RE = re.compile(
    r"(qbr|status[-_]call|status|incident[-_]review|incident|kickoff|renewal)",
    re.I,
)


def _extract_meeting_type(source: str) -> str:
    """Extract meeting type keyword from filename. Returns empty string if not found."""
    name = os.path.basename(source or "").lower()
    m = _MEETING_TYPE_RE.search(name)
    return m.group(1).replace("_", "-") if m else ""


def _merge_short_turns(turns: List[Turn]) -> List[Turn]:
    """Merge consecutive same-speaker turns and turns shorter than MIN_WORDS_PER_TURN.

    Very short utterances ("OK", "Right", "Got it") create noisy single-word chunks
    that hurt retrieval precision. Merge them with the next same-speaker turn.
    Cap merged runs at MAX_WORDS_PER_CHUNK so long monologues still get split.
    """
    if not turns:
        return []

    result = []
    i = 0
    while i < len(turns):
        speaker = turns[i].speaker
        text = turns[i].text
        start_ms = turns[i].start_ms
        word_count = len(text.split())

        # Merge consecutive same-speaker turns while within the word cap
        while (i + 1 < len(turns)
               and turns[i + 1].speaker == speaker
               and word_count + len(turns[i + 1].text.split()) <= MAX_WORDS_PER_CHUNK):
            i += 1
            text = text + " " + turns[i].text
            word_count = len(text.split())

        result.append(Turn(speaker=speaker, text=text.strip(), start_ms=start_ms))
        i += 1

    return result


def chunk(turns: List[Turn], source: str = "") -> List[Document]:
    """Convert transcript turns into per-speaker-segment LangChain Documents."""
    if not turns:
        return []

    meeting_type = _extract_meeting_type(source)
    all_speakers = sorted({t.speaker for t in turns})

    merged = _merge_short_turns(turns)

    docs: List[Document] = []
    for idx, turn in enumerate(merged):
        # Include the preceding speaker's turn as a context line so the LLM can
        # reason about what this speaker is responding to, without mixing both
        # speakers' semantics into the same embedding vector.
        prior_line = ""
        if idx > 0:
            prev = merged[idx - 1]
            prior_text = prev.text[:100] + ("..." if len(prev.text) > 100 else "")
            prior_line = f"[Prior: {prev.speaker}: {prior_text}]\n\n"

        content = f"{prior_line}{turn.speaker}: {turn.text}"

        meta = {
            "source": source,
            "doc_type": "transcript",
            "speaker": turn.speaker,        # primary speaker — used by _person_filter in exec nodes
            "speakers": turn.speaker,       # kept for backward-compat with any filters on "speakers"
            "all_speakers": ", ".join(all_speakers),
            "start_ms": turn.start_ms,
            "chunk_index": idx,
        }
        if meeting_type:
            meta["meeting_type"] = meeting_type

        docs.append(Document(page_content=content, metadata=meta))

    return docs
