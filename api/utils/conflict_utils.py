"""
utils/conflict_utils.py — Detect contradictory claims across source documents.

When two chunks from different files make different statements about the same
entity (an SLA percentage, a date, a ticket status, a version number),
surface both claims rather than letting the LLM pick one.

This is a heuristic approach:
  - Layer 1 (here): regex-based entity extraction + cross-source comparison
  - Layer 3 (langchain_utils): LLM judge catches relational conflicts
    (e.g. "X agreed to Y" vs "X declined Y") that regex cannot catch.

False positives (flagging non-conflicts) are better than false negatives
(missing real conflicts) — FDE verifies manually.
"""
import re
import os
import logging
from typing import Any, Dict, List, Optional
from langchain_core.documents import Document

_log = logging.getLogger(__name__)

# ── Entity extraction patterns ────────────────────────────────────────────────

_PERCENT_RE = re.compile(r'\b(\d+(?:\.\d+)?)\s*%')
_DATE_RE = re.compile(r'\b(\d{4}-\d{2}-\d{2})\b')
_VERSION_RE = re.compile(r'\bv?(\d+\.\d+(?:\.\d+)?)\b')
_TICKET_RE = re.compile(r'\b([A-Z]+-\d{3,6})\b')
_DAYS_RE = re.compile(r'\b(\d+)\s+(?:business\s+)?days?\b', re.I)


def extract_entities(text: str) -> Dict[str, List[str]]:
    """Extract typed entities from a text string."""
    return {
        "percentages": list(set(_PERCENT_RE.findall(text))),
        "dates":       list(set(_DATE_RE.findall(text))),
        "versions":    list(set(_VERSION_RE.findall(text))),
        "tickets":     list(set(_TICKET_RE.findall(text))),
        "day_counts":  list(set(_DAYS_RE.findall(text))),
    }


def detect_conflicts(
    claims: List[str],
    docs: List[Document],
    max_conflicts: int = 5,
) -> List[Dict[str, Any]]:
    """
    Detect contradictions across source documents.

    Strategy:
      1. Group docs by source filename.
      2. For each doc, extract numeric/typed entities.
      3. If two different sources have the same entity TYPE but different VALUES
         in a context that looks like the same fact, flag it as a conflict.

    Returns a list of conflict dicts compatible with the Conflict pydantic model.
    """
    if not docs or len(docs) < 2:
        return []

    # Group by source filename
    by_source: Dict[str, List[Document]] = {}
    for doc in docs:
        src = doc.metadata.get("filename") or doc.metadata.get("source") or "unknown"
        src = os.path.basename(src)
        by_source.setdefault(src, []).append(doc)

    if len(by_source) < 2:
        return []

    sources = list(by_source.keys())
    conflicts = []

    # Compare every pair of sources
    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            src_a, src_b = sources[i], sources[j]
            text_a = " ".join(d.page_content for d in by_source[src_a])
            text_b = " ".join(d.page_content for d in by_source[src_b])

            ents_a = extract_entities(text_a)
            ents_b = extract_entities(text_b)

            # Check for contradictions in numeric/typed entities
            for entity_type in ("percentages", "versions", "day_counts"):
                vals_a = set(ents_a.get(entity_type, []))
                vals_b = set(ents_b.get(entity_type, []))
                if not vals_a or not vals_b:
                    continue
                # Values differ AND there's overlap in surrounding context
                if vals_a != vals_b and _context_overlaps(text_a, text_b):
                    # Find the specific sentences containing these values
                    claim_a = _find_sentence_with_value(text_a, list(vals_a)[0])
                    claim_b = _find_sentence_with_value(text_b, list(vals_b)[0])
                    if claim_a and claim_b and claim_a != claim_b:
                        date_a = by_source[src_a][0].metadata.get("doc_date", "")
                        date_b = by_source[src_b][0].metadata.get("doc_date", "")
                        conflicts.append({
                            "claim_a": claim_a,
                            "source_a": {
                                "document": src_a,
                                "doc_date": date_a,
                                "location": "",
                                "is_stale": False,
                                "is_latest_version": True,
                            },
                            "claim_b": claim_b,
                            "source_b": {
                                "document": src_b,
                                "doc_date": date_b,
                                "location": "",
                                "is_stale": False,
                                "is_latest_version": True,
                            },
                        })
                        if len(conflicts) >= max_conflicts:
                            return conflicts

    return conflicts


def _context_overlaps(text_a: str, text_b: str, min_overlap: int = 2) -> bool:
    """
    Check if two texts share enough context words to indicate they're
    describing the same fact (not just coincidentally having the same number).
    """
    _STOP = {"the", "a", "an", "is", "are", "was", "were", "in", "of",
             "to", "for", "and", "or", "with", "at", "by", "on", "this"}
    words_a = {w.lower() for w in re.findall(r'\b[a-z]{4,}\b', text_a.lower())} - _STOP
    words_b = {w.lower() for w in re.findall(r'\b[a-z]{4,}\b', text_b.lower())} - _STOP
    overlap = len(words_a & words_b)
    return overlap >= min_overlap


def _find_sentence_with_value(text: str, value: str) -> Optional[str]:
    """Find the sentence in text that contains the given value."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for sent in sentences:
        if value in sent:
            return sent.strip()[:200]
    return None
