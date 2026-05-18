"""
utils/staleness.py — Single source of truth for document freshness checks.

Previously brief_generator.py used as_of_date as the reference and
answer_generator.py used datetime.now() — two different staleness semantics
in the same codebase. This module unifies both under one function:

  is_stale(doc_date, reference_date=None, doc_type=None)
    reference_date=None  →  use today (answer_generator behavior)
    reference_date=str   →  use that date (brief_generator behavior)
    doc_type             →  use per-type threshold when provided

STALE_DAYS is the default threshold for unknown or unspecified doc types.
"""

from datetime import datetime, timezone
from typing import Optional

STALE_DAYS = int(30)   # default; used by callers that don't know the doc type

# Per-doc-type staleness thresholds (days).
# Transcripts and tickets are transient — 30 days is appropriate.
# Commitment trackers move fast — flag anything older than two weeks.
# Reference documents (QBR decks, architecture docs) age slowly.
_STALE_DAYS_BY_TYPE: dict = {
    "transcript":           30,
    "ticket":               30,
    "commitment_tracker":   14,
    "account_notes":        30,
    "qbr_deck":             90,
    "solution_architecture": 180,
}


def get_stale_days(doc_type: Optional[str] = None) -> int:
    """Return the staleness threshold in days for the given doc_type."""
    if doc_type:
        return _STALE_DAYS_BY_TYPE.get(doc_type, STALE_DAYS)
    return STALE_DAYS


def is_stale(
    doc_date: str,
    reference_date: Optional[str] = None,
    doc_type: Optional[str] = None,
) -> bool:
    """Return True if doc_date is more than get_stale_days(doc_type) before reference_date.

    Args:
        doc_date:       ISO date string (YYYY-MM-DD) of the source document.
        reference_date: ISO date string to compare against. Defaults to today UTC.
        doc_type:       Optional doc type for per-type threshold lookup.

    Returns False for empty/unparseable dates rather than raising.
    """
    if not doc_date:
        return False
    try:
        doc_dt = datetime.strptime(doc_date[:10], "%Y-%m-%d")
        if reference_date:
            ref_dt = datetime.strptime(reference_date[:10], "%Y-%m-%d")
        else:
            ref_dt = datetime.now(timezone.utc).replace(tzinfo=None)
        threshold = get_stale_days(doc_type)
        return (ref_dt - doc_dt).days > threshold
    except Exception:
        return False


def recency_flag(
    doc_date: str,
    reference_date: Optional[str] = None,
    doc_type: Optional[str] = None,
) -> Optional[str]:
    """Return 'stale' if the document is stale, else None."""
    return "stale" if is_stale(doc_date, reference_date, doc_type) else None
