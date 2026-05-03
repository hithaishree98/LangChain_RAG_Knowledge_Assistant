"""
utils/staleness.py — Single source of truth for document freshness checks.

Previously brief_generator.py used as_of_date as the reference and
answer_generator.py used datetime.now() — two different staleness semantics
in the same codebase. This module unifies both under one function:

  is_stale(doc_date, reference_date=None)
    reference_date=None  →  use today (answer_generator behavior)
    reference_date=str   →  use that date (brief_generator behavior)

STALE_DAYS is the single threshold for the whole system.
"""

from datetime import datetime, timezone
from typing import Optional

STALE_DAYS = int(30)   # docs older than this many days relative to the reference date are stale


def is_stale(doc_date: str, reference_date: Optional[str] = None) -> bool:
    """Return True if doc_date is more than STALE_DAYS before reference_date.

    Args:
        doc_date:       ISO date string (YYYY-MM-DD) of the source document.
        reference_date: ISO date string to compare against. Defaults to today UTC.

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
        return (ref_dt - doc_dt).days > STALE_DAYS
    except Exception:
        return False


def recency_flag(doc_date: str, reference_date: Optional[str] = None) -> Optional[str]:
    """Return 'stale' if the document is stale, else None."""
    return "stale" if is_stale(doc_date, reference_date) else None
