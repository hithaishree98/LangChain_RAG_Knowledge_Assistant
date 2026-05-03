"""
utils/doc_type_utils.py — Single source of truth for doc_type inference,
validation, and filename convention enforcement.

Used by:
  - api/main.py        (upload endpoint)
  - api/chroma_utils.py (loader dispatch)
  - api/scripts/reindex_with_doctype.py (backfill migration)

Naming convention: YYYY-MM-DD_<keyword>_<descriptor>.<ext>
                or YYYY-Q{1-4}_<keyword>_<descriptor>.<ext>  (for QBR decks)
"""

import os
import re
from typing import Optional, Tuple

VALID_DOC_TYPES = frozenset({
    "transcript",
    "ticket",
    "account_notes",
    "qbr_deck",
    "solution_architecture",
    "commitment_tracker",
})

# Filename keyword rules — checked in order; first match wins.
# More specific patterns (commitment, account-notes) come before general ones (transcript).
_INFERENCE_RULES = [
    (("commitment", "commitments"),                                         "commitment_tracker"),
    (("account-notes", "account_notes", "account-note", "crm-notes"),      "account_notes"),
    (("qbr-deck", "qbr_deck", "slide", "deck", "presentation"),            "qbr_deck"),
    (("qbr", "status-call", "status_call", "incident-review",
      "incident_review", "kickoff", "renewal", "transcript"),              "transcript"),
    (("architecture", "solution", "design", "infra", "system"),            "solution_architecture"),
    (("ticket", "inc-", "bug-", "case-"),                                   "ticket"),
]

# Extension fallbacks when no keyword matched.
# .json is intentionally absent — ticket vs commitment_tracker can't be
# resolved by extension alone; callers must require an explicit value.
_EXT_DEFAULTS = {
    ".pdf":  "qbr_deck",
    ".docx": "qbr_deck",
    ".html": "solution_architecture",
    ".txt":  "transcript",
}

# ── Filename date-prefix patterns ─────────────────────────────────────────────
# Accepts: YYYY-MM-DD_... or YYYY-Qn_...
_ISO_DATE_PREFIX_RE    = re.compile(r"^(\d{4}-\d{2}-\d{2})_")
_QUARTER_PREFIX_RE     = re.compile(r"^(\d{4})-Q([1-4])_")

# Stricter naming pattern used by validate_filename:
# YYYY-MM-DD_<word>_<descriptor>.<ext>  OR  YYYY-Qn_<word>_<descriptor>.<ext>
_NAMING_PATTERN = re.compile(
    r"^(\d{4}-\d{2}-\d{2}|\d{4}-Q[1-4])_([a-z][a-z0-9_-]*)_(.+)\.(txt|pdf|docx|html|csv|json)$",
    re.IGNORECASE,
)

# Quarter → first month of that quarter
_QUARTER_MONTH = {"1": "01", "2": "04", "3": "07", "4": "10"}


def infer_doc_type(filename: str) -> str | None:
    """Infer doc_type from filename keywords then extension.

    Returns None when the type is ambiguous (e.g. .json without keywords).
    Callers should raise a 400 and prompt for explicit doc_type in that case.
    """
    lower = (filename or "").lower()
    for keywords, doc_type in _INFERENCE_RULES:
        if any(kw in lower for kw in keywords):
            return doc_type
    ext = os.path.splitext(lower)[1]
    return _EXT_DEFAULTS.get(ext)


def validate_filename(filename: str) -> Tuple[bool, str]:
    """Check that *filename* follows the required date-prefix naming convention.

    Accepted formats:
      YYYY-MM-DD_<keyword>_<descriptor>.<ext>   e.g. 2024-09-15_transcript_status-call.txt
      YYYY-Qn_<keyword>_<descriptor>.<ext>      e.g. 2024-Q3_qbr_deck.pdf

    Returns (True, "") when valid, or (False, <human-readable error>) otherwise.
    """
    name = os.path.basename(filename or "")
    m = _NAMING_PATTERN.match(name)
    if not m:
        return (
            False,
            f"Filename '{name}' does not match the required format "
            f"YYYY-MM-DD_<keyword>_<descriptor>.<ext>  "
            f"(e.g. 2024-09-15_transcript_status-call.txt or 2024-Q3_qbr_deck.pdf)",
        )
    inferred = infer_doc_type(name)
    if inferred is None:
        doc_type_part = m.group(2)
        return (
            False,
            f"Cannot determine document type from keyword '{doc_type_part}'. "
            f"Use one of: transcript, qbr_deck, ticket, commitment_tracker, "
            f"solution_architecture, account_notes",
        )
    return True, ""


def extract_date_from_filename(filename: str) -> Optional[str]:
    """Return the ISO date string (YYYY-MM-DD) embedded in *filename*.

    Returns None when neither a full ISO date nor a quarter prefix is found.
    Quarter shorthand is expanded to the first day of the quarter:
      2024-Q3  →  2024-07-01
    """
    name = os.path.basename(filename or "")
    m = _ISO_DATE_PREFIX_RE.match(name)
    if m:
        return m.group(1)
    m = _QUARTER_PREFIX_RE.match(name)
    if m:
        year, quarter = m.group(1), m.group(2)
        return f"{year}-{_QUARTER_MONTH[quarter]}-01"
    return None


def normalize_doc_type(doc_type: str) -> str:
    """Normalize legacy or variant doc_type values to canonical names.

    Raises ValueError for values that are neither a known alias nor already
    in VALID_DOC_TYPES — prevents bad data from reaching Chroma.
    """
    _NORMALIZE = {
        "tickets":            "ticket",
        "commitments":        "commitment_tracker",
        "plain_text":         "transcript",
        "notes":              "transcript",
        "generic_json":       "ticket",
    }
    normalized = _NORMALIZE.get(doc_type, doc_type)
    if normalized not in VALID_DOC_TYPES:
        raise ValueError(
            f"Unknown doc_type {doc_type!r}. "
            f"Valid types: {sorted(VALID_DOC_TYPES)}"
        )
    return normalized
