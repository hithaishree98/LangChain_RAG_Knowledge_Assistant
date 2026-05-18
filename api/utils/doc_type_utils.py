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
from typing import List, Optional, Tuple

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


def check_content_descriptor_consistency(
    filename: str, doc_type: str, content_sample: str
) -> List[str]:
    """Return warning strings when file content doesn't match its filename descriptor.

    Warnings are advisory only — they do not block the upload.
    ``content_sample`` should be the first ~4 KB of the file decoded as UTF-8.
    """
    warnings: List[str] = []
    lower = content_sample.lower()

    if doc_type == "transcript":
        # A real transcript should have speaker-label patterns
        has_labels = bool(
            re.search(r"^[A-Za-z][A-Za-z0-9 _-]{1,40}:\s+\S", content_sample, re.MULTILINE)
        )
        has_keywords = any(kw in lower for kw in ("speaker", "interviewer", "moderator", "host"))
        if not has_labels and not has_keywords:
            warnings.append(
                f"'{filename}' is marked as a transcript but no speaker labels were found in "
                f"the content. It will be indexed as plain text."
            )

    elif doc_type == "commitment_tracker":
        # Should contain at least two commitment-related field signals
        signals = sum([
            any(kw in lower for kw in ("commitment", "promised", "due_date", "target_date")),
            "status" in lower,
            any(kw in lower for kw in ("owner", "assignee", "responsible")),
            any(kw in lower for kw in ("delivered", "slipped", "deferred", "active")),
        ])
        if signals < 2:
            warnings.append(
                f"'{filename}' is marked as a commitment tracker but the content does not "
                f"appear to contain commitment records. Verify the file type."
            )

    elif doc_type == "ticket":
        # Should contain ticket-shaped fields
        signals = sum([
            any(kw in lower for kw in ("ticket", "issue", "bug", "case", "incident", "id")),
            any(kw in lower for kw in ("status", "priority", "severity")),
            any(kw in lower for kw in ("assignee", "reporter", "created_at", "description")),
        ])
        if signals < 2:
            warnings.append(
                f"'{filename}' is marked as a ticket but the content does not appear to "
                f"contain ticket fields. Verify the file type."
            )

    return warnings


def sniff_doc_type(content_bytes: bytes, filename: str) -> tuple[str | None, str]:
    """Detect document type from content, independent of filename convention.

    Returns (detected_type, confidence) where:
      detected_type : one of VALID_DOC_TYPES, or None when indeterminate
      confidence    : "high" | "medium" | "low"

    Detection hierarchy:
      1. Extension + content structure for structured formats (JSON, CSV)
      2. Content patterns for plain text (speaker labels → transcript)
      3. Extension-only for binary formats (PDF, DOCX, HTML) — always "low"

    This is used by the upload endpoint to pre-fill the doc_type so users do
    not need to rename files to follow the internal naming convention.
    """
    ext = os.path.splitext(filename.lower())[1]

    # ── JSON ─────────────────────────────────────────────────────────────────
    if ext == ".json":
        try:
            sample = content_bytes[:8192].decode("utf-8", errors="replace")
            obj = __import__("json").loads(sample)
        except Exception:
            return None, "low"

        # Top-level array or {"commitments": [...]}
        items = obj if isinstance(obj, list) else obj.get("commitments") or []
        if items and isinstance(items, list):
            first = items[0] if items else {}
            keys = {k.lower() for k in (first.keys() if isinstance(first, dict) else [])}
            commitment_signals = keys & {"commitment_id", "promised_date", "current_target_date",
                                         "target_date", "is_slipped", "is_open"}
            ticket_signals = keys & {"ticket_id", "subject", "priority", "assignee",
                                     "reporter", "resolution"}
            if commitment_signals:
                return "commitment_tracker", "high"
            if ticket_signals:
                return "ticket", "high"

        # Flat single-object ticket (Jira REST or custom)
        if isinstance(obj, dict):
            keys = {k.lower() for k in obj}
            # Jira REST API wraps everything in "fields"
            if "fields" in keys:
                return "ticket", "high"
            ticket_signals = keys & {"ticket_id", "subject", "description", "status", "priority"}
            if len(ticket_signals) >= 3:
                return "ticket", "high"
            commitment_signals = keys & {"commitment_id", "promised_date", "current_target_date"}
            if commitment_signals:
                return "commitment_tracker", "high"

        return None, "low"

    # ── CSV ──────────────────────────────────────────────────────────────────
    if ext == ".csv":
        try:
            import csv as _csv
            import io as _io
            sample = content_bytes[:4096].decode("utf-8", errors="replace")
            reader = _csv.DictReader(_io.StringIO(sample))
            headers = {(h or "").strip().lower() for h in (reader.fieldnames or [])}
        except Exception:
            return None, "low"

        ticket_cols = headers & {"ticket_id", "id", "summary", "issue_key", "issuetype"}
        commit_cols = headers & {"commitment_id", "promised_date", "current_target_date", "target_date"}
        if commit_cols:
            return "commitment_tracker", "high"
        if ticket_cols & {"ticket_id", "issue_key", "summary"}:
            return "ticket", "high"
        if "status" in headers and "priority" in headers:
            return "ticket", "medium"

        return None, "low"

    # ── Plain text / transcript formats ───────────────────────────────────────
    if ext in (".txt", ".vtt", ".srt"):
        if ext in (".vtt", ".srt"):
            return "transcript", "high"
        sample = content_bytes[:4096].decode("utf-8", errors="replace")
        # Look for speaker-label pattern: "Name: text" at the start of a line
        speaker_lines = re.findall(
            r"^[A-Za-z][A-Za-z0-9 _\-]{1,40}:\s+\S", sample, re.MULTILINE
        )
        if len(speaker_lines) >= 2:
            return "transcript", "high"
        # Otter JSON inside a .txt is rare but check anyway
        if '"speaker"' in sample and '"words"' in sample:
            return "transcript", "medium"
        # Plain prose — could be account_notes or transcript with no labels
        if any(kw in sample.lower() for kw in ("account notes", "relationship", "account summary")):
            return "account_notes", "medium"
        return "transcript", "low"

    # ── Binary / markup — cannot sniff reliably ───────────────────────────────
    if ext == ".pdf":
        return "account_notes", "low"   # most FDE PDFs are QBR or notes; let user decide
    if ext == ".docx":
        return "account_notes", "low"
    if ext == ".html":
        return "solution_architecture", "low"

    return None, "low"


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
