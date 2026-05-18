"""
commitment_parser.py — Parse commitment tracker files (JSON or CSV).

Supported formats
-----------------
JSON — top-level array or {"commitments": [...]}:
  [
    {
      "commitment_id": "COM-012",
      "description": "Deliver Predictive ETA early access",
      "promised_date": "2025-09-30",
      "current_target_date": "2026-09-30",
      "status": "slipped",
      "owner": "Helmsworth Engineering",
      "source_doc": "2024-01-15_qbr-deck_meridian.pdf",
      "source_section": "Roadmap commitments, slide 14",
      "last_updated": "2026-04-01",
      "customer_aware": true
    }
  ]

CSV — header row required; column names are case-insensitive:
  commitment_id,description,promised_date,current_target_date,status,owner,
  source_doc,source_section,last_updated,customer_aware

Valid status values: active | slipped | delivered | deferred
customer_aware: any truthy string ("true", "yes", "1") is accepted.
"""

import csv
import json
import os
from dataclasses import dataclass
from typing import List

# Statuses that mean the commitment is no longer active / needs no action.
_TERMINAL_COMMITMENT_STATUSES = frozenset({
    "delivered", "deferred", "closed", "resolved", "done",
    "complete", "completed", "cancelled", "canceled", "won't_do",
})

# Map raw status strings (from user-supplied data) → canonical vocabulary.
# Keys are lowercase; values are one of the canonical statuses.
_COMMIT_STATUS_NORMALIZE = {
    "wip":          "in_progress",
    "in-progress":  "in_progress",
    "in progress":  "in_progress",
    "started":      "in_progress",
    "working":      "in_progress",
    "open":         "active",
    "new":          "active",
    "pending":      "active",
    "on-hold":      "deferred",
    "on_hold":      "deferred",
    "blocked":      "deferred",
    "hold":         "deferred",
    "done":         "delivered",
    "complete":     "delivered",
    "completed":    "delivered",
    "closed":       "delivered",
    "resolved":     "delivered",
    "fixed":        "delivered",
    "cancelled":    "deferred",
    "canceled":     "deferred",
    "won't_do":     "deferred",
    "wontdo":       "deferred",
}


def _normalize_commitment_status(raw: str) -> str:
    """Map raw status to canonical value; unknown values pass through lowercased."""
    cleaned = raw.strip().lower()
    return _COMMIT_STATUS_NORMALIZE.get(cleaned, cleaned)


@dataclass
class Commitment:
    commitment_id: str
    description: str
    promised_date: str
    current_target_date: str
    status: str           # canonical: active | in_progress | slipped | delivered | deferred
    owner: str
    source_doc: str
    source_section: str
    last_updated: str
    customer_aware: bool
    is_slipped: bool = False   # True when current_target_date > promised_date
    is_open: bool = True       # False for terminal statuses (delivered, deferred, etc.)
    is_overdue: bool = False   # True when current_target_date < today and is_open
    days_overdue: int = 0      # Calendar days past target date; 0 when not overdue


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "yes", "1", "y")


def _row_to_commitment(item: dict) -> Commitment:
    """Convert a dict (from JSON or CSV DictReader) to a Commitment.

    Accepts both 'description' and 'commitment' as the human-readable text
    field so CSV exports that use 'commitment' as the column header work
    without renaming.
    """
    # Normalise keys to lowercase so CSV headers don't need exact casing
    lower = {k.strip().lower(): v for k, v in item.items()}
    # Resolve variant field names to canonical ones (e.g. "id" → "commitment_id").
    # Canonical keys present in the original dict take priority over alias-mapped ones.
    _aliased = {_COMMIT_ALIASES.get(k, k): v for k, v in lower.items()}
    _aliased.update({k: v for k, v in lower.items() if k not in _COMMIT_ALIASES})
    lower = _aliased
    # After aliasing, "description" maps to "commitment" — check "commitment" first
    description = str(lower.get("commitment") or lower.get("description", ""))
    promised_date = str(lower.get("promised_date", ""))
    current_target_date = str(lower.get("current_target_date", ""))
    # is_slipped: explicit field in JSON OR computed when target > promised
    explicit_slipped = lower.get("is_slipped")
    if explicit_slipped is not None:
        is_slipped = _parse_bool(explicit_slipped)
    else:
        # Compute: target date slipped past the original promise
        is_slipped = bool(
            current_target_date and promised_date
            and current_target_date > promised_date
        )
    status = _normalize_commitment_status(str(lower.get("status", "active")))
    return Commitment(
        commitment_id=str(lower.get("commitment_id", "")),
        description=description,
        promised_date=promised_date,
        current_target_date=current_target_date,
        status=status,
        owner=str(lower.get("owner", "")),
        source_doc=str(lower.get("source_doc", "")),
        source_section=str(lower.get("source_section", "")),
        last_updated=str(lower.get("last_updated", "")),
        customer_aware=_parse_bool(lower.get("customer_aware", False)),
        is_slipped=is_slipped,
        is_open=status not in _TERMINAL_COMMITMENT_STATUSES,
    )


def parse(file_path: str) -> List[Commitment]:
    """Return a list of Commitment objects from a JSON or CSV commitment tracker file."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        return _parse_csv(file_path)
    return _parse_json(file_path)


def _parse_json(file_path: str) -> List[Commitment]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("commitments", [])
    else:
        items = []

    return [_row_to_commitment(item) for item in items if isinstance(item, dict)]


def _parse_csv(file_path: str) -> List[Commitment]:
    result = []
    with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not any(row.values()):
                continue  # skip blank rows
            result.append(_row_to_commitment(row))
    return result


# Public alias used by test_ingestion.py so the import surface is consistent
# with ticket_csv_parser.parse_csv.
def parse_csv(file_path: str) -> List[Commitment]:
    """Parse a commitment tracker CSV file.  Thin alias for parse()."""
    return _parse_csv(file_path)


# ── Extended CSV parser (Google Sheets export format) ─────────────────────────
# CommitmentRecord is a lightweight dataclass for Google Sheets-style exports
# where columns may use different names than the canonical Commitment dataclass.
# Alias resolution maps common variants to canonical field names.

import csv as _csv
from dataclasses import dataclass as _dataclass
from typing import Optional as _Optional, List as _List

_COMMIT_REQUIRED = {"commitment", "status"}

_COMMIT_ALIASES = {
    # ── Commitment text ───────
    "description":      "commitment",
    "promise":          "commitment",
    "item":             "commitment",
    "task":             "commitment",
    # ── Identifier ───────────
    "id":               "commitment_id",
    # ── Date fields ──────────
    "due_date":         "promised_date",
    "original_due":     "promised_date",
    "commit_date":      "promised_date",
    "target":           "current_target_date",
    "target_date":      "current_target_date",
    "current_due":      "current_target_date",
    "revised_date":     "current_target_date",
    "updated_due":      "current_target_date",
    "created_date":     "last_updated",
    # ── People ───────────────
    "responsible":      "owner",
    "assignee":         "owner",
    "team":             "owner",
    # ── Visibility ───────────
    "customer_visible": "customer_aware",
    "visible":          "customer_aware",
    "external":         "customer_aware",
}


@_dataclass
class CommitmentRecord:
    description: str
    status: str
    promised_date: _Optional[str] = None
    current_target_date: _Optional[str] = None
    owner: _Optional[str] = None
    customer_aware: bool = False
    is_slipped: bool = False   # computed: current_target_date > promised_date
    is_open: bool = True       # False for terminal statuses
    commitment_id: str = ""


def parse_csv_sheets(file_path: str) -> _List[CommitmentRecord]:
    """
    Parse Google Sheets commitment tracker CSV export into CommitmentRecord objects.

    Expected columns: commitment, promised_date, current_target_date,
                      status, owner, customer_aware

    Column names are case-insensitive and resolved via _COMMIT_ALIASES.
    Raises ValueError if required columns (commitment, status) are absent.
    """
    import logging
    _log = logging.getLogger(__name__)

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        reader = _csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header row: {os.path.basename(file_path)}")

        col_map = {}
        for h in reader.fieldnames:
            normalized = h.lower().strip().replace(" ", "_")
            canonical = _COMMIT_ALIASES.get(normalized, normalized)
            col_map[canonical] = h

        missing = _COMMIT_REQUIRED - set(col_map.keys())
        if missing:
            raise ValueError(
                f"Commitment CSV missing required columns: {missing}. "
                f"Found: {list(col_map.keys())}"
            )

        records = []
        for row_num, row in enumerate(reader, start=2):
            def get(field_name: str, default: str = "") -> str:
                header = col_map.get(field_name)
                if not header:
                    return default
                return (row.get(header) or "").strip()

            desc = get("commitment")
            if not desc:
                continue

            promised = _normalize_commit_date(get("promised_date"))
            target = _normalize_commit_date(get("current_target_date")) or promised
            is_slipped = _compute_slipped(promised, target)

            aware_raw = get("customer_aware", "false").lower()
            customer_aware = aware_raw in ("true", "yes", "1", "x")

            commitment_id = get("commitment_id") or f"C{row_num - 1:03d}"
            status = _normalize_commitment_status(get("status") or "open")

            records.append(CommitmentRecord(
                description=desc,
                status=status,
                promised_date=promised,
                current_target_date=target,
                owner=get("owner") or None,
                customer_aware=customer_aware,
                is_slipped=is_slipped,
                is_open=status not in _TERMINAL_COMMITMENT_STATUSES,
                commitment_id=commitment_id,
            ))

    _log.info("commitment_csv_parsed file=%s count=%d",
              os.path.basename(file_path), len(records))
    return records


def _normalize_commit_date(s: str) -> _Optional[str]:
    if not s:
        return None
    from datetime import datetime
    s = s.strip()
    for fmt in (
        "%Y-%m-%d",     # 2026-01-15
        "%m/%d/%Y",     # 01/15/2026
        "%d/%m/%Y",     # 15/01/2026
        "%d-%b-%Y",     # 15-Jan-2026
        "%Y/%m/%d",     # 2026/01/15
        "%B %d, %Y",    # January 15, 2026  ← common in Google Sheets / Notion exports
        "%d %B %Y",     # 15 January 2026
    ):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue
    # ISO-8601 with time component: "2026-01-15T10:30:00Z" — strip to date only
    if len(s) >= 10:
        try:
            return datetime.strptime(s[:10], "%Y-%m-%d").strftime("%Y-%m-%d")
        except Exception:
            pass
    return None  # unparseable strings (e.g. "Q3 2026") are dropped rather than silently truncated


def _compute_slipped(promised: _Optional[str], target: _Optional[str]) -> bool:
    """True if current_target_date is later than promised_date."""
    if not promised or not target:
        return False
    try:
        from datetime import datetime
        p = datetime.strptime(promised, "%Y-%m-%d")
        t = datetime.strptime(target, "%Y-%m-%d")
        return t > p
    except Exception:
        return False
