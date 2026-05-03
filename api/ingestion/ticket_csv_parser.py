"""
ingestion/ticket_csv_parser.py — Parse Jira/Zendesk CSV exports into ticket records.

Expected CSV columns (flexible — uses aliases for different export formats):
  ticket_id, summary, status, priority, created_date, updated_date,
  reporter, assignee, description, resolution

Missing columns degrade gracefully; only ticket_id, summary, status are required.
"""
import csv
import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional

_log = logging.getLogger(__name__)

# Required columns — everything else is optional
_REQUIRED = {"ticket_id", "summary", "status"}

# Column name aliases for different export formats (Jira, Zendesk, etc.)
_ALIASES = {
    "id":                "ticket_id",
    "key":               "ticket_id",
    "issue_key":         "ticket_id",
    "issue_id":          "ticket_id",
    "ticket":            "ticket_id",
    "title":             "summary",
    "subject":           "summary",
    "name":              "summary",
    "state":             "status",
    "issue_state":       "status",
    "created":           "created_date",
    "create_date":       "created_date",
    "creation_date":     "created_date",
    "date_created":      "created_date",
    "updated":           "updated_date",
    "update_date":       "updated_date",
    "last_updated":      "updated_date",
    "date_updated":      "updated_date",
    "desc":              "description",
    "body":              "description",
    "issue_description": "description",
    "resolve":           "resolution",
    "resolution_notes":  "resolution",
    "fix_notes":         "resolution",
}


@dataclass
class TicketRecord:
    ticket_id: str
    summary: str
    status: str
    priority: str = "normal"
    created_date: Optional[str] = None
    updated_date: Optional[str] = None
    reporter: Optional[str] = None
    assignee: Optional[str] = None
    description: str = ""
    resolution: Optional[str] = None


def parse_csv(file_path: str) -> List[TicketRecord]:
    """
    Parse a CSV file of tickets. Flexible column detection with alias resolution.

    Raises ValueError if required columns are missing.
    """
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header row: {os.path.basename(file_path)}")

        # Normalize column names: lowercase, strip spaces, resolve aliases
        col_map = {}  # canonical_name → original_header
        for h in reader.fieldnames:
            normalized = h.lower().strip().replace(" ", "_")
            canonical = _ALIASES.get(normalized, normalized)
            col_map[canonical] = h

        missing = _REQUIRED - set(col_map.keys())
        if missing:
            raise ValueError(
                f"CSV missing required columns: {missing}. "
                f"Found: {list(col_map.keys())}"
            )

        tickets = []
        for row_num, row in enumerate(reader, start=2):
            def get(field_name: str, default: str = "") -> str:
                header = col_map.get(field_name)
                if not header:
                    return default
                return (row.get(header) or "").strip()

            ticket_id = get("ticket_id")
            if not ticket_id:
                _log.debug("Skipping row %d: empty ticket_id", row_num)
                continue

            tickets.append(TicketRecord(
                ticket_id=ticket_id,
                summary=get("summary"),
                status=get("status").lower() or "open",
                priority=get("priority") or "normal",
                created_date=_normalize_date(get("created_date")),
                updated_date=_normalize_date(get("updated_date")),
                reporter=get("reporter") or None,
                assignee=get("assignee") or None,
                description=get("description"),
                resolution=get("resolution") or None,
            ))

    _log.info("ticket_csv_parsed file=%s count=%d",
              os.path.basename(file_path), len(tickets))
    return tickets


def _normalize_date(s: str) -> Optional[str]:
    """Normalize various date formats to YYYY-MM-DD."""
    if not s:
        return None
    s = s.strip()
    from datetime import datetime
    formats = [
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%d-%b-%Y",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%b %d, %Y",
        "%B %d, %Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue
    # Try first 10 chars as ISO
    if len(s) >= 10:
        try:
            return datetime.strptime(s[:10], "%Y-%m-%d").strftime("%Y-%m-%d")
        except Exception:
            pass
    return None
