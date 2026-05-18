"""
ticket_parser.py — Parse support ticket JSON files.

Expected JSON schema (flexible — missing fields are skipped):
{
  "id": "TICK-1234",
  "subject": "...",
  "description": "...",
  "priority": "P1",
  "status": "open",
  "created_at": "2024-01-15",
  "comments": [
    {"author": "...", "body": "...", "created_at": "..."}
  ],
  "resolution": "..."
}
"""

import json
from dataclasses import dataclass, field
from typing import List, Optional

# Statuses that mean the ticket is no longer active.
_TERMINAL_TICKET_STATUSES = frozenset({
    "closed", "resolved", "done", "complete", "completed",
    "fixed", "won't fix", "wontfix", "duplicate", "cancelled", "canceled",
    "rejected", "invalid",
})

# Map raw status strings → canonical vocabulary.
_TICKET_STATUS_NORMALIZE = {
    "new":          "open",
    "todo":         "open",
    "backlog":      "open",
    "wip":          "in_progress",
    "in-progress":  "in_progress",
    "in progress":  "in_progress",
    "started":      "in_progress",
    "working":      "in_progress",
    "active":       "in_progress",
    "investigating": "in_progress",
    "done":         "closed",
    "complete":     "closed",
    "completed":    "closed",
    "resolved":     "closed",
    "fixed":        "closed",
    "won't fix":    "closed",
    "wontfix":      "closed",
    "duplicate":    "closed",
    "cancelled":    "closed",
    "canceled":     "closed",
    "rejected":     "closed",
    "invalid":      "closed",
}


def _normalize_ticket_status(raw: str) -> str:
    """Map raw status to canonical value; unknown values pass through lowercased."""
    cleaned = raw.strip().lower()
    return _TICKET_STATUS_NORMALIZE.get(cleaned, cleaned)


@dataclass
class Comment:
    author: str
    body: str
    created_at: str = ""


@dataclass
class TicketStructure:
    ticket_id: str
    subject: str
    description: str
    priority: str = ""
    status: str = ""        # canonical: open | in_progress | pending | closed | escalated
    created_at: str = ""
    updated_at: str = ""
    reporter: str = ""
    assignee: str = ""
    comments: List[Comment] = field(default_factory=list)
    resolution: str = ""
    is_open: bool = True    # False for terminal statuses


def _extract_adf_text(node: dict) -> str:
    """Recursively extract plain text from Atlassian Document Format (Jira Cloud)."""
    if not isinstance(node, dict):
        return str(node)
    text = node.get("text", "")
    node_type = node.get("type", "")
    for child in node.get("content") or []:
        text += _extract_adf_text(child)
    if node_type in ("paragraph", "heading", "listItem", "bulletList", "orderedList", "rule"):
        text += "\n"
    return text


def _normalize_jira_json(data: dict) -> dict:
    """Flatten Jira REST API export format (all fields nested under 'fields': {}).

    Jira's REST API wraps everything under a 'fields' key. Jira Cloud also uses
    Atlassian Document Format (ADF) for description and comment bodies instead of
    plain strings. This function detects both patterns and normalises to the flat
    schema that parse() expects, leaving non-Jira JSON untouched.
    """
    if "fields" not in data:
        return data

    fields = data["fields"]
    normalized: dict = dict(data)

    # ticket_id: Jira uses "key" (e.g. "PROJ-123") at the top level
    normalized["ticket_id"] = data.get("key") or data.get("id") or ""

    normalized["subject"] = (fields.get("summary") or "").strip()

    desc = fields.get("description") or ""
    normalized["description"] = (_extract_adf_text(desc) if isinstance(desc, dict) else str(desc)).strip()

    status_raw = fields.get("status") or {}
    normalized["status"] = (
        status_raw.get("name") if isinstance(status_raw, dict) else str(status_raw)
    ).strip()

    priority_raw = fields.get("priority") or {}
    normalized["priority"] = (
        priority_raw.get("name") if isinstance(priority_raw, dict) else str(priority_raw)
    ).strip()

    assignee_raw = fields.get("assignee") or {}
    normalized["assignee"] = (
        (assignee_raw.get("displayName") or assignee_raw.get("name") or "")
        if isinstance(assignee_raw, dict) else str(assignee_raw)
    )

    reporter_raw = fields.get("reporter") or {}
    normalized["reporter"] = (
        (reporter_raw.get("displayName") or reporter_raw.get("name") or "")
        if isinstance(reporter_raw, dict) else str(reporter_raw)
    )

    # Jira timestamps are ISO-8601; slice to YYYY-MM-DD
    normalized["created_at"] = (fields.get("created") or "")[:10]
    normalized["updated_at"] = (fields.get("updated") or "")[:10]

    # Comments live under fields.comment.comments
    comment_container = fields.get("comment") or {}
    raw_comments = (
        comment_container.get("comments") or []
        if isinstance(comment_container, dict) else []
    )
    normalized_comments = []
    for c in raw_comments:
        body = c.get("body") or ""
        if isinstance(body, dict):
            body = _extract_adf_text(body)
        author_raw = c.get("author") or {}
        author_name = (
            (author_raw.get("displayName") or author_raw.get("name") or "Unknown")
            if isinstance(author_raw, dict) else str(author_raw)
        )
        normalized_comments.append({
            "author": author_name,
            "body": str(body).strip(),
            "created_at": (c.get("created") or "")[:10],
        })
    normalized["comments"] = normalized_comments

    res_raw = fields.get("resolution") or {}
    res_name = (
        res_raw.get("name") if isinstance(res_raw, dict) else str(res_raw)
    ) or ""
    normalized["resolution"] = "" if res_name.lower() in ("unresolved", "none", "null", "") else res_name

    return normalized


def parse(file_path: str) -> TicketStructure:
    """Return a TicketStructure from a ticket JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data = _normalize_jira_json(data)

    comments = [
        Comment(
            author=c.get("author", "Unknown"),
            body=c.get("body", "").strip(),
            # Support both "created_at" and "timestamp" (used by some ticket systems).
            # Truncate ISO timestamps to YYYY-MM-DD for consistent date handling.
            created_at=(c.get("created_at") or c.get("timestamp", ""))[:10],
        )
        for c in data.get("comments", [])
        if c.get("body", "").strip()
    ]

    # Support both "id" and "ticket_id" as the primary key field
    ticket_id = str(data.get("ticket_id") or data.get("id") or "")
    status = _normalize_ticket_status(data.get("status", ""))

    return TicketStructure(
        ticket_id=ticket_id,
        subject=data.get("subject", "").strip(),
        description=data.get("description", "").strip(),
        priority=data.get("priority", ""),
        status=status,
        # Truncate ISO timestamps (e.g. "2026-04-30T16:40:00Z") to date-only.
        created_at=(data.get("created_at", "") or "")[:10],
        updated_at=(data.get("updated_at", "") or "")[:10],
        reporter=data.get("reporter", ""),
        assignee=data.get("assignee", ""),
        comments=comments,
        resolution=(data.get("resolution") or "").strip(),
        is_open=status not in _TERMINAL_TICKET_STATUSES,
    )
