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
    status: str = ""
    created_at: str = ""
    updated_at: str = ""
    reporter: str = ""
    assignee: str = ""
    comments: List[Comment] = field(default_factory=list)
    resolution: str = ""


def parse(file_path: str) -> TicketStructure:
    """Return a TicketStructure from a ticket JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    comments = [
        Comment(
            author=c.get("author", "Unknown"),
            body=c.get("body", "").strip(),
            created_at=c.get("created_at", ""),
        )
        for c in data.get("comments", [])
        if c.get("body", "").strip()
    ]

    # Support both "id" and "ticket_id" as the primary key field
    ticket_id = str(data.get("ticket_id") or data.get("id") or "")

    return TicketStructure(
        ticket_id=ticket_id,
        subject=data.get("subject", "").strip(),
        description=data.get("description", "").strip(),
        priority=data.get("priority", ""),
        status=data.get("status", ""),
        created_at=data.get("created_at", ""),
        updated_at=data.get("updated_at", ""),
        reporter=data.get("reporter", ""),
        assignee=data.get("assignee", ""),
        comments=comments,
        resolution=(data.get("resolution") or "").strip(),
    )
