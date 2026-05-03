"""
utils/diff_utils.py — Compute diffs between two brief versions.

Used by GET /briefs/diff to show FDE what changed since their last brief
for a customer. Shows: new items, resolved items, changed items.
"""
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

_log = logging.getLogger(__name__)


def diff_briefs(
    brief_old: Dict[str, Any],
    brief_new: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compare two brief dicts and return a diff showing what changed.

    Returns:
        {
            "new_items":      list of items in new but not old
            "resolved_items": list of items in old but not new
            "changed_items":  list of items in both but with different status
            "unchanged_count": number of items unchanged
            "as_of_old":      date of old brief
            "as_of_new":      date of new brief
        }
    """
    new_items = []
    resolved_items = []
    changed_items = []
    unchanged_count = 0

    # Compare open_items (tickets)
    old_open = {_item_key(i): i for i in brief_old.get("open_items", [])}
    new_open = {_item_key(i): i for i in brief_new.get("open_items", [])}
    _diff_section(old_open, new_open, "open_item",
                  new_items, resolved_items, changed_items)

    # Compare outstanding_commitments
    old_commits = {_commit_key(c): c for c in brief_old.get("outstanding_commitments", [])}
    new_commits = {_commit_key(c): c for c in brief_new.get("outstanding_commitments", [])}
    _diff_section(old_commits, new_commits, "commitment",
                  new_items, resolved_items, changed_items)

    # Compare overdue_commitments
    old_overdue = {_commit_key(c): c for c in brief_old.get("overdue_commitments", [])}
    new_overdue = {_commit_key(c): c for c in brief_new.get("overdue_commitments", [])}
    _diff_section(old_overdue, new_overdue, "overdue_commitment",
                  new_items, resolved_items, changed_items)

    total = len(new_items) + len(resolved_items) + len(changed_items)
    total_old = (len(old_open) + len(old_commits) + len(old_overdue))
    unchanged_count = max(0, total_old - len(resolved_items) - len(changed_items))

    return {
        "new_items":      new_items,
        "resolved_items": resolved_items,
        "changed_items":  changed_items,
        "unchanged_count": unchanged_count,
        "as_of_old":      brief_old.get("as_of_date", ""),
        "as_of_new":      brief_new.get("as_of_date", ""),
    }


def _diff_section(
    old_map: Dict[str, Any],
    new_map: Dict[str, Any],
    item_type: str,
    new_items: List,
    resolved_items: List,
    changed_items: List,
) -> None:
    all_keys = set(old_map) | set(new_map)
    for key in all_keys:
        if key in new_map and key not in old_map:
            new_items.append({"type": item_type, "item": new_map[key]})
        elif key in old_map and key not in new_map:
            resolved_items.append({"type": item_type, "item": old_map[key]})
        elif key in old_map and key in new_map:
            old_status = _get_status(old_map[key])
            new_status = _get_status(new_map[key])
            if old_status != new_status:
                changed_items.append({
                    "type": item_type,
                    "old": old_map[key],
                    "new": new_map[key],
                    "status_change": f"{old_status} → {new_status}",
                })


def _item_key(item: Dict[str, Any]) -> str:
    """Stable identity key for an open item."""
    return (item.get("title") or item.get("ticket_id") or "")[:80].lower().strip()


def _commit_key(commit: Dict[str, Any]) -> str:
    """Stable identity key for a commitment."""
    return (commit.get("description") or "")[:80].lower().strip()


def _get_status(item: Dict[str, Any]) -> str:
    return (item.get("status") or "").lower()
