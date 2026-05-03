"""
scripts/daily_overdue_check.py — Daily overdue commitment check.

Finds all commitments across all customers that are past their target date
with status != closed/resolved, then posts a digest to Slack.

Run via APScheduler (configured in main.py lifespan) or directly:
    python -m api.scripts.daily_overdue_check
"""
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List

# Allow running as a script from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_log = logging.getLogger(__name__)


def find_overdue_commitments() -> List[Dict[str, Any]]:
    """
    Scan the most recent brief per customer for overdue commitments.
    Returns list of {customer_name, commitment, was_due, owner} dicts.
    """
    from db_utils import get_db_connection

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    overdue = []

    with get_db_connection() as conn:
        # Get the most recent brief per customer
        rows = conn.execute("""
            SELECT bl.id, bl.customer_id, bl.brief_json, c.name as customer_name
            FROM brief_logs bl
            JOIN customers c ON c.slug = bl.customer_id
            WHERE bl.created_at = (
                SELECT MAX(bl2.created_at)
                FROM brief_logs bl2
                WHERE bl2.customer_id = bl.customer_id
            )
        """).fetchall()

    for row in rows:
        try:
            brief = json.loads(row["brief_json"])
        except (json.JSONDecodeError, TypeError):
            continue

        all_commitments = (
            brief.get("outstanding_commitments", []) +
            brief.get("overdue_commitments", [])
        )

        for commitment in all_commitments:
            target_date = commitment.get("target_date") or commitment.get("promised_date")
            status = (commitment.get("status") or "").lower()
            if status in ("closed", "resolved", "done", "completed"):
                continue
            if not target_date:
                continue
            try:
                if target_date < today:
                    overdue.append({
                        "customer_name": row["customer_name"],
                        "commitment":    commitment.get("description", "Unknown commitment"),
                        "was_due":       target_date,
                        "owner":         commitment.get("owner"),
                        "status":        status,
                    })
            except TypeError:
                continue

    _log.info("overdue_check found=%d today=%s", len(overdue), today)
    return overdue


async def run() -> None:
    """Run the overdue check and post digest to Slack if items found."""
    try:
        items = find_overdue_commitments()
        if not items:
            _log.info("overdue_check_no_items")
            return
        from notification_utils import notify_overdue_digest
        sent = await notify_overdue_digest(items)
        _log.info("overdue_digest_sent=%s count=%d", sent, len(items))
    except Exception as e:
        _log.error("overdue_check_failed: %s", e)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run())
