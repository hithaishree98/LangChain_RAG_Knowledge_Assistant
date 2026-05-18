import logging
import re
import httpx
import os

_WEBHOOK_URL_RE = re.compile(r"^https?://\S+", re.IGNORECASE)


def _validate_webhook_url(url: str) -> bool:
    """Return True only if url looks like a valid http(s) URL."""
    return bool(url and _WEBHOOK_URL_RE.match(url))

_log = logging.getLogger(__name__)

# Confidence-label thresholds for Slack messages.
# - HIGH:    "we'd ship this" — above the high-confidence floor
# - MEDIUM:  "review before sending" — between escalation floor and high floor
# - LOW:     escalated by the API (matches main.CONFIDENCE_THRESHOLD = 0.4)
# Kept as module constants instead of bare numerics so the labels and the
# escalation logic in main.py can be reasoned about together.
async def notify_overdue_digest(items: list) -> bool:
    """Send a daily digest of overdue commitments to Slack.

    Non-fatal — callers should wrap in try/except and ignore failures.
    Returns False when no webhook is configured or the call fails.
    """
    webhook_url = os.getenv("SLACK_WEBHOOK_URL", "")
    if not _validate_webhook_url(webhook_url):
        if webhook_url:
            _log.warning("slack_invalid_webhook_url url=%r", webhook_url)
        return False

    lines = []
    for item in items:
        customer = item.get("customer_name", "Unknown")
        commitment = item.get("commitment", "Unknown commitment")
        was_due = item.get("was_due", "?")
        owner = item.get("owner")
        owner_str = f" (owner: {owner})" if owner else ""
        lines.append(f"• *{customer}*: {commitment} — due {was_due}{owner_str}")

    text = "\n".join(lines)
    payload = {
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f":rotating_light: *{len(items)} overdue commitment(s)*\n{text}",
                },
            }
        ]
    }

    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(webhook_url, json=payload, timeout=10)
        return r.status_code == 200
    except Exception as e:
        _log.warning("overdue_digest_slack_failed error=%s", e)
        return False


async def notify_transcript_uploaded(customer_id: str, filename: str) -> bool:
    """Send a Slack notification when a new transcript is uploaded for a customer.

    Non-fatal — callers should wrap in try/except and ignore failures.
    Returns False when no webhook is configured or the call fails.
    """
    webhook_url = os.getenv("SLACK_WEBHOOK_URL", "")
    if not _validate_webhook_url(webhook_url):
        if webhook_url:
            _log.warning("slack_invalid_webhook_url url=%r", webhook_url)
        return False

    payload = {
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f":memo: *New transcript uploaded*\n"
                        f"*Customer:* `{customer_id}`\n"
                        f"*File:* `{filename}`"
                    ),
                },
            }
        ]
    }

    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(webhook_url, json=payload, timeout=10)
        return r.status_code == 200
    except Exception as e:
        _log.warning("transcript_notify_slack_failed customer=%s file=%s error=%s",
                     customer_id, filename, e)
        return False