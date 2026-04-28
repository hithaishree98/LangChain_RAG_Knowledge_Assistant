import logging
import httpx
import os

_log = logging.getLogger(__name__)

# Confidence-label thresholds for Slack messages.
# - HIGH:    "we'd ship this" — above the high-confidence floor
# - MEDIUM:  "review before sending" — between escalation floor and high floor
# - LOW:     escalated by the API (matches main.CONFIDENCE_THRESHOLD = 0.4)
# Kept as module constants instead of bare numerics so the labels and the
# escalation logic in main.py can be reasoned about together.
_CONFIDENCE_HIGH = 0.7
_CONFIDENCE_ESCALATION = 0.4   # mirrors main.CONFIDENCE_THRESHOLD


async def send_to_slack(question: str, answer: str, sources: list,
                        confidence: float, session_id: str) -> bool:
    webhook_url = os.getenv("SLACK_WEBHOOK_URL", "")   # read at call time, not import time
    if not webhook_url:
        return False

    if confidence > _CONFIDENCE_HIGH:
        conf_label = "High"
    elif confidence > _CONFIDENCE_ESCALATION:
        conf_label = "Medium"
    else:
        conf_label = "Low — verify before sharing"

    sources_text = ", ".join(sources) if sources else "None identified"

    payload = {
        "blocks": [
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Question:*\n{question}"},
                    {"type": "mrkdwn", "text": f"*Confidence:* {conf_label}"}
                ]
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Answer:*\n{answer}"}
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Sources:* {sources_text}"},
                    {"type": "mrkdwn", "text": f"*Session:* `{session_id}`"}
                ]
            }
        ]
    }

    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(webhook_url, json=payload, timeout=10)
        return r.status_code == 200
    except Exception as e:
        _log.warning("slack_notification_failed: %s", e)
        return False