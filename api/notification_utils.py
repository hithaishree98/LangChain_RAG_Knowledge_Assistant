import requests
import os

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")


def send_to_slack(question: str, answer: str, sources: list, confidence: float, session_id: str):
    if not SLACK_WEBHOOK_URL:
        return False

    if confidence > 0.7:
        conf_label = "High"
    elif confidence > 0.4:
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
        r = requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=10)
        return r.status_code == 200
    except Exception as e:
        print(f"Slack notification failed: {e}")
        return False