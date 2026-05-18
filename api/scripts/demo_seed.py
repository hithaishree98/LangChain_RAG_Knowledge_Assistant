"""
demo_seed.py — Seed the API with three demo customer workspaces.

Creates three accounts that demonstrate distinct health states:

  Cascadia Health   → Healthy (score ~95):  no P0s, all commitments on track
  Meridian          → At Risk (score ~65):  1 P1 open, 1 overdue commitment
  Novus Financial   → Critical (score  ~0): 3 P0s, 5 overdue commitments

All document content is generated in-memory — no external sample files required.
Safe to re-run: existing resources are skipped (409) rather than duplicated.

Usage:
    python api/scripts/demo_seed.py

Expects the API running on localhost:8000 (or API_BASE_URL env var).
Credentials are read from DEMO_WORKSPACE / DEMO_PASSKEY (default: "demo" / "demo").
"""

import io
import json
import os
import sys
import time
from typing import Any

import requests

API_BASE  = os.getenv("API_BASE_URL", "http://localhost:8000")
WORKSPACE = os.getenv("DEMO_WORKSPACE", "demo")
PASSKEY   = os.getenv("DEMO_PASSKEY",   "demo")
API_KEY   = os.getenv("API_KEY", "")

# ---------------------------------------------------------------------------
# Synthetic document content
# All dates are relative to 2026-05-10 (the reference "today" for the demo).
# ---------------------------------------------------------------------------

# ── Cascadia Health — Healthy ───────────────────────────────────────────────

_CASCADIA_NOTES = """\
Cascadia Health — Account Notes
Last updated: 2026-05-05

Executive sponsor: Michael Torres (VP Product)
Relationship health: Strong. Customer is satisfied with platform performance.
They are on track for a June renewal at $420k ARR. No escalation risk.

Key context:
- Cascadia is a mid-market health-tech company with 180 seats.
- Primary use-case: patient data analytics pipelines.
- They have been a customer since 2024-02 and have never opened a P0 ticket.
- Michael Torres is a strong internal champion who has recommended the platform to two peer companies.
- Outstanding item: SSO integration is GA-targeted for 2026-07-01 — Cascadia is in beta.
"""

_CASCADIA_TRANSCRIPT = """\
2026-05-05 — Cascadia Health Status Call
Attendees: Rachel (FDE), Michael Torres (VP Product), Priya Nair (Eng Lead)

Rachel: Thanks for joining. Quick check-in on the SSO beta and the upcoming renewal.

Michael Torres: Things are going really well. The SSO beta is performing exactly as expected.
Priya and the team are happy with the integration docs.

Priya Nair: One minor thing — the session timeout is set to 60 minutes.
We would prefer 90 minutes for our clinical users. Can you log that?

Rachel: Absolutely, I will create a feature request.

Michael Torres: On renewal — we are planning to expand by another 40 seats in Q3.
The legal team will have the paperwork ready before June 30.

Rachel: Perfect. I will prepare the renewal proposal by May 20.
"""

_CASCADIA_TICKET_P1 = {
    "ticket_id": "TICK-C001",
    "subject": "Session timeout should be 90 minutes for clinical users",
    "description": (
        "Clinical workflows require longer sessions. The current 60-minute timeout "
        "is causing disruptions for nursing staff mid-shift. Requested by Priya Nair."
    ),
    "status": "open",
    "priority": "P1",
    "created_at": "2026-05-05",
    "updated_at": "2026-05-05",
    "comments": [
        {
            "author": "Rachel (FDE)",
            "body": "Logged on behalf of Priya Nair. Routing to product.",
            "created_at": "2026-05-05",
        }
    ],
    "resolution": "",
}

_CASCADIA_COMMITMENTS = [
    {
        "commitment_id": "CAS-001",
        "description": "SSO integration GA release",
        "promised_date": "2026-07-01",
        "current_target_date": "2026-07-01",
        "status": "active",
        "owner": "Platform Team",
        "customer_aware": True,
    },
    {
        "commitment_id": "CAS-002",
        "description": "Renewal proposal with 40-seat expansion pricing",
        "promised_date": "2026-05-20",
        "current_target_date": "2026-05-20",
        "status": "active",
        "owner": "Rachel (FDE)",
        "customer_aware": True,
    },
    {
        "commitment_id": "CAS-003",
        "description": "Session timeout feature request submitted to product roadmap",
        "promised_date": "2026-05-12",
        "current_target_date": "2026-05-12",
        "status": "active",
        "owner": "Rachel (FDE)",
        "customer_aware": False,
    },
    {
        "commitment_id": "CAS-004",
        "description": "SSO beta performance summary shared with Cascadia executive team",
        "promised_date": "2026-05-07",
        "current_target_date": "2026-05-07",
        "status": "delivered",
        "owner": "Rachel (FDE)",
        "customer_aware": True,
        "last_updated": "2026-05-08",
    },
]

# ── Meridian — At Risk ──────────────────────────────────────────────────────

_MERIDIAN_NOTES = """\
Meridian — Account Notes
Last updated: 2026-04-24

Executive sponsor: Sarah Chen (VP Engineering)
Relationship health: Cautious. The login latency issue has eroded trust.
Sarah is professional but has made clear that a repeat P0 will escalate.

Key context:
- Meridian is an enterprise fintech company, 320 seats, $280k ARR.
- Primary pain point: sticky-session misconfiguration caused P0 login spikes in April.
- The P0 was resolved but the root-cause fix (sticky-session reconfig) slipped by 10 days.
- Sarah has requested daily status updates until the underlying issue is closed.
- At risk: the SOC 2 report delivery has also slipped. Legal is watching.
- Renewal is due 2026-08-01. No explicit churn signal but confidence is low.
"""

_MERIDIAN_TRANSCRIPT = """\
2026-04-24 — Meridian Engineering Sync
Attendees: Rachel (FDE), Sarah Chen (VP Engineering), David Park (Tech Lead)

Rachel: Thanks for the time. I want to walk through the status on the open items.

Sarah Chen: Before we start — I need to be direct. The login issue last month was
not acceptable. We had two customer-facing incidents in a single week.
I told my CEO it was resolved. It needs to stay resolved.

Rachel: Understood and I completely agree. The sticky-session fix is in staging now.
David, can you give the status?

David Park: We expect the staging validation to complete by April 30. The config
change itself is a 10-minute deployment but we want to validate against Meridian's
actual session volume before pushing to prod.

Sarah Chen: April 30. I am holding you to that. If it slips again I will be escalating.

Rachel: Noted. On SOC 2 — the audit firm confirmed they will have the Type II report
ready by May 31. I know the original promise was April 1 but the auditors needed
additional evidence on our encryption key management.

Sarah Chen: Our legal team is not happy. We need that report for an enterprise deal
we are trying to close. Please make sure nothing else slips on this account.
"""

_MERIDIAN_TICKET_P1 = {
    "ticket_id": "TICK-M001",
    "subject": "Login latency P95 exceeds 2.4 seconds — sticky-session root cause fix pending",
    "description": (
        "Root cause identified: sticky-session misconfiguration on load balancer. "
        "Incident mitigated via cache-clear workaround. Permanent fix (reconfig) "
        "validated in staging, pending prod deployment. Target: 2026-04-30."
    ),
    "status": "open",
    "priority": "P1",
    "created_at": "2026-04-01",
    "updated_at": "2026-04-24",
    "comments": [
        {
            "author": "David Park",
            "body": "Staging validation in progress. On track for April 30 deployment.",
            "created_at": "2026-04-22",
        },
        {
            "author": "Sarah Chen",
            "body": "This needs to close by April 30. Escalation risk if it slips.",
            "created_at": "2026-04-24",
        },
    ],
    "resolution": "",
}

_MERIDIAN_TICKET_CLOSED = {
    "ticket_id": "TICK-M002",
    "subject": "Duplicate invoice emails sent to 12 Meridian users",
    "description": "Stripe webhook idempotency key bug caused duplicate invoice emails.",
    "status": "closed",
    "priority": "P2",
    "created_at": "2026-03-20",
    "updated_at": "2026-04-10",
    "comments": [],
    "resolution": "Fixed: added idempotency key to stripe_event_id field. Deployed 2026-04-10.",
}

_MERIDIAN_COMMITMENTS = [
    {
        "commitment_id": "MER-001",
        "description": "Sticky-session permanent fix deployed to production",
        "promised_date": "2026-04-20",
        "current_target_date": "2026-04-30",
        "status": "active",
        "owner": "Platform Team",
        "customer_aware": True,
    },
    {
        "commitment_id": "MER-002",
        "description": "SOC 2 Type II audit report delivered to Meridian legal",
        "promised_date": "2026-04-01",
        "current_target_date": "2026-05-31",
        "status": "slipped",
        "owner": "Compliance Team",
        "customer_aware": True,
    },
    {
        "commitment_id": "MER-003",
        "description": "Daily status update cadence maintained until TICK-M001 closes",
        "promised_date": "2026-04-24",
        "current_target_date": "2026-04-24",
        "status": "active",
        "owner": "Rachel (FDE)",
        "customer_aware": True,
    },
    {
        "commitment_id": "MER-004",
        "description": "Predictive ETA feature early access for Meridian data team",
        "promised_date": "2026-05-30",
        "current_target_date": "2026-05-30",
        "status": "active",
        "owner": "Product Team",
        "customer_aware": False,
    },
    {
        "commitment_id": "MER-005",
        "description": "Renewal proposal with updated enterprise tier pricing",
        "promised_date": "2026-07-01",
        "current_target_date": "2026-07-01",
        "status": "active",
        "owner": "Rachel (FDE)",
        "customer_aware": False,
    },
    {
        "commitment_id": "MER-006",
        "description": "Cache-clear workaround validation report shared with Meridian SRE team",
        "promised_date": "2026-04-25",
        "current_target_date": "2026-04-25",
        "status": "delivered",
        "owner": "Rachel (FDE)",
        "customer_aware": True,
        "last_updated": "2026-04-25",
    },
]

# ── Novus Financial — Critical ──────────────────────────────────────────────

_NOVUS_NOTES = """\
Novus Financial — Account Notes
Last updated: 2026-04-01

Executive sponsor: Jennifer Walsh (CTO)
Relationship health: Critical. Multiple open P0 incidents. Jennifer has threatened
to invoke the SLA penalty clause if uptime does not improve by end of May 2026.

Key context:
- Novus is a large enterprise financial services firm, 850 seats, $680k ARR.
- Three P0 incidents open simultaneously — unprecedented for any account.
- Jennifer Walsh is technically sharp and does not accept vague timelines.
  She wants specific root-cause analysis for each P0, not just ETAs.
- Five commitments are overdue, including the disaster recovery runbook
  that was promised in January 2026.
- The account has not had a meaningful call in 39 days. Last call on 2026-04-01
  ended with Jennifer requesting a weekly war-room cadence that never happened.
- Renewal is 2026-09-01. At current trajectory this is high churn risk.
- Internal note: escalate to VP of Customer Success before next call.
"""

_NOVUS_TRANSCRIPT = """\
2026-04-01 — Novus Financial Executive War-Room
Attendees: Rachel (FDE), Jennifer Walsh (CTO), Omar Reyes (Head of Infrastructure)

Jennifer Walsh: I want to be very direct. We have three active P0s.
That has never happened with any vendor in my fifteen years.
I need a war-room cadence — weekly at minimum — starting this week.

Rachel: Agreed. I will set up a standing call starting April 8.

Jennifer Walsh: What is the root cause of the payment processing delay?

Omar Reyes: We believe it is a connection pool exhaustion in the transaction service.
We have a fix in testing but it has not been validated against peak load yet.

Jennifer Walsh: TICK-N003 has been open for three weeks. Three weeks.
Every day this is open costs us real money in SLA penalties to our clients.

Rachel: I understand. I will escalate internally to get dedicated engineering
resources on TICK-N003 today.

Jennifer Walsh: I also need the disaster recovery runbook that was promised
in January. We are now four months past the original commit date.
If I do not have it by April 30, I am invoking the SLA penalty clause.

Rachel: I will personally track that with the engineering team and confirm
the delivery date by end of this week.

Jennifer Walsh: And the authentication outage — TICK-N002. What is the status?

Omar Reyes: We identified the issue — a cert rotation that did not propagate
correctly. We have a hotfix deployed but we need to do a full cert audit
before we can close the ticket.

Jennifer Walsh: That audit needs to happen this week. Not next week.
"""

def _novus_ticket(ticket_id, subject, description, priority, status,
                   created, updated, comments=None, resolution=""):
    return {
        "ticket_id": ticket_id,
        "subject": subject,
        "description": description,
        "status": status,
        "priority": priority,
        "created_at": created,
        "updated_at": updated,
        "comments": comments or [],
        "resolution": resolution,
    }

_NOVUS_TICKETS = [
    _novus_ticket(
        "TICK-N001",
        "Production database failover — standby replica not promoting",
        (
            "Primary DB failover triggered 2026-03-15. Standby replica failed to "
            "promote automatically. Manual intervention required, causing 47-minute "
            "downtime. Root cause: replication lag exceeded failover threshold."
        ),
        "P0", "open", "2026-03-15", "2026-04-20",
        comments=[
            {"author": "Omar Reyes", "body": "Replication lag fix deployed to staging. Needs prod validation.", "created_at": "2026-04-18"},
        ],
    ),
    _novus_ticket(
        "TICK-N002",
        "Authentication service — cert rotation caused 90-minute outage",
        (
            "TLS cert rotation on 2026-03-28 did not propagate to all auth nodes. "
            "Result: 90-minute authentication failure for 850 Novus users. "
            "Hotfix deployed but full cert audit still pending."
        ),
        "P0", "open", "2026-03-28", "2026-04-10",
        comments=[
            {"author": "Rachel (FDE)", "body": "Cert audit must complete this week per Jennifer Walsh.", "created_at": "2026-04-01"},
        ],
    ),
    _novus_ticket(
        "TICK-N003",
        "Payment processing delays >30 seconds — connection pool exhaustion",
        (
            "Transaction service connection pool exhaustion causing payment "
            "processing to queue for 30-90 seconds. Affects all Novus payment "
            "workflows. Fix in testing but not yet validated against peak load."
        ),
        "P0", "open", "2026-03-10", "2026-04-01",
        comments=[
            {"author": "Jennifer Walsh", "body": "Three weeks open. This is causing us SLA penalties with our clients.", "created_at": "2026-04-01"},
            {"author": "Rachel (FDE)", "body": "Escalated internally for dedicated engineering resources.", "created_at": "2026-04-01"},
        ],
    ),
    _novus_ticket(
        "TICK-N004",
        "Audit log entries missing for 12-hour window on 2026-03-20",
        (
            "Audit log gap discovered during internal compliance review. "
            "12-hour window (2026-03-20 00:00–12:00) has no entries. "
            "Possible log rotation misconfiguration."
        ),
        "P1", "open", "2026-03-22", "2026-04-05",
    ),
    _novus_ticket(
        "TICK-N005",
        "API quota exceeded during month-end batch processing",
        (
            "Month-end batch processing on 2026-03-31 exceeded API rate limits, "
            "causing batch job failures and manual reprocessing. "
            "Novus needs higher API quota or batch mode support."
        ),
        "P1", "open", "2026-04-01", "2026-04-02",
    ),
]

_NOVUS_COMMITMENTS = [
    {
        "commitment_id": "NOV-001",
        "description": "Disaster recovery runbook delivered and validated with Novus infrastructure team",
        "promised_date": "2026-01-15",
        "current_target_date": "2026-04-30",
        "status": "slipped",
        "owner": "Platform Team",
        "customer_aware": True,
    },
    {
        "commitment_id": "NOV-002",
        "description": "TICK-N001 root-cause fix deployed to production with validation report",
        "promised_date": "2026-02-01",
        "current_target_date": "2026-04-15",
        "status": "slipped",
        "owner": "Engineering",
        "customer_aware": True,
    },
    {
        "commitment_id": "NOV-003",
        "description": "Full TLS certificate audit completed and documented",
        "promised_date": "2026-03-01",
        "current_target_date": "2026-04-20",
        "status": "slipped",
        "owner": "Security Team",
        "customer_aware": True,
    },
    {
        "commitment_id": "NOV-004",
        "description": "TICK-N003 payment processing fix deployed and peak-load validated",
        "promised_date": "2026-03-15",
        "current_target_date": "2026-05-01",
        "status": "slipped",
        "owner": "Engineering",
        "customer_aware": True,
    },
    {
        "commitment_id": "NOV-005",
        "description": "Audit log gap root-cause analysis report (TICK-N004)",
        "promised_date": "2026-04-01",
        "current_target_date": "2026-05-05",
        "status": "slipped",
        "owner": "Engineering",
        "customer_aware": False,
    },
    {
        "commitment_id": "NOV-006",
        "description": "Higher API quota tier provisioned for Novus batch workloads",
        "promised_date": "2026-05-01",
        "current_target_date": "2026-06-01",
        "status": "slipped",
        "owner": "Platform Team",
        "customer_aware": True,
    },
    {
        "commitment_id": "NOV-007",
        "description": "Weekly war-room cadence established (standing call every Wednesday)",
        "promised_date": "2026-04-08",
        "current_target_date": "2026-04-08",
        "status": "active",
        "owner": "Rachel (FDE)",
        "customer_aware": True,
    },
    {
        "commitment_id": "NOV-008",
        "description": "SLA penalty assessment report shared with Jennifer Walsh",
        "promised_date": "2026-05-15",
        "current_target_date": "2026-05-15",
        "status": "active",
        "owner": "Rachel (FDE)",
        "customer_aware": True,
    },
]

# ---------------------------------------------------------------------------
# Account definitions
# ---------------------------------------------------------------------------

ACCOUNTS: list[dict[str, Any]] = [
    {
        "name": "Cascadia Health",
        "slug": "cascadia",
        "person": {"name": "Michael Torres", "role": "VP Product", "email": "mtorres@cascadiahealth.example.com"},
        "last_call_date": "2026-05-05",
        "docs": [
            # (upload_filename, doc_type, content_bytes)
            ("2026-05-05_account-notes_cascadia.txt", "account_notes", _CASCADIA_NOTES.encode()),
            ("2026-05-05_transcript_status-call.txt",  "transcript",    _CASCADIA_TRANSCRIPT.encode()),
            ("2026-05-05_ticket_TICK-C001.json",        "ticket",        json.dumps(_CASCADIA_TICKET_P1).encode()),
            ("2026-05-05_commitment-tracker_cascadia.json", "commitment_tracker",
             json.dumps(_CASCADIA_COMMITMENTS).encode()),
        ],
    },
    {
        "name": "Meridian",
        "slug": "meridian",
        "person": {"name": "Sarah Chen", "role": "VP Engineering", "email": "sarah@meridian.example.com"},
        "last_call_date": "2026-04-24",
        "docs": [
            ("2026-04-24_account-notes_meridian.txt",   "account_notes",        _MERIDIAN_NOTES.encode()),
            ("2026-04-24_transcript_eng-sync.txt",       "transcript",           _MERIDIAN_TRANSCRIPT.encode()),
            ("2026-04-24_ticket_TICK-M001.json",         "ticket",               json.dumps(_MERIDIAN_TICKET_P1).encode()),
            ("2026-04-10_ticket_TICK-M002.json",         "ticket",               json.dumps(_MERIDIAN_TICKET_CLOSED).encode()),
            ("2026-04-24_commitment-tracker_meridian.json", "commitment_tracker", json.dumps(_MERIDIAN_COMMITMENTS).encode()),
        ],
    },
    {
        "name": "Novus Financial",
        "slug": "novus",
        "person": {"name": "Jennifer Walsh", "role": "CTO", "email": "jwalsh@novusfinancial.example.com"},
        "last_call_date": "2026-04-01",
        "docs": [
            ("2026-04-01_account-notes_novus.txt",      "account_notes",        _NOVUS_NOTES.encode()),
            ("2026-04-01_transcript_war-room.txt",       "transcript",           _NOVUS_TRANSCRIPT.encode()),
            ("2026-03-15_ticket_TICK-N001.json",         "ticket",               json.dumps(_NOVUS_TICKETS[0]).encode()),
            ("2026-03-28_ticket_TICK-N002.json",         "ticket",               json.dumps(_NOVUS_TICKETS[1]).encode()),
            ("2026-03-10_ticket_TICK-N003.json",         "ticket",               json.dumps(_NOVUS_TICKETS[2]).encode()),
            ("2026-03-22_ticket_TICK-N004.json",         "ticket",               json.dumps(_NOVUS_TICKETS[3]).encode()),
            ("2026-04-01_ticket_TICK-N005.json",         "ticket",               json.dumps(_NOVUS_TICKETS[4]).encode()),
            ("2026-04-01_commitment-tracker_novus.json", "commitment_tracker",   json.dumps(_NOVUS_COMMITMENTS).encode()),
        ],
    },
]

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

_token: "str | None" = None


def _get_token() -> str:
    global _token
    if _token:
        return _token
    r = requests.post(
        f"{API_BASE}/auth/token",
        json={"workspace": WORKSPACE, "passkey": PASSKEY},
        timeout=10,
    )
    if not r.ok:
        raise RuntimeError(f"auth/token failed {r.status_code}: {r.text[:200]}")
    _token = r.json()["token"]
    return _token


def _headers(include_content_type: bool = True) -> dict:
    h = {}
    if API_KEY:
        h["X-API-Key"] = API_KEY
    h["Authorization"] = f"Bearer {_get_token()}"
    if include_content_type:
        h["Content-Type"] = "application/json"
    return h


def _ensure_customer(name: str, slug: str) -> None:
    r = requests.post(
        f"{API_BASE}/customers",
        json={"name": name, "slug": slug},
        headers=_headers(),
        timeout=15,
    )
    if r.ok:
        print(f"  [customer] created '{slug}'")
    elif r.status_code == 409:
        print(f"  [customer] '{slug}' already exists — skipping")
    else:
        raise RuntimeError(f"create customer failed {r.status_code}: {r.text[:200]}")


def _ensure_person(slug: str, person: dict) -> None:
    r = requests.get(f"{API_BASE}/customers/{slug}/people", headers=_headers(), timeout=10)
    if r.ok and any(p.get("name") == person["name"] for p in r.json()):
        print(f"  [person]   '{person['name']}' already exists — skipping")
        return
    r = requests.post(
        f"{API_BASE}/customers/{slug}/people",
        json=person,
        headers=_headers(),
        timeout=10,
    )
    if r.ok:
        print(f"  [person]   created '{person['name']}' ({person.get('role', '')})")
    else:
        print(f"  [person]   WARN: {r.status_code}: {r.text[:100]}")


def _upload_doc(slug: str, filename: str, doc_type: str, content: bytes) -> None:
    files = {"file": (filename, io.BytesIO(content), "application/octet-stream")}
    data  = {"doc_type": doc_type}
    t0 = time.perf_counter()
    r = requests.post(
        f"{API_BASE}/customers/{slug}/upload",
        files=files,
        data=data,
        headers=_headers(include_content_type=False),
        timeout=600,
    )
    dt = (time.perf_counter() - t0) * 1000
    if r.ok:
        chunks = r.json().get("chunks", "?")
        print(f"  [upload]   OK  {chunks:>3} chunks  [{doc_type:<20}]  {filename}  ({dt:.0f}ms)")
    elif r.status_code == 409:
        print(f"  [upload]   already indexed — {filename}")
    else:
        print(f"  [upload]   FAIL {r.status_code}  {filename}")
        print(f"             {r.text[:200]}")


def _seed_account(account: dict) -> None:
    slug = account["slug"]
    print(f"\n{'─' * 60}")
    print(f"  {account['name']} ({slug})")
    print(f"{'─' * 60}")
    _ensure_customer(account["name"], slug)
    _ensure_person(slug, account["person"])
    for filename, doc_type, content in account["docs"]:
        _upload_doc(slug, filename, doc_type, content)
    # Stamp last_call_date so days_since_last_call is correct in the health score
    # (the transcript upload will set doc_date but not last_call_date on the customer row)
    # The transcript parser does not update last_call_date automatically — that is done
    # via the customers table. For demo purposes we patch it directly via the notes field.
    # (No dedicated endpoint exists yet; the brief workflow reads last_call_date from the
    # transcript doc_date instead, so this is informational only for now.)


def main() -> None:
    print("=" * 60)
    print("DEMO SEED — 3 accounts")
    print(f"  API:       {API_BASE}")
    print(f"  Workspace: {WORKSPACE}")
    print("=" * 60)

    try:
        resp = requests.get(f"{API_BASE}/health", timeout=5)
        assert resp.ok, f"health check failed: {resp.status_code}"
    except Exception as e:
        print(f"\n[error] API not reachable at {API_BASE}: {e}")
        sys.exit(1)

    for account in ACCOUNTS:
        _seed_account(account)

    print(f"\n{'=' * 60}")
    print("DONE")
    print()
    print("  Account          Slug       Expected Health")
    print("  ─────────────────────────────────────────────")
    print("  Cascadia Health  cascadia   Healthy  (~95/100)")
    print("  Meridian         meridian   At Risk  (~65/100)")
    print("  Novus Financial  novus      Critical (~0/100)")
    print()
    print("Open the UI → Account Health tab to verify scores.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
