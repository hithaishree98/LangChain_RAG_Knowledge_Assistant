"""
demo_seed.py — Seed the API with a complete demo customer workspace.

Creates the "meridian" demo customer, uploads all sample documents,
and adds a demo person (needed for the Exec 1:1 tab). Safe to re-run:
existing resources are skipped (409 / already-exists) rather than duplicated.

Usage:
    python api/scripts/demo_seed.py

Expects the API running on localhost:8000 (or API_BASE_URL env var).
Credentials are read from DEMO_WORKSPACE / DEMO_PASSKEY (default: "demo" / "demo").
"""

import os
import sys
import time
from pathlib import Path

import requests

API_BASE   = os.getenv("API_BASE_URL", "http://localhost:8000")
WORKSPACE  = os.getenv("DEMO_WORKSPACE", "demo")
PASSKEY    = os.getenv("DEMO_PASSKEY",   "demo")
API_KEY    = os.getenv("API_KEY", "")

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "experiment_kit" / "sample_docs"

CUSTOMER_NAME = "Meridian"
CUSTOMER_SLUG = "meridian"

DEMO_PERSON = {
    "name": "Sarah Chen",
    "role": "VP Engineering",
    "email": "sarah@meridian.example.com",
}

SAMPLE_DOCS = [
    ("Meridian_SOW_v2.pdf",           "2024-01-01_account-notes_sow.pdf",            "account_notes"),
    ("Q3_QBR_Notes.docx",             "2024-Q3_qbr_notes.docx",                      "account_notes"),
    ("Orion_Integration_Guide.html",  "2024-01-01_solution_orion-guide.html",         "account_notes"),
    ("customer_call_2024_09_15.txt",  "2024-09-15_transcript_status-call.txt",        "transcript"),
    ("TICK-4521.json",                "2024-09-01_ticket_TICK-4521.json",             "ticket"),
    ("TICK-4602.json",                "2024-09-15_ticket_TICK-4602.json",             "ticket"),
    ("Meridian_commitments.json",     "2024-10-01_commitments_meridian.json",         "commitment_tracker"),
]

_token: "str | None" = None


def _get_token() -> str:
    global _token
    if _token:
        return _token
    r = requests.post(f"{API_BASE}/auth/token",
                      json={"workspace": WORKSPACE, "passkey": PASSKEY}, timeout=10)
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


def ensure_customer() -> None:
    r = requests.post(
        f"{API_BASE}/customers",
        json={"name": CUSTOMER_NAME, "slug": CUSTOMER_SLUG},
        headers=_headers(),
        timeout=15,
    )
    if r.ok:
        print(f"[customer] created '{CUSTOMER_SLUG}'")
    elif r.status_code == 409:
        print(f"[customer] '{CUSTOMER_SLUG}' already exists — skipping")
    else:
        raise RuntimeError(f"create customer failed {r.status_code}: {r.text[:200]}")


def ensure_person() -> None:
    r = requests.get(
        f"{API_BASE}/customers/{CUSTOMER_SLUG}/people",
        headers=_headers(),
        timeout=10,
    )
    if r.ok:
        people = r.json()
        if any(p.get("name") == DEMO_PERSON["name"] for p in people):
            print(f"[person]   '{DEMO_PERSON['name']}' already exists — skipping")
            return
    r = requests.post(
        f"{API_BASE}/customers/{CUSTOMER_SLUG}/people",
        json=DEMO_PERSON,
        headers=_headers(),
        timeout=10,
    )
    if r.ok:
        print(f"[person]   created '{DEMO_PERSON['name']}' ({DEMO_PERSON['role']})")
    else:
        print(f"[person]   WARN: {r.status_code}: {r.text[:100]}")


def upload_docs() -> None:
    for disk_name, upload_name, doc_type in SAMPLE_DOCS:
        path = SAMPLE_DIR / disk_name
        if not path.exists():
            print(f"[skip]     missing sample file: {path.name}")
            continue
        with open(path, "rb") as f:
            files = {"file": (upload_name, f, "application/octet-stream")}
            data  = {"doc_type": doc_type}
            t0 = time.perf_counter()
            r = requests.post(
                f"{API_BASE}/customers/{CUSTOMER_SLUG}/upload",
                files=files, data=data,
                headers=_headers(include_content_type=False),
                timeout=600,
            )
        dt = (time.perf_counter() - t0) * 1000
        if r.ok:
            chunks = r.json().get("chunks", "?")
            print(f"[upload]   OK  {chunks:>4} chunks  [{doc_type}]  {upload_name}  ({dt:.0f}ms)")
        else:
            print(f"[upload]   FAIL {r.status_code}  [{doc_type}]  {upload_name}")
            print(f"           {r.text[:150]}")


def main() -> None:
    print("=" * 70)
    print("DEMO SEED — Meridian workspace")
    print(f"  API:       {API_BASE}")
    print(f"  Workspace: {WORKSPACE}")
    print("=" * 70)

    # Verify API is up
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=5)
        assert resp.ok, f"health check failed: {resp.status_code}"
    except Exception as e:
        print(f"\n[error] API not reachable at {API_BASE}: {e}")
        sys.exit(1)

    ensure_customer()
    ensure_person()

    print(f"\n[upload]   uploading {len(SAMPLE_DOCS)} sample documents ...")
    upload_docs()

    print("\n[done]")
    print(f"  Customer slug : {CUSTOMER_SLUG}")
    print(f"  Workspace     : {WORKSPACE}")
    print(f"  Person added  : {DEMO_PERSON['name']} ({DEMO_PERSON['role']})")
    print("\nAll primary tabs (Pre-Meeting Brief, Exec 1:1, Q&A, Upload) are now ready.")


if __name__ == "__main__":
    main()
