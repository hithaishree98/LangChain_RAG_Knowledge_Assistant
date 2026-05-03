"""
bootstrap_upload.py — upload all sample docs to a customer workspace.

Usage:
    python bootstrap_upload.py --customer meridian
    python bootstrap_upload.py --customer eval_full --wipe

Expects the API running on localhost:8000.
Filenames are automatically mapped to the required YYYY-MM-DD_keyword_descriptor.ext convention.
Any upload failure prints the full API error so filename/type problems are immediately visible.
"""
import argparse
import os
import sys
import time

import requests

API_BASE    = os.getenv("API_BASE_URL", "http://localhost:8000")
API_KEY     = os.getenv("API_KEY", "")
WORKSPACE   = os.getenv("EVAL_WORKSPACE", "eval-default")
PASSKEY     = os.getenv("EVAL_PASSKEY",   "eval-default-passkey")

_TOKEN: "str | None" = None


def _get_token() -> str:
    global _TOKEN
    if _TOKEN:
        return _TOKEN
    r = requests.post(
        f"{API_BASE}/auth/token",
        json={"workspace": WORKSPACE, "passkey": PASSKEY},
        timeout=10,
    )
    if not r.ok:
        raise RuntimeError(f"auth/token failed {r.status_code}: {r.text[:200]}")
    _TOKEN = r.json()["token"]
    return _TOKEN


def _headers() -> dict:
    h = {}
    if API_KEY:
        h["X-API-Key"] = API_KEY
    h["Authorization"] = f"Bearer {_get_token()}"
    return h

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "sample_docs")

# Each entry is (disk_filename, upload_filename, doc_type).
# upload_filename must follow the API convention: YYYY-MM-DD_<keyword>_<descriptor>.<ext>
# The keyword must let the API infer a valid doc_type (or pass doc_type explicitly).
# .json files have no extension fallback so the keyword is required (ticket / commitments).
SAMPLES = [
    ("Meridian_SOW_v2.pdf",           "2024-01-01_account-notes_sow.pdf",           "account_notes"),
    ("Q3_QBR_Notes.docx",             "2024-Q3_qbr_notes.docx",                     "account_notes"),
    ("Orion_Integration_Guide.html",  "2024-01-01_solution_orion-guide.html",        "account_notes"),
    ("customer_call_2024_09_15.txt",  "2024-09-15_transcript_status-call.txt",       "transcript"),
    ("TICK-4521.json",                "2024-09-01_ticket_TICK-4521.json",            "ticket"),
    ("TICK-4602.json",                "2024-09-15_ticket_TICK-4602.json",            "ticket"),
    ("Meridian_commitments.json",     "2024-10-01_commitments_meridian.json",        "commitment_tracker"),
]


def ensure_customer(slug: str, name: str = None) -> None:
    """Create the customer if it doesn't already exist."""
    display_name = name or slug.replace("-", " ").title()
    r = requests.post(
        f"{API_BASE}/customers",
        json={"name": display_name, "slug": slug},
        headers=_headers(),
        timeout=15,
    )
    if r.ok:
        print(f"[customer] created '{slug}'")
    elif r.status_code == 409:
        print(f"[customer] '{slug}' already exists — skipping creation")
    else:
        raise RuntimeError(f"create customer failed {r.status_code}: {r.text[:200]}")


def wipe_existing(slug: str) -> None:
    """Delete all documents already uploaded for this customer."""
    r = requests.get(f"{API_BASE}/customers/{slug}/documents", headers=_headers(), timeout=15)
    if r.status_code != 200:
        print(f"[warn] list documents returned {r.status_code}: {r.text}")
        return
    for d in r.json():
        fid = d.get("id") or d.get("file_id")
        del_r = requests.delete(
            f"{API_BASE}/customers/{slug}/documents/{fid}", headers=_headers(), timeout=15)
        status = "deleted" if del_r.ok else f"FAIL({del_r.status_code}): {del_r.text}"
        print(f"  {status}  {d.get('filename', fid)}")


def upload_all(slug: str) -> None:
    for disk_name, upload_name, doc_type in SAMPLES:
        path = os.path.join(SAMPLE_DIR, disk_name)
        if not os.path.exists(path):
            print(f"[err] missing sample: {path}"); continue
        with open(path, "rb") as f:
            files = {"file": (upload_name, f, "application/octet-stream")}
            data = {"doc_type": doc_type}
            t0 = time.perf_counter()
            r = requests.post(
                f"{API_BASE}/customers/{slug}/upload",
                files=files, data=data, headers=_headers(), timeout=1800,
            )
            dt = (time.perf_counter() - t0) * 1000
            if r.ok:
                chunks = r.json().get("chunks", "?")
                print(f"  OK  {chunks:>4} chunks  [{doc_type}]  {upload_name}  ({dt:.0f}ms)")
            else:
                print(f"  FAIL({r.status_code})  [{doc_type}]  {upload_name}  ({dt:.0f}ms)")
                print(f"    ERROR: {r.text}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--customer", required=True,
                   help="customer slug to upload under (e.g. meridian / eval_full)")
    p.add_argument("--wipe", action="store_true",
                   help="delete existing docs for this customer before uploading")
    args = p.parse_args()

    ensure_customer(args.customer)

    if args.wipe:
        print(f"[wipe] clearing existing docs for customer={args.customer}")
        wipe_existing(args.customer)

    print(f"[upload] uploading {len(SAMPLES)} docs to customer '{args.customer}' ...")
    upload_all(args.customer)
    print("done.")
