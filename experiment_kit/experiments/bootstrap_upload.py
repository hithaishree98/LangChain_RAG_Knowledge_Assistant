"""
bootstrap_upload.py — upload all sample docs to the API under a given user_id.

Usage:
    python bootstrap_upload.py --user_id eval_baseline
    python bootstrap_upload.py --user_id eval_sentence
    python bootstrap_upload.py --user_id eval_full

Expects the API running on localhost:8000. If API_KEY is set in env, sends it as X-API-Key.
"""
import argparse
import os
import sys
import time

import requests

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "")
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "sample_docs")
SAMPLES = [
    "Meridian_SOW_v2.pdf",
    "Q3_QBR_Notes.docx",
    "Orion_Integration_Guide.html",
    "customer_call_2024_09_15.txt",
    "TICK-4521.json",
    "TICK-4602.json",
]


def wipe_existing(user_id):
    """Delete anything this user has already uploaded."""
    r = requests.get(f"{API_BASE}/list-docs", params={"user_id": user_id},
                     headers=HEADERS, timeout=15)
    if r.status_code != 200:
        print(f"[warn] list-docs returned {r.status_code}")
        return
    docs = r.json()
    for d in docs:
        del_r = requests.post(f"{API_BASE}/delete-doc",
            json={"file_id": d["id"], "user_id": user_id},
            headers=HEADERS, timeout=15)
        print(f"  deleted existing: {d['filename']}  ({del_r.status_code})")


def upload_all(user_id):
    for name in SAMPLES:
        path = os.path.join(SAMPLE_DIR, name)
        if not os.path.exists(path):
            print(f"[err] missing sample: {path}"); continue
        with open(path, "rb") as f:
            files = {"file": (name, f, "application/octet-stream")}
            params = {"user_id": user_id, "doc_type": "auto"}
            t0 = time.perf_counter()
            r = requests.post(f"{API_BASE}/upload-doc", files=files, params=params,
                              headers=HEADERS, timeout=1800)
            dt = (time.perf_counter() - t0) * 1000
            status = "OK" if r.status_code == 200 else f"FAIL({r.status_code})"
            print(f"  {status:12s}  {name:45s}  {dt:.0f}ms")
            if r.status_code != 200:
                print(f"    body: {r.text[:200]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--user_id", required=True,
                   help="workspace to upload under (e.g. eval_baseline / eval_sentence / eval_full)")
    p.add_argument("--wipe", action="store_true", help="delete any existing docs for this user first")
    args = p.parse_args()

    if args.wipe:
        print(f"[wipe] clearing existing docs for user_id={args.user_id}")
        wipe_existing(args.user_id)

    print(f"[upload] uploading {len(SAMPLES)} docs under user_id={args.user_id}")
    upload_all(args.user_id)
    print("done.")
