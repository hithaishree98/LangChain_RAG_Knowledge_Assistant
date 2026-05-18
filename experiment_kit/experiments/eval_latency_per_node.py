"""
Experiment 5 — Per-node latency breakdown

Starts the API with NODE_TIMING=1, fires all queries from eval_set.csv at /query,
then prints p50/p90/p95 per node (query_rewrite, retrieve, reason, completeness).

Usage:
    python experiment_kit/experiments/exp5_latency.py

Requires: exp1 completed first (eval_full workspace must exist).
"""
import csv
import os
import sys
import time
from pathlib import Path

import requests

REPO_ROOT   = Path(__file__).resolve().parents[2]
KIT_ROOT    = REPO_ROOT / "experiment_kit"
TIMING_LOG  = REPO_ROOT / "api" / "data" / "node_timings.jsonl"
API_BASE    = "http://localhost:8000"

sys.path.insert(0, str(KIT_ROOT / "experiments"))

from api_utils import start_api, stop_api


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("EXPERIMENT 5 — PER-NODE LATENCY BREAKDOWN")
    print("=" * 72)

    # Clear timing log
    TIMING_LOG.parent.mkdir(parents=True, exist_ok=True)
    TIMING_LOG.write_text("")
    print(f"[cleared] {TIMING_LOG}\n")

    api_key = os.getenv("API_KEY", "")
    headers = {"X-API-Key": api_key} if api_key else {}

    # Load questions from eval set
    questions = []
    with open(KIT_ROOT / "eval_set.csv", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            q = row.get("question", "").strip()
            if q:
                questions.append(q)

    proc = start_api({
        "CHUNKING_MODE": "full",
        "RETRIEVAL_MODE": "full",
        "NODE_TIMING": "1",
    })

    try:
        print(f"[run] firing {len(questions)} queries at /query\n")
        print(f"{'#':>3}  {'wall_ms':>8}  {'status':>7}  query")
        print("-" * 80)

        for i, q in enumerate(questions, 1):
            t0 = time.perf_counter()
            try:
                r = requests.post(
                    f"{API_BASE}/query",
                    json={"question": q, "customer_id": "eval-full"},
                    headers=headers,
                    timeout=120,
                )
                status = r.status_code
            except Exception as e:
                status = 0
                print(f"  [err] {e}")
            elapsed = (time.perf_counter() - t0) * 1000
            print(f"{i:>3}  {elapsed:>8.0f}  {status:>7}  {q[:60]}")
    finally:
        stop_api(proc)

    print("\n[analyze]")
    import exp5_analyze
    exp5_analyze.main()


if __name__ == "__main__":
    main()
