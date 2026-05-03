"""
exp11_brief_type.py — compare /brief/pre-meeting vs /brief/exec-1on1.

Each endpoint produces a structurally different brief for the same customer:
  /brief/pre-meeting — account-wide view: open items, commitments, posture
  /brief/exec-1on1   — person-centric view: role, signals, asks, approach

This experiment answers:
  1. Which sections are populated vs empty under each brief type?
  2. Are the section_population_rates different between the two endpoints?
  3. How do latency and stale_warnings compare?
  4. Does /brief/exec-1on1 surface person-specific signals not in pre-meeting?

Usage:
    python experiment_kit/experiments/exp11_brief_type.py \
        --customer cascadia-inc --person-id 42

Requires: GOOGLE_API_KEY, API_KEY, EVAL_WORKSPACE, EVAL_PASSKEY set in env.
          API running on localhost:8000.
          At least one customer with uploaded documents, and at least one person
          registered via POST /customers/{slug}/people.
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
KIT_ROOT = REPO_ROOT / "experiment_kit"

BASE_URL        = os.getenv("API_BASE_URL", "http://localhost:8000")
AUTH_URL        = BASE_URL + "/auth/token"
PRE_MEETING_URL = BASE_URL + "/brief/pre-meeting"
EXEC_1ON1_URL   = BASE_URL + "/brief/exec-1on1"
API_KEY         = os.getenv("API_KEY", "")
EVAL_WORKSPACE  = os.getenv("EVAL_WORKSPACE", "eval-default")
EVAL_PASSKEY    = os.getenv("EVAL_PASSKEY", "eval-default-passkey")
RESULTS_DIR     = KIT_ROOT / "eval_results"

INTER_CALL_SLEEP = float(os.getenv("INTER_QUERY_SLEEP", "5"))

# How many as_of_date variants to test per brief type.
# Each date tests whether recency-sensitive sections (recent_changes) vary.
TEST_DATES = [None]   # None = today (API default)


def _mint_token() -> str:
    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    r = requests.post(
        AUTH_URL,
        json={"workspace": EVAL_WORKSPACE, "passkey": EVAL_PASSKEY},
        headers=headers,
        timeout=10,
    )
    r.raise_for_status()
    return r.json()["token"]


def _headers(token: str) -> dict:
    h = {"Authorization": f"Bearer {token}"}
    if API_KEY:
        h["X-API-Key"] = API_KEY
    return h


def _section_presence_pre_meeting(brief: dict) -> dict:
    """Which sections are non-empty in a PreMeetingBrief."""
    return {
        "account_summary":          bool(brief.get("account_summary")),
        "open_items":               bool(brief.get("open_items")),
        "recent_changes":           bool(brief.get("recent_changes")),
        "outstanding_commitments":  bool(brief.get("outstanding_commitments")),
        "overdue_commitments":      bool(brief.get("overdue_commitments")),
        "anticipated_questions":    bool(brief.get("anticipated_questions")),
        "recommended_posture":      bool(brief.get("recommended_posture")),
    }


def _section_presence_exec(brief: dict) -> dict:
    """Which sections are non-empty in an ExecBrief."""
    return {
        "role_and_tenure":    bool(brief.get("role_and_tenure")),
        "stated_position":    bool(brief.get("stated_position")),
        "recent_signals":     bool(brief.get("recent_signals")),
        "open_asks":          bool(brief.get("open_asks")),
        "recommended_approach": bool(brief.get("recommended_approach")),
    }


def run_pre_meeting(customer_id: str, token: str, n_runs: int = 3) -> dict:
    """Call /brief/pre-meeting n_runs times and aggregate metrics."""
    print(f"\n── PRE-MEETING BRIEF  (customer={customer_id}, runs={n_runs}) ──")
    lats = []
    section_hits: dict = {}
    stale_counts: list = []
    conflict_counts: list = []
    errors = 0

    for i in range(1, n_runs + 1):
        t0 = time.perf_counter()
        try:
            r = requests.post(
                PRE_MEETING_URL,
                json={"customer_id": customer_id},
                headers=_headers(token),
                timeout=180,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if not r.ok:
                errors += 1
                print(f"  [{i}] ERROR HTTP {r.status_code}: {r.text[:150]}")
                if i < n_runs:
                    time.sleep(INTER_CALL_SLEEP)
                continue
            brief = r.json()
        except Exception as e:
            errors += 1
            print(f"  [{i}] ERROR {e}")
            if i < n_runs:
                time.sleep(INTER_CALL_SLEEP)
            continue

        lats.append(elapsed_ms)
        stale_counts.append(len(brief.get("stale_warnings", [])))
        conflict_counts.append(len(brief.get("conflicts", [])))
        sec = _section_presence_pre_meeting(brief)
        for s, present in sec.items():
            section_hits[s] = section_hits.get(s, 0) + (1 if present else 0)

        print(f"  [{i}] lat={elapsed_ms:.0f}ms  "
              f"stale={stale_counts[-1]}  conflicts={conflict_counts[-1]}")
        print(f"       sections: " + "  ".join(k for k, v in sec.items() if v))
        if i < n_runs:
            time.sleep(INTER_CALL_SLEEP)

    n = len(lats)
    if n == 0:
        return {"brief_type": "pre_meeting", "error": "all calls failed", "n": 0}

    p50 = statistics.median(lats)
    p95 = sorted(lats)[min(int(n * 0.95), n - 1)]
    return {
        "brief_type":              "pre_meeting",
        "n":                       n,
        "errors":                  errors,
        "p50_latency_ms":          round(p50, 1),
        "p95_latency_ms":          round(p95, 1),
        "avg_stale_warnings":      round(sum(stale_counts) / n, 2),
        "avg_conflicts":           round(sum(conflict_counts) / n, 2),
        "section_population_rate": {s: round(h / n, 2) for s, h in section_hits.items()},
    }


def run_exec_1on1(customer_id: str, person_id: str, token: str, n_runs: int = 3) -> dict:
    """Call /brief/exec-1on1 n_runs times and aggregate metrics."""
    print(f"\n── EXEC 1:1 BRIEF  (customer={customer_id}, person={person_id}, runs={n_runs}) ──")
    lats = []
    section_hits: dict = {}
    stale_counts: list = []
    conflict_counts: list = []
    errors = 0

    for i in range(1, n_runs + 1):
        t0 = time.perf_counter()
        try:
            r = requests.post(
                EXEC_1ON1_URL,
                json={"customer_id": customer_id, "person_id": person_id},
                headers=_headers(token),
                timeout=180,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if not r.ok:
                errors += 1
                print(f"  [{i}] ERROR HTTP {r.status_code}: {r.text[:150]}")
                if i < n_runs:
                    time.sleep(INTER_CALL_SLEEP)
                continue
            brief = r.json()
        except Exception as e:
            errors += 1
            print(f"  [{i}] ERROR {e}")
            if i < n_runs:
                time.sleep(INTER_CALL_SLEEP)
            continue

        lats.append(elapsed_ms)
        stale_counts.append(len(brief.get("stale_warnings", [])))
        conflict_counts.append(len(brief.get("conflicts", [])))
        sec = _section_presence_exec(brief)
        for s, present in sec.items():
            section_hits[s] = section_hits.get(s, 0) + (1 if present else 0)

        print(f"  [{i}] lat={elapsed_ms:.0f}ms  "
              f"stale={stale_counts[-1]}  conflicts={conflict_counts[-1]}")
        print(f"       sections: " + "  ".join(k for k, v in sec.items() if v))
        if i < n_runs:
            time.sleep(INTER_CALL_SLEEP)

    n = len(lats)
    if n == 0:
        return {"brief_type": "exec_1on1", "error": "all calls failed", "n": 0}

    p50 = statistics.median(lats)
    p95 = sorted(lats)[min(int(n * 0.95), n - 1)]
    return {
        "brief_type":              "exec_1on1",
        "n":                       n,
        "errors":                  errors,
        "p50_latency_ms":          round(p50, 1),
        "p95_latency_ms":          round(p95, 1),
        "avg_stale_warnings":      round(sum(stale_counts) / n, 2),
        "avg_conflicts":           round(sum(conflict_counts) / n, 2),
        "section_population_rate": {s: round(h / n, 2) for s, h in section_hits.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--customer", required=True,
                        help="Customer slug, e.g. cascadia-inc")
    parser.add_argument("--person-id", required=True,
                        help="Person ID (integer) for exec-1on1 brief")
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of calls per brief type (default: 3)")
    args = parser.parse_args()

    print("=" * 72)
    print("EXPERIMENT 11 — /brief/pre-meeting vs /brief/exec-1on1 COMPARISON")
    print("=" * 72)
    print()
    print("Minting eval token ...")
    try:
        token = _mint_token()
    except Exception as e:
        print(f"[err] auth failed: {e}")
        sys.exit(1)
    print("  OK")

    pre = run_pre_meeting(args.customer, token, n_runs=args.runs)
    exec_ = run_exec_1on1(args.customer, args.person_id, token, n_runs=args.runs)

    results = {"pre_meeting": pre, "exec_1on1": exec_}

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "exp11_brief_type_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}")

    # ── Comparison table ─────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("COMPARISON")
    print("=" * 72)
    cols = ["p50_latency_ms", "p95_latency_ms", "avg_stale_warnings", "avg_conflicts"]
    print(f"  {'metric':<24} {'pre_meeting':>14} {'exec_1on1':>14}")
    print("  " + "-" * 54)
    for col in cols:
        pm  = pre.get(col, "N/A")
        ex  = exec_.get(col, "N/A")
        print(f"  {col:<24} {str(pm):>14} {str(ex):>14}")

    print()
    print("  section_population_rate  (fraction of runs where section was non-empty):")
    all_sections = set(pre.get("section_population_rate", {}).keys()) | \
                   set(exec_.get("section_population_rate", {}).keys())
    for sec in sorted(all_sections):
        pm  = pre.get("section_population_rate", {}).get(sec, "—")
        ex  = exec_.get("section_population_rate", {}).get(sec, "—")
        print(f"    {sec:<30} {str(pm):>8}  {str(ex):>8}")

    print()
    print("Reading these results:")
    print("  pre_meeting: expect high section_population_rate for open_items,")
    print("    outstanding_commitments, and recommended_posture.")
    print("    recent_changes depends on whether docs were uploaded < 14 days ago.")
    print()
    print("  exec_1on1: expect role_and_tenure and recommended_approach always")
    print("    populated; recent_signals/open_asks depend on transcript coverage.")
    print()
    print("  avg_stale_warnings > 0: corpus has docs older than the staleness")
    print("    threshold. Upload fresher transcripts or adjust STALE_DAYS in utils/staleness.py.")
    print()
    print("  If exec_1on1 sections are mostly empty, the person has no docs that")
    print("    mention them by name. Transcripts need to reference the person.")


if __name__ == "__main__":
    main()
