"""
Experiment 6b — Real cost measurement and tiktoken validation

Runs real /query requests, captures the LLM's actual token usage from the
response metadata, and compares against tiktoken's estimate (produced by
experiment 6a).

Output tells you three things:
  1. What it ACTUALLY costs per query (from the LLM usage field — exact)
  2. How accurate tiktoken is as an estimator (so you can trust 6a or not)
  3. Total cost of running this experiment (so you know what you spent)

Preconditions:
  - API running with TOKEN_LOGGING=1 set at launch
  - Sample docs uploaded under user_id=eval_full
  - experiment 6a has already been run (produces exp6_cost.json for comparison)

Note on token log schema: log entries still use keys `groq_prompt_tokens` and
`groq_completion_tokens` for historical continuity. They hold the provider's
reported token counts regardless of which provider is active (Gemini today).

Usage:
    # Run 6a first to generate the tiktoken baseline
    python experiment_kit/experiments/exp6_cost.py

    # Start API with token logging enabled (see instructions below)
    # Then run this:
    python experiment_kit/experiments/exp6b_real_cost.py
"""
import json
import os
import statistics
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
KIT_ROOT  = REPO_ROOT / "experiment_kit"

sys.path.insert(0, str(KIT_ROOT / "experiments"))
from api_utils import start_api, stop_api

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
API_KEY  = os.getenv("API_KEY", "")
HEADERS  = {"X-API-Key": API_KEY} if API_KEY else {}

# Same 10 queries as 6a — this is how we enable direct comparison
SAMPLE_QUERIES = [
    "What is the current open P1 issue for Meridian?",
    "What is Meridian's default retrieval top-k value?",
    "What are Meridian's analyst seat usage and upsell opportunity?",
    "What did Sarah Chen say about the login latency issue on the call?",
    "What caused the duplicate invoice issue in TICK-4602?",
    "When is the EU region (eu-west-2) targeted for GA?",
    "How often are backup snapshots taken?",
    "What is the workaround for the login latency issue?",
    "What version is the Salesforce connector and what is its volume limit?",
    "What is Meridian's monthly platform fee?",
]

# Pricing — Gemini 2.5-flash. Verify at https://ai.google.dev/pricing.
# Override via env if rates change or a different model is used.
INPUT_PRICE_PER_M  = float(os.getenv("LLM_INPUT_PRICE_PER_M",  "0.30"))
OUTPUT_PRICE_PER_M = float(os.getenv("LLM_OUTPUT_PRICE_PER_M", "2.50"))

# Where token log is written (must match TOKEN_LOG_FILE in graph/nodes.py)
TOKEN_LOG_FILE = REPO_ROOT / "api" / "data" / "token_usage.jsonl"


def clear_log():
    """Start with a fresh token log so we only see this experiment's data."""
    if TOKEN_LOG_FILE.exists():
        TOKEN_LOG_FILE.unlink()
    TOKEN_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_LOG_FILE.touch()


def run_query(query: str, user_id: str = "eval-full") -> dict:
    """Call /query with a single question. Returns the API response + wall-clock time."""
    t0 = time.perf_counter()
    try:
        r = requests.post(
            f"{API_BASE}/query",
            json={"question": query, "customer_id": user_id},
            headers=HEADERS,
            timeout=120,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {"status": r.status_code, "body": r.json() if r.ok else r.text,
                "elapsed_ms": elapsed_ms}
    except Exception as e:
        return {"status": 0, "body": str(e),
                "elapsed_ms": (time.perf_counter() - t0) * 1000}


def read_token_log() -> list:
    """Read all token usage entries from the log file.

    Normalizes legacy key names (groq_prompt_tokens / groq_completion_tokens,
    written before the provider-agnostic rename) to the current schema
    (prompt_tokens / completion_tokens). Consumers downstream in this file
    reference the legacy keys for historical continuity, so we also mirror
    the normalized values back under the legacy keys.
    """
    if not TOKEN_LOG_FILE.exists():
        return []
    entries = []
    with TOKEN_LOG_FILE.open() as f:
        for line in f:
            try:
                e = json.loads(line)
            except Exception:
                continue
            # New schema: prompt_tokens / completion_tokens
            # Old schema: groq_prompt_tokens / groq_completion_tokens
            prompt = e.get("prompt_tokens", e.get("groq_prompt_tokens", 0)) or 0
            completion = e.get("completion_tokens", e.get("groq_completion_tokens", 0)) or 0
            e["prompt_tokens"]          = prompt
            e["completion_tokens"]      = completion
            e["groq_prompt_tokens"]     = prompt      # back-compat for existing code paths
            e["groq_completion_tokens"] = completion
            entries.append(e)
    return entries


def load_6a_baseline() -> dict:
    """Load 6a's tiktoken estimates for side-by-side comparison."""
    path = KIT_ROOT / "eval_results" / "exp6_cost.json"
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def main():
    print("=" * 72)
    print("EXPERIMENT 6b — REAL COST + TIKTOKEN VALIDATION")
    print("=" * 72)
    print(f"Pricing: input=${INPUT_PRICE_PER_M}/M  output=${OUTPUT_PRICE_PER_M}/M")
    print()

    print("[check] clearing token log so we only see this experiment's data")
    clear_log()

    proc = start_api({
        "CHUNKING_MODE": "full",
        "RETRIEVAL_MODE": "full",
        "TOKEN_LOGGING": "1",
    })

    try:
        # ── Fire the queries ────────────────────────────────────────────────
        print(f"\n[run] sending {len(SAMPLE_QUERIES)} queries to /brief\n")
        print(f"{'#':>3}  {'wall_ms':>8}  {'status':>7}  query")
        print("-" * 100)

        wall_times = []
        per_query_wall = []
        for i, q in enumerate(SAMPLE_QUERIES, 1):
            resp = run_query(q)
            print(f"{i:>3}  {resp['elapsed_ms']:>8.0f}  {resp['status']:>7}  {q[:70]}")
            wall_times.append(resp["elapsed_ms"])
            per_query_wall.append((q, resp["elapsed_ms"]))
            if resp["status"] != 200:
                print(f"     [err body]: {str(resp['body'])[:200]}")
            time.sleep(0.3)
    finally:
        stop_api(proc)

    # ── Read the token log ──────────────────────────────────────────────────
    print("\n[read] parsing token usage log")
    entries = read_token_log()
    if not entries:
        print("[err] no token entries found in log — TOKEN_LOGGING may not have fired.")
        print("      Expected log file:", TOKEN_LOG_FILE)
        return 1
    print(f"[read] got {len(entries)} log entries across {len(SAMPLE_QUERIES)} queries")

    # ── Per-call-type aggregation ───────────────────────────────────────────
    print()
    print("=" * 72)
    print("REAL TOKEN USAGE (by call type)")
    print("=" * 72)
    by_call = {}
    for e in entries:
        by_call.setdefault(e["call"], []).append(e)

    print(f"{'call':<16} {'count':>6} {'llm_in_avg':>12} {'llm_out_avg':>13} "
          f"{'tik_in_avg':>11} {'tik_error':>10}")
    print("-" * 75)
    for call, es in sorted(by_call.items()):
        groq_in  = [e["groq_prompt_tokens"]     for e in es]
        groq_out = [e["groq_completion_tokens"] for e in es]
        tik_in   = [e["tiktoken_estimate"]       for e in es]
        tik_err  = [
            (abs(e["tiktoken_estimate"] - e["groq_prompt_tokens"]) / e["groq_prompt_tokens"] * 100)
            if e["groq_prompt_tokens"] > 0 else 0
            for e in es
        ]
        print(f"{call:<16} {len(es):>6} "
              f"{statistics.mean(groq_in):>12.1f} "
              f"{statistics.mean(groq_out):>13.1f} "
              f"{statistics.mean(tik_in):>11.1f} "
              f"{statistics.mean(tik_err):>9.1f}%")

    # ── Real cost per query ─────────────────────────────────────────────────
    # Assumes calls-per-query distributes evenly. More rigorous would be to
    # correlate by timestamp; for 10 queries the simpler model is fine.
    total_groq_in  = sum(e["groq_prompt_tokens"]     for e in entries)
    total_groq_out = sum(e["groq_completion_tokens"] for e in entries)
    real_cost_total = (total_groq_in  * INPUT_PRICE_PER_M  / 1_000_000 +
                       total_groq_out * OUTPUT_PRICE_PER_M / 1_000_000)
    real_cost_per_query = real_cost_total / len(SAMPLE_QUERIES)

    print()
    print("=" * 72)
    print("REAL COST")
    print("=" * 72)
    print(f"  Total LLM input tokens  (all queries): {total_groq_in:>10,}")
    print(f"  Total LLM output tokens (all queries): {total_groq_out:>10,}")
    print(f"  Real cost for this experiment:         ${real_cost_total:.6f}")
    print(f"  Real cost per query (avg):             ${real_cost_per_query:.6f}")
    print(f"  Avg wall-clock latency per query:       {statistics.mean(wall_times):.0f} ms")

    # ── Compare to 6a ───────────────────────────────────────────────────────
    baseline = load_6a_baseline()
    if baseline:
        est_cost_per_query = baseline.get("avg_cost_per_query", 0)
        est_input_avg      = baseline.get("avg_input_tokens", 0)
        est_output_avg     = baseline.get("avg_output_tokens", 0)

        # Compute per-query real averages from our groq log
        real_input_avg  = total_groq_in  / len(SAMPLE_QUERIES)
        real_output_avg = total_groq_out / len(SAMPLE_QUERIES)

        in_error  = abs(est_input_avg  - real_input_avg)  / max(real_input_avg, 1)  * 100
        out_error = abs(est_output_avg - real_output_avg) / max(real_output_avg, 1) * 100
        cost_error = abs(est_cost_per_query - real_cost_per_query) / max(real_cost_per_query, 1e-9) * 100

        print()
        print("=" * 72)
        print("TIKTOKEN ACCURACY vs REAL LLM USAGE (does 6a match reality?)")
        print("=" * 72)
        print(f"                        {'6a (tiktoken)':>16}  {'6b (real LLM)':>16}  {'error':>8}")
        print(f"  avg input tokens/q    {est_input_avg:>16.1f}  {real_input_avg:>16.1f}  {in_error:>7.1f}%")
        print(f"  avg output tokens/q   {est_output_avg:>16.1f}  {real_output_avg:>16.1f}  {out_error:>7.1f}%")
        print(f"  cost per query        ${est_cost_per_query:>14.6f}  ${real_cost_per_query:>14.6f}  {cost_error:>7.1f}%")

        print()
        if cost_error < 15:
            print(f"  [OK] 6a's tiktoken estimator is within {cost_error:.0f}% of real cost.")
            print("       You can trust 6a for quick cost projections without running real queries.")
        else:
            print(f"  [warn] 6a's estimate is {cost_error:.0f}% off from reality.")
            print("         Recalibrate or use 6b numbers for any production sizing.")

    # ── Monthly projections based on REAL numbers ──────────────────────────
    print()
    print("=" * 72)
    print("REAL MONTHLY COST PROJECTIONS")
    print("=" * 72)
    print(f"  {'queries/day':>12}   {'per day':>10}   {'per month':>12}   {'per year':>12}")
    for qpd in [10, 100, 1000, 10_000]:
        day = real_cost_per_query * qpd
        mo  = day * 30
        yr  = day * 365
        print(f"  {qpd:>12d}   ${day:>9.4f}   ${mo:>11.2f}   ${yr:>11.2f}")

    # ── Save the data ──────────────────────────────────────────────────────
    out_path = KIT_ROOT / "eval_results" / "exp6b_real_cost.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump({
            "n_queries": len(SAMPLE_QUERIES),
            "n_log_entries": len(entries),
            "total_llm_input_tokens":  total_groq_in,
            "total_llm_output_tokens": total_groq_out,
            "real_cost_total_usd":      round(real_cost_total, 6),
            "real_cost_per_query_usd":  round(real_cost_per_query, 6),
            "avg_wall_latency_ms":      round(statistics.mean(wall_times), 1),
            "pricing_input_per_M":      INPUT_PRICE_PER_M,
            "pricing_output_per_M":     OUTPUT_PRICE_PER_M,
            "projections_monthly": {
                "10_per_day":    round(real_cost_per_query * 10   * 30, 4),
                "100_per_day":   round(real_cost_per_query * 100  * 30, 4),
                "1000_per_day":  round(real_cost_per_query * 1000 * 30, 4),
                "10000_per_day": round(real_cost_per_query * 10000* 30, 4),
            },
            "by_call_type": {
                call: {
                    "count":              len(es),
                    "avg_llm_input":     round(statistics.mean([e["groq_prompt_tokens"]     for e in es]), 1),
                    "avg_llm_output":    round(statistics.mean([e["groq_completion_tokens"] for e in es]), 1),
                    "avg_tiktoken_input": round(statistics.mean([e["tiktoken_estimate"]       for e in es]), 1),
                }
                for call, es in by_call.items()
            },
        }, f, indent=2)
    print(f"\nSaved: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())