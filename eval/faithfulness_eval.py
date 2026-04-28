"""
faithfulness_eval.py — Run the full /brief pipeline on a golden query set
and measure claim grounding against retrieved chunks.

Target: faithfulness_score >= 0.90 across the golden set.
This is the headline portfolio metric.

Usage:
    python faithfulness_eval.py
    python faithfulness_eval.py --csv eval_set_meridian.csv --user_id my_workspace_id
"""

import argparse
import csv
import json
import os
import statistics
import time
import requests
from typing import List

BRIEF_URL         = "http://localhost:8000/brief"
AUTH_URL          = BRIEF_URL.rsplit("/", 1)[0] + "/auth/token"
API_KEY           = os.getenv("API_KEY", "")
INTER_QUERY_SLEEP = float(os.getenv("INTER_QUERY_SLEEP", "15"))

# Tenant identity is JWT-only — the API used to honor a `?user_id=` query param,
# but that was an IDOR. The eval mints a token once and reuses it.
EVAL_WORKSPACE = os.getenv("EVAL_WORKSPACE", "eval-default")
EVAL_PASSKEY   = os.getenv("EVAL_PASSKEY", "eval-default-passkey")


def _mint_token(workspace: str, passkey: str) -> tuple:
    """Authenticate once and return (token, user_id) for all subsequent calls."""
    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    r = requests.post(
        AUTH_URL,
        json={"workspace": workspace, "passkey": passkey},
        headers=headers,
        timeout=10,
    )
    r.raise_for_status()
    data = r.json()
    return data["token"], data["user_id"]


_TOKEN, _USER_ID = _mint_token(EVAL_WORKSPACE, EVAL_PASSKEY)


def _headers() -> dict:
    h = {"Authorization": f"Bearer {_TOKEN}"}
    if API_KEY:
        h["X-API-Key"] = API_KEY
    return h


def call_brief(query: str, customer_id: str) -> dict:
    """Call the /brief endpoint and return the response JSON."""
    try:
        r = requests.post(
            BRIEF_URL,
            json={"query": query, "customer_id": customer_id},
            headers=_headers(),
            timeout=180,
        )
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}: {r.text[:200]}"}
        data = r.json()
        # Detect silent LLM failures: HTTP 200 but reason_node failed internally
        # (Groq rate limit after retries exhausted, or JSON parse error).
        # Without this, failed queries get faithfulness=0.0 and drag averages down.
        brief = data.get("brief", {})
        open_qs = brief.get("open_questions", [])
        if (not brief.get("issues") and not brief.get("risks")
                and not brief.get("talking_points")
                and any(isinstance(q, str)
                        and ("could not analyze" in q.lower()
                             or "could not parse analyst" in q.lower())
                        for q in open_qs)):
            msg = next((q for q in open_qs if isinstance(q, str)), "silent LLM failure")
            return {"error": f"silent_llm_failure: {msg[:150]}"}
        return data
    except Exception as e:
        return {"error": str(e)}


def evaluate_faithfulness(csv_path: str, customer_id: str = None, out_dir: str = None):
    # Default to the JWT-derived workspace id so the brief is logged under the
    # tenant we authenticated as. Callers can still override.
    if customer_id is None:
        customer_id = _USER_ID
    """
    For each row in the golden CSV, call /brief and record:
      - faithfulness_score  (from the API: grounded claims / total claims)
      - loop_count          (how many retrieval loops the graph ran)
      - has_issues          (brief returned at least one issue)
      - has_risks           (brief returned at least one risk)
      - latency_ms

    Aggregate: mean faithfulness, fraction reaching target (>= 0.90).
    """
    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    scores: List[float] = []
    loop_counts: List[int] = []
    lats: List[float] = []
    errors = 0

    print(f"Running faithfulness eval on {len(rows)} queries...\n")

    for i, row in enumerate(rows, 1):
        query = row.get("question", row.get("query", "")).strip()
        if not query:
            continue

        t0 = time.perf_counter()
        result = call_brief(query, customer_id)
        lat = (time.perf_counter() - t0) * 1000.0

        if "error" in result:
            errors += 1
            print(f"[{i}] ERROR: {result['error']}")
            print(f"     Q: {query}\n")
            if i < len(rows):
                time.sleep(INTER_QUERY_SLEEP)
            continue

        fs = result.get("faithfulness_score", 0.0)
        lc = result.get("loop_count", 0)
        brief = result.get("brief", {})
        has_issues = bool(brief.get("issues"))
        has_risks  = bool(brief.get("risks"))

        scores.append(fs)
        loop_counts.append(lc)
        lats.append(lat)

        status = "PASS" if fs >= 0.90 else "FAIL"
        print(
            f"[{i}] {status}  faithfulness={fs:.2f}  loops={lc}  "
            f"issues={len(brief.get('issues', []))}  risks={len(brief.get('risks', []))}  "
            f"lat={lat:.0f}ms"
        )
        print(f"     Q: {query}\n")

        if i < len(rows):
            time.sleep(INTER_QUERY_SLEEP)

    n = len(scores)
    if n == 0:
        print("No results.")
        return

    mean_fs   = statistics.mean(scores)
    pass_rate = sum(1 for s in scores if s >= 0.90) / n
    p50_lat   = statistics.median(lats) if lats else 0.0

    out = {
        "total_queries":        n + errors,
        "evaluated":            n,
        "errors":               errors,
        "mean_faithfulness":    round(mean_fs, 3),
        "pass_rate_at_0_90":    round(pass_rate, 3),
        "p50_latency_ms":       round(p50_lat, 1),
        "avg_loop_count":       round(statistics.mean(loop_counts), 2) if loop_counts else 0,
    }

    save_dir = out_dir if out_dir else os.path.dirname(csv_path)
    out_path = os.path.join(save_dir, "faithfulness_metrics.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("\n─── FAITHFULNESS EVAL SUMMARY ───")
    print(json.dumps(out, indent=2))
    if pass_rate >= 0.90:
        print("\nTarget met (>= 90% of queries at faithfulness >= 0.90)")
    else:
        print(f"\nTarget NOT met. {int(pass_rate * 100)}% reached 0.90 (need 90%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=os.path.join(os.path.dirname(__file__), "eval_set_meridian.csv"),
    )
    parser.add_argument(
        "--user_id",
        default=None,
        help="Customer/workspace ID to record on brief logs. Defaults to the "
             "user_id derived from the JWT minted via EVAL_WORKSPACE/EVAL_PASSKEY.",
    )
    args = parser.parse_args()
    evaluate_faithfulness(args.csv, customer_id=args.user_id)
