"""
faithfulness_eval.py — Run the /query pipeline on a golden query set and
measure answer quality (semantic similarity + fact coverage) as a faithfulness
proxy.

Target: semantic_similarity_avg >= 0.70 and key_facts_coverage_avg >= 0.60.

Usage:
    python faithfulness_eval.py
    python faithfulness_eval.py --csv eval_set_meridian.csv --customer_id my_customer_slug
"""

import argparse
import csv
import datetime
import json
import os
import statistics
import time
import requests
from typing import List

QUERY_URL         = os.getenv("QUERY_URL", "http://localhost:8000/query")
AUTH_URL          = QUERY_URL.rsplit("/", 1)[0] + "/auth/token"
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


# Lazy token: prevents importing this module from any context (notebooks,
# tests, helper scripts) from hitting the network and crashing if the API
# is offline. Mint on first use and cache.
_TOKEN = None
_USER_ID = None


def _ensure_token():
    global _TOKEN, _USER_ID
    if _TOKEN is None:
        _TOKEN, _USER_ID = _mint_token(EVAL_WORKSPACE, EVAL_PASSKEY)
    return _TOKEN, _USER_ID


def _headers() -> dict:
    token, _ = _ensure_token()
    h = {"Authorization": f"Bearer {token}"}
    if API_KEY:
        h["X-API-Key"] = API_KEY
    return h


def call_query(question: str, customer_id: str) -> dict:
    """Call the /query endpoint and return the response JSON."""
    try:
        r = requests.post(
            QUERY_URL,
            json={"question": question, "customer_id": customer_id},
            headers=_headers(),
            timeout=180,
        )
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}: {r.text[:200]}"}
        data = r.json()
        if not data.get("answer") or data.get("answer_status") == "not_found":
            return {"error": f"answer_status={data.get('answer_status', 'not_found')}"}
        return data
    except Exception as e:
        return {"error": str(e)}


def evaluate_faithfulness(csv_path: str, customer_id: str = None, out_dir: str = None):
    """For each row in the golden CSV, call /query and record:
      - semantic_similarity  (answer vs. reference_answer)
      - key_facts_coverage   (fraction of key_facts present in the answer)
      - answer_status        (ok / partial / not_found)
      - latency_ms

    Aggregate: mean similarity, mean coverage, fraction reaching targets
    (similarity >= 0.70 and coverage >= 0.60).
    """
    # Default to the JWT-derived workspace id so queries are logged under the
    # tenant we authenticated as. Callers can still override.
    if customer_id is None:
        _, customer_id = _ensure_token()

    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        _embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        def _semantic_sim(a: str, b: str) -> float:
            v = _embedder.encode([a, b], convert_to_numpy=True)
            a_n = v[0] / (np.linalg.norm(v[0]) + 1e-8)
            b_n = v[1] / (np.linalg.norm(v[1]) + 1e-8)
            return float(max(0.0, min(1.0, np.dot(a_n, b_n))))
    except ImportError:
        def _semantic_sim(a: str, b: str) -> float:
            return 0.0

    import re

    def _coverage(answer: str, facts: list) -> float:
        if not facts:
            return 0.0
        ans_lower = answer.strip().lower()
        sentences = [s.strip() for s in re.split(r"[.!?]\s+", answer) if len(s.strip()) > 15]
        hit = 0
        for f in facts:
            f = f.strip()
            if not f:
                continue
            if f.lower() in ans_lower:
                hit += 1
                continue
            if any(_semantic_sim(s, f) >= 0.6 for s in sentences):
                hit += 1
        return hit / max(1, len(facts))

    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    sims: List[float] = []
    covs: List[float] = []
    lats: List[float] = []
    loop_counts: List[int] = []
    statuses: dict = {}
    errors = 0

    print(f"Running answer quality eval on {len(rows)} queries...\n")

    for i, row in enumerate(rows, 1):
        query = row.get("question", row.get("query", "")).strip()
        ref   = row.get("reference_answer", "").strip()
        facts = [x for x in row.get("key_facts", "").split(";") if x.strip()]
        if not query:
            continue

        t0 = time.perf_counter()
        result = call_query(query, customer_id)
        lat = (time.perf_counter() - t0) * 1000.0

        if "error" in result:
            errors += 1
            print(f"[{i}] ERROR: {result['error']}")
            print(f"     Q: {query}\n")
            if i < len(rows):
                time.sleep(INTER_QUERY_SLEEP)
            continue

        answer = result.get("answer", "")
        ans_status = result.get("answer_status", "ok")
        statuses[ans_status] = statuses.get(ans_status, 0) + 1
        loop_counts.append(result.get("loop_count", 1))

        sim = _semantic_sim(answer, ref) if ref else 0.0
        cov = _coverage(answer, facts)

        sims.append(sim)
        covs.append(cov)
        lats.append(lat)

        status_label = "PASS" if sim >= 0.70 and cov >= 0.60 else "FAIL"
        print(
            f"[{i}] {status_label}  sim={sim:.2f}  cov={cov:.2f}  "
            f"status={ans_status}  lat={lat:.0f}ms"
        )
        print(f"     Q: {query}\n")

        if i < len(rows):
            time.sleep(INTER_QUERY_SLEEP)

    n = len(sims)
    if n == 0:
        print("No results.")
        return

    mean_sim = statistics.mean(sims)
    mean_cov = statistics.mean(covs)
    pass_rate = sum(1 for s, c in zip(sims, covs) if s >= 0.70 and c >= 0.60) / n
    p50_lat   = statistics.median(lats) if lats else 0.0
    avg_loop_count = statistics.mean(loop_counts) if loop_counts else 0.0

    out = {
        "run_metadata": {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "embedder_model": "sentence-transformers/all-MiniLM-L6-v2",
            "eval_workspace": EVAL_WORKSPACE,
            "customer_id": customer_id,
            "query_url": QUERY_URL,
            "thresholds": {
                "pass_sim": 0.70,
                "pass_cov": 0.60,
                "pass_rate_target": 0.90,
                "sentence_sim_coverage": 0.60,
            },
        },
        "total_queries":            n + errors,
        "evaluated":                n,
        "errors":                   errors,
        "mean_semantic_similarity": round(mean_sim, 3),
        "mean_coverage":            round(mean_cov, 3),
        "pass_rate_sim70_cov60":    round(pass_rate, 3),
        "avg_loop_count":           round(avg_loop_count, 2),
        "p50_latency_ms":           round(p50_lat, 1),
        "answer_status_dist":       statuses,
    }

    save_dir = out_dir if out_dir else os.path.dirname(csv_path)
    out_path = os.path.join(save_dir, "faithfulness_metrics.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("\n─── ANSWER QUALITY EVAL SUMMARY ───")
    print(json.dumps(out, indent=2))
    if pass_rate >= 0.90:
        print("\nTarget met (>= 90% of queries passed sim>=0.70 and cov>=0.60)")
    else:
        print(f"\nTarget NOT met. {int(pass_rate * 100)}% passed (need 90%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=os.path.join(os.path.dirname(__file__), "eval_set_meridian.csv"),
    )
    parser.add_argument(
        "--customer_id",
        default=None,
        help="Customer slug to query against. Defaults to the "
             "user_id derived from the JWT minted via EVAL_WORKSPACE/EVAL_PASSKEY.",
    )
    args = parser.parse_args()
    evaluate_faithfulness(args.csv, customer_id=args.customer_id)
