"""
exp10_lookup.py — evaluate the /query endpoint for answer quality,
answer_status distribution, and latency.

/query is the single-pass Q&A endpoint:
  - Adaptive query rewriter (focused queries skip decomposition)
  - One LLM call to answer; no completeness loop
  - Returns {answer, answer_status, citation, recency_flag, conflicts, missing_doc_types}

This experiment answers:
  - Does /query produce accurate answers on the eval set?
  - What fraction of queries return answer_status=ok vs not_found vs partial?
  - What is the p50/p95 latency?

Usage:
    python experiment_kit/experiments/exp10_lookup.py

Requires: GOOGLE_API_KEY, API_KEY, EVAL_WORKSPACE, EVAL_PASSKEY set in env.
          API running on localhost:8000 (or override with QUERY_URL env var).
"""
import csv
import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[2]
KIT_ROOT = REPO_ROOT / "experiment_kit"
sys.path.insert(0, str(REPO_ROOT / "eval"))

QUERY_URL = os.getenv("QUERY_URL", "http://localhost:8000/query")
AUTH_URL = QUERY_URL.rsplit("/", 1)[0] + "/auth/token"
API_KEY = os.getenv("API_KEY", "")
EVAL_WORKSPACE = os.getenv("EVAL_WORKSPACE", "eval-default")
EVAL_PASSKEY = os.getenv("EVAL_PASSKEY", "eval-default-passkey")
EVAL_CUSTOMER_ID = os.getenv("EVAL_CUSTOMER_ID", "eval-full")
INTER_QUERY_SLEEP = float(os.getenv("INTER_QUERY_SLEEP", "10"))

EVAL_CSV = KIT_ROOT / "eval_set.csv"
RESULTS_DIR = KIT_ROOT / "eval_results"

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMB_MODEL)
    return _embedder


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def semantic_sim(a: str, b: str) -> float:
    emb = _get_embedder()
    v = emb.encode([a, b], convert_to_numpy=True)
    return max(0.0, min(1.0, _cosine(v[0], v[1])))


def coverage(answer: str, facts: list) -> float:
    """Fraction of facts that appear in the answer (substring or sentence-level sim)."""
    if not facts:
        return 0.0
    import re
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
        if any(semantic_sim(s, f) >= 0.6 for s in sentences):
            hit += 1
    return hit / max(1, len(facts))


def recall_at_k(sources: list, gold: str, k: int = 5) -> float:
    if not gold:
        return 0.0
    return 1.0 if gold.lower() in [s.lower() for s in sources[:k]] else 0.0


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


def query(question: str, customer_id: str, token: str) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    r = requests.post(
        QUERY_URL,
        json={"question": question, "customer_id": customer_id},
        headers=headers,
        timeout=120,
    )
    if not r.ok:
        return {"error": f"HTTP {r.status_code}: {r.text[:200]}"}
    return r.json()


def main():
    print("=" * 72)
    print("EXPERIMENT 10 — /query QUALITY & LATENCY EVAL")
    print("=" * 72)
    print()
    print("Minting eval token ...")
    try:
        token = _mint_token()
    except Exception as e:
        print(f"[err] auth failed: {e}")
        sys.exit(1)
    customer_id = EVAL_CUSTOMER_ID
    print(f"  OK  (customer_id={customer_id})")
    print()

    rows = []
    with open(EVAL_CSV, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} questions from {EVAL_CSV.name}")
    print()

    sims, covs, lats = [], [], []
    recalls = []
    statuses: Counter = Counter()
    recency_flags: Counter = Counter()
    errors = 0
    row_results = []

    for i, row in enumerate(rows, 1):
        q        = row["question"].strip()
        ref      = row.get("reference_answer", "").strip()
        facts    = [x for x in row.get("key_facts", "").split(";") if x.strip()]
        gold_src = row.get("gold_source", row.get("source_filename", "")).strip()

        t0 = time.perf_counter()
        resp = query(q, customer_id, token)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if "error" in resp:
            errors += 1
            print(f"[{i:02d}] ERROR  {resp['error']}")
            print(f"       Q: {q[:80]}")
            print()
            if i < len(rows):
                time.sleep(INTER_QUERY_SLEEP)
            continue

        answer     = resp.get("answer") or ""
        ans_status = resp.get("answer_status", "not_found")
        recency    = resp.get("recency_flag")
        citation   = resp.get("citation") or {}
        sources    = [citation["document"]] if citation.get("document") else []

        sim = semantic_sim(answer, ref) if ref else 0.0
        cov = coverage(answer, facts)
        r5  = recall_at_k(sources, gold_src) if gold_src else None

        sims.append(sim)
        covs.append(cov)
        lats.append(elapsed_ms)
        statuses[ans_status] += 1
        if recency:
            recency_flags[recency] += 1
        if r5 is not None:
            recalls.append(r5)

        print(
            f"[{i:02d}] status={ans_status:<14} sim={sim:.2f}  cov={cov:.2f}  "
            f"lat={elapsed_ms:.0f}ms"
        )
        print(f"       recency={recency}  source={citation.get('document', 'none')}")
        print(f"       Q: {q[:80]}")
        print()

        row_results.append({
            "question":      q,
            "answer_status": ans_status,
            "semantic_sim":  round(sim, 3),
            "coverage":      round(cov, 3),
            "latency_ms":    round(elapsed_ms, 1),
            "recency_flag":  recency,
            "recall_at_5":   round(r5, 3) if r5 is not None else None,
            "source":        citation.get("document"),
        })

        if i < len(rows):
            time.sleep(INTER_QUERY_SLEEP)

    # ── Aggregate metrics ────────────────────────────────────────────────────
    n = len(sims)
    if n == 0:
        print("[err] all queries failed — nothing to aggregate")
        sys.exit(1)

    sorted_lats = sorted(lats)
    p50 = statistics.median(sorted_lats)
    p95 = sorted_lats[min(int(len(sorted_lats) * 0.95), len(sorted_lats) - 1)]
    p99 = sorted_lats[min(int(len(sorted_lats) * 0.99), len(sorted_lats) - 1)]

    metrics = {
        "n_queries":               n,
        "error_count":             errors,
        "error_rate":              round(errors / max(1, n + errors), 3),
        "semantic_similarity_avg": round(sum(sims) / n, 3),
        "key_facts_coverage_avg":  round(sum(covs) / n, 3),
        "recall_at_5":             round(sum(recalls) / len(recalls), 3) if recalls else None,
        "p50_latency_ms":          round(p50, 1),
        "p95_latency_ms":          round(p95, 1),
        "p99_latency_ms":          round(p99, 1),
        "answer_status_dist":      dict(statuses),
        "recency_flag_dist":       dict(recency_flags),
        "rows":                    row_results,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "exp10_query_results.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"Results written to {out_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("SUMMARY — /query quality & latency")
    print("=" * 72)
    print(f"  n_queries          : {n}  (errors: {errors})")
    print(f"  semantic_sim avg   : {metrics['semantic_similarity_avg']}")
    print(f"  coverage avg       : {metrics['key_facts_coverage_avg']}")
    print(f"  recall@5           : {metrics['recall_at_5']}")
    print(f"  p50 latency        : {p50:.0f}ms")
    print(f"  p95 latency        : {p95:.0f}ms")
    print(f"  p99 latency        : {p99:.0f}ms")
    print()
    print("  answer_status distribution:")
    total = sum(statuses.values())
    for st, cnt in sorted(statuses.items(), key=lambda x: -x[1]):
        print(f"    {st:<20} {cnt:>3}  ({cnt/total*100:.0f}%)")
    print()
    print("Reading these results:")
    print("  semantic_sim / coverage: answer quality vs ground truth.")
    print("    <0.5 sim or <0.4 coverage suggests retrieval gap.")
    print()
    print("  answer_status=not_found: chunks didn't contain the answer.")
    print("    High not_found rate (~>20%) means retrieval needs improvement")
    print("    or the question is outside the uploaded corpus.")
    print()
    print("  answer_status=partial: answered some parts of a multi-part question.")
    print("    Use /brief/pre-meeting for broader multi-section questions.")
    print()
    print("  latency: p95 >> p50 on first run = cross-encoder cold start (lazy load).")


if __name__ == "__main__":
    main()
