"""
eval_exec_1on1.py — Evaluates /brief/exec-1on1 against a golden CSV.

For a given customer + person, calls the exec-1on1 endpoint and checks
whether each expected section claim appears in the brief.

CSV columns:
    person_name         — display name of the person (informational)
    section             — e.g. "recent_signals", "open_asks", "role_and_tenure"
    expected_content    — what should appear in that section
    key_facts           — semicolon-separated facts that must appear

Target: section_coverage_avg >= 0.60 for key person-specific signals/asks.

Usage:
    python eval_exec_1on1.py --customer cascadia-inc --person-id 42
    python eval_exec_1on1.py --customer cascadia-inc --person-id 42 \\
        --csv golden/meridian_exec_1on1_golden.csv
"""

import argparse
import csv
import datetime
import json
import os
import re
import statistics
import time
from typing import Any, Dict, List, Optional

import requests
import numpy as np
from sentence_transformers import SentenceTransformer

EXEC_1ON1_URL = os.getenv("EXEC_1ON1_URL", "http://localhost:8000/brief/exec-1on1")
AUTH_URL      = EXEC_1ON1_URL.rsplit("/brief", 1)[0] + "/auth/token"
API_KEY       = os.getenv("API_KEY", "")
EVAL_WORKSPACE = os.getenv("EVAL_WORKSPACE", "eval-default")
EVAL_PASSKEY  = os.getenv("EVAL_PASSKEY", "eval-default-passkey")

_TOKEN: Optional[str] = None
_embedder: Optional[SentenceTransformer] = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedder


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def _semantic_sim(a: str, b: str) -> float:
    emb = _get_embedder()
    v = emb.encode([a, b], convert_to_numpy=True)
    return max(0.0, min(1.0, _cosine(v[0], v[1])))


def _coverage(text: str, facts: List[str]) -> float:
    if not facts:
        return 0.0
    text_lower = text.strip().lower()
    sentences = [s.strip() for s in re.split(r"[.!?]\s+", text) if len(s.strip()) > 15]
    hit = 0
    for f in facts:
        f = f.strip()
        if not f:
            continue
        if f.lower() in text_lower:
            hit += 1
            continue
        if any(_semantic_sim(s, f) >= 0.6 for s in sentences):
            hit += 1
    return hit / max(1, len(facts))


def _mint_token() -> str:
    global _TOKEN
    if _TOKEN is not None:
        return _TOKEN
    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    r = requests.post(
        AUTH_URL,
        json={"workspace": EVAL_WORKSPACE, "passkey": EVAL_PASSKEY},
        headers=headers,
        timeout=10,
    )
    r.raise_for_status()
    _TOKEN = r.json()["token"]
    return _TOKEN


def _headers() -> dict:
    h = {"Authorization": f"Bearer {_mint_token()}"}
    if API_KEY:
        h["X-API-Key"] = API_KEY
    return h


def _flatten_section(brief: Dict[str, Any], section: str) -> str:
    """Flatten a section to text for scoring."""
    value = brief.get(section)
    if not value:
        return ""
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return str(value)
    parts = []
    for item in value:
        if isinstance(item, dict):
            text = (
                item.get("event") or item.get("ask") or item.get("content") or
                item.get("directive") or ""
            )
            extra = item.get("date") or item.get("status") or ""
            parts.append(f"{text} {extra}".strip())
        else:
            parts.append(str(item))
    return " ".join(p for p in parts if p)


def evaluate_exec_1on1(
    customer_id: str,
    person_id: str,
    csv_path: str,
    out_dir: Optional[str] = None,
):
    """Call /brief/exec-1on1, then score each golden row against the returned brief."""
    print(f"Calling /brief/exec-1on1 for customer='{customer_id}' person='{person_id}'...")
    t0 = time.perf_counter()
    try:
        r = requests.post(
            EXEC_1ON1_URL,
            json={"customer_id": customer_id, "person_id": person_id},
            headers=_headers(),
            timeout=180,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if not r.ok:
            print(f"ERROR: HTTP {r.status_code}: {r.text[:200]}")
            return
        brief = r.json()
    except Exception as e:
        print(f"ERROR: {e}")
        return

    print(f"  Got brief in {elapsed_ms:.0f}ms\n")

    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    sims: List[float] = []
    covs: List[float] = []
    results: List[dict] = []

    print(f"Evaluating {len(rows)} golden assertions...\n")

    for i, row in enumerate(rows, 1):
        person_name   = row.get("person_name", "").strip()
        section       = row["section"].strip()
        expected      = row.get("expected_content", "").strip()
        facts         = [x for x in row.get("key_facts", "").split(";") if x.strip()]

        section_text = _flatten_section(brief, section)
        if not section_text:
            sim, cov = 0.0, 0.0
            status = "EMPTY"
        else:
            sim = _semantic_sim(section_text, expected) if expected else 0.0
            cov = _coverage(section_text, facts)
            status = "PASS" if sim >= 0.50 and cov >= 0.50 else "FAIL"

        sims.append(sim)
        covs.append(cov)
        results.append({
            "person":   person_name,
            "section":  section,
            "sim":      round(sim, 3),
            "coverage": round(cov, 3),
            "status":   status,
            "expected": expected[:80],
        })

        print(f"[{i:02d}] {status:<5}  [{person_name}] {section:<22} sim={sim:.2f}  cov={cov:.2f}")
        if status in ("FAIL", "EMPTY"):
            print(f"       expected: {expected[:100]}")
        print()

    n = len(sims)
    if n == 0:
        print("No results.")
        return

    pass_rate = sum(1 for r in results if r["status"] == "PASS") / n
    out = {
        "run_metadata": {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "embedder_model": "sentence-transformers/all-MiniLM-L6-v2",
            "eval_workspace": EVAL_WORKSPACE,
            "api_url": EXEC_1ON1_URL,
            "thresholds": {
                "pass_sim": 0.50,
                "pass_cov": 0.50,
                "sentence_sim_coverage": 0.60,
            },
        },
        "customer_id":            customer_id,
        "person_id":              person_id,
        "n_assertions":           n,
        "mean_semantic_sim":      round(statistics.mean(sims), 3),
        "mean_coverage":          round(statistics.mean(covs), 3),
        "pass_rate_sim50_cov50":  round(pass_rate, 3),
        "latency_ms":             round(elapsed_ms, 1),
        "stale_warnings":         brief.get("stale_warnings", []),
        "conflicts":              len(brief.get("conflicts", [])),
        "section_results":        results,
    }

    save_dir = out_dir if out_dir else os.path.dirname(csv_path)
    out_path = os.path.join(save_dir, f"exec_1on1_eval_{customer_id}_{person_id}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("─── EXEC 1:1 BRIEF EVAL SUMMARY ───")
    print(json.dumps({k: v for k, v in out.items() if k != "section_results"}, indent=2))
    if pass_rate >= 0.60:
        print(f"\nTarget met (>= 60% of assertions passed sim>=0.50 and cov>=0.50)")
    else:
        print(f"\nTarget NOT met. {int(pass_rate * 100)}% passed (need 60%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--customer", required=True,
                        help="Customer slug, e.g. cascadia-inc")
    parser.add_argument("--person-id", required=True,
                        help="Person ID (integer) from POST /customers/{slug}/people")
    parser.add_argument("--csv",
                        default=os.path.join(os.path.dirname(__file__),
                                             "golden", "meridian_exec_1on1_golden.csv"),
                        help="Path to golden exec 1:1 CSV")
    parser.add_argument("--out-dir", default=None,
                        help="Directory to write output JSON")
    args = parser.parse_args()
    evaluate_exec_1on1(args.customer, args.person_id, args.csv, args.out_dir)
