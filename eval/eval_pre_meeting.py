"""
eval_pre_meeting.py — Per-section faithfulness evaluation for /brief/pre-meeting.

For each row in the golden CSV, checks whether the named section in the brief
contains the expected content and key facts.

CSV columns:
    section             — e.g. "open_items", "outstanding_commitments"
    expected_content    — what the section should say (used for semantic sim)
    key_facts           — semicolon-separated facts that must appear
    gold_source         — (optional) filename expected to be cited

Target: section_coverage_avg >= 0.70 across golden rows.

Usage:
    python eval_pre_meeting.py --customer cascadia-inc
    python eval_pre_meeting.py --customer cascadia-inc --csv golden/meridian_brief_sections_golden.csv
"""

import argparse
import csv
import json
import datetime
import os
import re
import statistics
import time
from typing import Any, Dict, List, Optional

import requests
import numpy as np
from sentence_transformers import SentenceTransformer

PRE_MEETING_URL = os.getenv("PRE_MEETING_URL", "http://localhost:8000/brief/pre-meeting")
AUTH_URL        = PRE_MEETING_URL.rsplit("/brief", 1)[0] + "/auth/token"
API_KEY         = os.getenv("API_KEY", "")
EVAL_WORKSPACE  = os.getenv("EVAL_WORKSPACE", "eval-default")
EVAL_PASSKEY    = os.getenv("EVAL_PASSKEY", "eval-default-passkey")

# LLM-as-judge: Anthropic Claude Haiku used as a tiebreaker when cosine sim
# is in the ambiguous range (0.35–0.50). Set ANTHROPIC_API_KEY to enable.
# Requires: pip install anthropic
_ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
_LLM_JUDGE_ENABLED = bool(_ANTHROPIC_API_KEY)
_LLM_JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "claude-haiku-4-5-20251001")

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


def _llm_judge(section_text: str, expected: str, facts: List[str]) -> Optional[float]:
    """Call Claude Haiku to score whether section_text covers expected content.

    Returns a 0.0–1.0 score, or None if the judge is disabled or fails.
    Only invoked when sim is ambiguous (0.35 ≤ sim < 0.60) to save cost.
    """
    if not _LLM_JUDGE_ENABLED:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=_ANTHROPIC_API_KEY)
        facts_str = "\n".join(f"- {f}" for f in facts) if facts else "(none)"
        prompt = f"""You are evaluating whether a brief section adequately covers the expected content.

SECTION TEXT:
{section_text[:800]}

EXPECTED CONTENT:
{expected[:400]}

KEY FACTS THAT SHOULD BE PRESENT:
{facts_str}

Score from 0.0 to 1.0:
- 1.0 = section covers all expected content and key facts
- 0.7 = section covers most content with minor gaps
- 0.4 = section partially covers but misses significant facts
- 0.0 = section is completely wrong or empty

Return ONLY a JSON object: {{"score": 0.0}}"""

        message = client.messages.create(
            model=_LLM_JUDGE_MODEL,
            max_tokens=64,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = (message.content[0].text or "").strip()
        raw = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
        return float(json.loads(raw)["score"])
    except Exception as e:
        print(f"  [judge] error: {e}")
        return None


def _coverage(text: str, facts: List[str]) -> float:
    """Fraction of facts present in text by substring or sentence-level similarity."""
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
    """Convert a section's list items or string value to a flat text for scoring."""
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
            # Try common text fields in order of preference
            text = (
                item.get("title") or item.get("description") or item.get("what") or
                item.get("topic") or item.get("event") or item.get("ask") or
                item.get("directive") or item.get("content") or ""
            )
            extra = item.get("status") or item.get("date") or ""
            parts.append(f"{text} {extra}".strip())
        else:
            parts.append(str(item))
    return " ".join(p for p in parts if p)


def evaluate_pre_meeting(
    customer_id: str,
    csv_path: str,
    out_dir: Optional[str] = None,
):
    """Call /brief/pre-meeting once, then score each golden row against the returned brief."""
    print(f"Calling /brief/pre-meeting for customer '{customer_id}'...")
    t0 = time.perf_counter()
    try:
        r = requests.post(
            PRE_MEETING_URL,
            json={"customer_id": customer_id},
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
    judge_scores: List[Optional[float]] = []
    section_results: List[dict] = []

    print(f"Evaluating {len(rows)} golden assertions...")
    if _LLM_JUDGE_ENABLED:
        print(f"  LLM judge enabled (model={_LLM_JUDGE_MODEL}) — used in ambiguous range 0.35–0.50\n")
    else:
        print("  LLM judge disabled (set ANTHROPIC_API_KEY to enable)\n")

    for i, row in enumerate(rows, 1):
        section       = row["section"].strip()
        expected      = row.get("expected_content", "").strip()
        facts         = [x for x in row.get("key_facts", "").split(";") if x.strip()]

        section_text = _flatten_section(brief, section)
        judge_score: Optional[float] = None

        if not section_text:
            sim, cov = 0.0, 0.0
            status = "EMPTY"
        else:
            sim = _semantic_sim(section_text, expected) if expected else 0.0
            cov = _coverage(section_text, facts)

            # Three-signal scoring:
            # 1. sim >= 0.50 AND cov >= 0.50  → PASS (confident)
            # 2. sim < 0.35 OR section empty  → FAIL (confident)
            # 3. 0.35 <= sim < 0.50 (ambiguous): call LLM judge as tiebreaker
            if sim >= 0.50 and cov >= 0.50:
                status = "PASS"
            elif sim < 0.35:
                status = "FAIL"
            else:
                # Ambiguous zone — ask the judge
                judge_score = _llm_judge(section_text, expected, facts)
                if judge_score is not None:
                    status = "PASS" if judge_score >= 0.60 else "FAIL"
                    status += "_JUDGE"
                else:
                    status = "FAIL_NO_JUDGE"

        sims.append(sim)
        covs.append(cov)
        judge_scores.append(judge_score)
        section_results.append({
            "section":     section,
            "sim":         round(sim, 3),
            "coverage":    round(cov, 3),
            "judge_score": round(judge_score, 3) if judge_score is not None else None,
            "status":      status,
            "expected":    expected[:80],
        })

        judge_note = f"  judge={judge_score:.2f}" if judge_score is not None else ""
        print(f"[{i:02d}] {status:<10}  section={section:<28} sim={sim:.2f}  cov={cov:.2f}{judge_note}")
        if status.startswith("FAIL") or status == "EMPTY":
            print(f"       expected: {expected[:100]}")
        print()

    n = len(sims)
    if n == 0:
        print("No results.")
        return

    # Count passes: "PASS" (confident) + "PASS_JUDGE" (judge-confirmed)
    # "FAIL_NO_JUDGE" = judge was needed but errored/disabled — counted as FAIL
    pass_count = sum(1 for r in section_results if r["status"].startswith("PASS"))
    pass_rate = pass_count / n
    judge_used = [r for r in section_results if r.get("judge_score") is not None]
    judge_scores_valid = [r["judge_score"] for r in judge_used if r["judge_score"] is not None]

    out = {
        "run_metadata": {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "embedder_model": "sentence-transformers/all-MiniLM-L6-v2",
            "llm_judge_model": _LLM_JUDGE_MODEL if _LLM_JUDGE_ENABLED else None,
            "eval_workspace": EVAL_WORKSPACE,
            "api_url": PRE_MEETING_URL,
            "thresholds": {
                "pass_sim": 0.50,
                "pass_cov": 0.50,
                "fail_sim_below": 0.35,
                "judge_pass_score": 0.60,
                "sentence_sim_coverage": 0.60,
            },
        },
        "customer_id":            customer_id,
        "n_assertions":           n,
        "mean_semantic_sim":      round(statistics.mean(sims), 3),
        "mean_coverage":          round(statistics.mean(covs), 3),
        "pass_rate_sim50_cov50":  round(pass_rate, 3),
        "llm_judge_enabled":      _LLM_JUDGE_ENABLED,
        "llm_judge_calls":        len(judge_used),
        "mean_judge_score":       round(statistics.mean(judge_scores_valid), 3) if judge_scores_valid else None,
        "latency_ms":             round(elapsed_ms, 1),
        "section_results":        section_results,
        "stale_warnings":         brief.get("stale_warnings", []),
        "conflicts":              len(brief.get("conflicts", [])),
        "corpus_health_overall":  (brief.get("corpus_health") or {}).get("overall"),
    }

    save_dir = out_dir if out_dir else os.path.dirname(csv_path)
    out_path = os.path.join(save_dir, f"pre_meeting_eval_{customer_id}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("─── PRE-MEETING BRIEF EVAL SUMMARY ───")
    summary = {k: v for k, v in out.items() if k != "section_results"}
    print(json.dumps(summary, indent=2))
    if pass_rate >= 0.70:
        print(f"\nTarget met (>= 70% of sections passed)")
    else:
        print(f"\nTarget NOT met. {int(pass_rate * 100)}% passed (need 70%)")
    if judge_used:
        print(f"LLM judge broke {len(judge_used)} tie(s) with mean score {summary.get('mean_judge_score', 'n/a')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--customer", required=True,
                        help="Customer slug to evaluate, e.g. cascadia-inc")
    parser.add_argument("--csv",
                        default=os.path.join(os.path.dirname(__file__),
                                             "golden", "meridian_brief_sections_golden.csv"),
                        help="Path to golden sections CSV")
    parser.add_argument("--out-dir", default=None,
                        help="Directory to write output JSON (defaults to same dir as csv)")
    args = parser.parse_args()
    evaluate_pre_meeting(args.customer, args.csv, args.out_dir)
