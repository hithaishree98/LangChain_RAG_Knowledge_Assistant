"""
Experiment 4 — Tiered hallucination detection

Tests three layers:
  Layer 1: Regex detector      — catches atomic fact lies (dates, amounts, versions)
  Layer 2: Classifier          — routes claims to the right verifier
  Layer 3: LLM-as-judge        — catches relational/named-entity hallucinations

Five cases, each targeting what a different layer should catch.

Usage:
    python experiment_kit/experiments/exp4_hallucination.py

Requires: GOOGLE_API_KEY set. The LLM judge actually calls Gemini for
Cases C-E, so each case takes ~1-3s. Cases A and B skip the LLM and run
instantly.
"""
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "api"))

from langchain_core.documents import Document
from output.brief_generator import generate_brief


# ── Real chunks from the sample docs — these are what would be "retrieved" ──
faithful_chunks = [
    Document(
        page_content=(
            "Base platform fee: $28,000 per month (Enterprise tier). "
            "Includes 5 named admin users, 50 analyst seats, unlimited read-only "
            "access, and up to 200 million events ingested per month."
        ),
        metadata={"source": "Meridian_SOW_v2.pdf", "chunk_id": "P1"},
    ),
    Document(
        page_content=(
            "Status update for Sarah Chen: fix is in staging; validation run "
            "scheduled for 2024-09-20 with Mike Rodriguez's team observing. "
            "Targeting production rollout on 2024-09-25 per the call commitment. "
            "David Park committed to daily status updates until ship."
        ),
        metadata={"source": "TICK-4521.json", "chunk_id": "P2"},
    ),
    Document(
        page_content=(
            "Primary region: us-east-2 (Ohio). Secondary region for disaster "
            "recovery: us-west-2. The Orion knowledge retrieval subsystem is "
            "configured with k = 2 for Meridian's tenant."
        ),
        metadata={"source": "Meridian_SOW_v2.pdf", "chunk_id": "P3"},
    ),
]


# ── Five test cases — each targets a different defense layer ───────────────

# Case A: all claims are faithful — atomic facts and relationships all match
case_A = {
    "issues": [
        {"claim": "The login latency hotfix is targeting production rollout on 2024-09-25.",
         "chunk_id": "P2"},
    ],
    "risks": [],
    "open_questions": [],
    "talking_points": [
        {"point": "Meridian pays $28,000 per month for the Enterprise tier.",
         "chunk_id": "P1"},
    ],
}

# Case B: atomic fact lies — regex should catch $50,000 and 2024-09-30 and v9.9.9
case_B = {
    "issues": [
        {"claim": "The login latency hotfix is targeting production rollout on 2024-09-30.",
         "chunk_id": "P2"},
    ],
    "risks": [
        {"claim": "Meridian is paying $50,000 per month and the P0 SLA is 8 hours.",
         "chunk_id": "P1"},
    ],
    "open_questions": [],
    "talking_points": [
        {"point": "The Salesforce connector v9.9.9 supports 100,000 records per hour.",
         "chunk_id": "P1"},
    ],
}

# Case C: correct numbers, WRONG name — Sarah Park (not Chen) said this.
# Regex sees $28,000 matches → classifier routes to judge because "said" is relational.
case_C = {
    "issues": [],
    "risks": [],
    "open_questions": [],
    "talking_points": [
        {"point": "Sarah Park said the $28,000 monthly fee was too expensive.",
         "chunk_id": "P1"},
    ],
}

# Case D: fabricated company name — no regex patterns match at all.
# Classifier sees no atomic facts and no relational verbs → still routes to judge
# because there's nothing regex-verifiable to anchor it.
case_D = {
    "issues": [
        {"claim": "Meridian is planning to migrate their analytics workload to Databricks.",
         "chunk_id": "P1"},
    ],
    "risks": [],
    "open_questions": [],
    "talking_points": [],
}

# Case E: pure relational invention — David blocked something that didn't happen.
# Classifier routes to judge because "blocked" is a relational verb.
case_E = {
    "issues": [
        {"claim": "David Park blocked the hotfix deployment due to unresolved security concerns.",
         "chunk_id": "P2"},
    ],
    "risks": [],
    "open_questions": [],
    "talking_points": [],
}


def build_state(reasoning):
    return {
        "customer_id": "demo",
        "original_query": "test",
        "sub_queries": [],
        "retrieved_chunks": faithful_chunks,
        "parent_chunks": faithful_chunks,
        "reasoning_output": reasoning,
        "iteration_count": 1,
        "is_sufficient": True,
        "brief": None,
        "information_gaps": [],
        "audit_trail": [],
    }


def print_brief_summary(case_name, brief):
    stats = brief.get("verification_stats", {})
    print(f"\n--- CASE {case_name} ---")
    print(f"  faithfulness_score  : {brief['faithfulness_score']}")
    print(f"  suspicious_facts    : {brief['suspicious_facts']}")
    sc = brief.get("suspicious_claims", [])
    print(f"  suspicious_claims   : {len(sc)} flagged")
    for s in sc:
        print(f"    [{s['caught_by']}] {s['claim'][:80]}")
        print(f"        reason: {s['reason'][:100]}")
    print(f"  verification_stats  : total={stats.get('claims_total', 0)}  "
          f"regex_verified={stats.get('verified_by_regex', 0)}  "
          f"regex_flagged={stats.get('flagged_by_regex', 0)}  "
          f"sent_to_judge={stats.get('sent_to_llm_judge', 0)}")


def run():
    print("=" * 72)
    print("EXPERIMENT 4 — TIERED HALLUCINATION DETECTION")
    print("=" * 72)
    print("Testing 3 layers: Regex → Classifier → LLM Judge")
    print()

    results = {}
    for name, reasoning in [("A", case_A), ("B", case_B),
                             ("C", case_C), ("D", case_D), ("E", case_E)]:
        brief = generate_brief(build_state(reasoning))
        results[name] = brief
        print_brief_summary(name, brief)

    # ── Assertions — what each case should demonstrate ──────────────────────
    print()
    print("=" * 72)
    print("ASSERTIONS")
    print("=" * 72)

    checks = []

    # --- Case A: everything is faithful, minimal flagging ---
    a = results["A"]
    checks.append(("A: faithful case produces zero suspicious_claims",
                   f"len={len(a.get('suspicious_claims', []))}",
                   len(a.get("suspicious_claims", [])) == 0))
    checks.append(("A: faithfulness_score is high (>=0.8)",
                   f"{a['faithfulness_score']}",
                   a["faithfulness_score"] >= 0.8))

    # --- Case B: regex catches atomic lies ---
    b = results["B"]
    b_caught_by = {s["caught_by"] for s in b.get("suspicious_claims", [])}
    checks.append(("B: regex layer flagged at least one claim",
                   f"caught_by={b_caught_by}",
                   "regex" in b_caught_by))
    checks.append(("B: suspicious_facts contains atomic lies",
                   f"{b['suspicious_facts']}",
                   any("50,000" in s or "$50,000" in s for s in b["suspicious_facts"])))

    # --- Case C: LLM judge catches the wrong name ---
    c = results["C"]
    c_caught_by = {s["caught_by"] for s in c.get("suspicious_claims", [])}
    checks.append(("C: wrong name routed to LLM judge",
                   f"caught_by={c_caught_by}, "
                   f"sent_to_judge={c.get('verification_stats', {}).get('sent_to_llm_judge', 0)}",
                   c.get("verification_stats", {}).get("sent_to_llm_judge", 0) >= 1))
    checks.append(("C: LLM judge flagged the wrong-name claim",
                   f"claims={[s['claim'][:40] for s in c.get('suspicious_claims', [])]}",
                   "llm_judge" in c_caught_by))

    # --- Case D: fabricated company caught by judge ---
    d = results["D"]
    d_caught_by = {s["caught_by"] for s in d.get("suspicious_claims", [])}
    checks.append(("D: fabricated company caught by LLM judge",
                   f"caught_by={d_caught_by}",
                   "llm_judge" in d_caught_by))

    # --- Case E: pure relational invention caught by judge ---
    e = results["E"]
    e_caught_by = {s["caught_by"] for s in e.get("suspicious_claims", [])}
    checks.append(("E: relational invention routed to judge",
                   f"sent_to_judge={e.get('verification_stats', {}).get('sent_to_llm_judge', 0)}",
                   e.get("verification_stats", {}).get("sent_to_llm_judge", 0) >= 1))
    checks.append(("E: LLM judge flagged the fake relational claim",
                   f"caught_by={e_caught_by}",
                   "llm_judge" in e_caught_by))

    passed = 0
    for name, detail, ok in checks:
        marker = "PASS" if ok else "FAIL"
        print(f"  [{marker}] {name}")
        print(f"         {detail}")
        if ok:
            passed += 1

    print()
    print(f"  {passed}/{len(checks)} assertions passed")
    print()

    # ── Honest summary ──────────────────────────────────────────────────────
    print("=" * 72)
    print("WHAT THIS PROVES")
    print("=" * 72)
    print("  - Layer 1 (regex)    : catches atomic-fact lies (dates, amounts, versions)")
    print("                          without an LLM call. Fast and cheap.")
    print("  - Layer 2 (classifier): correctly routed claims with relational verbs or")
    print("                          no atomic facts to the LLM judge — no waste.")
    print("  - Layer 3 (LLM judge): catches hallucinations regex cannot see —")
    print("                          wrong names, fabricated entities, fake relations.")
    print()
    print("  Known residual risk: an LLM judge that agrees with another LLM's lie")
    print("  (correlated errors) or subtle paraphrases both may miss. No single-model")
    print("  defense is perfect. A future version would use a DIFFERENT model as judge.")
    print()

    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    sys.exit(run())