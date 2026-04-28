"""
Experiment 2 — Retrieval ablation

Reuses the eval_full workspace from exp1. Tests three retrieval modes:
  dense        — embedding search only
  dense_bm25   — dense + BM25 via RRF, no reranker
  full         — dense + BM25 + cross-encoder reranker + parent fetch

Usage:
    python experiment_kit/experiments/exp2_retrieval.py

Requires: exp1 completed first (eval_full workspace must exist).
"""
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
KIT_ROOT  = REPO_ROOT / "experiment_kit"

sys.path.insert(0, str(REPO_ROOT / "eval"))
sys.path.insert(0, str(KIT_ROOT / "experiments"))

from api_utils import start_api, stop_api, assert_workspace_ready

EVAL_RESULTS = KIT_ROOT / "eval_results"


# ── Helpers ───────────────────────────────────────────────────────────────────

def run_eval(user_id: str, config_label: str):
    import eval_simple as E
    orig_ask = E.ask
    E.ask = lambda q, uid=None, _u=user_id: orig_ask(q, user_id=_u)
    EVAL_RESULTS.mkdir(exist_ok=True)
    try:
        E.evaluate(str(KIT_ROOT / "eval_set.csv"), chunking_config=config_label,
                   out_dir=str(EVAL_RESULTS))
    finally:
        E.ask = orig_ask


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("EXPERIMENT 2 — RETRIEVAL ABLATION")
    print("=" * 72)
    print("Reuses eval_full workspace from exp1 (index unchanged across modes).")

    for mode in ("dense", "dense_bm25", "full"):
        label = f"ret_{mode}"
        print(f"\n── [{mode}] ──")
        proc = start_api({"CHUNKING_MODE": "full", "RETRIEVAL_MODE": mode})
        try:
            assert_workspace_ready("eval_full")
            print(f"  [eval] running eval against user_id=eval_full")
            run_eval("eval_full", label)
        finally:
            stop_api(proc)
            time.sleep(2)

    print("\n=== Results ===")
    for mode in ("dense", "dense_bm25", "full"):
        p = EVAL_RESULTS / f"metrics_open_simple_ret_{mode}.json"
        if p.exists():
            data = json.loads(p.read_text())
            print(f"  {p.name}:")
            print(f"    semantic_sim={data.get('semantic_similarity_avg')}  "
                  f"coverage={data.get('key_facts_coverage_avg')}  "
                  f"faithfulness={data.get('mean_faithfulness_score')}  "
                  f"avg_loops={data.get('avg_loop_count')}  "
                  f"recall@5={data.get('recall_at_5')}  "
                  f"mrr={data.get('mrr')}  "
                  f"p50={data.get('p50_latency_ms')}ms")
        else:
            print(f"  {p.name}: MISSING")


if __name__ == "__main__":
    main()
