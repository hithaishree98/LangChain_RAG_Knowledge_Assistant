"""
Experiment 3 — Workflow loop vs single pass

Runs faithfulness_eval twice against eval_full:
  single_pass — completeness node always returns sufficient (loop disabled)
  loop        — default behavior (graph may loop up to 3 times)

Measures whether the loop actually improves faithfulness or just burns tokens.

Usage:
    python experiment_kit/experiments/exp3_workflow_loop.py

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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("EXPERIMENT 3 — WORKFLOW LOOP ABLATION")
    print("=" * 72)

    import faithfulness_eval as FE

    EVAL_RESULTS.mkdir(exist_ok=True)

    for mode in ("single_pass", "loop"):
        out_path = EVAL_RESULTS / f"faithfulness_metrics_wf_{mode}.json"
        print(f"\n── [{mode}] ──")
        proc = start_api({
            "CHUNKING_MODE": "full",
            "RETRIEVAL_MODE": "full",
            "WORKFLOW_MODE": mode,
        })
        try:
            assert_workspace_ready("eval_full")
            print(f"  [eval] running faithfulness eval against eval_full")
            FE.evaluate_faithfulness(
                str(KIT_ROOT / "eval_set.csv"),
                customer_id="eval_full",
                out_dir=str(EVAL_RESULTS),
            )
            # faithfulness_eval now writes directly into eval_results/
            # Use replace() not rename() — on Windows rename fails if target
            # exists, which blocks re-runs after a prior exp3 attempt.
            src = EVAL_RESULTS / "faithfulness_metrics.json"
            if src.exists():
                src.replace(out_path)
                print(f"  → saved {out_path.name}")
            else:
                print(f"  [warn] expected output file not found at {src}")
        finally:
            stop_api(proc)
            time.sleep(2)

    print("\n=== Results ===")
    for mode in ("single_pass", "loop"):
        p = EVAL_RESULTS / f"faithfulness_metrics_wf_{mode}.json"
        if p.exists():
            data = json.loads(p.read_text())
            print(f"  {p.name}:")
            print(f"    mean_faithfulness={data.get('mean_faithfulness')}  "
                  f"pass_rate={data.get('pass_rate_at_0_90')}  "
                  f"avg_loops={data.get('avg_loop_count')}  "
                  f"p50={data.get('p50_latency_ms')}ms")
        else:
            print(f"  {p.name}: MISSING")

    print()
    print("Key question: does 'loop' show higher mean_faithfulness than 'single_pass'?")
    print("If avg_loop_count is ~1.0 in loop mode, the loop never fired — it's dead weight.")


if __name__ == "__main__":
    main()
