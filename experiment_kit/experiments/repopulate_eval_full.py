"""
Repopulate the eval_full workspace with vanilla (non-contextual) ingestion.

Use case: after a failed or partial exp1 run leaves eval_full in a mixed or
under-populated state. Faster than re-running all of exp1 because it skips
the eval phase and only repopulates the one workspace.

Usage:
    python experiment_kit/experiments/repopulate_eval_full.py

Forces CONTEXTUAL_RETRIEVAL=0 in the API subprocess regardless of the parent
shell's env, so this script always produces a clean vanilla-ingested workspace.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))

from api_utils import start_api, stop_api
import bootstrap_upload as bu


def main():
    print("=" * 72)
    print("REPOPULATE eval_full — vanilla (no contextual retrieval)")
    print("=" * 72)
    proc = start_api({
        "CHUNKING_MODE": "full",
        "RETRIEVAL_MODE": "full",
        "CONTEXTUAL_RETRIEVAL": "0",
    })
    try:
        bu.ensure_customer("eval-full")
        print("\n[wipe] clearing existing eval_full docs")
        bu.wipe_existing("eval-full")
        print(f"\n[upload] uploading {len(bu.SAMPLES)} sample docs (vanilla)")
        bu.upload_all("eval-full")
    finally:
        stop_api(proc)
    print("\nDone. eval_full now has 6 vanilla docs and is ready for exp3.")


if __name__ == "__main__":
    main()
