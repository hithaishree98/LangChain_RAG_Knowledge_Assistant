"""
Experiment 1 — Chunking ablation

Tests three chunking strategies on the same 6 sample documents:

  baseline  — PyPDFLoader/Docx2txtLoader/BeautifulSoup + RecursiveCST(800,200).
               No noise filtering. No structure awareness. Flat index.

  sentence  — Advanced loaders: PyMuPDF+noise+sentence_chunk(256w) for PDF,
               heading-aware RecursiveCST(800) for DOCX, HTMLHeaderSplitter for
               HTML, specialized parsers for TXT/JSON. Flat index (no parent store).

  full      — Same advanced loaders as sentence, BUT indexed as parent-child:
               children (500 chars) are searched, parents (1600 chars) are fetched
               and returned as context. This gives precise embedding matching with
               full-paragraph context.

What each mode is expected to win at:
  baseline  — fastest indexing, simplest pipeline
  sentence  — better structure preservation than baseline, no retrieval overhead
  full      — best recall on multi-paragraph questions (parent context retrieved)

Phase 1: upload 6 docs under user_id=eval_{mode} with CHUNKING_MODE={mode}
Phase 2: evaluate all three workspaces with CHUNKING_MODE=full RETRIEVAL_MODE=full

Usage:
    python experiment_kit/experiments/exp1_chunking.py

Requires: GOOGLE_API_KEY set, api/requirements.txt installed.

Phase 2 note: all three chunking modes are evaluated with RETRIEVAL_MODE=full.
For baseline and sentence modes (flat index, no parent chunks), the "full"
retrieval path's fetch_parents() silently falls back to child chunks because
no parent_chunk_id metadata is present. This means Phase 2 is not a pure
chunking ablation — the retrieval path differs subtly between modes. Results
reflect end-to-end config performance, not isolated chunking effect.
"""
import csv
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
KIT_ROOT  = REPO_ROOT / "experiment_kit"

sys.path.insert(0, str(REPO_ROOT / "eval"))
sys.path.insert(0, str(KIT_ROOT / "experiments"))

from api_utils import start_api, stop_api

EVAL_RESULTS = KIT_ROOT / "eval_results"


# ── Helpers ───────────────────────────────────────────────────────────────────

def bootstrap(user_id: str, wipe: bool = True):
    import bootstrap_upload as bu
    bu.ensure_customer(user_id)
    if wipe:
        print(f"  [wipe] clearing existing docs for {user_id}")
        bu.wipe_existing(user_id)
    print(f"  [upload] uploading {len(bu.SAMPLES)} sample docs under {user_id}")
    bu.upload_all(user_id)


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
    print("EXPERIMENT 1 — CHUNKING ABLATION")
    print("=" * 72)

    import bootstrap_upload as _bu

    # Phase 1: index each chunking config
    for mode in ("baseline", "sentence", "full"):
        user_id = f"eval-{mode}"
        print(f"\n── Phase 1 [{mode}] ──")
        proc = start_api({"CHUNKING_MODE": mode, "RETRIEVAL_MODE": "full"})
        try:
            bootstrap(user_id, wipe=True)
        finally:
            stop_api(proc)
            _bu._TOKEN = None  # JWT_SECRET regenerates on restart; cached token is invalid
            time.sleep(2)

    # Phase 2: run eval for each config (retrieval mode fixed to full)
    print("\n── Phase 2: eval runs ──")
    proc = start_api({"CHUNKING_MODE": "full", "RETRIEVAL_MODE": "full"})
    try:
        for mode in ("baseline", "sentence", "full"):
            print(f"\n  [eval:{mode}] running eval against user_id=eval_{mode}")
            run_eval(f"eval-{mode}", mode)
    finally:
        stop_api(proc)

    print("\n=== Results ===")
    for mode in ("baseline", "sentence", "full"):
        p = EVAL_RESULTS / f"metrics_open_simple_{mode}.json"
        if p.exists():
            data = json.loads(p.read_text())
            print(f"  {p.name}:")
            print(f"    semantic_sim={data.get('semantic_similarity_avg')}  "
                  f"coverage={data.get('key_facts_coverage_avg')}  "
                  f"recall@1={data.get('recall_at_1')}  "
                  f"mrr={data.get('mrr')}  "
                  f"p50={data.get('p50_latency_ms')}ms")
        else:
            print(f"  {p.name}: MISSING")


if __name__ == "__main__":
    main()
