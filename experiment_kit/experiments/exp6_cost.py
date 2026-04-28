"""
Experiment 6 — Token and cost math

Goal: produce a defensible "this is not expensive" claim with real numbers.

Strategy:
  We reproduce the EXACT prompts the LLM would see for a sample of queries,
  using the real retrieval pipeline. We don't need to actually call the LLM —
  counting input tokens is enough, and we can estimate output tokens from
  the schema size.

Three LLM calls per query (all on the same model):
  1. query_rewrite — prompt is ~1 sentence + the user query
  2. reason        — prompt is the analyst system + retrieved chunks + query
  3. llm_judge     — prompt is the retrieved chunks + the claims to verify

Usage:
    # API must be running with CHUNKING_MODE=full, RETRIEVAL_MODE=full,
    # and the eval_full workspace populated.
    python experiment_kit/experiments/exp6_cost.py

Output: per-query token counts, projected cost at current Gemini pricing,
plus a projected monthly cost at 100 / 1k / 10k queries per day.
"""
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "api"))

# ── LLM pricing — Gemini 2.5-flash for all three call types ──────────────────
# After the Gemini swap every graph node (query_rewrite, reason, llm_judge)
# uses the same model, so the old 8b/70b dual-pricing collapses to one tier.
# Verify at https://ai.google.dev/pricing — override via env if rates change.
INPUT_PRICE_PER_M  = float(os.getenv("LLM_INPUT_PRICE_PER_M",  "0.30"))   # $ per 1M input
OUTPUT_PRICE_PER_M = float(os.getenv("LLM_OUTPUT_PRICE_PER_M", "2.50"))   # $ per 1M output

SAMPLE_QUERIES = [
    "What is the current open P1 issue for Meridian?",
    "What is Meridian's default retrieval top-k value?",
    "What are Meridian's analyst seat usage and upsell opportunity?",
    "What did Sarah Chen say about the login latency issue on the call?",
    "What caused the duplicate invoice issue in TICK-4602?",
    "When is the EU region (eu-west-2) targeted for GA?",
    "How often are backup snapshots taken?",
    "What is the workaround for the login latency issue?",
    "What version is the Salesforce connector and what is its volume limit?",
    "What is Meridian's monthly platform fee?",
]


def count_tokens(text: str) -> int:
    """
    Approximate token count. Tries tiktoken (cl100k_base — close to Llama's
    tokenizer) and falls back to a char-heuristic if tiktoken isn't installed.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        # Fallback: ~4 chars per token is a decent rough estimate
        return max(1, len(text) // 4)


def build_prompts_for_query(query: str, user_id: str) -> dict:
    """
    Reconstruct the exact prompts the LLM would see for this query.
    Does NOT call the LLM. Uses the real retrieval pipeline so chunks are realistic.
    """
    # Import inside the function so the print() in chroma_utils fires once
    from graph.nodes import _DECOMPOSE_PROMPT, _ANALYST_SYSTEM, _build_context_str
    from chroma_utils import get_retriever_for_user, fetch_parents

    # Prompt 1: query_rewrite
    rewrite_prompt = _DECOMPOSE_PROMPT.format(query=query)

    # Prompt 2: reason — we must retrieve chunks first
    retriever = get_retriever_for_user(user_id)
    # Handle both old (get_relevant_documents) and new (invoke) LangChain APIs
    try:
        child_chunks = retriever.invoke(query)
    except AttributeError:
        child_chunks = retriever.get_relevant_documents(query)
    for i, doc in enumerate(child_chunks):
        if "chunk_id" not in doc.metadata:
            doc.metadata["chunk_id"] = f"C{i+1}"
    parent_chunks = fetch_parents(child_chunks, user_id=user_id)
    for i, doc in enumerate(parent_chunks):
        doc.metadata["chunk_id"] = f"P{i+1}"

    docs = parent_chunks or child_chunks
    ctx = _build_context_str(docs)
    reason_prompt = _ANALYST_SYSTEM.format(context=ctx, query=query)

    from langchain_utils import _JUDGE_PROMPT
    mock_claims = "\n".join([f"{i+1}. [claim placeholder ~60 chars of typical content]"
                              for i in range(3)])
    judge_prompt = _JUDGE_PROMPT.format(context=ctx, claims_numbered=mock_claims)

    return {
        "rewrite_prompt": rewrite_prompt,
        "reason_prompt": reason_prompt,
        "judge_prompt": judge_prompt,
        "retrieved_chunk_count": len(docs),
    }


# Rough per-call output token estimates, observed from the schemas:
#   query_rewrite returns a JSON array of 2-4 strings → ~40 tokens
#   reason returns the 4-section JSON brief → varies but ~400-600 tokens typical
OUTPUT_TOKENS_REWRITE = 40
OUTPUT_TOKENS_REASON  = 500


def main():
    user_id = os.getenv("USER_ID", "eval_full")
    print("=" * 72)
    print("EXPERIMENT 6 — TOKEN & COST MATH")
    print("=" * 72)
    print(f"LLM pricing (gemini-2.5-flash): input=${INPUT_PRICE_PER_M}/M, output=${OUTPUT_PRICE_PER_M}/M")
    print(f"Workspace: user_id={user_id}")
    print()

    rows = []
    print(f"{'#':>3} {'rw_in':>7} {'rn_in':>7} {'tot_in':>8} {'tot_out':>8} "
          f"{'cost_$':>9}  query")
    print("-" * 110)

    totals = {"rw_in": 0, "rn_in": 0, "out": 0, "cost": 0.0}

    for i, q in enumerate(SAMPLE_QUERIES, 1):
        try:
            bundle = build_prompts_for_query(q, user_id)
        except Exception as e:
            print(f"  [err] query {i}: {e}")
            continue

        rw_in = count_tokens(bundle["rewrite_prompt"])
        rn_in = count_tokens(bundle["reason_prompt"])
        jg_in = count_tokens(bundle["judge_prompt"])
        total_in = rw_in + rn_in + jg_in
        total_out = OUTPUT_TOKENS_REWRITE + OUTPUT_TOKENS_REASON + 200  # +200 for judge

        # Single model (gemini-2.5-flash) priced uniformly across all three calls
        cost_in  = total_in  * INPUT_PRICE_PER_M  / 1_000_000
        cost_out = total_out * OUTPUT_PRICE_PER_M / 1_000_000
        cost = cost_in + cost_out

        totals["rw_in"] += rw_in
        totals["rn_in"] += rn_in
        totals["jg_in"] = totals.get("jg_in", 0) + jg_in
        totals["out"]   += total_out
        totals["cost"]  += cost

        print(f"{i:>3} {rw_in:>7d} {rn_in:>7d} {total_in:>8d} {total_out:>8d} "
              f"{cost:>9.6f}  {q[:60]}")
        rows.append((q, rw_in, rn_in, total_in, total_out, cost))

    n = len(rows)
    if n == 0:
        print("no data — is the API running and user_id populated?")
        return

    avg_in = (totals["rw_in"] + totals["rn_in"] + totals.get("jg_in", 0)) / n
    avg_out = totals["out"] / n
    avg_cost = totals["cost"] / n

    print("-" * 110)
    print(f"{'avg':>3} {'':>7} {'':>7} {avg_in:>8.0f} {avg_out:>8.0f} "
          f"{avg_cost:>9.6f}  (across {n} queries)")

    # ── Projections ──────────────────────────────────────────────────────────
    print()
    print("PROJECTED COST at different daily query volumes")
    print("-" * 72)
    print(f"  {'queries/day':>12}   {'per day':>10}   {'per month':>12}   {'per year':>12}")
    for qpd in [10, 100, 1000, 10_000]:
        day = avg_cost * qpd
        mo  = day * 30
        yr  = day * 365
        print(f"  {qpd:>12d}   ${day:>9.4f}   ${mo:>11.2f}   ${yr:>11.2f}")

    # ── Dollar context ───────────────────────────────────────────────────────
    print()
    print("WHAT THIS MEANS")
    print("-" * 72)
    # At 1000 queries/day the cost is essentially negligible for a SaaS use case
    cost_1k_daily = avg_cost * 1000 * 30
    print(f"  At 1,000 queries/day the LLM bill is ~${cost_1k_daily:.2f}/month.")
    print("  Embeddings run locally (zero marginal cost).")
    print("  Vector store (Chroma) runs locally (zero marginal cost).")
    print("  SQLite is a file (zero marginal cost).")
    print()
    print("  The entire operating cost is LLM tokens. If someone says the")
    print("  architecture is 'expensive,' the answer is:")
    print(f"    '${cost_1k_daily:.2f}/month at 1k queries/day. Where is the expense?'")

    # Save the numbers
    out_path = Path(__file__).parent.parent / "eval_results" / "exp6_cost.json"
    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("w") as f:
        json.dump({
            "pricing_input_per_M":  INPUT_PRICE_PER_M,
            "pricing_output_per_M": OUTPUT_PRICE_PER_M,
            "n_queries":            n,
            "avg_input_tokens":     round(avg_in, 1),
            "avg_output_tokens":    round(avg_out, 1),
            "avg_cost_per_query":   round(avg_cost, 6),
            "projections": {
                "10_per_day_monthly":     round(avg_cost * 10 * 30, 4),
                "100_per_day_monthly":    round(avg_cost * 100 * 30, 4),
                "1000_per_day_monthly":   round(avg_cost * 1000 * 30, 4),
                "10000_per_day_monthly":  round(avg_cost * 10000 * 30, 4),
            },
            "per_query_rows": [
                {
                    "query": q,
                    "tiktoken_rewrite_input":  rw,
                    "tiktoken_reason_input":   rn,
                    "tiktoken_total_input":    ti,
                    "estimated_output":        to,
                    "estimated_cost_usd":      c,
                }
                for q, rw, rn, ti, to, c in rows
            ],
        }, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
