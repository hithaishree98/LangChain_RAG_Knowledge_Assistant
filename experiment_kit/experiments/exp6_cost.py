"""
Experiment 6 — Token and cost math (tiktoken estimates)

Goal: produce a defensible "this is what it costs" claim with real numbers.

Strategy:
  Reproduce the EXACT prompts the LLM would see for a sample of queries using
  the real retrieval pipeline. Count input tokens via tiktoken. Estimate output
  tokens from exp6b actuals (calibrated from real LLM usage logs).

LLM calls per /brief request (current workflow):
  1. query_rewrite     — decomposes query into 2-4 sub-queries
  2. reason            — main analyst prompt (largest call)
  3. account_summary   — 2-3 sentence posture paragraph
  4. recent_changes    — fires only when docs within 14 days exist
  5. anticipated_topics— fires only when open tickets/transcripts found
  6. posture           — synthesises sections into 3-5 directives
  7. llm_judge         — verifies claims against context (batch, 1 call per query)

Calls 4 and 5 are conditional. All experiments use 2024 sample docs, so
recent_changes and anticipated_topics return early without an LLM call.
To get the full-cost upper bound include them; the default here models the
unconditional calls (1, 2, 3, 6, 7).

Output token estimates are calibrated from exp6b real measurements:
  query_rewrite : ~920 tokens  (exp6b avg_llm_output)
  reason        : ~3159 tokens (exp6b avg_llm_output)
  account_summary: ~80 tokens  (short prose, ~80 words)
  posture        : ~200 tokens (3-5 directive objects)
  llm_judge      : ~2143 tokens (exp6b avg_llm_output)

Tokenizer note: tiktoken cl100k_base (OpenAI/GPT-4 tokenizer) underestimates
Gemini input token counts by ~10% per exp6b cross-validation. Treat these
as lower-bound input estimates; real input cost is ~10% higher.

Usage:
    # API must be running with CHUNKING_MODE=full, RETRIEVAL_MODE=full,
    # and the eval_full workspace populated.
    python experiment_kit/experiments/exp6_cost.py

    # Run exp6b after this to measure real LLM usage and validate these estimates.
"""
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "api"))

# ── LLM pricing — Gemini 2.5-flash for all call types ────────────────────────
# All seven call types use the same model and pricing tier.
# Verify at https://ai.google.dev/pricing — override via env if rates change.
INPUT_PRICE_PER_M  = float(os.getenv("LLM_INPUT_PRICE_PER_M",  "0.30"))   # $ per 1M input
OUTPUT_PRICE_PER_M = float(os.getenv("LLM_OUTPUT_PRICE_PER_M", "2.50"))   # $ per 1M output

# ── Embedding pricing — only relevant when OPENAI_API_KEY is set ─────────────
# text-embedding-3-small: $0.02 per 1M tokens.
# At query time: ~3 sub-queries × ~30 tokens each = ~90 tokens per /brief call.
# At 1000 queries/day × 30 days = 2.7M tokens/month → $0.054/month.
# Against ~$500+/month LLM cost this is <0.01% — negligible but non-zero.
# If OPENAI_API_KEY is not set, the local MiniLM-L6-v2 model is used and
# embedding cost is $0.
EMBEDDING_PRICE_PER_M = float(os.getenv("EMBEDDING_PRICE_PER_M", "0.02"))
USE_OPENAI_EMBEDDINGS = bool(os.getenv("OPENAI_API_KEY", ""))

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
    """Approximate token count using tiktoken cl100k_base (GPT-4 tokenizer).

    Underestimates Gemini token counts by ~10% per exp6b cross-validation.
    Use exp6b for exact numbers; use this for quick projections knowing the
    real input cost will be ~10% higher than reported here.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        return max(1, len(text) // 4)


def build_prompts_for_query(query: str, user_id: str) -> dict:
    """Reconstruct the exact prompts the LLM would see for this query.
    Does NOT call the LLM. Uses the real retrieval pipeline so chunks are realistic.
    """
    from graph.nodes import _QA_PROMPT, _build_context_str
    from graph.workflow import _ADAPTIVE_PROMPT
    from chroma_utils import get_retriever_for_user, fetch_parents

    rewrite_prompt = _ADAPTIVE_PROMPT.format(query=query)

    retriever = get_retriever_for_user(user_id)
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
    # Escape braces in chunk content so str.format() doesn't misinterpret JSON
    # keys like {overdue_commitments} as template slots. The resulting string
    # is identical after formatting — this only affects the format() call itself.
    ctx_safe = ctx.replace("{", "{{").replace("}", "}}")
    ctx6_safe = _build_context_str(docs[:6]).replace("{", "{{").replace("}", "}}")

    # reason prompt (query workflow answer prompt)
    reason_prompt = _QA_PROMPT.format(context=ctx_safe, query=query)

    from langchain_utils import _JUDGE_PROMPT
    mock_claims = "\n".join(
        [f"{i+1}. [claim placeholder ~60 chars of typical content]" for i in range(3)]
    )
    judge_prompt = _JUDGE_PROMPT.format(context=ctx_safe, claims_numbered=mock_claims)

    # account_summary prompt (short — uses first 6 docs and issue/risk text)
    from graph.nodes import _ACCOUNT_SUMMARY_PROMPT
    account_summary_prompt = _ACCOUNT_SUMMARY_PROMPT.format(
        context=ctx6_safe,
        as_of_date="2024-09-25",
    )

    # posture prompt (no doc context — synthesises section outputs)
    from graph.nodes import _POSTURE_PROMPT
    posture_prompt = _POSTURE_PROMPT.format(
        account_summary="(placeholder account summary ~60 words)",
        overdue_commitments="[]",
        open_items="[]",
        recent_changes="[]",
        commitments="[]",
        anticipated_questions="[]",
    )

    return {
        "rewrite_prompt":          rewrite_prompt,
        "reason_prompt":           reason_prompt,
        "judge_prompt":            judge_prompt,
        "account_summary_prompt":  account_summary_prompt,
        "posture_prompt":          posture_prompt,
        "retrieved_chunk_count":   len(docs),
    }


# Output token estimates calibrated from exp6b real LLM measurements.
# query_rewrite and reason and llm_judge come directly from exp6b avg_llm_output.
# account_summary and posture were not in exp6b (added after that run) —
# estimates are based on prompt constraints (80-word cap, 3-5 directives).
OUTPUT_TOKENS = {
    "query_rewrite":    920,   # exp6b avg_llm_output: 920
    "reason":          3159,   # exp6b avg_llm_output: 3158.6
    "account_summary":   80,   # prompt cap: 80 words ≈ ~80 tokens
    "posture":          200,   # 3-5 directive objects ≈ ~200 tokens
    "llm_judge":       2143,   # exp6b avg_llm_output: 2142.9
}
# Calls 4 (recent_changes) and 5 (anticipated_topics) are conditional.
# In experiments with 2024 sample docs they never fire. Excluded from default.
# Upper-bound estimates if they do fire:
#   recent_changes:    ~300 tokens (JSON array of change objects)
#   anticipated_topics: ~250 tokens (JSON array of 3-5 topic objects)


def main():
    user_id = os.getenv("USER_ID", "eval-full")
    print("=" * 72)
    print("EXPERIMENT 6 — TOKEN & COST MATH (tiktoken estimates)")
    print("=" * 72)
    print(f"LLM pricing (gemini-2.5-flash): input=${INPUT_PRICE_PER_M}/M, "
          f"output=${OUTPUT_PRICE_PER_M}/M")
    print(f"Workspace: user_id={user_id}")
    print(f"Tiktoken note: cl100k_base underestimates Gemini input by ~10%.")
    print(f"               Run exp6b for exact real-LLM measurements.")
    print()

    rows = []
    print(f"{'#':>3} {'rw_in':>7} {'rn_in':>7} {'jg_in':>7} {'as_in':>7} {'ps_in':>6} "
          f"{'tot_in':>8} {'tot_out':>8} {'cost_$':>9}  query")
    print("-" * 120)

    totals = {k: 0 for k in ["rw_in", "rn_in", "jg_in", "as_in", "ps_in", "out", "cost"]}

    for i, q in enumerate(SAMPLE_QUERIES, 1):
        try:
            bundle = build_prompts_for_query(q, user_id)
        except Exception as e:
            print(f"  [err] query {i}: {e}")
            continue

        rw_in = count_tokens(bundle["rewrite_prompt"])
        rn_in = count_tokens(bundle["reason_prompt"])
        jg_in = count_tokens(bundle["judge_prompt"])
        as_in = count_tokens(bundle["account_summary_prompt"])
        ps_in = count_tokens(bundle["posture_prompt"])
        total_in = rw_in + rn_in + jg_in + as_in + ps_in

        total_out = sum(OUTPUT_TOKENS.values())

        cost_in  = total_in  * INPUT_PRICE_PER_M  / 1_000_000
        cost_out = total_out * OUTPUT_PRICE_PER_M / 1_000_000
        cost = cost_in + cost_out

        for k, v in [("rw_in", rw_in), ("rn_in", rn_in), ("jg_in", jg_in),
                     ("as_in", as_in), ("ps_in", ps_in),
                     ("out", total_out), ("cost", cost)]:
            totals[k] += v

        print(f"{i:>3} {rw_in:>7d} {rn_in:>7d} {jg_in:>7d} {as_in:>7d} {ps_in:>6d} "
              f"{total_in:>8d} {total_out:>8d} {cost:>9.6f}  {q[:55]}")
        rows.append((q, rw_in, rn_in, jg_in, as_in, ps_in, total_in, total_out, cost))

    n = len(rows)
    if n == 0:
        print("no data — is the API running and user_id populated?")
        return

    avg_in  = (totals["rw_in"] + totals["rn_in"] + totals["jg_in"] +
               totals["as_in"] + totals["ps_in"]) / n
    avg_out  = totals["out"] / n
    avg_cost = totals["cost"] / n

    print("-" * 120)
    print(f"{'avg':>3} {'':>7} {'':>7} {'':>7} {'':>7} {'':>6} "
          f"{avg_in:>8.0f} {avg_out:>8.0f} {avg_cost:>9.6f}  (across {n} queries)")

    # ── Embedding cost addendum ───────────────────────────────────────────────
    embed_tokens_per_query = 90  # ~3 sub-queries × ~30 tokens
    embed_cost_per_query = (embed_tokens_per_query * EMBEDDING_PRICE_PER_M / 1_000_000
                            if USE_OPENAI_EMBEDDINGS else 0.0)

    # ── Projections ──────────────────────────────────────────────────────────
    print()
    print("PROJECTED MONTHLY COST at different daily query volumes")
    print(f"  (LLM only — 5 unconditional calls: query_rewrite, reason, "
          f"account_summary, posture, llm_judge)")
    print("-" * 72)
    print(f"  {'queries/day':>12}   {'LLM/day':>10}   {'LLM/month':>12}   "
          f"{'embed/month':>12}   {'total/month':>12}")
    for qpd in [10, 100, 1000, 10_000]:
        llm_day   = avg_cost * qpd
        llm_mo    = llm_day * 30
        emb_mo    = embed_cost_per_query * qpd * 30
        total_mo  = llm_mo + emb_mo
        print(f"  {qpd:>12d}   ${llm_day:>9.4f}   ${llm_mo:>11.2f}   "
              f"${emb_mo:>11.4f}   ${total_mo:>11.2f}")

    # ── Dollar context ───────────────────────────────────────────────────────
    print()
    print("WHAT THIS MEANS")
    print("-" * 72)
    cost_1k_daily = avg_cost * 1000 * 30
    emb_str = (f"OpenAI text-embedding-3-small (~${embed_cost_per_query * 1000 * 30:.2f}/month "
               f"at 1k/day — negligible)"
               if USE_OPENAI_EMBEDDINGS
               else "MiniLM-L6-v2 local model (zero marginal cost)")
    print(f"  At 1,000 queries/day the LLM bill is ~${cost_1k_daily:.2f}/month.")
    print(f"  Embeddings: {emb_str}")
    print(f"  Vector store (Chroma) runs locally (zero marginal cost).")
    print(f"  SQLite is a file (zero marginal cost).")
    print()
    print("  Two conditional calls NOT included in these projections:")
    print("  - recent_changes: fires when docs exist within last 14 days.")
    print("    Expected cost if fired: ~$0.001/query (small context, short output).")
    print("  - anticipated_topics: fires when open tickets/transcripts found.")
    print("    Expected cost if fired: ~$0.001/query.")
    print()
    print("  Tiktoken underestimates Gemini input by ~10%. Actual LLM input cost")
    print("  is ~10% higher than shown. Run exp6b for exact real measurements.")
    print()
    print("  IMPORTANT: Compare against exp6b before quoting these numbers.")
    print(f"  exp6b (3-call model, pre-section-nodes): $0.017/query.")
    print(f"  This run (5-call model): ~${avg_cost:.4f}/query.")
    print(f"  Difference = account_summary + posture output token cost.")

    # Save
    out_path = Path(__file__).parent.parent / "eval_results" / "exp6_cost.json"
    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("w") as f:
        json.dump({
            "pricing_input_per_M":      INPUT_PRICE_PER_M,
            "pricing_output_per_M":     OUTPUT_PRICE_PER_M,
            "embedding_price_per_M":    EMBEDDING_PRICE_PER_M,
            "openai_embeddings_active": USE_OPENAI_EMBEDDINGS,
            "llm_calls_modeled":        ["query_rewrite", "reason", "account_summary",
                                         "posture", "llm_judge"],
            "llm_calls_conditional_excluded": ["recent_changes", "anticipated_topics"],
            "output_token_estimates":   OUTPUT_TOKENS,
            "tiktoken_undercount_note": "cl100k_base underestimates Gemini input by ~10%",
            "n_queries":                n,
            "avg_input_tokens":         round(avg_in, 1),
            "avg_output_tokens":        round(avg_out, 1),
            "avg_cost_per_query":       round(avg_cost, 6),
            "projections": {
                "10_per_day_monthly":    round(avg_cost * 10  * 30, 4),
                "100_per_day_monthly":   round(avg_cost * 100 * 30, 4),
                "1000_per_day_monthly":  round(avg_cost * 1000* 30, 4),
                "10000_per_day_monthly": round(avg_cost * 10000*30, 4),
            },
            "per_query_rows": [
                {
                    "query": q,
                    "tiktoken_rewrite_input":          rw,
                    "tiktoken_reason_input":            rn,
                    "tiktoken_judge_input":             jg,
                    "tiktoken_account_summary_input":   as_,
                    "tiktoken_posture_input":           ps,
                    "tiktoken_total_input":             ti,
                    "estimated_output":                 to,
                    "estimated_cost_usd":               c,
                }
                for q, rw, rn, jg, as_, ps, ti, to, c in rows
            ],
        }, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
