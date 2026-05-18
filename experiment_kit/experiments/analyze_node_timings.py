"""
analyze_node_timings.py — aggregate per-node timings from api/data/node_timings.jsonl

Reads the JSONL file emitted by the @_timed decorator in nodes.py and prints:
count, sum, p50, p90, p95, p99, max — per node.

Also prints the implied per-query breakdown. Note: account_summary,
recent_changes, commitments, anticipated_topics run in PARALLEL after
completeness — their individual times cannot be summed to get wall-clock
latency for that parallel group. Wall-clock per-query latency is better
read from exp5_latency.py's outer timer, which measures end-to-end time.
"""
import json
import statistics
from collections import defaultdict
from pathlib import Path


TIMING_FILE = Path(__file__).resolve().parents[2] / "api" / "data" / "node_timings.jsonl"

# Nodes that run sequentially in the main loop (query → retrieve → reason → check).
_SEQUENTIAL_NODES = ["query_rewrite", "retrieve", "reason", "completeness"]

# Nodes that run in parallel after completeness passes.
# Wall-clock time for this group = max(individual times), not sum.
_PARALLEL_NODES = ["account_summary", "recent_changes", "commitments", "anticipated_topics"]

# Runs after the parallel group converges.
_POST_PARALLEL_NODES = ["posture", "faithfulness"]


def percentile(values, p):
    if not values: return 0.0
    s = sorted(values)
    k = int(len(s) * p)
    k = min(k, len(s) - 1)
    return s[k]


def main():
    if not TIMING_FILE.exists():
        print(f"[err] {TIMING_FILE} not found — did you run exp5_latency.py first?")
        return

    buckets = defaultdict(list)
    with TIMING_FILE.open() as f:
        for line in f:
            try:
                entry = json.loads(line)
            except Exception:
                continue
            buckets[entry["node"]].append(entry["elapsed_ms"])

    if not buckets:
        print("[err] no timing entries found. Was NODE_TIMING=1 set?")
        return

    print()
    print(f"{'node':<18} {'calls':>7} {'sum_ms':>10} {'mean':>8} {'p50':>8} "
          f"{'p90':>8} {'p95':>8} {'p99':>8} {'max':>8}")
    print("-" * 90)

    node_order = _SEQUENTIAL_NODES + _PARALLEL_NODES + _POST_PARALLEL_NODES
    seen = set(node_order)
    extras = [n for n in buckets if n not in seen]

    for node in node_order + extras:
        if node not in buckets:
            continue
        xs = buckets[node]
        print(f"{node:<18} {len(xs):>7d} "
              f"{sum(xs):>10.1f} {statistics.mean(xs):>8.1f} "
              f"{percentile(xs, 0.5):>8.1f} {percentile(xs, 0.9):>8.1f} "
              f"{percentile(xs, 0.95):>8.1f} {percentile(xs, 0.99):>8.1f} "
              f"{max(xs):>8.1f}")

    # ── Implied per-query total ──────────────────────────────────────────────
    counts = [len(v) for v in buckets.values()]
    if counts:
        n_queries = min(counts)
        print()
        print(f"  queries observed (min node count): ~{n_queries}")

        # Sequential nodes: sum their averages → real sequential cost per query.
        seq_ms = sum(
            statistics.mean(buckets[n]) for n in _SEQUENTIAL_NODES if n in buckets
        )
        # Parallel nodes: wall-clock = max of the group, not sum.
        par_ms = max(
            (statistics.mean(buckets[n]) for n in _PARALLEL_NODES if n in buckets),
            default=0.0,
        )
        post_ms = sum(
            statistics.mean(buckets[n]) for n in _POST_PARALLEL_NODES if n in buckets
        )
        implied_wall = seq_ms + par_ms + post_ms
        print(f"  implied wall-clock per query:      {implied_wall:.1f} ms")
        print(f"    sequential (loop nodes):         {seq_ms:.1f} ms")
        print(f"    parallel group (max of group):   {par_ms:.1f} ms")
        print(f"    post-parallel (posture+faith):   {post_ms:.1f} ms")
        print()
        print("  share of sequential time by node:")
        total_seq = sum(sum(buckets[n]) for n in _SEQUENTIAL_NODES if n in buckets)
        for node in _SEQUENTIAL_NODES:
            if node not in buckets:
                continue
            share = sum(buckets[node]) / max(total_seq, 1) * 100
            print(f"    {node:<18} {share:>5.1f}%")

    print()
    print("Reading the results:")
    print("  Sequential nodes (run in order per iteration):")
    print("  - query_rewrite   : Gemini LLM call for query decomposition.")
    print("  - retrieve        : dense embed + BM25 + rerank + parent-fetch stack.")
    print("    If retrieve is <15% of sequential time, 'over-engineered retrieval'")
    print("    arguments don't hold up.")
    print("  - reason          : Gemini LLM call — the main analyst prompt.")
    print("    Typically the largest single-node cost.")
    print("  - completeness    : pure Python citation-rate check, sub-millisecond.")
    print()
    print("  Parallel section nodes (run concurrently after completeness passes):")
    print("  - account_summary : Gemini LLM call — 2-3 sentence posture paragraph.")
    print("  - recent_changes  : Gemini LLM call — only fires if docs within 14 days.")
    print("    In experiments with 2024 sample docs this node returns immediately.")
    print("    Set EVAL_TODAY env var to a 2024 date to exercise this path.")
    print("  - commitments     : metadata-only, no LLM call. Should be near zero.")
    print("  - anticipated_topics: Gemini LLM call — only fires if tickets/transcripts found.")
    print()
    print("  Post-parallel nodes:")
    print("  - posture         : Gemini LLM call — synthesises all sections into directives.")
    print("  - faithfulness    : cosine similarity check, no LLM call.")
    print()
    print("  If retrieve p95 >> p50, the cross-encoder is the culprit on cold runs")
    print("  (lazy load on first call). Subsequent calls are fast.")


if __name__ == "__main__":
    main()
