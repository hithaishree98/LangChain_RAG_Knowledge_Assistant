"""
exp5_analyze.py — aggregate per-node timings from /tmp/node_timings.jsonl

Reads the JSONL file emitted by the @_timed decorator in nodes_patched.py
and prints: count, sum, p50, p90, p95, max — per node.

Also prints the implied per-query breakdown (assuming each query runs
each node once per iteration, so totals / N_queries ≈ avg per query).
"""
import json
import statistics
from collections import defaultdict
from pathlib import Path


TIMING_FILE = Path(__file__).resolve().parents[2] / "api" / "data" / "node_timings.jsonl"


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
    print(f"{'node':<16} {'calls':>7} {'sum_ms':>10} {'mean':>8} {'p50':>8} {'p90':>8} {'p95':>8} {'max':>8}")
    print("-" * 82)
    node_order = ["query_rewrite", "retrieve", "reason", "completeness"]
    # any nodes we didn't list but appeared, print at the bottom
    seen = set(node_order)
    extras = [n for n in buckets if n not in seen]
    for node in node_order + extras:
        if node not in buckets: continue
        xs = buckets[node]
        print(f"{node:<16} {len(xs):>7d} "
              f"{sum(xs):>10.1f} {statistics.mean(xs):>8.1f} "
              f"{percentile(xs, 0.5):>8.1f} {percentile(xs, 0.9):>8.1f} "
              f"{percentile(xs, 0.95):>8.1f} {max(xs):>8.1f}")

    # ── Implied per-query total ──────────────────────────────────────────────
    # Each query goes through each node at least once. If the loop fires,
    # query_rewrite / retrieve / reason / completeness run extra times.
    # Rough estimate: take min(count) across nodes as "number of queries",
    # and max(count) across nodes as "total node invocations".
    counts = [len(v) for v in buckets.values()]
    if counts:
        n_queries = min(counts)
        print()
        print(f"  queries observed (min node count): ~{n_queries}")
        # Average latency PER QUERY (sum across nodes / n_queries)
        total_ms_per_query = sum(sum(v) for v in buckets.values()) / max(n_queries, 1)
        print(f"  avg node time per query:           {total_ms_per_query:.1f} ms")
        # Now break down where it goes
        print()
        print("  share of per-query time by node:")
        for node in node_order + extras:
            if node not in buckets: continue
            share = sum(buckets[node]) / sum(sum(v) for v in buckets.values()) * 100
            print(f"    {node:<16} {share:>5.1f}%")

    print()
    print("Reading the results:")
    print("  - reason and query_rewrite are LLM calls — this is Groq latency,")
    print("    not something you can optimize inside the retrieval pipeline.")
    print("  - retrieve is your dense + BM25 + rerank + parent-fetch stack.")
    print("    If it's <20% of total time, arguments that it is 'overengineered")
    print("    for the latency cost' don't hold up.")
    print("  - completeness is pure Python and should be sub-millisecond.")
    print("  - If retrieve p95 dwarfs its p50, the cross-encoder is the culprit")
    print("    (it loads models lazily — first call is expensive, later calls aren't).")


if __name__ == "__main__":
    main()
