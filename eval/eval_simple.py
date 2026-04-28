import csv, re, json, time, statistics, requests
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
import os

BRIEF_URL = os.getenv("BRIEF_URL", "http://localhost:8000/brief")
# Derive the auth-token URL from BRIEF_URL so the eval works against any deployment
# (localhost in dev, a staging URL in CI). The /brief and /auth/token endpoints
# share the same host.
AUTH_URL  = BRIEF_URL.rsplit("/", 1)[0] + "/auth/token"
MODEL    = "llama-3.1-8b-instant"
API_KEY  = os.getenv("API_KEY", "")

# Tenant identity is JWT-only — the API used to honor a `?user_id=` query param,
# but that was an IDOR. The eval mints a token once via /auth/token and reuses
# it across all queries.
EVAL_WORKSPACE = os.getenv("EVAL_WORKSPACE", "eval-default")
EVAL_PASSKEY   = os.getenv("EVAL_PASSKEY", "eval-default-passkey")

INTER_QUERY_SLEEP = float(os.getenv("INTER_QUERY_SLEEP", "15"))


def _mint_token(workspace: str, passkey: str) -> tuple:
    """Authenticate once and return (token, user_id) for all subsequent calls.

    Raises if /auth/token is unreachable so the eval fails loudly rather than
    silently degrading to unauthenticated requests against the "default" tenant.
    """
    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    r = requests.post(
        AUTH_URL,
        json={"workspace": workspace, "passkey": passkey},
        headers=headers,
        timeout=10,
    )
    r.raise_for_status()
    data = r.json()
    return data["token"], data["user_id"]


_TOKEN, _USER_ID = _mint_token(EVAL_WORKSPACE, EVAL_PASSKEY)

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMB_MODEL)


def normalize(t: str) -> str:
    t = t.strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def semantic_sim(a: str, b: str) -> float:
    v = embedder.encode([a, b], convert_to_numpy=True)
    return max(0.0, min(1.0, cosine(v[0], v[1])))


def coverage(answer: str, facts: List[str]) -> float:
    if not facts: return 0.0
    ans = normalize(answer)
    hit = 0
    for f in facts:
        f = f.strip()
        if not f: continue
        if f.lower() in ans or semantic_sim(answer, f) >= 0.6:
            hit += 1
    return hit / max(1, len(facts))


def ask(question: str, user_id: str = None) -> dict:
    # user_id arg is kept for back-compat with callers but ignored — tenancy
    # comes from the JWT minted at module import. Pass the same id as customer_id
    # so the brief logs attribute the query to the eval workspace.
    customer_id = user_id or _USER_ID
    headers = {"Authorization": f"Bearer {_TOKEN}"}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    r = requests.post(
        BRIEF_URL,
        json={"query": question, "customer_id": customer_id},
        headers=headers,
        timeout=180,
    )
    if not r.ok:
        return {"error": f"HTTP {r.status_code}: {r.text[:200]}"}
    data = r.json()
    brief = data.get("brief", {})
    # Detect silent LLM failures: brief returned HTTP 200 but the reason node failed
    # (e.g. Groq rate limit after retries exhausted). In that case issues/risks/
    # talking_points are empty and open_questions contains an error string.
    open_qs = brief.get("open_questions", [])
    if (not brief.get("issues") and not brief.get("risks")
            and not brief.get("talking_points")
            and any(isinstance(q, str)
                    and ("could not analyze" in q.lower()
                         or "could not parse analyst" in q.lower())
                    for q in open_qs)):
        msg = next((q for q in open_qs if isinstance(q, str)), "silent LLM failure")
        return {"error": f"silent_llm_failure: {msg[:150]}"}
    claims = (
        [item.get("claim", "") for item in brief.get("issues", []) if isinstance(item, dict)]
        + [item.get("claim", "") for item in brief.get("risks", []) if isinstance(item, dict)]
        + [item.get("point", "") for item in brief.get("talking_points", []) if isinstance(item, dict)]
        + [q for q in brief.get("open_questions", []) if isinstance(q, str)]
    )
    answer = " ".join(c for c in claims if c)
    return {
        "answer":             answer,
        "confidence":         data.get("faithfulness_score", 0.0),
        "sources":            [s["filename"] for s in data.get("sources", []) if isinstance(s, dict) and "filename" in s],
        "faithfulness_score": data.get("faithfulness_score", 0.0),
        "loop_count":         data.get("loop_count", 0),
    }

# ── Retrieval quality: source-level ──────────────────────────────────────────

def recall_at_k(retrieved_sources: List[str], gold_source: str, k: int = 5) -> float:
    """1.0 if gold_source appears in the top-k retrieved sources, else 0.0."""
    if not gold_source:
        return 0.0
    return 1.0 if gold_source.lower() in [s.lower() for s in retrieved_sources[:k]] else 0.0


def reciprocal_rank(retrieved_sources: List[str], gold_source: str) -> float:
    """1/rank of the first hit, or 0.0 if not found."""
    if not gold_source:
        return 0.0
    for i, s in enumerate(retrieved_sources, start=1):
        if gold_source.lower() in s.lower():
            return 1.0 / i
    return 0.0


# ── Retrieval quality: chunk-level ────────────────────────────────────────────

def chunk_precision_at_k(retrieved_chunk_ids: List[str], gold_chunk_ids: List[str],
                          k: int = 5) -> float:
    """
    Fraction of the top-k retrieved chunks that are in the gold set.
    Requires gold_chunks column in the eval CSV (semicolon-separated chunk IDs).
    """
    if not gold_chunk_ids:
        return 0.0
    top_k = retrieved_chunk_ids[:k]
    if not top_k:
        return 0.0
    gold_set = set(g.strip().lower() for g in gold_chunk_ids)
    hits = sum(1 for cid in top_k if cid.strip().lower() in gold_set)
    return hits / len(top_k)


def chunk_recall_at_k(retrieved_chunk_ids: List[str], gold_chunk_ids: List[str],
                       k: int = 5) -> float:
    """
    Fraction of gold chunks that appear in the top-k retrieved chunks.
    """
    if not gold_chunk_ids:
        return 0.0
    top_k = retrieved_chunk_ids[:k]
    gold_set = set(g.strip().lower() for g in gold_chunk_ids)
    hits = sum(1 for cid in top_k if cid.strip().lower() in gold_set)
    return hits / len(gold_set)


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate(csv_path: str, chunking_config: Optional[str] = None, out_dir: Optional[str] = None):
    """
    Run evaluation against a CSV test set.

    CSV columns:
      question, reference_answer, key_facts (semicolon-separated)
      Optional: gold_source (filename), gold_chunks (semicolon-separated chunk IDs)

    chunking_config: optional label to tag the output file, useful for
    comparing multiple chunking strategies side-by-side.
    out_dir: directory to write the metrics JSON; defaults to same dir as csv_path.
    """
    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    sims, covs, lats = [], [], []
    recalls, rrs = [], []
    chunk_precs, chunk_recs = [], []
    faith_scores, loop_counts = [], []
    errors = 0

    for i, row in enumerate(rows, 1):
        q         = row["question"].strip()
        ref       = row.get("reference_answer", "").strip()
        facts     = [x for x in row.get("key_facts", "").split(";") if x.strip()]
        gold_src  = row.get("source_filename", row.get("gold_source", "")).strip()
        gold_cids = [x for x in row.get("gold_chunks", "").split(";") if x.strip()]

        t0 = time.perf_counter()
        resp = ask(q)
        t1 = time.perf_counter()

        if "error" in resp:
            errors += 1
            print(f"[{i}] ERROR {resp['error']} | Q: {q}")
            if i < len(rows):
                time.sleep(INTER_QUERY_SLEEP)
            continue

        ans     = resp.get("answer", "")
        sources = resp.get("sources", [])
        faith   = resp.get("faithfulness_score", 0.0)
        loops   = resp.get("loop_count", 0)
        # The /brief response doesn't expose chunk IDs — only filenames.
        # Chunk-level precision/recall requires the API to return chunk_ids in the
        # brief payload. Until then this list stays empty and those metrics stay null.
        retrieved_chunk_ids = []

        lat = (t1 - t0) * 1000.0
        lats.append(lat)
        faith_scores.append(faith)
        loop_counts.append(loops)

        sim = semantic_sim(ans, ref) if ref else 0.0
        cov = coverage(ans, facts)
        sims.append(sim)
        covs.append(cov)

        # Source-level retrieval quality
        r5 = recall_at_k(sources, gold_src, k=5)
        rr = reciprocal_rank(sources, gold_src)
        if gold_src:
            recalls.append(r5)
            rrs.append(rr)

        # Chunk-level retrieval quality
        if gold_cids and retrieved_chunk_ids:
            cp = chunk_precision_at_k(retrieved_chunk_ids, gold_cids, k=5)
            cr = chunk_recall_at_k(retrieved_chunk_ids, gold_cids, k=5)
            chunk_precs.append(cp)
            chunk_recs.append(cr)
        else:
            cp = cr = None

        print(
            f"[{i}] sim={sim:.2f} cov={cov:.2f} faith={faith:.2f} loops={loops} "
            f"R@5={r5:.0f} MRR={rr:.2f} lat={lat:.1f}ms"
            + (f" chunkP={cp:.2f} chunkR={cr:.2f}" if cp is not None else "")
        )
        print(f"     sources={sources}")
        print(f"     Q: {q}")
        print()

        if i < len(rows):
            time.sleep(INTER_QUERY_SLEEP)

    n    = len(sims)
    p50  = statistics.median(lats) if lats else 0.0
    # quantiles(n=20) gives vigintiles; [-1] is the 95th percentile.
    # With < 2 samples quantiles() errors, so fall back to the single value.
    if len(lats) >= 2:
        p95 = statistics.quantiles(lats, n=20)[-1]
    elif lats:
        p95 = lats[0]
    else:
        p95 = 0.0
    err_rate = errors / (n + errors) if (n + errors) else 0.0

    out = {
        "chunking_config":         chunking_config,
        "semantic_similarity_avg": round(sum(sims) / n, 3) if n else 0.0,
        "key_facts_coverage_avg":  round(sum(covs) / n, 3) if n else 0.0,
        "mean_faithfulness_score": round(sum(faith_scores) / len(faith_scores), 3) if faith_scores else None,
        "avg_loop_count":          round(sum(loop_counts) / len(loop_counts), 2) if loop_counts else None,
        "recall_at_5":             round(sum(recalls) / len(recalls), 3) if recalls else None,
        "mrr":                     round(sum(rrs) / len(rrs), 3) if rrs else None,
        "chunk_precision_at_5":    round(sum(chunk_precs) / len(chunk_precs), 3) if chunk_precs else None,
        "chunk_recall_at_5":       round(sum(chunk_recs) / len(chunk_recs), 3) if chunk_recs else None,
        "p50_latency_ms":          round(p50, 1),
        "p95_latency_ms":          round(p95, 1),
        "error_rate":              round(err_rate, 3),
        "count":                   n,
    }

    # Tag output file with config name so multiple runs don't overwrite each other
    suffix = f"_{chunking_config}" if chunking_config else ""
    save_dir = out_dir if out_dir else os.path.dirname(csv_path)
    out_path = os.path.join(save_dir, f"metrics_open_simple{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print("METRICS")
    print(json.dumps(out, indent=2))


def evaluate_custom(questions: List[dict]):
    sims, covs, lats, faith_scores, loop_counts = [], [], [], [], []
    for i, row in enumerate(questions, 1):
        facts = [x for x in row.get("key_facts", "").split(";") if x.strip()]
        t0 = time.perf_counter()
        resp = ask(row["question"])
        lat = (time.perf_counter() - t0) * 1000.0
        lats.append(lat)
        if "error" not in resp:
            ans = resp.get("answer", "")
            sim = semantic_sim(ans, row.get("reference_answer", ""))
            cov = coverage(ans, facts)
            faith = resp.get("faithfulness_score", 0.0)
            loops = resp.get("loop_count", 0)
            sims.append(sim); covs.append(cov)
            faith_scores.append(faith); loop_counts.append(loops)
            print(f"[{i}] sim={sim:.2f} cov={cov:.2f} faith={faith:.2f} loops={loops} "
                  f"lat={lat:.1f}ms | {row['question']}")

    n = len(sims)
    return {
        "semantic_similarity_avg": round(sum(sims) / n, 3) if n else 0.0,
        "key_facts_coverage_avg":  round(sum(covs) / n, 3) if n else 0.0,
        "mean_faithfulness_score": round(sum(faith_scores) / n, 3) if n else None,
        "avg_loop_count":          round(sum(loop_counts) / n, 2) if n else None,
        "p50_latency_ms":          round(statistics.median(lats), 1) if lats else 0.0,
        "count": n,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=os.path.join(os.path.dirname(__file__),
                                                        "eval_set_meridian.csv"))
    parser.add_argument("--config", default=None,
                        help="Label for this chunking config, e.g. 'sentence_256' or 'recursive_800'")
    args = parser.parse_args()
    evaluate(args.csv, chunking_config=args.config)
