import csv, re, json, time, statistics, requests, datetime
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
import os

QUERY_URL = os.getenv("QUERY_URL", "http://localhost:8000/query")
# Derive the auth-token URL from QUERY_URL so the eval works against any deployment.
AUTH_URL  = QUERY_URL.rsplit("/", 1)[0] + "/auth/token"
API_KEY  = os.getenv("API_KEY", "")

# Tenant identity is JWT-only — the API used to honor a `?user_id=` query param,
# but that was an IDOR. The eval mints a token once via /auth/token and reuses
# it across all queries.
EVAL_WORKSPACE = os.getenv("EVAL_WORKSPACE", "eval-default")
EVAL_PASSKEY   = os.getenv("EVAL_PASSKEY", "eval-default-passkey")

INTER_QUERY_SLEEP = float(os.getenv("INTER_QUERY_SLEEP", "15"))

# Customer slug to query against. Must be created via POST /customers first.
# If not set, falls back to the JWT user_id (legacy behaviour).
EVAL_CUSTOMER_ID = os.getenv("EVAL_CUSTOMER_ID", "")


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


# Lazy token: the previous module-level call meant `import eval_simple`
# from ANY context (Jupyter notebook, test, helper script) hit the network
# and crashed if the API was offline. Now we mint on first use and cache.
_TOKEN: Optional[str] = None
_USER_ID: Optional[str] = None


def _ensure_token():
    global _TOKEN, _USER_ID
    if _TOKEN is None:
        _TOKEN, _USER_ID = _mint_token(EVAL_WORKSPACE, EVAL_PASSKEY)
    return _TOKEN, _USER_ID

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
    """Fraction of `facts` that the answer covers — by substring match OR
    by sentence-level semantic similarity.

    Previous implementation called ``semantic_sim(answer, f)`` (whole answer
    vs. one short fact). On long answers the cosine similarity is high
    against essentially any topic-related fact — inflated the metric to the
    point of meaninglessness. We instead split the answer into sentences and
    require AT LEAST ONE sentence to be similar to the fact, which is the
    semantically correct definition of "this fact is covered somewhere in
    the answer."
    """
    if not facts:
        return 0.0
    ans_lower = normalize(answer)
    # Split answer into sentence-ish units for fact-level similarity. Filter
    # short fragments so noise like "OK." doesn't get weighted as a sentence.
    sentences = [s.strip() for s in re.split(r"[.!?]\s+", answer) if len(s.strip()) > 15]
    hit = 0
    for f in facts:
        f = f.strip()
        if not f:
            continue
        # Cheap path first: substring match on lowercased answer.
        if f.lower() in ans_lower:
            hit += 1
            continue
        # Fall back to per-sentence semantic similarity. A fact is "covered"
        # iff some sentence is similar to it (>= 0.6 cosine).
        if any(semantic_sim(s, f) >= 0.6 for s in sentences):
            hit += 1
    return hit / max(1, len(facts))


def ask(question: str, user_id: str = None) -> dict:
    # customer_id resolution order:
    #   1. EVAL_CUSTOMER_ID env var (explicit customer slug)
    #   2. user_id arg (back-compat)
    #   3. JWT user_id (legacy fallback — only works if a customer with that slug exists)
    token, default_user = _ensure_token()
    customer_id = EVAL_CUSTOMER_ID or user_id or default_user
    headers = {"Authorization": f"Bearer {token}"}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    r = requests.post(
        QUERY_URL,
        json={"question": question, "customer_id": customer_id},
        headers=headers,
        timeout=180,
    )
    if not r.ok:
        return {"error": f"HTTP {r.status_code}: {r.text[:200]}"}
    data = r.json()
    answer_status = data.get("answer_status", "not_found")
    answer = data.get("answer") or ""
    if not answer or answer_status == "not_found":
        return {"error": f"answer_status={answer_status}"}
    # Extract source from the single citation the /query endpoint returns
    citation = data.get("citation") or {}
    sources = [citation["document"]] if citation.get("document") else []
    return {
        "answer":       answer,
        "confidence":   1.0 if answer_status == "ok" else 0.5,
        "sources":      sources,
        "answer_status": answer_status,
        "recency_flag": data.get("recency_flag"),
    }

# ── Retrieval quality: source-level ──────────────────────────────────────────

def recall_at_k(retrieved_sources: List[str], gold_source: str, k: int = 5) -> float:
    """1.0 if gold_source appears (as substring) in the top-k retrieved sources, else 0.0."""
    if not gold_source:
        return 0.0
    return 1.0 if any(gold_source.lower() in s.lower() for s in retrieved_sources[:k]) else 0.0


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

        lat = (t1 - t0) * 1000.0
        lats.append(lat)

        sim = semantic_sim(ans, ref) if ref else 0.0
        cov = coverage(ans, facts)
        sims.append(sim)
        covs.append(cov)

        # Source-level retrieval quality (/query exposes one citation; k=1 is the effective limit)
        r5 = recall_at_k(sources, gold_src, k=5)
        rr = reciprocal_rank(sources, gold_src)
        if gold_src:
            recalls.append(r5)
            rrs.append(rr)

        print(
            f"[{i}] sim={sim:.2f} cov={cov:.2f} "
            f"R@1={r5:.0f} MRR={rr:.2f} lat={lat:.1f}ms"
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
        "run_metadata": {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "embedder_model": EMB_MODEL,
            "chunking_config": chunking_config,
            "eval_workspace": EVAL_WORKSPACE,
            "customer_id": EVAL_CUSTOMER_ID or "jwt_derived",
            "query_url": QUERY_URL,
            "thresholds": {
                "sentence_sim_coverage": 0.6,
            },
        },
        "chunking_config":         chunking_config,
        "semantic_similarity_avg": round(sum(sims) / n, 3) if n else 0.0,
        "key_facts_coverage_avg":  round(sum(covs) / n, 3) if n else 0.0,
        "recall_at_1":             round(sum(recalls) / len(recalls), 3) if recalls else None,
        "mrr":                     round(sum(rrs) / len(rrs), 3) if rrs else None,
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
    sims, covs, lats = [], [], []
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
            sims.append(sim); covs.append(cov)
            print(f"[{i}] sim={sim:.2f} cov={cov:.2f} "
                  f"lat={lat:.1f}ms | {row['question']}")

    n = len(sims)
    return {
        "semantic_similarity_avg": round(sum(sims) / n, 3) if n else 0.0,
        "key_facts_coverage_avg":  round(sum(covs) / n, 3) if n else 0.0,
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
