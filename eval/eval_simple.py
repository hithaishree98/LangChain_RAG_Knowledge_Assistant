import csv, re, json, time, statistics, requests
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np
import os

API_URL = "http://localhost:8000/chat"
MODEL = "llama-3.1-8b-instant"

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

def ask(question):
    r = requests.post(
        "http://localhost:8000/chat",
        json={"question": question}
    )

    data = r.json()

    return {
        "answer": data["answer"],
        "confidence": data["confidence"],
        "sources": data["sources"]
    }

def evaluate(csv_path: str):
    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    sims, covs, lats = [], [], []
    errors = 0

    for i, row in enumerate(rows, 1):
        q = row["question"].strip()
        ref = row.get("reference_answer","").strip()
        facts = [x for x in row.get("key_facts","").split(";") if x.strip()]

        t0 = time.perf_counter()
        resp = ask(q)
        t1 = time.perf_counter()

        if "error" in resp:
            errors += 1
            print(f"[{i}] ERROR {resp['error']} | Q: {q}")
            continue

        ans = resp.get("answer", "")
        conf = resp.get("confidence", 0.0)
        sources = resp.get("sources", [])

        lat = (t1 - t0) * 1000.0
        lats.append(lat)

        sim = semantic_sim(ans, ref) if ref else 0.0
        cov = coverage(ans, facts)

        sims.append(sim)
        covs.append(cov)

        print(f"[{i}] sim={sim:.2f} cov={cov:.2f} conf={conf:.2f} lat={lat:.1f}ms")
        print(f"     sources={sources}")
        print(f"     Q: {q}")
        print()

    n = len(sims)
    p50 = statistics.median(lats) if lats else 0.0
    p95 = statistics.quantiles(lats, n=20)[-1] if len(lats) >= 20 else (max(lats) if lats else 0.0)
    err_rate = errors / (n + errors) if (n + errors) else 0.0

    out = {
        "semantic_similarity_avg": round(sum(sims)/n, 3) if n else 0.0,
        "key_facts_coverage_avg": round(sum(covs)/n, 3) if n else 0.0,
        "p50_latency_ms": round(p50, 1),
        "p95_latency_ms": round(p95, 1),
        "error_rate": round(err_rate, 3),
        "count": n
    }
    out_path = os.path.join(os.path.dirname(csv_path), "metrics_open_simple.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print("SIMPLE METRICS")
    print(json.dumps(out, indent=2))

def evaluate_custom(questions: list[dict]):
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
            sims.append(sim)
            covs.append(cov)
            print(f"[{i}] sim={sim:.2f} cov={cov:.2f} lat={lat:.1f}ms | {row['question']}")

    n = len(sims)
    return {
        "semantic_similarity_avg": round(sum(sims)/n, 3) if n else 0.0,
        "key_facts_coverage_avg": round(sum(covs)/n, 3) if n else 0.0,
        "p50_latency_ms": round(statistics.median(lats), 1) if lats else 0.0,
        "count": n
    }

if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(__file__), "eval_set_multi_company.csv")
    evaluate(csv_path)
