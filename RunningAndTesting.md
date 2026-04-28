# Running and Testing Guide

How to run the project locally, how to run tests inside Docker, and what scenarios to cover.

---

## Running locally

### Option A — Docker (recommended)

**Prerequisites:** Docker Desktop installed and running.

**Step 1 — Create `.env` in the project root:**
```
# Required — get a free key at https://aistudio.google.com/apikey
GOOGLE_API_KEY=your_google_api_key

# Optional — override the LLM model (default: gemini-2.5-flash)
# LLM_MODEL=gemini-2.5-flash

# Optional — enable Anthropic-style contextual retrieval at ingest time.
# Adds ~5s/chunk to upload time. Off by default.
# CONTEXTUAL_RETRIEVAL=1

# Required in production, optional in dev
API_KEY=any_secret_string
JWT_SECRET=a_long_random_string_at_least_32_chars
```

**Step 2 — Build and start:**
```bash
docker-compose up --build
```

First build takes a few minutes (downloads PyTorch, sentence-transformers, spaCy model). Subsequent starts are fast.

**Step 3 — Open:**
- UI: http://localhost:8501
- API docs (Swagger): http://localhost:8000/docs
- Health check: http://localhost:8000/health

**Stop:**
```bash
docker-compose down
```

**Stop and wipe data** (clears SQLite + Chroma):
```bash
docker-compose down -v
```

---

### Option B — Without Docker

**Prerequisites:** Python 3.10+, pip.

**Step 1 — Install the spaCy model** (one-time):
```bash
pip install spacy
python -m spacy download en_core_web_sm
```

**Step 2 — API:**
```bash
cd api
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**Step 3 — Streamlit (separate terminal):**
```bash
cd app
pip install -r requirements.txt
streamlit run streamlit_app.py
```

**Step 4 — Set env vars:**

On Linux/Mac:
```bash
export GOOGLE_API_KEY=your_key
export API_KEY=any_secret
export JWT_SECRET=a_long_random_string
```

On Windows (PowerShell):
```powershell
$env:GOOGLE_API_KEY="your_key"
$env:API_KEY="any_secret"
$env:JWT_SECRET="a_long_random_string"
```

---

## Running tests

### Unit/integration smoke tests with Docker

The test file is `tests/test_api.py`. It uses `TestClient(app)` — FastAPI's built-in test client — so no running server is needed.

**Run tests inside the API container:**
```bash
# If the stack is running:
docker-compose exec api pytest tests/ -v

# If you want to spin up a one-off container just to run tests:
docker-compose run --rm api pytest tests/ -v
```

**Run tests locally (without Docker):**
```bash
cd api
pip install pytest httpx
pytest ../tests/ -v
```

Expected output:
```
tests/test_api.py::test_health PASSED
tests/test_api.py::test_brief_end_to_end PASSED          # requires GOOGLE_API_KEY
tests/test_api.py::test_circuit_breaker_opens_after_threshold PASSED
tests/test_api.py::test_circuit_breaker_recovery_path PASSED
tests/test_api.py::test_contextualize_chunks_happy_path PASSED
tests/test_api.py::test_contextualize_chunks_llm_failure_fallback PASSED
tests/test_api.py::test_contextualize_chunks_empty_llm_response_is_fallback PASSED
tests/test_api.py::test_strip_context_prefix_removes_prefix_when_flag_set PASSED
tests/test_api.py::test_strip_context_prefix_no_op_without_flag PASSED
tests/test_api.py::test_strip_context_prefix_no_op_when_content_mismatches_flag PASSED
... (40 tests total, 0 failures expected)
```

`test_brief_end_to_end` is gated on `GOOGLE_API_KEY` being set — it makes a real Gemini call to validate the full /brief pipeline including hallucination layers and judge_status surfacing. It will skip silently if the key isn't set, so you'll see 39 passed + 1 skipped instead of 40 passed.

---

### Evaluation harness (requires running API + documents)

These scripts hit the live API. Start the stack first.

**Generation + retrieval eval:**
```bash
cd eval
python eval_simple.py --csv eval_set_multi_company.csv

# Compare two chunking configs:
python eval_simple.py --csv eval_set_multi_company.csv --config sentence_256
python eval_simple.py --csv eval_set_multi_company.csv --config recursive_800
```

Output: `eval/metrics_open_simple.json` (or `metrics_open_simple_sentence_256.json`).

**Faithfulness eval (headline metric):**
```bash
cd eval
python faithfulness_eval.py --user_id your_workspace_id
```

Output: `eval/faithfulness_metrics.json`. Target: `pass_rate_at_0_90 >= 0.90`.

---

## What to test — manual scenarios

Work through these in order. Each builds on the previous.

---

### 1. Health and auth

**Check the API is up:**
```bash
curl http://localhost:8000/health
```
Expected: `{"status": "healthy", "checks": {"database": "ok", "vector_store": "ok", "llm_key": "ok", ...}}`

**Get a JWT:**
```bash
curl -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"workspace": "acme", "passkey": "secret123"}'
```
Expected: `{"token": "eyJ...", "user_id": "..."}`

**Save the token and user_id for use below:**
```bash
TOKEN="eyJ..."
USER_ID="..."
```

---

### 2. Document upload — each supported format

Upload a PDF:
```bash
curl -X POST "http://localhost:8000/upload-doc?user_id=$USER_ID" \
  -H "X-API-Key: your_api_key" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@/path/to/document.pdf"
```
Expected: `{"message": "'document.pdf' uploaded successfully.", "file_id": 1}`

Upload a plain-text transcript (`speaker: text` format):
```bash
curl -X POST "http://localhost:8000/upload-doc?user_id=$USER_ID&doc_type=transcript" \
  -H "X-API-Key: your_api_key" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@/path/to/transcript.txt"
```

Upload a ticket JSON:
```bash
curl -X POST "http://localhost:8000/upload-doc?user_id=$USER_ID&doc_type=ticket" \
  -H "X-API-Key: your_api_key" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@/path/to/ticket.json"
```

List documents to confirm indexing:
```bash
curl "http://localhost:8000/list-docs?user_id=$USER_ID" \
  -H "Authorization: Bearer $TOKEN"
```
Expected: JSON array with `id`, `filename`, `upload_timestamp`, `user_id` for each uploaded file.

---

### 3. Brief generation — core workflow

Submit a pre-call query:
```bash
curl -X POST http://localhost:8000/brief \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"query": "What are the open issues and risks I should know before this call?", "customer_id": "'$USER_ID'"}'
```

**What to verify in the response:**
- `brief.issues` is a non-empty list where each item has `claim`, `chunk_id`, `source_doc`, and `passage`
- `brief.risks` follows the same structure
- `brief.talking_points` each have `point` with citation
- `brief.open_questions` is a list of strings
- `faithfulness_score` is between 0.0 and 1.0 (ideally > 0.4)
- `loop_count` is 1, 2, or 3
- `sources` lists the filenames that were used

**Test with a specific factual query** (requires uploaded documents containing the answer):
- Expected: `faithfulness_score >= 0.6`, citations pointing to correct source doc

**Test with an out-of-scope query** (nothing in documents matches):
- Expected: `open_questions` contains unanswered items, `faithfulness_score` near 0, `brief.issues` empty

---

### 4. Brief viewer in Streamlit

1. Open http://localhost:8501
2. Enter workspace `acme` and passkey `secret123` → Continue
3. In the sidebar, select document type and upload a PDF → "Upload and index"
4. In the Brief tab, enter: `"What were the main concerns discussed?"` → Generate brief
5. Expand an issue — verify the source doc name and passage text appear
6. Check the Faithfulness metric is displayed
7. If faithfulness < 40%, verify the escalation warning banner appears
8. Expand "Potentially ungrounded facts" (if present) — verify suspicious facts are listed

---

### 5. Auth enforcement

**No API key on a protected endpoint:**
```bash
curl -X POST http://localhost:8000/brief \
  -H "Content-Type: application/json" \
  -d '{"query": "test"}'
```
Expected: `403 Forbidden` (if `API_KEY` is set in `.env`)

**Expired/invalid JWT:**
```bash
curl http://localhost:8000/list-docs?user_id=default \
  -H "Authorization: Bearer invalidtoken"
```
Expected: `401 Unauthorized`

**Cross-workspace isolation — upload a doc as user A, query as user B:**
- Create workspace A (`acme/secret123`), upload a PDF
- Create workspace B (`other/password`), run a brief query
- Expected: workspace B's brief returns no results from workspace A's documents

---

### 6. Upload validation

**Unsupported file type:**
```bash
curl -X POST "http://localhost:8000/upload-doc?user_id=$USER_ID" \
  -H "X-API-Key: your_api_key" \
  -F "file=@/path/to/image.png"
```
Expected: `400 Bad Request` — "Unsupported file type '.png'"

**Duplicate filename:**
Upload the same file twice.
Expected: second upload returns `409 Conflict`

**Empty file:**
```bash
touch /tmp/empty.pdf
curl -X POST "http://localhost:8000/upload-doc?user_id=$USER_ID" \
  -H "X-API-Key: your_api_key" \
  -F "file=@/tmp/empty.pdf"
```
Expected: `400 Bad Request` — "Uploaded file is empty."

---

### 7. Bulk questionnaire

Create `test_questions.csv`:
```csv
question
What are the main issues?
What risks were discussed?
What is the agreed SLA?
```

```bash
curl -X POST "http://localhost:8000/answer-questionnaire?user_id=$USER_ID" \
  -H "X-API-Key: your_api_key" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@test_questions.csv"
```

**What to verify:**
- `results` array has one entry per row
- Each result has `question`, `answer`, `confidence`, `sources`, `needs_review`
- `needs_review: true` on low-confidence answers

**Test the row limit:**
Create a CSV with 201 rows.
Expected: `400 Bad Request` — "CSV too large. Maximum 200 rows allowed."

**Test rate limit (2/minute):**
Call the endpoint 3 times within one minute.
Expected: third call returns `429 Too Many Requests`

---

### 8. Circuit breaker (`llm_breaker`)

Set `GOOGLE_API_KEY` to an invalid key, restart the API:
```bash
GOOGLE_API_KEY=invalid docker-compose up api
```

Call `/brief` five times. After 5 failures:
```bash
curl http://localhost:8000/health
```
Expected: `"circuit_breaker": "open"`

Next `/brief` call:
Expected: `503 Service Unavailable` immediately (no LLM call attempted)

Wait 30 seconds, call `/brief` again — circuit transitions to `HALF_OPEN` and makes one probe attempt. Successful probe → `CLOSED`. Failed probe → back to `OPEN`.

The breaker name is provider-agnostic (`llm_breaker`), so this works the same regardless of which LLM provider you've configured via `LLM_MODEL`.

---

### 9. Document deletion

```bash
# Delete file_id 1
curl -X POST http://localhost:8000/delete-doc \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{"file_id": 1, "user_id": "'$USER_ID'"}'
```
Expected: `{"message": "Document deleted."}`

Verify it's gone:
```bash
curl "http://localhost:8000/list-docs?user_id=$USER_ID" \
  -H "Authorization: Bearer $TOKEN"
```
Expected: file_id 1 no longer in the list.

Run a brief query that relied on that document — verify results no longer reference it.

**Delete nonexistent file:**
```bash
curl -X POST http://localhost:8000/delete-doc \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{"file_id": 9999, "user_id": "'$USER_ID'"}'
```
Expected: `404 Not Found`

---

### 10. Analytics and audit log

After running several brief queries:
```bash
curl "http://localhost:8000/analytics?user_id=$USER_ID" \
  -H "X-API-Key: your_api_key" \
  -H "Authorization: Bearer $TOKEN"
```
Expected: `total_queries > 0`, `avg_confidence` between 0 and 1, `top_questions` list populated.

```bash
curl "http://localhost:8000/audit-log?user_id=$USER_ID" \
  -H "X-API-Key: your_api_key" \
  -H "Authorization: Bearer $TOKEN"
```
Expected: list of log entries with `user_query`, `confidence`, `escalated`, `sources`.

---

### 11. Eval harness (end-to-end quality check)

Run after uploading documents and verifying basic brief generation works.

```bash
cd eval
python eval_simple.py
```

Check `metrics_open_simple.json`:
- `semantic_similarity_avg` should be > 0.5 for a good document set
- `recall_at_5` and `mrr` require `gold_source` column in the CSV
- `chunk_precision_at_5` and `chunk_recall_at_5` require `gold_chunks` column

```bash
python faithfulness_eval.py --user_id your_workspace_id
```

Check `faithfulness_metrics.json`:
- `mean_faithfulness` — target > 0.70
- `pass_rate_at_0_90` — target > 0.90

---

## Experiment kit

`experiment_kit/experiments/` holds the ablation studies that produce the numbers in README's "Results" section. Each experiment is independently runnable.

### Required env vars before running any experiment

```powershell
$env:GOOGLE_API_KEY="your_gemini_key"
$env:CONTEXTUAL_RETRIEVAL="0"        # vanilla chunking; set to 1 to test Anthropic's method
$env:INTER_QUERY_SLEEP="1"           # paid Gemini; bump higher on free-tier
$env:PYTHONIOENCODING="utf-8"        # Windows console fix
$env:TOKEN_LOGGING="1"               # capture real token counts for exp6b
```

These five stay fixed for the full experiment session — per-experiment overrides like `CHUNKING_MODE`, `RETRIEVAL_MODE`, `WORKFLOW_MODE`, `NODE_TIMING` are set by the experiment scripts themselves via the API subprocess.

### Run order

```bash
# Repopulate eval_full if a prior partial run left it inconsistent
python experiment_kit/experiments/repopulate_eval_full.py

# Chunking ablation — baseline vs sentence vs full (~50 min)
python experiment_kit/experiments/exp1_chunking.py

# Retrieval ablation — dense vs +BM25 vs +reranker (~45 min)
python experiment_kit/experiments/exp2_retrieval.py

# Workflow loop ablation — single_pass vs the citation-rate-triggered loop (~50 min)
python experiment_kit/experiments/exp3_workflow_loop.py

# Hallucination assertions — 9 cases across 3 layers (~5 min)
python experiment_kit/experiments/exp4_hallucination.py

# Per-node latency breakdown (~30 min)
python experiment_kit/experiments/exp5_latency.py
python experiment_kit/experiments/exp5_analyze.py

# Cost — tiktoken estimate (needs API running in another shell)
python experiment_kit/experiments/exp6_cost.py

# Cost — real LLM-reported usage at Gemini 2.5-flash pricing (~25 min)
python experiment_kit/experiments/exp6b_real_cost.py
```

All output JSONs land in `experiment_kit/eval_results/`. Compare across runs by suffixing filenames (e.g., `_vanilla.json` vs `_contextual.json` if you flip `CONTEXTUAL_RETRIEVAL`).

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| API container exits on startup | Missing `GOOGLE_API_KEY` (or whatever `LLM_MODEL` you set requires) | Add to `.env` |
| `/health` never returns 200 | `warmup_models()` is loading the 440MB nomic-embed model | Wait 3-5 min on first start; subsequent starts use cached weights |
| `/health` returns 200 but `circuit_breaker: open` | LLM provider key invalid or 5 consecutive 429s | Check the key, wait 30s for HALF_OPEN |
| `503` on `/brief` immediately | Circuit breaker open | Check `/health`, fix LLM key, wait 30s |
| `judge_status: "parse_error"` in briefs | Gemini occasionally returns malformed JSON for the judge prompt | Transient — retry the brief. If persistent, see `_JUDGE_PROMPT` in `langchain_utils.py` |
| `judge_status: "skipped_breaker_open"` | LLM breaker tripped during this request | Same fix as 503 — wait for HALF_OPEN |
| `judge_status: "no_context_all_unsupported"` | Retrieval returned zero chunks | Check `user_id` matches an actually-uploaded workspace |
| `[WARN] workspace 'X' has existing chunks with ingest_contextual_retrieval=...` at upload | You're mixing CONTEXTUAL_RETRIEVAL=0 and =1 chunks in the same workspace — vector space is inconsistent | Wipe the workspace and re-ingest under a single flag value |
| Empty brief, zero issues | No documents uploaded | Upload documents first |
| `401` on all requests | `API_KEY` set but not passed | Add `X-API-Key` header |
| Streamlit shows "API offline" | API container not running | `docker-compose up api` |
| Upload succeeds but brief still empty | Chroma indexing failed | Check `GET /logs?level=ERROR` |
| Upload times out at 5 minutes with `CONTEXTUAL_RETRIEVAL=1` | Per-chunk LLM calls take ~5s each; 100-child docs need ~10 min | Bump the upload timeout in your client (the experiment kit's `bootstrap_upload.py` uses 1800s for this reason). See ImprovementsForProd.md #14 for the planned batching fix |
| `spacy` import error | Model not downloaded | `python -m spacy download en_core_web_sm` |
| Very slow first brief | Cross-encoder loading on first request | Should be rare — `warmup_models()` pre-loads it. Check the lifespan logs for `warmup_failed` |
| Streamlit "Verified" chip never appears | `judge_status` isn't `"ok"` for any of your queries — verification is failing | Check the brief's `judge_status` value in the response |
