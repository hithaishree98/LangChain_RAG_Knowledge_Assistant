# Testing and Evals

## Start the system

```bash
docker compose up --build
```

Open http://localhost:8501. You should see the sidebar with a customer dropdown and corpus health panel. If the sidebar shows a connection error, the API container is still starting — wait 10–15 seconds and reload.

For demo data:

```bash
docker compose exec api python scripts/demo_seed.py
```

---

## Manual browser scenarios

Run these in order — later ones depend on data from earlier ones.

---

### Scenario 1: Create a customer

1. In the left sidebar, expand **Create new customer**.
2. Enter a name (e.g. "Acme Corp") and a slug (e.g. "acme-corp"). The slug must be lowercase alphanumeric with hyphens only.
3. Click **Create**.

**Expected:** The customer appears in the sidebar dropdown and is immediately selectable. The corpus health panel shows all doc types as empty.

**What to check:** Switch to another customer if one exists, then switch back to Acme Corp. The corpus health should show Acme Corp's counts only. Each customer's documents are stored in the same ChromaDB collection but filtered by user_id — this is your first check that isolation is working.

**Variation:** Try creating a second customer with the same slug. You should get an error — slugs are unique per workspace.

---

### Scenario 2: Upload a document

1. Select "Acme Corp" in the sidebar and go to the **Upload Documents** tab.
2. Upload a `.csv` file with columns `ticket_id`, `summary`, `priority`, `status`.
3. Check the auto-detected type label beneath the selector.

**Expected:** The selector pre-fills as "ticket" and shows "✓ Auto-detected as ticket". Upload succeeds with "Chunks indexed: N."

After upload, the corpus health in the sidebar should show a count under "ticket" for this customer.

**Variation:** Upload the same file again without checking "Replace if already uploaded" — you should get a duplicate error. Try uploading a `.pdf` file — the type detection won't pre-fill (PDF sniffing is server-side only), so you'll need to select the type manually before uploading.

---

### Scenario 3: Account health score

1. Make sure at least one ticket in your uploaded CSV has priority "P0" or "critical".
2. Go to the **Account Health** tab and click **Refresh**.

**Expected:** A health band card appears (Healthy / At Risk / Critical) with a numeric score. The "Open P0 Tickets" tile shows a non-zero count.

The score formula is `100 − (P0×20) − (P1×5) − (overdue commitments×10) − (slip rate×30)` plus a penalty for days since last call after 14 days. You can verify this against your uploaded data. If you uploaded two P0 tickets, the score should be at most 60. If it's 95, something is wrong with the ticket priority field.

This tab makes no LLM calls — all numbers come from document metadata.

---

### Scenario 4: Pre-meeting brief on an empty corpus

1. Create a new customer with no documents uploaded.
2. Select that customer, go to **Pre-Meeting Brief**, and click **Generate Brief**.

**Expected:** The brief generates without erroring. An amber warning banner appears at the top: "No documents uploaded for this customer. Upload a transcript, ticket export, and commitment tracker to generate a useful brief."

The section panels (Open Items, Outstanding Commitments, etc.) should show empty state labels, not invented content. The recommended posture section should be empty.

---

### Scenario 5: Overdue commitment detection

1. Upload a commitment tracker JSON or CSV for "Acme Corp" with at least one commitment where `target_date` is in the past and `status` is not "completed."
2. Go to **Pre-Meeting Brief** and click **Generate Brief**.

**Expected:** The Overdue Commitments section (red banner at the top of the brief) shows the past-due item. The snapshot bar shows a non-zero overdue count in red.

**What to check:** The `is_overdue` flag is computed at ingest from the `target_date` field, not by the LLM. To verify this is working correctly: if you set `target_date` to 2025-01-01, that commitment should appear as overdue. If you set it to 2026-06-01, it should appear in Outstanding Commitments but not in Overdue.

Open the "Outstanding Commitments" expander — the same commitment should appear there in the table with its target date and owner fields visible.

**Variation:** Upload a commitment with `status = "completed"`. It should not appear in either section. Upload one with `target_date` in the future and `status = "in_progress"` — it shows in Outstanding only.

---

### Scenario 6: Query — answer found

1. With at least one ticket or account_notes document uploaded for Acme Corp, go to the **Query** tab.
2. Ask a question that has a direct answer in the uploaded content — e.g. "What is the priority of the login bug ticket?"

**Expected:** An `OK` status badge. The answer block contains a response. The "Sources (N) — M chunks searched" expander shows the source document filename.

**What to check:** Expand "Sources." The document name should match a file you uploaded. The `doc_date` shown comes from the filename — it should match the date in the filename you used (e.g. `ticket_2025-03-15_acme.csv` → doc_date: 2025-03-15).

**Variation:** Ask a broad question like "What are all the issues and commitments for this customer?" The query rewrite node will decompose this into sub-queries. The answer will likely return `partial` for a broad synthesis question — that's expected. Check that Sources shows multiple documents.

---

### Scenario 7: Query — answer not found

With only ticket data uploaded (no transcripts), ask something that requires a transcript to answer, like "What did the customer say about their Q3 budget?"

You should see a `NOT_FOUND` badge and no answer text — not a fabricated answer. A hint may appear below the answer block suggesting which document type would help: "Uploading a transcript could improve results for this question."

If you ask a question that's partially answerable (the tickets mention a keyword but the full answer needs a transcript), you should see `PARTIAL` with a confidence explanation string saying why the answer is incomplete.

---

### Scenario 8: Staleness warning

1. Upload an account_notes document and a commitment_tracker document, both with dates more than 31 days ago in the filename (e.g. `account_notes_2024-01-01_acme.txt` and `commitment_tracker_2024-01-01_acme.csv`).
2. Generate a pre-meeting brief.
3. Open the **Data Sources & Confidence** expander at the bottom.

**Expected:** The expander label shows "⚠️ N stale" in the header. Stale warnings list both documents.

**What to check:** Now upload a solution_architecture document with a date from 100 days ago. Regenerate the brief. The solution architecture document should not appear as stale — its threshold is 180 days. Each doc type has a different freshness window: commitment tracker 14d, transcript/ticket/account_notes 30d, QBR deck 90d, solution_architecture 180d.

---

### Scenario 9: Exec 1:1 brief

1. In the **Exec 1:1 Brief** tab, expand **Add stakeholder** and add a person whose name appears in your uploaded transcript (e.g. "Jane Smith"). Click **Add**.
2. Select "Jane Smith" from the person dropdown and click **Generate 1:1 Brief**.

**Expected:** Role & Tenure, Stated Position, Recent Signals, and Open Asks sections appear. Statements that are quoted speech from Jane Smith appear in blockquote format.

**What to check:** The source documents listed under each statement should be from your transcript. The exec brief only surfaces what it can actually trace — if Jane Smith is not mentioned in any uploaded document, all sections except Role & Tenure will show "No statements on record" / "No recent signals."

**Variation:** Add a person who isn't mentioned in any document and generate their brief. All sections should show empty states cleanly — no error, no fabricated content.

---

### Scenario 10: Multi-tenant isolation

Both customers share the same ChromaDB collection — isolation is enforced through user_id metadata filters, not separate namespaces. This scenario verifies that filter is actually working.

1. Upload a document for Acme Corp with a distinctive phrase in the content (e.g. a ticket mentioning "Project Thunderbird").
2. Create a second customer "Beta Inc" and go to their **Query** tab.
3. Ask "What is Project Thunderbird?"

**Expected:** `NOT_FOUND` or an answer that makes no reference to Acme Corp's document. The Sources expander should be empty or show only Beta Inc documents.

**What to check:** Switch back to Acme Corp and ask the same question. The answer should now be found, citing the document you uploaded under Acme Corp.

---

## Test suite

```bash
docker compose exec api python -m pytest tests/ -v
```

| File | What it tests |
|---|---|
| `tests/test_api.py` | Auth token, customer CRUD, document upload pipeline, brief generation, query endpoint, feedback |
| `tests/test_ingestion.py` | Parser and chunker output per doc type: transcript turn boundaries, ticket section splits, commitment field extraction |
| `tests/test_new_modules.py` | Hallucination detection layers, staleness threshold values, circuit breaker state transitions |

Run a single file:

```bash
docker compose exec api python -m pytest tests/test_ingestion.py -v
```

---

## Evals

Seed demo data first, then run these from the containers.

```bash
docker compose exec api python experiment_kit/experiments/eval_retrieval_modes.py
```
Compares BM25-only, dense-only, and hybrid RRF on a fixed question set. Reports MRR and top-k hit rate per mode.

```bash
docker compose exec api python experiment_kit/experiments/eval_chunking_strategies.py
```
Tests parent-child vs flat chunking on retrieval recall. Shows whether returning the parent chunk improves answer quality.

```bash
docker compose exec api python experiment_kit/experiments/eval_hallucination_layers.py
```
Runs the three-layer detection pipeline against synthetic claims and reports how many get caught at each layer (regex, classifier, LLM judge).

```bash
docker compose exec api python experiment_kit/experiments/eval_latency_per_node.py
```
Profiles each LangGraph node in the pre-meeting brief. Reports p50/p95 latency — useful for knowing which node to optimize first.

```bash
docker compose exec api python experiment_kit/experiments/eval_cost_estimate.py
```
Estimates token cost per brief and per query by doc count using tiktoken. Run this before enabling `CONTEXTUAL_RETRIEVAL` to understand the ingest cost multiplier.

```bash
docker compose exec api python experiment_kit/experiments/eval_cost_real.py
```
Same but measures actual token usage from the Gemini billing API rather than estimating.

```bash
docker compose exec api python experiment_kit/experiments/eval_brief_types.py
```
Compares pre-meeting brief vs exec 1:1 brief output on the same corpus — section completeness, citation density.

```bash
docker compose exec api python experiment_kit/experiments/eval_query_endpoint.py
```
Runs a fixed question set against `/query` and reports answer status distribution and mean chunks searched.

```bash
docker compose exec api python experiment_kit/experiments/eval_workflow_loop.py
```
Stress-tests the LangGraph workflow with concurrent invocations. Runs for ~60 seconds — expect it to take a while. Surfaces any state accumulation or race conditions in parallel section nodes.

```bash
docker compose exec api python experiment_kit/experiments/analyze_node_timings.py
```
Post-processes timing logs written by `eval_latency_per_node.py`. Run this after that eval, not standalone.
