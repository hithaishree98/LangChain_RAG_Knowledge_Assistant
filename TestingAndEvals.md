# Testing and Evals

## Start the system

```bash
docker compose up --build
```

Open http://localhost:8501. The UI connects to the API at http://localhost:8000 automatically. You should see the sidebar with a customer dropdown and corpus health panel. If the sidebar shows a connection error, the API container hasn't finished starting — wait 10–15 seconds and reload.

For demo data:

```bash
docker compose exec api python scripts/demo_seed.py
```

---

## Manual browser scenarios

Each scenario below tests one named design decision. Run them in order — later scenarios depend on data created in earlier ones.

---

### Scenario 1: Create a customer

**Design decision tested:** Multi-tenant isolation — customers are scoped to the authenticated workspace.

1. In the left sidebar, expand **Create new customer**.
2. Enter a name (e.g. "Acme Corp") and a slug (e.g. "acme-corp"). The slug must be lowercase alphanumeric with hyphens only.
3. Click **Create**.

**Expected:** The customer appears in the sidebar dropdown and is immediately selectable. The corpus health panel shows all doc types as empty.

**What to check:** Select a different customer (if one exists), then re-select "Acme Corp." The corpus health should show counts for Acme Corp only, not any other customer's documents.

**Variation:** Try creating a second customer with the same slug. You should get an error — slugs are unique per workspace.

---

### Scenario 2: Upload a document

**Design decision tested:** Format-aware ingestion — the system detects doc type from file content and pre-fills the selector.

1. Select "Acme Corp" in the sidebar.
2. Go to the **Upload Documents** tab.
3. Upload a `.csv` file with columns `ticket_id`, `summary`, `priority`, `status`. (Create a minimal one if needed — three rows is enough.)
4. Check the auto-detected type label beneath the selector.

**Expected:** The selector pre-fills as "ticket" (or shows "Auto-detected as ticket"). The upload succeeds and shows "Chunks indexed: N."

**What to check:** After upload, return to the sidebar. The corpus health panel should show a count under "ticket" for this customer.

**Variation:** Upload a `.pdf` file and check the type selector — it should stay blank and let you choose, because PDF content sniffing is not implemented client-side. Override to "account_notes" and confirm the upload succeeds. Upload the same file again without checking "Replace if already uploaded" — the system should return a duplicate error rather than creating a second copy.

---

### Scenario 3: Account health score

**Design decision tested:** Account health is computed from metadata alone — no LLM call is made.

1. Upload a ticket CSV that includes at least one P0 ticket (priority field = "P0" or "critical").
2. Go to the **Account Health** tab.
3. Click **Refresh**.

**Expected:** A health band card appears (Healthy / At Risk / Critical) with a numeric score. The "Open P0 Tickets" tile shows a count greater than 0.

**What to check:** The score formula shown in the caption is `100 − (P0×20) − (P1×5) − (overdue commitments×10) − (slip rate×30) − (days since call penalty)`. Verify the displayed score is consistent with what you uploaded. Upload a second ticket CSV with no P0 tickets (keeping the first one) and refresh — if the new file replaces the old one (same filename, "Replace if already uploaded" checked), the score should change. If you uploaded under a different filename, both count.

**Variation:** Check the score for a customer with no documents at all. It should show 0 or a minimal score with all KPI tiles at 0, not an error.

---

### Scenario 4: Pre-meeting brief on an empty corpus

**Design decision tested:** The brief generator emits a corpus warning rather than fabricating content when there are no documents.

1. Create a new customer with no documents uploaded.
2. Select that customer and go to the **Pre-Meeting Brief** tab.
3. Click **Generate Brief**.

**Expected:** The brief generates (does not error). An amber warning banner appears at the top: something like "No documents uploaded for this customer" or "Corpus is empty."

**What to check:** The section panels (Open Items, Outstanding Commitments, etc.) show "empty" status labels, not fabricated content. The recommended posture section should be empty or show an empty state message, not invented directives.

---

### Scenario 5: Pre-meeting brief with commitment tracker

**Design decision tested:** Overdue commitments are computed from field values at ingest, not by the LLM at query time.

1. Upload a commitment tracker JSON or CSV for "Acme Corp" with at least one commitment where `target_date` is in the past and `status` is not "completed."
2. Go to the **Pre-Meeting Brief** tab.
3. Click **Generate Brief**.

**Expected:** The Overdue Commitments section (red banner) shows the past-due commitment. The snapshot bar at the top shows a non-zero overdue count in red.

**What to check:** Open the "Outstanding Commitments" expander. The table should include the same commitment. Cross-reference the `target_date` in the table against today's date — the "is_overdue" determination should match your calculation. If you set `target_date` to yesterday, it should appear as overdue. If you set it to tomorrow, it should appear in outstanding but not overdue.

**Variation:** Upload a commitment with `status = "completed"`. It should not appear in either Overdue or Outstanding Commitments. Upload a commitment with `target_date` in the future and status "in_progress" — it should appear in Outstanding but not Overdue.

---

### Scenario 6: Query — answer found

**Design decision tested:** Hybrid retrieval returns a grounded answer with a citation that traces to a real document.

1. Ensure "Acme Corp" has at least one ticket or account_notes document uploaded.
2. Go to the **Query** tab.
3. Ask a question that has a direct answer in the uploaded content, e.g. "What is the priority of the login bug ticket?"

**Expected:** An `OK` status badge appears. The answer block contains the answer. The "Sources (N) — M chunks searched" expander lists the source document filename and date.

**What to check:** Expand "Sources." The document name should match the file you uploaded. The `doc_date` shown should match the date in the filename or the document's creation date. If any source shows "⚠️ stale," check that the document is actually older than 30 days for tickets/transcripts.

**Variation:** Ask a vague, broad question like "What are all the issues and commitments and recent changes for this customer?" The query rewrite node should decompose this into sub-queries. The answer should still return a status badge (likely `partial` for a broad synthesis question). Check that the Sources expander shows multiple documents cited.

---

### Scenario 7: Query — answer not found

**Design decision tested:** The system returns `not_found` rather than hallucinating an answer when no relevant source exists.

1. With "Acme Corp" having only ticket data uploaded (no transcripts, no commitment tracker), go to the **Query** tab.
2. Ask a question that cannot be answered from the available documents, e.g. "What did the customer say about their Q3 budget?"

**Expected:** A `NOT_FOUND` status badge appears in red. The answer block shows "No answer found in the uploaded documents." A hint may appear below suggesting which document type to upload.

**What to check:** Confirm the answer block is empty or explicitly says "not found" — it should not contain a fabricated answer. If a "💡 Uploading a transcript could improve results" hint appears, that is the `missing_doc_types` hint from the answer generator checking the actual corpus composition.

**Variation:** Ask a question that is partially answerable, e.g. one where the tickets mention a keyword but the full answer requires a transcript. You should see a `PARTIAL` badge with a confidence explanation string describing why the answer is incomplete.

---

### Scenario 8: Staleness warning

**Design decision tested:** Staleness thresholds are enforced per doc type with specific cutoffs, not a single global age.

1. Upload an account_notes document and a commitment_tracker document, both dated more than 31 days ago (use a filename like `account_notes_2024-01-01_acme.txt`).
2. Generate a pre-meeting brief.
3. Open the **Data Sources & Confidence** expander at the bottom of the brief.

**Expected:** The expander label shows "⚠️ N stale" in its header. Stale source warnings list the account_notes document (30d threshold exceeded) and the commitment_tracker (14d threshold exceeded).

**What to check:** Upload a solution_architecture document dated 100 days ago (e.g. `solution_architecture_2025-01-01_acme.pdf`). Regenerate the brief. The solution_architecture document should NOT appear in stale warnings — its threshold is 180 days.

**Variation:** Upload a commitment_tracker dated 13 days ago. It should not appear as stale (threshold is 14 days). Upload one dated 15 days ago. It should appear as stale.

---

### Scenario 9: Exec 1:1 brief

**Design decision tested:** Person-filtered retrieval pulls only content where the named person appears, not all account content.

1. Select "Acme Corp."
2. In the **Exec 1:1 Brief** tab, expand **Add stakeholder**.
3. Add a person whose name appears in your uploaded transcript (e.g. "Jane Smith").
4. Select "Jane Smith" from the person dropdown and click **Generate 1:1 Brief**.

**Expected:** The brief shows Role & Tenure, Stated Position, Recent Signals, and Open Asks sections. Stated Position items that are quoted speech from Jane Smith appear in blockquote format.

**What to check:** The source documents listed under each statement should be from the uploaded transcript. If Jane Smith is not mentioned in any uploaded document, the Stated Position and Recent Signals sections should show "No statements on record" / "No recent signals" — not fabricated content.

**Variation:** Add a second person who is not mentioned in any document. Generate a brief for them. All sections except Role & Tenure should show empty states. The brief should still render cleanly rather than erroring.

---

### Scenario 10: Multi-tenant isolation

**Design decision tested:** Documents uploaded for one customer are not visible to another customer in the same workspace.

1. Upload a uniquely identifiable document (with a distinctive phrase in the content) under "Acme Corp."
2. Create a second customer, "Beta Inc."
3. Go to the **Query** tab with "Beta Inc" selected.
4. Ask a question that would match the distinctive phrase from the Acme Corp document.

**Expected:** `NOT_FOUND` or an answer that does not reference the Acme Corp document. The Acme Corp document should not appear in the Sources expander.

**What to check:** Switch back to "Acme Corp" and ask the same question. The answer should now be found, citing the document you uploaded.

---

## Test suite

```bash
docker compose exec api python -m pytest tests/ -v
```

| File | What it tests |
|---|---|
| `tests/test_api.py` | API endpoints: auth token, customer CRUD, document upload, brief generation, query, feedback |
| `tests/test_ingestion.py` | Parser and chunker output per doc type: transcript turn boundaries, ticket section splits, commitment field extraction |
| `tests/test_new_modules.py` | Hallucination detection layers, staleness threshold values, circuit breaker state transitions |

Run a single file:

```bash
docker compose exec api python -m pytest tests/test_ingestion.py -v
```

---

## Evals

These scripts measure system quality rather than correctness. Run them after the containers are up and the demo data is seeded.

```bash
docker compose exec api python experiment_kit/experiments/eval_retrieval_modes.py
```
Compares BM25-only, dense-only, and hybrid RRF retrieval on a fixed question set. Reports MRR and top-k hit rate per mode.

```bash
docker compose exec api python experiment_kit/experiments/eval_chunking_strategies.py
```
Tests parent-child chunking vs flat chunking on retrieval recall. Measures whether returning the parent chunk improves answer quality.

```bash
docker compose exec api python experiment_kit/experiments/eval_hallucination_layers.py
```
Runs the three-layer detection pipeline (regex → classifier → LLM judge) against synthetic claims and reports how many are correctly flagged at each layer.

```bash
docker compose exec api python experiment_kit/experiments/eval_latency_per_node.py
```
Profiles each LangGraph node in the pre-meeting brief workflow. Reports p50/p95 latency and identifies which nodes dominate wall-clock time.

```bash
docker compose exec api python experiment_kit/experiments/eval_cost_estimate.py
```
Estimates token cost per brief and per query by doc count. Useful for capacity planning before enabling `CONTEXTUAL_RETRIEVAL`.

```bash
docker compose exec api python experiment_kit/experiments/eval_cost_real.py
```
Same as above but measures actual API token usage from live calls against the Gemini billing API.

```bash
docker compose exec api python experiment_kit/experiments/eval_brief_types.py
```
Compares pre-meeting brief vs exec 1:1 brief output quality on the same customer corpus. Reports section completeness and citation density.

```bash
docker compose exec api python experiment_kit/experiments/eval_query_endpoint.py
```
Runs a fixed query set against the `/query` endpoint and reports answer status distribution (ok/partial/not_found) and mean sources searched.

```bash
docker compose exec api python experiment_kit/experiments/eval_workflow_loop.py
```
Stress-tests the LangGraph workflow with concurrent invocations to surface any state accumulation or race conditions.

```bash
docker compose exec api python experiment_kit/experiments/analyze_node_timings.py
```
Post-processes latency logs from `eval_latency_per_node.py` to produce a per-node timing breakdown chart.
