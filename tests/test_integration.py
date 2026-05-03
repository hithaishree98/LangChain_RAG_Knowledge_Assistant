"""
tests/test_integration.py — End-to-end integration tests for the pre-meeting brief pipeline.

Uploads all four Cascadia golden fixture documents once per module, then:
  - Verifies the brief pipeline produces the correct schema and key facts
  - Runs three golden queries and asserts answer_status

LLM calls are mocked via patch so the test runs without API credentials.
The module-scoped fixture uploads documents once; all assertions share it.
"""

import io
import json
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

from main import app, _create_token
from fastapi.testclient import TestClient

# ── Constants ─────────────────────────────────────────────────────────────────

GOLDEN_DOCS = os.path.join(
    os.path.dirname(__file__), "..", "eval", "golden", "cascadia", "docs"
)
TEST_SLUG = "cascadia-integration"
_USER = "integration-fde"
_HDRS = {"Authorization": f"Bearer {_create_token(_USER)}"}

client = TestClient(app)


# ── Mock LLM ──────────────────────────────────────────────────────────────────

class _FakeResp:
    def __init__(self, text: str):
        self.content = text
        self.usage_metadata = {"input_tokens": 50, "output_tokens": 20}


def _mock_llm(llm, prompt, **kw):
    p = str(prompt).lower()

    # query_rewrite_node (workflow.py) — returns sub-queries
    if "query analyzer" in p or "focused" in p and "broad" in p:
        return _FakeResp('["original query"]')

    # open_items_node — returns JSON array of ticket items
    if "open action items" in p:
        return _FakeResp(json.dumps([
            {
                "title": "TICK-4521: us-east-2 deployment failing",
                "status": "open", "priority": "P0", "owner": "eng-team",
                "last_update": "2024-10-12",
                "source_doc": "2024-10-01_tickets_open.csv",
                "doc_date": "2024-10-12",
            },
            {
                "title": "TICK-4601: Uptime below SLA threshold",
                "status": "open", "priority": "P0", "owner": "sre-team",
                "last_update": "2024-10-20",
                "source_doc": "2024-10-01_tickets_open.csv",
                "doc_date": "2024-10-20",
            },
        ]))

    # account_summary_node — returns prose text
    if "account status summary" in p:
        return _FakeResp(
            "This account is at-risk due to an overdue SSO integration commitment "
            "and two open P0 tickets including the us-east-2 deployment failure. "
            "The October 22 call is a critical escalation touchpoint."
        )

    # recent_changes_node — returns JSON array
    if "summarize what changed" in p:
        return _FakeResp(json.dumps([
            {
                "what": "TICK-4601 opened tracking SLA uptime breach",
                "date": "2024-10-01",
                "source_doc": "2024-10-01_tickets_open.csv",
                "customer_aware": True,
            }
        ]))

    # anticipated_questions_node — returns JSON array
    if "customer is likely to raise" in p:
        return _FakeResp(json.dumps([
            {
                "topic": "SSO integration status",
                "evidence": "SSO commitment due Oct 1 is overdue",
                "source_quote": "The SSO work is scheduled for completion by October 1st. That's a firm commitment.",
                "source_doc": "2024-09-15_transcript_status-call.txt",
                "urgency": "high",
            },
            {
                "topic": "us-east-2 deployment resolution",
                "evidence": "TICK-4521 still open P0",
                "source_quote": "We need this resolved by end of month",
                "source_doc": "2024-09-15_transcript_status-call.txt",
                "urgency": "high",
            },
        ]))

    # posture_node — returns JSON array of directives
    if "recommended posture" in p:
        return _FakeResp(json.dumps([
            {
                "verb": "Acknowledge",
                "directive": "Open with SSO integration delay — Oct 1 commitment is now overdue",
                "basis": "Customer was told Oct 1 is a firm commitment",
                "grounding_item": "SSO integration delivery",
            },
            {
                "verb": "Acknowledge",
                "directive": "Address TICK-4521 us-east-2 deployment — open P0 since September 1",
                "basis": "Customer flagged this as blocking production deploy",
                "grounding_item": "TICK-4521: us-east-2 deployment failing",
            },
        ]))

    # answer_node — /query path
    if "you answer factual questions" in p or "role:" in p:
        # Check specific question phrases first (the query text is embedded verbatim in the prompt)
        if "sla uptime" in p or "uptime commitment" in p:
            return _FakeResp(json.dumps({
                "answer": "As of 2024-09-15: The customer's SLA requires 99.9% uptime; they reported 99.1% this quarter [C1].",
                "answer_status": "ok",
                "answer_date": "2024-09-15",
                "citations": [{"claim": "SLA 99.9%, actual 99.1%", "chunk_id": "C1", "date": "2024-09-15"}],
            }))
        if "status of tick-4521" in p:
            return _FakeResp(json.dumps({
                "answer": "As of 2024-10-12: TICK-4521 is open at P0, tracking us-east-2 deployment failures with timeout errors [C3].",
                "answer_status": "ok",
                "answer_date": "2024-10-12",
                "citations": [{"claim": "TICK-4521 open P0", "chunk_id": "C3", "date": "2024-10-12"}],
            }))
        if "sso integration due" in p or "when is the sso" in p or "sso" in p:
            return _FakeResp(json.dumps({
                "answer": "As of 2024-10-15: The SSO integration was committed for October 1, 2024 [C5]. It is currently overdue.",
                "answer_status": "ok",
                "answer_date": "2024-10-15",
                "citations": [{"claim": "SSO due Oct 1", "chunk_id": "C5", "date": "2024-10-15"}],
            }))
        if "uptime" in p or "sla" in p:
            return _FakeResp(json.dumps({
                "answer": "As of 2024-09-15: The customer's SLA requires 99.9% uptime; they reported 99.1% this quarter [C1].",
                "answer_status": "ok",
                "answer_date": "2024-09-15",
                "citations": [{"claim": "SLA 99.9%, actual 99.1%", "chunk_id": "C1", "date": "2024-09-15"}],
            }))
        if "tick-4521" in p:
            return _FakeResp(json.dumps({
                "answer": "As of 2024-10-12: TICK-4521 is open at P0, tracking us-east-2 deployment failures with timeout errors [C3].",
                "answer_status": "ok",
                "answer_date": "2024-10-12",
                "citations": [{"claim": "TICK-4521 open P0", "chunk_id": "C3", "date": "2024-10-12"}],
            }))

    return _FakeResp(json.dumps({
        "answer": "",
        "answer_status": "not_found",
        "citations": [],
    }))


# ── Module-scoped setup ───────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def uploaded_cascadia():
    """Create the integration customer and upload all four golden fixture docs.

    Runs once per module. Returns the customer slug.
    """
    # Create customer (idempotent-ish — if slug exists from a prior run in the
    # same session, ignore the 409 and proceed with the existing record)
    r = client.post(
        "/customers",
        headers=_HDRS,
        json={"name": "Cascadia Integration", "slug": TEST_SLUG},
    )
    assert r.status_code in (200, 201, 409), f"create_customer failed: {r.status_code} {r.text}"

    uploads = [
        # (filename, doc_type_override)
        ("2024-09-15_transcript_status-call.txt", None),          # inferred → transcript
        ("2024-10-01_tickets_open.csv",            "ticket"),      # explicit: ticket
        ("2024-10-15_commitments_tracker.csv",     "commitment_tracker"),  # explicit
        ("2024-09-15_notes_status-call.txt",       "account_notes"),       # explicit: not transcript
    ]

    for fname, doc_type in uploads:
        fpath = os.path.join(GOLDEN_DOCS, fname)
        assert os.path.exists(fpath), f"Missing fixture: {fpath}"

        with open(fpath, "rb") as fh:
            content = fh.read()

        form_data = {}
        if doc_type:
            form_data["doc_type"] = doc_type

        r = client.post(
            f"/customers/{TEST_SLUG}/upload",
            headers=_HDRS,
            files={"file": (fname, io.BytesIO(content), "text/plain")},
            data=form_data,
        )
        assert r.status_code == 200, f"Upload failed for {fname}: {r.status_code} {r.text}"
        body = r.json()
        assert body.get("chunks", 0) > 0, f"No chunks indexed for {fname}"

    return TEST_SLUG


@pytest.fixture(scope="module")
def cascadia_brief(uploaded_cascadia):
    """Generate a pre-meeting brief with mocked LLM. Runs once per module."""
    with patch("graph.nodes._llm_invoke_with_retry", _mock_llm), \
         patch("graph.workflow._llm_invoke_with_retry", _mock_llm):
        r = client.post(
            "/brief/pre-meeting",
            headers=_HDRS,
            json={"customer_id": uploaded_cascadia},
        )
    assert r.status_code == 200, f"Brief failed: {r.status_code} {r.text}"
    return r.json()


# ── Brief structure assertions ────────────────────────────────────────────────

class TestBriefStructure:
    def test_has_required_top_level_keys(self, cascadia_brief):
        required = {
            "overdue_commitments", "open_items", "account_summary",
            "outstanding_commitments", "anticipated_questions",
            "recommended_posture", "as_of_date", "section_status",
        }
        assert required.issubset(set(cascadia_brief.keys()))

    def test_section_status_no_unavailable(self, cascadia_brief):
        statuses = cascadia_brief.get("section_status", {})
        unavailable = [k for k, v in statuses.items() if v == "unavailable"]
        assert not unavailable, f"Sections unavailable: {unavailable}"

    def test_as_of_date_is_string(self, cascadia_brief):
        assert isinstance(cascadia_brief.get("as_of_date"), str)
        assert len(cascadia_brief["as_of_date"]) == 10  # YYYY-MM-DD


class TestOverdueCommitments:
    def test_sso_commitment_is_overdue(self, cascadia_brief):
        """SSO integration promised 2024-10-01 is past today — must appear overdue."""
        overdue = cascadia_brief.get("overdue_commitments", [])
        assert overdue, "No overdue commitments found"
        descriptions = " ".join(c.get("description", "").lower() for c in overdue)
        assert "sso" in descriptions, f"SSO not in overdue: {descriptions}"

    def test_overdue_commitment_is_flagged(self, cascadia_brief):
        overdue = cascadia_brief.get("overdue_commitments", [])
        for c in overdue:
            if "sso" in c.get("description", "").lower():
                assert c.get("is_overdue") is True
                return
        pytest.fail("SSO commitment not found in overdue list")


class TestOpenItems:
    def test_p0_tickets_present(self, cascadia_brief):
        items = cascadia_brief.get("open_items", [])
        assert items, "No open items returned"
        titles = " ".join(i.get("title", "") for i in items)
        assert "TICK-4521" in titles, f"TICK-4521 missing from open items: {titles}"
        assert "TICK-4601" in titles, f"TICK-4601 missing from open items: {titles}"

    def test_open_items_have_priority(self, cascadia_brief):
        items = cascadia_brief.get("open_items", [])
        for item in items:
            assert "priority" in item, f"Item missing priority: {item}"


class TestAccountSummary:
    def test_at_risk_mentioned(self, cascadia_brief):
        summary = cascadia_brief.get("account_summary", "")
        assert "at-risk" in summary.lower(), f"Expected 'at-risk' in account_summary: {summary}"

    def test_summary_is_prose(self, cascadia_brief):
        summary = cascadia_brief.get("account_summary", "")
        assert isinstance(summary, str) and len(summary) > 20


class TestRecommendedPosture:
    def test_posture_not_empty(self, cascadia_brief):
        posture = cascadia_brief.get("recommended_posture", [])
        assert posture, "No posture directives returned"

    def test_acknowledge_verb_present(self, cascadia_brief):
        posture = cascadia_brief.get("recommended_posture", [])
        verbs = {d.get("verb") for d in posture}
        assert "Acknowledge" in verbs, f"Acknowledge not in posture verbs: {verbs}"

    def test_posture_directives_have_required_fields(self, cascadia_brief):
        for d in cascadia_brief.get("recommended_posture", []):
            assert "verb" in d
            assert "directive" in d
            assert "basis" in d
            assert d["verb"] in {"Lead", "Acknowledge", "Defer", "Push"}


class TestAnticipatedQuestions:
    def test_sso_topic_anticipated(self, cascadia_brief):
        questions = cascadia_brief.get("anticipated_questions", [])
        topics = " ".join(q.get("topic", "").lower() for q in questions)
        assert "sso" in topics or "integration" in topics, \
            f"SSO not in anticipated topics: {topics}"

    def test_source_quotes_present(self, cascadia_brief):
        questions = cascadia_brief.get("anticipated_questions", [])
        assert questions, "No anticipated questions"
        with_quotes = [q for q in questions if q.get("source_quote")]
        assert with_quotes, "No anticipated questions have source_quote"


# ── Golden query assertions ───────────────────────────────────────────────────

def _query(customer_id: str, question: str) -> dict:
    with patch("graph.nodes._llm_invoke_with_retry", _mock_llm), \
         patch("graph.workflow._llm_invoke_with_retry", _mock_llm):
        r = client.post(
            "/query",
            headers=_HDRS,
            json={"customer_id": customer_id, "question": question},
        )
    assert r.status_code == 200, f"Query failed ({r.status_code}): {r.text}"
    return r.json()


class TestGoldenQueries:
    def test_sso_commitment_query(self, uploaded_cascadia):
        resp = _query(uploaded_cascadia, "When is the SSO integration due?")
        assert resp.get("answer_status") == "ok"
        assert "sso" in resp.get("answer", "").lower() or "october" in resp.get("answer", "").lower()

    def test_uptime_sla_query(self, uploaded_cascadia):
        resp = _query(uploaded_cascadia, "What is the SLA uptime commitment?")
        assert resp.get("answer_status") == "ok"
        assert "99.9" in resp.get("answer", "") or "sla" in resp.get("answer", "").lower()

    def test_tick_4521_query(self, uploaded_cascadia):
        resp = _query(uploaded_cascadia, "What is the status of TICK-4521?")
        assert resp.get("answer_status") == "ok"
        assert "TICK-4521" in resp.get("answer", "") or "us-east-2" in resp.get("answer", "").lower()

    def test_query_returns_citations(self, uploaded_cascadia):
        resp = _query(uploaded_cascadia, "What is the status of TICK-4521?")
        assert resp.get("citations") is not None

    def test_query_has_as_of_date(self, uploaded_cascadia):
        resp = _query(uploaded_cascadia, "What is the status of TICK-4521?")
        # The answer should start with "As of" per the QA prompt spec
        answer = resp.get("answer", "")
        assert answer.startswith("As of"), f"Answer doesn't start with 'As of': {answer[:60]}"
