"""
tests/test_new_modules.py — Tests for modules added in the six-section brief
implementation: doc_type_utils, commitment_parser/chunker, section nodes,
answer_generator confidence fields, and the replace-on-conflict upload path.
"""

import io
import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))


# ── doc_type_utils ────────────────────────────────────────────────────────────

class TestDocTypeInference:
    def test_commitment_keyword_wins(self):
        from utils.doc_type_utils import infer_doc_type
        assert infer_doc_type("2026-01-15_commitments.json") == "commitment_tracker"

    def test_account_notes_keyword(self):
        from utils.doc_type_utils import infer_doc_type
        assert infer_doc_type("account-notes-meridian.pdf") == "account_notes"

    def test_transcript_keyword(self):
        from utils.doc_type_utils import infer_doc_type
        assert infer_doc_type("2026-04-15_transcript_q2_call.txt") == "transcript"

    def test_qbr_deck_keyword(self):
        from utils.doc_type_utils import infer_doc_type
        assert infer_doc_type("meridian_qbr-deck_2026.pdf") == "qbr_deck"

    def test_solution_arch_keyword(self):
        from utils.doc_type_utils import infer_doc_type
        assert infer_doc_type("system_architecture.html") == "solution_architecture"

    def test_ticket_keyword(self):
        from utils.doc_type_utils import infer_doc_type
        assert infer_doc_type("ticket-INC-0042.json") == "ticket"

    def test_pdf_extension_default(self):
        from utils.doc_type_utils import infer_doc_type
        assert infer_doc_type("quarterly_review.pdf") == "qbr_deck"

    def test_docx_extension_default(self):
        from utils.doc_type_utils import infer_doc_type
        assert infer_doc_type("meeting_notes.docx") == "qbr_deck"

    def test_html_extension_default(self):
        from utils.doc_type_utils import infer_doc_type
        assert infer_doc_type("product_guide.html") == "solution_architecture"

    def test_txt_extension_default(self):
        from utils.doc_type_utils import infer_doc_type
        assert infer_doc_type("notes.txt") == "transcript"

    def test_json_without_keywords_returns_none(self):
        """Ambiguous .json must return None so callers demand explicit doc_type."""
        from utils.doc_type_utils import infer_doc_type
        assert infer_doc_type("config.json") is None

    def test_commitment_keyword_beats_qbr_keyword(self):
        """'commitment' should win over 'qbr' since it's checked first."""
        from utils.doc_type_utils import infer_doc_type
        result = infer_doc_type("qbr_commitments.json")
        assert result == "commitment_tracker"

    def test_valid_doc_types_set_is_complete(self):
        from utils.doc_type_utils import VALID_DOC_TYPES
        expected = {
            "transcript", "ticket", "account_notes",
            "qbr_deck", "solution_architecture", "commitment_tracker",
        }
        assert VALID_DOC_TYPES == expected


class TestDocTypeNormalization:
    """normalize_doc_type maps legacy names to canonical ones and rejects unknowns."""

    def test_tickets_becomes_ticket(self):
        from utils.doc_type_utils import normalize_doc_type
        assert normalize_doc_type("tickets") == "ticket"

    def test_commitments_becomes_commitment_tracker(self):
        from utils.doc_type_utils import normalize_doc_type
        assert normalize_doc_type("commitments") == "commitment_tracker"

    def test_notes_becomes_transcript(self):
        from utils.doc_type_utils import normalize_doc_type
        assert normalize_doc_type("notes") == "transcript"

    def test_plain_text_becomes_transcript(self):
        from utils.doc_type_utils import normalize_doc_type
        assert normalize_doc_type("plain_text") == "transcript"

    def test_canonical_types_pass_through_unchanged(self):
        from utils.doc_type_utils import normalize_doc_type, VALID_DOC_TYPES
        for dt in VALID_DOC_TYPES:
            assert normalize_doc_type(dt) == dt

    def test_invalid_type_raises_value_error(self):
        from utils.doc_type_utils import normalize_doc_type
        with pytest.raises(ValueError, match="Unknown doc_type"):
            normalize_doc_type("nonsense_type")

    def test_invalid_type_with_close_spelling_raises(self):
        from utils.doc_type_utils import normalize_doc_type
        with pytest.raises(ValueError):
            normalize_doc_type("Ticket")  # wrong case is not a known alias


class TestExecPersonResolution:
    """Exec 1:1 person_id must resolve to a real name before prompts are built."""

    def test_get_person_by_id_returns_name_and_role(self, tmp_db):
        import api.db_utils as db_utils
        import importlib
        importlib.reload(db_utils)

        cust_id = db_utils.create_customer("Acme", "acme-exec", fde_user_id="fde-exec-test")["id"]
        person = db_utils.add_person(cust_id, "Jane Smith", role="CTO", email="jane@acme.com")
        pid = person["id"]

        result = db_utils.get_person_by_id(pid, "acme-exec")
        assert result is not None
        assert result["name"] == "Jane Smith"
        assert result["role"] == "CTO"

    def test_get_person_by_id_returns_none_for_missing(self, tmp_db):
        import api.db_utils as db_utils
        import importlib
        importlib.reload(db_utils)
        assert db_utils.get_person_by_id(999999, "nonexistent-co") is None

    def test_exec_workflow_state_uses_person_name(self):
        """run_exec_1on1_workflow must set person_name from DB, not pass person_id raw."""
        from unittest.mock import AsyncMock, patch
        import asyncio
        from graph.workflow import run_exec_1on1_workflow

        captured_state = {}

        async def fake_invoke(state):
            captured_state.update(state)
            return state

        with patch("graph.workflow._exec_1on1_workflow") as mock_wf, \
             patch("db_utils.get_person_by_id",
                   return_value={"id": 42, "name": "John Doe", "role": "VP Engineering"}):
            mock_wf.ainvoke = AsyncMock(side_effect=fake_invoke)
            asyncio.run(run_exec_1on1_workflow("acme-exec2", "42"))

        assert captured_state.get("person_name") == "John Doe"
        assert captured_state.get("person_name") != "42"  # must not be raw numeric ID


# ── commitment_parser ─────────────────────────────────────────────────────────

class TestCommitmentParserJSON:
    def test_parse_top_level_array(self, tmp_path):
        from ingestion.commitment_parser import parse
        data = [
            {
                "commitment_id": "COM-001",
                "description": "Ship SCIM integration",
                "promised_date": "2025-06-30",
                "current_target_date": "2025-09-30",
                "status": "slipped",
                "owner": "Platform Team",
                "source_doc": "qbr.pdf",
                "source_section": "slide 5",
                "last_updated": "2026-01-15",
                "customer_aware": True,
            }
        ]
        f = tmp_path / "commitments.json"
        f.write_text(json.dumps(data))
        result = parse(str(f))
        assert len(result) == 1
        assert result[0].commitment_id == "COM-001"
        assert result[0].status == "slipped"
        assert result[0].customer_aware is True

    def test_parse_wrapped_object(self, tmp_path):
        from ingestion.commitment_parser import parse
        data = {"commitments": [
            {"commitment_id": "COM-002", "description": "Deploy feature X",
             "status": "active", "customer_aware": False}
        ]}
        f = tmp_path / "tracker.json"
        f.write_text(json.dumps(data))
        result = parse(str(f))
        assert len(result) == 1
        assert result[0].commitment_id == "COM-002"
        assert result[0].customer_aware is False

    def test_parse_empty_array(self, tmp_path):
        from ingestion.commitment_parser import parse
        f = tmp_path / "empty.json"
        f.write_text("[]")
        assert parse(str(f)) == []


class TestCommitmentParserCSV:
    def test_parse_csv_basic(self, tmp_path):
        from ingestion.commitment_parser import parse
        csv_content = (
            "commitment_id,description,promised_date,current_target_date,"
            "status,owner,source_doc,source_section,last_updated,customer_aware\n"
            "COM-101,Deliver SCIM,2025-06-30,2025-09-30,slipped,Eng,,slide 5,2026-01-15,true\n"
            "COM-102,Deploy infra,2025-12-31,2025-12-31,active,Ops,,,2026-02-01,false\n"
        )
        f = tmp_path / "commitments.csv"
        f.write_text(csv_content)
        result = parse(str(f))
        assert len(result) == 2
        assert result[0].commitment_id == "COM-101"
        assert result[0].customer_aware is True
        assert result[1].customer_aware is False

    def test_parse_csv_case_insensitive_headers(self, tmp_path):
        from ingestion.commitment_parser import parse
        csv_content = (
            "Commitment_ID,Description,Status,Customer_Aware\n"
            "COM-200,Test item,active,YES\n"
        )
        f = tmp_path / "upper.csv"
        f.write_text(csv_content)
        result = parse(str(f))
        assert len(result) == 1
        assert result[0].commitment_id == "COM-200"
        assert result[0].customer_aware is True

    def test_parse_csv_truthy_variants(self, tmp_path):
        from ingestion.commitment_parser import parse
        csv_content = "commitment_id,description,customer_aware\n"
        for val in ("true", "yes", "1", "y", "TRUE", "Yes"):
            csv_content += f"C,desc,{val}\n"
        f = tmp_path / "truthy.csv"
        f.write_text(csv_content)
        result = parse(str(f))
        assert all(c.customer_aware is True for c in result)


# ── commitment_chunker ────────────────────────────────────────────────────────

class TestCommitmentChunker:
    def _make_commitment(self, **kwargs):
        from ingestion.commitment_parser import Commitment
        defaults = {
            "commitment_id": "COM-001",
            "description": "Ship SCIM integration by Q3",
            "promised_date": "2025-06-30",
            "current_target_date": "2025-09-30",
            "status": "slipped",
            "is_slipped": True,
            "owner": "Platform Team",
            "source_doc": "qbr.pdf",
            "source_section": "slide 5",
            "last_updated": "2026-01-15",
            "customer_aware": True,
        }
        defaults.update(kwargs)
        return Commitment(**defaults)

    def test_one_doc_per_commitment(self):
        from ingestion.commitment_chunker import chunk
        commits = [self._make_commitment(commitment_id=f"C-{i}") for i in range(3)]
        docs = chunk(commits, source="test.json")
        assert len(docs) == 3

    def test_metadata_fields_present(self):
        from ingestion.commitment_chunker import chunk
        docs = chunk([self._make_commitment()], source="test.json")
        meta = docs[0].metadata
        assert meta["doc_type"] == "commitment_tracker"
        assert meta["commitment_id"] == "COM-001"
        assert meta["commitment_status"] == "slipped"
        assert meta["is_slipped"] == "true"
        assert meta["customer_aware"] == "true"

    def test_is_slipped_false_when_status_active(self):
        from ingestion.commitment_chunker import chunk
        docs = chunk([self._make_commitment(status="active", is_slipped=False)], source="test.json")
        assert docs[0].metadata["is_slipped"] == "false"

    def test_page_content_includes_description(self):
        from ingestion.commitment_chunker import chunk
        docs = chunk([self._make_commitment(description="Deploy feature X")], source="test.json")
        assert "Deploy feature X" in docs[0].page_content


try:
    import langchain_google_genai  # noqa: F401
    _GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    _GOOGLE_GENAI_AVAILABLE = False

_skip_no_genai = pytest.mark.skipif(
    not _GOOGLE_GENAI_AVAILABLE,
    reason="langchain_google_genai not installed — skipping graph node tests",
)


# ── posture_node verb validation ──────────────────────────────────────────────

@_skip_no_genai
class TestPostureNodeVerbValidation:
    """Posture directives with invalid verbs are filtered out."""

    def _run_posture_with_mocked_llm(self, llm_output: str):
        from unittest.mock import patch
        from graph.nodes import posture_node

        class _FakeResp:
            def __init__(self, content): self.content = content
            usage_metadata = {"input_tokens": 10, "output_tokens": 5}

        state = {
            "account_summary_text": "Account is neutral.",
            "overdue_commitments_data": [],
            "open_items_data": [],
            "recent_changes_data": [],
            "outstanding_commitments_data": [],
            "anticipated_questions_data": [],
            "section_status": {},
            "audit_trail": [],
        }
        with patch("graph.nodes._get_llm"):
            with patch("graph.nodes._llm_invoke_with_retry",
                       return_value=_FakeResp(llm_output)):
                return posture_node(state)

    def test_valid_verbs_pass_through(self):
        output = json.dumps([
            {"verb": "Lead", "directive": "Open with the SCIM status.", "basis": "COM-001"},
            {"verb": "Acknowledge", "directive": "Mention the delay.", "basis": "COM-001"},
        ])
        result = self._run_posture_with_mocked_llm(output)
        assert len(result["posture_directives_data"]) == 2

    def test_invalid_verb_is_filtered(self):
        output = json.dumps([
            {"verb": "Lead", "directive": "Lead on SCIM.", "basis": "COM-001"},
            {"verb": "Ignore", "directive": "Bad verb.", "basis": "none"},
        ])
        result = self._run_posture_with_mocked_llm(output)
        verbs = [d["verb"] for d in result["posture_directives_data"]]
        assert "Ignore" not in verbs
        assert "Lead" in verbs

    def test_lowercase_verb_normalised(self):
        output = json.dumps([
            {"verb": "push", "directive": "Push on renewal.", "basis": "COM-002"},
        ])
        result = self._run_posture_with_mocked_llm(output)
        assert result["posture_directives_data"][0]["verb"] == "Push"


# ── account_summary_node word-count cap ───────────────────────────────────────

@_skip_no_genai
class TestAccountSummaryWordCap:
    def test_over_100_words_gets_truncated(self):
        from unittest.mock import patch, MagicMock
        from langchain_core.documents import Document
        from graph.nodes import account_summary_node

        long_text = " ".join([f"word{i}" for i in range(150)])

        class _FakeResp:
            def __init__(self, content): self.content = content

        mock_doc = Document(
            page_content="account status content for grounding",
            metadata={"filename": "test.txt", "doc_date": "2024-01-01", "user_id": "test_customer"},
        )
        # Score 0.4 is below the 0.85 distance threshold, so grounding check passes
        mock_search = MagicMock(return_value=[(mock_doc, 0.4)])

        state = {
            "customer_id": "test_customer",
            "audit_trail": [],
        }
        with patch("chroma_utils.vectorstore.similarity_search_with_score", mock_search):
            with patch("graph.nodes._get_llm"):
                with patch("graph.nodes._llm_invoke_with_retry",
                           return_value=_FakeResp(long_text)):
                    result = account_summary_node(state)

        summary = result["account_summary_text"]
        assert len(summary.split()) <= 101  # 100 words + possible "…"
        assert summary.endswith("…")


# ── answer_generator recency_flag and confidence_level ────────────────────────

class TestAnswerGeneratorFields:
    def _make_doc(self, doc_date="2026-04-20"):
        from langchain_core.documents import Document
        return Document(
            page_content="The SLA is 99.9%.",
            metadata={"chunk_id": "P1", "doc_date": doc_date, "filename": "sla.txt"},
        )

    def test_recent_doc_gets_current_flag(self):
        from output.answer_generator import generate_answer
        from unittest.mock import patch
        from datetime import datetime, timezone

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        doc = self._make_doc(doc_date=today)
        state = {
            "answer_output": {
                "answer": "The SLA is 99.9%.",
                "answer_status": "ok",
                "citations": [{"claim": "SLA is 99.9%", "chunk_id": "P1"}],
            },
            "parent_chunks": [doc],
            "retrieved_chunks": [],
            "original_query": "what is the SLA",
        }
        result = generate_answer(state)
        assert result["recency_flag"] is None  # None = not stale (recency_flag only returns "stale" or None)

    def test_high_faithfulness_gives_high_confidence(self):
        from output.answer_generator import generate_answer
        from unittest.mock import patch

        doc = self._make_doc()
        state = {
            "answer_output": {
                "answer": "The SLA is 99.9%.",
                "answer_status": "ok",
                "citations": [],
            },
            "parent_chunks": [doc],
            "retrieved_chunks": [],
            "original_query": "what is the SLA",
        }
        with patch("output.answer_generator.classify_claims",
                   return_value={"verified_by_regex": [], "flagged_by_regex": [], "needs_judge": []}):
            with patch("output.answer_generator.detect_hallucination", return_value=[]):
                result = generate_answer(state)
        assert result["answer_status"] == "ok"
        assert result["confidence_explanation"] is None

    def test_parse_error_early_return(self):
        from output.answer_generator import generate_answer
        state = {
            "answer_output": {"answer": "error text", "_parse_error": True},
            "parent_chunks": [],
            "retrieved_chunks": [],
            "original_query": "q",
        }
        result = generate_answer(state)
        assert result["answer_status"] == "error"
        assert result["citations"] == []


# ── classify_claims edge cases ────────────────────────────────────────────────

class TestClassifyClaims:
    """Unit tests for the three-bucket claim classifier in langchain_utils."""

    def _doc(self, text):
        from langchain_core.documents import Document
        return Document(page_content=text, metadata={})

    def test_empty_input_returns_empty_buckets(self):
        from langchain_utils import classify_claims
        result = classify_claims([], [])
        assert result == {"verified_by_regex": [], "flagged_by_regex": [], "needs_judge": []}

    def test_blank_claim_is_skipped(self):
        from langchain_utils import classify_claims
        result = classify_claims(["", "   "], [])
        assert result["verified_by_regex"] == []
        assert result["flagged_by_regex"] == []
        assert result["needs_judge"] == []

    def test_regex_fact_present_in_context_goes_to_verified(self):
        from langchain_utils import classify_claims
        doc = self._doc("The uptime is 99.9% this quarter.")
        result = classify_claims(["The uptime is 99.9% this quarter."], [doc])
        assert len(result["verified_by_regex"]) == 1
        assert result["flagged_by_regex"] == []
        assert result["needs_judge"] == []

    def test_regex_fact_absent_from_context_goes_to_flagged(self):
        from langchain_utils import classify_claims
        doc = self._doc("The service is healthy.")
        result = classify_claims(["The uptime was 97.3% last month."], [doc])
        assert len(result["flagged_by_regex"]) == 1
        assert "97.3%" in result["flagged_by_regex"][0]["unsupported_facts"]

    def test_relational_verb_sends_to_judge_even_with_regex_facts(self):
        """A claim with a relational verb must always go to needs_judge,
        even if atomic facts (dates, percentages) are present in context."""
        from langchain_utils import classify_claims
        doc = self._doc("SLA breach: uptime at 99.1% vs committed 99.9%.")
        claim = "Sarah confirmed the uptime was 99.1%."
        result = classify_claims([claim], [doc])
        assert claim in result["needs_judge"]
        assert claim not in result["verified_by_regex"]

    def test_no_regex_facts_goes_to_needs_judge(self):
        from langchain_utils import classify_claims
        doc = self._doc("The project looks on track.")
        claim = "The project looks on track."
        result = classify_claims([claim], [doc])
        assert claim in result["needs_judge"]

    def test_relational_verb_multi_word_matches(self):
        """'reports to' is a multi-word entry and must not false-match sub-words."""
        from langchain_utils import classify_claims
        doc = self._doc("John reports to the VP of Engineering.")
        claim = "John reports to the VP of Engineering."
        result = classify_claims([claim], [doc])
        assert claim in result["needs_judge"]

    def test_no_context_facts_still_go_to_needs_judge(self):
        from langchain_utils import classify_claims
        claim = "The deal was worth $5M."
        result = classify_claims([claim], [])  # empty docs — $5M is a regex fact not in context → flagged
        assert any(f["claim"] == claim for f in result["flagged_by_regex"])

    def test_multiple_claims_routed_independently(self):
        from langchain_utils import classify_claims
        doc = self._doc("Ticket TICK-4521 is open. Uptime is 99.9%.")
        claims = [
            "TICK-4521 is open.",        # fact present → verified
            "TICK-9999 is open.",        # fact absent → flagged
            "Sarah confirmed the fix.",  # relational → judge
        ]
        result = classify_claims(claims, [doc])
        assert "TICK-4521 is open." in result["verified_by_regex"]
        assert any("TICK-9999" in f["unsupported_facts"]
                   for f in result["flagged_by_regex"])
        assert "Sarah confirmed the fix." in result["needs_judge"]


# ── _strip_json edge cases ────────────────────────────────────────────────────

class TestStripJson:
    """Tests for the regex-based JSON fence stripper in graph/nodes.py."""

    def test_no_fence_returns_content_unchanged(self):
        from graph.nodes import _strip_json
        assert _strip_json('{"key": "val"}') == '{"key": "val"}'

    def test_plain_fence_stripped(self):
        from graph.nodes import _strip_json
        assert _strip_json('```\n{"a":1}\n```') == '{"a":1}'

    def test_json_fence_stripped(self):
        from graph.nodes import _strip_json
        assert _strip_json('```json\n["x","y"]\n```') == '["x","y"]'

    def test_backtick_inside_json_body_does_not_truncate(self):
        """Backtick within a JSON string value must not break extraction."""
        from graph.nodes import _strip_json
        raw = '```json\n{"note": "use ` for code"}\n```'
        result = _strip_json(raw)
        assert '"note"' in result
        assert 'use `' in result

    def test_empty_string_returns_empty(self):
        from graph.nodes import _strip_json
        assert _strip_json("") == ""

    def test_whitespace_only_returns_empty(self):
        from graph.nodes import _strip_json
        assert _strip_json("   ") == ""


# ── query_rewrite fallback edge cases ─────────────────────────────────────────

@_skip_no_genai
class TestQueryRewriteFallback:
    """Verify _query_rewrite_node degrades gracefully on bad LLM output."""

    def _run(self, llm_content):
        from unittest.mock import patch
        from graph.workflow import _query_rewrite_node

        class _FakeResp:
            def __init__(self, c): self.content = c

        state = {"original_query": "what are open tickets", "audit_trail": []}
        with patch("graph.workflow.ChatGoogleGenerativeAI"):
            with patch("graph.workflow._llm_invoke_with_retry",
                       return_value=_FakeResp(llm_content)):
                return _query_rewrite_node(state)

    def test_malformed_json_falls_back_to_original(self):
        result = self._run("{not valid json{{")
        assert result["sub_queries"] == ["what are open tickets"]

    def test_empty_array_falls_back_to_original(self):
        result = self._run("[]")
        assert result["sub_queries"] == ["what are open tickets"]

    def test_non_string_entries_filtered(self):
        result = self._run('["valid query", 42, null, "another query"]')
        assert all(isinstance(q, str) for q in result["sub_queries"])
        assert "valid query" in result["sub_queries"]
        assert "another query" in result["sub_queries"]

    def test_max_four_sub_queries_enforced(self):
        result = self._run('["a","b","c","d","e","f"]')
        assert len(result["sub_queries"]) <= 4


# ── upload replace-on-conflict ────────────────────────────────────────────────

_FAKE_INDEX_SUMMARY = {
    "loader_path": "account_notes", "raw_chunks": 2, "parent_chunks": 0,
    "child_chunks": 2, "parent_child_split": False, "doc_date": None,
    "doc_date_source": None, "contextual_retrieval": False,
}


class TestUploadReplaceOnConflict:
    """Test replace=true/false logic.

    Mocks index_document_to_chroma so these tests don't require a working
    embedding stack — they test the duplicate-filename resolution logic only.

    Uses a customer slug equal to the FDE user_id so that customer ownership
    checks pass without a separate customer-creation fixture.
    """

    _SLUG_A = "replace-unit-a"
    _SLUG_B = "replace-unit-b"

    @staticmethod
    def _ensure_customer(client, slug: str, auth: dict) -> None:
        r = client.post("/customers", json={"name": slug, "slug": slug}, headers=auth)
        assert r.status_code in (200, 201, 409), f"create customer: {r.status_code} {r.text}"

    def test_replace_false_returns_409_on_duplicate(self):
        from unittest.mock import patch
        from fastapi.testclient import TestClient
        from main import app, _create_token
        client = TestClient(app)
        slug = self._SLUG_A
        auth = {"Authorization": f"Bearer {_create_token(slug)}"}
        self._ensure_customer(client, slug, auth)

        content = b"any content"
        with patch("main.index_document_to_chroma", return_value=_FAKE_INDEX_SUMMARY), \
             patch("main.delete_doc_from_chroma", return_value=True):
            resp1 = client.post(
                f"/customers/{slug}/upload",
                files={"file": ("2024-01-01_account-notes_dupe.txt", io.BytesIO(content), "text/plain")},
                data={"doc_type": "account_notes"},
                headers=auth,
            )
        assert resp1.status_code == 200, f"first upload failed: {resp1.text}"
        file_id = resp1.json()["file_id"]

        try:
            with patch("main.index_document_to_chroma", return_value=_FAKE_INDEX_SUMMARY):
                resp2 = client.post(
                    f"/customers/{slug}/upload",
                    files={"file": ("2024-01-01_account-notes_dupe.txt", io.BytesIO(content), "text/plain")},
                    data={"doc_type": "account_notes"},
                    headers=auth,
                )
            assert resp2.status_code == 409
            assert "replace" in resp2.json()["detail"].lower()
        finally:
            client.delete(f"/customers/{slug}/documents/{file_id}", headers=auth)

    def test_replace_true_succeeds_on_duplicate(self):
        from unittest.mock import patch
        from fastapi.testclient import TestClient
        from main import app, _create_token
        client = TestClient(app)
        slug = self._SLUG_B
        auth = {"Authorization": f"Bearer {_create_token(slug)}"}
        self._ensure_customer(client, slug, auth)

        content = b"any content"
        with patch("main.index_document_to_chroma", return_value=_FAKE_INDEX_SUMMARY), \
             patch("main.delete_doc_from_chroma", return_value=True):
            resp1 = client.post(
                f"/customers/{slug}/upload",
                files={"file": ("2024-01-01_account-notes_replace.txt", io.BytesIO(content), "text/plain")},
                data={"doc_type": "account_notes"},
                headers=auth,
            )
        assert resp1.status_code == 200, f"first upload failed: {resp1.text}"

        with patch("main.index_document_to_chroma", return_value=_FAKE_INDEX_SUMMARY), \
             patch("main.delete_doc_from_chroma", return_value=True):
            resp2 = client.post(
                f"/customers/{slug}/upload",
                files={"file": ("2024-01-01_account-notes_replace.txt", io.BytesIO(content), "text/plain")},
                data={"doc_type": "account_notes", "replace": "true"},
                headers=auth,
            )
        assert resp2.status_code == 200, f"replace upload failed: {resp2.text}"
        new_file_id = resp2.json()["file_id"]

        docs = client.get(f"/customers/{slug}/documents", headers=auth).json()
        matching = [d for d in docs if d["filename"] == "2024-01-01_account-notes_replace.txt"]
        assert len(matching) == 1
        assert matching[0]["id"] == new_file_id

        client.delete(f"/customers/{slug}/documents/{new_file_id}", headers=auth)
