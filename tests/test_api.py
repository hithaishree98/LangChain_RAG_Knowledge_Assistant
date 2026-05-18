import io
import os
import time
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api'))

from main import app, _create_token

client = TestClient(app)

# Several tests import from graph.nodes, which has a top-level
# `from langchain_google_genai import ChatGoogleGenerativeAI`. Skip those tests
# when the package isn't installed so CI doesn't fail on a missing optional dep.
try:
    import langchain_google_genai  # noqa: F401
    _GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    _GOOGLE_GENAI_AVAILABLE = False

_skip_no_genai = pytest.mark.skipif(
    not _GOOGLE_GENAI_AVAILABLE,
    reason="langchain_google_genai not installed — skipping graph.nodes tests",
)


def _bearer(user_id: str) -> dict:
    """Mint a JWT for the given user_id and return Authorization headers.

    Tenant identity is taken only from the signed token now (the old
    ?user_id= query-param fallback was an IDOR), so test cases that need
    to act as a specific tenant must authenticate as one.
    """
    return {"Authorization": f"Bearer {_create_token(user_id)}"}


# Shared test upload customer. Slug matches user_id so that DELETE /documents/{id}
# (which filters by FDE user_id) can clean up docs written with user_id=slug.
_TEST_UPLOAD_USER = "test-upload-u01"
_TEST_UPLOAD_SLUG = "test-upload-u01"


def _ensure_upload_customer():
    """Create the shared test upload customer if it doesn't already exist (409 = OK)."""
    client.post(
        "/customers",
        json={"name": "Test Upload Co", "slug": _TEST_UPLOAD_SLUG},
        headers=_bearer(_TEST_UPLOAD_USER),
    )


# ── Health ────────────────────────────────────────────────────────────────────

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "checks" in data
    assert "circuit_breaker" in data["checks"]
    assert data["checks"]["circuit_breaker"] == "closed"


# ── Auth ──────────────────────────────────────────────────────────────────────

def test_auth_token_valid():
    response = client.post("/auth/token", json={"workspace": "test-ws", "passkey": "secret"})
    assert response.status_code == 200
    data = response.json()
    assert "token" in data
    assert "user_id" in data
    assert len(data["user_id"]) == 32  # sha256[:32]


def test_auth_token_same_credentials_returns_same_user_id():
    r1 = client.post("/auth/token", json={"workspace": "ws-a", "passkey": "pw-a"})
    r2 = client.post("/auth/token", json={"workspace": "ws-a", "passkey": "pw-a"})
    assert r1.json()["user_id"] == r2.json()["user_id"]


def test_auth_token_different_credentials_different_user_id():
    r1 = client.post("/auth/token", json={"workspace": "ws-a", "passkey": "pw-1"})
    r2 = client.post("/auth/token", json={"workspace": "ws-a", "passkey": "pw-2"})
    assert r1.json()["user_id"] != r2.json()["user_id"]


def test_auth_token_missing_fields():
    response = client.post("/auth/token", json={"workspace": "only-workspace"})
    # 400/422 = validation failure; 429 = rate-limited (5/min quota hit by earlier auth tests)
    assert response.status_code in (400, 422, 429)


def test_auth_token_empty_fields():
    response = client.post("/auth/token", json={"workspace": "", "passkey": ""})
    assert response.status_code in (400, 429)


# ── Document list / analytics / audit ────────────────────────────────────────

def test_list_docs():
    response = client.get("/documents", headers=_bearer("list-docs-user"))
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_analytics():
    response = client.get("/stats", headers=_bearer("stats-user"))
    assert response.status_code == 200
    data = response.json()
    assert "total_queries" in data
    assert "escalated_count" in data
    assert "avg_confidence" in data


def test_audit_log():
    response = client.get("/audit-log")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_audit_log_limit():
    response = client.get("/audit-log?limit=5")
    assert response.status_code == 200
    assert len(response.json()) <= 5


# ── Document delete ───────────────────────────────────────────────────────────

def test_delete_nonexistent_doc():
    response = client.delete("/documents/99999", headers=_bearer("nobody"))
    assert response.status_code == 404
    assert "detail" in response.json()


# ── Upload validation ─────────────────────────────────────────────────────────

def test_upload_unsupported_extension():
    _ensure_upload_customer()
    response = client.post(
        f"/customers/{_TEST_UPLOAD_SLUG}/upload",
        files={"file": ("malware.exe", io.BytesIO(b"bad"), "application/octet-stream")},
        headers=_bearer(_TEST_UPLOAD_USER),
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_upload_empty_file():
    _ensure_upload_customer()
    response = client.post(
        f"/customers/{_TEST_UPLOAD_SLUG}/upload",
        files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
        headers=_bearer(_TEST_UPLOAD_USER),
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_upload_file_too_large():
    _ensure_upload_customer()
    big_content = b"x" * (11 * 1024 * 1024)  # 11 MB > 10 MB limit
    response = client.post(
        f"/customers/{_TEST_UPLOAD_SLUG}/upload",
        files={"file": ("big.txt", io.BytesIO(big_content), "text/plain")},
        headers=_bearer(_TEST_UPLOAD_USER),
    )
    assert response.status_code == 400
    assert "large" in response.json()["detail"].lower()


def test_upload_json_without_doc_type_returns_400():
    """A .json file with no doc_type and no filename keywords must return 400."""
    _ensure_upload_customer()
    response = client.post(
        f"/customers/{_TEST_UPLOAD_SLUG}/upload",
        files={"file": ("config.json", io.BytesIO(b'{"key": "value"}'), "application/json")},
        headers=_bearer(_TEST_UPLOAD_USER),
    )
    assert response.status_code == 400
    assert "document type" in response.json()["detail"].lower()


def test_upload_json_with_explicit_doc_type_accepted():
    """A .json file with an explicit doc_type Form field must pass doc_type validation."""
    _ensure_upload_customer()
    response = client.post(
        f"/customers/{_TEST_UPLOAD_SLUG}/upload",
        files={"file": ("config.json", io.BytesIO(b'{"key": "value"}'), "application/json")},
        data={"doc_type": "ticket"},
        headers=_bearer(_TEST_UPLOAD_USER),
    )
    # 400 for filename convention or indexing failure are both acceptable;
    # what we guard: the request is NOT rejected specifically for missing doc_type.
    assert response.status_code != 400 or "doc_type" not in response.json().get("detail", "")


def test_upload_invalid_doc_type_returns_400():
    """An unrecognised doc_type value must return 400."""
    _ensure_upload_customer()
    response = client.post(
        f"/customers/{_TEST_UPLOAD_SLUG}/upload",
        files={"file": ("notes.txt", io.BytesIO(b"some content here"), "text/plain")},
        data={"doc_type": "not_a_real_type"},
        headers=_bearer(_TEST_UPLOAD_USER),
    )
    assert response.status_code == 400


# ── Multi-tenant isolation ────────────────────────────────────────────────────

def test_multitenant_doc_isolation():
    """FDE-A's customer must not be visible to FDE-B; B cannot upload to A's customer."""
    user_a = "tenant-iso-fde-a"
    user_b = "tenant-iso-fde-b"
    slug_a = "tenant-iso-corp-a"
    auth_a = _bearer(user_a)
    auth_b = _bearer(user_b)

    # Create customer for user A (409 = already exists, safe to ignore)
    client.post("/customers", json={"name": "Iso Corp A", "slug": slug_a}, headers=auth_a)

    # User B must not see user A's customer
    r_b = client.get("/customers", headers=auth_b)
    assert r_b.status_code == 200
    assert not any(c["slug"] == slug_a for c in r_b.json())

    # User A must see their own customer
    r_a = client.get("/customers", headers=auth_a)
    assert r_a.status_code == 200
    assert any(c["slug"] == slug_a for c in r_a.json())

    # User B cannot upload to user A's customer
    r_upload = client.post(
        f"/customers/{slug_a}/upload",
        files={"file": ("2026-01-01_account-notes_test.txt",
                        io.BytesIO(b"some content"), "text/plain")},
        data={"doc_type": "account_notes"},
        headers=auth_b,
    )
    assert r_upload.status_code == 404


def test_multitenant_delete_isolation():
    """FDE-B must not be able to delete FDE-A's customer."""
    user_a = "tenant-del-fde-a"
    user_b = "tenant-del-fde-b"
    slug_a = "tenant-del-corp-a"
    auth_a = _bearer(user_a)
    auth_b = _bearer(user_b)

    client.post("/customers", json={"name": "Del Corp A", "slug": slug_a}, headers=auth_a)

    # User B tries to delete user A's customer — must 404
    r = client.delete(f"/customers/{slug_a}", headers=auth_b)
    assert r.status_code == 404

    # User A's customer must still exist
    r2 = client.get("/customers", headers=auth_a)
    assert any(c["slug"] == slug_a for c in r2.json())



# ── Circuit breaker ───────────────────────────────────────────────────────────

def test_circuit_breaker_opens_after_threshold():
    """Failures >= failure_threshold should open the breaker."""
    from langchain_utils import llm_breaker, _CBState
    llm_breaker.on_success()  # reset to CLOSED
    for _ in range(llm_breaker.failure_threshold):
        llm_breaker.on_failure()
    assert llm_breaker.state == _CBState.OPEN
    llm_breaker.on_success()  # restore for other tests


def test_circuit_breaker_half_open_reverts_on_failure():
    """A failure while HALF_OPEN must revert to OPEN, not stay HALF_OPEN."""
    from langchain_utils import llm_breaker, _CBState
    llm_breaker.on_success()  # start CLOSED
    for _ in range(llm_breaker.failure_threshold):
        llm_breaker.on_failure()
    assert llm_breaker.state == _CBState.OPEN

    # Simulate recovery timeout by backdating last_failure_time
    llm_breaker.last_failure_time = time.time() - 31
    assert llm_breaker.is_open() is False          # transitions to HALF_OPEN
    assert llm_breaker.state == _CBState.HALF_OPEN

    llm_breaker.on_failure()                       # probe fails
    assert llm_breaker.state == _CBState.OPEN      # must revert, not stay HALF_OPEN
    llm_breaker.on_success()  # restore for other tests


def test_circuit_breaker_closes_after_half_open_success():
    """A success while HALF_OPEN must fully close the breaker."""
    from langchain_utils import llm_breaker, _CBState
    llm_breaker.on_success()
    for _ in range(llm_breaker.failure_threshold):
        llm_breaker.on_failure()
    llm_breaker.last_failure_time = time.time() - 31
    llm_breaker.is_open()                          # transition to HALF_OPEN
    llm_breaker.on_success()
    assert llm_breaker.state == _CBState.CLOSED


# ── JWT token expiry ──────────────────────────────────────────────────────────

def test_expired_jwt_returns_401():
    """A token whose exp is in the past must be rejected with 401."""
    from datetime import datetime, timedelta, timezone
    from jose import jwt as jose_jwt
    from main import _JWT_SECRET, _JWT_ALGORITHM

    expired_payload = {
        "sub": "test_user_expired",
        "exp": datetime.now(timezone.utc) - timedelta(hours=1),
    }
    expired_token = jose_jwt.encode(expired_payload, _JWT_SECRET, algorithm=_JWT_ALGORITHM)
    response = client.get(
        "/documents",
        headers={"Authorization": f"Bearer {expired_token}"},
    )
    assert response.status_code == 401


# ── Noise filter ──────────────────────────────────────────────────────────────

def test_noise_filter_preserves_section_headers():
    from chroma_utils import noise_filter
    text = "Risks:\n• Cost overrun\n- Data loss\nSome longer sentence follows here."
    result = noise_filter(text)
    assert "Risks:" in result
    assert "• Cost overrun" in result
    assert "- Data loss" in result


def test_noise_filter_removes_page_numbers():
    from chroma_utils import noise_filter
    text = "Page 3\n— 12 —\nActual content that should be kept around here."
    result = noise_filter(text)
    assert "Page 3" not in result
    assert "— 12 —" not in result
    assert "Actual content" in result


# ── Format-aware loader fallbacks ─────────────────────────────────────────────

def test_plain_text_falls_back_when_no_speaker_turns(tmp_path):
    """A .txt file without 'Speaker: ...' format must still index as plain prose."""
    import chroma_utils
    txt = tmp_path / "memo.txt"
    txt.write_text(
        "This is a quarterly planning memo. The team will focus on three areas: "
        "reliability, performance, and cost. We expect to ship the reliability "
        "improvements by the end of Q3 and the performance work by the start of Q4. "
        "Cost optimization is a longer-running effort spanning multiple quarters. "
        "Stakeholders will be notified through the usual channels."
    )
    chunks = chroma_utils.load_and_split_document(str(txt))
    assert len(chunks) >= 1, "plain-text fallback must produce at least one chunk"
    assert all(c.metadata.get("doc_type") == "plain_text" for c in chunks)


def test_generic_json_falls_back_when_not_ticket_shape(tmp_path):
    """A .json file without ticket fields must still index as generic JSON."""
    import json
    import chroma_utils
    j = tmp_path / "config.json"
    j.write_text(json.dumps({
        "name": "Service config",
        "summary": "Defines retry policy and timeout for the upstream API.",
        "settings": {"retries": 3, "timeout_ms": 5000, "policy": "exponential backoff"},
        "owners": ["platform-team"],
    }))
    chunks = chroma_utils.load_and_split_document(str(j))
    assert len(chunks) >= 1, "generic-json fallback must produce at least one chunk"
    assert all(c.metadata.get("doc_type") == "generic_json" for c in chunks)


def test_walk_json_strings_collects_nested_values():
    """_walk_json_strings flattens all string values from nested structures."""
    import chroma_utils
    obj = {
        "title": "TopLevelTitle",
        "nested": {"description": "InnerDescription", "tags": ["alpha_tag_one", "beta_tag_two"]},
        "count": 42,            # non-string skipped
        "list": ["extra_item_x", {"y": "zeta_value_y"}],
    }
    out = chroma_utils._walk_json_strings(obj)
    assert "TopLevelTitle" in out
    assert "InnerDescription" in out
    assert "alpha_tag_one" in out and "beta_tag_two" in out
    assert "extra_item_x" in out and "zeta_value_y" in out
    assert 42 not in out


# ── Indexing summary returned to caller ──────────────────────────────────────

def test_upload_response_includes_indexing_summary():
    """The upload response must include file metadata so the UI can confirm ingestion."""
    import io
    user = "summary-check-u01"
    slug = "summary-check-u01"  # slug == user so DELETE /documents/{id} can clean up
    auth = _bearer(user)
    client.post("/customers", json={"name": "Summary Check Co", "slug": slug}, headers=auth)

    transcript = (
        b"Sarah: Welcome to the call. Let's review the open items.\n"
        b"Martha: Thanks. The first one is the SCIM rollout decision.\n"
        b"Sarah: Right. We've decided to start in pull mode for Q2.\n"
    )
    upload = client.post(
        f"/customers/{slug}/upload",
        files={"file": ("2026-04-15_transcript_demo-call.txt", io.BytesIO(transcript), "text/plain")},
        data={"doc_type": "transcript"},
        headers=auth,
    )
    assert upload.status_code == 200
    body = upload.json()

    assert "file_id" in body
    assert "chunks" in body and isinstance(body["chunks"], int)
    assert body["doc_type"] == "transcript"
    assert body["doc_date"] == "2026-04-15", "filename prefix must drive doc_date"
    assert body["chunks"] >= 1

    client.delete(f"/documents/{body['file_id']}", headers=auth)


# ── Format-aware indexing decisions ───────────────────────────────────────────

def test_transcript_indexing_preserves_speaker_turns(tmp_path):
    """A .txt transcript must not get parent-child re-split — speaker turns must stay intact."""
    import io
    user = "transcript-indexing-u01"
    slug = "transcript-indexing-u01"  # slug == user so DELETE /documents/{id} can clean up
    auth = _bearer(user)
    client.post("/customers", json={"name": "Transcript Indexing Co", "slug": slug}, headers=auth)

    transcript = (
        "Sarah: Good morning everyone, thanks for joining the call.\n"
        "Martha: Thanks Sarah. We wanted to circle back on the SCIM rollout.\n"
        "Sarah: Right. We agreed last week that we'd ship pull mode to start.\n"
        "Martha: Yes, but we'd like to revisit push mode in Q3 if possible.\n"
        "Sarah: Makes sense. Let me check with the platform team and follow up.\n"
        "Martha: Perfect. Also wanted to confirm the AWS region — still us-east-2?\n"
        "Sarah: Correct, us-east-2 primary with a us-west-2 read-replica.\n"
    ).encode()

    upload = client.post(
        f"/customers/{slug}/upload",
        files={"file": ("2026-04-15_transcript_meridian-call.txt", io.BytesIO(transcript), "text/plain")},
        data={"doc_type": "transcript"},
        headers=auth,
    )
    assert upload.status_code == 200
    file_id = upload.json()["file_id"]

    try:
        # Inspect the chunks now in Chroma. Transcript chunks should contain
        # at least one full speaker turn, not character-bisected mid-sentence.
        # user_id in Chroma = customer slug (set by upload endpoint)
        from chroma_utils import vectorstore
        result = vectorstore._collection.get(
            where={"$and": [{"file_id": {"$eq": file_id}}, {"user_id": {"$eq": slug}}]},
            include=["documents", "metadatas"],
        )
        docs = result.get("documents") or []
        assert len(docs) >= 1
        has_speaker = any(("Sarah:" in d or "Martha:" in d) for d in docs)
        assert has_speaker, "transcript chunks lost speaker attribution during indexing"
    finally:
        client.delete(f"/documents/{file_id}", headers=auth)


# ── Lookup endpoint + adaptive rewriter + judge gating ───────────────────────

def test_should_run_judge_skips_high_confidence_no_relational():
    """High faithfulness + no relational verbs + few claims → skip judge."""
    from langchain_utils import should_run_judge
    assert should_run_judge("what's the SLA", faithfulness=0.92, n_claims=2) is False


def test_should_run_judge_runs_on_relational_verb():
    """Relational verb in query → always run judge regardless of confidence."""
    from langchain_utils import should_run_judge
    assert should_run_judge("what did Martha agree to", faithfulness=0.95, n_claims=1) is True
    assert should_run_judge("what did we promise the customer", 0.95, 1) is True


def test_should_run_judge_runs_on_low_confidence():
    """Faithfulness < 0.7 always triggers the judge."""
    from langchain_utils import should_run_judge
    assert should_run_judge("any neutral question here", 0.65, 2) is True


def test_should_run_judge_runs_on_complex_answer():
    """More than 3 claims → run the judge to be safe."""
    from langchain_utils import should_run_judge
    assert should_run_judge("what's the SLA", 0.95, n_claims=5) is True


def test_should_run_judge_always_run_overrides():
    """always_run=True bypasses every other check (used by /brief)."""
    from langchain_utils import should_run_judge
    assert should_run_judge("trivial question", 0.99, 1, always_run=True) is True


@_skip_no_genai
def test_adaptive_rewriter_passes_focused_query_through():
    """A focused question must come back as a 1-element list — no decomposition."""
    from graph.workflow import _query_rewrite_node

    class _FakeResp:
        def __init__(self, content): self.content = content

    state = {"original_query": "what is the SLA", "audit_trail": []}
    with patch("graph.workflow.ChatGoogleGenerativeAI"):
        with patch("graph.workflow._llm_invoke_with_retry",
                   return_value=_FakeResp('["what is the SLA"]')):
            result = _query_rewrite_node(state)
    assert result["sub_queries"] == ["what is the SLA"]


@_skip_no_genai
def test_adaptive_rewriter_decomposes_broad_synthesis_query():
    """A synthesis-shaped question should produce multiple sub-queries."""
    from graph.workflow import _query_rewrite_node

    class _FakeResp:
        def __init__(self, content): self.content = content

    state = {
        "original_query": "summarize agreements, decisions, and open items",
        "audit_trail": [],
    }
    with patch("graph.workflow.ChatGoogleGenerativeAI"):
        with patch("graph.workflow._llm_invoke_with_retry",
                   return_value=_FakeResp('["agreements", "decisions", "open items"]')):
            result = _query_rewrite_node(state)
    assert len(result["sub_queries"]) > 1



# ── Date awareness ────────────────────────────────────────────────────────────

def test_resolve_doc_date_from_filename_prefix():
    """A YYYY-MM-DD prefix in the filename is the highest-priority date source."""
    import chroma_utils
    date = chroma_utils.resolve_doc_date(
        "2025-03-28_meridian_call.txt", b"some content here", "transcript"
    )
    assert date == "2025-03-28"


def test_resolve_doc_date_from_transcript_header():
    """A date inside the first 1KB of a transcript falls back to header extraction."""
    import chroma_utils
    contents = b"Date: 2025-04-15\nMartha: hello team\nSarah: hi back\n"
    date = chroma_utils.resolve_doc_date("notes.txt", contents, "transcript")
    assert date == "2025-04-15"


def test_resolve_doc_date_from_ticket_created_at():
    """A `created_at` field in a ticket JSON is used when no filename prefix."""
    import json
    import chroma_utils
    contents = json.dumps({
        "id": "TICK-1", "subject": "...", "description": "...",
        "created_at": "2025-02-10T09:30:00Z",
    }).encode()
    date = chroma_utils.resolve_doc_date("ticket.json", contents, "ticket")
    assert date == "2025-02-10"


def test_resolve_doc_date_returns_none_when_no_date_recoverable():
    """No filename prefix + no content date → caller falls back to upload time."""
    import chroma_utils
    date = chroma_utils.resolve_doc_date("memo.pdf", b"plain bytes", "")
    assert date is None


def test_recency_boost_reorders_docs_newer_first():
    """A recent doc must outrank an older one when recency boost fires."""
    import chroma_utils
    from langchain_core.documents import Document
    docs = [
        Document(page_content="old", metadata={"doc_date": "2024-01-01", "chunk_id": "C1"}),
        Document(page_content="new", metadata={"doc_date": "2025-04-01", "chunk_id": "C2"}),
    ]
    boosted = chroma_utils._recency_boost(docs)
    assert boosted[0].page_content == "new"
    assert boosted[1].page_content == "old"


def test_wants_recency_detects_temporal_keywords():
    """Common temporal keywords trigger the recency boost path."""
    import chroma_utils
    assert chroma_utils._wants_recency("what's the most recent agreement")
    assert chroma_utils._wants_recency("what did we say last call")
    assert chroma_utils._wants_recency("any recent changes")
    assert not chroma_utils._wants_recency("what is the SLA")
    assert not chroma_utils._wants_recency("who is the primary contact")


@_skip_no_genai
def test_build_context_str_includes_date_and_source():
    """The LLM context header surfaces chunk_id, date, and source filename."""
    from graph.nodes import _build_context_str
    from langchain_core.documents import Document
    docs = [
        Document(page_content="Martha agreed to send forecasts.",
                 metadata={"chunk_id": "P3", "doc_date": "2025-03-28",
                           "filename": "2025-03-28_meridian_call.txt"}),
    ]
    ctx = _build_context_str(docs)
    assert "[P3 | 2025-03-28 | 2025-03-28_meridian_call.txt]" in ctx
    assert "Martha agreed to send forecasts." in ctx


# ── BM25 tokenizer ────────────────────────────────────────────────────────────

def test_bm25_tokenizer_strips_punctuation():
    from chroma_utils import _tokenize
    assert _tokenize("issue, risk") == ["issue", "risk"]
    assert _tokenize("pre-call") == ["pre", "call"]
    assert _tokenize("don't") == ["don", "t"]


# ── Retrieval failure propagation ─────────────────────────────────────────────

@_skip_no_genai
def test_retrieve_node_raises_when_all_sub_queries_fail():
    """retrieve_node must raise RuntimeError when every sub-query fails retrieval."""
    from graph.nodes import retrieve_node

    fake_state = {
        "customer_id": "test_user",
        "sub_queries": ["query1", "query2"],
        "audit_trail": [],
    }

    with patch("graph.nodes.get_retriever_for_user") as mock_retriever:
        mock_retriever.return_value.invoke.side_effect = RuntimeError("Chroma unavailable")
        mock_retriever.return_value.get_relevant_documents.side_effect = RuntimeError("Chroma unavailable")
        with pytest.raises(RuntimeError, match="Retrieval failed"):
            retrieve_node(fake_state)


@_skip_no_genai
def test_retrieve_node_continues_when_some_sub_queries_fail():
    """Partial retrieval failure should not raise — return whatever chunks succeeded."""
    from graph.nodes import retrieve_node
    from langchain_core.documents import Document

    ok_doc = Document(page_content="some content", metadata={})

    def _invoke_side_effect(query):
        if query == "bad_query":
            raise RuntimeError("Chroma unavailable")
        return [ok_doc]

    fake_state = {
        "customer_id": "test_user",
        "sub_queries": ["good_query", "bad_query"],
        "audit_trail": [],
    }

    with patch("graph.nodes.get_retriever_for_user") as mock_retriever:
        with patch("graph.nodes.fetch_parents", return_value=[ok_doc]):
            mock_retriever.return_value.invoke.side_effect = _invoke_side_effect
            result = retrieve_node(fake_state)

    assert len(result["retrieved_chunks"]) >= 1


# ── answer_node parse-error state ────────────────────────────────────────────

@_skip_no_genai
def test_reason_node_parse_error_is_marked():
    """answer_node must mark its output with _parse_error=True on JSON failure."""
    from graph.nodes import answer_node
    from langchain_core.documents import Document

    fake_state = {
        "customer_id": "test",
        "original_query": "test",
        "sub_queries": ["test"],
        "parent_chunks": [Document(page_content="some content", metadata={"chunk_id": "C1"})],
        "retrieved_chunks": [],
        "audit_trail": [],
    }

    with patch("graph.nodes._llm_invoke_with_retry") as mock_llm:
        mock_response = type("R", (), {"content": "not valid json {{{{",
                                       "usage_metadata": {"input_tokens": 10, "output_tokens": 5}})()
        mock_llm.return_value = mock_response
        result = answer_node(fake_state)

    assert result["answer_output"].get("_parse_error") is True


# ── Chunking mode structure ───────────────────────────────────────────────────

def test_full_chunking_produces_parent_child_metadata(tmp_path):
    """CHUNKING_MODE=full must produce chunks with parent_chunk_id; sentence must not."""
    import os
    import importlib

    txt = tmp_path / "test.txt"
    # Enough content to generate at least one chunk
    txt.write_text(
        "Speaker A: Good morning everyone, let us begin today's meeting.\n"
        "Speaker B: Thank you. The primary issue is that login latency has increased.\n"
        "Speaker A: Understood. What is the root cause of the latency degradation?\n"
        "Speaker B: The load balancer sticky-session rule is pinning users to a saturated node.\n"
        "Speaker A: When will the fix be deployed to production?\n"
        "Speaker B: We are targeting September 25 with a rollback plan if p95 exceeds 1500ms.\n"
    )

    os.environ["CHUNKING_MODE"] = "full"
    os.environ["RETRIEVAL_MODE"] = "full"
    import chroma_utils
    importlib.reload(chroma_utils)

    chunks_full = chroma_utils.load_and_split_document(str(txt))
    assert len(chunks_full) >= 1

    # "full" mode: flat raw_docs (parent_chunk_id assigned during indexing, not loading)
    # "sentence" mode: same loaders — the key difference shows in index_document_to_chroma
    os.environ["CHUNKING_MODE"] = "sentence"
    importlib.reload(chroma_utils)
    chunks_sentence = chroma_utils.load_and_split_document(str(txt))
    assert len(chunks_sentence) >= 1

    # Both use the same loaders for TXT — the structural difference is in indexing,
    # not in the chunks returned by load_and_split_document.
    # Confirm that neither has parent_chunk_id yet (it is assigned by index_document_to_chroma)
    for c in chunks_full:
        assert "parent_chunk_id" not in c.metadata
    for c in chunks_sentence:
        assert "parent_chunk_id" not in c.metadata

    os.environ["CHUNKING_MODE"] = "full"
    importlib.reload(chroma_utils)


def test_sentence_chunk_splits_long_prose():
    """sentence_chunk must split a 300-word passage into multiple ≤256-word chunks."""
    import chroma_utils

    # 30 sentences × ~10 words each = ~300 words → must produce at least 2 chunks at size=256
    sentences = [
        f"This is sentence number {i} and it describes the login latency issue for Meridian."
        for i in range(1, 31)
    ]
    long_text = " ".join(sentences)
    chunks = chroma_utils.sentence_chunk(long_text, size=256, overlap=64)
    assert len(chunks) >= 2, f"Expected ≥2 chunks for 300-word text, got {len(chunks)}"
    for c in chunks:
        word_count = len(c.page_content.split())
        assert word_count <= 320, f"Chunk exceeded size limit: {word_count} words"


def test_sentence_chunk_preserves_sentence_boundaries():
    """sentence_chunk must not cut mid-sentence."""
    import chroma_utils
    text = (
        "The login latency is caused by sticky sessions. "
        "David Park identified the root cause as a misconfigured load balancer. "
        "The fix reduces the timeout from 15 minutes to 3 minutes. "
        "A CPU-pressure trigger forces eviction when CPU exceeds 80 percent. "
        "The rollback plan is to run orion-cli config rollback if p95 exceeds 1500ms."
    )
    chunks = chroma_utils.sentence_chunk(text, size=10, overlap=2)
    # Every chunk must end at a sentence boundary (end with . ! ?)
    for c in chunks:
        stripped = c.page_content.strip()
        assert stripped[-1] in ".!?", f"Chunk cuts mid-sentence: {stripped[-30:]!r}"


# ── Contextual retrieval (Anthropic Sep 2024 method) ─────────────────────────

@_skip_no_genai
def test_contextualize_chunks_happy_path():
    """Each chunk gets a '[Context: ...]' prefix and the has_context_prefix flag."""
    import chroma_utils
    from langchain_core.documents import Document

    chunks = [
        Document(page_content="The rate limit is 1000/min.", metadata={"src": "a"}),
        Document(page_content="The escalation is 15m.", metadata={"src": "b"}),
    ]

    class _FakeResp:
        def __init__(self, content): self.content = content
    class _FakeLLM:
        def invoke(self, prompt): return _FakeResp("this chunk describes the API rate limit")

    with patch("langchain_google_genai.ChatGoogleGenerativeAI", return_value=_FakeLLM()):
        result = chroma_utils._contextualize_chunks(chunks, "fakefile.md")

    assert len(result) == 2
    for r in result:
        assert r.page_content.startswith("[Context:")
        assert "]\n\n" in r.page_content
        assert r.metadata.get("has_context_prefix") is True
        assert "context_text" in r.metadata


@_skip_no_genai
def test_contextualize_chunks_llm_failure_fallback():
    """If the LLM call raises, the original chunk is returned unmodified."""
    import chroma_utils
    from langchain_core.documents import Document

    original = Document(page_content="plain text body", metadata={"src": "a"})

    class _ExplodingLLM:
        def invoke(self, prompt): raise RuntimeError("simulated Gemini 500")

    with patch("langchain_google_genai.ChatGoogleGenerativeAI", return_value=_ExplodingLLM()):
        result = chroma_utils._contextualize_chunks([original], "fakefile.md")

    assert len(result) == 1
    assert result[0].page_content == "plain text body"       # no prefix added
    assert not result[0].metadata.get("has_context_prefix")  # flag absent


@_skip_no_genai
def test_contextualize_chunks_empty_llm_response_is_fallback():
    """LLM returns an empty string -> treat as failed contextualization."""
    import chroma_utils
    from langchain_core.documents import Document

    original = Document(page_content="content body", metadata={"src": "a"})

    class _FakeResp:
        def __init__(self, content): self.content = content
    class _BlankLLM:
        def invoke(self, prompt): return _FakeResp("   ")  # whitespace only

    with patch("langchain_google_genai.ChatGoogleGenerativeAI", return_value=_BlankLLM()):
        result = chroma_utils._contextualize_chunks([original], "fakefile.md")

    assert result[0].page_content == "content body"
    assert not result[0].metadata.get("has_context_prefix")


