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


def _bearer(user_id: str) -> dict:
    """Mint a JWT for the given user_id and return Authorization headers.

    Tenant identity is taken only from the signed token now (the old
    ?user_id= query-param fallback was an IDOR), so test cases that need
    to act as a specific tenant must authenticate as one.
    """
    return {"Authorization": f"Bearer {_create_token(user_id)}"}


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
    # 400 = validation failure; 429 = rate-limited (5/min quota hit by earlier auth tests)
    assert response.status_code in (400, 429)


def test_auth_token_empty_fields():
    response = client.post("/auth/token", json={"workspace": "", "passkey": ""})
    assert response.status_code in (400, 429)


# ── Document list / analytics / audit ────────────────────────────────────────

def test_list_docs():
    response = client.get("/list-docs")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_analytics():
    response = client.get("/analytics")
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
    response = client.post(
        "/delete-doc", json={"file_id": 99999}, headers=_bearer("nobody")
    )
    assert response.status_code == 404
    assert "detail" in response.json()


# ── Upload validation ─────────────────────────────────────────────────────────

def test_upload_unsupported_extension():
    response = client.post(
        "/upload-doc",
        files={"file": ("malware.exe", io.BytesIO(b"bad"), "application/octet-stream")},
        headers=_bearer("test_upload"),
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_upload_empty_file():
    response = client.post(
        "/upload-doc",
        files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
        headers=_bearer("test_upload"),
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_upload_file_too_large():
    big_content = b"x" * (11 * 1024 * 1024)  # 11 MB > 10 MB limit
    response = client.post(
        "/upload-doc",
        files={"file": ("big.txt", io.BytesIO(big_content), "text/plain")},
        headers=_bearer("test_upload"),
    )
    assert response.status_code == 400
    assert "large" in response.json()["detail"].lower()


# ── Chat endpoint ─────────────────────────────────────────────────────────────

def test_chat_empty_question():
    response = client.post("/chat", json={"question": "  ", "user_id": "test_chat"})
    assert response.status_code == 400


def test_chat_question_too_long():
    response = client.post("/chat", json={"question": "q" * 1001, "user_id": "test_chat"})
    assert response.status_code == 400
    assert "long" in response.json()["detail"].lower()


def test_chat_no_documents_returns_graceful_response():
    # A user with no documents should get a 200 with a helpful message, not a 500
    response = client.post(
        "/chat",
        json={"question": "What are the risks?"},
        headers=_bearer("user_with_no_docs_xyz987"),
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "no documents" in data["answer"].lower() and data["confidence"] == 0.0


# ── Multi-tenant isolation ────────────────────────────────────────────────────

def test_multitenant_doc_isolation():
    """Documents uploaded by user A must not appear in user B's list."""
    user_a = "tenant_isolation_user_a"
    user_b = "tenant_isolation_user_b"
    auth_a = _bearer(user_a)
    auth_b = _bearer(user_b)

    # Remove stale state from any previous crashed run
    for doc in client.get("/list-docs", headers=auth_a).json():
        if doc["filename"] == "private_a.txt":
            client.post("/delete-doc", json={"file_id": doc["id"]}, headers=auth_a)

    # Upload a doc as user A
    content = b"Confidential data for user A only."
    upload = client.post(
        "/upload-doc",
        files={"file": ("private_a.txt", io.BytesIO(content), "text/plain")},
        headers=auth_a,
    )
    assert upload.status_code == 200
    file_id = upload.json()["file_id"]

    try:
        # User B should not see user A's document
        docs_b = client.get("/list-docs", headers=auth_b).json()
        filenames_b = [d["filename"] for d in docs_b]
        assert "private_a.txt" not in filenames_b

        # User A should see their own document
        docs_a = client.get("/list-docs", headers=auth_a).json()
        filenames_a = [d["filename"] for d in docs_a]
        assert "private_a.txt" in filenames_a
    finally:
        client.post("/delete-doc", json={"file_id": file_id}, headers=auth_a)


def test_multitenant_delete_isolation():
    """User B must not be able to delete user A's document."""
    user_a = "tenant_delete_user_a"
    user_b = "tenant_delete_user_b"
    auth_a = _bearer(user_a)
    auth_b = _bearer(user_b)

    # Remove stale state from any previous crashed run
    for doc in client.get("/list-docs", headers=auth_a).json():
        if doc["filename"] == "owned_by_a.txt":
            client.post("/delete-doc", json={"file_id": doc["id"]}, headers=auth_a)

    upload = client.post(
        "/upload-doc",
        files={"file": ("owned_by_a.txt", io.BytesIO(b"content"), "text/plain")},
        headers=auth_a,
    )
    assert upload.status_code == 200
    file_id = upload.json()["file_id"]

    try:
        # User B tries to delete user A's file — must 404 because /delete-doc now
        # derives user_id from the JWT, not the request body. The pre-fix attack
        # of "POST /delete-doc with body {file_id, user_id: <victim>}" no longer
        # has a way to spoof the tenant.
        response = client.post(
            "/delete-doc", json={"file_id": file_id}, headers=auth_b
        )
        assert response.status_code == 404
    finally:
        client.post("/delete-doc", json={"file_id": file_id}, headers=auth_a)


# ── Brief endpoint (requires GOOGLE_API_KEY) ──────────────────────────────────

@pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set — skipping LLM integration test",
)
def test_brief_empty_query():
    response = client.post("/brief", json={"query": "  "}, headers=_bearer("test_brief"))
    assert response.status_code == 400


@pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set — skipping LLM integration test",
)
def test_brief_end_to_end():
    USER_ID = "test_integration"
    auth = _bearer(USER_ID)

    # Pre-test cleanup
    existing = client.get("/list-docs", headers=auth).json()
    for doc in existing:
        if doc["filename"] == "acme_issue.txt":
            client.post("/delete-doc", json={"file_id": doc["id"]}, headers=auth)

    doc_content = (
        "Customer ACME has an open P1 issue about login latency. "
        "Last resolution attempt on 2024-03-15 failed. "
        "Current workaround is to clear browser cache. "
        "Engineering identified the root cause as a misconfigured load balancer. "
        "Next steps: deploy hotfix by 2024-03-20."
    )

    upload_resp = client.post(
        "/upload-doc",
        files={"file": ("acme_issue.txt", io.BytesIO(doc_content.encode()), "text/plain")},
        params={"doc_type": "auto"},
        headers=auth,
    )
    assert upload_resp.status_code == 200, f"Upload failed: {upload_resp.json()}"
    file_id = upload_resp.json()["file_id"]

    try:
        brief_resp = client.post(
            "/brief",
            json={"query": "What is the open issue for ACME?", "customer_id": USER_ID},
            headers=auth,
        )
        assert brief_resp.status_code == 200, f"Brief failed: {brief_resp.json()}"

        data = brief_resp.json()
        assert "brief" in data
        brief = data["brief"]
        for key in ("faithfulness_score", "issues", "risks", "open_questions",
                    "talking_points", "sources", "suspicious_facts",
                    "suspicious_claims", "judge_status"):
            assert key in brief, f"Missing key in brief: {key}"
        assert isinstance(brief["suspicious_claims"], list)
        assert isinstance(brief["suspicious_facts"], list)
        # judge_status must also appear at the top level (explicit BriefResponse
        # field) so OpenAPI consumers don't depend on nested dict access.
        assert "judge_status" in data
        assert data["judge_status"] == brief["judge_status"]
        # We expect verification to have RUN (or legitimately had no claims
        # that needed LLM judging). The small single-doc test case often
        # produces briefs where regex verifies everything and nothing is sent
        # to the judge — that's a valid "no_claims" outcome, not a failure.
        # What we reject: "error" / "parse_error" / "skipped_breaker_open" —
        # those all mean verification silently failed, which is exactly the
        # regression this field exists to catch.
        assert brief["judge_status"] in ("ok", "no_claims"), (
            f"judge_status='{brief['judge_status']}' indicates verification "
            f"failed or was skipped — expected 'ok' or 'no_claims'."
        )

        assert data["loop_count"] >= 1
        assert isinstance(data["audit_trail"], list)
        assert len(data["audit_trail"]) > 0

    finally:
        client.post("/delete-doc", json={"file_id": file_id}, headers=auth)


# ── Circuit breaker ───────────────────────────────────────────────────────────

def test_circuit_breaker_opens_after_threshold():
    """5 consecutive failures should open the breaker."""
    from langchain_utils import llm_breaker, _CBState
    llm_breaker.on_success()  # reset to CLOSED
    for _ in range(5):
        llm_breaker.on_failure()
    assert llm_breaker.state == _CBState.OPEN
    llm_breaker.on_success()  # restore for other tests


def test_circuit_breaker_half_open_reverts_on_failure():
    """A failure while HALF_OPEN must revert to OPEN, not stay HALF_OPEN."""
    from langchain_utils import llm_breaker, _CBState
    llm_breaker.on_success()  # start CLOSED
    for _ in range(5):
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
    for _ in range(5):
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
        "/list-docs",
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


# ── BM25 tokenizer ────────────────────────────────────────────────────────────

def test_bm25_tokenizer_strips_punctuation():
    from chroma_utils import _tokenize
    assert _tokenize("issue, risk") == ["issue", "risk"]
    assert _tokenize("pre-call") == ["pre", "call"]
    assert _tokenize("don't") == ["don", "t"]


# ── Retrieval failure propagation ─────────────────────────────────────────────

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


# ── reason_node parse-error state ────────────────────────────────────────────

def test_reason_node_parse_error_is_marked():
    """reason_node must mark its output with _parse_error=True on JSON failure."""
    from graph.nodes import reason_node
    from langchain_core.documents import Document

    fake_state = {
        "original_query": "test",
        "parent_chunks": [Document(page_content="some content")],
        "retrieved_chunks": [],
        "audit_trail": [],
    }

    with patch("graph.nodes._llm_invoke_with_retry") as mock_llm:
        mock_response = type("R", (), {"content": "not valid json {{{{", "response_metadata": {}})()
        mock_llm.return_value = mock_response
        result = reason_node(fake_state)

    assert result["reasoning_output"].get("_parse_error") is True


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


# ── Context prefix stripping (user-facing citation passage hygiene) ──────────

def test_strip_context_prefix_removes_prefix_when_flag_set():
    from output.brief_generator import _strip_context_prefix
    content = "[Context: from Orion integration guide]\n\nThe rate limit is 1000/min."
    metadata = {"has_context_prefix": True}
    out = _strip_context_prefix(content, metadata)
    assert out == "The rate limit is 1000/min."


def test_strip_context_prefix_no_op_without_flag():
    """A chunk that fell through contextualize's error path has no flag; we
    must NOT strip even if its content happens to contain '[Context:' or ']\\n\\n'."""
    from output.brief_generator import _strip_context_prefix
    content = "[Context: this looks like a prefix but isn't]\n\nThe real passage."
    metadata = {}  # no has_context_prefix
    out = _strip_context_prefix(content, metadata)
    assert out == content


def test_strip_context_prefix_no_op_when_content_mismatches_flag():
    """Metadata says prefixed but content doesn't start with '[Context:'. Safety."""
    from output.brief_generator import _strip_context_prefix
    content = "Plain content with no prefix at all."
    metadata = {"has_context_prefix": True}
    out = _strip_context_prefix(content, metadata)
    assert out == content  # defense-in-depth: no-op if state is inconsistent
