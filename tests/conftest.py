"""
conftest.py — pytest fixtures for the full FDE Knowledge Assistant test suite.

Module-level env-var setup (CHROMA_DB_DIR, RAG_DB_PATH, TESTING_DISABLE_RATELIMIT)
runs before any test-module import so that chroma_utils / db_utils pick up
the isolated paths at import time.  Per-test DB fixtures (tmp_db / db_conn)
build on top of that foundation with a fresh SQLite file in a pytest tmp_path.
"""

import os
import sys
import json
import shutil
import sqlite3
import tempfile

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock
from langchain_core.documents import Document

# ── sys.path so `from db_utils import ...` and `import api.db_utils` both work ──
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "api"))
sys.path.insert(0, ROOT)  # needed for `import api.db_utils` in isolation/versioning tests

# ── Session-level isolated data stores ───────────────────────────────────────
# Must happen at module level — chroma_utils and db_utils read env vars once
# at import time, which happens during pytest collection (before fixtures fire).
_TEST_DATA_DIR = tempfile.mkdtemp(prefix="rag_test_data_")
_CHROMA_TEST_DIR = os.path.join(_TEST_DATA_DIR, "chroma_db")
os.makedirs(_CHROMA_TEST_DIR, exist_ok=True)
os.environ["CHROMA_DB_DIR"] = _CHROMA_TEST_DIR
os.environ["RAG_DB_PATH"] = os.path.join(_TEST_DATA_DIR, "rag_app.db")
os.environ["TESTING_DISABLE_RATELIMIT"] = "1"

# ── Sample raw document content ───────────────────────────────────────────────

SAMPLE_TRANSCRIPT = """2024-09-15 Status Call - Cascadia Inc

Sarah (FDE): Good morning everyone. Let's review where we stand.
John (Customer): Thanks Sarah. We're still seeing the us-east-2 deployment issues.
Sarah (FDE): I understand. We've escalated TICK-4521 to P0 internally.
John (Customer): We need this resolved by end of month. Also, what's the status on the SSO integration you promised?
Sarah (FDE): The SSO work is scheduled for completion by October 1st. That's a firm commitment.
John (Customer): Good. Our SLA requires 99.9% uptime and we've been at 99.1% this quarter.
Sarah (FDE): We're aware and engineering is on it. I'll send you a detailed update by Friday.
"""

SAMPLE_TICKETS_CSV = """ticket_id,summary,status,priority,created_date,updated_date,reporter,assignee,description,resolution
TICK-4521,us-east-2 deployment failing,open,P0,2024-09-01,2024-10-12,john@cascadia.com,eng-team,Deployment pipeline fails in us-east-2 region with timeout errors,
TICK-4580,SSO integration request,in_progress,P1,2024-09-10,2024-10-15,john@cascadia.com,dev-team,Customer requesting SSO integration for their Okta instance,
TICK-4601,Uptime below SLA threshold,open,P0,2024-10-01,2024-10-20,john@cascadia.com,sre-team,Uptime at 99.1% vs committed 99.9%,
"""

SAMPLE_COMMITMENTS_CSV = """commitment,promised_date,current_target_date,status,owner,customer_aware
SSO integration delivery,2024-10-01,2024-10-01,open,dev-team,true
AWS us-east-2 region support,2024-11-30,2024-11-30,in_progress,eng-team,true
Uptime improvement to 99.9%,2024-12-31,2024-12-31,in_progress,sre-team,true
"""

SAMPLE_NOTES = """Post-call notes - September 15 2024
Account: Cascadia Inc
Attendees: Sarah (FDE), John (Customer CTO)

Key points:
- Customer is unhappy with us-east-2 deployment issues (ongoing 3+ weeks)
- SSO integration promised for Oct 1 - HIGH PRIORITY
- SLA breach: uptime at 99.1% vs 99.9% committed
- Action item: Send detailed update email by Friday Sep 20

Customer sentiment: At-risk. John was visibly frustrated about the SSO delay.
Next call: October 22, 2024
"""

# ── Session-level teardown ─────────────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def _cleanup_test_data_dir():
    """Remove the temp directory created at module level after all tests."""
    yield
    for key in ("CHROMA_DB_DIR", "RAG_DB_PATH", "TESTING_DISABLE_RATELIMIT"):
        os.environ.pop(key, None)
    shutil.rmtree(_TEST_DATA_DIR, ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def _force_local_embedder_for_tests():
    """Unset OPENAI_API_KEY for the duration of the pytest session."""
    saved = os.environ.pop("OPENAI_API_KEY", None)
    yield
    if saved is not None:
        os.environ["OPENAI_API_KEY"] = saved


@pytest.fixture(scope="session", autouse=True)
def _run_migrations_before_tests():
    """Apply SQLite migrations once at session start (idempotent)."""
    from db_utils import run_migrations
    run_migrations()
    yield


@pytest.fixture(autouse=True)
def _reset_circuit_breaker():
    """Restore the LLM circuit breaker to CLOSED after every test."""
    yield
    try:
        from langchain_utils import llm_breaker, _CBState
        llm_breaker.state = _CBState.CLOSED
        llm_breaker.failures = 0
    except Exception:
        pass

# ── Per-test DB fixture ────────────────────────────────────────────────────────

@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Fresh SQLite DB with all migrations applied, isolated to this test."""
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("RAG_DB_PATH", db_path)
    # Reload db_utils so DB_NAME picks up the new env var
    import importlib
    import api.db_utils as db_utils
    importlib.reload(db_utils)
    db_utils.run_migrations()
    yield db_path


@pytest.fixture
def db_conn(tmp_db):
    """Direct SQLite connection for assertion queries against tmp_db."""
    conn = sqlite3.connect(tmp_db)
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()

# ── Chroma fixture ─────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_chroma(tmp_path, monkeypatch):
    """Isolated Chroma directory in tmp_path."""
    chroma_dir = str(tmp_path / "chroma")
    monkeypatch.setenv("CHROMA_DB_DIR", chroma_dir)
    return chroma_dir

# ── Mock LLM ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm():
    """LLM mock returning pre-written JSON strings keyed on prompt content."""
    llm = MagicMock()

    def _respond(prompt, *args, **kwargs):
        resp = MagicMock()
        lp = prompt.lower() if isinstance(prompt, str) else ""
        if "decompos" in lp or "sub-quer" in lp:
            resp.content = '["what are open tickets?", "what commitments are outstanding?"]'
        elif "account_summary" in lp or "posture" in lp:
            resp.content = (
                "This account is at-risk due to ongoing deployment issues "
                "and a missed commitment deadline."
            )
        elif "recent" in lp and "change" in lp:
            resp.content = "[]"
        elif "anticipated" in lp:
            resp.content = (
                '[{"topic": "SSO status", "evidence": "TICK-4580", '
                '"chunk_id": "C1", "urgency": "high"}]'
            )
        elif "directive" in lp:
            resp.content = (
                '[{"verb": "Acknowledge", "directive": "Address the SSO delay directly", '
                '"basis": "SSO commitment missed Oct 1"}]'
            )
        elif "answer" in lp:
            resp.content = (
                '{"answer": "The SLA is 99.9% uptime.", "answer_status": "ok", '
                '"citations": [{"claim": "SLA is 99.9%", "chunk_id": "C1"}]}'
            )
        else:
            resp.content = (
                '{"issues": [], "risks": [], "open_questions": [], "talking_points": []}'
            )
        resp.usage_metadata = {"input_tokens": 100, "output_tokens": 50}
        return resp

    llm.invoke = _respond
    return llm

# ── Sample LangChain Documents ─────────────────────────────────────────────────

@pytest.fixture
def sample_transcript_docs():
    """Transcript chunks as if parsed and chunked from the Cascadia status call."""
    return [
        Document(
            page_content=(
                "Sarah (FDE): We've escalated TICK-4521 to P0 internally.\n"
                "John (Customer): We need this resolved by end of month."
            ),
            metadata={
                "doc_type": "transcript", "doc_date": "2024-09-15",
                "filename": "2024-09-15_transcript_status-call.txt",
                "chunk_id": "C1", "user_id": "cascadia-test",
                "is_latest_version": "true",
            },
        ),
        Document(
            page_content=(
                "John (Customer): What's the status on the SSO integration you promised?\n"
                "Sarah (FDE): The SSO work is scheduled for completion by October 1st. "
                "That's a firm commitment."
            ),
            metadata={
                "doc_type": "transcript", "doc_date": "2024-09-15",
                "filename": "2024-09-15_transcript_status-call.txt",
                "chunk_id": "C2", "user_id": "cascadia-test",
                "is_latest_version": "true",
            },
        ),
    ]


@pytest.fixture
def sample_ticket_docs():
    """Ticket chunks with full metadata for Cascadia open tickets."""
    return [
        Document(
            page_content=(
                "TICK-4521: us-east-2 deployment failing\n"
                "Status: open | Priority: P0\n"
                "Deployment pipeline fails in us-east-2 region with timeout errors"
            ),
            metadata={
                "doc_type": "ticket", "doc_date": "2024-10-12",
                "filename": "2024-10-01_tickets_open.csv",
                "ticket_id": "TICK-4521", "status": "open", "priority": "P0",
                "assignee": "eng-team", "updated_date": "2024-10-12",
                "chunk_id": "C3", "user_id": "cascadia-test",
                "is_latest_version": "true",
            },
        ),
        Document(
            page_content=(
                "TICK-4601: Uptime below SLA threshold\n"
                "Status: open | Priority: P0\n"
                "Uptime at 99.1% vs committed 99.9%"
            ),
            metadata={
                "doc_type": "ticket", "doc_date": "2024-10-20",
                "filename": "2024-10-01_tickets_open.csv",
                "ticket_id": "TICK-4601", "status": "open", "priority": "P0",
                "assignee": "sre-team", "updated_date": "2024-10-20",
                "chunk_id": "C4", "user_id": "cascadia-test",
                "is_latest_version": "true",
            },
        ),
    ]


@pytest.fixture
def sample_commitment_docs_overdue():
    """Commitment chunks where the SSO integration is overdue."""
    return [
        Document(
            page_content=(
                "SSO integration delivery\n"
                "Status: open | Owner: dev-team | Customer aware: true"
            ),
            metadata={
                "doc_type": "commitment_tracker", "doc_date": "2024-10-15",
                "filename": "2024-10-15_commitments_tracker.csv",
                "commitment_id": "C001", "promised_date": "2024-10-01",
                "current_target_date": "2024-10-01", "commitment_status": "open",
                "owner": "dev-team", "customer_aware": "true",
                "is_slipped": "false", "chunk_id": "C5",
                "user_id": "cascadia-test", "is_latest_version": "true",
            },
        ),
        Document(
            page_content=(
                "AWS us-east-2 region support\n"
                "Status: in_progress | Owner: eng-team | Customer aware: true"
            ),
            metadata={
                "doc_type": "commitment_tracker", "doc_date": "2024-10-15",
                "filename": "2024-10-15_commitments_tracker.csv",
                "commitment_id": "C002", "promised_date": "2024-11-30",
                "current_target_date": "2024-11-30", "commitment_status": "in_progress",
                "owner": "eng-team", "customer_aware": "true",
                "is_slipped": "false", "chunk_id": "C6",
                "user_id": "cascadia-test", "is_latest_version": "true",
            },
        ),
    ]


@pytest.fixture
def sample_customer():
    """Sample customer dict as returned by DB layer."""
    return {
        "id": 1, "name": "Cascadia Inc", "slug": "cascadia",
        "fde_user_id": "test-fde-user",
        "last_call_date": "2024-09-15",
        "created_at": "2024-01-01T00:00:00",
    }

# ── Temp file helpers ─────────────────────────────────────────────────────────

@pytest.fixture
def sample_transcript_file(tmp_path):
    f = tmp_path / "2024-09-15_transcript_status-call.txt"
    f.write_text(SAMPLE_TRANSCRIPT)
    return str(f)


@pytest.fixture
def sample_tickets_csv_file(tmp_path):
    f = tmp_path / "2024-10-01_tickets_open.csv"
    f.write_text(SAMPLE_TICKETS_CSV)
    return str(f)


@pytest.fixture
def sample_commitments_csv_file(tmp_path):
    f = tmp_path / "2024-10-15_commitments_tracker.csv"
    f.write_text(SAMPLE_COMMITMENTS_CSV)
    return str(f)


@pytest.fixture
def sample_notes_file(tmp_path):
    f = tmp_path / "2024-09-15_notes_status-call.txt"
    f.write_text(SAMPLE_NOTES)
    return str(f)

# ── Date helpers ──────────────────────────────────────────────────────────────

@pytest.fixture
def today():
    return datetime.now().strftime("%Y-%m-%d")


@pytest.fixture
def stale_date():
    """40 days ago — older than the 30-day stale threshold."""
    return (datetime.now() - timedelta(days=40)).strftime("%Y-%m-%d")


@pytest.fixture
def recent_date():
    """10 days ago — within the 30-day recency window."""
    return (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
