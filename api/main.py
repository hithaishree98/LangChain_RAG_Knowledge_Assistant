import asyncio
import hashlib
import json
import logging
import os
import secrets
import uuid
import csv
import io
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security.api_key import APIKeyHeader

from pydantic_models import (
    QueryInput, QueryResponse, DocumentInfo, DeleteFileRequest,
    BriefRequest, BriefResponse,
)
from langchain_utils import llm_breaker
from chroma_utils import index_document_to_chroma, delete_doc_from_chroma, vectorstore
from db_utils import (
    insert_application_logs, get_all_documents,
    insert_document_record, delete_document_record,
    get_query_stats, get_audit_log, run_migrations, insert_brief_log,
)
from notification_utils import send_to_slack  # noqa: F401 — available for future Slack escalation wiring


# ── Structured JSON logger ────────────────────────────────────────────────────

class StructuredLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, "app.log"))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("%(message)s"))

        self.logger.addHandler(console)
        self.logger.addHandler(file_handler)

    def _write(self, level, event, **kwargs):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "event": event,
            **kwargs,
        }
        getattr(self.logger, level.lower())(json.dumps(entry))

    def info(self, event, **kwargs):    self._write("INFO", event, **kwargs)
    def warning(self, event, **kwargs): self._write("WARNING", event, **kwargs)
    def error(self, event, **kwargs):   self._write("ERROR", event, **kwargs)
    def debug(self, event, **kwargs):   self._write("DEBUG", event, **kwargs)


log = StructuredLogger("rag_api")


_ENV = os.getenv("ENVIRONMENT", "development").lower()


@asynccontextmanager
async def lifespan(_: FastAPI):
    run_migrations()
    if not os.getenv("API_KEY"):
        if _ENV == "production":
            raise RuntimeError(
                "API_KEY must be set in production. "
                "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
            )
        log.warning("startup_warning",
            message="API_KEY is not set — all protected endpoints are publicly accessible")
    if not os.getenv("JWT_SECRET"):
        if _ENV == "production":
            raise RuntimeError(
                "JWT_SECRET must be set in production. "
                "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
            )
    # Pre-load the embedding model and cross-encoder so first-request latency
    # is bounded. Without this, the 440MB nomic-embed model cold-loads lazily
    # on the first /upload-doc call and can exceed any reasonable client timeout.
    # Warmup failures (missing internet, HF hub down, disk full) degrade to
    # lazy-load rather than crashing the API — otherwise /health never returns
    # ready and the server appears permanently broken on cold environments.
    from chroma_utils import warmup_models
    log.info("warmup_start")
    try:
        warmup_models()
        log.info("warmup_complete")
    except Exception as e:
        log.warning("warmup_failed", error=str(e),
                    note="models will lazy-load on first request")
    yield

app = FastAPI(title="FDE Assistant API", lifespan=lifespan)

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:8501").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key", "Authorization"],
)

# ── Rate limiter (slowapi) ────────────────────────────────────────────────────
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

_limiter = Limiter(key_func=get_remote_address)
app.state.limiter = _limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


def _limit(rate: str):
    return _limiter.limit(rate)


# ── Constants ─────────────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.4
MAX_QUESTION_LENGTH = 1000
MAX_FILE_SIZE_MB = 10
ALLOWED_EXTENSIONS = [".pdf", ".docx", ".html", ".txt", ".json"]
MAX_QUESTIONNAIRE_ROWS = 200
QUESTIONNAIRE_SEMAPHORE = asyncio.Semaphore(5)

# ── API-key auth ──────────────────────────────────────────────────────────────

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def verify_api_key(api_key: str = Depends(api_key_header)):
    expected = os.getenv("API_KEY")
    if not expected:
        # No API_KEY configured — open access allowed in development only.
        return
    if not api_key or api_key != expected:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="Invalid or missing API key")


# ── JWT workspace tokens ──────────────────────────────────────────────────────

_JWT_SECRET = os.getenv("JWT_SECRET")
if not _JWT_SECRET:
    _JWT_SECRET = secrets.token_hex(32)
    logging.warning(
        "JWT_SECRET is not set — using a random secret. "
        "All tokens will be invalidated on every restart. "
        "Set JWT_SECRET in your .env for persistent sessions."
    )
_JWT_ALGORITHM = "HS256"
_JWT_EXPIRE_HOURS = 24


def _create_token(user_id: str) -> str:
    # Don't swallow the ImportError — returning an empty token here used to
    # silently break /auth/token (clients got a 200 with a useless empty
    # string), while _decode_token would raise on the next request. Letting
    # the ImportError propagate makes the dependency issue surface at startup
    # / first request rather than as a confusing auth failure later.
    from jose import jwt
    payload = {
        "sub": user_id,
        "exp": datetime.now(timezone.utc) + timedelta(hours=_JWT_EXPIRE_HOURS),
    }
    return jwt.encode(payload, _JWT_SECRET, algorithm=_JWT_ALGORITHM)


def _decode_token(token: str) -> str:
    from jose import jwt, JWTError
    try:
        payload = jwt.decode(token, _JWT_SECRET, algorithms=[_JWT_ALGORITHM])
        return payload["sub"]
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


async def get_current_user(authorization: str = Header(None)) -> str:
    """Extract user_id from a JWT Bearer token.

    Tenant identity is derived ONLY from the signed token. Earlier versions
    accepted a `user_id` query parameter as a fallback, but that allowed any
    caller to spoof another tenant's id by passing `?user_id=<target>`. The
    fallback is removed; callers must mint a token via `/auth/token`.

    When no Authorization header is present, returns the literal "default"
    so unauthenticated dev-mode requests still resolve to a single shared
    tenant rather than crashing. In production `API_KEY` enforcement on
    write endpoints prevents anonymous access regardless.
    """
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:]
        return _decode_token(token)
    return "default"


# ── Exception handler ─────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error("unhandled_exception",
        path=str(request.url),
        method=request.method,
        error=str(exc),
        error_type=type(exc).__name__,
    )
    return JSONResponse(status_code=500, content={
        "message": "Something went wrong. Please try again."
    })


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    checks = {}
    try:
        get_all_documents()
        checks["database"] = "ok"
    except Exception as e:
        checks["database"] = f"error: {str(e)}"
    try:
        vectorstore.get()
        checks["vector_store"] = "ok"
    except Exception as e:
        checks["vector_store"] = f"error: {str(e)}"
    checks["llm_key"] = "ok" if os.getenv("GOOGLE_API_KEY") else "missing"
    checks["slack"] = "configured" if os.getenv("SLACK_WEBHOOK_URL") else "not configured"
    from langchain_utils import _CBState
    checks["circuit_breaker"] = llm_breaker.state.value
    _OK_VALUES = {"ok", "configured", "not configured", _CBState.CLOSED.value}
    all_ok = all(v in _OK_VALUES for v in checks.values())
    return {"status": "healthy" if all_ok else "degraded", "checks": checks}


# ── Auth endpoint ─────────────────────────────────────────────────────────────

@app.post("/auth/token")
@_limit("5/minute")
async def get_token(request: Request, body: dict):  # request is used by slowapi for IP-based limiting
    """Issue a short-lived JWT for a workspace. user_id = sha256(workspace:passkey)."""
    workspace = body.get("workspace", "").strip().lower()
    passkey = body.get("passkey", "").strip()
    if not workspace or not passkey:
        raise HTTPException(status_code=400, detail="workspace and passkey are required")
    raw = f"{workspace}:{passkey}"
    user_id = hashlib.sha256(raw.encode()).hexdigest()[:32]
    token = _create_token(user_id)
    return {"token": token, "user_id": user_id}


# ── Brief (primary endpoint) ──────────────────────────────────────────────────

@app.post("/brief", response_model=BriefResponse, dependencies=[Depends(verify_api_key)])
@_limit("10/minute")
async def get_brief(request: Request, req: BriefRequest, user_id: str = Depends(get_current_user)):
    """
    Primary endpoint. Runs the LangGraph workflow and returns a structured
    pre-call brief with citations, faithfulness score, and audit trail.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query cannot be empty")

    if llm_breaker.is_open():
        raise HTTPException(status_code=503,
            detail="AI service temporarily unavailable. Please try again later.")

    customer_id = req.customer_id or user_id
    log.info("brief_request", customer_id=customer_id, query_length=len(req.query))

    try:
        from graph.workflow import run_workflow
        state = await run_workflow(customer_id, req.query.strip())
    except Exception as e:
        log.error("brief_workflow_failed", error=str(e))
        raise HTTPException(status_code=503,
            detail=f"Brief generation failed: {str(e)}")

    brief = state.get("brief") or {}
    faithfulness = brief.get("faithfulness_score", 0.0)
    loop_count = state.get("iteration_count", 0)
    sources = brief.get("sources", [])
    suspicious = brief.get("suspicious_facts", [])

    if suspicious:
        log.warning("hallucination_suspected",
            customer_id=customer_id, facts_not_in_context=suspicious)

    try:
        insert_brief_log(customer_id, req.query, json.dumps(brief), faithfulness, loop_count)
    except Exception as e:
        log.error("brief_log_failed", error=str(e))
    log.info("brief_response", customer_id=customer_id,
             faithfulness=faithfulness, loop_count=loop_count)

    return BriefResponse(
        brief=brief,
        sources=sources,
        faithfulness_score=faithfulness,
        judge_status=brief.get("judge_status", "disabled"),
        loop_count=loop_count,
        audit_trail=state.get("audit_trail", []),
    )


# ── Chat (stub — delegates to brief, returns prose summary) ──────────────────

@app.post("/chat", response_model=QueryResponse, dependencies=[Depends(verify_api_key)])
async def chat(query_input: QueryInput, user_id: str = Depends(get_current_user)):
    """
    Legacy chat endpoint kept for backwards compatibility.
    Internally calls the /brief workflow and returns a prose summary.
    """
    question = query_input.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if len(question) > MAX_QUESTION_LENGTH:
        raise HTTPException(status_code=400,
            detail=f"Question too long. Maximum {MAX_QUESTION_LENGTH} characters.")

    docs = get_all_documents(user_id=user_id)
    if not docs:
        return QueryResponse(
            answer="No documents uploaded yet. Please upload some documents first.",
            session_id=query_input.session_id or str(uuid.uuid4()),
            model=query_input.model,
            confidence=0.0,
            sources=[],
            escalated=True,
        )

    if llm_breaker.is_open():
        raise HTTPException(status_code=503,
            detail="AI service temporarily unavailable. Please try again later.")

    session_id = query_input.session_id or str(uuid.uuid4())

    try:
        from graph.workflow import run_workflow
        state = await run_workflow(user_id, question)
    except Exception as e:
        llm_breaker.on_failure()
        raise HTTPException(status_code=503,
            detail="AI service temporarily unavailable. Please try again.")

    brief = state.get("brief") or {}
    faithfulness = brief.get("faithfulness_score", 0.0)

    # Build a prose answer from the brief sections
    answer_parts = []
    if brief.get("issues"):
        answer_parts.append("Issues: " + "; ".join(i["claim"] for i in brief["issues"][:3]))
    if brief.get("risks"):
        answer_parts.append("Risks: " + "; ".join(r["claim"] for r in brief["risks"][:3]))
    if brief.get("talking_points"):
        answer_parts.append("Key points: " + "; ".join(t["point"] for t in brief["talking_points"][:3]))
    if not answer_parts:
        answer_parts.append(brief.get("summary", "No findings."))
    answer = "\n\n".join(answer_parts)

    sources = [s["filename"] for s in brief.get("sources", [])]
    escalated = faithfulness < CONFIDENCE_THRESHOLD

    try:
        insert_application_logs(
            session_id, question, answer, query_input.model.value,
            faithfulness, escalated, ", ".join(sources), user_id=user_id,
        )
    except Exception as e:
        log.error("db_write_failed", session_id=session_id, error=str(e))

    return QueryResponse(
        answer=answer,
        session_id=session_id,
        model=query_input.model,
        confidence=faithfulness,
        sources=sources,
        escalated=escalated,
    )


# ── Streaming chat (stub — streams brief summary) ────────────────────────────

@app.post("/chat/stream", dependencies=[Depends(verify_api_key)])
async def chat_stream(query_input: QueryInput, user_id: str = Depends(get_current_user)):
    """Stream answer tokens for the brief. Final line: '---META---' + JSON metadata."""
    question = query_input.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if llm_breaker.is_open():
        raise HTTPException(status_code=503, detail="AI service temporarily unavailable.")

    session_id = query_input.session_id or str(uuid.uuid4())

    async def generate():
        try:
            from graph.workflow import run_workflow
            state = await run_workflow(user_id, question)
        except Exception as e:
            llm_breaker.on_failure()
            yield json.dumps({"error": str(e), "session_id": session_id}) + "\n"
            return

        brief = state.get("brief") or {}
        faithfulness = brief.get("faithfulness_score", 0.0)
        sources = [s["filename"] for s in brief.get("sources", [])]
        escalated = faithfulness < CONFIDENCE_THRESHOLD

        # Stream the brief as plain text sections
        if brief.get("summary"):
            yield brief["summary"] + "\n\n"
        for issue in brief.get("issues", []):
            yield f"Issue: {issue['claim']}\n"
        for risk in brief.get("risks", []):
            yield f"Risk: {risk['claim']}\n"
        for pt in brief.get("talking_points", []):
            yield f"Talking point: {pt['point']}\n"

        try:
            insert_application_logs(
                session_id, question,
                brief.get("summary", ""),
                query_input.model.value,
                faithfulness, escalated, ", ".join(sources), user_id=user_id,
            )
        except Exception:
            pass

        meta = json.dumps({
            "session_id": session_id,
            "confidence": faithfulness,
            "sources": sources,
            "escalated": escalated,
        })
        yield f"\n---META---{meta}"

    return StreamingResponse(generate(), media_type="text/plain")


# ── Document upload ───────────────────────────────────────────────────────────

@app.post("/upload-doc", dependencies=[Depends(verify_api_key)])
@_limit("10/minute")
async def upload_and_index_document(
    request: Request,
    file: UploadFile = File(...),
    doc_type: str = "auto",
    user_id: str = Depends(get_current_user),
):
    """
    Upload and index a document. Supports PDF, DOCX, HTML, TXT (transcripts),
    and JSON (tickets). Set doc_type to 'transcript', 'ticket', 'pdf', or 'auto'
    (infers from extension).
    """
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)

    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(status_code=400,
            detail=f"File too large ({size_mb:.1f}MB). Maximum is {MAX_FILE_SIZE_MB}MB.")

    safe_name = os.path.basename(file.filename)
    suffix = os.path.splitext(safe_name)[1]

    existing = [doc["filename"] for doc in get_all_documents(user_id=user_id)]
    if safe_name in existing:
        raise HTTPException(status_code=409,
            detail=f"'{safe_name}' already exists. Delete it first or rename the file.")
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    temp_path = tmp.name

    try:
        tmp.write(contents)
        tmp.close()

        log.info("upload_started", filename=safe_name, size_mb=round(size_mb, 2),
                 doc_type=doc_type)
        file_id = insert_document_record(safe_name, user_id=user_id)
        success = index_document_to_chroma(temp_path, file_id, user_id=user_id, filename=safe_name)

        if not success:
            # Pass user_id so the cleanup actually matches the row we just inserted.
            # Without it, delete_document_record's user_id default ("default") leaves
            # an orphan row in document_store with no Chroma vectors backing it.
            delete_document_record(file_id, user_id=user_id)
            raise HTTPException(status_code=500,
                detail=f"Failed to index '{file.filename}'. The file may be corrupted.")

        log.info("upload_success", filename=safe_name, file_id=file_id)
        return {"message": f"'{safe_name}' uploaded successfully.", "file_id": file_id}

    except HTTPException:
        raise
    except Exception as e:
        log.error("upload_error", filename=safe_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ── Document list / delete ────────────────────────────────────────────────────

@app.get("/list-docs", response_model=list[DocumentInfo])
def list_documents(user_id: str = Depends(get_current_user)):
    try:
        return get_all_documents(user_id=user_id)
    except Exception as e:
        log.error("list_docs_error", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve documents.")


@app.post("/delete-doc", dependencies=[Depends(verify_api_key)])
def delete_document(request: DeleteFileRequest, user_id: str = Depends(get_current_user)):
    # user_id is JWT-derived; we never trust a tenant id from the request body.
    existing = get_all_documents(user_id=user_id)
    if not any(doc["id"] == request.file_id for doc in existing):
        raise HTTPException(status_code=404, detail="Document not found in your workspace.")

    chroma_ok = delete_doc_from_chroma(request.file_id, user_id=user_id)
    db_ok = delete_document_record(request.file_id, user_id=user_id)

    if chroma_ok and db_ok:
        log.info("delete_success", file_id=request.file_id)
        return {"message": "Document deleted."}
    elif db_ok and not chroma_ok:
        log.warning("delete_partial", file_id=request.file_id)
        return {"warning": "Removed from database but failed to remove from vector store."}
    else:
        raise HTTPException(status_code=500, detail="Failed to delete document.")


# ── Analytics / audit / logs ──────────────────────────────────────────────────

@app.get("/analytics", dependencies=[Depends(verify_api_key)])
def get_analytics(user_id: str = Depends(get_current_user)):
    try:
        return get_query_stats(user_id=user_id)
    except Exception as e:
        log.error("analytics_error", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to load analytics.")


@app.get("/audit-log", dependencies=[Depends(verify_api_key)])
def audit_log(limit: int = 100, user_id: str = Depends(get_current_user)):
    try:
        return get_audit_log(user_id=user_id, limit=limit)
    except Exception as e:
        log.error("audit_log_error", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to load audit log.")


@app.get("/logs", dependencies=[Depends(verify_api_key)])
def get_logs(level: str = None, limit: int = 100):
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "app.log")
    if not os.path.exists(log_path):
        return {"logs": [], "total": 0}

    with open(log_path, "r") as f:
        lines = f.readlines()

    parsed = []
    for line in lines:
        try:
            parsed.append(json.loads(line.strip()))
        except Exception:
            continue

    if level:
        parsed = [l for l in parsed if l.get("level") == level.upper()]

    parsed = parsed[-limit:][::-1]
    return {"logs": parsed, "total": len(parsed)}


# ── Bulk questionnaire ────────────────────────────────────────────────────────

@app.post("/answer-questionnaire", dependencies=[Depends(verify_api_key)])
@_limit("2/minute")
async def answer_questionnaire(
    request: Request,
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user),
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="CSV file is empty.")

    try:
        reader = csv.DictReader(io.StringIO(content.decode("utf-8")))
        rows = list(reader)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not parse CSV. Check file encoding.")

    if not rows:
        raise HTTPException(status_code=400, detail="CSV has no rows.")
    if "question" not in rows[0]:
        raise HTTPException(status_code=400, detail="CSV must have a 'question' column.")
    if len(rows) > MAX_QUESTIONNAIRE_ROWS:
        raise HTTPException(status_code=400,
            detail=f"CSV too large. Maximum {MAX_QUESTIONNAIRE_ROWS} rows allowed.")

    results = []

    async def process_row(row):
        question = row.get("question", "").strip()
        if not question:
            return None
        async with QUESTIONNAIRE_SEMAPHORE:
            try:
                from graph.workflow import run_workflow
                state = await run_workflow(user_id, question)
                brief = state.get("brief") or {}
                faithfulness = brief.get("faithfulness_score", 0.0)
                sources = [s["filename"] for s in brief.get("sources", [])]
                # Build a one-line answer from the brief
                parts = [i["claim"] for i in brief.get("issues", [])[:2]]
                parts += [r["claim"] for r in brief.get("risks", [])[:2]]
                answer = "; ".join(parts) if parts else brief.get("summary", "No findings.")
                return {
                    "question": question,
                    "answer": answer,
                    "confidence": faithfulness,
                    "sources": sources,
                    "needs_review": faithfulness < CONFIDENCE_THRESHOLD,
                    "error": None,
                }
            except Exception as e:
                log.error("bulk_question_failed", question=question, error=str(e))
                return {
                    "question": question,
                    "answer": "Failed to get answer — retry manually.",
                    "confidence": 0.0,
                    "sources": [],
                    "needs_review": True,
                    "error": str(e),
                }

    tasks = [process_row(row) for row in rows]
    raw = await asyncio.gather(*tasks)
    results = [r for r in raw if r is not None]

    return {
        "results": results,
        "total": len(results),
        "needs_review_count": sum(1 for r in results if r["needs_review"]),
    }
