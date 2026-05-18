import asyncio
import hashlib
import json
import logging
import os
import secrets
import tempfile
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader

from pydantic_models import (
    PreMeetingBriefRequest, PreMeetingBrief,
    ExecBriefRequest, ExecBrief,
    QueryRequest, QueryResult,
    CustomerCreate, CustomerResponse, CorpusHealth, AccountHealth,
    PersonCreate, BriefFeedback,
    TokenRequest,
    DocumentInfo,
)
from langchain_utils import llm_breaker
from chroma_utils import index_document_to_chroma, delete_doc_from_chroma, vectorstore, get_account_health_metrics, get_latest_chunks_by_doctype
from db_utils import (
    get_all_documents, document_exists,
    insert_document_record, delete_document_record,
    get_query_stats, get_audit_log, run_migrations, insert_brief_log,
    set_latest_version_flag,
    create_customer, get_customers, get_customer_by_slug, delete_customer,
    update_last_call_date, get_corpus_health,
    add_person, get_people,
    insert_brief_feedback,
)

from utils.doc_type_utils import (
    VALID_DOC_TYPES, infer_doc_type, validate_filename, extract_date_from_filename,
    check_content_descriptor_consistency, sniff_doc_type,
)


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

# Holds references to background notification tasks so they aren't GC'd mid-flight.
_background_tasks: set = set()


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

    # Start APScheduler for daily overdue digest
    _scheduler = None
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from scripts.daily_overdue_check import run as _run_overdue_check
        _scheduler = AsyncIOScheduler()
        _scheduler.add_job(_run_overdue_check, "cron", hour=8, minute=0,
                           id="daily_overdue_check", replace_existing=True)
        _scheduler.start()
        log.info("scheduler_started")
    except ImportError:
        log.warning("apscheduler_not_installed scheduler_disabled")
    except Exception as e:
        log.warning("scheduler_start_failed", error=str(e))

    yield

    # Graceful shutdown: let any running scheduled job finish before the process exits.
    if _scheduler is not None:
        try:
            _scheduler.shutdown(wait=True)
            log.info("scheduler_stopped")
        except Exception as e:
            log.warning("scheduler_shutdown_failed", error=str(e))


app = FastAPI(title="FDE Assistant API", lifespan=lifespan)

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:8501").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "X-API-Key", "Authorization"],
)

# ── Rate limiter (slowapi) ────────────────────────────────────────────────────
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

_RATELIMIT_ENABLED = os.getenv("TESTING_DISABLE_RATELIMIT", "0") != "1"
_limiter = Limiter(key_func=get_remote_address, enabled=_RATELIMIT_ENABLED)
app.state.limiter = _limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


def _limit(rate: str):
    return _limiter.limit(rate)


# ── Constants ─────────────────────────────────────────────────────────────────

MAX_QUESTION_LENGTH = 1000
MAX_FILE_SIZE_MB = 10
ALLOWED_EXTENSIONS = [".pdf", ".docx", ".html", ".txt", ".json", ".csv"]

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
    # Surface which embedding provider the API will use. "openai" = hosted
    # (fast, requires OPENAI_API_KEY); "huggingface" = local sentence-transformer
    # fallback (slower on CPU, free, no network). Either is healthy.
    try:
        from chroma_utils import get_embedding_provider
        checks["embedding_provider"] = get_embedding_provider()
    except Exception as e:
        checks["embedding_provider"] = f"error: {str(e)[:60]}"
    checks["slack"] = "configured" if os.getenv("SLACK_WEBHOOK_URL") else "not configured"
    from langchain_utils import _CBState
    checks["circuit_breaker"] = llm_breaker.state.value
    _OK_VALUES = {"ok", "configured", "not configured", "openai", "huggingface",
                  _CBState.CLOSED.value}
    all_ok = all(v in _OK_VALUES for v in checks.values())
    return {"status": "healthy" if all_ok else "degraded", "checks": checks}


# ── Auth endpoint ─────────────────────────────────────────────────────────────

@app.post("/auth/token")
@_limit("5/minute")
async def get_token(request: Request, body: TokenRequest):  # request is used by slowapi for IP-based limiting
    """Issue a short-lived JWT for a workspace. user_id = sha256(workspace:passkey)."""
    workspace = body.workspace.strip().lower()
    passkey = body.passkey.strip()
    if not workspace or not passkey:
        raise HTTPException(status_code=400, detail="workspace and passkey are required")
    raw = f"{workspace}:{passkey}"
    user_id = hashlib.sha256(raw.encode()).hexdigest()[:32]
    token = _create_token(user_id)
    return {"token": token, "user_id": user_id}


# ── Customer CRUD ─────────────────────────────────────────────────────────────

@app.post("/customers", response_model=CustomerResponse, dependencies=[Depends(verify_api_key)])
async def create_customer_endpoint(req: CustomerCreate, user_id: str = Depends(get_current_user)):
    """Create a new customer workspace scoped to the authenticated FDE."""
    import sqlite3
    try:
        row = create_customer(req.name, req.slug, fde_user_id=user_id)
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409,
                            detail=f"Customer slug '{req.slug}' already exists.")
    except Exception as e:
        log.error("create_customer_failed", error=str(e), user_id=user_id)
        raise HTTPException(status_code=500, detail="Failed to create customer.")
    return CustomerResponse(**row)


@app.get("/customers", response_model=List[CustomerResponse], dependencies=[Depends(verify_api_key)])
async def list_customers(user_id: str = Depends(get_current_user)):
    """List all customers belonging to the authenticated FDE."""
    try:
        rows = get_customers(user_id)
    except Exception as e:
        log.error("list_customers_failed", error=str(e), user_id=user_id)
        raise HTTPException(status_code=500, detail="Failed to retrieve customers.")
    return [CustomerResponse(**r) for r in rows]


@app.delete("/customers/{slug}", dependencies=[Depends(verify_api_key)])
async def delete_customer_endpoint(slug: str, user_id: str = Depends(get_current_user)):
    """Delete a customer workspace and all associated data. Only the owning FDE may delete."""
    try:
        file_ids = delete_customer(slug, user_id)
    except Exception as e:
        log.error("delete_customer_failed", slug=slug, error=str(e), user_id=user_id)
        raise HTTPException(status_code=500, detail="Failed to delete customer.")
    if file_ids is None:
        raise HTTPException(status_code=404, detail=f"Customer '{slug}' not found.")
    for file_id in file_ids:
        try:
            delete_doc_from_chroma(file_id, user_id=slug)
        except Exception as e:
            log.warning("chroma_cleanup_partial", file_id=file_id, slug=slug, error=str(e))
    log.info("customer_deleted", slug=slug, user_id=user_id, docs_removed=len(file_ids))
    return {"deleted": True}


@app.get("/customers/{customer_id}/corpus-health", response_model=CorpusHealth,
         dependencies=[Depends(verify_api_key)])
async def get_corpus_health_endpoint(customer_id: str, user_id: str = Depends(get_current_user)):
    """Return per-doc-type freshness for a customer's corpus."""
    if get_customer_by_slug(customer_id, user_id) is None:
        raise HTTPException(status_code=404, detail=f"Customer '{customer_id}' not found.")
    try:
        health = get_corpus_health(customer_id)
    except Exception as e:
        log.error("corpus_health_failed", customer_id=customer_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve corpus health.")
    return CorpusHealth(**health)


@app.get("/customers/{customer_id}/health-score", response_model=AccountHealth,
         dependencies=[Depends(verify_api_key)])
async def get_health_score_endpoint(customer_id: str, user_id: str = Depends(get_current_user)):
    """Return computed account health KPIs. No LLM calls — pure metadata aggregation."""
    if get_customer_by_slug(customer_id, user_id) is None:
        raise HTTPException(status_code=404, detail=f"Customer '{customer_id}' not found.")
    try:
        metrics = get_account_health_metrics(customer_id)
        corpus = get_corpus_health(customer_id)

        days_since_call: int | None = None
        last_call = corpus.get("last_call_date")
        if last_call:
            try:
                delta = (date.today() - date.fromisoformat(last_call)).days
                days_since_call = delta
                # Penalise the health score for stale contact (>14 days)
                if delta > 14:
                    penalty = min(20, (delta - 14) * 2)
                    metrics["health_score"] = max(0, metrics["health_score"] - penalty)
                    if metrics["health_score"] >= 75:
                        metrics["health_band"] = "Healthy"
                    elif metrics["health_score"] >= 45:
                        metrics["health_band"] = "At Risk"
                    else:
                        metrics["health_band"] = "Critical"
            except ValueError:
                pass

        return AccountHealth(
            **metrics,
            days_since_last_call=days_since_call,
            missing_doc_types=corpus.get("missing_doc_types", []),
        )
    except Exception as e:
        log.error("health_score_failed", customer_id=customer_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to compute health score.")


@app.post("/customers/{customer_id}/people", dependencies=[Depends(verify_api_key)])
async def add_person_endpoint(
    customer_id: str,
    req: PersonCreate,
    user_id: str = Depends(get_current_user),
):
    """Add a stakeholder/person to a customer workspace."""
    customer = get_customer_by_slug(customer_id, user_id)
    if customer is None:
        raise HTTPException(status_code=404,
                            detail=f"Customer '{customer_id}' not found.")
    customer_numeric_id = customer["id"]
    try:
        person = add_person(customer_numeric_id, req.name, req.role, req.email)
    except Exception as e:
        import sqlite3
        if isinstance(e, sqlite3.IntegrityError):
            raise HTTPException(
                status_code=409,
                detail=f"A person named '{req.name}' already exists for this customer.",
            )
        log.error("add_person_failed", customer_id=customer_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to add person.")
    return person


@app.get("/customers/{customer_id}/people", dependencies=[Depends(verify_api_key)])
def list_people_endpoint(
    customer_id: str,
    user_id: str = Depends(get_current_user),
):
    """List all stakeholders/people for a customer workspace."""
    customer = get_customer_by_slug(customer_id, user_id)
    if customer is None:
        raise HTTPException(status_code=404, detail=f"Customer '{customer_id}' not found.")
    try:
        from db_utils import get_people
        return get_people(customer["id"])
    except Exception as e:
        log.error("list_people_failed", customer_id=customer_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to list people.")


# ── Document upload (per-customer) ────────────────────────────────────────────

@app.post("/customers/{customer_id}/upload", dependencies=[Depends(verify_api_key)])
@_limit("20/minute")
async def upload_document(
    request: Request,
    customer_id: str,
    file: UploadFile = File(...),
    doc_type: str = Form(None),
    replace: str = Form("false"),
    user_id: str = Depends(get_current_user),
):
    """Upload and index a document into a specific customer's corpus.

    Filename must follow the convention: YYYY-MM-DD_<keyword>_<descriptor>.<ext>
    doc_type is optional; when omitted it is inferred from the filename.
    """
    # 0. Ownership check
    if get_customer_by_slug(customer_id, user_id) is None:
        raise HTTPException(status_code=404, detail=f"Customer '{customer_id}' not found.")

    # 1. Extension check
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

    # 2. Read and validate file size
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(status_code=400,
            detail=f"File too large ({size_mb:.1f}MB). Maximum is {MAX_FILE_SIZE_MB}MB.")

    # 3. Resolve doc_type
    # Priority: (1) caller-supplied explicit value, (2) filename keyword, (3) content sniff.
    # Files no longer need to follow the date-prefix naming convention — users can upload
    # files with any name and the system will detect the type automatically.
    filename = os.path.basename(file.filename)
    sniff_detected: str | None = None
    sniff_confidence: str = "low"

    if doc_type:
        try:
            from utils.doc_type_utils import normalize_doc_type
            doc_type = normalize_doc_type(doc_type)
        except ValueError:
            raise HTTPException(status_code=400,
                detail=f"doc_type must be one of: {sorted(VALID_DOC_TYPES)}")
    else:
        # Try filename keyword first (fast, no I/O)
        doc_type = infer_doc_type(filename)
        if doc_type is None:
            # Fall back to content sniffing
            sniff_detected, sniff_confidence = sniff_doc_type(contents, filename)
            doc_type = sniff_detected
        if doc_type is None:
            raise HTTPException(status_code=400,
                detail=(
                    f"Cannot determine document type for '{filename}'. "
                    f"Please select a type from the dropdown: "
                    f"{sorted(VALID_DOC_TYPES)}"
                ))

    # 4. Filename convention check — advisory only, does not block uploads.
    # Users can upload any filename; the convention is encouraged but not required.
    upload_warnings: list = []
    valid, err_msg = validate_filename(filename)
    if not valid:
        upload_warnings.append(
            f"Tip: renaming files as YYYY-MM-DD_<type>_<descriptor>.ext improves "
            f"automatic date detection (e.g. 2024-09-15_transcript_status-call.txt)."
        )

    # 4b. Lightweight content/descriptor consistency check (warning, does not block)
    if doc_type in ("transcript", "commitment_tracker", "ticket"):
        try:
            content_sample = contents[:4096].decode("utf-8", errors="ignore")
            consistency_warnings = check_content_descriptor_consistency(filename, doc_type, content_sample)
            upload_warnings.extend(consistency_warnings)
            for w in consistency_warnings:
                log.warning("upload_content_filename_mismatch", filename=filename,
                            customer_id=customer_id, detail=w)
        except Exception:
            pass

    # 5. Duplicate filename check
    replace_existing = replace.lower() in ("true", "1", "yes")
    existing_docs = get_all_documents(user_id=customer_id)
    duplicate = next((d for d in existing_docs if d["filename"] == filename), None)
    if duplicate:
        if not replace_existing:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"A document named '{filename}' already exists for this customer. "
                    f"Set replace=true to overwrite it."
                ),
            )
        # Delete the old record so the new upload becomes the sole version
        delete_doc_from_chroma(duplicate["id"], user_id=customer_id)
        delete_document_record(duplicate["id"], user_id=customer_id)
        log.info("upload_replaced_existing", filename=filename, old_file_id=duplicate["id"],
                 customer_id=customer_id)

    log.info("upload_started", filename=filename, customer_id=customer_id,
             doc_type=doc_type, size_mb=round(size_mb, 2), user_id=user_id)

    suffix = ext
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    temp_path = tmp.name

    try:
        tmp.write(contents)
        tmp.close()

        # 6. Extract doc_date from filename
        doc_date = extract_date_from_filename(filename)

        # 7. Insert DB record
        file_id = insert_document_record(
            filename, user_id=customer_id, doc_type=doc_type, doc_date=doc_date
        )

        # 8. Index into Chroma — must succeed before we stamp the version flag.
        # If we flagged first and indexing then failed, the DB record gets rolled
        # back but the version flag already flipped prior uploads to is_latest=0,
        # leaving the corpus without any "latest" version for this doc_type.
        index_summary = index_document_to_chroma(
            temp_path, file_id, user_id=customer_id, filename=filename, doc_type=doc_type
        )

        if not index_summary:
            delete_document_record(file_id, user_id=customer_id)
            raise HTTPException(status_code=500,
                detail=f"Failed to index '{filename}'. "
                       f"The file may be empty after format-aware filtering, or corrupted.")

        # 9. Mark this as the latest version only after indexing succeeds,
        #    then demote prior Chroma chunks so retrieval filters stay in sync.
        set_latest_version_flag(customer_id, doc_type, file_id)
        try:
            from chroma_utils import demote_old_versions_in_chroma
            demote_old_versions_in_chroma(customer_id, doc_type, file_id, filename=filename)
        except Exception as e:
            log.warning("chroma_demotion_failed", customer_id=customer_id,
                        doc_type=doc_type, file_id=file_id, error=str(e))

        log.info("upload_success", filename=filename, file_id=file_id,
                 customer_id=customer_id, **index_summary)

        # 10. Invalidate cache for this customer
        try:
            from utils.cache_utils import invalidate_customer
            invalidate_customer(customer_id)
        except Exception as e:
            log.warning("cache_invalidate_failed", customer_id=customer_id, error=str(e))

        # 11. Notify on transcript upload (non-blocking)
        if doc_type == "transcript":
            async def _notify():
                try:
                    from notification_utils import notify_transcript_uploaded
                    await notify_transcript_uploaded(customer_id, filename)
                except Exception as exc:
                    log.warning("transcript_notify_failed", customer_id=customer_id,
                                filename=filename, error=str(exc))
            task = asyncio.create_task(_notify())
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)

        response: dict = {
            "file_id": file_id,
            "filename": filename,
            "doc_type": doc_type,
            "doc_date": doc_date,
            "chunks": index_summary.get("child_chunks", 0),
        }
        if upload_warnings:
            response["warnings"] = upload_warnings
        return response

    except HTTPException:
        raise
    except Exception as e:
        log.error("upload_error", filename=filename, customer_id=customer_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ── Customer-scoped document list and delete ──────────────────────────────────

@app.get("/customers/{customer_id}/documents", dependencies=[Depends(verify_api_key)])
def list_customer_documents(
    customer_id: str,
    user_id: str = Depends(get_current_user),
):
    """List all documents uploaded to a specific customer workspace."""
    if get_customer_by_slug(customer_id, user_id) is None:
        raise HTTPException(status_code=404, detail=f"Customer '{customer_id}' not found.")
    try:
        return get_all_documents(user_id=customer_id)
    except Exception as e:
        log.error("list_customer_docs_error", customer_id=customer_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve documents.")


@app.delete("/customers/{customer_id}/documents/{file_id}", dependencies=[Depends(verify_api_key)])
def delete_customer_document(
    customer_id: str,
    file_id: int,
    user_id: str = Depends(get_current_user),
):
    """Delete a document from a specific customer workspace."""
    if get_customer_by_slug(customer_id, user_id) is None:
        raise HTTPException(status_code=404, detail=f"Customer '{customer_id}' not found.")
    if not document_exists(file_id, user_id=customer_id):
        raise HTTPException(status_code=404, detail="Document not found in this customer workspace.")
    chroma_ok = delete_doc_from_chroma(file_id, user_id=customer_id)
    db_ok = delete_document_record(file_id, user_id=customer_id)
    if chroma_ok and db_ok:
        log.info("customer_delete_success", file_id=file_id, customer_id=customer_id)
        return {"message": "Document deleted."}
    elif db_ok and not chroma_ok:
        log.warning("customer_delete_partial", file_id=file_id, customer_id=customer_id)
        return {"message": "Document removed. Vector store cleanup incomplete.", "partial": True}
    else:
        raise HTTPException(status_code=500, detail="Failed to delete document.")


# ── Brief: pre-meeting ────────────────────────────────────────────────────────

@app.post("/brief/pre-meeting", dependencies=[Depends(verify_api_key)])
@_limit("10/minute")
async def pre_meeting_brief(
    request: Request,
    req: PreMeetingBriefRequest,
    user_id: str = Depends(get_current_user),
):
    """Generate a structured pre-meeting brief for a customer."""
    if llm_breaker.is_open():
        raise HTTPException(status_code=503,
            detail="AI service temporarily unavailable. Please try again later.")

    as_of_date = req.as_of_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Check cache
    try:
        from utils.cache_utils import get_cached, set_cached
        cached = get_cached(req.customer_id, as_of_date, "pre_meeting")
        if cached is not None:
            log.info("pre_meeting_cache_hit", customer_id=req.customer_id, as_of_date=as_of_date)
            return cached
    except Exception as e:
        log.warning("cache_read_failed", error=str(e))

    # Verify ownership and get last_call_date
    customer = get_customer_by_slug(req.customer_id, user_id)
    if customer is None:
        raise HTTPException(status_code=404, detail=f"Customer '{req.customer_id}' not found.")
    last_call_date = customer.get("last_call_date")

    log.info("pre_meeting_brief_start", customer_id=req.customer_id, as_of_date=as_of_date,
             last_call_date=last_call_date)

    try:
        from graph.workflow import run_pre_meeting_workflow
        final_state = await run_pre_meeting_workflow(
            req.customer_id, as_of_date, last_call_date
        )
    except Exception as e:
        log.error("pre_meeting_workflow_failed", customer_id=req.customer_id, error=str(e))
        raise HTTPException(status_code=503,
            detail=f"Brief generation failed: {str(e)}")

    # Build brief object
    brief = final_state.get("brief")
    if brief is None:
        brief = PreMeetingBrief(
            as_of_date=as_of_date,
            section_status=final_state.get("section_status", {}),
        )

    # Convert to dict for storage and response
    if hasattr(brief, "model_dump"):
        brief_dict = brief.model_dump()
    else:
        brief_dict = brief if isinstance(brief, dict) else {}

    # Log to DB
    try:
        insert_brief_log(req.customer_id, "pre_meeting", json.dumps(brief_dict))
    except Exception as e:
        log.error("brief_log_failed", customer_id=req.customer_id, error=str(e))

    # Sync last_call_date from most recent transcript to DB so health score can compute days-since-call
    try:
        t_chunks = get_latest_chunks_by_doctype(req.customer_id, "transcript")
        dates = [c.metadata.get("doc_date", "") for c in t_chunks if c.metadata.get("doc_date")]
        if dates:
            update_last_call_date(customer["id"], max(dates))
    except Exception as e:
        log.warning("last_call_date_sync_failed", customer_id=req.customer_id, error=str(e))

    # Cache the result
    try:
        set_cached(req.customer_id, as_of_date, "pre_meeting", brief_dict)
    except Exception as e:
        log.warning("cache_write_failed", error=str(e))

    log.info("pre_meeting_brief_complete", customer_id=req.customer_id, as_of_date=as_of_date)
    return brief_dict


# ── Brief: exec 1:1 ──────────────────────────────────────────────────────────

@app.post("/brief/exec-1on1", dependencies=[Depends(verify_api_key)])
@_limit("10/minute")
async def exec_1on1_brief(
    request: Request,
    req: ExecBriefRequest,
    user_id: str = Depends(get_current_user),
):
    """Generate an exec 1:1 brief for a specific person at a customer."""
    if llm_breaker.is_open():
        raise HTTPException(status_code=503,
            detail="AI service temporarily unavailable. Please try again later.")

    as_of_date = req.as_of_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Check cache
    cache_key = f"exec_{req.person_id}"
    try:
        from utils.cache_utils import get_cached, set_cached
        cached = get_cached(req.customer_id, as_of_date, cache_key)
        if cached is not None:
            log.info("exec_1on1_cache_hit", customer_id=req.customer_id, person_id=req.person_id)
            return cached
    except Exception as e:
        log.warning("cache_read_failed", error=str(e))

    # Verify ownership and get last_call_date
    customer = get_customer_by_slug(req.customer_id, user_id)
    if customer is None:
        raise HTTPException(status_code=404, detail=f"Customer '{req.customer_id}' not found.")
    last_call_date = customer.get("last_call_date")

    # Verify person belongs to this customer (prevents cross-customer data access)
    try:
        from db_utils import get_person_by_id
        person = get_person_by_id(int(req.person_id), req.customer_id)
    except (ValueError, TypeError):
        person = None
    if person is None:
        raise HTTPException(status_code=404,
            detail=f"Person '{req.person_id}' not found for customer '{req.customer_id}'.")

    log.info("exec_1on1_brief_start", customer_id=req.customer_id, person_id=req.person_id,
             as_of_date=as_of_date)

    try:
        from graph.workflow import run_exec_1on1_workflow
        final_state = await run_exec_1on1_workflow(
            req.customer_id, req.person_id, as_of_date, last_call_date
        )
    except Exception as e:
        log.error("exec_1on1_workflow_failed", customer_id=req.customer_id,
                  person_id=req.person_id, error=str(e))
        raise HTTPException(status_code=503,
            detail=f"Exec brief generation failed: {str(e)}")

    result = final_state.get("exec_brief_result") or {}

    # Log to DB
    try:
        insert_brief_log(req.customer_id, f"exec_1on1:{req.person_id}", json.dumps(result))
    except Exception as e:
        log.error("brief_log_failed", customer_id=req.customer_id, error=str(e))

    # Cache
    try:
        set_cached(req.customer_id, as_of_date, cache_key, result)
    except Exception as e:
        log.warning("cache_write_failed", error=str(e))

    log.info("exec_1on1_brief_complete", customer_id=req.customer_id, person_id=req.person_id)
    return result


# ── Query ─────────────────────────────────────────────────────────────────────

@app.post("/query", dependencies=[Depends(verify_api_key)])
@_limit("20/minute")
async def query(
    request: Request,
    req: QueryRequest,
    user_id: str = Depends(get_current_user),
):
    """Single-pass focused Q&A against a customer's corpus."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")

    if not req.customer_id:
        raise HTTPException(status_code=400,
            detail="customer_id is required. Pass the customer slug in the request body.")

    if llm_breaker.is_open():
        raise HTTPException(status_code=503,
            detail="AI service temporarily unavailable. Please try again later.")

    # Verify the requested customer belongs to the authenticated FDE.
    customer = get_customer_by_slug(req.customer_id, user_id)
    if customer is None:
        raise HTTPException(status_code=404,
            detail=f"Customer '{req.customer_id}' not found.")

    customer_id = req.customer_id
    log.info("query_request", customer_id=customer_id, question_length=len(req.question))

    try:
        from graph.workflow import run_query_workflow
        final_state = await run_query_workflow(customer_id, req.question.strip())
    except Exception as e:
        log.error("query_workflow_failed", customer_id=customer_id, error=str(e))
        raise HTTPException(status_code=503, detail=f"Query failed: {str(e)}")

    payload = final_state.get("lookup_response")
    if not payload:
        log.error("query_empty_response", customer_id=customer_id)
        raise HTTPException(status_code=503, detail="Query workflow returned no result.")
    log.info("query_response", customer_id=customer_id,
             answer_status=payload.get("answer_status", "?"))
    return payload


# ── Brief feedback ────────────────────────────────────────────────────────────

@app.post("/brief/feedback", dependencies=[Depends(verify_api_key)])
async def brief_feedback(req: BriefFeedback, user_id: str = Depends(get_current_user)):
    """Record thumbs-up/down feedback on a brief section."""
    try:
        insert_brief_feedback(req.brief_log_id, req.customer_id, req.section, req.rating, req.flagged_claim)
    except Exception as e:
        log.error("brief_feedback_failed", brief_log_id=req.brief_log_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to record feedback.")
    return {"status": "recorded"}


# ── Documents: list / delete ──────────────────────────────────────────────────

@app.get("/documents", response_model=List[DocumentInfo], dependencies=[Depends(verify_api_key)])
def list_documents(user_id: str = Depends(get_current_user)):
    """List all documents for the authenticated user's workspace."""
    try:
        return get_all_documents(user_id=user_id)
    except Exception as e:
        log.error("list_docs_error", error=str(e), user_id=user_id)
        raise HTTPException(status_code=500, detail="Failed to retrieve documents.")


@app.delete("/documents/{file_id}", dependencies=[Depends(verify_api_key)])
def delete_document(file_id: int, user_id: str = Depends(get_current_user)):
    """Delete a document and its Chroma vectors from the workspace."""
    if not document_exists(file_id, user_id=user_id):
        raise HTTPException(status_code=404, detail="Document not found in your workspace.")

    chroma_ok = delete_doc_from_chroma(file_id, user_id=user_id)
    db_ok = delete_document_record(file_id, user_id=user_id)

    if chroma_ok and db_ok:
        log.info("delete_success", file_id=file_id, user_id=user_id)
        return {"message": "Document deleted."}
    elif db_ok and not chroma_ok:
        log.warning("delete_partial", file_id=file_id, user_id=user_id)
        return {"message": "Document removed. Vector store cleanup incomplete — re-indexing may help.",
                "partial": True}
    else:
        raise HTTPException(status_code=500, detail="Failed to delete document.")


# ── Stats / audit log ─────────────────────────────────────────────────────────

@app.get("/stats", dependencies=[Depends(verify_api_key)])
def get_stats(user_id: str = Depends(get_current_user)):
    """Return query statistics for the authenticated user's workspace."""
    try:
        return get_query_stats(user_id=user_id)
    except Exception as e:
        log.error("stats_error", error=str(e), user_id=user_id)
        raise HTTPException(status_code=500, detail="Failed to load stats.")


@app.get("/audit-log", dependencies=[Depends(verify_api_key)])
def audit_log(limit: int = 100, user_id: str = Depends(get_current_user)):
    """Return recent audit log entries for the authenticated user's workspace."""
    try:
        return get_audit_log(user_id=user_id, limit=limit)
    except Exception as e:
        log.error("audit_log_error", error=str(e), user_id=user_id)
        raise HTTPException(status_code=500, detail="Failed to load audit log.")


# ── Logs ──────────────────────────────────────────────────────────────────────

@app.get("/logs", dependencies=[Depends(verify_api_key)])
def get_logs(level: str = None,
             limit: int = 100,
             user_id: str = Depends(get_current_user)):
    """Return recent log entries scoped to the caller's workspace.

    Two behaviors that closed an earlier cross-tenant leak:
      1. Tenant filter — only entries that carry a matching ``user_id`` or
         ``customer_id`` (request-scoped event) are returned. Entries without
         a tenant tag (server lifecycle / startup / system events) are
         INCLUDED only for the unauthenticated dev "default" tenant; for any
         real workspace they are stripped, since they may reference other
         tenants' filenames or queries in passing.
      2. Tail-only read — we read at most the last LOGS_READ_BYTES bytes of
         app.log so a large log file can't OOM the API. Default ~512 KB.

    Authentication:
      - Bearer JWT REQUIRED to scope to a real workspace. Without it we fall
        through to "default" which only contains untagged events.
    """
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "app.log")
    if not os.path.exists(log_path):
        return {"logs": [], "total": 0}

    LOGS_READ_BYTES = 512 * 1024  # 512 KB tail
    LOGS_HARD_LIMIT = 500
    limit = max(1, min(int(limit or 100), LOGS_HARD_LIMIT))

    try:
        with open(log_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - LOGS_READ_BYTES))
            tail = f.read().decode("utf-8", errors="replace")
    except Exception as e:
        log.error("logs_read_failed", error=str(e))
        return {"logs": [], "total": 0}

    lines = tail.splitlines()
    if size > LOGS_READ_BYTES and lines:
        lines = lines[1:]

    parsed = []
    for line in lines:
        try:
            parsed.append(json.loads(line.strip()))
        except Exception:
            continue

    def _belongs_to_caller(entry: Dict[str, Any]) -> bool:
        tag = entry.get("user_id") or entry.get("customer_id")
        if user_id == "default":
            return tag is None or tag == "default"
        return tag == user_id

    parsed = [e for e in parsed if _belongs_to_caller(e)]

    if level:
        parsed = [l for l in parsed if l.get("level") == level.upper()]

    parsed = parsed[-limit:][::-1]
    return {"logs": parsed, "total": len(parsed)}


