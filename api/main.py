from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status, Request
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic_models import QueryInput, QueryResponse, DocumentInfo, DeleteFileRequest
from langchain_utils import get_rag_chain, calculate_confidence, extract_sources
from chroma_utils import index_document_to_chroma, delete_doc_from_chroma, vectorstore, get_retriever_for_user
from db_utils import (
    insert_application_logs, get_chat_history, get_all_documents,
    insert_document_record, delete_document_record,
    get_query_stats, get_audit_log
)
from notification_utils import send_to_slack
import os, uuid, time, csv, io, json, logging
from datetime import datetime, timezone


# Simple structured logger that writes JSON to file and plain text to console
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
            **kwargs
        }
        getattr(self.logger, level.lower())(json.dumps(entry))

    def info(self, event, **kwargs):    self._write("INFO", event, **kwargs)
    def warning(self, event, **kwargs): self._write("WARNING", event, **kwargs)
    def error(self, event, **kwargs):   self._write("ERROR", event, **kwargs)
    def debug(self, event, **kwargs):   self._write("DEBUG", event, **kwargs)


log = StructuredLogger("rag_api")

app = FastAPI(title="FDE Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

CONFIDENCE_THRESHOLD = 0.4
MAX_QUESTION_LENGTH = 1000
MAX_FILE_SIZE_MB = 10
ALLOWED_EXTENSIONS = [".pdf", ".docx", ".html"]
MAX_RETRIES = 2

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def verify_api_key(api_key: str = Depends(api_key_header)):
    expected = os.getenv("API_KEY")
    if not expected:
        return
    if api_key != expected:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="Invalid or missing API key")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error("unhandled_exception",
        path=str(request.url),
        method=request.method,
        error=str(exc),
        error_type=type(exc).__name__
    )
    return JSONResponse(status_code=500, content={
        "message": "Something went wrong. Please try again."
    })


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
    checks["llm_key"] = "ok" if os.getenv("GROQ_API_KEY") else "missing"
    checks["slack"] = "configured" if os.getenv("SLACK_WEBHOOK_URL") else "not configured"
    all_ok = all(v in ("ok", "configured", "not configured") for v in checks.values())
    return {"status": "healthy" if all_ok else "degraded", "checks": checks}


@app.post("/chat", response_model=QueryResponse, dependencies=[Depends(verify_api_key)])
def chat(query_input: QueryInput):
    user_id = query_input.user_id
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
            escalated=True
        )

    session_id = query_input.session_id or str(uuid.uuid4())
    log.info("chat_request", session_id=session_id, question_length=len(question))

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            retriever = get_retriever_for_user(user_id)
            rag_chain = get_rag_chain(query_input.model.value, retriever=retriever)
            chat_history = get_chat_history(session_id, user_id=user_id)
            result = rag_chain.invoke({"input": question, "chat_history": chat_history})
            break
        except Exception as e:
            last_error = e
            log.warning("llm_retry", session_id=session_id, attempt=attempt + 1, error=str(e))
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
    else:
        log.error("llm_failed", session_id=session_id, error=str(last_error))
        raise HTTPException(status_code=503,
            detail="AI service temporarily unavailable. Please try again.")

    answer = result["answer"]
    retrieved_docs = result.get("context", [])
    confidence = calculate_confidence(answer, retrieved_docs)
    sources = extract_sources(retrieved_docs)
    escalated = confidence < CONFIDENCE_THRESHOLD

    try:
        insert_application_logs(
            session_id, question, answer, query_input.model.value,
            confidence, escalated, ", ".join(sources), user_id=user_id
        )
    except Exception as e:
        log.error("db_write_failed", session_id=session_id, error=str(e))

    if query_input.notify_slack:
        try:
            send_to_slack(question, answer, sources, confidence, session_id)
        except Exception as e:
            log.error("slack_failed", session_id=session_id, error=str(e))

    log.info("chat_response", session_id=session_id, confidence=confidence, escalated=escalated)

    return QueryResponse(
        answer=answer,
        session_id=session_id,
        model=query_input.model,
        confidence=confidence,
        sources=sources,
        escalated=escalated
    )


@app.post("/upload-doc")
async def upload_and_index_document(file: UploadFile = File(...), user_id: str = "default"):
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

    existing = [doc["filename"] for doc in get_all_documents(user_id=user_id)]
    if file.filename in existing:
        raise HTTPException(status_code=409,
            detail=f"'{file.filename}' already exists. Delete it first or rename the file.")

    temp_path = f"temp_{uuid.uuid4()}_{file.filename}"

    try:
        with open(temp_path, "wb") as f:
            f.write(contents)

        log.info("upload_started", filename=file.filename, size_mb=round(size_mb, 2))

        file_id = insert_document_record(file.filename, user_id=user_id)
        success = index_document_to_chroma(temp_path, file_id, user_id=user_id)

        if not success:
            delete_document_record(file_id)
            raise HTTPException(status_code=500,
                detail=f"Failed to index '{file.filename}'. The file may be corrupted.")

        log.info("upload_success", filename=file.filename, file_id=file_id)
        return {"message": f"'{file.filename}' uploaded successfully.", "file_id": file_id}

    except HTTPException:
        raise
    except Exception as e:
        log.error("upload_error", filename=file.filename, error=str(e))
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/list-docs", response_model=list[DocumentInfo])
def list_documents(user_id: str = "default"):
    try:
        return get_all_documents(user_id=user_id)
    except Exception as e:
        log.error("list_docs_error", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve documents.")


@app.post("/delete-doc")
def delete_document(request: DeleteFileRequest):
    existing = get_all_documents(user_id=request.user_id)
    if not any(doc["id"] == request.file_id for doc in existing):
        raise HTTPException(status_code=404, detail="Document not found in your workspace.")

    chroma_ok = delete_doc_from_chroma(request.file_id, user_id=request.user_id)
    db_ok = delete_document_record(request.file_id, user_id=request.user_id)

    if chroma_ok and db_ok:
        log.info("delete_success", file_id=request.file_id)
        return {"message": "Document deleted."}
    elif db_ok and not chroma_ok:
        log.warning("delete_partial", file_id=request.file_id)
        return {"warning": "Removed from database but failed to remove from vector store."}
    else:
        raise HTTPException(status_code=500, detail="Failed to delete document.")


@app.get("/analytics")
def get_analytics(user_id: str = "default"):
    try:
        return get_query_stats(user_id=user_id)
    except Exception as e:
        log.error("analytics_error", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to load analytics.")


@app.get("/audit-log")
def audit_log(limit: int = 100, user_id: str = "default"):
    try:
        return get_audit_log(user_id=user_id, limit=limit)
    except Exception as e:
        log.error("audit_log_error", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to load audit log.")


@app.get("/logs")
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


@app.post("/answer-questionnaire")
async def answer_questionnaire(file: UploadFile = File(...), user_id: str = "default"):
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

    retriever = get_retriever_for_user(user_id)
    rag_chain = get_rag_chain(retriever=retriever)
    results = []

    for row in rows:
        question = row.get("question", "").strip()
        if not question:
            continue
        try:
            result = rag_chain.invoke({"input": question, "chat_history": []})
            answer = result["answer"]
            retrieved = result.get("context", [])
            confidence = calculate_confidence(answer, retrieved)
            sources = extract_sources(retrieved)
            results.append({
                "question": question,
                "answer": answer,
                "confidence": confidence,
                "sources": sources,
                "needs_review": confidence < CONFIDENCE_THRESHOLD,
                "error": None
            })
        except Exception as e:
            log.error("bulk_question_failed", question=question, error=str(e))
            results.append({
                "question": question,
                "answer": "Failed to get answer — retry manually.",
                "confidence": 0.0,
                "sources": [],
                "needs_review": True,
                "error": str(e)
            })

    return {
        "results": results,
        "total": len(results),
        "needs_review_count": sum(1 for r in results if r["needs_review"])
    }