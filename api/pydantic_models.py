from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from enum import Enum
from datetime import datetime


class ModelName(str, Enum):
    GROQ_DEFAULT = "llama-3.1-8b-instant"


class QueryInput(BaseModel):
    question: str
    session_id: Optional[str] = None
    model: ModelName = ModelName.GROQ_DEFAULT
    user_id: str = "default"


class QueryResponse(BaseModel):
    answer: str
    session_id: str
    model: ModelName
    confidence: float
    sources: List[str]
    escalated: bool


class DocumentInfo(BaseModel):
    id: int
    filename: str
    upload_timestamp: datetime
    user_id: str


class DeleteFileRequest(BaseModel):
    file_id: int


# ── Brief endpoints ───────────────────────────────────────────────────────────

class BriefRequest(BaseModel):
    query: str
    customer_id: Optional[str] = None  # falls back to JWT user_id if not set


class BriefResponse(BaseModel):
    brief: Dict[str, Any]
    sources: List[Dict[str, str]]
    faithfulness_score: float
    # Explicit schema-level field so OpenAPI documents it and Pydantic
    # serialization never silently drops it. Mirror of brief["judge_status"].
    # Values: ok | no_claims | skipped_breaker_open | no_context_all_unsupported
    #       | parse_error | error | disabled
    judge_status: str = "disabled"
    loop_count: int
    audit_trail: List[Dict[str, Any]]