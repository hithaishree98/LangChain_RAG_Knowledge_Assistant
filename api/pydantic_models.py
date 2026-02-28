from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum
from datetime import datetime


class ModelName(str, Enum):
    GROQ_DEFAULT = "llama-3.1-8b-instant"


class QueryInput(BaseModel):
    question: str
    session_id: Optional[str] = None
    model: ModelName = ModelName.GROQ_DEFAULT
    notify_slack: bool = False
    notify_email: Optional[str] = None
    user_id: str = "default"


class QueryResponse(BaseModel):
    answer: str
    session_id: str
    model: ModelName
    confidence: float
    sources: list[str]
    escalated: bool


class DocumentInfo(BaseModel):
    id: int
    filename: str
    upload_timestamp: datetime
    user_id: str


class DeleteFileRequest(BaseModel):
    file_id: int
    user_id: str = "default"