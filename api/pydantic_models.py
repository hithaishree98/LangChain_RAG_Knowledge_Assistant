from pydantic import BaseModel, Field, field_validator
from typing import Any, Dict, List, Optional
from datetime import datetime


# ── Shared building blocks ────────────────────────────────────────────────────

class SourceCitation(BaseModel):
    document: str
    doc_date: str
    location: str = ""
    is_stale: bool = False
    is_latest_version: bool = True


class ClaimVerification(BaseModel):
    verified: bool = True
    flag: Optional[str] = None
    # flag: None | "stale_source" | "verify_before_quoting" | "conflict"


class Conflict(BaseModel):
    claim_a: str
    source_a: SourceCitation
    claim_b: str
    source_b: SourceCitation


# ── Pre-meeting brief sections ────────────────────────────────────────────────

class OpenItem(BaseModel):
    title: str
    status: str
    last_update: Optional[str] = None
    owner: Optional[str] = None
    priority: str = "normal"
    source: SourceCitation
    verification: ClaimVerification = ClaimVerification()


class RecentChange(BaseModel):
    what: str
    date: str
    source: SourceCitation
    customer_aware: bool = False


class Commitment(BaseModel):
    description: str
    promised_date: Optional[str] = None
    target_date: Optional[str] = None
    status: str
    owner: Optional[str] = None
    is_slipped: bool = False
    is_overdue: bool = False
    customer_aware: bool = False
    source: SourceCitation


class AnticipatedQuestion(BaseModel):
    topic: str
    evidence: str
    source_quote: Optional[str] = None   # verbatim phrase from source doc
    source: SourceCitation
    urgency: str = "medium"

    @field_validator("urgency")
    @classmethod
    def validate_urgency(cls, v):
        if v not in ("high", "medium", "low"):
            raise ValueError(f"urgency must be high/medium/low, got {v!r}")
        return v


class PostureDirective(BaseModel):
    verb: str
    directive: str
    basis: str
    grounding_item: Optional[str] = None  # specific ticket/commitment that drives this

    @field_validator("verb")
    @classmethod
    def validate_verb(cls, v):
        if v not in ("Lead", "Acknowledge", "Defer", "Push"):
            raise ValueError(f"verb must be Lead/Acknowledge/Defer/Push, got {v!r}")
        return v


# ── PreMeetingBrief ────────────────────────────────────────────────────────────

class PreMeetingBriefRequest(BaseModel):
    customer_id: str
    as_of_date: Optional[str] = None


class PreMeetingBrief(BaseModel):
    overdue_commitments: List[Commitment] = Field(default_factory=list)
    account_summary: str = ""
    open_items: List[OpenItem] = Field(default_factory=list)
    recent_changes: List[RecentChange] = Field(default_factory=list)
    outstanding_commitments: List[Commitment] = Field(default_factory=list)
    anticipated_questions: List[AnticipatedQuestion] = Field(default_factory=list)
    recommended_posture: List[PostureDirective] = Field(default_factory=list)
    as_of_date: str
    last_call_date: Optional[str] = None
    stale_warnings: List[str] = Field(default_factory=list)
    conflicts: List[Conflict] = Field(default_factory=list)
    corpus_health: Dict[str, Any] = Field(default_factory=dict)
    section_status: Dict[str, str] = Field(default_factory=dict)
    section_sources: Dict[str, List[str]] = Field(default_factory=dict)
    section_data_as_of: Dict[str, str] = Field(default_factory=dict)
    corpus_warning: Optional[str] = None


# ── Exec 1:1 brief ────────────────────────────────────────────────────────────

class ExecBriefRequest(BaseModel):
    customer_id: str
    person_id: str
    as_of_date: Optional[str] = None


class PersonStatement(BaseModel):
    content: str
    said_by: str = "other"           # "person" | "other"
    stated_date: Optional[str] = None   # YYYY-MM-DD from source document
    sentiment: Optional[str] = None     # positive | neutral | concern | request
    source: SourceCitation
    verification: ClaimVerification = ClaimVerification()


class Signal(BaseModel):
    event: str
    date: str
    source: SourceCitation


class Ask(BaseModel):
    ask: str
    date: str
    status: str = "open"
    source: SourceCitation


class ExecBrief(BaseModel):
    role_and_tenure: str = ""
    stated_position: List[PersonStatement] = Field(default_factory=list)
    recent_signals: List[Signal] = Field(default_factory=list)
    open_asks: List[Ask] = Field(default_factory=list)
    recommended_approach: str = ""
    stale_warnings: List[str] = Field(default_factory=list)
    conflicts: List[Conflict] = Field(default_factory=list)


# ── Query ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    customer_id: Optional[str] = None


class QueryResult(BaseModel):
    answer: Optional[str] = None
    answer_status: str = "not_found"
    citation: Optional[SourceCitation] = None
    citations: List[SourceCitation] = Field(default_factory=list)
    answer_as_of: Optional[str] = None
    recency_flag: Optional[str] = None
    confidence_explanation: Optional[str] = None
    sources_searched: int = 0
    conflicts: List[Conflict] = Field(default_factory=list)
    missing_doc_types: List[str] = Field(default_factory=list)


# ── Customer management ────────────────────────────────────────────────────────

class TokenRequest(BaseModel):
    workspace: str
    passkey: str


class CustomerCreate(BaseModel):
    name: str
    slug: str

    @field_validator("slug")
    @classmethod
    def validate_slug(cls, v):
        import re
        v = v.lower().strip()
        if not re.match(r'^[a-z0-9-]+$', v):
            raise ValueError("slug must be lowercase alphanumeric and hyphens only")
        return v


class CustomerResponse(BaseModel):
    id: int
    name: str
    slug: str
    last_call_date: Optional[str] = None
    created_at: str


class CorpusHealth(BaseModel):
    doc_types: Dict[str, Dict] = Field(default_factory=dict)
    overall: str = "empty"
    last_call_date: Optional[str] = None
    missing_doc_types: List[str] = Field(default_factory=list)


class AccountHealth(BaseModel):
    health_score: int                        # 0–100
    health_band: str                         # "Healthy" | "At Risk" | "Critical"
    open_p0_count: int = 0
    open_p1_count: int = 0
    overdue_commitment_count: int = 0
    total_open_commitments: int = 0
    slipped_commitment_count: int = 0
    total_commitments: int = 0
    commitment_slip_rate: float = 0.0        # 0.0–1.0
    days_since_last_call: Optional[int] = None
    missing_doc_types: List[str] = Field(default_factory=list)


class PersonCreate(BaseModel):
    name: str
    role: Optional[str] = None
    email: Optional[str] = None


# ── Feedback ──────────────────────────────────────────────────────────────────

class BriefFeedback(BaseModel):
    brief_log_id: int
    customer_id: str
    section: str
    rating: int
    flagged_claim: Optional[str] = None

    @field_validator("rating")
    @classmethod
    def validate_rating(cls, v):
        if v not in (1, -1):
            raise ValueError("rating must be 1 or -1")
        return v


# ── Document management ───────────────────────────────────────────────────────

class DocumentInfo(BaseModel):
    id: int
    filename: str
    upload_timestamp: datetime
    user_id: str


