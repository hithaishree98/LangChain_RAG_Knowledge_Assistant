"""
output/exec_brief_generator.py — Assembles ExecBrief from exec graph state sections.

Reads the exec-specific state keys written by exec section nodes and converts
them into the typed ExecBrief Pydantic model.
Detects stale sources (signal/ask doc older than 30 days) and surfaces conflicts
from conflicts_raw populated by section nodes.
"""

from typing import Any, Dict, List

from pydantic import BaseModel

from pydantic_models import (
    ExecBrief,
    PersonStatement,
    Signal,
    Ask,
    SourceCitation,
    ClaimVerification,
    Conflict,
)
from utils.staleness import is_stale as _is_stale, STALE_DAYS as _STALE_DAYS


class _RawStatement(BaseModel):
    content: str = ""
    said_by: str = "person"
    stated_date: str = ""
    sentiment: str = "neutral"
    source_doc: str = ""
    doc_date: str = ""


class _RawSignal(BaseModel):
    event: str = ""
    date: str = ""
    source_doc: str = ""


class _RawAsk(BaseModel):
    ask: str = ""
    date: str = ""
    status: str = "open"
    source_doc: str = ""


def generate_exec_brief(state: Dict[str, Any]) -> ExecBrief:
    """Assemble an ExecBrief from the populated GraphState.

    Reads exec_role_tenure, exec_stated_position, exec_recent_signals,
    exec_open_asks, and exec_recommended_approach from state and returns
    a fully typed ExecBrief.
    """
    as_of_date = state.get("as_of_date") or ""

    role_and_tenure = state.get("exec_role_tenure") or ""

    # ── Stated position ───────────────────────────────────────────────────────
    stated_position = []
    for raw in (state.get("exec_stated_position") or []):
        if not isinstance(raw, dict):
            continue
        try:
            item = _RawStatement.model_validate(raw)
        except Exception:
            continue
        if not item.source_doc or item.source_doc == "unknown":
            continue  # drop items with no source evidence
        source = SourceCitation(
            document=item.source_doc,
            doc_date=item.doc_date,
        )
        stated_position.append(PersonStatement(
            content=item.content,
            said_by=item.said_by,
            source=source,
        ))

    # ── Recent signals ────────────────────────────────────────────────────────
    recent_signals = []
    for raw in (state.get("exec_recent_signals") or []):
        if not isinstance(raw, dict):
            continue
        try:
            item = _RawSignal.model_validate(raw)
        except Exception:
            continue
        if not item.source_doc or item.source_doc == "unknown":
            continue  # drop items with no source evidence
        source = SourceCitation(
            document=item.source_doc,
            doc_date=item.date,
            is_stale=_is_stale(item.date, as_of_date),
        )
        recent_signals.append(Signal(
            event=item.event,
            date=item.date,
            source=source,
        ))

    # ── Open asks ─────────────────────────────────────────────────────────────
    open_asks = []
    for raw in (state.get("exec_open_asks") or []):
        if not isinstance(raw, dict):
            continue
        try:
            item = _RawAsk.model_validate(raw)
        except Exception:
            continue
        if not item.source_doc or item.source_doc == "unknown":
            continue  # drop items with no source evidence
        source = SourceCitation(
            document=item.source_doc,
            doc_date=item.date,
            is_stale=_is_stale(item.date, as_of_date),
        )
        open_asks.append(Ask(
            ask=item.ask,
            date=item.date,
            status=item.status,
            source=source,
        ))

    recommended_approach = state.get("exec_recommended_approach") or ""

    # ── Stale warnings ────────────────────────────────────────────────────────
    stale_warnings: List[str] = list(state.get("stale_warnings") or [])
    for sig in recent_signals:
        if sig.source.is_stale:
            w = f"Signal '{sig.event[:60]}' cites document older than {_STALE_DAYS} days ({sig.source.document})."
            if w not in stale_warnings:
                stale_warnings.append(w)
    for ask in open_asks:
        if ask.source.is_stale:
            w = f"Ask '{ask.ask[:60]}' cites document older than {_STALE_DAYS} days ({ask.source.document})."
            if w not in stale_warnings:
                stale_warnings.append(w)

    # ── Conflicts ─────────────────────────────────────────────────────────────
    conflicts: List[Conflict] = []
    for raw in (state.get("conflicts_raw") or []):
        if not isinstance(raw, dict):
            continue
        try:
            sa_raw = raw.get("source_a") or {}
            sb_raw = raw.get("source_b") or {}
            conflicts.append(Conflict(
                claim_a=raw.get("claim_a") or raw.get("claim") or "",
                source_a=SourceCitation(
                    document=sa_raw.get("document") or sa_raw.get("chunk_id") or "unknown",
                    doc_date=sa_raw.get("doc_date") or "",
                ),
                claim_b=raw.get("claim_b") or sb_raw.get("claim") or "",
                source_b=SourceCitation(
                    document=sb_raw.get("document") or sb_raw.get("chunk_id") or "unknown",
                    doc_date=sb_raw.get("doc_date") or "",
                ),
            ))
        except Exception:
            pass

    return ExecBrief(
        role_and_tenure=role_and_tenure,
        stated_position=stated_position,
        recent_signals=recent_signals,
        open_asks=open_asks,
        recommended_approach=recommended_approach,
        stale_warnings=stale_warnings,
        conflicts=conflicts,
    )
