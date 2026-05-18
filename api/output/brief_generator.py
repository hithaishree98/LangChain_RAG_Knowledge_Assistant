"""
output/brief_generator.py — Assembles PreMeetingBrief from graph state sections.

Converts raw dict lists from section nodes into Pydantic models.
Applies 3-layer hallucination check per claim.
Detects stale sources (doc_date > as_of_date - 30 days).
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

_log = logging.getLogger(__name__)

from pydantic_models import (
    PreMeetingBrief,
    Commitment,
    OpenItem,
    RecentChange,
    AnticipatedQuestion,
    PostureDirective,
    SourceCitation,
    ClaimVerification,
    Conflict,
)
from langchain_utils import detect_hallucination, classify_claims, llm_judge_claims
from utils.staleness import is_stale as _is_stale


# ── Source / staleness helpers ────────────────────────────────────────────────

def _make_source(item: dict, as_of_date: str) -> SourceCitation:
    """Build a SourceCitation from a raw dict item, flagging stale docs."""
    doc_date = item.get("doc_date") or ""
    return SourceCitation(
        document=item.get("source_doc") or item.get("filename") or "unknown",
        doc_date=doc_date,
        location=item.get("location") or "",
        is_stale=_is_stale(doc_date, as_of_date),
    )


# ── Section converters ────────────────────────────────────────────────────────

def _dict_to_commitment(item: dict, as_of_date: str) -> Commitment:
    source = _make_source(item, as_of_date)
    return Commitment(
        description=item.get("description") or "",
        promised_date=item.get("promised_date") or None,
        target_date=item.get("target_date") or item.get("current_target_date") or None,
        status=item.get("status") or "open",
        owner=item.get("owner") or None,
        is_slipped=bool(item.get("is_slipped")),
        is_overdue=bool(item.get("is_overdue")),
        customer_aware=bool(item.get("customer_aware")),
        source=source,
    )


def _dict_to_open_item(item: dict, as_of_date: str) -> OpenItem:
    source = _make_source(item, as_of_date)
    if source.document in ("unknown", "", None):
        verification = ClaimVerification(verified=False, flag="no_source")
    elif source.is_stale:
        verification = ClaimVerification(verified=True, flag="stale_source")
    else:
        verification = ClaimVerification(verified=True, flag=None)
    return OpenItem(
        title=item.get("title") or "",
        status=item.get("status") or "open",
        last_update=item.get("last_update") or None,
        owner=item.get("owner") or None,
        priority=item.get("priority") or "normal",
        source=source,
        verification=verification,
    )


def _dict_to_recent_change(item: dict, as_of_date: str) -> RecentChange:
    source = _make_source(item, as_of_date)
    return RecentChange(
        what=item.get("what") or item.get("description") or "",
        date=item.get("date") or item.get("doc_date") or "",
        source=source,
        customer_aware=bool(item.get("customer_aware")),
    )


def _dict_to_anticipated_question(item: dict, as_of_date: str) -> AnticipatedQuestion:
    source = _make_source(item, as_of_date)
    return AnticipatedQuestion(
        topic=item.get("topic") or "",
        evidence=item.get("evidence") or "",
        source_quote=item.get("source_quote") or None,
        source=source,
        urgency=item.get("urgency") or "medium",
    )


def _dict_to_posture_directive(item: dict) -> PostureDirective:
    return PostureDirective(
        verb=item.get("verb") or "Lead",
        directive=item.get("directive") or "",
        basis=item.get("basis") or "",
        grounding_item=item.get("grounding_item") or None,
    )


# ── 3-layer hallucination check for open items ────────────────────────────────

def _run_hallucination_check_on_open_items(
    open_items: List[OpenItem],
    all_docs: List[Any],
) -> List[OpenItem]:
    """Run classify_claims + optional llm_judge on open_item titles in aggregate.

    Updates verification.flag to "verify_before_quoting" for flagged items.
    Items already flagged "stale_source" keep that flag.
    """
    if not open_items or not all_docs:
        return open_items

    # Collect titles (claims) to classify
    titles = [item.title for item in open_items if item.title]
    if not titles:
        return open_items

    classification = classify_claims(titles, all_docs)

    # Build a set of titles that are regex-flagged
    regex_flagged_titles = {entry["claim"] for entry in classification.get("flagged_by_regex", [])}

    # Run LLM judge on needs_judge claims if enabled
    llm_flagged_titles: set = set()
    needs_judge = classification.get("needs_judge", [])
    if needs_judge and os.getenv("ENABLE_LLM_JUDGE", "1") not in ("0", "false", "False"):
        judge_output = llm_judge_claims(needs_judge, all_docs)
        for unsupported in judge_output.get("unsupported", []):
            llm_flagged_titles.add(unsupported["claim"])

    # Update verification flags
    updated = []
    for item in open_items:
        if item.title in regex_flagged_titles or item.title in llm_flagged_titles:
            # Don't downgrade an existing stale_source flag — but do add verify flag
            if item.verification.flag is None:
                item = item.model_copy(update={
                    "verification": ClaimVerification(verified=True, flag="verify_before_quoting")
                })
        updated.append(item)

    return updated


# ── Main assembly function ─────────────────────────────────────────────────────

def generate_pre_meeting_brief(state: Dict[str, Any]) -> PreMeetingBrief:
    """Assemble a PreMeetingBrief from the populated GraphState.

    Reads raw dict lists written by section nodes, converts each to its
    Pydantic model, runs hallucination checks on open_items, collects
    stale warnings, and returns the fully typed PreMeetingBrief.
    """
    as_of_date = state.get("as_of_date") or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    last_call_date = state.get("last_call_date") or None

    # ── Overdue commitments ───────────────────────────────────────────────────
    overdue_commitments: List[Commitment] = [
        _dict_to_commitment(item, as_of_date)
        for item in (state.get("overdue_commitments_data") or [])
        if isinstance(item, dict)
    ]

    # ── Account summary ───────────────────────────────────────────────────────
    account_summary: str = (
        state.get("account_summary_text")
        or state.get("account_summary")         # legacy key from old workflow
        or "Insufficient data to assess posture."
    )

    # ── Open items — with hallucination check ─────────────────────────────────
    # Collect all docs from retrieved_chunks + parent_chunks for verification
    all_docs = list(state.get("parent_chunks") or []) + list(state.get("retrieved_chunks") or [])

    raw_open_items: List[OpenItem] = [
        _dict_to_open_item(item, as_of_date)
        for item in (state.get("open_items_data") or [])
        if isinstance(item, dict)
    ]
    # Deduplicate by normalized title — same ticket can appear in multiple chunks
    _seen_titles: set = set()
    _deduped: List[OpenItem] = []
    for _item in raw_open_items:
        _key = (_item.title or "").strip().lower()
        if _key and _key in _seen_titles:
            continue
        if _key:
            _seen_titles.add(_key)
        _deduped.append(_item)
    raw_open_items = _deduped

    open_items = _run_hallucination_check_on_open_items(raw_open_items, all_docs)

    # ── Recent changes ────────────────────────────────────────────────────────
    recent_changes: List[RecentChange] = [
        _dict_to_recent_change(item, as_of_date)
        for item in (state.get("recent_changes_data") or [])
        if isinstance(item, dict)
    ]

    # ── Outstanding commitments ───────────────────────────────────────────────
    outstanding_commitments: List[Commitment] = [
        _dict_to_commitment(item, as_of_date)
        for item in (state.get("outstanding_commitments_data") or [])
        if isinstance(item, dict)
    ]

    # ── Anticipated questions ─────────────────────────────────────────────────
    anticipated_questions: List[AnticipatedQuestion] = [
        _dict_to_anticipated_question(item, as_of_date)
        for item in (state.get("anticipated_questions_data") or [])
        if isinstance(item, dict)
    ]

    # ── Recommended posture ───────────────────────────────────────────────────
    recommended_posture: List[PostureDirective] = []
    for item in (state.get("posture_directives_data") or []):
        if not isinstance(item, dict):
            continue
        try:
            recommended_posture.append(_dict_to_posture_directive(item))
        except Exception:
            pass  # Skip malformed PostureDirective (bad verb, etc.)

    # ── Stale warnings ────────────────────────────────────────────────────────
    stale_warnings: List[str] = list(state.get("stale_warnings") or [])

    # Add warnings for any items with stale sources not already captured
    for item in open_items:
        if item.source.is_stale:
            warning = f"Open item '{item.title[:60]}' cites document older than 30 days ({item.source.document})."
            if warning not in stale_warnings:
                stale_warnings.append(warning)

    for item in recent_changes:
        if item.source.is_stale:
            warning = f"Recent change '{item.what[:60]}' cites document older than 30 days ({item.source.document})."
            if warning not in stale_warnings:
                stale_warnings.append(warning)

    for item in overdue_commitments + outstanding_commitments:
        if item.source.is_stale:
            warning = f"Commitment '{item.description[:60]}' cites document older than 30 days ({item.source.document})."
            if warning not in stale_warnings:
                stale_warnings.append(warning)

    # ── Conflicts ─────────────────────────────────────────────────────────────
    conflicts: List[Conflict] = []
    conflict_sources: set = set()   # filenames that appear in any conflict
    for raw in (state.get("conflicts_raw") or []):
        if not isinstance(raw, dict):
            continue
        try:
            sa_raw = raw.get("source_a") or {}
            sb_raw = raw.get("source_b") or {}
            claim_a = raw.get("claim") or sa_raw.get("claim") or raw.get("topic") or ""
            claim_b = sb_raw.get("claim") or ""
            if not claim_a or not claim_b:
                _log.warning("conflict_skipped_empty_claim raw=%s", str(raw)[:120])
                continue
            source_a = SourceCitation(
                document=sa_raw.get("chunk_id") or sa_raw.get("document") or "unknown",
                doc_date=sa_raw.get("doc_date") or "",
            )
            source_b = SourceCitation(
                document=sb_raw.get("chunk_id") or sb_raw.get("document") or "unknown",
                doc_date=sb_raw.get("doc_date") or "",
            )
            conflicts.append(Conflict(
                claim_a=claim_a,
                source_a=source_a,
                claim_b=claim_b,
                source_b=source_b,
            ))
            conflict_sources.add(source_a.document)
            conflict_sources.add(source_b.document)
        except Exception:
            pass

    # Promote conflicts to open items whose source doc is in a conflict
    if conflict_sources:
        promoted = []
        for item in open_items:
            if item.source.document in conflict_sources and item.verification.flag is None:
                item = item.model_copy(update={
                    "verification": ClaimVerification(verified=False, flag="conflict")
                })
            promoted.append(item)
        open_items = promoted

    # ── Section status ────────────────────────────────────────────────────────
    section_status: Dict[str, str] = dict(state.get("section_status") or {})

    # ── Section source provenance ─────────────────────────────────────────────
    section_sources: Dict[str, List[str]] = {
        "overdue_commitments":    state.get("overdue_sources") or [],
        "open_items":             state.get("open_items_sources") or [],
        "recent_changes":         state.get("recent_changes_sources") or [],
        "outstanding_commitments": state.get("outstanding_sources") or [],
        "account_summary":        state.get("account_summary_sources") or [],
        "anticipated_questions":  state.get("anticipated_questions_sources") or [],
    }
    section_data_as_of: Dict[str, str] = {
        k: v for k, v in {
            "overdue_commitments":    state.get("overdue_as_of") or "",
            "open_items":             state.get("open_items_as_of") or "",
            "recent_changes":         state.get("recent_changes_as_of") or "",
            "outstanding_commitments": state.get("outstanding_as_of") or "",
            "account_summary":        state.get("account_summary_as_of") or "",
        }.items() if v
    }

    # ── Corpus warning ────────────────────────────────────────────────────────
    corpus_warning: str | None = None
    health = state.get("corpus_health") or {}
    if health.get("overall") == "empty":
        corpus_warning = (
            "No documents uploaded for this customer. "
            "Upload a transcript, ticket export, and commitment tracker to generate a useful brief."
        )
    elif health.get("overall") == "stale":
        stale_types = [
            dt for dt, info in (health.get("doc_types") or {}).items()
            if isinstance(info, dict) and info.get("status") == "stale"
        ]
        if stale_types:
            corpus_warning = (
                f"Corpus is stale: {', '.join(stale_types)} not updated in 30+ days. "
                "Upload fresh documents before this call for accurate results."
            )

    return PreMeetingBrief(
        overdue_commitments=overdue_commitments,
        account_summary=account_summary,
        open_items=open_items,
        recent_changes=recent_changes,
        outstanding_commitments=outstanding_commitments,
        anticipated_questions=anticipated_questions,
        recommended_posture=recommended_posture,
        as_of_date=as_of_date,
        last_call_date=last_call_date,
        stale_warnings=stale_warnings,
        conflicts=conflicts,
        corpus_health=health,
        section_status=section_status,
        section_sources=section_sources,
        section_data_as_of=section_data_as_of,
        corpus_warning=corpus_warning,
    )


def generate_brief(state: Dict[str, Any]) -> Dict[str, Any]:
    """Backward-compat wrapper: assemble PreMeetingBrief and return as dict.

    Called by workflow.py's generate_brief_node, which stores the result in
    state["brief"]. Callers expecting the old six-section dict format will
    still find the same top-level keys via model.dict().
    """
    brief = generate_pre_meeting_brief(state)
    return brief.model_dump()

