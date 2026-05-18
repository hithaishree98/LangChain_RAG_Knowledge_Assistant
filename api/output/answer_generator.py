"""
output/answer_generator.py — Format the /query workflow's answer_node output.

Takes the GraphState after answer_node and builds a QueryResult-shaped dict:
  answer, answer_status, citation, citations, answer_as_of,
  confidence_explanation, sources_searched, recency_flag,
  conflicts, missing_doc_types.

Uses 3-layer hallucination check (regex → classify → llm_judge) and maps the
result to a confidence_explanation string the FDE can read directly.
"""

import logging
import os
import re as _re
from typing import Any, Dict, List, Optional

from langchain_utils import (
    detect_hallucination,
    classify_claims,
    llm_judge_claims,
    should_run_judge,
)
from utils.staleness import recency_flag as _recency_flag

_log = logging.getLogger(__name__)

# Doc types that are relevant to common query intents — used to populate
# missing_doc_types when a query's answer isn't found in the corpus.
_QUERY_DOC_TYPE_HINTS: List[tuple] = [
    (("sla", "uptime", "availability", "commitment", "promised", "deliver"), "commitment_tracker"),
    (("ticket", "bug", "issue", "incident", "outage", "error", "p0", "p1"), "ticket"),
    (("call", "meeting", "said", "transcript", "discussed", "agreed"), "transcript"),
    (("architecture", "design", "integration", "solution", "infra"), "solution_architecture"),
    (("notes", "account", "crm", "internal"), "account_notes"),
]


def _infer_missing_doc_types(query: str, retrieved_docs: list, customer_id: str = "") -> List[str]:
    """Return doc types likely needed to answer this query but not in the indexed corpus.

    Queries the actual indexed corpus for this customer (via vectorstore) rather than
    checking only the current retrieval result — a doc type that was uploaded but not
    retrieved for this query should not be flagged as missing.
    """
    query_lower = query.lower()

    corpus_types: set = set()
    if customer_id:
        try:
            from chroma_utils import vectorstore
            result = vectorstore._collection.get(
                where={"$and": [
                    {"user_id": {"$eq": customer_id}},
                    {"is_latest_version": {"$eq": 1}},
                ]},
                include=["metadatas"],
            )
            corpus_types = {
                (m.get("doc_type") or "") for m in (result.get("metadatas") or [])
            }
        except Exception:
            corpus_types = {d.metadata.get("doc_type", "") for d in retrieved_docs}
    else:
        corpus_types = {d.metadata.get("doc_type", "") for d in retrieved_docs}

    missing = []
    for keywords, doc_type in _QUERY_DOC_TYPE_HINTS:
        if any(kw in query_lower for kw in keywords) and doc_type not in corpus_types:
            if doc_type not in missing:
                missing.append(doc_type)
    return missing[:3]


def _build_citation(doc: Any) -> Optional[Dict[str, Any]]:
    """Build a SourceCitation-shaped dict from a Document."""
    if doc is None:
        return None
    filename = (
        doc.metadata.get("filename")
        or os.path.basename(doc.metadata.get("source", ""))
        or "unknown"
    )
    doc_date = doc.metadata.get("doc_date") or ""
    location = doc.metadata.get("location") or doc.metadata.get("chunk_id") or ""
    return {
        "document": filename,
        "doc_date": doc_date,
        "location": location,
        "is_stale": False,
        "is_latest_version": True,
    }


def _build_all_citations(
    ao: Dict[str, Any], chunk_map: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Build one citation per unique source document that contributed to the answer.

    Only citations whose chunk_id exists in chunk_map (the actual retrieved set) are
    included. LLM-fabricated chunk_ids that don't match any retrieved chunk are logged
    and silently dropped so forged citations never reach the user.
    """
    seen_files: set = set()
    citations: List[Dict[str, Any]] = []

    # Detect and log forged chunk IDs (present in LLM output but not retrieved)
    raw_cids = [
        c.get("chunk_id") for c in (ao.get("citations") or [])
        if isinstance(c, dict) and c.get("chunk_id")
    ]
    forged = [cid for cid in raw_cids if cid not in chunk_map]
    if forged:
        _log.warning("citation_validation_forged_ids count=%d ids=%s",
                     len(forged), str(forged[:5]))

    # Walk citations in answer_output order so the primary source comes first
    for citation_ref in (ao.get("citations") or []):
        if not isinstance(citation_ref, dict):
            continue
        cid = citation_ref.get("chunk_id")
        doc = chunk_map.get(cid) if cid else None
        if doc is None:
            continue
        fname = doc.metadata.get("filename") or os.path.basename(doc.metadata.get("source", ""))
        if fname and fname not in seen_files:
            seen_files.add(fname)
            c = _build_citation(doc)
            if c:
                c["is_stale"] = _recency_flag(c["doc_date"]) == "stale"
                citations.append(c)

    # If the LLM cited no chunk_ids, fall back to all docs in the result set
    if not citations:
        for doc in list(chunk_map.values())[:4]:
            fname = doc.metadata.get("filename") or ""
            if fname and fname not in seen_files:
                seen_files.add(fname)
                c = _build_citation(doc)
                if c:
                    c["is_stale"] = _recency_flag(c["doc_date"]) == "stale"
                    citations.append(c)

    return citations


def generate_answer(state: Dict[str, Any]) -> Dict[str, Any]:
    """Convert state['answer_output'] into a QueryResult-shaped dict.

    Returns:
        {
            "answer": str,
            "answer_status": "ok" | "not_found" | "partial" | "error",
            "citation": SourceCitation-shaped dict or None,   # primary (compat)
            "citations": [SourceCitation-shaped dict, ...],   # all sources
            "answer_as_of": "YYYY-MM-DD" | None,
            "confidence_explanation": str | None,
            "sources_searched": int,
            "recency_flag": "stale" | None,
            "conflicts": [],
            "missing_doc_types": [],
        }
    """
    ao = state.get("answer_output") or {}
    parent_chunks = state.get("parent_chunks") or []
    child_chunks = state.get("retrieved_chunks") or []
    query = state.get("original_query", "")
    customer_id = state.get("customer_id", "")
    # reference_date: the calendar date this answer is computed FOR (today)
    reference_date: Optional[str] = state.get("today_date") or state.get("as_of_date") or None

    # ── Short-circuit on system failures ─────────────────────────────────────
    if ao.get("_parse_error") or ao.get("_breaker_open"):
        return {
            "answer": ao.get("answer", ""),
            "answer_status": "error",
            "citation": None,
            "citations": [],
            "answer_as_of": None,
            "reference_date": reference_date,
            "latest_source_date": None,
            "confidence_explanation": "AI service temporarily unavailable — try again shortly.",
            "sources_searched": len(parent_chunks) or len(child_chunks),
            "recency_flag": None,
            "conflicts": [],
            "missing_doc_types": [],
        }

    # ── Hard not_found: no docs retrieved at all ──────────────────────────────
    if not parent_chunks and not child_chunks:
        missing = _infer_missing_doc_types(query, [], customer_id=customer_id)
        explanation = "No documents were found for this customer."
        if missing:
            explanation += f" Upload a {missing[0]} to answer this type of question."
        return {
            "answer": "",
            "answer_status": "not_found",
            "citation": None,
            "citations": [],
            "answer_as_of": None,
            "reference_date": reference_date,
            "latest_source_date": None,
            "confidence_explanation": explanation,
            "sources_searched": 0,
            "recency_flag": None,
            "conflicts": [],
            "missing_doc_types": missing,
        }

    # ── Build chunk lookup (parent ids take precedence) ───────────────────────
    chunk_map: Dict[str, Any] = {}
    for doc in child_chunks:
        cid = doc.metadata.get("chunk_id")
        if cid:
            chunk_map[cid] = doc
    for doc in parent_chunks:
        cid = doc.metadata.get("chunk_id")
        if cid:
            chunk_map[cid] = doc

    all_docs = list(chunk_map.values())
    answer_text: str = ao.get("answer", "")
    answer_status: str = ao.get("answer_status", "ok")

    # ── Hard not_found from LLM ───────────────────────────────────────────────
    if answer_status == "not_found" or not answer_text.strip():
        missing = _infer_missing_doc_types(query, all_docs, customer_id=customer_id)
        explanation = "The uploaded documents don't contain a clear answer to this question."
        if missing:
            explanation += f" Consider uploading a {missing[0]}."
        return {
            "answer": "",
            "answer_status": "not_found",
            "citation": None,
            "citations": [],
            "answer_as_of": None,
            "reference_date": reference_date,
            "latest_source_date": None,
            "confidence_explanation": explanation,
            "sources_searched": len(all_docs),
            "recency_flag": None,
            "conflicts": [],
            "missing_doc_types": missing,
        }

    # ── Build all citations ───────────────────────────────────────────────────
    all_citations = _build_all_citations(ao, chunk_map)
    primary_citation = all_citations[0] if all_citations else None

    # ── Date handling: split reference date (today) from source date ─────────
    # latest_source_date: max doc_date across all cited sources (WHEN the content is from)
    # reference_date:     today (WHEN the answer is computed for) — set earlier from state
    citation_dates = [c["doc_date"] for c in all_citations if c.get("doc_date")]
    latest_source_date: Optional[str] = max(citation_dates) if citation_dates else (
        ao.get("answer_date") or None
    )
    answer_as_of = latest_source_date  # backward compat: callers expecting answer_as_of get source date

    # ── Recency flag on primary citation ─────────────────────────────────────
    doc_recency: Optional[str] = None
    if primary_citation:
        doc_recency = _recency_flag(primary_citation["doc_date"])
        primary_citation["is_stale"] = doc_recency == "stale"

    # ── 3-layer hallucination check ───────────────────────────────────────────
    # Layer 1: regex — logs suspicious facts as side effect
    detect_hallucination(answer_text, all_docs)

    # Layer 2: classify claims into verified / flagged / needs_judge
    claim_texts = [
        c.get("claim", "") for c in (ao.get("citations") or [])
        if isinstance(c, dict) and c.get("claim")
    ]
    if not claim_texts:
        claim_texts = [
            s.strip() for s in _re.split(r"[.!?]", answer_text)
            if len(s.strip()) > 20
        ]
    classification = classify_claims(claim_texts, all_docs)

    flagged_by_regex = classification.get("flagged_by_regex", [])
    has_regex_issues = len(flagged_by_regex) > 0

    # Layer 3: conditional LLM judge
    needs_judge = classification.get("needs_judge", [])
    conflicts: List[Dict[str, Any]] = []
    has_llm_issues = False
    judge_output = None

    enable_judge = os.getenv("ENABLE_LLM_JUDGE", "1") not in ("0", "false", "False")
    if enable_judge and needs_judge:
        if should_run_judge(query, 0.0, len(needs_judge), always_run=False):
            judge_output = llm_judge_claims(needs_judge, all_docs)
            conflicts = judge_output.get("conflicts", []) or []
            has_llm_issues = len(judge_output.get("unsupported", [])) > 0

    # ── Build confidence_explanation ──────────────────────────────────────────
    explanation_parts: List[str] = []
    if flagged_by_regex:
        unsupported_facts = [
            f for item in flagged_by_regex
            for f in (item.get("unsupported_facts") or [])
        ]
        if unsupported_facts:
            display = ", ".join(f'"{f}"' for f in unsupported_facts[:3])
            explanation_parts.append(
                f"These specific facts weren't found in the source documents: {display}"
            )
    if has_llm_issues and judge_output is not None:
        unsupported = judge_output.get("unsupported", [])
        if unsupported:
            first = unsupported[0].get("claim", "")[:100]
            explanation_parts.append(
                f'This claim couldn\'t be verified: "{first}"'
            )
    if doc_recency == "stale" and primary_citation:
        explanation_parts.append(
            f"Most relevant source is over 30 days old ({primary_citation['doc_date']})"
        )
    confidence_explanation = " | ".join(explanation_parts) if explanation_parts else None

    # ── Downgrade status when hallucination signals found ────────────────────
    if answer_status == "ok" and (has_regex_issues or has_llm_issues):
        answer_status = "partial"

    # ── missing_doc_types (only for partial/not_found) ────────────────────────
    missing_doc_types: List[str] = []
    if answer_status in ("partial", "not_found"):
        missing_doc_types = _infer_missing_doc_types(query, all_docs, customer_id=customer_id)

    return {
        "answer": answer_text,
        "answer_status": answer_status,
        "citation": primary_citation,
        "citations": all_citations,
        "answer_as_of": answer_as_of,          # backward compat: latest source date
        "reference_date": reference_date,       # date the answer was computed for (today)
        "latest_source_date": latest_source_date,  # max doc_date across cited sources
        "confidence_explanation": confidence_explanation,
        "sources_searched": len(all_docs),
        "recency_flag": doc_recency,
        "conflicts": conflicts,
        "missing_doc_types": missing_doc_types,
    }
