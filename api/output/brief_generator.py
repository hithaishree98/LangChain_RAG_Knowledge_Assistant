"""
output/brief_generator.py — Formats reasoning_output into a structured brief.

Takes the GraphState after reason_node and maps it to the brief schema:
  summary, issues[], risks[], open_questions[], talking_points[],
  information_gaps[], sources[]

Each cited item resolves chunk_id → source doc name + passage text.
Runs hallucination check before returning.
"""

import os
from typing import Any, Dict, List

from langchain_utils import detect_hallucination, calculate_faithfulness, classify_claims, llm_judge_claims


def generate_brief(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert state["reasoning_output"] into the final brief dict.
    Resolves chunk_ids, runs hallucination check, builds sources list.
    """

    ro = state.get("reasoning_output") or {}
    parent_chunks = state.get("parent_chunks") or []
    child_chunks = state.get("retrieved_chunks") or []

    # Build chunk_id → Document lookup.
    # Parent chunks are tagged P1, P2, ... by retrieve_node and take precedence.
    # Child chunks carry C1, C2, ... ids and act as fallback citations.
    chunk_map: Dict[str, Any] = {}
    for doc in child_chunks:
        cid = doc.metadata.get("chunk_id")
        if cid:
            chunk_map[cid] = doc
    for doc in parent_chunks:
        cid = doc.metadata.get("chunk_id")  # P-style id set by retrieve_node
        if cid:
            chunk_map[cid] = doc

    # ── Resolve issues ────────────────────────────────────────────────────────
    issues = []
    for item in ro.get("issues", []):
        issues.append(_resolve_cited_item(item, chunk_map, "claim"))

    # ── Resolve risks ─────────────────────────────────────────────────────────
    risks = []
    for item in ro.get("risks", []):
        risks.append(_resolve_cited_item(item, chunk_map, "claim"))

    # ── Open questions ────────────────────────────────────────────────────────
    open_questions = [q for q in ro.get("open_questions", []) if q]

    # ── Talking points ────────────────────────────────────────────────────────
    talking_points = []
    for item in ro.get("talking_points", []):
        talking_points.append(_resolve_cited_item(item, chunk_map, "point"))

    # ── Sources list ──────────────────────────────────────────────────────────
    all_docs = list(chunk_map.values())
    sources = _build_sources(all_docs)

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = _build_summary(issues, risks, open_questions, talking_points)

    # ── Hallucination check ───────────────────────────────────────────────────
    # ── Layer 1: Regex hallucination check (atomic facts) ────────────────────
    # Kept as-is for backwards compatibility. Flags individual fact patterns
    # (dates, amounts, versions, IDs) that appear in the claims but not context.
    all_text_items = (
        [i["claim"] for i in issues]
        + [r["claim"] for r in risks]
        + [t["point"] for t in talking_points]
    )
    full_text = " ".join(all_text_items)
    suspicious_facts = detect_hallucination(full_text, all_docs) if all_docs else []

    # ── Layer 2: Classify each claim → regex / flagged / needs-judge ─────────
    # Each claim is evaluated individually so we can route it to the right
    # verification tier. This avoids sending regex-verifiable claims to the LLM.
    classification = classify_claims(all_text_items, all_docs)

    # ── Layer 3: LLM-as-judge (only for claims regex cannot verify) ──────────
    # Single batched call — ALL needs-judge claims in one LLM request.
    # Disabled via env var ENABLE_LLM_JUDGE=0 if you want regex-only behavior.
    suspicious_claims: List[Dict[str, Any]] = []

    # Add claims that regex already flagged (no LLM call needed for these)
    for flagged in classification["flagged_by_regex"]:
        suspicious_claims.append({
            "claim": flagged["claim"],
            "reason": f"Regex found unsupported facts: {', '.join(flagged['unsupported_facts'])}",
            "caught_by": "regex",
        })

    # Run LLM judge on the remaining claims
    judge_status = "disabled"
    if os.getenv("ENABLE_LLM_JUDGE", "1") not in ("0", "false", "False"):
        judge_output = llm_judge_claims(classification["needs_judge"], all_docs)
        judge_status = judge_output["status"]
        for j in judge_output["unsupported"]:
            suspicious_claims.append({
                "claim": j["claim"],
                "reason": j["reason"],
                "caught_by": "llm_judge",
            })

    # Faithfulness score: complementary semantic check alongside regex/judge.
    import time as _t
    _faith_start = _t.perf_counter()
    faithfulness = calculate_faithfulness(full_text, all_docs) if all_docs else 0.0
    _faith_elapsed_ms = (_t.perf_counter() - _faith_start) * 1000
    try:
        from graph.nodes import _log_timing
        _log_timing("faithfulness", _faith_elapsed_ms)
    except Exception:
        pass
    
    return {
        "summary": summary,
        "issues": issues,
        "risks": risks,
        "open_questions": open_questions,
        "talking_points": talking_points,
        "information_gaps": state.get("information_gaps", []),
        "sources": sources,
        "faithfulness_score": faithfulness,
        "suspicious_facts": suspicious_facts,          # Layer 1 output (atomic facts)
        "suspicious_claims": suspicious_claims,         # Layers 2+3 output (whole claims)
        "judge_status": judge_status,                   # "ok" | "parse_error" | "error" |
                                                        # "skipped_breaker_open" | "no_claims" |
                                                        # "no_context_all_unsupported" | "disabled"
        "verification_stats": {                         # For transparency / writeup
            "claims_total": len(all_text_items),
            "verified_by_regex": len(classification["verified_by_regex"]),
            "flagged_by_regex": len(classification["flagged_by_regex"]),
            "sent_to_llm_judge": len(classification["needs_judge"]),
        },
        "loop_count": state.get("iteration_count", 0),
    }


def _strip_context_prefix(content: str, metadata: Dict[str, Any]) -> str:
    """Remove the '[Context: ...]\\n\\n' prefix added by contextual retrieval.

    When CONTEXTUAL_RETRIEVAL=1, _contextualize_chunks in chroma_utils prepends
    an LLM-generated context sentence to each chunk's page_content before
    embedding. The LLM reasoning node benefits from seeing that context, but
    end users shouldn't see it in citation passages — they want the raw source
    text. This helper strips the prefix only for display purposes.

    Two guards against mis-stripping:
    1. The has_context_prefix metadata flag must be True. Chunks that
       fell through _contextualize_chunks' error path have no flag, so we
       won't accidentally strip from an un-prefixed chunk that happens to
       contain ']\\n\\n' in its real content.
    2. The content must start with '[Context:'. If the flag-vs-content state
       is inconsistent (shouldn't happen, but defense in depth), we no-op.
    """
    if not metadata.get("has_context_prefix"):
        return content
    if not content.startswith("[Context:"):
        return content
    # Find the FIRST unescaped ']' that closes the opening '[Context:', then
    # look for the '\n\n' immediately after. This avoids mis-stripping if the
    # LLM-generated context itself contains ']\n\n' somewhere in the middle.
    close_bracket = content.find("]", len("[Context:"))
    if close_bracket == -1:
        return content
    # Require exactly ']\n\n' right after the closing bracket.
    if content[close_bracket:close_bracket + 3] != "]\n\n":
        return content
    return content[close_bracket + 3:]


def _resolve_cited_item(
    item: Any,
    chunk_map: Dict[str, Any],
    text_key: str,
) -> Dict[str, Any]:
    """Add source_doc and passage fields to a cited item dict."""
    if not isinstance(item, dict):
        return {text_key: str(item), "chunk_id": None, "source_doc": None, "passage": None}

    chunk_id = item.get("chunk_id")
    doc = chunk_map.get(chunk_id) if chunk_id else None

    source_doc = None
    passage = None
    if doc:
        source_doc = doc.metadata.get("filename") or os.path.basename(doc.metadata.get("source", "")) or None
        # Show first 200 chars of the supporting passage. Strip any contextual-
        # retrieval prefix first so citations show the real source text, not
        # the LLM-generated context sentence we prepended at ingest time.
        raw = _strip_context_prefix(doc.page_content, doc.metadata)
        passage = raw[:200].strip()

    return {
        text_key: item.get(text_key, ""),
        "chunk_id": chunk_id,
        "source_doc": source_doc,
        "passage": passage,
    }


def _build_sources(docs: List[Any]) -> List[Dict[str, str]]:
    """Deduplicated list of source doc names from retrieved documents."""
    seen: set = set()
    sources = []
    for doc in docs:
        name = doc.metadata.get("filename") or os.path.basename(doc.metadata.get("source", "")) or "unknown"
        if name not in seen:
            seen.add(name)
            sources.append({
                "filename": name,
                "doc_type": doc.metadata.get("doc_type", ""),
            })
    return sources


def _build_summary(
    issues: List[Dict],
    risks: List[Dict],
    open_questions: List[str],
    talking_points: List[Dict],
) -> str:
    parts = []
    if issues:
        parts.append(f"{len(issues)} issue(s) identified")
    if risks:
        parts.append(f"{len(risks)} risk(s) flagged")
    if open_questions:
        parts.append(f"{len(open_questions)} open question(s)")
    if talking_points:
        parts.append(f"{len(talking_points)} talking point(s)")
    if not parts:
        return "No significant findings in the retrieved documents."
    return "Pre-call brief: " + ", ".join(parts) + "."
