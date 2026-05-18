import json
import logging
import os
import re
import time
import threading
from typing import Optional
import numpy as np
from enum import Enum
from dotenv import load_dotenv
from pathlib import Path

_log = logging.getLogger(__name__)

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


# ── LLM model configuration ───────────────────────────────────────────────────
# Single source of truth for the LLM model. Every node (query_rewrite, reason,
# llm_judge) uses this same identifier. Overridable at deploy time via env so
# ops can swap models without code changes.
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")


# ── Circuit breaker ───────────────────────────────────────────────────────────

class _CBState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Three-state circuit breaker (CLOSED → OPEN → HALF_OPEN → CLOSED)."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 30):
        self.state = _CBState.CLOSED
        self.failures = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.last_failure_time = 0.0
        self._lock = threading.Lock()

    def is_open(self) -> bool:
        with self._lock:
            if self.state == _CBState.OPEN:
                if time.time() - self.last_failure_time >= self.recovery_timeout:
                    self.state = _CBState.HALF_OPEN
                    return False
                return True
            return False

    def on_success(self):
        with self._lock:
            self.failures = 0
            self.state = _CBState.CLOSED

    def on_failure(self):
        with self._lock:
            self.failures += 1
            self.last_failure_time = time.time()
            # A failure while probing (HALF_OPEN) means the service is still broken —
            # revert immediately instead of staying in HALF_OPEN indefinitely.
            if self.state == _CBState.HALF_OPEN or self.failures >= self.failure_threshold:
                self.state = _CBState.OPEN


llm_breaker = CircuitBreaker(failure_threshold=10, recovery_timeout=30)


# ── LLM judge gating ──────────────────────────────────────────────────────────

# Words that signal a relational claim ("X agreed", "Y promised") — these are
# exactly the claims regex CAN'T verify and that an FDE will quote back to
# customers. Always send these to the LLM judge regardless of confidence.
_RELATIONAL_QUERY_RE = re.compile(
    r"\b(agreed?|committed?|promised?|told|said|confirmed?|denied?|refused?|"
    r"stated?|approved?|rejected?|escalated?)\b",
    re.I,
)


def should_run_judge(query: str, faithfulness: float, n_claims: int,
                     always_run: bool = False) -> bool:
    """Decide whether to run Layer 3 (LLM judge) for a /lookup answer.

    The /brief endpoint always runs the judge (briefs are speculative and
    pre-call, so latency is acceptable). /lookup is in-call territory — we
    skip Layer 3 when we have high confidence the answer is grounded AND no
    relational verbs are involved.

    Always run when:
      - faithfulness < 0.7 (low confidence — must verify)
      - query has a relational verb (high stakes, regex can't catch it)
      - n_claims > 3 (complex answer — verify)
      - always_run flag is True (brief path)
    """
    if always_run:
        return True
    if faithfulness < 0.7:
        return True
    if _RELATIONAL_QUERY_RE.search(query or ""):
        return True
    if n_claims > 3:
        return True
    return False


# ── Faithfulness-based confidence (RAGAS metric) ──────────────────────────────

# Threshold calibrated for the active embedding provider.
# HuggingFace all-MiniLM-L6-v2 (normalize_embeddings=True) produces higher
# dot-product scores (~0.7-0.9 for related pairs) than OpenAI text-embedding-3-small
# (~0.4-0.7 for the same pairs). Using separate defaults avoids both false-positives
# (threshold too low) and the current 0.0-always bug (threshold too high for OpenAI).
# Override either via env var for custom calibration.
_THRESHOLD_BY_PROVIDER = {"openai": 0.45, "huggingface": 0.65}
try:
    _FAITHFULNESS_THRESHOLD = float(os.getenv("FAITHFULNESS_THRESHOLD", "0"))  # 0 = auto
except ValueError:
    _log.warning("FAITHFULNESS_THRESHOLD env var is not a valid float; using auto-select")
    _FAITHFULNESS_THRESHOLD = 0.0


def _get_faithfulness_threshold() -> float:
    if _FAITHFULNESS_THRESHOLD > 0:
        return _FAITHFULNESS_THRESHOLD
    from chroma_utils import get_embedding_provider
    return _THRESHOLD_BY_PROVIDER.get(get_embedding_provider(), 0.50)


def _cosine(a: tuple, b: tuple) -> float:
    """Dot product of two normalized embedding vectors = cosine similarity."""
    return float(np.dot(np.array(a), np.array(b)))


def _safe_embed(text: str):
    """Return embedding tuple, or None and log on failure."""
    from chroma_utils import embed_cached
    try:
        return embed_cached(text)
    except Exception as e:
        _log.warning("embed_cached_failed: %s | text=%r", e, text[:60])
        return None


def calculate_faithfulness(answer: str, retrieved_docs) -> float:
    """
    Faithfulness score: fraction of claim sentences grounded in retrieved context.
    A hallucinated answer no longer scores 1.0.
    """
    if "don't have enough information" in answer.lower():
        return 0.0
    if not retrieved_docs:
        return 0.0

    sentences = [s.strip() for s in re.split(r"[.!?]", answer) if len(s.strip()) > 20]
    if not sentences:
        return 0.0

    chunk_embs = [e for e in (_safe_embed(doc.page_content[:800]) for doc in retrieved_docs) if e is not None]
    if not chunk_embs:
        return 0.0

    threshold = _get_faithfulness_threshold()
    grounded = 0
    for sentence in sentences:
        s_emb = _safe_embed(sentence)
        if s_emb is not None and any(_cosine(s_emb, c_emb) >= threshold for c_emb in chunk_embs):
            grounded += 1

    return round(grounded / len(sentences), 2)



# ── Hallucination detection ───────────────────────────────────────────────────

_FACT_PATTERNS = [
    r"\b\d{1,4}[-/]\d{1,2}[-/]\d{2,4}\b",                    # dates: 2024-09-25, 9/25/2024
    r"\$[\d,]+\.?\d*[KkMm]?",                                 # currency: $28,000 / $4.5M
    r"\b\d+\.?\d*\s*%",                                       # percentages: 99.9%, 23%
    r"\b\d+\.?\d*\s*(?:hours?|days?|weeks?|months?|years?|minutes?|seconds?|ms|GB|MB|TB|KB)\b",  # quantities with time/size units
    r"\bv?\d+\.\d+(?:\.\d+)*\b",                              # version numbers: v2.4.1, 3.2.1
    r"\bP[0-4]\b",                                            # priority labels: P0 through P4
    r"\b(?:Sev|SEV)[\s-]?[0-4]\b",                            # severity labels: Sev 2, SEV-1
    r"\b[A-Z]{2,5}-\d{3,6}\b",                                # ticket/case IDs: TICK-4521, INC-0892
    r"\b(?:us|eu|ap|sa|ca|af|me)-(?:east|west|central|north|south|northeast|southeast|northwest|southwest)-[0-9]\b",  # cloud regions: us-east-2
    r"\b\d{1,3}(?:,\d{3})+\s*(?:events|records|requests|users|transactions|seats|rows)\b",  # large-quantity+unit: 50,000 records
]

# Relational verbs — if a claim contains any of these, it's making an assertion
# about who did/said/caused what. Regex can verify the nouns but not the relationship.
# Claims with these verbs get routed to the LLM judge layer.
_RELATIONAL_VERBS = {
    "said", "told", "asked", "agreed", "refused", "confirmed", "denied",
    "stated", "reported", "claimed", "acknowledged", "committed",
    "caused", "blocked", "approved", "rejected", "flagged", "raised",
    "escalated", "owns", "manages", "reports to",
}

# ── Claim classifier: routes work between regex layer and LLM judge ──────────

def classify_claims(claim_texts: list, retrieved_docs) -> dict:
    """
    Sort a list of claim sentences into three buckets based on what can verify them:
      - "verified_by_regex"   : claim contains regex-matchable facts and ALL of them
                                appear in the context. No further check needed.
      - "flagged_by_regex"    : claim contains regex-matchable facts and at least
                                ONE does not appear in context. Already caught.
      - "needs_judge"         : claim has no regex-matchable facts OR contains
                                relational verbs (who said what, who caused what).
                                These require semantic verification by the LLM judge.

    Returns:
      {
        "verified_by_regex": [claim_text, ...],
        "flagged_by_regex":  [{"claim": text, "unsupported_facts": [...]}],
        "needs_judge":       [claim_text, ...],
      }

    Called before running the LLM judge, so the judge only sees the claims it
    actually needs to reason about.
    """
    if not claim_texts:
        return {"verified_by_regex": [], "flagged_by_regex": [], "needs_judge": []}

    context_text = " ".join(doc.page_content for doc in retrieved_docs).lower() if retrieved_docs else ""

    verified, flagged, needs_judge = [], [], []

    for claim in claim_texts:
        if not claim or not claim.strip():
            continue

        claim_lower = claim.lower()

        # Extract all regex-matchable atomic facts from the claim
        atomic_facts = []
        for pattern in _FACT_PATTERNS:
            atomic_facts.extend(re.findall(pattern, claim, re.IGNORECASE))

        # Check which atomic facts appear in the retrieved context
        unsupported = [f for f in atomic_facts if f.lower() not in context_text]
        has_regex_facts = len(atomic_facts) > 0

        # Check if the claim contains relational assertions — even if atomic facts
        # match, the relationship between them may still be hallucinated.
        # Use whole-string `in` (not `.split()`) so multi-word verbs like
        # "reports to" also match. The previous `.split()` membership check
        # would silently miss any space-containing entry in _RELATIONAL_VERBS.
        # We pad with spaces to avoid spurious sub-word matches (e.g. "raised"
        # would otherwise match against "praised" / "appraised").
        padded = f" {claim_lower} "
        has_relational = any(f" {verb} " in padded for verb in _RELATIONAL_VERBS)

        # Routing logic
        if has_regex_facts and not unsupported and not has_relational:
            # All checkable facts verified, no relational claims — trust regex
            verified.append(claim)
        elif has_regex_facts and unsupported:
            # Regex found a lie — flag without bothering the LLM
            flagged.append({"claim": claim, "unsupported_facts": list(set(unsupported))})
        else:
            # Either no atomic facts, or has relational verbs — needs LLM
            needs_judge.append(claim)

    return {
        "verified_by_regex": verified,
        "flagged_by_regex": flagged,
        "needs_judge": needs_judge,
    }

def detect_hallucination(answer: str, retrieved_docs) -> list:
    """
    Return specific facts in the answer not found in retrieved context.
    Wire into output layer so every brief gets checked before returning.
    """
    if not retrieved_docs:
        return []
    context_text = " ".join(doc.page_content for doc in retrieved_docs).lower()
    suspicious = []
    for pattern in _FACT_PATTERNS:
        for match in re.findall(pattern, answer, re.IGNORECASE):
            if match.lower() not in context_text:
                suspicious.append(match)
    return list(set(suspicious))


# ── LLM-as-judge layer ───────────────────────────────────────────────────────

_JUDGE_PROMPT = """ROLE:
You are a strict fact-verification judge with zero tolerance for unverified claims.

TASK A — CLAIM VERIFICATION:
For each CLAIM below, determine if it is fully supported by the CONTEXT.
A claim is SUPPORTED only when EVERY assertion in it — every name, relationship,
attribution, and fact — can be traced word-for-word or as an unambiguous paraphrase
to the provided CONTEXT. Any doubt → UNSUPPORTED.

RULES (apply ALL of them):
1. NAMES: Every person, company, and product name must appear in the context EXACTLY.
   "Sarah Park" and "Sarah Chen" are DIFFERENT people. If the claim names someone not
   present in the context, or names a slightly different person, mark UNSUPPORTED.
2. ENTITIES: If the claim introduces any organization, tool, or system (e.g. "Databricks",
   "Snowflake", a specific vendor) that does not appear in the context, mark UNSUPPORTED.
3. RELATIONSHIPS: If the claim asserts who did/said/caused/blocked something, verify that
   BOTH the actor and the action are explicitly in the context. "X committed to Y" and
   "X blocked Y" are opposite assertions — confirm the exact verb.
4. CONTRADICTIONS: If the context states the OPPOSITE of what the claim asserts, mark UNSUPPORTED.
5. PARAPHRASES: A faithful restatement of context content in different words is SUPPORTED,
   but only if it does not change the meaning or introduce any new entity or relationship.
6. DEFAULT: When in doubt, mark UNSUPPORTED. False positives (over-flagging) are safer than
   false negatives (missing a hallucination).

TASK B — CONFLICT DETECTION:
After verifying the claims, scan the context chunks against each other.
Identify cases where two chunks assert contradictory facts about the same topic, e.g.:
  - One chunk says a project is "on track", another says it is "slipping"
  - Different dates for the same deadline or promised delivery
  - Different owners, statuses, or agreed terms for the same item

Recency rule: if one chunk has a newer date in its header, its version is more likely
correct. Set recommendation to "trust_newer" in that case, otherwise "needs_verification".

<context>
{context}
</context>

<claims_to_judge>
{claims_numbered}
</claims_to_judge>

OUTPUT:
Return ONLY a single JSON object (not an array) with two keys:

{{
  "verdicts": [
    {{"index": 1, "verdict": "supported", "reason": "Context explicitly states this."}},
    {{"index": 2, "verdict": "unsupported", "reason": "Name 'Sarah Park' does not appear in context; context names 'Sarah Chen'."}}
  ],
  "conflicts": [
    {{
      "topic": "brief description of what the conflict is about",
      "source_a": {{"chunk_id": "chunk 1", "claim": "what this chunk says"}},
      "source_b": {{"chunk_id": "chunk 3", "claim": "what this chunk says"}},
      "recommendation": "trust_newer | needs_verification"
    }}
  ]
}}

If no conflicts are found, return "conflicts": [].
Return the JSON object only. No prose before or after."""

def _llm_invoke_with_retry(llm, prompt, max_retries: int = 3):
    """Invoke LLM, sleeping and retrying on 429 / quota / rate-limit responses.

    Covers error-message shapes from both Groq ("try again in 6.5s") and
    Gemini ("ResourceExhausted", "quota", "retry_delay { seconds: 20 }").

    Always either returns the LLM response or raises — never silently returns
    None. The defensive guard + final ``raise`` at the end of the function
    ensure that even pathological inputs (max_retries < 1, or a future loop
    refactor) can't produce a None that crashes callers with an opaque
    AttributeError on ``response.content``.
    """
    if max_retries < 1:
        raise ValueError(
            f"_llm_invoke_with_retry requires max_retries >= 1, got {max_retries}"
        )
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return llm.invoke(prompt)
        except Exception as e:
            last_exc = e
            msg = str(e)
            low = msg.lower()
            is_rate_limit = (
                "429" in msg
                or "rate_limit" in low or "rate limit" in low
                or "resource_exhausted" in low or "resourceexhausted" in low
                or "quota" in low
            )
            if is_rate_limit and attempt < max_retries - 1:
                m = (re.search(r"try again in\s+([\d.]+)s", msg, re.IGNORECASE)
                     or re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", msg, re.IGNORECASE)
                     or re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+)", msg, re.IGNORECASE))
                wait = float(m.group(1)) + 1.0 if m else 10.0
                _log.warning("llm_rate_limit: sleeping %.1fs (attempt %d/%d)", wait, attempt + 1, max_retries)
                time.sleep(wait)
            else:
                raise
    # Belt-and-braces: should be unreachable since the loop above either
    # returns or raises on every iteration. Re-raise the last captured
    # exception (or a synthetic one) so an opaque None can never propagate
    # to callers like reason_node / answer_node which then crash on .content.
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("_llm_invoke_with_retry exhausted retries without success")


def llm_judge_claims(claim_texts: list, retrieved_docs) -> dict:
    """
    Verify claims against retrieved context using an LLM as judge.

    Runs ONE batched LLM call regardless of how many claims are passed in.
    Returns a dict:
      {
        "unsupported": [{"claim": "...", "reason": "...", "judge_verdict": "unsupported"}, ...],
        "status": "ok" | "no_claims" | "skipped_breaker_open" |
                  "no_context_all_unsupported" | "parse_error" | "error",
      }

    The status lets callers distinguish "judge ran and found nothing suspicious"
    from "judge didn't run at all" — silent skips previously looked identical to
    a clean bill of health.
    """
    import time as _t
    _judge_start = _t.perf_counter()

    def _log_and_return(unsupported, status, conflicts=None):
        """Helper so every return path logs timing consistently."""
        _judge_elapsed_ms = (_t.perf_counter() - _judge_start) * 1000
        try:
            from graph.nodes import _log_timing
            _log_timing("llm_judge", _judge_elapsed_ms)
        except Exception:
            pass
        return {"unsupported": unsupported, "status": status, "conflicts": conflicts or []}

    if not claim_texts:
        return _log_and_return([], "no_claims")

    # If circuit breaker is open, skip the LLM call entirely
    if llm_breaker.is_open():
        return _log_and_return([], "skipped_breaker_open")

    # With no context there's nothing to verify against — everything is unsupported
    if not retrieved_docs:
        return _log_and_return([
            {"claim": c, "reason": "No retrieved context to verify against.",
             "judge_verdict": "unsupported"} for c in claim_texts
        ], "no_context_all_unsupported")

    # Build the prompt
    context_str = "\n\n---\n\n".join(
        f"[chunk {i+1}]\n{doc.page_content}" for i, doc in enumerate(retrieved_docs)
    )
    claims_numbered = "\n".join(
        f"{i+1}. {c}" for i, c in enumerate(claim_texts)
    )
    prompt = _JUDGE_PROMPT.format(context=context_str, claims_numbered=claims_numbered)

    # Make the LLM call
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0,
        )
        response = _llm_invoke_with_retry(llm, prompt)
        # Token usage logging
        try:
            from graph.nodes import _log_token_usage
            _log_token_usage("llm_judge", response, prompt)
        except Exception:
            pass

        content = (response.content or "").strip()

        # Strip markdown fences using the same robust regex helper used by nodes.
        try:
            from graph.nodes import _strip_json
            content = _strip_json(content)
        except ImportError:
            pass

        parsed = json.loads(content)

        # The new prompt returns {"verdicts": [...], "conflicts": [...]}.
        # Guard against the old array shape in case a cached/stale prompt fires.
        if isinstance(parsed, list):
            verdicts = parsed
            conflicts_raw = []
        elif isinstance(parsed, dict):
            verdicts = parsed.get("verdicts", [])
            conflicts_raw = parsed.get("conflicts", [])
        else:
            raise ValueError(f"judge returned unexpected type: {type(parsed).__name__}")

        if not isinstance(verdicts, list):
            raise ValueError("judge 'verdicts' is not a list")

        llm_breaker.on_success()

        # Filter to unsupported claims only
        unsupported = []
        for v in verdicts:
            if not isinstance(v, dict):
                continue
            if v.get("verdict", "").lower() == "unsupported":
                raw_idx = v.get("index")
                if raw_idx is None:
                    continue  # missing index → can't map to a claim safely
                idx = raw_idx - 1  # LLM uses 1-based numbering
                if 0 <= idx < len(claim_texts):
                    unsupported.append({
                        "claim": claim_texts[idx],
                        "reason": v.get("reason", "Judge marked unsupported."),
                        "judge_verdict": "unsupported",
                    })

        # Normalise conflicts list — drop malformed entries
        conflicts = [
            c for c in (conflicts_raw or [])
            if isinstance(c, dict) and c.get("topic") and c.get("source_a") and c.get("source_b")
        ]
        return _log_and_return(unsupported, "ok", conflicts)

    except json.JSONDecodeError as e:
        llm_breaker.on_failure()
        _log.error("llm_judge_json_parse_failed: %s | raw=%s", e, content[:200])
        return _log_and_return([], "parse_error", [])
    except Exception as e:
        llm_breaker.on_failure()
        _log.error("llm_judge_claims_failed: %s", e)
        return _log_and_return([], "error", [])