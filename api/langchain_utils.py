import json
import logging
import os
import re
import time
import threading
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


llm_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30)


# ── Faithfulness-based confidence (RAGAS metric) ──────────────────────────────

# Calibrated against eval runs: 0.65 gives < 5% false-positive grounding rate.
# Override via FAITHFULNESS_THRESHOLD env var if you re-calibrate on your own dataset.
_FAITHFULNESS_THRESHOLD = float(os.getenv("FAITHFULNESS_THRESHOLD", "0.65"))


def _cosine(a: tuple, b: tuple) -> float:
    """Dot product of two normalized embedding vectors = cosine similarity."""
    return float(np.dot(np.array(a), np.array(b)))


def calculate_faithfulness(answer: str, retrieved_docs) -> float:
    """
    Faithfulness score: fraction of claim sentences grounded in retrieved context.
    A hallucinated answer no longer scores 1.0.
    """
    if "don't have enough information" in answer.lower():
        return 0.0
    if not retrieved_docs:
        return 0.0

    from chroma_utils import embed_cached

    sentences = [s.strip() for s in re.split(r"[.!?]", answer) if len(s.strip()) > 20]
    if not sentences:
        return 0.0

    chunk_embs = [embed_cached(doc.page_content[:300]) for doc in retrieved_docs]

    grounded = 0
    for sentence in sentences:
        s_emb = embed_cached(sentence)
        if any(_cosine(s_emb, c_emb) >= _FAITHFULNESS_THRESHOLD for c_emb in chunk_embs):
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
        has_relational = any(verb in claim_lower.split() for verb in _RELATIONAL_VERBS)

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

TASK:
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

CONTEXT:
{context}

CLAIMS TO JUDGE:
{claims_numbered}

OUTPUT:
Return ONLY a JSON array, one object per claim in the same order, with:
  - "index": the claim number (1-indexed, matching input)
  - "verdict": "supported" or "unsupported"
  - "reason": one short sentence explaining the verdict (quote the exact mismatch when flagging)

Example output:
[
  {{"index": 1, "verdict": "supported", "reason": "Context explicitly states this."}},
  {{"index": 2, "verdict": "unsupported", "reason": "Name 'Sarah Park' does not appear in context; context names 'Sarah Chen'."}}
]

Return the JSON array only. No prose before or after."""

def _llm_invoke_with_retry(llm, prompt, max_retries: int = 3):
    """Invoke LLM, sleeping and retrying on 429 / quota / rate-limit responses.

    Covers error-message shapes from both Groq ("try again in 6.5s") and
    Gemini ("ResourceExhausted", "quota", "retry_delay { seconds: 20 }").
    """
    for attempt in range(max_retries):
        try:
            return llm.invoke(prompt)
        except Exception as e:
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

    def _log_and_return(unsupported, status):
        """Helper so every return path logs timing consistently."""
        _judge_elapsed_ms = (_t.perf_counter() - _judge_start) * 1000
        try:
            from graph.nodes import _log_timing
            _log_timing("llm_judge", _judge_elapsed_ms)
        except Exception:
            pass
        return {"unsupported": unsupported, "status": status}

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

        content = response.content.strip()

        # Strip markdown fences if present
        if "```" in content:
            content = content.split("```")[1].strip()
            if content.startswith("json"):
                content = content[4:].strip()

        verdicts = json.loads(content)
        if not isinstance(verdicts, list):
            raise ValueError("judge did not return a list")

        llm_breaker.on_success()

        # Filter to unsupported claims only
        unsupported = []
        for v in verdicts:
            if not isinstance(v, dict):
                continue
            if v.get("verdict", "").lower() == "unsupported":
                idx = v.get("index", 0) - 1
                if 0 <= idx < len(claim_texts):
                    unsupported.append({
                        "claim": claim_texts[idx],
                        "reason": v.get("reason", "Judge marked unsupported."),
                        "judge_verdict": "unsupported",
                    })
        return _log_and_return(unsupported, "ok")

    except json.JSONDecodeError as e:
        llm_breaker.on_failure()
        _log.error("llm_judge_json_parse_failed: %s | raw=%s", e, content[:200] if 'content' in dir() else "")
        return _log_and_return([], "parse_error")
    except Exception as e:
        llm_breaker.on_failure()
        _log.error("llm_judge_claims_failed: %s", e)
        return _log_and_return([], "error")