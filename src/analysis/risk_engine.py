"""
Core risk analysis engine.

Uses a map-reduce strategy:
  Map:    Analyze each clause independently against retrieved references.
  Reduce: Synthesize clause results into a document-level summary.

Pydantic validation with retry ensures structured output.
"""

import hashlib
import json
import logging
import re
import threading
import time
from typing import Optional

from config.settings import (
    ANTHROPIC_API_KEY,
    OPENAI_API_KEY,
    GROQ_API_KEY,
    LLM_PROVIDER,
    ANTHROPIC_MODEL,
    OPENAI_MODEL,
    GROQ_BASE_URL,
    GROQ_MODEL,
    LMSTUDIO_BASE_URL,
    LMSTUDIO_MODEL,
    MAX_RETRIES,
    CONFIDENCE_THRESHOLD,
    MAX_CLAUSES_PER_ANALYSIS,
)
from src.analysis.schemas import (
    ClauseRisk,
    DocumentSummary,
    FullAnalysisResult,
    MissingClause,
    RiskLevel,
)
from src.analysis.prompts import (
    CLAUSE_ANALYSIS_SYSTEM,
    CLAUSE_ANALYSIS_USER,
    DOCUMENT_SUMMARY_SYSTEM,
    DOCUMENT_SUMMARY_USER,
    MISSING_CLAUSE_CHECK_SYSTEM,
    MISSING_CLAUSE_CHECK_USER,
)
from src.ingestion.metadata import EnrichedChunk
from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.reranker import CrossEncoderReranker
from src.agents.verification import VerificationAgent
from src.agents.obligation import ObligationAgent
from src.agents.contradiction import ContradictionAgent
from src.audit.logger import AuditLogger

logger = logging.getLogger(__name__)


# ── Question → Clause-Type Router ────────────────────────────────────────────
# Keyword signals that map a natural-language question to a clause category.
# Weights are additive; the top-scoring type is used as a search filter unless
# the race is too close (ambiguity margin) or signals are too weak.

_QUESTION_CLAUSE_SIGNALS: dict[str, list[tuple[str, float]]] = {
    "limitation_of_liability": [
        (r"\bliabilit", 2.5),
        (r"\bdamage.*limit\b", 2.0),
        (r"\bnot.*exceed\b", 2.0),
        (r"\bindirect.*damage", 2.0),
        (r"\bconsequential", 2.0),
        (r"\bpunitive", 1.5),
        (r"\bmaximum.*liable\b", 2.5),
        (r"\bcap\b.*damage", 2.0),
    ],
    "indemnification": [
        (r"\bindemnif", 3.0),
        (r"\bhold harmless\b", 2.5),
        (r"\bdefend.*claim\b", 2.0),
        (r"\bwho pays.*claim", 2.0),
        (r"\bresponsible.*lawsuit", 2.0),
    ],
    "confidentiality": [
        (r"\bconfidential", 2.5),
        (r"\bnon.?disclos\b", 3.0),
        (r"\bnda\b", 4.0),           # unambiguous acronym: always wins over incidental keywords
        (r"\bdisclos", 1.5),
        (r"\btrade secret\b", 2.5),
        (r"\bproprietary.*information\b", 2.0),
        (r"\bcan.*share.*information", 1.5),
    ],
    "termination": [
        (r"\bterminat", 2.5),
        (r"\bcancel\b", 2.0),
        (r"\bend.*contract\b", 2.0),
        (r"\bnotice.*period\b", 2.0),
        (r"\bexpir", 1.5),
        (r"\bauto.?renew", 2.5),
        (r"\bwithout.*cause\b", 2.0),
        (r"\bfor.*cause\b", 1.5),
        (r"\bhow.*end\b", 1.5),
    ],
    "payment_terms": [
        (r"\bpayment\b", 2.0),
        (r"\binvoice\b", 2.0),
        (r"\bfee\b", 1.5),
        (r"\bprice\b", 1.5),
        (r"\bsubscription\b", 2.0),
        (r"\brefund\b", 2.0),
        (r"\blate.*payment\b", 2.5),
        (r"\bwhen.*due\b", 2.0),
        (r"\bdue.*date\b", 2.0),
    ],
    "ip_assignment": [
        (r"\bintellectual property\b", 2.5),
        (r"\bwork.*for.*hire\b", 3.0),
        (r"\bown.*invention", 2.5),
        (r"\bpatent\b", 2.0),
        (r"\bcopyright\b", 2.0),
        (r"\bwork product\b", 2.5),
        (r"\bwho owns\b", 1.5),
        (r"\bip rights\b", 2.5),
        (r"\bassign.*right", 2.0),
    ],
    "non_compete": [
        (r"\bnon.?compet", 3.0),
        (r"\bcompetitor\b", 2.0),
        (r"\bnon.?solicit", 3.0),
        (r"\brestrict.*work\b", 2.0),
        (r"\bcompeting.*business\b", 2.0),
    ],
    "dispute_resolution": [
        (r"\barbitrat", 3.0),
        (r"\bdisput", 2.5),           # prefix: matches both "dispute" and "disputes"
        (r"\bmediat", 2.0),
        (r"\blitigat", 2.0),
        (r"\bforum\b", 2.0),
        (r"\bclass.*action\b", 2.5),
    ],
    "governing_law": [
        (r"\bgoverning law\b", 3.0),
        (r"\bwhich.*law\b", 2.0),
        (r"\bwhat.*law\b", 2.0),
        (r"\blaw.*govern", 2.5),     # prefix: matches "governs", "governed", "governing"
        (r"\bstate.*law.*appli", 2.5),
        (r"\bjurisdiction\b", 2.0),
    ],
    "warranty": [
        (r"\bwarrant", 2.5),
        (r"\bguarantee\b", 2.0),
        (r"\bas.?is\b", 3.0),
        (r"\bno.*warrant", 2.5),
        (r"\bfit.*purpose\b", 2.0),
    ],
    "data_protection": [
        (r"\bpersonal.*data\b", 2.5),
        (r"\bgdpr\b", 3.0),
        (r"\bccpa\b", 3.0),
        (r"\bdata.*breach\b", 2.5),
        (r"\bmy data\b", 2.0),
        (r"\bcustomer.*data\b", 2.0),
        (r"\bpersonal.*information\b", 2.0),
    ],
    "force_majeure": [
        (r"\bforce majeure\b", 3.0),
        (r"\bact of god\b", 3.0),
        (r"\bbeyond.*control\b", 2.0),
        (r"\bnatural disaster\b", 2.0),
        (r"\bpandemic\b", 1.5),
    ],
}

_COMPILED_QUESTION_SIGNALS: dict[str, list[tuple[re.Pattern, float]]] = {
    clause_type: [
        (re.compile(pattern, re.IGNORECASE), weight)
        for pattern, weight in signals
    ]
    for clause_type, signals in _QUESTION_CLAUSE_SIGNALS.items()
}

_MIN_SIGNAL_SCORE = 1.5   # ignore types with weaker total signal
_AMBIGUITY_MARGIN = 1.5   # if top-2 within this margin AND both strong → no filter


def _infer_all_clause_types(question: str) -> list[str]:
    """
    Return every clause type whose signal score meets the minimum threshold.

    Used for multi-hop questions that span categories (e.g. "What happens if
    I breach confidentiality AND miss a payment?").  The caller retrieves for
    each type independently then merges the contexts before the LLM call.
    Returns an empty list when no type meets the threshold (full-corpus search).
    """
    types: list[tuple[str, float]] = []
    for clause_type, patterns in _COMPILED_QUESTION_SIGNALS.items():
        score = sum(w for pat, w in patterns if pat.search(question))
        if score >= _MIN_SIGNAL_SCORE:
            types.append((clause_type, score))
    # Only return multiple when the question is genuinely multi-signal
    # (two or more types each scoring independently above the threshold).
    return [t for t, _ in sorted(types, key=lambda x: x[1], reverse=True)]


def _infer_question_clause_type(question: str) -> Optional[str]:
    """
    Map a natural-language question to a clause category for targeted retrieval.

    Returns a clause_type string when a clear category wins, or None when the
    question is off-topic or genuinely ambiguous between two categories.
    Caller should fall back to full-corpus search on None.
    """
    scores: dict[str, float] = {}
    for clause_type, patterns in _COMPILED_QUESTION_SIGNALS.items():
        score = sum(w for pat, w in patterns if pat.search(question))
        if score >= _MIN_SIGNAL_SCORE:
            scores[clause_type] = score

    if not scores:
        return None

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_type, top_score = ranked[0]

    # If second-place is close and strong enough, call it ambiguous
    if len(ranked) >= 2:
        _, second_score = ranked[1]
        if top_score - second_score < _AMBIGUITY_MARGIN and second_score >= _MIN_SIGNAL_SCORE:
            return None

    return top_type


class _GroqRateLimiter:
    """Sliding-window TPM limiter for Groq free tier (6 000 TPM)."""

    def __init__(self, tpm_limit: int = 5500):
        self._tpm_limit = tpm_limit
        self._window: list[tuple[float, int]] = []
        self._lock = threading.Lock()

    def wait(self, estimated_tokens: int) -> None:
        with self._lock:
            while True:
                now = time.time()
                self._window = [(t, tok) for t, tok in self._window if now - t < 60]
                used = sum(tok for _, tok in self._window)
                if used + estimated_tokens <= self._tpm_limit:
                    self._window.append((now, estimated_tokens))
                    return
                sleep_for = 60 - (now - self._window[0][0]) + 0.5
                logger.info(f"Rate limiter: {used}/{self._tpm_limit} TPM — sleeping {sleep_for:.1f}s")
                time.sleep(sleep_for)


def _detect_contract_type(chunks: list[EnrichedChunk]) -> str:
    """Heuristic contract type detection from clause types present."""
    types = {c.clause_type for c in chunks}
    text_sample = " ".join(c.text[:200] for c in chunks[:5]).lower()

    if "non_compete" in types or "employment" in text_sample or "at-will" in text_sample:
        return "Employment Contract"
    if "data_protection" in types or "saas" in text_sample or "subscription" in text_sample:
        return "SaaS Agreement"
    if "confidentiality" in types and len(types) <= 4:
        return "NDA"
    if "lease" in text_sample or "tenant" in text_sample or "landlord" in text_sample:
        return "Lease Agreement"
    return "Service Agreement"


def _extract_json_from_response(text: str):
    """Extract JSON from LLM response, handling markdown code blocks."""
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("```").strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find first JSON object/array
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue

    raise ValueError(f"Could not extract valid JSON from response: {text[:300]}")


class LLMClient:
    """Unified LLM client supporting Anthropic, OpenAI, and LM Studio."""

    def __init__(self):
        self.provider = LLM_PROVIDER

        if self.provider == "anthropic":
            if not ANTHROPIC_API_KEY:
                raise ValueError("ANTHROPIC_API_KEY not set. Please add it to your .env file.")
            import anthropic
            self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            self.model = ANTHROPIC_MODEL
        elif self.provider == "groq":
            if not GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY not set. Please add it to your .env file.")
            from openai import OpenAI
            self.client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
            self.model = GROQ_MODEL
            self._rate_limiter: Optional[_GroqRateLimiter] = _GroqRateLimiter()
            logger.info(f"Groq client initialized: {GROQ_BASE_URL} | model={GROQ_MODEL}")
        elif self.provider == "lmstudio":
            from openai import OpenAI
            self.client = OpenAI(api_key="lm-studio", base_url=LMSTUDIO_BASE_URL)
            self.model = LMSTUDIO_MODEL
            logger.info(f"LM Studio client initialized: {LMSTUDIO_BASE_URL} | model={LMSTUDIO_MODEL}")
        else:
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not set. Please add it to your .env file.")
            from openai import OpenAI
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.model = OPENAI_MODEL

    def chat(self, system: str, user: str, max_tokens: int = 2048) -> str:
        """Send a chat completion request and return the text response."""
        if getattr(self, "_rate_limiter", None):
            estimated = (len(system) + len(user)) // 4 + max_tokens
            self._rate_limiter.wait(estimated)
        if self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return response.content[0].text
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return response.choices[0].message.content


class RiskAnalysisEngine:
    """
    Orchestrates the full contract analysis pipeline:
      1. Retrieve reference clauses for each contract chunk
      2. LLM analysis per clause (map step)
      3. Document-level summary synthesis (reduce step)
    """

    def __init__(
        self,
        hybrid_search: HybridSearchEngine,
        reranker: CrossEncoderReranker,
    ):
        self.hybrid_search = hybrid_search
        self.reranker = reranker
        self.llm = LLMClient()
        self.verifier = VerificationAgent(self.llm)
        self.obligation_agent = ObligationAgent(self.llm)
        self.contradiction_agent = ContradictionAgent(self.llm)
        self._summary_cache: dict[tuple[str, str], object] = {}
        self.audit = AuditLogger()

    def analyze_contract(
        self,
        chunks: list[EnrichedChunk],
        filename: str,
        progress_callback=None,
    ) -> FullAnalysisResult:
        """
        Full contract analysis pipeline.

        Args:
            chunks: Enriched contract chunks.
            filename: Source filename for display.
            progress_callback: Optional callable(message: str, pct: float)
                               for Gradio progress updates.

        Returns:
            FullAnalysisResult with all clause analyses and document summary.
        """
        if not chunks:
            raise ValueError("No chunks to analyze — document may be empty.")

        # Limit to avoid runaway processing on huge documents
        chunks_to_analyze = chunks[:MAX_CLAUSES_PER_ANALYSIS]
        if len(chunks) > MAX_CLAUSES_PER_ANALYSIS:
            logger.warning(
                f"Document has {len(chunks)} chunks; analyzing first {MAX_CLAUSES_PER_ANALYSIS}"
            )

        contract_type = _detect_contract_type(chunks_to_analyze)
        logger.info(f"Detected contract type: {contract_type}")

        # ── Map step: analyze each clause ─────────────────────────────────────
        clause_analyses: list[ClauseRisk] = []
        total = len(chunks_to_analyze)

        for i, chunk in enumerate(chunks_to_analyze):
            if progress_callback:
                progress_callback(
                    f"Analyzing clause {i+1}/{total}: {chunk.section_title[:50]}...",
                    (i + 1) / total * 0.8,
                )

            analysis = self._analyze_single_clause(chunk)
            if analysis:
                clause_analyses.append(analysis)

        if not clause_analyses:
            raise RuntimeError("Analysis produced no results. Check API key and connectivity.")

        # ── Obligation extraction ──────────────────────────────────────────────
        if progress_callback:
            progress_callback("Extracting obligations, deadlines, and notice periods...", 0.82)

        obligation_table = None
        try:
            obligation_table = self.obligation_agent.extract(chunks_to_analyze)
            self.audit.log_obligation(
                obligation_count=len(obligation_table.obligations),
                high_priority=obligation_table.high_priority_count,
                note=obligation_table.extraction_note or "",
            )
        except Exception as e:
            logger.warning(f"Obligation extraction failed (non-fatal): {e}")

        # ── Contradiction detection ────────────────────────────────────────────
        if progress_callback:
            progress_callback("Checking for cross-clause contradictions...", 0.87)

        contradiction_report = None
        try:
            contradiction_report = self.contradiction_agent.detect(clause_analyses)
            self.audit.log_contradiction(
                contradictions_found=len(contradiction_report.contradictions),
                has_critical=contradiction_report.has_critical,
            )
        except Exception as e:
            logger.warning(f"Contradiction detection failed (non-fatal): {e}")

        # ── Missing clause check ───────────────────────────────────────────────
        if progress_callback:
            progress_callback("Checking for missing standard clauses...", 0.90)

        missing = self._check_missing_clauses(
            contract_type, [c.clause_type for c in clause_analyses]
        )

        # ── Reduce step: document summary (cached by content hash) ────────────
        if progress_callback:
            progress_callback("Generating document summary...", 0.90)

        content_hash = hashlib.sha256(
            "".join(c.clause_text[:120] for c in clause_analyses).encode()
        ).hexdigest()[:16]
        cache_key = (filename, content_hash)

        cached = self._summary_cache.get(cache_key)
        if cached is not None:
            logger.info(f"Summary cache hit for '{filename}' (hash={content_hash})")
            doc_summary = cached
        else:
            doc_summary = self._generate_document_summary(
                filename, clause_analyses, missing, contract_type
            )
            self._summary_cache[cache_key] = doc_summary

        self.audit.log_summary(
            filename=filename,
            overall_risk=doc_summary.overall_risk_level.value,
            risk_score=doc_summary.overall_risk_score,
        )

        if progress_callback:
            progress_callback("Analysis complete.", 1.0)

        result = FullAnalysisResult.from_clause_list(
            filename=filename,
            clause_analyses=clause_analyses,
            document_summary=doc_summary,
        )
        result.obligation_table = obligation_table
        result.contradiction_report = contradiction_report
        return result

    # ── Map step ──────────────────────────────────────────────────────────────

    def _analyze_single_clause(self, chunk: EnrichedChunk) -> Optional[ClauseRisk]:
        """
        Analyze one clause with Pydantic-validation retries AND a
        post-analysis faithfulness check (LLM-as-judge).

        Flow:
          1. Retrieve reference clauses (filtered by clause type).
          2. LLM analysis → Pydantic validation (up to MAX_RETRIES).
          3. VerificationAgent judges faithfulness of the result.
          4. If not verified: re-retrieve without type filter (broader),
             re-analyze once more, attach verification to final result.
        """
        reranked, reference_text = self._retrieve_for_chunk(
            chunk, use_type_filter=True
        )

        if not reranked:
            return ClauseRisk(
                clause_text=chunk.text[:500],
                clause_type=chunk.clause_type,
                risk_level=RiskLevel.MEDIUM,
                risk_description="No reference clause found for comparison. Manual review recommended.",
                key_concerns=["Insufficient reference data for automated assessment"],
                reference_clause="",
                source_citation="No reference available",
                suggested_revision="Please have this clause reviewed by qualified legal counsel.",
                confidence_score=0.1,
            )

        result = self._llm_analyze(chunk, reference_text, reranked)
        if result is None:
            return None

        # ── Verify faithfulness ───────────────────────────────────────────────
        verification = self.verifier.verify(result, reference_text)
        result.verification = verification

        if not verification.is_verified:
            logger.info(
                f"[{chunk.clause_type}] verification failed "
                f"(score={verification.faithfulness_score:.2f}), re-retrieving broadly"
            )
            # One retry with broader retrieval (no clause-type filter)
            reranked2, reference_text2 = self._retrieve_for_chunk(
                chunk, use_type_filter=False
            )
            if reranked2:
                result2 = self._llm_analyze(chunk, reference_text2, reranked2)
                if result2 is not None:
                    verification2 = self.verifier.verify(result2, reference_text2)
                    result2.verification = verification2
                    return result2

        return result

    def _retrieve_for_chunk(
        self,
        chunk: EnrichedChunk,
        use_type_filter: bool,
    ) -> tuple[list[dict], str]:
        """Retrieve and rerank reference clauses for a chunk."""
        clause_filter = (
            chunk.clause_type
            if use_type_filter and chunk.clause_type != "general"
            else None
        )
        candidates = self.hybrid_search.search_reference(
            query=chunk.text,
            top_k=10,
            clause_type_filter=clause_filter,
        )
        reranked = self.reranker.rerank(query=chunk.text, candidates=candidates, top_n=4)
        reference_text = self._format_references(reranked) if reranked else ""
        return reranked, reference_text

    def _llm_analyze(
        self,
        chunk: EnrichedChunk,
        reference_text: str,
        reranked: list[dict],
    ) -> Optional[ClauseRisk]:
        """Run LLM clause analysis with Pydantic-validation retries."""
        _NON_RETRYABLE = (
            "credit balance", "insufficient credits", "payment required",
            "authentication", "invalid api key", "permission denied",
        )
        for attempt in range(MAX_RETRIES):
            try:
                prompt = CLAUSE_ANALYSIS_USER.format(
                    clause_text=chunk.text[:800],
                    clause_type=chunk.clause_type,
                    reference_chunks=reference_text,
                )
                raw = self.llm.chat(
                    system=CLAUSE_ANALYSIS_SYSTEM,
                    user=prompt,
                    max_tokens=1024,
                )
                data = _extract_json_from_response(raw)
                if isinstance(data, list):
                    data = data[0] if data else {}
                return ClauseRisk(**data)

            except Exception as e:
                err_lower = str(e).lower()
                if any(phrase in err_lower for phrase in _NON_RETRYABLE):
                    raise RuntimeError(f"LLM API error (non-retryable): {e}") from e
                logger.warning(f"Clause analysis attempt {attempt+1} failed: {e}")
                if attempt == MAX_RETRIES - 1:
                    return ClauseRisk(
                        clause_text=chunk.text[:500],
                        clause_type=chunk.clause_type,
                        risk_level=RiskLevel.MEDIUM,
                        risk_description=f"Analysis failed after {MAX_RETRIES} attempts. Manual review required.",
                        key_concerns=["Automated analysis unavailable"],
                        reference_clause=reranked[0]["text"][:300] if reranked else "",
                        source_citation=reranked[0]["metadata"].get("source_filename", "unknown") if reranked else "",
                        suggested_revision="Manual legal review required.",
                        confidence_score=0.0,
                    )
        return None

    # ── Reduce step ───────────────────────────────────────────────────────────

    def _generate_document_summary(
        self,
        filename: str,
        clause_analyses: list[ClauseRisk],
        missing_clauses: list[MissingClause],
        contract_type: str,
    ) -> DocumentSummary:
        """Synthesize clause-level results into a document-level summary."""
        # Create a compact summary of each clause for the prompt
        clause_summaries = "\n".join([
            f"- [{c.clause_type}] {c.risk_level.value}: {c.risk_description[:150]}"
            for c in clause_analyses
        ])

        prompt = DOCUMENT_SUMMARY_USER.format(
            filename=filename,
            clause_summaries=clause_summaries,
            contract_type_hint=contract_type,
        )

        _NON_RETRYABLE = ("credit balance", "insufficient credits", "payment required", "authentication", "invalid api key", "permission denied")

        for attempt in range(MAX_RETRIES):
            try:
                raw = self.llm.chat(
                    system=DOCUMENT_SUMMARY_SYSTEM,
                    user=prompt,
                    max_tokens=1500,
                )
                data = _extract_json_from_response(raw)

                # Inject pre-computed missing clauses if LLM didn't catch them
                if "missing_clauses" not in data or not data["missing_clauses"]:
                    data["missing_clauses"] = [m.model_dump() for m in missing_clauses]

                return DocumentSummary(**data)

            except Exception as e:
                err_lower = str(e).lower()
                if any(phrase in err_lower for phrase in _NON_RETRYABLE):
                    raise RuntimeError(f"LLM API error (non-retryable): {e}") from e
                logger.warning(f"Document summary attempt {attempt+1} failed: {e}")
                if attempt == MAX_RETRIES - 1:
                    return self._fallback_summary(filename, clause_analyses, missing_clauses, contract_type)

    def _fallback_summary(
        self,
        filename: str,
        clauses: list[ClauseRisk],
        missing: list[MissingClause],
        contract_type: str,
    ) -> DocumentSummary:
        """Compute a basic summary without LLM when generation fails."""
        risk_weights = {"LOW": 1, "MEDIUM": 3, "HIGH": 6, "CRITICAL": 10}
        total_score = sum(risk_weights.get(c.risk_level.value, 3) for c in clauses)
        avg_score = min(10.0, total_score / max(len(clauses), 1))

        if avg_score >= 7:
            overall = RiskLevel.HIGH
        elif avg_score >= 4:
            overall = RiskLevel.MEDIUM
        else:
            overall = RiskLevel.LOW

        return DocumentSummary(
            overall_risk_level=overall,
            overall_risk_score=round(avg_score, 1),
            contract_type=contract_type,
            party_analysis="Automated summary generation was unavailable. Review individual clause assessments.",
            critical_issues=[
                c.risk_description[:100]
                for c in clauses
                if c.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
            ][:5],
            missing_clauses=missing,
            positive_observations=[
                c.risk_description[:100]
                for c in clauses
                if c.risk_level == RiskLevel.LOW
            ][:3],
            executive_summary=(
                f"This {contract_type} contains {sum(1 for c in clauses if c.risk_level in ('HIGH','CRITICAL'))} "
                f"high-risk clauses out of {len(clauses)} analyzed. "
                f"Overall risk score: {avg_score:.1f}/10."
            ),
        )

    def _check_missing_clauses(
        self, contract_type: str, present_types: list[str]
    ) -> list[MissingClause]:
        """Ask LLM to identify missing standard clauses."""
        try:
            prompt = MISSING_CLAUSE_CHECK_USER.format(
                contract_type=contract_type,
                present_clause_types=", ".join(set(present_types)),
            )
            raw = self.llm.chat(
                system=MISSING_CLAUSE_CHECK_SYSTEM,
                user=prompt,
                max_tokens=800,
            )
            data = _extract_json_from_response(raw)
            if isinstance(data, list):
                return [MissingClause(**item) for item in data]
        except Exception as e:
            logger.warning(f"Missing clause check failed: {e}")
        return []

    @staticmethod
    def _format_references(results: list[dict]) -> str:
        """Format retrieved references for injection into the LLM prompt."""
        parts: list[str] = []
        for i, r in enumerate(results, 1):
            meta = r.get("metadata", {})
            source = meta.get("source_filename", "unknown")
            clause_type = meta.get("clause_type", "general")
            parts.append(
                f"[Reference {i}] Source: {source} | Type: {clause_type}\n{r['text'][:600]}"
            )
        return "\n\n".join(parts)

    def answer_question(
        self,
        question: str,
        contract_chunks: list[EnrichedChunk],
        history: Optional[list[tuple[str, str]]] = None,
        max_history_turns: int = 4,
    ) -> dict:
        """
        Answer a user question about the uploaded contract (chat mode).

        Routing: infer the most likely clause category from the question and
        filter the vector search to that category first.  This eliminates
        irrelevant chunks, reduces token usage, and cuts cross-category
        confusion.  Falls back to full-corpus search when the category signal
        is ambiguous or the filtered search returns too few results (the clause
        may have been mis-classified during ingestion).

        history: last N (user, assistant) turn pairs from the Gradio chatbot,
        injected as context so follow-up references ("that cap", "the other
        party") resolve correctly.
        """
        from src.analysis.prompts import CHAT_SYSTEM, CHAT_USER
        from src.analysis.schemas import ChatResponse

        # ── Build history block ───────────────────────────────────────────────
        history_block = ""
        if history:
            recent = history[-max_history_turns:]
            lines = ["PRIOR CONVERSATION:"]
            for user_msg, assistant_msg in recent:
                lines.append(f"User: {user_msg}")
                # Strip JSON wrapper if the stored answer is raw JSON
                try:
                    stored = json.loads(assistant_msg)
                    lines.append(f"Assistant: {stored.get('answer', assistant_msg)[:300]}")
                except Exception:
                    lines.append(f"Assistant: {str(assistant_msg)[:300]}")
            history_block = "\n".join(lines) + "\n\n"

        # ── Rewrite query for retrieval using last turn context ───────────────
        retrieval_query = question
        if history:
            last_user, _ = history[-1]
            # Append the previous question so pronouns resolve in vector search
            retrieval_query = f"{last_user} {question}"

        # ── Category routing (single-type) or decomposition (multi-type) ──────
        clause_type = _infer_question_clause_type(question)
        all_types = _infer_all_clause_types(question)
        routed = False

        if len(all_types) >= 2:
            # Multi-hop: retrieve independently for each detected category,
            # deduplicate by chunk text, then rerank the merged pool.
            logger.debug(f"Multi-type decomposition: {all_types} for question={question!r}")
            seen_texts: set[str] = set()
            merged: list[dict] = []
            for ct in all_types[:3]:   # cap at 3 types to bound latency
                sub = self.hybrid_search.search_contract(
                    query=retrieval_query, top_k=4, clause_type_filter=ct
                )
                for r in sub:
                    key = r["text"][:80]
                    if key not in seen_texts:
                        seen_texts.add(key)
                        merged.append(r)
            results = merged if merged else self.hybrid_search.search_contract(
                query=retrieval_query, top_k=8
            )
            routed = bool(merged)
        elif clause_type:
            logger.debug(f"Chat routing: clause_type={clause_type!r} for question={question!r}")
            results = self.hybrid_search.search_contract(
                query=retrieval_query, top_k=6, clause_type_filter=clause_type
            )
            if len(results) >= 2:
                routed = True
            else:
                logger.debug(
                    f"Filtered search returned {len(results)} result(s); "
                    "falling back to full-corpus search"
                )
                results = self.hybrid_search.search_contract(query=retrieval_query, top_k=8)
        else:
            results = self.hybrid_search.search_contract(query=retrieval_query, top_k=8)

        reranked = self.reranker.rerank(query=retrieval_query, candidates=results, top_n=4)

        if reranked:
            context = "\n\n".join(
                f"[Section: {r['metadata'].get('section_title', 'Unknown')} | "
                f"Type: {r['metadata'].get('clause_type', 'general')}]\n{r['text'][:600]}"
                for r in reranked
            )
        else:
            context = "No relevant clauses found in this contract."

        for attempt in range(MAX_RETRIES):
            try:
                prompt = CHAT_USER.format(
                    question=question,
                    context=context,
                    history_block=history_block,
                )
                raw = self.llm.chat(system=CHAT_SYSTEM, user=prompt, max_tokens=800)
                data = _extract_json_from_response(raw)
                result = ChatResponse(**data).model_dump()
                result["routed_clause_type"] = clause_type
                result["category_filter_applied"] = routed
                self.audit.log_chat(
                    question=question,
                    routed_clause_type=clause_type,
                    category_filter_applied=routed,
                    retrieved_count=len(reranked),
                    answer_preview=result.get("answer", "")[:300],
                    confidence=result.get("confidence", 0.0),
                )
                return result
            except Exception as e:
                logger.warning(f"Chat attempt {attempt+1} failed: {e}")
                if attempt == MAX_RETRIES - 1:
                    return {
                        "answer": "I was unable to generate a response. Please try rephrasing your question.",
                        "relevant_clauses": [],
                        "citations": [],
                        "confidence": 0.0,
                        "disclaimer": "This is AI-generated analysis. Not legal advice.",
                        "routed_clause_type": clause_type,
                        "category_filter_applied": False,
                    }
