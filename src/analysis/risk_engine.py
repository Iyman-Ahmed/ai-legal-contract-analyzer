"""
Core risk analysis engine.

Uses a map-reduce strategy:
  Map:    Analyze each clause independently against retrieved references.
  Reduce: Synthesize clause results into a document-level summary.

Pydantic validation with retry ensures structured output.
"""

import json
import logging
import re
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

logger = logging.getLogger(__name__)


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

        # ── Missing clause check ───────────────────────────────────────────────
        if progress_callback:
            progress_callback("Checking for missing standard clauses...", 0.85)

        missing = self._check_missing_clauses(
            contract_type, [c.clause_type for c in clause_analyses]
        )

        # ── Reduce step: document summary ─────────────────────────────────────
        if progress_callback:
            progress_callback("Generating document summary...", 0.90)

        doc_summary = self._generate_document_summary(
            filename, clause_analyses, missing, contract_type
        )

        if progress_callback:
            progress_callback("Analysis complete.", 1.0)

        return FullAnalysisResult.from_clause_list(
            filename=filename,
            clause_analyses=clause_analyses,
            document_summary=doc_summary,
        )

    # ── Map step ──────────────────────────────────────────────────────────────

    def _analyze_single_clause(self, chunk: EnrichedChunk) -> Optional[ClauseRisk]:
        """Analyze one clause with retries for Pydantic validation failures."""
        # Retrieve relevant reference clauses
        candidates = self.hybrid_search.search_reference(
            query=chunk.text,
            top_k=10,
            clause_type_filter=chunk.clause_type if chunk.clause_type != "general" else None,
        )
        reranked = self.reranker.rerank(query=chunk.text, candidates=candidates, top_n=4)

        if not reranked:
            # No reference found — create a default assessment
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

        reference_text = self._format_references(reranked)

        _NON_RETRYABLE = ("credit balance", "billing", "authentication", "invalid api key", "permission")

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
                return ClauseRisk(**data)

            except Exception as e:
                err_lower = str(e).lower()
                if any(phrase in err_lower for phrase in _NON_RETRYABLE):
                    raise RuntimeError(
                        f"LLM API error (non-retryable): {e}"
                    ) from e
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

        _NON_RETRYABLE = ("credit balance", "billing", "authentication", "invalid api key", "permission")

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
    ) -> dict:
        """
        Answer a user question about the uploaded contract (chat mode).
        """
        from src.analysis.prompts import CHAT_SYSTEM, CHAT_USER
        from src.analysis.schemas import ChatResponse

        # Search contract for relevant context
        results = self.hybrid_search.search_contract(query=question, top_k=8)
        reranked = self.reranker.rerank(query=question, candidates=results, top_n=4)

        context = "\n\n".join(
            f"[Section: {r['metadata'].get('section_title', 'Unknown')}]\n{r['text'][:600]}"
            for r in reranked
        ) if reranked else "No relevant clauses found in this contract."

        for attempt in range(MAX_RETRIES):
            try:
                prompt = CHAT_USER.format(question=question, context=context)
                raw = self.llm.chat(system=CHAT_SYSTEM, user=prompt, max_tokens=800)
                data = _extract_json_from_response(raw)
                return ChatResponse(**data).model_dump()
            except Exception as e:
                logger.warning(f"Chat attempt {attempt+1} failed: {e}")
                if attempt == MAX_RETRIES - 1:
                    return {
                        "answer": "I was unable to generate a response. Please try rephrasing your question.",
                        "relevant_clauses": [],
                        "citations": [],
                        "confidence": 0.0,
                        "disclaimer": "This is AI-generated analysis. Not legal advice.",
                    }
