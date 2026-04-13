"""
Citation verification guardrail.

Ensures every citation in the LLM output actually corresponds to
a retrieved reference document. Catches hallucinated source names.
"""

import logging
from src.analysis.schemas import ClauseRisk, FullAnalysisResult

logger = logging.getLogger(__name__)


class CitationVerifier:
    """
    Verifies that citations in analysis outputs match actual retrieved sources.
    """

    def verify_result(
        self,
        result: FullAnalysisResult,
        retrieved_sources: set[str],
    ) -> tuple[FullAnalysisResult, list[str]]:
        """
        Verify citations in a FullAnalysisResult against retrieved sources.

        Args:
            result: The full analysis result to verify.
            retrieved_sources: Set of source filenames that were actually retrieved.

        Returns:
            (verified_result, list_of_warnings)
            Clauses with unverifiable citations have confidence_score reduced.
        """
        warnings: list[str] = []
        verified_clauses: list[ClauseRisk] = []

        for clause in result.clause_analyses:
            clause_copy = clause.model_copy()
            citation = clause.source_citation.strip()

            if not citation or citation in ("No reference available", "unknown"):
                # No citation — flag but don't penalize (may be legitimate)
                warnings.append(
                    f"Clause '{clause.clause_type}' has no source citation"
                )
                verified_clauses.append(clause_copy)
                continue

            # Check if citation references a known source
            citation_verified = any(
                src.lower() in citation.lower() or citation.lower() in src.lower()
                for src in retrieved_sources
            )

            if not citation_verified and retrieved_sources:
                warning = (
                    f"Unverified citation in '{clause.clause_type}': '{citation}' "
                    f"(known sources: {list(retrieved_sources)[:3]})"
                )
                warnings.append(warning)
                logger.warning(warning)

                # Reduce confidence for unverified citations
                object.__setattr__(
                    clause_copy,
                    "confidence_score",
                    max(0.0, clause.confidence_score - 0.3)
                )
                object.__setattr__(
                    clause_copy,
                    "risk_description",
                    clause.risk_description + " [Note: citation could not be verified against retrieved sources]"
                )

            verified_clauses.append(clause_copy)

        # Return a modified result with verified clauses
        # (using model_copy since Pydantic models are immutable by default)
        verified = result.model_copy(update={"clause_analyses": verified_clauses})
        return verified, warnings

    def extract_sources_from_results(self, retrieved_results: list[dict]) -> set[str]:
        """Extract set of source filenames from retrieval results."""
        sources: set[str] = set()
        for r in retrieved_results:
            meta = r.get("metadata", {})
            source = meta.get("source_filename", "")
            if source:
                sources.add(source)
        return sources
