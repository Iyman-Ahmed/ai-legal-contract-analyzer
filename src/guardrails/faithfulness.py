"""
Faithfulness guardrail.

Checks that LLM outputs are grounded in the retrieved context.
Uses a lightweight heuristic approach (keyword overlap) to avoid
adding another LLM call to the pipeline — fast and deterministic.
"""

import re
import logging
from src.analysis.schemas import ClauseRisk

logger = logging.getLogger(__name__)

_MIN_OVERLAP_THRESHOLD = 0.15  # 15% token overlap required


def _tokenize(text: str) -> set[str]:
    """Simple tokenizer for overlap calculation."""
    text = text.lower()
    tokens = re.findall(r"\b\w+\b", text)
    stopwords = {"the", "a", "an", "in", "of", "to", "and", "or", "for", "is", "are",
                 "was", "were", "this", "that", "it", "with", "by", "from"}
    return {t for t in tokens if t not in stopwords and len(t) > 2}


class FaithfulnessChecker:
    """
    Verifies that generated risk descriptions are grounded in retrieved context.
    Uses token overlap as a proxy for faithfulness.
    """

    def check_clause(
        self,
        clause: ClauseRisk,
        retrieved_context: str,
    ) -> tuple[bool, float, str]:
        """
        Check if a clause analysis is faithful to retrieved context.

        Args:
            clause: The generated clause analysis.
            retrieved_context: Combined text of retrieved reference documents.

        Returns:
            (is_faithful, overlap_score, explanation)
        """
        if not retrieved_context.strip():
            return True, 1.0, "No context to check against"

        # Tokens from the generated risk description
        generated_tokens = _tokenize(clause.risk_description)
        # Tokens from retrieved reference clause
        ref_tokens = _tokenize(clause.reference_clause)
        # Tokens from the retrieved context
        context_tokens = _tokenize(retrieved_context)

        if not generated_tokens:
            return True, 1.0, "Empty generated text"

        # Check overlap: how many generated assertion tokens appear in context
        all_context = context_tokens | ref_tokens
        overlap = generated_tokens & all_context
        overlap_ratio = len(overlap) / len(generated_tokens) if generated_tokens else 1.0

        is_faithful = overlap_ratio >= _MIN_OVERLAP_THRESHOLD

        explanation = (
            f"Token overlap: {overlap_ratio:.1%} "
            f"({len(overlap)}/{len(generated_tokens)} tokens found in context)"
        )

        if not is_faithful:
            logger.warning(
                f"Low faithfulness for {clause.clause_type}: {explanation}"
            )

        return is_faithful, overlap_ratio, explanation

    def check_all(
        self,
        clauses: list[ClauseRisk],
        retrieved_context: str,
    ) -> list[dict]:
        """
        Check faithfulness for all clauses.

        Returns:
            List of {clause_type, is_faithful, score, explanation} dicts.
        """
        return [
            {
                "clause_type": c.clause_type,
                **dict(zip(["is_faithful", "score", "explanation"],
                           self.check_clause(c, retrieved_context))),
            }
            for c in clauses
        ]
