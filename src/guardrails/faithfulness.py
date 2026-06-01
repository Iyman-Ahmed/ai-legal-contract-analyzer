"""
Faithfulness guardrail.

When an LLMClient is provided the checker delegates to VerificationAgent
(LLM-as-judge). Without one it falls back to the original token-overlap
heuristic — this keeps existing unit tests working without an API key.
"""

import re
import logging
from typing import Optional, TYPE_CHECKING

from src.analysis.schemas import ClauseRisk

if TYPE_CHECKING:
    from src.analysis.risk_engine import LLMClient

logger = logging.getLogger(__name__)

_MIN_OVERLAP_THRESHOLD = 0.15


def _tokenize(text: str) -> set:
    text = text.lower()
    tokens = re.findall(r"\b\w+\b", text)
    stopwords = {"the", "a", "an", "in", "of", "to", "and", "or", "for", "is", "are",
                 "was", "were", "this", "that", "it", "with", "by", "from"}
    return {t for t in tokens if t not in stopwords and len(t) > 2}


class FaithfulnessChecker:
    """
    Verifies that generated risk descriptions are grounded in retrieved context.

    With an LLM client: uses LLM-as-judge (VerificationAgent).
    Without one: uses token-overlap heuristic (fast, no API call needed).
    """

    def __init__(self, llm_client: Optional["LLMClient"] = None):
        self._agent = None
        if llm_client is not None:
            from src.agents.verification import VerificationAgent
            self._agent = VerificationAgent(llm_client)

    def check_clause(
        self,
        clause: ClauseRisk,
        retrieved_context: str,
    ) -> tuple[bool, float, str]:
        """
        Returns (is_faithful, score, explanation).
        Score is faithfulness_score from judge or overlap ratio from heuristic.
        """
        if self._agent is not None:
            result = self._agent.verify(clause, retrieved_context)
            return result.is_verified, result.faithfulness_score, result.judge_reasoning

        return self._heuristic_check(clause, retrieved_context)

    def check_all(
        self,
        clauses: list[ClauseRisk],
        retrieved_context: str,
    ) -> list[dict]:
        return [
            {
                "clause_type": c.clause_type,
                **dict(zip(
                    ["is_faithful", "score", "explanation"],
                    self.check_clause(c, retrieved_context),
                )),
            }
            for c in clauses
        ]

    # ── heuristic fallback ────────────────────────────────────────────────────

    def _heuristic_check(
        self,
        clause: ClauseRisk,
        retrieved_context: str,
    ) -> tuple[bool, float, str]:
        if not retrieved_context.strip():
            return True, 1.0, "No context to check against"

        generated_tokens = _tokenize(clause.risk_description)
        ref_tokens = _tokenize(clause.reference_clause)
        context_tokens = _tokenize(retrieved_context)

        if not generated_tokens:
            return True, 1.0, "Empty generated text"

        all_context = context_tokens | ref_tokens
        overlap = generated_tokens & all_context
        overlap_ratio = len(overlap) / len(generated_tokens)
        is_faithful = overlap_ratio >= _MIN_OVERLAP_THRESHOLD

        explanation = (
            f"Token overlap: {overlap_ratio:.1%} "
            f"({len(overlap)}/{len(generated_tokens)} tokens found in context)"
        )
        if not is_faithful:
            logger.warning(f"Low faithfulness for {clause.clause_type}: {explanation}")

        return is_faithful, overlap_ratio, explanation
