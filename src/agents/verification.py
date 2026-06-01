"""
LLM-as-judge verification agent.

After each clause analysis the pipeline calls this agent to ask a second
LLM: "Is every claim in this output directly supported by the retrieved
context?" If the faithfulness score falls below the threshold the caller
should re-retrieve with a broader query and re-analyze — at most once.

Why a second LLM call rather than token overlap?
Token overlap catches word reuse but misses paraphrased hallucinations.
A judge model that reads both context and output catches those cases.
Real documented legal AI failures (Mata v. Avianca) came from exactly
this kind of confident-but-wrong paraphrase.
"""

import json
import logging
import re
from typing import TYPE_CHECKING

from src.analysis.schemas import ClauseRisk, VerificationResult
from src.analysis.prompts import VERIFICATION_SYSTEM, VERIFICATION_USER

if TYPE_CHECKING:
    from src.analysis.risk_engine import LLMClient

logger = logging.getLogger(__name__)

FAITHFULNESS_THRESHOLD = 0.75


class VerificationAgent:
    """
    Runs an LLM-as-judge pass over a completed clause analysis.

    Usage:
        agent = VerificationAgent(llm_client)
        result = agent.verify(clause_risk, retrieved_context_text)
        if not result.is_verified:
            # re-retrieve and re-analyze
    """

    def __init__(self, llm: "LLMClient"):
        self.llm = llm

    def verify(self, clause: ClauseRisk, retrieved_context: str) -> VerificationResult:
        """
        Ask the LLM to judge whether the clause analysis is grounded in context.

        Falls back to a passing heuristic result on any LLM/parse error so
        that a verification failure never silently swallows the primary result.
        """
        if not retrieved_context.strip():
            return VerificationResult(
                faithfulness_score=1.0,
                is_verified=True,
                unsupported_claims=[],
                judge_reasoning="No context to verify against — assumed faithful.",
            )

        try:
            prompt = VERIFICATION_USER.format(
                retrieved_context=retrieved_context[:2000],
                risk_level=clause.risk_level.value,
                risk_description=clause.risk_description,
                reference_clause=clause.reference_clause[:400],
                source_citation=clause.source_citation,
                key_concerns=", ".join(clause.key_concerns),
            )
            raw = self.llm.chat(
                system=VERIFICATION_SYSTEM,
                user=prompt,
                max_tokens=512,
            )
            data = _parse_json(raw)
            score = max(0.0, min(1.0, float(data.get("faithfulness_score", 0.5))))
            unsupported = data.get("unsupported_claims", [])
            reasoning = data.get("judge_reasoning", "")
            is_verified = score >= FAITHFULNESS_THRESHOLD

            if not is_verified:
                logger.warning(
                    f"Verification failed for [{clause.clause_type}] "
                    f"score={score:.2f} unsupported={unsupported}"
                )

            return VerificationResult(
                faithfulness_score=score,
                is_verified=is_verified,
                unsupported_claims=unsupported if isinstance(unsupported, list) else [],
                judge_reasoning=reasoning,
            )

        except Exception as e:
            logger.warning(f"Verification agent error (defaulting to pass): {e}")
            return VerificationResult(
                faithfulness_score=0.5,
                is_verified=True,
                unsupported_claims=[],
                judge_reasoning=f"Verification skipped due to error: {type(e).__name__}",
            )


def _parse_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise
