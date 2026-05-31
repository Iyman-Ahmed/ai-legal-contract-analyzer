"""
Cross-clause contradiction detector.

After all clauses have been individually analyzed, this agent makes one
LLM call with a compact summary of every clause and asks the model to find
logical conflicts between them.

Why this matters: the risk analysis engine processes clauses in isolation.
A real-world failure pattern is: liability cap in §3 + unlimited indemnity
carve-out in §9. Neither clause is "wrong" alone — the conflict only
appears when viewed together. This agent catches those cases.
"""

import json
import logging
import re
from typing import TYPE_CHECKING

from src.analysis.schemas import ClauseRisk, ContradictionFinding, ContradictionReport
from src.analysis.prompts import CONTRADICTION_SYSTEM, CONTRADICTION_USER

if TYPE_CHECKING:
    from src.analysis.risk_engine import LLMClient

logger = logging.getLogger(__name__)

# Max chars of clause summary fed to the contradiction LLM call
_MAX_SUMMARY_CHARS = 4000


class ContradictionAgent:
    """
    Detects logical conflicts between clauses in the same contract.

    Takes the full list of ClauseRisk objects produced by the risk engine,
    builds a compact clause summary, and asks the LLM to find contradictions.

    Usage:
        agent = ContradictionAgent(llm_client)
        report = agent.detect(clause_analyses)
    """

    def __init__(self, llm: "LLMClient"):
        self.llm = llm

    def detect(self, clause_analyses: list[ClauseRisk]) -> ContradictionReport:
        """
        Detect cross-clause contradictions.

        Returns an empty ContradictionReport (not an error) if there are
        fewer than 2 clauses or if the LLM call fails.
        """
        if len(clause_analyses) < 2:
            return ContradictionReport(
                contradictions=[],
                analysis_note="Too few clauses for contradiction analysis.",
            )

        summary = _build_clause_summary(clause_analyses)

        try:
            prompt = CONTRADICTION_USER.format(clause_summaries=summary)
            raw = self.llm.chat(
                system=CONTRADICTION_SYSTEM,
                user=prompt,
                max_tokens=1500,
            )
            findings = _parse_findings(raw)
            report = ContradictionReport(contradictions=findings)

            if findings:
                logger.info(
                    f"Contradiction detector found {len(findings)} conflict(s), "
                    f"critical={report.has_critical}"
                )
            return report

        except Exception as e:
            logger.warning(f"Contradiction detection failed (non-fatal): {e}")
            return ContradictionReport(
                contradictions=[],
                analysis_note=f"Contradiction analysis unavailable: {type(e).__name__}",
            )


def _build_clause_summary(clauses: list[ClauseRisk]) -> str:
    """
    Build a compact clause summary for the contradiction prompt.

    Each line: [TYPE | RISK] Section: ... | Key text snippet
    Truncated to _MAX_SUMMARY_CHARS to stay within context limits.
    """
    lines: list[str] = []
    for i, c in enumerate(clauses, 1):
        # Use source_citation as the reference if it has content, else a numbered label
        ref = c.source_citation.strip() or f"Clause {i}"
        snippet = c.clause_text[:150].replace("\n", " ")
        line = (
            f"[{c.clause_type.upper()} | {c.risk_level.value}] "
            f"Ref: {ref} | \"{snippet}\""
        )
        lines.append(line)

    full = "\n".join(lines)
    if len(full) > _MAX_SUMMARY_CHARS:
        full = full[:_MAX_SUMMARY_CHARS] + "\n[... truncated]"
    return full


def _parse_findings(text: str) -> list[ContradictionFinding]:
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()

    raw_list: list[dict] = []
    try:
        parsed = json.loads(text)
        raw_list = parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                raw_list = parsed if isinstance(parsed, list) else []
            except json.JSONDecodeError:
                logger.debug(f"Could not parse contradiction JSON: {text[:200]}")
                return []

    findings: list[ContradictionFinding] = []
    for entry in raw_list:
        try:
            findings.append(ContradictionFinding(**entry))
        except Exception as e:
            logger.debug(f"Skipping malformed contradiction entry: {e}")
    return findings
