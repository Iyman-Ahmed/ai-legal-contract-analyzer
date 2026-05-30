"""
Pydantic v2 schemas for all structured LLM outputs.

Using strict schemas forces the LLM to produce parseable JSON and
enables retry logic when the output doesn't match the expected structure.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class VerificationResult(BaseModel):
    """Output from the LLM-as-judge faithfulness check."""

    faithfulness_score: float = Field(
        ge=0.0, le=1.0,
        description="0-1 score: fraction of claims grounded in retrieved context"
    )
    is_verified: bool = Field(
        description="True when faithfulness_score >= 0.75"
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Claims in the analysis not traceable to retrieved context"
    )
    judge_reasoning: str = Field(
        default="",
        description="One-sentence explanation from the judge"
    )

    @field_validator("faithfulness_score", mode="before")
    @classmethod
    def clamp_score(cls, v) -> float:
        return max(0.0, min(1.0, float(v)))


class ClauseRisk(BaseModel):
    """Risk assessment for a single clause."""

    clause_text: str = Field(
        description="The original clause text from the uploaded contract"
    )
    clause_type: str = Field(
        description="Type of clause (e.g., indemnification, termination)"
    )
    risk_level: RiskLevel = Field(
        description="Risk severity: LOW, MEDIUM, HIGH, or CRITICAL"
    )
    risk_description: str = Field(
        description="Plain-English explanation of why this clause is risky or acceptable"
    )
    key_concerns: list[str] = Field(
        default_factory=list,
        description="Bullet-point list of specific concerns (2-4 items)"
    )
    reference_clause: str = Field(
        description="The standard/benchmark clause this was compared against"
    )
    source_citation: str = Field(
        description="Reference document name and section the comparison came from"
    )
    suggested_revision: str = Field(
        description="Concrete suggestion for how the clause could be improved"
    )
    confidence_score: float = Field(
        ge=0.0, le=1.0,
        description="Model confidence in this assessment (0.0 to 1.0)"
    )
    is_missing: bool = Field(
        default=False,
        description="True if this represents a missing standard clause"
    )
    verification: Optional[VerificationResult] = Field(
        default=None,
        description="LLM-as-judge faithfulness check result (None if not yet verified)"
    )

    @field_validator("reference_clause", "suggested_revision", "source_citation", mode="before")
    @classmethod
    def coerce_none_str(cls, v) -> str:
        return v if v is not None else ""

    @field_validator("risk_level", mode="before")
    @classmethod
    def normalize_risk_level(cls, v: str) -> str:
        return v.upper().strip() if isinstance(v, str) else v

    @field_validator("confidence_score", mode="before")
    @classmethod
    def clamp_confidence(cls, v) -> float:
        return max(0.0, min(1.0, float(v)))


class MissingClause(BaseModel):
    """Represents a standard clause that is absent from the contract."""
    clause_type: str
    description: str
    risk_level: RiskLevel
    recommended_text: str

    @field_validator("risk_level", mode="before")
    @classmethod
    def normalize_risk_level(cls, v) -> str:
        if not isinstance(v, str):
            return v
        v = v.upper().strip()
        if "-" in v or "/" in v:
            sep = "-" if "-" in v else "/"
            parts = [p.strip() for p in v.split(sep)]
            order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
            return max(parts, key=lambda x: order.get(x, 1))
        return v


class DocumentSummary(BaseModel):
    """Overall document-level risk summary."""

    overall_risk_level: RiskLevel = Field(
        description="Aggregate risk level for the entire document"
    )
    overall_risk_score: float = Field(
        ge=0.0, le=10.0,
        description="Numeric risk score 0-10 (10 = most risky)"
    )
    contract_type: str = Field(
        description="Detected contract type (NDA, SaaS, Employment, Lease, etc.)"
    )
    party_analysis: str = Field(
        description="Brief analysis of which party bears more risk"
    )
    critical_issues: list[str] = Field(
        description="Top 3-5 most critical concerns in the document"
    )
    missing_clauses: list[MissingClause] = Field(
        default_factory=list,
        description="Standard clauses not found in this document"
    )
    positive_observations: list[str] = Field(
        default_factory=list,
        description="Well-drafted clauses or protective provisions found"
    )
    executive_summary: str = Field(
        description="2-3 sentence executive summary for non-lawyers"
    )


class ObligationItem(BaseModel):
    """A single actionable obligation extracted from the contract."""

    obligation: str = Field(description="Plain-English description of what must be done")
    party: str = Field(description="Which party must fulfill this (e.g. 'Customer', 'Vendor', 'Both')")
    deadline: str = Field(description="When it must be done (date, event, or 'Ongoing')")
    consequence: str = Field(description="What happens if missed or breached")
    clause_reference: str = Field(description="Section or clause number where this appears")
    obligation_type: str = Field(
        description="Category: payment | notice | renewal | termination | delivery | reporting | other"
    )


class ObligationTable(BaseModel):
    """All actionable obligations extracted from the contract."""

    obligations: list[ObligationItem] = Field(default_factory=list)
    high_priority_count: int = Field(
        default=0,
        description="Obligations with hard deadlines or high financial consequence"
    )
    extraction_note: str = Field(
        default="",
        description="Any caveats about extraction completeness"
    )

    @classmethod
    def from_items(cls, items: list[ObligationItem]) -> "ObligationTable":
        high_priority_types = {"payment", "notice", "termination", "renewal"}
        high = sum(1 for i in items if i.obligation_type in high_priority_types)
        return cls(obligations=items, high_priority_count=high)


class FullAnalysisResult(BaseModel):
    """Complete analysis output for a contract."""
    filename: str
    clause_analyses: list[ClauseRisk]
    document_summary: DocumentSummary
    total_clauses_analyzed: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    verified_count: int = Field(
        default=0,
        description="Clauses that passed the LLM-as-judge faithfulness check"
    )
    obligation_table: Optional[ObligationTable] = Field(
        default=None,
        description="All actionable obligations extracted from the contract"
    )
    analysis_version: str = "2.0"

    @classmethod
    def from_clause_list(
        cls,
        filename: str,
        clause_analyses: list[ClauseRisk],
        document_summary: DocumentSummary,
    ) -> "FullAnalysisResult":
        risk_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
        for c in clause_analyses:
            risk_counts[c.risk_level.value] += 1

        verified_count = sum(
            1 for c in clause_analyses
            if c.verification is not None and c.verification.is_verified
        )

        return cls(
            filename=filename,
            clause_analyses=clause_analyses,
            document_summary=document_summary,
            total_clauses_analyzed=len(clause_analyses),
            high_risk_count=risk_counts["HIGH"] + risk_counts["CRITICAL"],
            medium_risk_count=risk_counts["MEDIUM"],
            low_risk_count=risk_counts["LOW"],
            verified_count=verified_count,
            obligation_table=None,  # populated by RiskAnalysisEngine after extraction
        )


class ChatResponse(BaseModel):
    """Structured response for the Q&A chat interface."""
    answer: str = Field(description="Direct answer to the user's question")
    relevant_clauses: list[str] = Field(
        default_factory=list,
        description="Relevant clause texts supporting the answer"
    )
    citations: list[str] = Field(
        default_factory=list,
        description="Source sections from the contract"
    )
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    disclaimer: str = Field(
        default="This is AI-generated analysis for informational purposes only. "
                "It does not constitute legal advice. Consult a qualified attorney "
                "before acting on any information provided."
    )
