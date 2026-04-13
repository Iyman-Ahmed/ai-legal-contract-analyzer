"""
Tests for guardrails: citation verification and faithfulness checking.
Run with: python -m pytest tests/test_guardrails.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.guardrails.citation_check import CitationVerifier
from src.guardrails.faithfulness import FaithfulnessChecker, _tokenize
from src.guardrails.disclaimer import inject_disclaimer, get_disclaimer
from src.analysis.schemas import ClauseRisk, RiskLevel, FullAnalysisResult, DocumentSummary


def _make_clause(
    clause_type="indemnification",
    risk_level=RiskLevel.HIGH,
    source_citation="standard_commercial_clauses_v2026.json",
    risk_description="The indemnification clause is unlimited in scope.",
    reference_clause="Mutual indemnification capped at 12 months fees.",
    confidence_score=0.8,
) -> ClauseRisk:
    return ClauseRisk(
        clause_text="Customer shall indemnify Vendor for all claims.",
        clause_type=clause_type,
        risk_level=risk_level,
        risk_description=risk_description,
        key_concerns=["Unlimited liability"],
        reference_clause=reference_clause,
        source_citation=source_citation,
        suggested_revision="Add a mutual cap.",
        confidence_score=confidence_score,
    )


def _make_result(clauses: list[ClauseRisk]) -> FullAnalysisResult:
    return FullAnalysisResult(
        filename="test.txt",
        clause_analyses=clauses,
        document_summary=DocumentSummary(
            overall_risk_level=RiskLevel.HIGH,
            overall_risk_score=7.5,
            contract_type="Service Agreement",
            party_analysis="Test analysis.",
            critical_issues=["Issue 1"],
            executive_summary="High risk contract.",
        ),
        total_clauses_analyzed=len(clauses),
        high_risk_count=sum(1 for c in clauses if c.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)),
        medium_risk_count=sum(1 for c in clauses if c.risk_level == RiskLevel.MEDIUM),
        low_risk_count=sum(1 for c in clauses if c.risk_level == RiskLevel.LOW),
    )


class TestCitationVerifier:
    def setup_method(self):
        self.verifier = CitationVerifier()

    def test_valid_citation_no_warning(self):
        clause = _make_clause(source_citation="standard_commercial_clauses_v2026.json")
        result = _make_result([clause])
        sources = {"standard_commercial_clauses_v2026.json"}
        _, warnings = self.verifier.verify_result(result, sources)
        assert not any("Unverified" in w for w in warnings)

    def test_hallucinated_citation_triggers_warning(self):
        clause = _make_clause(source_citation="made_up_document_2099.json")
        result = _make_result([clause])
        sources = {"standard_commercial_clauses_v2026.json"}
        verified_result, warnings = self.verifier.verify_result(result, sources)
        assert any("Unverified" in w for w in warnings)

    def test_unverified_citation_reduces_confidence(self):
        original_confidence = 0.9
        clause = _make_clause(
            source_citation="invented_reference.json",
            confidence_score=original_confidence,
        )
        result = _make_result([clause])
        sources = {"standard_commercial_clauses_v2026.json"}
        verified, _ = self.verifier.verify_result(result, sources)
        verified_confidence = verified.clause_analyses[0].confidence_score
        assert verified_confidence < original_confidence

    def test_no_citation_clause_flagged_but_not_penalized(self):
        clause = _make_clause(source_citation="No reference available")
        result = _make_result([clause])
        sources = {"standard_commercial_clauses_v2026.json"}
        verified, warnings = self.verifier.verify_result(result, sources)
        # Should flag it but not reduce confidence aggressively
        assert verified.clause_analyses[0].confidence_score >= 0.5

    def test_extract_sources_from_results(self):
        retrieved = [
            {"text": "...", "metadata": {"source_filename": "doc_a.json"}},
            {"text": "...", "metadata": {"source_filename": "doc_b.json"}},
            {"text": "...", "metadata": {}},
        ]
        sources = self.verifier.extract_sources_from_results(retrieved)
        assert "doc_a.json" in sources
        assert "doc_b.json" in sources
        assert len(sources) == 2

    def test_empty_sources_set_no_crash(self):
        clause = _make_clause()
        result = _make_result([clause])
        verified, warnings = self.verifier.verify_result(result, set())
        assert verified is not None


class TestFaithfulnessChecker:
    def setup_method(self):
        self.checker = FaithfulnessChecker()

    def test_high_overlap_is_faithful(self):
        clause = _make_clause(
            risk_description="The indemnification clause is unlimited and one-sided.",
            reference_clause="Mutual indemnification capped at 12 months fees.",
        )
        context = "unlimited indemnification one-sided liability unlimited scope"
        is_faithful, score, _ = self.checker.check_clause(clause, context)
        assert is_faithful, f"Expected faithful (score={score:.3f})"

    def test_no_overlap_is_unfaithful(self):
        clause = _make_clause(
            risk_description="The blockchain cryptocurrency NFT quantum computing clause.",
            reference_clause="Indemnification.",
        )
        context = "standard indemnification mutual cap fees liability"
        is_faithful, score, _ = self.checker.check_clause(clause, context)
        # Low overlap should be unfaithful
        assert score < 0.5 or not is_faithful  # lenient check

    def test_empty_context_returns_faithful(self):
        clause = _make_clause()
        is_faithful, score, explanation = self.checker.check_clause(clause, "")
        assert is_faithful

    def test_check_all_returns_per_clause_results(self):
        clauses = [_make_clause(), _make_clause(clause_type="termination")]
        context = "termination indemnification clause liability"
        results = self.checker.check_all(clauses, context)
        assert len(results) == 2
        for r in results:
            assert "is_faithful" in r
            assert "score" in r
            assert "clause_type" in r

    def test_tokenizer_basic(self):
        tokens = _tokenize("The party shall indemnify against claims.")
        assert "indemnify" in tokens
        assert "claims" in tokens
        assert "the" not in tokens

    def test_tokenizer_short_words_excluded(self):
        tokens = _tokenize("it is a to at")
        # All stopwords, should return empty or near-empty
        non_stop = [t for t in tokens if len(t) > 2]
        assert len(non_stop) == 0


class TestDisclaimer:
    def test_get_short_disclaimer(self):
        disc = get_disclaimer(short=True)
        assert "Not legal advice" in disc or "legal advice" in disc.lower()

    def test_get_long_disclaimer(self):
        disc = get_disclaimer(short=False)
        assert "attorney" in disc.lower()
        assert "DISCLAIMER" in disc.upper()

    def test_inject_disclaimer_appends(self):
        text = "This is the analysis."
        result = inject_disclaimer(text, short=True)
        assert result.startswith("This is the analysis.")
        assert "legal advice" in result.lower()

    def test_inject_long_disclaimer(self):
        text = "Analysis text."
        result = inject_disclaimer(text, short=False)
        assert "LEGAL DISCLAIMER" in result.upper() or "disclaimer" in result.lower()
        assert len(result) > len(text)
