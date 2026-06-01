"""
Smoke test for the three new trust-redesign agents.

Runs against LM Studio (or whichever LLM_PROVIDER is in .env) without
loading ChromaDB, embedding models, or Gradio — fast to execute.

Usage:
    cd contract-analyzer/.claude/worktrees/trust-redesign
    PYTHONPATH=. python scripts/smoke_test_agents.py

Make sure LM Studio is running on http://localhost:1234 before running.
"""

import sys
import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Load .env from this directory
load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.risk_engine import LLMClient
from src.analysis.schemas import (
    ClauseRisk, RiskLevel, VerificationResult,
    ObligationItem, ObligationTable,
    ContradictionFinding, ContradictionReport,
)
from src.agents.verification import VerificationAgent
from src.agents.obligation import ObligationAgent
from src.agents.contradiction import ContradictionAgent
from src.ingestion.metadata import EnrichedChunk

# ── Sample contract data ───────────────────────────────────────────────────────

SAMPLE_CLAUSE = ClauseRisk(
    clause_text=(
        "Customer shall indemnify, defend, and hold harmless Vendor and its officers "
        "from any and all claims, damages, losses, and expenses arising out of "
        "Customer's use of the Service, with no limitation on the amount of indemnification."
    ),
    clause_type="indemnification",
    risk_level=RiskLevel.HIGH,
    risk_description=(
        "This one-sided indemnification clause requires the Customer to cover all "
        "losses with no cap, placing unlimited financial exposure on the Customer."
    ),
    key_concerns=[
        "No cap on indemnification amount",
        "One-sided — only Customer indemnifies",
        "Broad 'any and all' scope",
    ],
    reference_clause=(
        "Each party shall indemnify the other for claims arising from its own negligence "
        "or willful misconduct, capped at the total fees paid in the prior 12 months."
    ),
    source_citation="Standard SaaS Mutual Indemnification Template, Section 8.1",
    suggested_revision=(
        "Replace with mutual indemnification capped at 12 months of fees paid."
    ),
    confidence_score=0.85,
)

RETRIEVED_CONTEXT = """
[Reference 1] Standard SaaS indemnification clauses should be mutual.
Each party should indemnify the other only for claims arising from its own negligence.
Caps tied to fees paid in the prior 12 months are market standard.
One-sided unlimited indemnification is a significant risk for the customer.
"""

SAMPLE_CHUNKS = [
    EnrichedChunk(
        chunk_id="c1",
        text=(
            "Section 3. Limitation of Liability. Vendor's total liability shall not exceed "
            "the fees paid by Customer in the six months prior to the claim."
        ),
        section_title="Section 3",
        clause_number="3",
        page_number=2,
        chunk_index=0,
        char_start=500,
        char_end=700,
        word_count=30,
        source_filename="sample_saas.txt",
        clause_type="limitation_of_liability",
        clause_type_confidence=0.9,
        detected_signals=["liability", "total liability", "fees paid"],
    ),
    EnrichedChunk(
        chunk_id="c2",
        text=(
            "Section 9. Indemnification. Customer shall indemnify Vendor from any and all "
            "claims with no limitation on amount, including claims arising from Vendor's own negligence."
        ),
        section_title="Section 9",
        clause_number="9",
        page_number=5,
        chunk_index=1,
        char_start=2100,
        char_end=2350,
        word_count=32,
        source_filename="sample_saas.txt",
        clause_type="indemnification",
        clause_type_confidence=0.88,
        detected_signals=["indemnify", "claims", "no limitation"],
    ),
    EnrichedChunk(
        chunk_id="c3",
        text=(
            "Section 12. Payment. Customer shall pay all invoices within 30 days of receipt. "
            "Late payments incur 2% monthly interest. Vendor may suspend service after 45 days."
        ),
        section_title="Section 12",
        clause_number="12",
        page_number=7,
        chunk_index=2,
        char_start=3100,
        char_end=3350,
        word_count=33,
        source_filename="sample_saas.txt",
        clause_type="payment_terms",
        clause_type_confidence=0.92,
        detected_signals=["pay", "invoices", "30 days", "interest"],
    ),
]

SAMPLE_CLAUSES_FOR_CONTRADICTION = [
    ClauseRisk(
        clause_text="Vendor's total liability shall not exceed fees paid in six months.",
        clause_type="limitation_of_liability",
        risk_level=RiskLevel.MEDIUM,
        risk_description="Liability cap of 6 months fees — standard but low.",
        key_concerns=["Cap may be too low for high-value contracts"],
        reference_clause="Standard cap is 12 months.",
        source_citation="Section 3",
        suggested_revision="Increase cap to 12 months fees.",
        confidence_score=0.8,
    ),
    ClauseRisk(
        clause_text=(
            "Customer shall indemnify Vendor from any and all claims with no limitation "
            "on amount, including claims arising from Vendor's own negligence."
        ),
        clause_type="indemnification",
        risk_level=RiskLevel.CRITICAL,
        risk_description="Unlimited one-sided indemnification overrides liability cap in §3.",
        key_concerns=["No cap", "Covers Vendor negligence", "Contradicts §3"],
        reference_clause="Mutual indemnification with cap.",
        source_citation="Section 9",
        suggested_revision="Add mutual indemnification with cap matching §3.",
        confidence_score=0.9,
    ),
]


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def fail(msg: str) -> None:
    print(f"  ✗ {msg}")


def run_all():
    print("\nSmoke Test — New Trust Redesign Agents")
    print(f"Provider: {os.getenv('LLM_PROVIDER')} | Model: {os.getenv('LMSTUDIO_MODEL', 'default')}")

    # ── Connect to LLM ─────────────────────────────────────────────────────────
    section("1. LLM Connection")
    try:
        llm = LLMClient()
        ping = llm.chat("You are a test assistant.", "Reply with just the word: CONNECTED", max_tokens=10)
        ok(f"LLM client connected → response: {ping.strip()[:40]!r}")
    except Exception as e:
        fail(f"LLM connection failed: {e}")
        print("\nMake sure LM Studio is running on http://localhost:1234")
        sys.exit(1)

    # ── Verification Agent ─────────────────────────────────────────────────────
    section("2. Verification Agent (LLM-as-judge)")
    try:
        agent = VerificationAgent(llm)
        result = agent.verify(SAMPLE_CLAUSE, RETRIEVED_CONTEXT)
        ok(f"faithfulness_score = {result.faithfulness_score:.2f}")
        ok(f"is_verified        = {result.is_verified}")
        ok(f"judge_reasoning    = {result.judge_reasoning[:80]!r}")
        if result.unsupported_claims:
            ok(f"unsupported_claims = {result.unsupported_claims}")
        assert isinstance(result.faithfulness_score, float), "score not float"
        assert isinstance(result.is_verified, bool), "is_verified not bool"
        ok("Schema validation passed")
    except Exception as e:
        fail(f"Verification agent failed: {e}")
        import traceback; traceback.print_exc()

    # ── Obligation Agent ───────────────────────────────────────────────────────
    section("3. Obligation Extractor Agent")
    try:
        agent = ObligationAgent(llm)
        table = agent.extract(SAMPLE_CHUNKS)
        ok(f"obligations extracted = {len(table.obligations)}")
        ok(f"high_priority_count   = {table.high_priority_count}")
        if table.obligations:
            first = table.obligations[0]
            ok(f"Sample obligation     = {first.obligation[:60]!r}")
            ok(f"  party={first.party!r}, deadline={first.deadline!r}")
            ok(f"  type={first.obligation_type!r}")
        if table.extraction_note:
            ok(f"Note: {table.extraction_note}")
        ok("Schema validation passed")
    except Exception as e:
        fail(f"Obligation agent failed: {e}")
        import traceback; traceback.print_exc()

    # ── Contradiction Agent ────────────────────────────────────────────────────
    section("4. Contradiction Detector Agent")
    try:
        agent = ContradictionAgent(llm)
        report = agent.detect(SAMPLE_CLAUSES_FOR_CONTRADICTION)
        ok(f"contradictions found = {len(report.contradictions)}")
        ok(f"has_critical         = {report.has_critical}")
        if report.contradictions:
            c = report.contradictions[0]
            ok(f"Conflict: {c.clause_a_reference} vs {c.clause_b_reference}")
            ok(f"  {c.conflict_description[:80]!r}")
            ok(f"  risk_level = {c.risk_level.value}")
        if report.analysis_note:
            ok(f"Note: {report.analysis_note}")
        ok("Schema validation passed")
    except Exception as e:
        fail(f"Contradiction agent failed: {e}")
        import traceback; traceback.print_exc()

    section("Done")
    print("  All agents tested. Check output above for any ✗ failures.\n")


if __name__ == "__main__":
    run_all()
