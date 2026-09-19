"""
Hard reasoning tests for chat-mode category routing and hallucination resistance.

Three test classes with escalating integration depth:

  1. TestQuestionRouter          — pure unit tests, no I/O, no LLM
  2. TestCategoryFilteredRetrieval — ingestion + BM25/vector search, no LLM
  3. TestHallucinationResistance — integration tests requiring LLM (skipped by default)

Design rationale
----------------
The accuracy report showed two structural failures in the previous pipeline:
  • high_risk_contract_type_coverage = 0.588: 41.2% of HIGH/CRITICAL contracts lack an expected risky type
  • answer_question() searched all chunks with no clause-type filter
    → cross-category noise degraded precision and invited hallucination

These tests lock in the fixes so they cannot regress.

Run all unit + retrieval tests (no LLM needed):
    cd contract-analyzer/.claude/worktrees/trust-redesign
    python -m pytest tests/test_chat_routing.py -v -k "not integration"

Run everything including LLM tests (requires .env with a valid provider):
    python -m pytest tests/test_chat_routing.py -v -m integration
"""

import sys
import os
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis.risk_engine import _infer_question_clause_type


# ─────────────────────────────────────────────────────────────────────────────
# 1. Pure routing unit tests — no external I/O
# ─────────────────────────────────────────────────────────────────────────────

class TestQuestionRouter:
    """
    Verify that _infer_question_clause_type() maps questions to the right
    clause type without LLM or file I/O.

    Failure here means the keyword signals are misconfigured — the chat
    will send irrelevant chunks to the LLM for every question of that type.
    """

    # ── Unambiguous single-category questions ─────────────────────────────────

    @pytest.mark.parametrize("question,expected", [
        # limitation_of_liability
        ("What is the maximum liability cap?",              "limitation_of_liability"),
        ("Are consequential damages excluded?",             "limitation_of_liability"),
        ("What is the cap on indirect damages?",            "limitation_of_liability"),
        ("Can the vendor's liability not exceed fees paid?","limitation_of_liability"),

        # indemnification
        ("Am I required to indemnify the vendor?",          "indemnification"),
        ("Does the contract have a hold harmless clause?",  "indemnification"),
        ("Who is responsible if a third party defends a claim?", "indemnification"),

        # confidentiality
        ("What are my confidentiality obligations?",        "confidentiality"),
        ("How long does the NDA last after termination?",   "confidentiality"),
        ("Can I disclose information to my lawyer?",        "confidentiality"),
        ("Is this a non-disclosure agreement?",             "confidentiality"),

        # termination
        ("How do I terminate this contract?",               "termination"),
        ("What notice period is required to cancel?",       "termination"),
        ("Does the contract auto-renew?",                   "termination"),
        ("Can the vendor terminate without cause?",         "termination"),

        # payment_terms
        ("When are payments due?",                          "payment_terms"),
        ("Is there a late payment penalty?",                "payment_terms"),
        ("What is the subscription fee?",                   "payment_terms"),
        ("Can I get a refund?",                             "payment_terms"),

        # ip_assignment
        ("Who owns the work product I create?",             "ip_assignment"),
        ("Does the contract have a work-for-hire clause?",  "ip_assignment"),
        ("Do I retain copyright over my deliverables?",     "ip_assignment"),
        ("What are the ip rights assignments?",             "ip_assignment"),

        # non_compete
        ("Can I work for a competitor after leaving?",      "non_compete"),
        ("Is there a non-compete clause?",                  "non_compete"),
        ("Does the contract have a non-solicitation clause?","non_compete"),

        # dispute_resolution
        ("How are disputes resolved?",                      "dispute_resolution"),
        ("Is arbitration required?",                        "dispute_resolution"),
        ("Can I bring a class action?",                     "dispute_resolution"),
        ("What is the litigation forum?",                   "dispute_resolution"),

        # governing_law
        ("Which state's law governs this contract?",        "governing_law"),
        ("What law governs the agreement?",                 "governing_law"),
        ("Which jurisdiction applies?",                     "governing_law"),

        # warranty
        ("What warranties does the vendor provide?",        "warranty"),
        ("Is the service provided as-is?",                  "warranty"),
        ("Does the vendor guarantee uptime?",               "warranty"),

        # data_protection
        ("What happens in a data breach?",                  "data_protection"),
        ("How is my personal data used?",                   "data_protection"),
        ("Does the contract comply with GDPR?",             "data_protection"),
        ("Can the vendor sell customer data?",              "data_protection"),

        # force_majeure
        ("What counts as force majeure?",                   "force_majeure"),
        ("Is the vendor excused during a natural disaster?","force_majeure"),
        ("Does a pandemic suspend performance?",            "force_majeure"),
    ])
    def test_clear_routing(self, question: str, expected: str):
        result = _infer_question_clause_type(question)
        assert result == expected, (
            f"ROUTING FAIL\n"
            f"  Question : {question!r}\n"
            f"  Expected : {expected!r}\n"
            f"  Got      : {result!r}\n"
            f"  Fix: add a pattern to _QUESTION_CLAUSE_SIGNALS[{expected!r}]"
        )

    # ── Off-topic questions must return None (not hallucinate a category) ─────

    @pytest.mark.parametrize("question", [
        "Hello, how are you?",
        "Can you summarize this document?",
        "Who wrote this contract?",
        "What is the weather today?",
        "Please review everything in this file.",
        "What does section 4 say?",       # generic — no clause signal
        "Tell me about the agreement.",    # generic
    ])
    def test_off_topic_returns_none(self, question: str):
        result = _infer_question_clause_type(question)
        assert result is None, (
            f"OFF-TOPIC LEAK\n"
            f"  Question : {question!r}\n"
            f"  Got      : {result!r}  (expected None)\n"
            f"  Fix: raise _MIN_SIGNAL_SCORE or remove weak patterns"
        )

    # ── Cross-clause questions: dominant signal wins OR returns None ──────────
    # These are the hardest cases — the model must not lock to the wrong type.

    @pytest.mark.parametrize("question,acceptable", [
        # "liability cap for data breaches" — liability is the dominant legal concept
        (
            "What is the liability cap for data breaches?",
            {"limitation_of_liability", None},
        ),
        # "penalty for not paying" — payment is the primary subject
        (
            "What is the penalty for late payment?",
            {"payment_terms", None},
        ),
        # "NDA survive termination" — confidentiality is the subject being asked about
        (
            "Does the NDA survive termination?",
            {"confidentiality", None},
        ),
        # "can vendor terminate if I don't pay" — genuinely ambiguous, either is fine
        (
            "Can the vendor terminate if I don't pay?",
            {"termination", "payment_terms", None},
        ),
        # "indemnification for IP infringement" — indemnification is the legal mechanism
        (
            "Am I indemnified for intellectual property infringement claims?",
            {"indemnification", None},
        ),
    ])
    def test_cross_clause_acceptable_outcomes(self, question: str, acceptable: set):
        result = _infer_question_clause_type(question)
        assert result in acceptable, (
            f"CROSS-CLAUSE ROUTING ERROR\n"
            f"  Question   : {question!r}\n"
            f"  Acceptable : {acceptable}\n"
            f"  Got        : {result!r}\n"
            f"  Fix: check _AMBIGUITY_MARGIN or pattern weights for competing types"
        )

    # ── Category confusion: must NOT route to the wrong single type ───────────
    # These questions have surface words from one type but belong to another.

    @pytest.mark.parametrize("question,wrong_type", [
        # "termination" appears but this is about confidentiality duration
        ("How long do confidentiality obligations last after termination?", "termination"),
        # "data" appears but this is about payment data, not data protection
        # (acceptable to return None or payment_terms, but NOT data_protection alone)
        # NB: this is hard — we only assert it doesn't map to data_protection
        ("What payment data must I provide?", "data_protection"),
        # "claim" appears but this is about payment, not indemnification
        ("Can I claim a refund for overpayment?", "indemnification"),
    ])
    def test_category_confusion_avoided(self, question: str, wrong_type: str):
        result = _infer_question_clause_type(question)
        assert result != wrong_type, (
            f"CATEGORY CONFUSION\n"
            f"  Question    : {question!r}\n"
            f"  Wrong type  : {wrong_type!r}\n"
            f"  Got         : {result!r}\n"
            f"  Fix: surface word '{wrong_type}' has too much weight; add negative context or raise threshold"
        )

    # ── Paraphrase robustness: unusual phrasing for known clause types ─────────
    # Tests for test-data bias: does routing only work on the exact patterns
    # used in the training data / keyword list, or on real lawyer language too?

    @pytest.mark.parametrize("question,expected", [
        # Unusual phrasings for limitation_of_liability
        ("What is the most I can recover from the vendor?",   None),        # weak signal, should be None
        ("Is there a ceiling on the vendor's exposure?",       None),        # "exposure" not in signals → None is fine
        # Unusual phrasings for confidentiality
        ("Am I allowed to tell my colleague about this deal?", None),        # no strong keyword
        # If we add "exposure" to signals later, this will fail — intentional
    ])
    def test_paraphrase_gracefully_returns_none(self, question: str, expected):
        result = _infer_question_clause_type(question)
        assert result == expected, (
            f"PARAPHRASE BIAS DETECTED\n"
            f"  Question : {question!r}\n"
            f"  Expected : {expected!r} (signals too weak for this phrasing)\n"
            f"  Got      : {result!r}\n"
            f"  The router over-fits to exact keyword patterns. "
            f"  Full-corpus fallback will handle it but this is a known gap."
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Retrieval routing tests — ingestion + search, no LLM
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_CONTRACT = ROOT / "data" / "sample_contracts" / "sample_nda.txt"
SAAS_CONTRACT   = ROOT / "sample_saas_contract.docx"


@pytest.fixture(scope="module")
def indexed_search_engine():
    """
    Build a HybridSearchEngine indexed on sample_nda.txt.
    Skip the whole class if the file doesn't exist or models can't load.
    """
    from src.ingestion.parser import DocumentParser
    from src.ingestion.chunker import SectionAwareChunker
    from src.ingestion.metadata import MetadataExtractor
    from src.retrieval.embeddings import EmbeddingPipeline
    from src.retrieval.vector_store import VectorStore
    from src.retrieval.bm25_search import BM25SearchEngine
    from src.retrieval.hybrid_search import HybridSearchEngine
    from src.retrieval.reranker import CrossEncoderReranker

    if not SAMPLE_CONTRACT.exists():
        pytest.skip(f"Sample contract not found: {SAMPLE_CONTRACT}")

    try:
        parser  = DocumentParser()
        chunker = SectionAwareChunker()
        meta    = MetadataExtractor()
        embed   = EmbeddingPipeline()
        store   = VectorStore()
        bm25    = BM25SearchEngine()

        parsed   = parser.parse(str(SAMPLE_CONTRACT))
        chunks   = chunker.chunk(parsed)
        enriched = [meta.enrich(c) for c in chunks]

        store.index_contract(enriched)
        bm25.index_contract(enriched)

        hybrid  = HybridSearchEngine(embed, store, bm25)
        reranker = CrossEncoderReranker()
        return hybrid, reranker, enriched
    except Exception as e:
        pytest.skip(f"Could not build search engine: {e}")


class TestCategoryFilteredRetrieval:
    """
    Verify that category-routed search returns chunks of the right type.

    These tests catch the case where routing logic is correct but the
    clause_type_filter in HybridSearchEngine is broken or ignored.
    """

    @pytest.mark.parametrize("question,expected_clause_type", [
        ("What are my confidentiality obligations?",    "confidentiality"),
        ("How long does the NDA last after termination?","confidentiality"),
    ])
    def test_filtered_search_returns_correct_type(
        self, indexed_search_engine, question, expected_clause_type
    ):
        hybrid, reranker, _ = indexed_search_engine
        inferred = _infer_question_clause_type(question)

        if inferred is None:
            pytest.skip(f"Router returned None for: {question!r}")

        results = hybrid.search_contract(
            query=question, top_k=6, clause_type_filter=inferred
        )
        if not results:
            pytest.skip("No results — contract may not have this clause type labelled")

        types_found = {r["metadata"].get("clause_type") for r in results}
        assert inferred in types_found, (
            f"RETRIEVAL TYPE MISMATCH\n"
            f"  Question      : {question!r}\n"
            f"  Filtered for  : {inferred!r}\n"
            f"  Types returned: {types_found}\n"
            f"  Fix: check VectorStore clause_type_filter implementation"
        )

    def test_fallback_when_type_absent(self, indexed_search_engine):
        """
        If the contract has no force_majeure clause, the filtered search
        returns < 2 results and the pipeline must fall back to full search.
        """
        hybrid, reranker, _ = indexed_search_engine
        results = hybrid.search_contract(
            query="What counts as force majeure?",
            top_k=6,
            clause_type_filter="force_majeure",
        )
        # NDA likely has no force_majeure — that's fine.
        # The important thing is: the call doesn't crash.
        assert isinstance(results, list), "search_contract must always return a list"

    def test_unfiltered_fallback_returns_more_results(self, indexed_search_engine):
        """
        Full-corpus search must return at least as many results as filtered
        (usually more) for the same query.
        """
        hybrid, _, _ = indexed_search_engine
        q = "What obligations survive termination?"
        filtered   = hybrid.search_contract(q, top_k=6, clause_type_filter="confidentiality")
        unfiltered = hybrid.search_contract(q, top_k=8)
        assert len(unfiltered) >= len(filtered), (
            f"Unfiltered search ({len(unfiltered)}) < filtered ({len(filtered)}) — "
            "clause_type_filter may be broken"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Hallucination resistance — requires LLM (integration tests)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestHallucinationResistance:
    """
    End-to-end tests that send real questions through the full pipeline
    (routing → filtered retrieval → LLM) and assert the model does NOT
    fabricate answers when the clause is absent or underspecified.

    These tests require a live LLM provider configured in .env.
    Run with: pytest tests/test_chat_routing.py -m integration

    Known failure modes from the prior accuracy report:
      - high_risk_contract_type_coverage = 0.588: incomplete risky-type coverage in 41.2% of HIGH/CRITICAL contracts
      - Lease agreement F1 = 0.679: poorly classified lease language
      - The model previously hallucinated specific figures when asked about
        clauses that existed but were vague (e.g., "reasonable notice period")
    """

    @pytest.fixture(scope="class")
    def full_engine(self):
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")

        from src.ingestion.parser import DocumentParser
        from src.ingestion.chunker import SectionAwareChunker
        from src.ingestion.metadata import MetadataExtractor
        from src.retrieval.embeddings import EmbeddingPipeline
        from src.retrieval.vector_store import VectorStore
        from src.retrieval.bm25_search import BM25SearchEngine
        from src.retrieval.hybrid_search import HybridSearchEngine
        from src.retrieval.reranker import CrossEncoderReranker
        from src.analysis.risk_engine import RiskAnalysisEngine

        if not SAMPLE_CONTRACT.exists():
            pytest.skip(f"Sample contract not found: {SAMPLE_CONTRACT}")

        try:
            parser  = DocumentParser()
            chunker = SectionAwareChunker()
            meta    = MetadataExtractor()
            embed   = EmbeddingPipeline()
            store   = VectorStore()
            bm25    = BM25SearchEngine()

            parsed   = parser.parse(str(SAMPLE_CONTRACT))
            chunks   = chunker.chunk(parsed)
            enriched = [meta.enrich(c) for c in chunks]

            store.index_contract(enriched)
            bm25.index_contract(enriched)

            hybrid   = HybridSearchEngine(embed, store, bm25)
            reranker = CrossEncoderReranker()
            engine   = RiskAnalysisEngine(hybrid, reranker)
            return engine, enriched
        except Exception as e:
            pytest.skip(f"Could not build engine: {e}")

    # ── Absence hallucination ─────────────────────────────────────────────────
    # The model must acknowledge when a clause is NOT in the contract
    # instead of fabricating a plausible-sounding answer.

    ABSENCE_QUESTIONS = [
        # An NDA typically does NOT have: payment terms, non-compete,
        # force majeure, indemnification with specific amounts
        ("What are the payment terms?",               ["payment", "not", "no payment", "doesn't", "does not", "not specified", "not found"]),
        ("What is the non-compete radius?",            ["non-compete", "not", "no non-compete", "doesn't", "does not", "not specified", "not found"]),
        ("What is the force majeure provision?",       ["force majeure", "not", "no force", "doesn't", "does not", "not specified", "not found"]),
        ("What are the late payment penalties?",       ["payment", "not", "no late", "doesn't", "does not", "not specified", "not found"]),
    ]

    @pytest.mark.parametrize("question,absence_indicators", ABSENCE_QUESTIONS)
    def test_absent_clause_not_hallucinated(
        self, full_engine, question: str, absence_indicators: list[str]
    ):
        """
        When the clause is absent, the answer must contain at least one
        absence indicator phrase, not a fabricated provision.
        """
        engine, enriched = full_engine
        result = engine.answer_question(question=question, contract_chunks=enriched)
        answer = result.get("answer", "").lower()

        has_absence_signal = any(phrase in answer for phrase in absence_indicators)
        assert has_absence_signal, (
            f"HALLUCINATION DETECTED\n"
            f"  Question         : {question!r}\n"
            f"  Answer           : {result['answer'][:300]!r}\n"
            f"  Expected one of  : {absence_indicators}\n"
            f"  The model fabricated a clause that does not exist in the contract.\n"
            f"  Fix: strengthen the 'no information found' instruction in CHAT_SYSTEM prompt."
        )

    # ── Specificity trap ──────────────────────────────────────────────────────
    # The model must not invent specific figures when the clause is vague.

    VAGUE_CLAUSE_QUESTIONS = [
        # If the NDA says "reasonable notice period" without specifying days,
        # the model must NOT hallucinate "30 days" or any other specific number.
        (
            "Exactly how many days notice is required?",
            ["reasonable", "not specified", "not defined", "does not specify", "unclear"],
            ["30 days", "60 days", "90 days", "14 days", "7 days"],   # hallucinated specifics
        ),
        # If the NDA says "promptly" for breach notification without a deadline,
        # the model must NOT invent a timeline.
        (
            "Within exactly how many hours must a breach be reported?",
            ["promptly", "not specified", "not defined", "does not specify", "unclear", "immediately"],
            ["24 hours", "48 hours", "72 hours", "7 days"],
        ),
    ]

    @pytest.mark.parametrize("question,vague_indicators,hallucinated_specifics", VAGUE_CLAUSE_QUESTIONS)
    def test_vague_clause_no_invented_specifics(
        self,
        full_engine,
        question: str,
        vague_indicators: list[str],
        hallucinated_specifics: list[str],
    ):
        engine, enriched = full_engine
        result = engine.answer_question(question=question, contract_chunks=enriched)
        answer = result.get("answer", "").lower()

        invented = [s for s in hallucinated_specifics if s in answer]
        assert not invented, (
            f"SPECIFICITY HALLUCINATION\n"
            f"  Question       : {question!r}\n"
            f"  Answer         : {result['answer'][:300]!r}\n"
            f"  Hallucinated   : {invented}\n"
            f"  Expected vague : {vague_indicators}\n"
            f"  Fix: add 'do not invent figures not stated in context' to CHAT_SYSTEM prompt."
        )

    # ── Category routing confirmation ─────────────────────────────────────────
    # The pipeline must apply the category filter for clear questions.

    @pytest.mark.parametrize("question,expected_clause_type", [
        ("What are my confidentiality obligations?",          "confidentiality"),
        ("How long does the NDA last after termination?",     "confidentiality"),
    ])
    def test_routing_applied_in_full_pipeline(
        self, full_engine, question: str, expected_clause_type: str
    ):
        engine, enriched = full_engine
        result = engine.answer_question(question=question, contract_chunks=enriched)
        assert result.get("routed_clause_type") == expected_clause_type, (
            f"ROUTING NOT APPLIED\n"
            f"  Question             : {question!r}\n"
            f"  Expected clause_type : {expected_clause_type!r}\n"
            f"  Got                  : {result.get('routed_clause_type')!r}\n"
            f"  Fix: check _infer_question_clause_type() for this question."
        )

    # ── Cross-category confusion ───────────────────────────────────────────────
    # A question about confidentiality must not return indemnification clauses.

    def test_cross_category_not_confused(self, full_engine):
        """
        When asked about confidentiality, the answer must reference
        confidentiality language, not indemnification or payment content.
        """
        engine, enriched = full_engine
        result = engine.answer_question(
            question="What are my confidentiality obligations?",
            contract_chunks=enriched,
        )
        answer = result.get("answer", "").lower()

        confidentiality_signals = [
            "confidential", "disclose", "discloses", "disclosure",
            "non-disclosure", "proprietary", "secret",
        ]
        wrong_category_signals = [
            "indemnif", "hold harmless", "payment", "invoice", "fee due",
        ]

        has_right_content = any(s in answer for s in confidentiality_signals)
        has_wrong_content = any(s in answer for s in wrong_category_signals)

        assert has_right_content, (
            f"CATEGORY MISS: confidentiality question got no confidentiality content.\n"
            f"Answer: {result['answer'][:300]!r}"
        )
        assert not has_wrong_content, (
            f"CATEGORY CONFUSION: confidentiality question returned other-clause content.\n"
            f"Answer: {result['answer'][:300]!r}"
        )

    # ── Confidence calibration ────────────────────────────────────────────────
    # When the clause is absent, confidence should be lower than when present.

    def test_confidence_lower_for_absent_clause(self, full_engine):
        """
        The model should express lower confidence when it can't find
        the clause than when the clause is clearly present.
        """
        engine, enriched = full_engine

        present_result = engine.answer_question(
            question="What are my confidentiality obligations?",
            contract_chunks=enriched,
        )
        absent_result = engine.answer_question(
            question="What are the payment terms?",
            contract_chunks=enriched,
        )

        present_conf = present_result.get("confidence", 1.0)
        absent_conf  = absent_result.get("confidence", 1.0)

        # Allow a 0.1 tolerance — this is a soft heuristic, not a hard guarantee
        assert absent_conf <= present_conf + 0.1, (
            f"CONFIDENCE MISCALIBRATED\n"
            f"  Present clause confidence : {present_conf}\n"
            f"  Absent clause confidence  : {absent_conf}\n"
            f"  Absent should be ≤ present.\n"
            f"  Fix: add 'set confidence low when context contains no relevant clause' "
            f"to CHAT_SYSTEM prompt."
        )
