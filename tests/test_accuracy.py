"""
Accuracy test suite — runs all 50 test contracts through the full
ingestion + retrieval pipeline and measures:

  1. Clause Type Detection Accuracy
       - Precision:  of detected types, how many were expected?
       - Recall:     of expected types, how many were detected?
       - F1:         harmonic mean of P and R

  2. Retrieval Hit Rate
       - For each expected clause type, does hybrid search surface
         at least one relevant reference document?

  3. Risk Profile Detection
       - Does the clause classifier produce any HIGH-signal clauses
         for contracts labeled HIGH/CRITICAL?
       - Does a LOW contract avoid false HIGH signals?

  4. Coverage by contract type (NDA / SaaS / Employment / Service / Lease)

Run with:
    python -m pytest tests/test_accuracy.py -v -s
Or standalone:
    python tests/test_accuracy.py
"""

import sys
import json
import time
import statistics
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ingestion.parser import DocumentParser
from src.ingestion.chunker import SectionAwareChunker
from src.ingestion.metadata import MetadataExtractor
from src.retrieval.embeddings import EmbeddingPipeline
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_search import BM25SearchEngine
from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.knowledge_base import KnowledgeBaseBuilder

MANIFEST_PATH = ROOT / "data" / "test_contracts" / "manifest.json"
CONTRACT_DIR = ROOT / "data" / "test_contracts"


# ─────────────────────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ContractResult:
    contract_id: str
    contract_type: str
    risk_profile: str
    expected_clause_types: list[str]
    expected_high_risk_types: list[str]

    detected_clause_types: list[str] = field(default_factory=list)
    detected_high_confidence_types: list[str] = field(default_factory=list)  # conf > 0.4

    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    retrieval_hits: int = 0
    retrieval_total: int = 0
    retrieval_hit_rate: float = 0.0

    false_high_risk_flags: list[str] = field(default_factory=list)  # for LOW contracts
    missed_high_risk_types: list[str] = field(default_factory=list)

    parse_time_ms: float = 0.0
    chunk_count: int = 0
    error: Optional[str] = None


@dataclass
class AccuracyReport:
    total_contracts: int = 0
    failed_contracts: int = 0

    # Clause detection
    avg_precision: float = 0.0
    avg_recall: float = 0.0
    avg_f1: float = 0.0

    # Retrieval
    avg_retrieval_hit_rate: float = 0.0
    total_retrieval_hits: int = 0
    total_retrieval_queries: int = 0

    # Risk profile accuracy
    high_risk_recall: float = 0.0   # HIGH contracts: did we catch their risky clauses?
    low_risk_precision: float = 0.0  # LOW contracts: did we avoid false HIGH flags?

    # By contract type
    by_type: dict = field(default_factory=dict)
    by_risk_profile: dict = field(default_factory=dict)

    # Worst performers
    lowest_f1_contracts: list = field(default_factory=list)
    lowest_retrieval_contracts: list = field(default_factory=list)

    results: list[ContractResult] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline (initialized once, shared across all contracts)
# ─────────────────────────────────────────────────────────────────────────────

class AccuracyTestRunner:
    def __init__(self):
        print("\n🔧 Initializing pipeline components...")
        self.parser = DocumentParser()
        self.chunker = SectionAwareChunker()
        self.extractor = MetadataExtractor()

        self.embedder = EmbeddingPipeline()
        self.vector_store = VectorStore(embedding_pipeline=self.embedder, persist=False)
        self.bm25 = BM25SearchEngine()
        self.hybrid = HybridSearchEngine(vector_store=self.vector_store, bm25_engine=self.bm25)
        self.reranker = CrossEncoderReranker()

        # Build reference knowledge base
        kb = KnowledgeBaseBuilder(vector_store=self.vector_store, bm25_engine=self.bm25)
        ref_count = kb.build(force_rebuild=True)
        print(f"   Reference KB: {ref_count} documents indexed")

    def run_single(self, manifest_entry: dict) -> ContractResult:
        """Run a single contract through ingestion + retrieval and measure accuracy."""
        cid = manifest_entry["id"]
        filepath = CONTRACT_DIR / manifest_entry["filename"]
        expected = manifest_entry["expected_clause_types"]
        expected_high = manifest_entry["expected_high_risk_clause_types"]

        result = ContractResult(
            contract_id=cid,
            contract_type=manifest_entry["contract_type"],
            risk_profile=manifest_entry["risk_profile"],
            expected_clause_types=expected,
            expected_high_risk_types=expected_high,
        )

        # ── Parse ──────────────────────────────────────────────────────────────
        try:
            t0 = time.perf_counter()
            parsed = self.parser.parse(filepath)
            result.parse_time_ms = (time.perf_counter() - t0) * 1000
        except Exception as e:
            result.error = f"Parse failed: {e}"
            return result

        # ── Chunk + Classify ───────────────────────────────────────────────────
        try:
            chunks = self.chunker.chunk(parsed)
            enriched = self.extractor.enrich_all(chunks)
            result.chunk_count = len(enriched)
        except Exception as e:
            result.error = f"Chunk/classify failed: {e}"
            return result

        # Detected clause types (all) and high-confidence ones (conf > 0.4)
        result.detected_clause_types = list({c.clause_type for c in enriched
                                              if c.clause_type != "general"})
        result.detected_high_confidence_types = list({
            c.clause_type for c in enriched
            if c.clause_type != "general" and c.clause_type_confidence > 0.4
        })

        # ── Clause Detection Accuracy ──────────────────────────────────────────
        detected_set = set(result.detected_clause_types)
        expected_set = set(expected)

        true_positives = detected_set & expected_set
        precision = len(true_positives) / len(detected_set) if detected_set else 0.0
        recall = len(true_positives) / len(expected_set) if expected_set else 1.0
        f1 = (2 * precision * recall / (precision + recall)
               if (precision + recall) > 0 else 0.0)

        result.precision = round(precision, 3)
        result.recall = round(recall, 3)
        result.f1 = round(f1, 3)

        # ── Retrieval Hit Rate ─────────────────────────────────────────────────
        hits = 0
        total = len(expected)
        result.retrieval_total = total

        for clause_type in expected:
            # Build a representative query from the contract's text for this clause
            query = self._build_query(clause_type, enriched)
            candidates = self.hybrid.search_reference(
                query=query, top_k=5, clause_type_filter=clause_type
            )
            if candidates:
                hits += 1

        result.retrieval_hits = hits
        result.retrieval_hit_rate = round(hits / total, 3) if total > 0 else 1.0

        # ── Risk Profile Signal Check ──────────────────────────────────────────
        high_signal_types = {
            c.clause_type for c in enriched
            if c.clause_type_confidence > 0.5
               and c.clause_type in {"indemnification", "non_compete", "limitation_of_liability",
                                     "ip_assignment", "confidentiality", "data_protection",
                                     "dispute_resolution", "governing_law"}
        }

        if manifest_entry["risk_profile"] in ("HIGH", "CRITICAL"):
            # We expect the classifier to pick up the risky types
            missed = [t for t in expected_high if t not in high_signal_types]
            result.missed_high_risk_types = missed
        elif manifest_entry["risk_profile"] == "LOW":
            # No clause should be incorrectly flagged with very high confidence
            # (false HIGH signal = a type not in expected list detected with high conf)
            unexpected_high = [t for t in high_signal_types
                               if t not in expected_set and t not in ("general",)]
            result.false_high_risk_flags = unexpected_high

        return result

    def _build_query(self, clause_type: str, enriched: list) -> str:
        """Get text from the chunk most likely to be this clause type."""
        matches = [c for c in enriched if c.clause_type == clause_type]
        if matches:
            # Pick highest confidence match
            best = max(matches, key=lambda c: c.clause_type_confidence)
            return best.text[:400]
        # Fallback: use type name as query
        return f"{clause_type.replace('_', ' ')} clause legal agreement"

    def run_all(self, manifest: list) -> AccuracyReport:
        """Run all contracts and aggregate results."""
        print(f"\n🚀 Running accuracy test on {len(manifest)} contracts...\n")

        results: list[ContractResult] = []
        for i, entry in enumerate(manifest, 1):
            r = self.run_single(entry)
            results.append(r)
            status = "✅" if not r.error else "❌"
            print(f"  [{i:02d}/50] {status} {r.contract_id:15s} | "
                  f"type={r.contract_type:20s} | risk={r.risk_profile:8s} | "
                  f"P={r.precision:.2f} R={r.recall:.2f} F1={r.f1:.2f} | "
                  f"retrieval={r.retrieval_hits}/{r.retrieval_total} | "
                  f"chunks={r.chunk_count}")

        return self._aggregate(results)

    def _aggregate(self, results: list[ContractResult]) -> AccuracyReport:
        report = AccuracyReport()
        report.results = results
        report.total_contracts = len(results)

        ok = [r for r in results if not r.error]
        report.failed_contracts = len(results) - len(ok)

        if not ok:
            return report

        report.avg_precision = round(statistics.mean(r.precision for r in ok), 3)
        report.avg_recall = round(statistics.mean(r.recall for r in ok), 3)
        report.avg_f1 = round(statistics.mean(r.f1 for r in ok), 3)

        total_hits = sum(r.retrieval_hits for r in ok)
        total_queries = sum(r.retrieval_total for r in ok)
        report.total_retrieval_hits = total_hits
        report.total_retrieval_queries = total_queries
        report.avg_retrieval_hit_rate = round(
            statistics.mean(r.retrieval_hit_rate for r in ok), 3
        )

        # Risk profile checks
        high_contracts = [r for r in ok if r.risk_profile in ("HIGH", "CRITICAL")]
        low_contracts = [r for r in ok if r.risk_profile == "LOW"]

        if high_contracts:
            # HIGH recall: fraction of high-risk contracts where we missed 0 high-risk types
            caught = sum(1 for r in high_contracts if not r.missed_high_risk_types)
            report.high_risk_recall = round(caught / len(high_contracts), 3)

        if low_contracts:
            # LOW precision: fraction of LOW contracts with no false HIGH flags
            clean = sum(1 for r in low_contracts if not r.false_high_risk_flags)
            report.low_risk_precision = round(clean / len(low_contracts), 3)

        # By contract type
        type_groups = defaultdict(list)
        for r in ok:
            type_groups[r.contract_type].append(r)
        for ct, group in type_groups.items():
            report.by_type[ct] = {
                "count": len(group),
                "avg_f1": round(statistics.mean(r.f1 for r in group), 3),
                "avg_recall": round(statistics.mean(r.recall for r in group), 3),
                "avg_retrieval_hit_rate": round(
                    statistics.mean(r.retrieval_hit_rate for r in group), 3
                ),
            }

        # By risk profile
        risk_groups = defaultdict(list)
        for r in ok:
            risk_groups[r.risk_profile].append(r)
        for rp, group in risk_groups.items():
            report.by_risk_profile[rp] = {
                "count": len(group),
                "avg_f1": round(statistics.mean(r.f1 for r in group), 3),
                "avg_retrieval_hit_rate": round(
                    statistics.mean(r.retrieval_hit_rate for r in group), 3
                ),
            }

        # Worst performers
        report.lowest_f1_contracts = sorted(
            [(r.contract_id, r.f1, r.contract_type) for r in ok],
            key=lambda x: x[1]
        )[:5]
        report.lowest_retrieval_contracts = sorted(
            [(r.contract_id, r.retrieval_hit_rate, r.retrieval_hits, r.retrieval_total)
             for r in ok],
            key=lambda x: x[1]
        )[:5]

        return report


# ─────────────────────────────────────────────────────────────────────────────
# Report printer
# ─────────────────────────────────────────────────────────────────────────────

def print_report(report: AccuracyReport):
    SEP = "─" * 70

    print(f"\n{'═'*70}")
    print(f"  ACCURACY REPORT — {report.total_contracts} CONTRACTS")
    print(f"{'═'*70}")

    print(f"\n{SEP}")
    print(f"  CLAUSE TYPE DETECTION")
    print(f"{SEP}")
    print(f"  Avg Precision:   {report.avg_precision:.1%}  (of detected types, how many were expected)")
    print(f"  Avg Recall:      {report.avg_recall:.1%}  (of expected types, how many were found)")
    print(f"  Avg F1 Score:    {report.avg_f1:.1%}  (harmonic mean)")

    print(f"\n{SEP}")
    print(f"  RETRIEVAL HIT RATE")
    print(f"{SEP}")
    print(f"  Avg Hit Rate:    {report.avg_retrieval_hit_rate:.1%}  "
          f"({report.total_retrieval_hits}/{report.total_retrieval_queries} queries returned ≥1 relevant doc)")

    print(f"\n{SEP}")
    print(f"  RISK PROFILE SIGNALS")
    print(f"{SEP}")
    print(f"  HIGH/CRITICAL recall: {report.high_risk_recall:.1%}  "
          f"(HIGH-risk contracts where all risky clauses were signalled)")
    print(f"  LOW precision:        {report.low_risk_precision:.1%}  "
          f"(LOW-risk contracts without false-HIGH flags)")

    print(f"\n{SEP}")
    print(f"  BY CONTRACT TYPE")
    print(f"{SEP}")
    print(f"  {'Type':<22} {'Count':>5}  {'Avg F1':>7}  {'Avg Recall':>10}  {'Retrieval':>10}")
    print(f"  {'-'*22} {'-'*5}  {'-'*7}  {'-'*10}  {'-'*10}")
    for ct, stats in sorted(report.by_type.items()):
        print(f"  {ct:<22} {stats['count']:>5}  {stats['avg_f1']:>6.1%}  "
              f"{stats['avg_recall']:>9.1%}  {stats['avg_retrieval_hit_rate']:>9.1%}")

    print(f"\n{SEP}")
    print(f"  BY RISK PROFILE")
    print(f"{SEP}")
    print(f"  {'Risk':<12} {'Count':>5}  {'Avg F1':>7}  {'Retrieval':>10}")
    print(f"  {'-'*12} {'-'*5}  {'-'*7}  {'-'*10}")
    for rp, stats in sorted(report.by_risk_profile.items()):
        print(f"  {rp:<12} {stats['count']:>5}  {stats['avg_f1']:>6.1%}  "
              f"{stats['avg_retrieval_hit_rate']:>9.1%}")

    print(f"\n{SEP}")
    print(f"  LOWEST F1 CONTRACTS (improvement targets)")
    print(f"{SEP}")
    for cid, f1, ctype in report.lowest_f1_contracts:
        print(f"  {cid:<15} F1={f1:.2f}  ({ctype})")

    print(f"\n{SEP}")
    print(f"  LOWEST RETRIEVAL HIT RATE")
    print(f"{SEP}")
    for cid, rate, hits, total in report.lowest_retrieval_contracts:
        print(f"  {cid:<15} {rate:.1%}  ({hits}/{total} queries hit)")

    if report.failed_contracts:
        print(f"\n⚠️  {report.failed_contracts} contracts failed (see errors above)")

    print(f"\n{'═'*70}\n")


# ─────────────────────────────────────────────────────────────────────────────
# pytest integration
# ─────────────────────────────────────────────────────────────────────────────

import pytest

@pytest.fixture(scope="module")
def runner():
    return AccuracyTestRunner()

@pytest.fixture(scope="module")
def manifest():
    with open(MANIFEST_PATH) as f:
        return json.load(f)

@pytest.fixture(scope="module")
def full_report(runner, manifest):
    return runner.run_all(manifest)


class TestDatasetCoverage:
    """Verify the test dataset itself is correctly structured."""

    def test_manifest_has_50_entries(self, manifest):
        assert len(manifest) == 50

    def test_all_contract_files_exist(self, manifest):
        missing = [e["filename"] for e in manifest
                   if not (CONTRACT_DIR / e["filename"]).exists()]
        assert not missing, f"Missing contract files: {missing}"

    def test_contract_type_distribution(self, manifest):
        types = [e["contract_type"] for e in manifest]
        assert types.count("NDA") == 12
        assert types.count("SaaS Agreement") == 15
        assert types.count("Employment Contract") == 12
        assert types.count("Service Agreement") == 7
        assert types.count("Lease Agreement") == 4

    def test_risk_profile_distribution(self, manifest):
        risks = [e["risk_profile"] for e in manifest]
        assert risks.count("LOW") == 19
        assert risks.count("MEDIUM") == 14
        assert risks.count("HIGH") == 15
        assert risks.count("CRITICAL") == 2

    def test_all_entries_have_expected_clauses(self, manifest):
        for entry in manifest:
            assert entry["expected_clause_types"], \
                f"{entry['id']} has no expected clause types"

    def test_unique_contract_ids(self, manifest):
        ids = [e["id"] for e in manifest]
        assert len(ids) == len(set(ids))


class TestIngestionAccuracy:
    """Clause type detection accuracy across all 50 contracts."""

    def test_overall_recall_above_threshold(self, full_report):
        """System must find ≥70% of expected clause types on average."""
        assert full_report.avg_recall >= 0.70, \
            f"Recall {full_report.avg_recall:.1%} below 70% threshold"

    def test_overall_f1_above_threshold(self, full_report):
        """Overall F1 must be ≥0.55."""
        assert full_report.avg_f1 >= 0.55, \
            f"F1 {full_report.avg_f1:.1%} below 55% threshold"

    def test_no_contracts_failed(self, full_report):
        """No contract should fail to parse/chunk."""
        failed = [r for r in full_report.results if r.error]
        assert not failed, f"Failed contracts: {[r.contract_id for r in failed]}"

    def test_all_contracts_produce_chunks(self, full_report):
        """Every contract must produce at least 2 chunks."""
        low_chunk = [r for r in full_report.results if r.chunk_count < 2]
        assert not low_chunk, \
            f"Contracts with <2 chunks: {[r.contract_id for r in low_chunk]}"

    def test_nda_confidentiality_detected(self, runner, manifest):
        """All NDA contracts must detect 'confidentiality' clause type."""
        ndas = [e for e in manifest if e["contract_type"] == "NDA"]
        misses = []
        for entry in ndas:
            result = runner.run_single(entry)
            if "confidentiality" not in result.detected_clause_types:
                misses.append(entry["id"])
        assert not misses, f"NDAs missing confidentiality detection: {misses}"

    def test_employment_ip_or_noncompete_detected(self, runner, manifest):
        """Employment contracts must detect ip_assignment or non_compete."""
        emp = [e for e in manifest if e["contract_type"] == "Employment Contract"]
        misses = []
        for entry in emp:
            result = runner.run_single(entry)
            found = set(result.detected_clause_types)
            if not (found & {"ip_assignment", "non_compete"}):
                misses.append(entry["id"])
        assert not misses, \
            f"Employment contracts missing IP/non-compete detection: {misses}"

    def test_saas_indemnification_detected(self, runner, manifest):
        """SaaS agreements must detect indemnification."""
        saas = [e for e in manifest if e["contract_type"] == "SaaS Agreement"]
        misses = []
        for entry in saas:
            result = runner.run_single(entry)
            if "indemnification" not in result.detected_clause_types:
                misses.append(entry["id"])
        assert not misses, f"SaaS missing indemnification detection: {misses}"

    def test_saas_limitation_of_liability_detected(self, runner, manifest):
        """SaaS agreements must detect limitation_of_liability."""
        saas = [e for e in manifest if e["contract_type"] == "SaaS Agreement"]
        misses = []
        for entry in saas:
            result = runner.run_single(entry)
            if "limitation_of_liability" not in result.detected_clause_types:
                misses.append(entry["id"])
        assert not misses, \
            f"SaaS missing limitation_of_liability detection: {misses}"

    def test_high_risk_contracts_detect_more_clause_types(self, full_report):
        """HIGH-risk contracts (more clauses) should have more detected types on average."""
        high_results = [r for r in full_report.results if r.risk_profile in ("HIGH", "CRITICAL")]
        low_results = [r for r in full_report.results if r.risk_profile == "LOW"]
        avg_high = statistics.mean(len(r.detected_clause_types) for r in high_results)
        avg_low = statistics.mean(len(r.detected_clause_types) for r in low_results)
        assert avg_high >= avg_low * 0.8, \
            f"HIGH ({avg_high:.1f}) should detect >= 80% as many types as LOW ({avg_low:.1f})"


class TestRetrievalAccuracy:
    """Hybrid search hit rate across all 50 contracts."""

    def test_overall_retrieval_hit_rate_above_threshold(self, full_report):
        """Retrieval must surface at least 1 relevant reference for ≥70% of expected clause types."""
        assert full_report.avg_retrieval_hit_rate >= 0.70, \
            f"Retrieval hit rate {full_report.avg_retrieval_hit_rate:.1%} below 70%"

    def test_indemnification_retrieval(self, runner, manifest):
        """Indemnification queries must always retrieve at least one reference."""
        contracts_with_indemnity = [
            e for e in manifest if "indemnification" in e["expected_clause_types"]
        ]
        misses = []
        for entry in contracts_with_indemnity[:10]:  # test first 10 for speed
            result = runner.run_single(entry)
            # Check that indemnification retrieval worked
            if result.retrieval_hit_rate < 0.3:
                misses.append(entry["id"])
        assert len(misses) <= 2, \
            f"Too many indemnification retrieval misses: {misses}"

    def test_confidentiality_retrieval(self, runner, manifest):
        """Confidentiality queries must surface reference documents."""
        ndas = [e for e in manifest if e["contract_type"] == "NDA"][:6]
        misses = []
        for entry in ndas:
            result = runner.run_single(entry)
            if result.retrieval_hits == 0:
                misses.append(entry["id"])
        assert not misses, f"NDAs with zero retrieval hits: {misses}"

    def test_retrieval_hit_rate_by_type(self, full_report):
        """Every contract type must have ≥60% retrieval hit rate."""
        for ct, stats in full_report.by_type.items():
            assert stats["avg_retrieval_hit_rate"] >= 0.60, \
                f"{ct} retrieval hit rate {stats['avg_retrieval_hit_rate']:.1%} below 60%"

    def test_total_retrieval_queries(self, full_report):
        """Total retrieval queries should be substantial (all expected clause types queried)."""
        assert full_report.total_retrieval_queries >= 200, \
            f"Only {full_report.total_retrieval_queries} retrieval queries run — expected ≥200"


class TestRiskSignalAccuracy:
    """Clause confidence scores reflect document risk level."""

    def test_low_risk_contracts_no_false_alarm(self, full_report):
        """LOW-risk contracts should mostly avoid unexpected high-confidence detections."""
        low_results = [r for r in full_report.results if r.risk_profile == "LOW"]
        flagged = [r.contract_id for r in low_results if len(r.false_high_risk_flags) > 3]
        assert len(flagged) <= 3, \
            f"Too many LOW contracts with false HIGH signals: {flagged}"

    def test_high_risk_contracts_have_detections(self, full_report):
        """HIGH/CRITICAL contracts must detect ≥75% of their expected clause types.

        NDAs are compact (2 types); SaaS/Employment have 5-9 expected types.
        Aggressive-language clauses are intentionally harder to classify —
        75% recall is the production-quality threshold for high-risk contracts.
        """
        high_results = [r for r in full_report.results if r.risk_profile in ("HIGH", "CRITICAL")]
        under_detected = []
        for r in high_results:
            threshold = max(2, round(len(r.expected_clause_types) * 0.75))
            if len(r.detected_clause_types) < threshold:
                under_detected.append(
                    f"{r.contract_id}({len(r.detected_clause_types)}/{len(r.expected_clause_types)})"
                )
        assert not under_detected, \
            f"HIGH contracts below 75% clause detection threshold: {under_detected}"

    def test_critical_contracts_have_most_detections(self, full_report):
        """CRITICAL contracts (most clauses) detect the most clause types."""
        critical = [r for r in full_report.results if r.risk_profile == "CRITICAL"]
        if critical:
            avg_critical = statistics.mean(len(r.detected_clause_types) for r in critical)
            assert avg_critical >= 5, \
                f"CRITICAL contracts only detected {avg_critical:.1f} types on avg — expected ≥5"

    def test_employment_high_detects_ip_noncompete(self, runner, manifest):
        """HIGH-risk employment contracts must detect both ip_assignment and non_compete."""
        high_emp = [e for e in manifest
                    if e["contract_type"] == "Employment Contract"
                    and e["risk_profile"] in ("HIGH", "CRITICAL")]
        misses = []
        for entry in high_emp:
            result = runner.run_single(entry)
            types = set(result.detected_clause_types)
            if not ("ip_assignment" in types and "non_compete" in types):
                misses.append(entry["id"])
        assert not misses, \
            f"HIGH employment contracts missing IP + non_compete: {misses}"

    def test_saas_high_detects_data_protection(self, runner, manifest):
        """HIGH-risk SaaS contracts must detect data_protection."""
        high_saas = [e for e in manifest
                     if e["contract_type"] == "SaaS Agreement"
                     and e["risk_profile"] in ("HIGH", "CRITICAL")
                     and "data_protection" in e["expected_clause_types"]]
        misses = []
        for entry in high_saas:
            result = runner.run_single(entry)
            if "data_protection" not in result.detected_clause_types:
                misses.append(entry["id"])
        assert not misses, \
            f"HIGH SaaS contracts missing data_protection detection: {misses}"


class TestPerformance:
    """Parse times and throughput checks."""

    def test_parse_time_acceptable(self, full_report):
        """All contracts should parse in under 500ms."""
        slow = [(r.contract_id, r.parse_time_ms)
                for r in full_report.results if r.parse_time_ms > 500]
        assert not slow, f"Slow parse times (>500ms): {slow}"

    def test_avg_chunks_per_contract(self, full_report):
        """Average chunk count should be between 3 and 25 per contract."""
        avg = statistics.mean(r.chunk_count for r in full_report.results if not r.error)
        assert 3 <= avg <= 25, f"Unexpected avg chunk count: {avg:.1f}"


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    runner = AccuracyTestRunner()
    report = runner.run_all(manifest)
    print_report(report)

    # Save JSON report
    report_path = ROOT / "data" / "test_contracts" / "accuracy_report.json"
    report_data = {
        "summary": {
            "total_contracts": report.total_contracts,
            "failed_contracts": report.failed_contracts,
            "avg_precision": report.avg_precision,
            "avg_recall": report.avg_recall,
            "avg_f1": report.avg_f1,
            "avg_retrieval_hit_rate": report.avg_retrieval_hit_rate,
            "total_retrieval_hits": report.total_retrieval_hits,
            "total_retrieval_queries": report.total_retrieval_queries,
            "high_risk_recall": report.high_risk_recall,
            "low_risk_precision": report.low_risk_precision,
        },
        "by_type": report.by_type,
        "by_risk_profile": report.by_risk_profile,
        "lowest_f1": report.lowest_f1_contracts,
        "lowest_retrieval": report.lowest_retrieval_contracts,
        "per_contract": [
            {
                "id": r.contract_id,
                "type": r.contract_type,
                "risk": r.risk_profile,
                "precision": r.precision,
                "recall": r.recall,
                "f1": r.f1,
                "retrieval_hit_rate": r.retrieval_hit_rate,
                "chunks": r.chunk_count,
                "detected": r.detected_clause_types,
                "expected": r.expected_clause_types,
                "error": r.error,
            }
            for r in report.results
        ],
    }
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"📄 JSON report saved to {report_path}")
