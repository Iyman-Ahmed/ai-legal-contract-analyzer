"""
Standalone RAGAS evaluation runner.

Indexes all three sample contracts into the uploaded_contract collection,
then runs the 15-question golden set through the full retrieval + scoring
pipeline using the configured LLM provider (or token-overlap heuristics if
no API key is available).

Usage:
    cd contract-analyzer/.claude/worktrees/trust-redesign
    PYTHONPATH=. python scripts/run_ragas_eval.py [--heuristic]

    --heuristic   Skip LLM judge; use token-overlap fallback (no API key needed)
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

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
from src.retrieval.knowledge_base import KnowledgeBaseBuilder
from src.evaluation.ragas_eval import (
    RAGEvaluator,
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextPrecisionMetric,
    ContextRecallMetric,
    ClauseF1Metric,
    _keyword_recall,
    _token_overlap,
)

GOLDEN_SET_PATH = ROOT / "data" / "evaluation" / "golden_set.json"
SAMPLE_CONTRACTS = {
    "sample_nda.txt":           ROOT / "data" / "sample_contracts" / "sample_nda.txt",
    "sample_saas_contract.txt": ROOT / "data" / "sample_contracts" / "sample_saas_agreement.txt",
    "sample_employment.txt":    ROOT / "data" / "sample_contracts" / "sample_employment_contract.txt",
}


def build_retrieval_stack():
    embed  = EmbeddingPipeline()
    store  = VectorStore(embed)
    bm25   = BM25SearchEngine()
    rerank = CrossEncoderReranker()

    print("Building reference knowledge base …", flush=True)
    kb = KnowledgeBaseBuilder(store, bm25)
    if not store.is_reference_indexed():
        kb.build()

    return embed, store, bm25, rerank


def index_contract(path: Path, embed, store, bm25):
    parser  = DocumentParser()
    chunker = SectionAwareChunker()
    meta    = MetadataExtractor()

    parsed   = parser.parse(str(path))
    chunks   = chunker.chunk(parsed)
    enriched = [meta.enrich(c) for c in chunks]
    dicts    = [e.to_dict() for e in enriched]

    store.add_contract_chunks(dicts)
    bm25.build_contract_index(dicts)
    return enriched


def run(use_heuristic: bool):
    print(f"\n{'='*60}")
    print("  RAGAS Evaluation — Legal Contract Analyzer")
    print(f"  Mode: {'token-overlap heuristic' if use_heuristic else 'LLM-as-judge'}")
    print(f"  Golden set: {GOLDEN_SET_PATH}")
    print(f"{'='*60}\n")

    with open(GOLDEN_SET_PATH) as f:
        golden = json.load(f)

    embed, store, bm25, rerank = build_retrieval_stack()
    hybrid = HybridSearchEngine(store, bm25)

    llm = None
    if not use_heuristic:
        try:
            from src.analysis.risk_engine import LLMClient
            llm = LLMClient()
            print(f"LLM provider: {llm.provider} / {llm.model}\n")
        except Exception as e:
            print(f"⚠  Could not init LLM ({e}) — falling back to heuristic mode.\n")

    faith_m  = FaithfulnessMetric(llm)
    relev_m  = AnswerRelevancyMetric(llm)
    prec_m   = ContextPrecisionMetric(llm)
    recall_m = ContextRecallMetric(llm)

    results = []
    t0 = time.time()

    for i, ex in enumerate(golden, 1):
        qid          = ex["id"]
        question     = ex["question"]
        ground_truth = ex.get("ground_truth", "")
        keywords     = ex.get("expected_answer_keywords", [])
        clause_type  = ex.get("clause_type", "general")
        contract_id  = ex.get("contract_id", "")
        is_present   = ex.get("is_present", True)
        difficulty   = ex.get("difficulty", "easy")

        # Index the relevant sample contract for this question
        contract_path = None
        for key, path in SAMPLE_CONTRACTS.items():
            if key in contract_id or contract_id.replace("_contract", "") in key:
                contract_path = path
                break
        if contract_path is None:
            # best-effort: try all filenames
            for key, path in SAMPLE_CONTRACTS.items():
                if any(w in contract_id for w in ["nda", "saas", "employment"]):
                    if any(w in key for w in ["nda", "saas", "employment"]):
                        contract_path = path
                        break

        if contract_path and contract_path.exists():
            index_contract(contract_path, embed, store, bm25)
        else:
            print(f"  [{qid}] Contract '{contract_id}' not found — skipping retrieval")

        # Retrieve from the indexed contract
        retrieved = hybrid.search_contract(
            query=question,
            top_k=5,
            clause_type_filter=clause_type if clause_type != "general" else None,
        )
        if retrieved:
            retrieved = rerank.rerank(query=question, candidates=retrieved, top_n=3)

        chunks_text   = [r["text"] for r in retrieved]
        combined_ctx  = "\n\n".join(chunks_text)

        f_score = faith_m.compute(ground_truth, combined_ctx)
        r_score = relev_m.compute(question, ground_truth)
        p_score = prec_m.compute(question, chunks_text)
        c_score = recall_m.compute(ground_truth, combined_ctx, keywords)

        retrieved_type = (
            retrieved[0].get("metadata", {}).get("clause_type", "general")
            if retrieved else "none"
        )

        row = {
            "id": qid,
            "question": question,
            "clause_type": clause_type,
            "is_present": is_present,
            "difficulty": difficulty,
            "retrieved_count": len(retrieved),
            "retrieved_clause_type": retrieved_type,
            "faithfulness": round(f_score, 3),
            "answer_relevancy": round(r_score, 3),
            "context_precision": round(p_score, 3),
            "context_recall": round(c_score, 3),
        }
        results.append(row)

        status = "✓" if c_score >= 0.5 else "✗"
        print(
            f"  [{i:02d}/{len(golden)}] {status} {qid:<10} "
            f"faith={f_score:.2f} relev={r_score:.2f} "
            f"prec={p_score:.2f} recall={c_score:.2f}  "
            f"[{difficulty}] retrieved={len(retrieved)}"
        )

    elapsed = round(time.time() - t0, 1)

    def avg(key):
        vals = [r[key] for r in results if r[key] is not None]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    pred_types = [r["retrieved_clause_type"] for r in results]
    true_types = [ex.get("clause_type", "general") for ex in golden]
    clause_f1  = ClauseF1Metric.compute(pred_types, true_types)

    overall = round(
        (avg("faithfulness") + avg("answer_relevancy") +
         avg("context_precision") + avg("context_recall")) / 4, 3
    )

    by_difficulty = {}
    for diff in ("easy", "medium", "hard"):
        subset = [r for r in results if r["difficulty"] == diff]
        if subset:
            by_difficulty[diff] = {
                "n": len(subset),
                "faithfulness":     round(sum(r["faithfulness"]     for r in subset) / len(subset), 3),
                "answer_relevancy": round(sum(r["answer_relevancy"] for r in subset) / len(subset), 3),
                "context_precision":round(sum(r["context_precision"]for r in subset) / len(subset), 3),
                "context_recall":   round(sum(r["context_recall"]   for r in subset) / len(subset), 3),
            }

    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Examples : {len(results)}  |  Time: {elapsed}s")
    print(f"  Mode     : {'heuristic' if llm is None else f'LLM ({llm.provider})'}")
    print()
    print(f"  {'Metric':<22}  {'Score':>6}")
    print(f"  {'-'*30}")
    print(f"  {'Faithfulness':<22}  {avg('faithfulness'):>6.3f}")
    print(f"  {'Answer Relevancy':<22}  {avg('answer_relevancy'):>6.3f}")
    print(f"  {'Context Precision':<22}  {avg('context_precision'):>6.3f}")
    print(f"  {'Context Recall':<22}  {avg('context_recall'):>6.3f}")
    print(f"  {'─'*30}")
    print(f"  {'OVERALL RAG SCORE':<22}  {overall:>6.3f}")
    print()
    print(f"  {'Clause Type F1':<22}  {clause_f1['f1']:>6.3f}")
    print(f"  {'  Precision':<22}  {clause_f1['precision']:>6.3f}")
    print(f"  {'  Recall':<22}  {clause_f1['recall']:>6.3f}")
    print()
    print("  By difficulty:")
    for diff, m in by_difficulty.items():
        print(
            f"    {diff:<8} (n={m['n']})  "
            f"faith={m['faithfulness']:.2f}  "
            f"recall={m['context_recall']:.2f}  "
            f"overall={round((m['faithfulness']+m['answer_relevancy']+m['context_precision']+m['context_recall'])/4,3):.3f}"
        )
    print(f"{'='*60}\n")

    # Write results JSON
    out_path = ROOT / "data" / "evaluation" / "ragas_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "mode": "heuristic" if llm is None else f"llm/{llm.provider}",
            "elapsed_seconds": elapsed,
            "metrics": {
                "faithfulness": avg("faithfulness"),
                "answer_relevancy": avg("answer_relevancy"),
                "context_precision": avg("context_precision"),
                "context_recall": avg("context_recall"),
                "overall": overall,
                "clause_type_f1": clause_f1["f1"],
                "clause_type_precision": clause_f1["precision"],
                "clause_type_recall": clause_f1["recall"],
            },
            "by_difficulty": by_difficulty,
            "per_question": results,
        }, f, indent=2)
    print(f"  Results saved → {out_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--heuristic", action="store_true",
                        help="Use token-overlap fallback instead of LLM judge")
    args = parser.parse_args()
    run(use_heuristic=args.heuristic)
