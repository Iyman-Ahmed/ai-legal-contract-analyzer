"""
RAG evaluation using custom metrics (RAGAS-inspired).

We implement lightweight versions of key metrics without requiring
the full RAGAS dependency (which has heavy torch requirements).
Metrics: context_precision, context_recall, answer_relevancy, faithfulness.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"


def _token_overlap(text1: str, text2: str) -> float:
    """Jaccard-like token overlap between two texts."""
    def tokens(t: str) -> set[str]:
        return set(re.findall(r"\b\w+\b", t.lower()))

    t1, t2 = tokens(text1), tokens(text2)
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def _keyword_recall(answer: str, keywords: list[str]) -> float:
    """What fraction of expected keywords appear in the answer."""
    if not keywords:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return hits / len(keywords)


class RAGEvaluator:
    """
    Evaluates the RAG pipeline using a golden test set.
    Computes lightweight versions of standard RAG metrics.
    """

    def __init__(self, hybrid_search=None, reranker=None, risk_engine=None):
        """
        Args:
            hybrid_search: HybridSearchEngine instance.
            reranker: CrossEncoderReranker instance.
            risk_engine: RiskAnalysisEngine instance (for answer generation).
        """
        self.hybrid_search = hybrid_search
        self.reranker = reranker
        self.risk_engine = risk_engine
        self.golden_set = self._load_golden_set()

    def run_evaluation(
        self, subset_size: Optional[int] = None
    ) -> dict:
        """
        Run evaluation against the golden test set.

        Args:
            subset_size: If set, only evaluate first N examples.

        Returns:
            Dict with per-metric scores and per-question results.
        """
        examples = self.golden_set
        if subset_size:
            examples = examples[:subset_size]

        if not examples:
            return {"error": "No golden set examples found"}

        results: list[dict] = []
        for example in examples:
            result = self._evaluate_single(example)
            results.append(result)

        # Aggregate metrics
        context_precisions = [r["context_precision"] for r in results]
        context_recalls = [r["context_recall"] for r in results]
        answer_relevancies = [r["answer_relevancy"] for r in results]

        avg = lambda lst: round(sum(lst) / len(lst), 3) if lst else 0.0

        return {
            "num_examples": len(results),
            "metrics": {
                "context_precision": avg(context_precisions),
                "context_recall": avg(context_recalls),
                "answer_relevancy": avg(answer_relevancies),
                "overall": avg(context_precisions + context_recalls + answer_relevancies),
            },
            "per_question": results,
        }

    def _evaluate_single(self, example: dict) -> dict:
        """Evaluate a single golden set example."""
        question = example["question"]
        expected_keywords = example.get("expected_answer_keywords", [])
        ground_truth = example.get("ground_truth", "")
        clause_type = example.get("clause_type")

        # Retrieve context
        retrieved = []
        if self.hybrid_search:
            retrieved = self.hybrid_search.search_reference(
                query=question,
                top_k=5,
                clause_type_filter=clause_type if clause_type != "general" else None,
            )
            if self.reranker and retrieved:
                retrieved = self.reranker.rerank(query=question, candidates=retrieved, top_n=3)

        context_texts = [r["text"] for r in retrieved]
        combined_context = " ".join(context_texts)

        # Context precision: how relevant are retrieved docs to the question?
        context_precision = (
            sum(_token_overlap(question, ctx) for ctx in context_texts) / len(context_texts)
            if context_texts else 0.0
        )

        # Context recall: does retrieved context contain expected keywords?
        context_recall = _keyword_recall(combined_context, expected_keywords)

        # Answer relevancy: does the context help answer the question?
        answer_relevancy = _token_overlap(combined_context, ground_truth) if ground_truth else context_recall

        return {
            "id": example.get("id", "unknown"),
            "question": question,
            "context_precision": round(context_precision, 3),
            "context_recall": round(context_recall, 3),
            "answer_relevancy": round(answer_relevancy, 3),
            "retrieved_count": len(retrieved),
            "top_sources": [
                r.get("metadata", {}).get("source_filename", "?")
                for r in retrieved[:2]
            ],
        }

    def _load_golden_set(self) -> list[dict]:
        """Load golden test set from JSON."""
        if GOLDEN_SET_PATH.exists():
            with open(GOLDEN_SET_PATH) as f:
                return json.load(f)
        logger.warning(f"Golden set not found at {GOLDEN_SET_PATH}")
        return []

    def format_results_for_display(self, eval_results: dict) -> str:
        """Format evaluation results as a Markdown table for Gradio display."""
        if "error" in eval_results:
            return f"**Error:** {eval_results['error']}"

        metrics = eval_results.get("metrics", {})
        lines = [
            "## RAG Evaluation Results",
            "",
            f"**Examples evaluated:** {eval_results.get('num_examples', 0)}",
            "",
            "### Overall Metrics",
            "",
            "| Metric | Score | Description |",
            "|--------|-------|-------------|",
            f"| Context Precision | {metrics.get('context_precision', 0):.3f} | Relevance of retrieved context to query |",
            f"| Context Recall | {metrics.get('context_recall', 0):.3f} | Coverage of expected information in context |",
            f"| Answer Relevancy | {metrics.get('answer_relevancy', 0):.3f} | Alignment between context and ground truth |",
            f"| **Overall** | **{metrics.get('overall', 0):.3f}** | Average across all metrics |",
            "",
            "### Per-Question Results",
            "",
            "| # | Question | Precision | Recall | Relevancy |",
            "|---|----------|-----------|--------|-----------|",
        ]

        for i, r in enumerate(eval_results.get("per_question", []), 1):
            q_short = r["question"][:60] + "..." if len(r["question"]) > 60 else r["question"]
            lines.append(
                f"| {i} | {q_short} | {r['context_precision']:.3f} | "
                f"{r['context_recall']:.3f} | {r['answer_relevancy']:.3f} |"
            )

        return "\n".join(lines)
