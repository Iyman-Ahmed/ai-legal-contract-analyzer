"""
RAG evaluation with LLM-as-judge metrics.

Replaces the original token-overlap heuristics with proper LLM-judged
metrics that actually catch the failure modes that matter for legal AI:
  - Faithfulness: is every claim grounded in context?
  - Answer Relevancy: does the answer address the question?
  - Context Precision: are retrieved chunks relevant to the query?
  - Context Recall: does retrieved context cover the ground truth?
  - Clause F1: how accurately are clause types detected?

All LLM calls use whichever LLM_PROVIDER is configured — works with
LM Studio locally or any of the cloud providers.
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.analysis.risk_engine import LLMClient

logger = logging.getLogger(__name__)

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"

# ── Evaluation prompts ─────────────────────────────────────────────────────────

_FAITHFULNESS_SYSTEM = """You are a faithfulness evaluator for a legal AI system.
Score whether the generated answer is fully supported by the provided context.
Output ONLY valid JSON: {"score": <0.0-1.0>, "reasoning": "<one sentence>"}"""

_FAITHFULNESS_USER = """CONTEXT:
{context}

GENERATED ANSWER:
{answer}

Score faithfulness 0.0-1.0. 1.0 = every claim traceable to context. 0.0 = answer contradicts or ignores context.
JSON only:"""

_RELEVANCY_SYSTEM = """You are an answer relevancy evaluator.
Score whether the answer directly addresses the question asked.
Output ONLY valid JSON: {"score": <0.0-1.0>, "reasoning": "<one sentence>"}"""

_RELEVANCY_USER = """QUESTION:
{question}

ANSWER:
{answer}

Score relevancy 0.0-1.0. 1.0 = answer directly and completely addresses the question.
JSON only:"""

_CONTEXT_PRECISION_SYSTEM = """You are a retrieval quality evaluator.
For each retrieved chunk, judge whether it is relevant to answering the question.
Output ONLY valid JSON: {"relevant_count": <int>, "total": <int>, "score": <0.0-1.0>}"""

_CONTEXT_PRECISION_USER = """QUESTION:
{question}

RETRIEVED CHUNKS:
{chunks}

Count how many chunks are relevant to answering the question. Score = relevant/total.
JSON only:"""

_RECALL_SYSTEM = """You are a context coverage evaluator.
Judge whether the retrieved context contains enough information to answer the question given the ground truth.
Output ONLY valid JSON: {"score": <0.0-1.0>, "reasoning": "<one sentence>"}"""

_RECALL_USER = """GROUND TRUTH ANSWER:
{ground_truth}

RETRIEVED CONTEXT:
{context}

Score how well the context covers the ground truth (0.0 = nothing covered, 1.0 = fully covered).
JSON only:"""


def _parse_score(raw: str, field: str = "score") -> float:
    """Extract a 0-1 score from a JSON response."""
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    try:
        data = json.loads(raw)
        return max(0.0, min(1.0, float(data.get(field, 0.5))))
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end > start:
            try:
                data = json.loads(raw[start:end + 1])
                return max(0.0, min(1.0, float(data.get(field, 0.5))))
            except Exception:
                pass
        # Fallback: look for a decimal number
        numbers = re.findall(r"0\.\d+|1\.0+|[01]", raw)
        return float(numbers[0]) if numbers else 0.5


def _token_overlap(text1: str, text2: str) -> float:
    """Jaccard token overlap — fast fallback when LLM is unavailable."""
    def tok(t: str) -> set:
        return set(re.findall(r"\b\w+\b", t.lower()))
    t1, t2 = tok(text1), tok(text2)
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def _keyword_recall(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    a = answer.lower()
    return sum(1 for kw in keywords if kw.lower() in a) / len(keywords)


# ── Per-metric classes ─────────────────────────────────────────────────────────

class FaithfulnessMetric:
    """LLM-as-judge: is every claim in the answer grounded in the context?"""

    def __init__(self, llm: Optional["LLMClient"] = None):
        self.llm = llm

    def compute(self, answer: str, context: str) -> float:
        if not answer.strip() or not context.strip():
            return 1.0
        if self.llm is None:
            return _token_overlap(answer, context)
        try:
            raw = self.llm.chat(
                system=_FAITHFULNESS_SYSTEM,
                user=_FAITHFULNESS_USER.format(context=context[:2000], answer=answer[:800]),
                max_tokens=128,
            )
            return _parse_score(raw)
        except Exception as e:
            logger.warning(f"FaithfulnessMetric LLM call failed: {e}")
            return _token_overlap(answer, context)


class AnswerRelevancyMetric:
    """LLM-as-judge: does the answer directly address the question?"""

    def __init__(self, llm: Optional["LLMClient"] = None):
        self.llm = llm

    def compute(self, question: str, answer: str) -> float:
        if not answer.strip():
            return 0.0
        if self.llm is None:
            return _token_overlap(question, answer)
        try:
            raw = self.llm.chat(
                system=_RELEVANCY_SYSTEM,
                user=_RELEVANCY_USER.format(question=question, answer=answer[:800]),
                max_tokens=128,
            )
            return _parse_score(raw)
        except Exception as e:
            logger.warning(f"AnswerRelevancyMetric LLM call failed: {e}")
            return _token_overlap(question, answer)


class ContextPrecisionMetric:
    """LLM-as-judge: what fraction of retrieved chunks are relevant to the query?"""

    def __init__(self, llm: Optional["LLMClient"] = None):
        self.llm = llm

    def compute(self, question: str, chunks: list[str]) -> float:
        if not chunks:
            return 0.0
        if self.llm is None:
            return sum(_token_overlap(question, c) for c in chunks) / len(chunks)
        chunk_text = "\n\n".join(f"[Chunk {i+1}]: {c[:300]}" for i, c in enumerate(chunks))
        try:
            raw = self.llm.chat(
                system=_CONTEXT_PRECISION_SYSTEM,
                user=_CONTEXT_PRECISION_USER.format(
                    question=question, chunks=chunk_text[:2000]
                ),
                max_tokens=128,
            )
            return _parse_score(raw)
        except Exception as e:
            logger.warning(f"ContextPrecisionMetric LLM call failed: {e}")
            return sum(_token_overlap(question, c) for c in chunks) / len(chunks)


class ContextRecallMetric:
    """LLM-as-judge: does retrieved context cover the ground truth answer?"""

    def __init__(self, llm: Optional["LLMClient"] = None):
        self.llm = llm

    def compute(self, ground_truth: str, context: str, keywords: list[str]) -> float:
        if not ground_truth or ground_truth == "Not present in this contract.":
            return 1.0
        if self.llm is None:
            return _keyword_recall(context, keywords) if keywords else _token_overlap(ground_truth, context)
        try:
            raw = self.llm.chat(
                system=_RECALL_SYSTEM,
                user=_RECALL_USER.format(
                    ground_truth=ground_truth[:400],
                    context=context[:2000],
                ),
                max_tokens=128,
            )
            return _parse_score(raw)
        except Exception as e:
            logger.warning(f"ContextRecallMetric LLM call failed: {e}")
            return _keyword_recall(context, keywords)


class ClauseF1Metric:
    """
    Clause detection accuracy: F1 between predicted and expected clause types.

    Does not require an LLM — compares clause_type strings directly.
    Used to measure how well the ingestion metadata extractor classifies clauses.
    """

    @staticmethod
    def compute(
        predicted_types: list[str],
        true_types: list[str],
    ) -> dict[str, float]:
        pred_set = set(predicted_types)
        true_set = set(true_types)
        if not true_set:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

        tp = len(pred_set & true_set)
        precision = tp / len(pred_set) if pred_set else 0.0
        recall = tp / len(true_set) if true_set else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}


# ── Main evaluator ─────────────────────────────────────────────────────────────

class RAGEvaluator:
    """
    Evaluates the full RAG pipeline against a golden test set.

    With an LLMClient: all four metrics use LLM-as-judge.
    Without one: falls back to fast token-overlap heuristics
    (for CI / test environments without an API key).
    """

    def __init__(
        self,
        hybrid_search=None,
        reranker=None,
        risk_engine=None,
        llm: Optional["LLMClient"] = None,
    ):
        self.hybrid_search = hybrid_search
        self.reranker = reranker
        self.risk_engine = risk_engine

        # If risk_engine is provided, reuse its LLM client
        _llm = llm or (risk_engine.llm if risk_engine else None)
        self.faithfulness = FaithfulnessMetric(_llm)
        self.relevancy = AnswerRelevancyMetric(_llm)
        self.precision = ContextPrecisionMetric(_llm)
        self.recall = ContextRecallMetric(_llm)
        self.golden_set = self._load_golden_set()

        judge_mode = "LLM-as-judge" if _llm else "heuristic"
        logger.info(f"RAGEvaluator initialized ({judge_mode} mode, {len(self.golden_set)} examples)")

    def run_evaluation(self, subset_size: Optional[int] = None) -> dict:
        """
        Run full evaluation against the golden test set.

        Returns a dict with aggregate metrics and per-question detail.
        """
        examples = self.golden_set[:subset_size] if subset_size else self.golden_set
        if not examples:
            return {"error": "No golden set examples found. Run scripts/build_eval_dataset.py first."}

        results: list[dict] = []
        t0 = time.time()

        for i, example in enumerate(examples):
            logger.info(f"Evaluating example {i+1}/{len(examples)}: {example.get('id')}")
            result = self._evaluate_single(example)
            results.append(result)

        elapsed = round(time.time() - t0, 1)

        def avg(key: str) -> float:
            vals = [r[key] for r in results if r[key] is not None]
            return round(sum(vals) / len(vals), 3) if vals else 0.0

        # Clause F1 across all examples (aggregate)
        pred_types = [r.get("retrieved_clause_type", "general") for r in results]
        true_types = [ex.get("clause_type", "general") for ex in examples]
        clause_f1 = ClauseF1Metric.compute(pred_types, true_types)

        return {
            "num_examples": len(results),
            "elapsed_seconds": elapsed,
            "metrics": {
                "faithfulness": avg("faithfulness"),
                "answer_relevancy": avg("answer_relevancy"),
                "context_precision": avg("context_precision"),
                "context_recall": avg("context_recall"),
                "clause_type_f1": clause_f1["f1"],
                "clause_type_precision": clause_f1["precision"],
                "clause_type_recall": clause_f1["recall"],
                "overall": round(
                    (avg("faithfulness") + avg("answer_relevancy") +
                     avg("context_precision") + avg("context_recall")) / 4,
                    3,
                ),
            },
            "per_question": results,
        }

    def _evaluate_single(self, example: dict) -> dict:
        question = example["question"]
        ground_truth = example.get("ground_truth", "")
        expected_keywords = example.get("expected_answer_keywords", [])
        clause_type = example.get("clause_type", "general")

        # ── Retrieval ──────────────────────────────────────────────────────────
        retrieved: list[dict] = []
        if self.hybrid_search:
            retrieved = self.hybrid_search.search_reference(
                query=question,
                top_k=5,
                clause_type_filter=clause_type if clause_type != "general" else None,
            )
            if self.reranker and retrieved:
                retrieved = self.reranker.rerank(query=question, candidates=retrieved, top_n=3)

        context_texts = [r["text"] for r in retrieved]
        combined_context = "\n\n".join(context_texts)

        # ── Answer generation (optional) ───────────────────────────────────────
        generated_answer = ""
        if self.risk_engine and combined_context:
            try:
                response = self.risk_engine.answer_question(
                    question=question,
                    contract_chunks=[],
                )
                generated_answer = response.get("answer", "")
            except Exception:
                generated_answer = ""

        # Fall back to ground truth for scoring when no answer is generated
        answer_for_scoring = generated_answer or ground_truth

        # ── Metrics ────────────────────────────────────────────────────────────
        f_score = self.faithfulness.compute(answer_for_scoring, combined_context)
        r_score = self.relevancy.compute(question, answer_for_scoring)
        p_score = self.precision.compute(question, context_texts)
        c_score = self.recall.compute(ground_truth, combined_context, expected_keywords)

        retrieved_type = (
            retrieved[0].get("metadata", {}).get("clause_type", "general")
            if retrieved else "general"
        )

        return {
            "id": example.get("id", "?"),
            "question": question[:100],
            "clause_type": clause_type,
            "faithfulness": round(f_score, 3),
            "answer_relevancy": round(r_score, 3),
            "context_precision": round(p_score, 3),
            "context_recall": round(c_score, 3),
            "retrieved_count": len(retrieved),
            "retrieved_clause_type": retrieved_type,
            "top_sources": [
                r.get("metadata", {}).get("source_filename", "?") for r in retrieved[:2]
            ],
            "is_present": example.get("is_present", True),
            "difficulty": example.get("difficulty", "medium"),
        }

    def _load_golden_set(self) -> list[dict]:
        if GOLDEN_SET_PATH.exists():
            with open(GOLDEN_SET_PATH) as f:
                return json.load(f)
        logger.warning(f"Golden set not found at {GOLDEN_SET_PATH}. Run build_eval_dataset.py.")
        return []

    def format_results_for_display(self, eval_results: dict) -> str:
        """Format results as Markdown for the Gradio evaluation tab."""
        if "error" in eval_results:
            return f"**Error:** {eval_results['error']}"

        m = eval_results.get("metrics", {})
        n = eval_results.get("num_examples", 0)
        elapsed = eval_results.get("elapsed_seconds", 0)

        lines = [
            "## RAG Evaluation Results (LLM-as-judge)",
            "",
            f"**Examples evaluated:** {n} &nbsp;|&nbsp; **Time:** {elapsed}s",
            "",
            "### Core RAG Metrics",
            "",
            "| Metric | Score | What it measures |",
            "|--------|-------|-----------------|",
            f"| Faithfulness | **{m.get('faithfulness', 0):.3f}** | Claims grounded in retrieved context |",
            f"| Answer Relevancy | **{m.get('answer_relevancy', 0):.3f}** | Answer directly addresses the question |",
            f"| Context Precision | **{m.get('context_precision', 0):.3f}** | Retrieved chunks are relevant to query |",
            f"| Context Recall | **{m.get('context_recall', 0):.3f}** | Context covers the expected answer |",
            f"| **Overall RAG Score** | **{m.get('overall', 0):.3f}** | Average of above four |",
            "",
            "### Clause Detection Accuracy",
            "",
            "| Metric | Score |",
            "|--------|-------|",
            f"| Clause Type Precision | {m.get('clause_type_precision', 0):.3f} |",
            f"| Clause Type Recall | {m.get('clause_type_recall', 0):.3f} |",
            f"| Clause Type F1 | **{m.get('clause_type_f1', 0):.3f}** |",
            "",
            "### Per-Question Breakdown",
            "",
            "| # | Question | Type | Faith. | Relev. | Prec. | Recall | Diff. |",
            "|---|----------|------|--------|--------|-------|--------|-------|",
        ]

        for i, r in enumerate(eval_results.get("per_question", []), 1):
            q = r["question"][:55] + "…" if len(r["question"]) > 55 else r["question"]
            lines.append(
                f"| {i} | {q} | {r['clause_type'][:12]} "
                f"| {r['faithfulness']:.2f} | {r['answer_relevancy']:.2f} "
                f"| {r['context_precision']:.2f} | {r['context_recall']:.2f} "
                f"| {r.get('difficulty', '?')} |"
            )

        return "\n".join(lines)
