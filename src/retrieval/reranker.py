"""
Cross-encoder reranker.

Takes top-k fused results from hybrid search and re-scores each
(query, document) pair with a cross-encoder model, which reads both
together (vs. separate embeddings) for much higher precision.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
- Fast on CPU (MiniLM architecture)
- Strong performance on passage retrieval tasks
- ~85 MB download
"""

import logging
from typing import Optional

from config.settings import RERANKER_MODEL, RERANK_TOP_K, FINAL_TOP_K

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranks a list of candidate documents using a cross-encoder model.
    """

    def __init__(self, model_name: str = RERANKER_MODEL):
        logger.info(f"Loading reranker: {model_name}")
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, max_length=512)
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_n: int = FINAL_TOP_K,
    ) -> list[dict]:
        """
        Rerank candidate documents for a given query.

        Args:
            query: The search query.
            candidates: List of candidate dicts (must have 'text' key).
            top_n: Number of results to return after reranking.

        Returns:
            Top-n results sorted by cross-encoder score descending.
        """
        if not candidates:
            return []

        # Truncate document text for cross-encoder (512 token limit)
        pairs = [(query, doc["text"][:1200]) for doc in candidates]

        scores = self.model.predict(pairs, show_progress_bar=False)

        for doc, score in zip(candidates, scores):
            doc["rerank_score"] = float(score)

        reranked = sorted(candidates, key=lambda d: d["rerank_score"], reverse=True)

        logger.debug(
            f"Reranked {len(candidates)} → top {top_n}: "
            f"scores {[round(d['rerank_score'], 3) for d in reranked[:top_n]]}"
        )

        return reranked[:top_n]
