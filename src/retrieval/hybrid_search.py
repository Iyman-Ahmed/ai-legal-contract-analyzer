"""
Hybrid search via Reciprocal Rank Fusion (RRF).

RRF combines dense (ChromaDB) and sparse (BM25) results without needing
to tune interpolation weights. Formula: sum(1 / (k + rank_i)) for each
result across all ranked lists.

Reference: Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet
and individual Rank Learning Methods" (SIGIR 2009).
"""

import logging
from collections import defaultdict
from typing import Optional

from config.settings import RRF_K, DENSE_TOP_K, SPARSE_TOP_K, FINAL_TOP_K
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_search import BM25SearchEngine

logger = logging.getLogger(__name__)


class HybridSearchEngine:
    """
    Fuses dense and sparse search results using RRF.
    Optionally filters by clause_type for like-for-like comparison.
    """

    def __init__(self, vector_store: VectorStore, bm25_engine: BM25SearchEngine):
        self.vector_store = vector_store
        self.bm25_engine = bm25_engine

    def search_reference(
        self,
        query: str,
        top_k: int = FINAL_TOP_K,
        clause_type_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Hybrid search against the reference corpus.

        Args:
            query: Search query (typically a clause from the uploaded contract).
            top_k: Number of final results to return after fusion.
            clause_type_filter: Optional clause type to narrow search.

        Returns:
            Merged, RRF-ranked list of result dicts.
        """
        dense_results = self.vector_store.query_reference(
            query=query,
            top_k=DENSE_TOP_K,
            clause_type_filter=clause_type_filter,
        )
        sparse_results = self.bm25_engine.search_reference(
            query=query,
            top_k=SPARSE_TOP_K,
        )

        return self._rrf_fuse(dense_results, sparse_results, top_k)

    def search_contract(
        self,
        query: str,
        top_k: int = FINAL_TOP_K,
        clause_type_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Hybrid search against the uploaded contract (for Q&A chat mode).
        """
        dense_results = self.vector_store.query_contract(
            query=query,
            top_k=DENSE_TOP_K,
            clause_type_filter=clause_type_filter,
        )
        sparse_results = self.bm25_engine.search_contract(
            query=query,
            top_k=SPARSE_TOP_K,
        )

        return self._rrf_fuse(dense_results, sparse_results, top_k)

    def _rrf_fuse(
        self,
        dense_results: list[dict],
        sparse_results: list[dict],
        top_k: int,
        k: int = RRF_K,
    ) -> list[dict]:
        """
        Reciprocal Rank Fusion.

        Score for each doc = sum over all result lists of 1 / (k + rank_in_list).
        """
        # Use a text fingerprint as the dedup key
        doc_scores: dict[str, float] = defaultdict(float)
        doc_data: dict[str, dict] = {}

        def _key(result: dict) -> str:
            """Deduplicate on first 120 chars of text."""
            return result["text"][:120].strip()

        for rank, result in enumerate(dense_results, start=1):
            key = _key(result)
            doc_scores[key] += 1.0 / (k + rank)
            if key not in doc_data:
                doc_data[key] = result

        for rank, result in enumerate(sparse_results, start=1):
            key = _key(result)
            doc_scores[key] += 1.0 / (k + rank)
            if key not in doc_data:
                doc_data[key] = result

        # Sort by RRF score descending
        sorted_keys = sorted(doc_scores, key=lambda k: doc_scores[k], reverse=True)

        fused: list[dict] = []
        for key in sorted_keys[:top_k]:
            result = dict(doc_data[key])
            result["rrf_score"] = round(doc_scores[key], 6)
            result["source"] = "hybrid"
            fused.append(result)

        logger.debug(
            f"RRF fusion: {len(dense_results)} dense + {len(sparse_results)} sparse → {len(fused)} merged"
        )
        return fused
