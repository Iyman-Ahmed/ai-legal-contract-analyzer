"""
BM25 sparse search for legal contract retrieval.

Uses rank_bm25 (Okapi BM25) — lightweight, no external service required.
Maintains separate indices for reference corpus and uploaded contract.
"""

import logging
import re
from typing import Optional

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# Simple legal-aware stopwords (supplements BM25's own TF-IDF weighting)
_STOPWORDS = frozenset({
    "the", "a", "an", "in", "of", "to", "and", "or", "for", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "shall",
    "this", "that", "these", "those", "it", "its", "with", "by", "from",
    "at", "on", "as", "if", "but", "not", "no", "any", "all", "such",
})


def _tokenize(text: str) -> list[str]:
    """
    Tokenize text for BM25. Lowercases, splits on non-alphanumeric,
    removes stopwords. Preserves legal compound terms (e.g., "non-compete").
    """
    text = text.lower()
    # Keep hyphens within words (non-compete → non-compete)
    tokens = re.findall(r"\b[\w][\w\-\']*\b", text)
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


class BM25Index:
    """
    Wraps rank_bm25 with document storage for result retrieval.
    """

    def __init__(self):
        self._docs: list[dict] = []
        self._bm25: Optional[BM25Okapi] = None

    def build(self, documents: list[dict]) -> None:
        """
        Build a BM25 index from a list of document dicts.

        Args:
            documents: List of dicts with at minimum a 'text' key.
        """
        if not documents:
            logger.warning("BM25 index built with 0 documents")
            return

        self._docs = documents
        tokenized = [_tokenize(doc["text"]) for doc in documents]
        self._bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25 index built with {len(documents)} documents")

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Search the BM25 index.

        Args:
            query: Raw query string (will be tokenized).
            top_k: Number of top results to return.

        Returns:
            List of result dicts with text, metadata, and BM25 score.
        """
        if self._bm25 is None or not self._docs:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        # Get top-k indices by score
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results: list[dict] = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            doc = self._docs[idx]
            results.append({
                "text": doc["text"],
                "metadata": {k: v for k, v in doc.items() if k != "text"},
                "score": float(scores[idx]),
                "source": "bm25",
            })

        return results

    def is_built(self) -> bool:
        return self._bm25 is not None

    def document_count(self) -> int:
        return len(self._docs)


class BM25SearchEngine:
    """
    Manages BM25 indices for both reference corpus and uploaded contracts.
    """

    def __init__(self):
        self.reference_index = BM25Index()
        self.contract_index = BM25Index()

    def build_reference_index(self, documents: list[dict]) -> None:
        """Build BM25 index for reference corpus."""
        self.reference_index.build(documents)

    def build_contract_index(self, documents: list[dict]) -> None:
        """Build BM25 index for uploaded contract (replaces any existing)."""
        self.contract_index = BM25Index()
        self.contract_index.build(documents)

    def search_reference(self, query: str, top_k: int = 10) -> list[dict]:
        """Search reference corpus with BM25."""
        return self.reference_index.search(query, top_k)

    def search_contract(self, query: str, top_k: int = 10) -> list[dict]:
        """Search uploaded contract with BM25."""
        return self.contract_index.search(query, top_k)
