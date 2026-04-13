"""
Embedding pipeline using BAAI/bge-base-en-v1.5.

BGE models prepend a query instruction for retrieval tasks — this
wrapper handles that automatically so callers don't need to think about it.
Embeddings are cached in memory to avoid recomputing identical texts.
"""

import hashlib
import logging
from functools import lru_cache
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# BGE models benefit from an instruction prefix for query embeddings
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class EmbeddingPipeline:
    """
    Wraps SentenceTransformer with BGE-aware query prefix and result caching.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        logger.info(f"Loading embedding model: {model_name}")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self._cache: dict[str, np.ndarray] = {}
        self._is_bge = "bge" in model_name.lower()

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        """
        Embed a list of document chunks (no query prefix).

        Args:
            texts: Raw text strings to embed.

        Returns:
            List of embedding vectors as numpy arrays.
        """
        return self._embed_batch(texts, add_query_prefix=False)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string (adds BGE query prefix if applicable).

        Args:
            query: The search query.

        Returns:
            Embedding vector as numpy array.
        """
        return self._embed_batch([query], add_query_prefix=True)[0]

    def _embed_batch(self, texts: list[str], add_query_prefix: bool) -> list[np.ndarray]:
        """Internal batch embedding with cache lookup."""
        results: list[Optional[np.ndarray]] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            cache_key = self._cache_key(text, add_query_prefix)
            if cache_key in self._cache:
                results[i] = self._cache[cache_key]
            else:
                uncached_indices.append(i)
                prefix = (_BGE_QUERY_PREFIX if self._is_bge and add_query_prefix else "")
                uncached_texts.append(prefix + text)

        if uncached_texts:
            logger.debug(f"Embedding {len(uncached_texts)} texts (model={self.model_name})")
            vectors = self.model.encode(
                uncached_texts,
                normalize_embeddings=True,  # cosine similarity via dot product
                show_progress_bar=False,
                batch_size=32,
            )
            for idx, vec in zip(uncached_indices, vectors):
                text = texts[idx]
                cache_key = self._cache_key(text, add_query_prefix)
                self._cache[cache_key] = vec
                results[idx] = vec

        return results  # type: ignore[return-value]

    @staticmethod
    def _cache_key(text: str, is_query: bool) -> str:
        h = hashlib.md5(text.encode()).hexdigest()
        return f"{'q' if is_query else 'd'}:{h}"

    def clear_cache(self) -> None:
        """Free the in-memory embedding cache."""
        self._cache.clear()
