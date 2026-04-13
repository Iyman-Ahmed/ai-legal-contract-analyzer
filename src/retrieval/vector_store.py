"""
ChromaDB vector store wrapper.

Manages two collections:
  - legal_reference: pre-indexed standard contract templates
  - uploaded_contract: the current user-uploaded document (cleared per session)

ChromaDB runs fully in-process on CPU with no external service required.
"""

import logging
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings

from config.settings import (
    CHROMA_DIR,
    CHROMA_COLLECTION_REFERENCE,
    CHROMA_COLLECTION_CONTRACT,
    EMBEDDING_DIMENSION,
)
from src.retrieval.embeddings import EmbeddingPipeline

logger = logging.getLogger(__name__)


class VectorStore:
    """
    ChromaDB-backed vector store with a unified add/query interface.
    """

    def __init__(self, embedding_pipeline: EmbeddingPipeline, persist: bool = True):
        """
        Args:
            embedding_pipeline: Initialized EmbeddingPipeline instance.
            persist: Whether to persist ChromaDB to disk (False for tests).
        """
        self.embedder = embedding_pipeline

        if persist:
            CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=str(CHROMA_DIR),
                settings=Settings(anonymized_telemetry=False),
            )
        else:
            self.client = chromadb.EphemeralClient(
                settings=Settings(anonymized_telemetry=False)
            )

        self.reference_col = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_REFERENCE,
            metadata={"hnsw:space": "cosine"},
        )
        self.contract_col = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_CONTRACT,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB initialized — reference={self.reference_col.count()} docs, "
            f"contract={self.contract_col.count()} docs"
        )

    # ── Add documents ─────────────────────────────────────────────────────────

    def add_reference_chunks(self, chunks: list[dict]) -> None:
        """
        Add reference contract chunks to the reference collection.

        Args:
            chunks: List of dicts with keys: chunk_id, text, + metadata fields.
        """
        self._add_chunks(self.reference_col, chunks)

    def add_contract_chunks(self, chunks: list[dict]) -> None:
        """
        Replace the uploaded contract collection with new chunks.
        Clears any previous upload first.
        """
        # Delete and recreate to clear previous upload
        self.client.delete_collection(CHROMA_COLLECTION_CONTRACT)
        self.contract_col = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_CONTRACT,
            metadata={"hnsw:space": "cosine"},
        )
        self._add_chunks(self.contract_col, chunks)

    def _add_chunks(self, collection: chromadb.Collection, chunks: list[dict]) -> None:
        if not chunks:
            return

        texts = [c["text"] for c in chunks]
        ids = [c["chunk_id"] for c in chunks]
        metadatas = [
            {k: v for k, v in c.items() if k not in ("text", "chunk_id")}
            for c in chunks
        ]

        # Embed in batches of 64
        batch_size = 64
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            batch_ids = ids[start : start + batch_size]
            batch_metas = metadatas[start : start + batch_size]

            embeddings = self.embedder.embed_documents(batch_texts)

            collection.add(
                ids=batch_ids,
                embeddings=[e.tolist() for e in embeddings],
                documents=batch_texts,
                metadatas=batch_metas,
            )

        logger.info(f"Added {len(chunks)} chunks to '{collection.name}'")

    # ── Query ─────────────────────────────────────────────────────────────────

    def query_reference(
        self,
        query: str,
        top_k: int = 10,
        clause_type_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Dense search against the reference collection.

        Args:
            query: Search query string.
            top_k: Number of results to return.
            clause_type_filter: If set, only return chunks of this clause type.

        Returns:
            List of result dicts with text, metadata, and distance score.
        """
        return self._query(self.reference_col, query, top_k, clause_type_filter)

    def query_contract(
        self,
        query: str,
        top_k: int = 10,
        clause_type_filter: Optional[str] = None,
    ) -> list[dict]:
        """Dense search against the uploaded contract collection."""
        return self._query(self.contract_col, query, top_k, clause_type_filter)

    def _query(
        self,
        collection: chromadb.Collection,
        query: str,
        top_k: int,
        clause_type_filter: Optional[str],
    ) -> list[dict]:
        if collection.count() == 0:
            return []

        query_embedding = self.embedder.embed_query(query)

        where_filter = None
        if clause_type_filter:
            where_filter = {"clause_type": {"$eq": clause_type_filter}}

        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, collection.count()),
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        output: list[dict] = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({
                "text": doc,
                "metadata": meta,
                "score": 1.0 - dist,  # cosine similarity from cosine distance
                "source": "dense",
            })

        return output

    # ── Status ────────────────────────────────────────────────────────────────

    def reference_count(self) -> int:
        return self.reference_col.count()

    def contract_count(self) -> int:
        return self.contract_col.count()

    def is_reference_indexed(self) -> bool:
        return self.reference_col.count() > 0
