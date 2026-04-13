"""
Reference knowledge base builder.

Loads standard contract clauses from JSON, enriches with metadata,
and indexes them into ChromaDB + BM25 for hybrid retrieval.
"""

import json
import logging
from pathlib import Path

from config.settings import REFERENCE_DIR, RISK_PATTERNS_DIR
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_search import BM25SearchEngine

logger = logging.getLogger(__name__)


class KnowledgeBaseBuilder:
    """
    Builds and manages the reference knowledge base.
    """

    def __init__(self, vector_store: VectorStore, bm25_engine: BM25SearchEngine):
        self.vector_store = vector_store
        self.bm25_engine = bm25_engine
        self._risk_rubrics: dict = {}

    def build(self, force_rebuild: bool = False) -> int:
        """
        Build the reference knowledge base.

        Skips rebuild if ChromaDB already has reference data (unless force_rebuild=True).

        Returns:
            Number of reference documents indexed.
        """
        if self.vector_store.is_reference_indexed() and not force_rebuild:
            count = self.vector_store.reference_count()
            logger.info(f"Reference KB already indexed ({count} docs). Skipping rebuild.")
            # Still need to rebuild BM25 (in-memory, not persisted)
            self._rebuild_bm25_from_json()
            return count

        logger.info("Building reference knowledge base...")
        documents = self._load_reference_documents()
        self._risk_rubrics = self._load_risk_rubrics()

        if not documents:
            raise RuntimeError(
                f"No reference documents found in {REFERENCE_DIR}. "
                "Ensure standard_clauses.json exists."
            )

        # Enrich each document with risk rubric data
        enriched = [self._enrich_with_rubric(doc) for doc in documents]

        self.vector_store.add_reference_chunks(enriched)
        self.bm25_engine.build_reference_index(enriched)

        logger.info(f"Reference KB built with {len(enriched)} documents")
        return len(enriched)

    def get_risk_rubric(self, clause_type: str) -> dict:
        """Get the risk rubric for a given clause type."""
        return self._risk_rubrics.get(clause_type, {})

    def _load_reference_documents(self) -> list[dict]:
        """Load all reference clause JSON files from the data directory."""
        documents: list[dict] = []

        clause_file = REFERENCE_DIR / "standard_clauses.json"
        if clause_file.exists():
            with open(clause_file, "r") as f:
                clauses = json.load(f)
            for clause in clauses:
                doc = {
                    "chunk_id": clause["id"],
                    "text": f"{clause['title']}\n\n{clause['text']}",
                    "clause_type": clause.get("clause_type", "general"),
                    "risk_level": clause.get("risk_level", "MEDIUM"),
                    "section_title": clause.get("title", ""),
                    "clause_number": "",
                    "page_number": 1,
                    "chunk_index": 0,
                    "char_start": 0,
                    "char_end": len(clause["text"]),
                    "word_count": len(clause["text"].split()),
                    "source_filename": clause.get("source_filename", "standard_clauses.json"),
                    "jurisdiction": clause.get("jurisdiction", "General"),
                    "contract_types": ", ".join(clause.get("contract_types", [])),
                }
                documents.append(doc)

        logger.info(f"Loaded {len(documents)} reference clauses")
        return documents

    def _load_risk_rubrics(self) -> dict:
        """Load risk scoring rubrics."""
        rubric_file = RISK_PATTERNS_DIR / "risk_rubrics.json"
        if rubric_file.exists():
            with open(rubric_file, "r") as f:
                return json.load(f)
        return {}

    def _enrich_with_rubric(self, doc: dict) -> dict:
        """Add risk rubric context to the document text for richer retrieval."""
        rubric = self._risk_rubrics.get(doc["clause_type"], {})
        if rubric:
            rubric_text = (
                f"\nStandard: {rubric.get('standard_market_position', '')}"
                f"\nHigh risk signals: {', '.join(rubric.get('high_risk_signals', []))}"
            )
            doc["text"] = doc["text"] + rubric_text
        return doc

    def _rebuild_bm25_from_json(self) -> None:
        """Rebuild in-memory BM25 index from JSON files (called on startup if ChromaDB exists)."""
        documents = self._load_reference_documents()
        self._risk_rubrics = self._load_risk_rubrics()
        enriched = [self._enrich_with_rubric(doc) for doc in documents]
        self.bm25_engine.build_reference_index(enriched)
        logger.info(f"BM25 index rebuilt with {len(enriched)} reference documents")
