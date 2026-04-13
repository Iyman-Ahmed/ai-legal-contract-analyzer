"""
Central configuration for the Legal Contract Analyzer.
All constants, model names, thresholds, and paths live here.
"""

import os
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REFERENCE_DIR = DATA_DIR / "reference_contracts"
RISK_PATTERNS_DIR = DATA_DIR / "risk_patterns"
SAMPLE_CONTRACTS_DIR = DATA_DIR / "sample_contracts"
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"

# ─── LLM ──────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# Primary LLM — Claude Sonnet via Anthropic API
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "anthropic")  # "anthropic" | "openai"
ANTHROPIC_MODEL: str = "claude-sonnet-4-6"
OPENAI_MODEL: str = "gpt-4o-mini"

# ─── Embeddings ───────────────────────────────────────────────────────────────
# BAAI/bge-base-en-v1.5 — top performer on MTEB for semantic search, CPU-friendly
EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIMENSION: int = 768

# ─── Reranker ─────────────────────────────────────────────────────────────────
RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ─── Retrieval ────────────────────────────────────────────────────────────────
DENSE_TOP_K: int = 20          # candidates from dense search
SPARSE_TOP_K: int = 20         # candidates from BM25
RERANK_TOP_K: int = 10         # candidates sent to reranker
FINAL_TOP_K: int = 5           # final results returned to LLM
RRF_K: int = 60                # RRF constant (standard = 60)

# ─── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = 512          # tokens — balanced for legal clause length
CHUNK_OVERLAP: int = 64        # slight overlap to catch cross-boundary context

# ─── Analysis ─────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD: float = 0.4   # below this, flag output as uncertain
MAX_RETRIES: int = 3                 # Pydantic validation retry attempts
MAX_CLAUSES_PER_ANALYSIS: int = 30  # prevent runaway processing

# ─── ChromaDB ─────────────────────────────────────────────────────────────────
CHROMA_COLLECTION_REFERENCE: str = "legal_reference"
CHROMA_COLLECTION_CONTRACT: str = "uploaded_contract"

# ─── UI ───────────────────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB: int = 20
SUPPORTED_FORMATS: list[str] = [".pdf", ".docx", ".txt"]

# ─── Clause Types ─────────────────────────────────────────────────────────────
CLAUSE_TYPES: list[str] = [
    "indemnification",
    "limitation_of_liability",
    "termination",
    "ip_assignment",
    "non_compete",
    "confidentiality",
    "governing_law",
    "dispute_resolution",
    "data_protection",
    "force_majeure",
    "payment_terms",
    "warranty",
    "representations",
    "general",
]

RISK_LEVELS: list[str] = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
