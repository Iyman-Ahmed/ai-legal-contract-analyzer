---
title: Legal Contract Analyzer With Clause Risk
emoji: ⚖️
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
short_description: RAG · Hybrid Search · Reranking · Claude LLM
---

# LegalRAG
### Hybrid BM25 + Dense Search · Cross-Encoder Reranking · Claude LLM · Clause Risk Detection

A production-grade RAG application for legal contract risk analysis. Upload any contract (PDF, DOCX, TXT) and receive clause-by-clause risk assessment with citations, severity scoring, and suggested revisions.

**Live demo:** [Hugging Face Spaces](https://huggingface.co/spaces/Iyman-ahmed/legal-contract-analyzer-with-clause-risk)

---

## Architecture

```
PDF/DOCX/TXT Upload
        │
        ▼
┌─────────────────────────┐
│  Document Ingestion      │
│  PyMuPDF → pdfplumber    │  Section-aware chunking
│  python-docx             │  Clause type detection
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐     ┌──────────────────────────┐
│  Hybrid Embedding        │     │  Reference Knowledge Base│
│  Dense: BGE-base-en-v1.5│     │  20 standard clause      │
│  Sparse: BM25 (rank_bm25)│    │  templates + risk rubrics│
│  Store: ChromaDB         │     │  (CUAD-inspired)         │
└────────┬────────────────┘     └──────────┬───────────────┘
         │                                  │
         ▼                                  ▼
┌──────────────────────────────────────────────────────┐
│  Hybrid Search + Reranking                            │
│  RRF fusion (dense + sparse) → cross-encoder rerank  │
│  Metadata filtering by clause type (like-for-like)   │
└────────────────────┬─────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────┐
│  Risk Analysis Engine (Claude Sonnet)                │
│  Map: clause-by-clause analysis with Pydantic schema │
│  Reduce: document-level summary + missing clauses    │
└────────────────────┬─────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────┐
│  Output Guardrails                                    │
│  Citation verification | Faithfulness check          │
│  Legal disclaimer injection                          │
└────────────────────┬─────────────────────────────────┘
                     │
                     ▼
            Gradio UI (5 tabs)
   Upload | Analysis | Chat | Export | Evaluation
```

---

## Tech Stack

| Component | Technology | Why |
|-----------|------------|-----|
| UI | Gradio 4.44 | Native to HF Spaces, rich components |
| LLM | Claude Sonnet (`claude-sonnet-4-6`) | Best structured output, cost-effective |
| Embeddings | `BAAI/bge-base-en-v1.5` | #1 MTEB open model, CPU-runnable, 768-dim |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Lightweight, strong precision, CPU-runnable |
| Vector DB | ChromaDB 0.5.5 | Persistent, no external service needed |
| Sparse Search | rank-bm25 | Lightweight BM25, legal-aware stopwords |
| PDF Parsing | PyMuPDF + pdfplumber fallback | Structure-preserving, fast |
| Output Schema | Pydantic v2 | Strict validation + retry on invalid JSON |

---

## Key Design Decisions

**Why BGE over OpenAI embeddings?**
BGE-base-en-v1.5 runs entirely on CPU at 768 dimensions, costs nothing per query, and ranks #1 on MTEB retrieval benchmarks among open models. For HF Spaces free tier (2 vCPU, 16 GB RAM), this is the only viable choice.

**Why RRF over weighted fusion?**
Reciprocal Rank Fusion requires no tuning — no α parameter to optimize. It consistently outperforms weighted combinations in benchmarks (Cormack 2009) and is immune to score distribution differences between dense and sparse results.

**Why cross-encoder reranking?**
Bi-encoder embeddings compute query and document independently, losing cross-attention signal. Cross-encoders read both together, giving 15–20% precision improvement on top-k results at the cost of latency — acceptable since we only rerank 10–20 candidates.

**Why section-aware chunking?**
Recursive character splitting cuts clauses mid-sentence, destroying legal meaning. The chunker splits on legal section boundaries first (numbered clauses, `ARTICLE`, `SECTION` patterns), then falls back to sentence boundaries for large sections — preserving clause integrity throughout.

**Why Pydantic with retry?**
LLMs occasionally produce malformed JSON. Pydantic v2 validates the schema strictly and raises `ValidationError` on failure. We retry up to 3 times before falling back to a degraded response — production behavior, not tutorial behavior.

---

## Accuracy Results

Evaluated on 50 generated contracts (NDA, SaaS, Employment, Service, Lease) at LOW / MEDIUM / HIGH / CRITICAL risk levels:

| Metric | Score |
|--------|-------|
| Clause Detection Precision | 90.0% |
| Clause Detection Recall | 96.8% |
| Clause Detection F1 | 92.4% |
| Retrieval Hit Rate | 100% (266/266 queries) |

Clause types covered: `indemnification` · `limitation_of_liability` · `termination` · `ip_assignment` · `non_compete` · `confidentiality` · `governing_law` · `dispute_resolution` · `data_protection` · `force_majeure` · `payment_terms` · `warranty` · `representations`

---

## Local Setup

```bash
git clone https://github.com/Iyman-Ahmed/ai-legal-contract-analyzer
cd ai-legal-contract-analyzer

# Install dependencies
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY=sk-ant-...

# Run
python app.py
# Open http://localhost:7860
```

---

## Running Tests

```bash
python -m pytest tests/ -v
# 91 tests across ingestion, retrieval, guardrails, and accuracy
```

| Test File | Coverage | Tests |
|-----------|----------|-------|
| `test_ingestion.py` | Parser, chunker, metadata extractor | 20 |
| `test_retrieval.py` | Embeddings, BM25, ChromaDB, hybrid search, reranker, knowledge base | 28 |
| `test_guardrails.py` | Citation verifier, faithfulness checker, disclaimer | 16 |
| `test_accuracy.py` | Clause detection F1, retrieval hit rate across 50 contracts | 27 |

---

## HF Spaces Deployment

1. Create Space at huggingface.co → **Gradio SDK**, **CPU Basic** (free tier)
2. Add `ANTHROPIC_API_KEY` as a Space Secret (Settings → Repository secrets)
3. Push code — entry point is `app.py`

Live: [huggingface.co/spaces/Iyman-ahmed/legal-contract-analyzer-with-clause-risk](https://huggingface.co/spaces/Iyman-ahmed/legal-contract-analyzer-with-clause-risk)

---

## Project Structure

```
├── app.py                          # Gradio entry point (5-tab UI)
├── requirements.txt
├── .env.example                    # API key template
├── config/settings.py              # All constants and model names
├── src/
│   ├── ingestion/
│   │   ├── parser.py               # PDF/DOCX/TXT parsing (PyMuPDF + fallback)
│   │   ├── chunker.py              # Section-aware legal chunking
│   │   └── metadata.py             # Keyword-based clause type classifier (13 types)
│   ├── retrieval/
│   │   ├── embeddings.py           # BGE embedding pipeline with caching
│   │   ├── vector_store.py         # ChromaDB wrapper
│   │   ├── bm25_search.py          # BM25 sparse search (legal stopwords)
│   │   ├── hybrid_search.py        # RRF fusion
│   │   ├── reranker.py             # Cross-encoder reranking
│   │   └── knowledge_base.py       # Reference corpus indexer
│   ├── analysis/
│   │   ├── schemas.py              # Pydantic v2 output models
│   │   ├── prompts.py              # All LLM prompt templates
│   │   └── risk_engine.py          # Map-reduce analysis pipeline
│   ├── guardrails/
│   │   ├── citation_check.py       # Citation verification
│   │   ├── faithfulness.py         # Token-overlap grounding check
│   │   └── disclaimer.py           # Legal disclaimer injection
│   ├── evaluation/
│   │   ├── ragas_eval.py           # RAG quality metrics
│   │   └── golden_set.json         # 10 curated golden Q&A pairs
│   └── ui/components.py            # Markdown rendering helpers
├── data/
│   ├── reference_contracts/        # 20 standard clause templates (LOW + HIGH variants)
│   ├── risk_patterns/              # Risk rubrics per clause type
│   └── sample_contracts/           # Demo contracts (NDA, SaaS, Employment)
├── scripts/
│   └── generate_test_dataset.py    # Generates 50 parameterized test contracts
└── tests/
    ├── test_ingestion.py
    ├── test_retrieval.py
    ├── test_guardrails.py
    └── test_accuracy.py            # Accuracy evaluation across 50 contracts
```

---

## Limitations and Future Work

- **Reference corpus size:** Currently 20 hand-curated clauses. Adding CUAD's 500+ contracts would improve retrieval quality.
- **PDF layout complexity:** Multi-column contracts and scanned PDFs (requiring OCR) are not fully supported.
- **Jurisdiction coverage:** Risk rubrics are US/California-centric. EU/UK contract norms differ.
- **LLM latency:** Claude Sonnet takes 2–5s per clause. For 30-clause contracts, total analysis is 1–2 minutes. Async processing would improve UX.

---

*Built to demonstrate production RAG engineering. Not a law firm. Not legal advice.*
