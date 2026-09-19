# LegalRAG
### Multi-Agent RAG · LLM-as-Judge Verification · Obligation Extraction · Contradiction Detection · Category-Routed Chat

A portfolio RAG prototype for legal contract risk analysis. Upload any contract (PDF, DOCX, TXT) and receive clause-by-clause risk assessment with citations, obligation tables, contradiction detection, and grounded Q&A — all self-verified by an LLM judge.

**Live demo:** [Hugging Face Spaces](https://huggingface.co/spaces/Iyman-ahmed/legal-contract-analyzer-with-clause-risk)

---

## Architecture

```
PDF/DOCX/TXT Upload
        │
        ▼
┌──────────────────────────────┐
│  Document Ingestion           │
│  PyMuPDF → pdfplumber         │  Section-aware chunking preserves
│  python-docx                  │  clause boundaries. Each chunk is
└─────────┬────────────────────┘  tagged with one of 14 clause types.
          │  EnrichedChunk[] with clause_type labels
          ▼
┌──────────────────────────────┐     ┌──────────────────────────┐
│  Hybrid Retrieval             │     │  Reference Knowledge Base│
│  Dense: BGE-base-en-v1.5      │◄───►│  20 clause templates     │
│  Sparse: BM25 (legal stopwords)│    │  LOW + HIGH risk variants │
│  RRF fusion → Cross-encoder   │     │  per clause type          │
└─────────┬────────────────────┘     └──────────────────────────┘
          │  top-5 reference clauses per uploaded clause
          ▼
┌─────────────────────────────────────────────────────────────┐
│  Multi-Agent Analysis Pipeline                               │
│                                                              │
│  ① Map — per-clause LLM analysis (Pydantic-validated JSON)   │
│     └─ VerificationAgent (LLM-as-judge)                      │
│          Asks: "Is every claim supported by the context?"    │
│          faithfulness score < 0.75 → re-retrieve + re-analyze│
│                                                              │
│  ② ObligationAgent                                           │
│     Dedicated LLM pass → structured table of:               │
│     deadlines · payments · renewals · termination triggers   │
│                                                              │
│  ③ ContradictionAgent                                        │
│     Feeds full clause set to LLM in one call →              │
│     finds cross-clause conflicts (e.g. liability cap in §3   │
│     vs. unlimited indemnification carve-out in §9)           │
│                                                              │
│  ④ Reduce — DocumentSummary + missing-clause check           │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Output Guardrails                                           │
│  Citation verification · Faithfulness check · Disclaimer     │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
             Gradio UI (5 tabs)
   Upload | Analysis | Obligations | Chat | Evaluation

                Chat Q&A mode:
         Question → Category Router
    (12-type weighted keyword scoring)
          │                    │
    clause_type inferred   ambiguous / off-topic
    filtered search first  full-corpus fallback
    (fewer tokens, less     (no clause missed)
     cross-category noise)
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
| Agents | VerificationAgent · ObligationAgent · ContradictionAgent | Self-verifying multi-agent pipeline |

---

## Key Design Decisions

**Why multi-agent verification instead of a single LLM pass?**
A single LLM call per clause produces plausible-sounding output but cannot catch its own unsupported claims. `VerificationAgent` runs a second LLM call as a judge: "Is every claim directly supported by the retrieved context?" If faithfulness score < 0.75, the pipeline re-retrieves with a broader query and re-analyzes — at most once. This is a verification heuristic; it does not establish legal correctness.

**Why a separate ObligationAgent?**
Clause risk analysis asks "how risky is this clause?" Obligation extraction asks "what must I actually do and by when?" They require different prompts, different schemas, and different reasoning. A dedicated agent with a targeted schema (deadline, party, consequence, clause_reference, obligation_type) produces far better results than mixing concerns in the risk prompt.

**Why ContradictionAgent runs over the full clause set?**
Individual clause analysis is blind to cross-clause conflicts. A liability cap in §3 and unlimited indemnification in §9 each look reasonable in isolation — together they create legal uncertainty that no per-clause agent can detect. The contradiction agent feeds the complete structured output to a single LLM call designed to find these conflicts.

**Why category-routed Q&A?**
When a user asks "What is the liability cap?" the old pipeline searched all contract chunks — indemnification, payment terms, IP clauses, everything. `_infer_question_clause_type()` maps the question to `limitation_of_liability` using weighted keyword scoring, then searches only that category. Results: fewer irrelevant chunks in context, less cross-category hallucination, lower token consumption. Falls back to full-corpus search if the category filter returns < 2 results (handles mis-classified or absent clauses).

**Why BGE over OpenAI embeddings?**
BGE-base-en-v1.5 runs entirely on CPU at 768 dimensions, costs nothing per query, and ranks #1 on MTEB retrieval benchmarks among open models. For HF Spaces free tier (2 vCPU, 16 GB RAM), this is the only viable choice.

**Why RRF over weighted fusion?**
Reciprocal Rank Fusion requires no tuning — no α parameter to optimize. It consistently outperforms weighted combinations in benchmarks (Cormack 2009) and is immune to score distribution differences between dense and sparse results.

**Why section-aware chunking?**
Recursive character splitting cuts clauses mid-sentence, destroying legal meaning. The chunker splits on legal section boundaries first (numbered clauses, `ARTICLE`, `SECTION` patterns), then falls back to sentence boundaries for large sections — preserving clause integrity throughout.

**Why Pydantic with retry?**
LLMs occasionally produce malformed JSON. Pydantic v2 validates the schema strictly and raises `ValidationError` on failure. We retry up to 3 times before falling back to a degraded response — a bounded error-handling mechanism.

---

## Accuracy Results

Evaluated on 50 generated contracts (NDA, SaaS, Employment, Service, Lease) at LOW / MEDIUM / HIGH / CRITICAL risk levels:

| Metric | Score |
|--------|-------|
| Mean Clause-Type Precision | 90.0% |
| Mean Clause-Type Recall | 96.8% |
| Mean Clause-Type F1 | 92.4% |
| Type-filtered Nonempty Retrieval | 100% (266/266 queries) |
| High-Risk Contracts With All Expected Risky Types | 58.8% *(known gap — lease + NDA miss rate)* |

These are historical results on 50 synthetic contracts. Precision and recall compare sets of clause types per contract, then average across contracts. They do not measure legal-answer correctness, individual clause extraction, or user-query retrieval relevance. The 266 queries use the expected clause type as a filter and count any returned candidate; there are no independently judged relevance labels. The 58.8% metric is a contract-level coverage proxy, not risk-classification recall.

**Per contract type:**

| Contract Type | F1 | Recall | Type-filtered Nonempty Retrieval |
|---|---|---|---|
| SaaS Agreement | 97.5% | 95.3% | 100% |
| Service Agreement | 98.2% | 96.6% | 100% |
| NDA | 94.5% | 94.5% | 100% |
| Employment Contract | 88.7% | 100% | 100% |
| Lease Agreement | 67.9% | 100% | 100% |

**Category routing test results (no LLM, unit tests):**

| Test Class | Cases | Result |
|---|---|---|
| Unambiguous clause-type routing | 43 | 43/43 pass |
| Off-topic → None (no false routing) | 7 | 7/7 pass |
| Cross-clause ambiguity (acceptable outcomes) | 5 | 5/5 pass |
| Category confusion avoided | 3 | 3/3 pass |
| Paraphrase graceful degradation | 3 | 3/3 pass (known gaps documented) |

Clause types covered: `indemnification` · `limitation_of_liability` · `termination` · `ip_assignment` · `non_compete` · `confidentiality` · `governing_law` · `dispute_resolution` · `data_protection` · `force_majeure` · `payment_terms` · `warranty` · `representations`

---

## Quickstart

```bash
git clone https://github.com/Iyman-Ahmed/ai-legal-contract-analyzer
cd ai-legal-contract-analyzer

python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Open .env and set your LLM provider key (see below)

python app.py
# Opens at http://localhost:7860
```

### Choosing a provider

Open `.env` and set `LLM_PROVIDER` to one of:

| Provider | Key needed | Notes |
|---|---|---|
| `groq` **(default, free)** | `GROQ_API_KEY` | Free tier at [console.groq.com](https://console.groq.com). Runs Llama 3.3-70B. |
| `anthropic` | `ANTHROPIC_API_KEY` | Claude Sonnet 4.6 — best output quality. |
| `openai` | `OPENAI_API_KEY` | GPT-4o-mini. |
| `lmstudio` | None | Fully local. Start LM Studio, load any model, enable the server on port 1234. |

Groq is **free** and recommended for first-time setup. No GPU required — all embedding and reranking models run on CPU.

---

## Running Tests

```bash
# All unit + retrieval + accuracy tests (no LLM required)
python -m pytest tests/ -v -k "not integration"
# 152 tests

# Integration tests (hallucination resistance, full pipeline — requires LLM in .env)
python -m pytest tests/test_chat_routing.py -m integration -v

# Smoke test for the three new agents (requires LLM + LM Studio or other provider)
cd contract-analyzer
PYTHONPATH=. python scripts/smoke_test_agents.py
```

| Test File | Coverage | Tests |
|-----------|----------|-------|
| `test_ingestion.py` | Parser, chunker, metadata extractor | 20 |
| `test_retrieval.py` | Embeddings, BM25, ChromaDB, hybrid search, reranker, knowledge base | 28 |
| `test_guardrails.py` | Citation verifier, faithfulness checker, disclaimer | 16 |
| `test_accuracy.py` | Clause detection F1, retrieval hit rate across 50 contracts | 27 |
| `test_chat_routing.py` | Category router correctness, hallucination resistance, cross-clause confusion | 61 unit + 14 integration |

---

## Running the RAGAS Evaluation

```bash
# Heuristic mode — no LLM or API key needed, fast
PYTHONPATH=. python scripts/run_ragas_eval.py --heuristic

# LLM-judge mode — requires a provider key in .env
PYTHONPATH=. python scripts/run_ragas_eval.py
```

Results are saved to `data/evaluation/ragas_results.json`.

---

## Project Structure

```
├── app.py                          # Gradio entry point (5-tab UI)
├── requirements.txt
├── .env.example                    # API key template
├── config/settings.py              # All constants and model names
├── src/
│   ├── agents/                     # Multi-agent pipeline (new)
│   │   ├── verification.py         # LLM-as-judge faithfulness verifier
│   │   ├── obligation.py           # Deadline/payment/renewal extractor
│   │   └── contradiction.py        # Cross-clause conflict detector
│   ├── ingestion/
│   │   ├── parser.py               # PDF/DOCX/TXT parsing (PyMuPDF + fallback)
│   │   ├── chunker.py              # Section-aware legal chunking
│   │   └── metadata.py             # Keyword-based clause type classifier (14 types)
│   ├── retrieval/
│   │   ├── embeddings.py           # BGE embedding pipeline with caching
│   │   ├── vector_store.py         # ChromaDB wrapper (two collections: reference + contract)
│   │   ├── bm25_search.py          # BM25 sparse search (legal stopwords)
│   │   ├── hybrid_search.py        # RRF fusion with optional clause_type_filter
│   │   ├── reranker.py             # Cross-encoder reranking
│   │   └── knowledge_base.py       # Reference corpus indexer
│   ├── analysis/
│   │   ├── schemas.py              # Pydantic v2 models (ClauseRisk, ObligationTable, ContradictionReport)
│   │   ├── prompts.py              # All LLM prompt templates
│   │   └── risk_engine.py          # Orchestrator: map-reduce + agents + category router
│   ├── guardrails/
│   │   ├── citation_check.py       # Citation verification
│   │   ├── faithfulness.py         # Token-overlap grounding check (LLM-as-judge upgrade)
│   │   └── disclaimer.py           # Legal disclaimer injection
│   ├── evaluation/
│   │   ├── ragas_eval.py           # RAG quality metrics
│   │   └── golden_set.json         # 15 curated golden Q&A pairs (CUAD-sourced)
│   └── ui/components.py            # Markdown rendering helpers
├── data/
│   ├── reference_contracts/        # 20 standard clause templates (LOW + HIGH variants)
│   ├── risk_patterns/              # Risk rubrics per clause type
│   ├── sample_contracts/           # Demo contracts (NDA, SaaS, Employment)
│   └── test_contracts/             # 55 parameterized contracts for accuracy testing
├── scripts/
│   ├── generate_test_dataset.py    # Generates 50 parameterized test contracts
│   ├── smoke_test_agents.py        # Fast agent smoke test (no ChromaDB/Gradio needed)
│   └── build_eval_dataset.py       # Builds golden set from CUAD dataset
└── tests/
    ├── test_ingestion.py
    ├── test_retrieval.py
    ├── test_guardrails.py
    ├── test_accuracy.py            # Clause detection F1 across 50 contracts
    └── test_chat_routing.py        # Category routing + hallucination resistance (new)
```

---

## Known Limitations

- **High-risk contract coverage gap:** 41.2% of HIGH/CRITICAL synthetic contracts lack at least one expected risky clause type (complete type coverage = 58.8%). Worst in lease agreements (F1 = 67.9%) where lease-specific vocabulary diverges from reference clause patterns. Mitigation: expand reference corpus with lease-specific templates.
- **Paraphrase routing gaps:** `_infer_question_clause_type()` uses keyword signals. Unusual legal phrasing ("ceiling on vendor's exposure" instead of "liability cap") falls back to full-corpus search — correct behavior, but slightly less efficient.
- **Reference corpus size:** 20 hand-curated clauses. Adding CUAD's 500+ contracts would improve retrieval quality for edge-case clause variants.
- **PDF layout complexity:** Multi-column contracts and scanned PDFs (requiring OCR) are not fully supported.
- **Jurisdiction coverage:** Risk rubrics are US/California-centric. EU/UK contract norms differ.
- **LLM latency:** Multi-agent pipeline (verification + obligation + contradiction) adds 3–5 LLM calls per contract. For 30-clause contracts, total analysis is 2–4 minutes. Async processing would improve UX.

---

*Built to demonstrate production RAG engineering. Not a law firm. Not legal advice.*
