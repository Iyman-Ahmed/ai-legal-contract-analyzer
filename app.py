"""
AI Legal Contract Analyzer — Main Gradio Application

Entry point for Hugging Face Spaces deployment.
Initializes all pipeline components on startup, then serves the UI.
"""

import os
import sys
import logging
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# ── Load environment variables ──────────────────────────────────────────────
load_dotenv()

# ── Logging setup ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Ensure src is importable ────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr

from config.settings import SAMPLE_CONTRACTS_DIR
from src.ingestion.parser import DocumentParser
from src.ingestion.chunker import SectionAwareChunker
from src.ingestion.metadata import MetadataExtractor
from src.retrieval.embeddings import EmbeddingPipeline
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_search import BM25SearchEngine
from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.knowledge_base import KnowledgeBaseBuilder
from src.analysis.risk_engine import RiskAnalysisEngine
from src.guardrails.citation_check import CitationVerifier
from src.guardrails.faithfulness import FaithfulnessChecker
from src.guardrails.disclaimer import get_disclaimer
from src.evaluation.ragas_eval import RAGEvaluator
from src.ui.components import (
    render_document_summary_md,
    render_clause_analysis_md,
    render_export_report,
)

# ═══════════════════════════════════════════════════════════════════════════
# Global pipeline components (initialized once at startup)
# ═══════════════════════════════════════════════════════════════════════════

logger.info("Initializing pipeline components...")

_parser = DocumentParser()
_chunker = SectionAwareChunker()
_meta_extractor = MetadataExtractor()
_embedder = EmbeddingPipeline()
_vector_store = VectorStore(embedding_pipeline=_embedder, persist=True)
_bm25 = BM25SearchEngine()
_hybrid_search = HybridSearchEngine(vector_store=_vector_store, bm25_engine=_bm25)
_reranker = CrossEncoderReranker()
_kb_builder = KnowledgeBaseBuilder(vector_store=_vector_store, bm25_engine=_bm25)
_citation_verifier = CitationVerifier()
_faithfulness_checker = FaithfulnessChecker()
_evaluator = RAGEvaluator(hybrid_search=_hybrid_search, reranker=_reranker)

# Build reference knowledge base at startup
try:
    ref_count = _kb_builder.build(force_rebuild=False)
    logger.info(f"Reference knowledge base ready: {ref_count} documents")
except Exception as e:
    logger.error(f"Failed to build reference knowledge base: {e}")
    ref_count = 0

# Risk engine (requires API key — initialized lazily)
_risk_engine: RiskAnalysisEngine | None = None
_current_enriched_chunks = []  # Store for chat context
_current_result = None


def get_risk_engine() -> Optional[RiskAnalysisEngine]:
    """Lazily initialize the risk engine (requires API key)."""
    global _risk_engine
    if _risk_engine is None:
        try:
            _risk_engine = RiskAnalysisEngine(
                hybrid_search=_hybrid_search,
                reranker=_reranker,
            )
        except ValueError as e:
            return None
    return _risk_engine


# ═══════════════════════════════════════════════════════════════════════════
# Core handler functions
# ═══════════════════════════════════════════════════════════════════════════

def load_sample_contract(contract_name: str) -> str:
    """Load a sample contract by name and return its path."""
    sample_map = {
        "📄 Sample NDA": "sample_nda.txt",
        "☁️ SaaS Agreement (High Risk)": "sample_saas_agreement.txt",
        "👔 Employment Contract (CA)": "sample_employment_contract.txt",
    }
    filename = sample_map.get(contract_name)
    if not filename:
        return None
    path = SAMPLE_CONTRACTS_DIR / filename
    return str(path) if path.exists() else None


def analyze_contract(
    file_obj,
    sample_choice: str,
    progress=gr.Progress(track_tqdm=True),
) -> tuple[str, str, str, str]:
    """
    Main analysis handler called by Gradio.

    Returns:
        (summary_md, clauses_md, export_report, status_message)
    """
    global _current_enriched_chunks, _current_result

    # ── Determine file source ───────────────────────────────────────────────
    file_path = None

    if file_obj is not None:
        file_path = file_obj.name
    elif sample_choice and sample_choice != "— Select a sample —":
        file_path = load_sample_contract(sample_choice)
        if not file_path:
            return "", "", "", "❌ Sample contract not found."
    else:
        return "", "", "", "⚠️ Please upload a contract file or select a sample contract."

    # ── Parse ───────────────────────────────────────────────────────────────
    try:
        progress(0.05, desc="Parsing document...")
        parsed = _parser.parse(file_path)
        logger.info(f"Parsed: {parsed.filename}, {parsed.total_pages} pages")
    except Exception as e:
        return "", "", "", f"❌ Document parsing failed: {str(e)}"

    # ── Chunk + enrich ──────────────────────────────────────────────────────
    try:
        progress(0.15, desc="Chunking document...")
        chunks = _chunker.chunk(parsed)
        enriched = _meta_extractor.enrich_all(chunks)
        _current_enriched_chunks = enriched

        # Index contract into vector store + BM25 for chat
        contract_docs = [c.to_dict() for c in enriched]
        _vector_store.add_contract_chunks(contract_docs)
        _bm25.build_contract_index(contract_docs)

        logger.info(f"Chunks: {len(enriched)}, types: {set(c.clause_type for c in enriched)}")
    except Exception as e:
        return "", "", "", f"❌ Document processing failed: {str(e)}"

    # ── Check API key ────────────────────────────────────────────────────────
    engine = get_risk_engine()
    if engine is None:
        return (
            "", "",
            "",
            "❌ No API key configured. Set ANTHROPIC_API_KEY or OPENAI_API_KEY in your .env file.",
        )

    # ── Analyze ─────────────────────────────────────────────────────────────
    try:
        def _progress_cb(msg: str, pct: float):
            progress(0.2 + pct * 0.7, desc=msg)

        result = engine.analyze_contract(
            chunks=enriched,
            filename=parsed.filename,
            progress_callback=_progress_cb,
        )
        _current_result = result

    except Exception as e:
        logger.exception("Analysis failed")
        return "", "", "", f"❌ Analysis failed: {str(e)}"

    # ── Guardrails ───────────────────────────────────────────────────────────
    try:
        progress(0.92, desc="Verifying citations...")
        all_retrieved = _hybrid_search.search_reference(
            query=parsed.full_text[:500], top_k=20
        )
        sources = _citation_verifier.extract_sources_from_results(all_retrieved)
        result, citation_warnings = _citation_verifier.verify_result(result, sources)

        if citation_warnings:
            logger.info(f"Citation warnings: {len(citation_warnings)}")
    except Exception as e:
        logger.warning(f"Guardrails failed (non-fatal): {e}")

    # ── Render output ────────────────────────────────────────────────────────
    progress(0.98, desc="Rendering results...")
    summary_md = render_document_summary_md(result)
    clauses_md = render_clause_analysis_md(result.clause_analyses)
    export_report = render_export_report(result)

    risk_val = result.document_summary.overall_risk_level.value
    risk_emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴", "CRITICAL": "⛔"}.get(risk_val, "⚪")
    status = (
        f"✅ Analysis complete — {result.total_clauses_analyzed} clauses analyzed | "
        f"Overall risk: {risk_emoji} {risk_val} ({result.document_summary.overall_risk_score:.1f}/10)"
    )

    return summary_md, clauses_md, export_report, status


def chat_with_contract(question: str, history: list) -> tuple[list, str]:
    """Handle chat Q&A about the uploaded contract."""
    if not _current_enriched_chunks:
        history.append((question, "⚠️ Please analyze a contract first before asking questions."))
        return history, ""

    engine = get_risk_engine()
    if engine is None:
        history.append((question, "❌ API key not configured."))
        return history, ""

    try:
        response = engine.answer_question(
            question=question,
            contract_chunks=_current_enriched_chunks,
        )
        answer = response.get("answer", "Unable to generate answer.")
        citations = response.get("citations", [])
        disclaimer = response.get("disclaimer", "")

        full_response = answer
        if citations:
            full_response += f"\n\n*Sources: {', '.join(citations[:3])}*"
        full_response += f"\n\n_{disclaimer}_"

        history.append((question, full_response))
    except Exception as e:
        history.append((question, f"❌ Error: {str(e)}"))

    return history, ""


def run_evaluation() -> str:
    """Run the RAG evaluation and return formatted results."""
    if not _vector_store.is_reference_indexed():
        return "❌ Reference knowledge base not indexed. Restart the application."

    try:
        results = _evaluator.run_evaluation()
        return _evaluator.format_results_for_display(results)
    except Exception as e:
        return f"❌ Evaluation failed: {str(e)}"


def get_system_status() -> str:
    """Return current system status for display."""
    api_key_set = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY"))
    provider = os.getenv("LLM_PROVIDER", "anthropic")

    lines = [
        "### System Status",
        "",
        f"- **Reference KB:** {'✅' if _vector_store.is_reference_indexed() else '❌'} "
        f"{_vector_store.reference_count()} documents indexed",
        f"- **API Key:** {'✅ Configured' if api_key_set else '❌ Not set (add to .env)'}",
        f"- **LLM Provider:** {provider.title()}",
        f"- **Embedding Model:** BAAI/bge-base-en-v1.5",
        f"- **Reranker:** cross-encoder/ms-marco-MiniLM-L-6-v2",
        f"- **Vector DB:** ChromaDB (persistent)",
        "",
        get_disclaimer(short=True),
    ]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Gradio UI
# ═══════════════════════════════════════════════════════════════════════════

with gr.Blocks(
    title="AI Legal Contract Analyzer",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
    ),
    css="""
    .risk-header { font-size: 1.4em; font-weight: bold; }
    .status-bar { font-size: 0.9em; color: #6b7280; }
    footer { visibility: hidden }
    """,
) as demo:

    # ── Header ───────────────────────────────────────────────────────────────
    gr.Markdown("""
    # ⚖️ AI Legal Contract Analyzer
    ### Powered by RAG — Hybrid Search + Cross-Encoder Reranking + LLM Analysis

    Upload any legal contract (PDF, DOCX, or TXT) to receive instant AI-powered risk analysis
    with clause-by-clause assessment, citation trails, and suggested revisions.
    """)

    with gr.Tabs():

        # ── Tab 1: Upload & Analyze ──────────────────────────────────────────
        with gr.Tab("📤 Upload & Analyze"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Upload Contract")
                    file_upload = gr.File(
                        label="Upload PDF, DOCX, or TXT",
                        file_types=[".pdf", ".docx", ".txt"],
                        file_count="single",
                    )

                    gr.Markdown("### Or try a sample contract")
                    sample_dropdown = gr.Dropdown(
                        choices=[
                            "— Select a sample —",
                            "📄 Sample NDA",
                            "☁️ SaaS Agreement (High Risk)",
                            "👔 Employment Contract (CA)",
                        ],
                        value="— Select a sample —",
                        label="Sample Contracts",
                    )

                    analyze_btn = gr.Button(
                        "🔍 Analyze Contract",
                        variant="primary",
                        size="lg",
                    )

                    status_text = gr.Textbox(
                        label="Status",
                        interactive=False,
                        placeholder="Upload a contract and click Analyze...",
                    )

                    gr.Markdown(get_system_status())

                with gr.Column(scale=2):
                    summary_output = gr.Markdown(
                        label="Document Risk Summary",
                        value="_Analysis results will appear here._",
                    )

            analyze_btn.click(
                fn=analyze_contract,
                inputs=[file_upload, sample_dropdown],
                outputs=[summary_output, gr.State(), gr.State(), status_text],
                show_progress=True,
            )

        # ── Tab 2: Clause Analysis ───────────────────────────────────────────
        with gr.Tab("📋 Clause Analysis"):
            gr.Markdown("### Detailed Clause-by-Clause Risk Assessment")
            gr.Markdown("*Run an analysis in the Upload tab first*")

            clause_output = gr.Markdown(
                value="_Analyze a contract to see clause-level results._"
            )

            analyze_btn.click(
                fn=analyze_contract,
                inputs=[file_upload, sample_dropdown],
                outputs=[gr.State(), clause_output, gr.State(), gr.State()],
            )

        # ── Tab 3: Chat ──────────────────────────────────────────────────────
        with gr.Tab("💬 Ask About the Contract"):
            gr.Markdown("""
            ### Ask Questions About Your Contract
            After analyzing a contract, you can ask follow-up questions in plain English.

            **Example questions:**
            - *"Is the indemnification clause one-sided?"*
            - *"What happens to my IP if I'm terminated?"*
            - *"Can this contract be terminated without notice?"*
            - *"What are my confidentiality obligations after leaving?"*
            """)

            chatbot = gr.Chatbot(height=450, bubble_full_width=False)
            with gr.Row():
                chat_input = gr.Textbox(
                    placeholder="Ask a question about the contract...",
                    label="Your question",
                    scale=4,
                )
                chat_submit = gr.Button("Ask", variant="primary", scale=1)

            chat_submit.click(
                fn=chat_with_contract,
                inputs=[chat_input, chatbot],
                outputs=[chatbot, chat_input],
            )
            chat_input.submit(
                fn=chat_with_contract,
                inputs=[chat_input, chatbot],
                outputs=[chatbot, chat_input],
            )

            gr.Button("🗑️ Clear Chat").click(
                fn=lambda: ([], ""),
                outputs=[chatbot, chat_input],
            )

        # ── Tab 4: Export ────────────────────────────────────────────────────
        with gr.Tab("📥 Export Report"):
            gr.Markdown("### Download Full Analysis Report")

            export_preview = gr.Markdown(
                value="_Analyze a contract first to generate an export._"
            )

            with gr.Row():
                download_md = gr.File(label="Download Markdown Report", interactive=False)

            def prepare_download(report_text: str):
                if not report_text or report_text.startswith("_"):
                    return None
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".md", delete=False, prefix="contract_analysis_"
                ) as f:
                    f.write(report_text)
                    return f.name

            export_state = gr.State("")

            analyze_btn.click(
                fn=analyze_contract,
                inputs=[file_upload, sample_dropdown],
                outputs=[gr.State(), gr.State(), export_state, gr.State()],
            )

            export_state.change(
                fn=lambda r: (r, prepare_download(r)),
                inputs=[export_state],
                outputs=[export_preview, download_md],
            )

        # ── Tab 5: Evaluation ────────────────────────────────────────────────
        with gr.Tab("📊 RAG Evaluation"):
            gr.Markdown("""
            ### RAG Pipeline Quality Metrics

            This evaluation runs a curated golden test set of 10 questions against the
            retrieval pipeline to measure quality. This is what separates production systems
            from tutorial projects — we measure and care about retrieval quality.

            **Metrics:**
            - **Context Precision:** Are retrieved documents relevant to the query?
            - **Context Recall:** Does retrieved context contain expected information?
            - **Answer Relevancy:** Does context align with ground-truth answers?
            """)

            eval_btn = gr.Button("▶️ Run Evaluation", variant="secondary")
            eval_output = gr.Markdown(value="_Click 'Run Evaluation' to benchmark the RAG pipeline._")

            eval_btn.click(
                fn=run_evaluation,
                outputs=[eval_output],
                show_progress=True,
            )

    # ── Footer ────────────────────────────────────────────────────────────────
    gr.Markdown("""
    ---
    ⚠️ **Disclaimer:** AI-generated analysis for informational purposes only.
    Not legal advice. Consult a qualified attorney before acting on any information.
    Built with LangChain · ChromaDB · sentence-transformers · Gradio · Claude Sonnet
    """)


# ═══════════════════════════════════════════════════════════════════════════
# Launch
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
