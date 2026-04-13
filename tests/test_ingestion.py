"""
Tests for document ingestion pipeline: parser, chunker, metadata extractor.
Run with: python -m pytest tests/test_ingestion.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.ingestion.parser import DocumentParser
from src.ingestion.chunker import SectionAwareChunker
from src.ingestion.metadata import MetadataExtractor
from config.settings import SAMPLE_CONTRACTS_DIR


SAMPLE_NDA = SAMPLE_CONTRACTS_DIR / "sample_nda.txt"
SAMPLE_SAAS = SAMPLE_CONTRACTS_DIR / "sample_saas_agreement.txt"
SAMPLE_EMPLOYMENT = SAMPLE_CONTRACTS_DIR / "sample_employment_contract.txt"


class TestDocumentParser:
    def setup_method(self):
        self.parser = DocumentParser()

    def test_parse_txt_nda(self):
        doc = self.parser.parse(SAMPLE_NDA)
        assert doc.filename == "sample_nda.txt"
        assert doc.file_type == "txt"
        assert len(doc.full_text) > 100
        assert doc.total_pages == 1

    def test_parse_txt_saas(self):
        doc = self.parser.parse(SAMPLE_SAAS)
        assert "SaaS" in doc.full_text or "saas" in doc.full_text.lower()
        assert len(doc.full_text) > 500

    def test_parse_txt_employment(self):
        doc = self.parser.parse(SAMPLE_EMPLOYMENT)
        assert "employment" in doc.full_text.lower()
        assert len(doc.full_text) > 500

    def test_parse_from_bytes(self):
        with open(SAMPLE_NDA, "rb") as f:
            file_bytes = f.read()
        doc = self.parser.parse(SAMPLE_NDA, file_bytes=file_bytes)
        assert doc.full_text

    def test_unsupported_format_raises(self):
        with pytest.raises(ValueError, match="Unsupported format"):
            self.parser.parse("document.xlsx")

    def test_file_size_validation(self):
        with pytest.raises(ValueError, match="20 MB"):
            self.parser.parse("test.txt", file_bytes=b"x" * (21 * 1024 * 1024))

    def test_empty_document_text_preserved(self):
        doc = self.parser.parse(SAMPLE_NDA)
        assert doc.full_text.strip() != ""


class TestSectionAwareChunker:
    def setup_method(self):
        self.parser = DocumentParser()
        self.chunker = SectionAwareChunker(max_chars=1800)

    def _parse(self, path):
        return self.parser.parse(path)

    def test_chunk_nda_produces_multiple_chunks(self):
        doc = self._parse(SAMPLE_NDA)
        chunks = self.chunker.chunk(doc)
        assert len(chunks) >= 2, f"Expected multiple chunks, got {len(chunks)}"

    def test_chunks_have_required_fields(self):
        doc = self._parse(SAMPLE_NDA)
        chunks = self.chunker.chunk(doc)
        for chunk in chunks:
            assert chunk.chunk_id
            assert chunk.text.strip()
            assert chunk.page_number >= 1
            assert chunk.source_filename == "sample_nda.txt"

    def test_chunk_ids_are_unique(self):
        doc = self._parse(SAMPLE_SAAS)
        chunks = self.chunker.chunk(doc)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_no_chunk_exceeds_max_chars(self):
        doc = self._parse(SAMPLE_SAAS)
        chunks = self.chunker.chunk(doc)
        for chunk in chunks:
            # Allow 20% slack for overlap
            assert len(chunk.text) <= 1800 * 1.2, (
                f"Chunk too large: {len(chunk.text)} chars in chunk {chunk.chunk_id}"
            )

    def test_all_text_covered(self):
        doc = self._parse(SAMPLE_NDA)
        chunks = self.chunker.chunk(doc)
        combined = " ".join(c.text for c in chunks)
        # Key terms from the document should appear somewhere in chunks
        assert "confidential" in combined.lower()

    def test_employment_contract_chunked(self):
        doc = self._parse(SAMPLE_EMPLOYMENT)
        chunks = self.chunker.chunk(doc)
        assert len(chunks) >= 3

    def test_empty_document_raises(self):
        from src.ingestion.parser import ParsedDocument, ParsedPage
        from src.ingestion.chunker import SectionAwareChunker
        empty_doc = ParsedDocument(
            filename="empty.txt",
            file_type="txt",
            pages=[ParsedPage(page_number=1, text="")],
            full_text="",
            total_pages=1,
        )
        with pytest.raises(ValueError, match="empty"):
            self.chunker.chunk(empty_doc)


class TestMetadataExtractor:
    def setup_method(self):
        self.parser = DocumentParser()
        self.chunker = SectionAwareChunker()
        self.extractor = MetadataExtractor()

    def _get_enriched(self, path):
        doc = self.parser.parse(path)
        chunks = self.chunker.chunk(doc)
        return self.extractor.enrich_all(chunks)

    def test_enrich_nda_detects_confidentiality(self):
        enriched = self._get_enriched(SAMPLE_NDA)
        types = {c.clause_type for c in enriched}
        assert "confidentiality" in types, f"Expected confidentiality, got: {types}"

    def test_enrich_saas_detects_indemnification(self):
        enriched = self._get_enriched(SAMPLE_SAAS)
        types = {c.clause_type for c in enriched}
        assert "indemnification" in types or "limitation_of_liability" in types, (
            f"Expected indemnification or limitation_of_liability, got: {types}"
        )

    def test_enrich_employment_detects_ip(self):
        enriched = self._get_enriched(SAMPLE_EMPLOYMENT)
        types = {c.clause_type for c in enriched}
        assert "ip_assignment" in types or "non_compete" in types, (
            f"Expected IP or non-compete detection, got: {types}"
        )

    def test_all_enriched_have_clause_type(self):
        enriched = self._get_enriched(SAMPLE_NDA)
        for chunk in enriched:
            assert chunk.clause_type, f"Empty clause_type in chunk {chunk.chunk_id}"

    def test_to_dict_has_all_fields(self):
        enriched = self._get_enriched(SAMPLE_NDA)
        d = enriched[0].to_dict()
        required = ["chunk_id", "text", "clause_type", "section_title", "page_number"]
        for field in required:
            assert field in d, f"Missing field: {field}"

    def test_confidence_score_range(self):
        enriched = self._get_enriched(SAMPLE_SAAS)
        for chunk in enriched:
            assert 0.0 <= chunk.clause_type_confidence <= 1.0
