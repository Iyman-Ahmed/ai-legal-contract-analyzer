"""
Section-aware semantic chunker for legal contracts.

Strategy: Split on clause/section boundaries first, then apply
token-size limits. Never cuts a sentence in half. Preserves
section hierarchy so each chunk knows which section it belongs to.
"""

import re
from dataclasses import dataclass, field
from typing import Optional
import logging

from src.ingestion.parser import ParsedDocument

logger = logging.getLogger(__name__)


@dataclass
class ContractChunk:
    """A single chunk from a legal contract with rich metadata."""
    chunk_id: str
    text: str
    section_title: str
    clause_number: str
    page_number: int
    chunk_index: int
    char_start: int
    char_end: int
    word_count: int
    source_filename: str


# Regex patterns for common legal section markers
_SECTION_PATTERNS = [
    # "1.", "1.1", "1.1.1", "Section 1", "ARTICLE I"
    re.compile(r"^(ARTICLE\s+[IVXLCDM]+|SECTION\s+\d+[\.\d]*|§\s*\d+[\.\d]*|\d+\.\s+[A-Z]|\d+\.\d+[\.\d]*\s)", re.MULTILINE | re.IGNORECASE),
    # "DEFINITIONS", "TERM AND TERMINATION", etc. (all-caps standalone lines)
    re.compile(r"^[A-Z][A-Z\s\-]{4,}$", re.MULTILINE),
]

_CLAUSE_NUMBER_RE = re.compile(
    r"^((?:Section|Clause|Article|§)\s*[\d\.]+|[\d]+\.[\d\.]*)",
    re.IGNORECASE,
)


class SectionAwareChunker:
    """
    Splits a parsed legal document into clause-aware chunks.

    Algorithm:
    1. Split full text at section/clause boundaries.
    2. If a section is too large (> max_chars), split on sentence
       boundaries within it, keeping sentence integrity intact.
    3. Attach metadata: section_title, clause_number, page_number.
    """

    def __init__(self, max_chars: int = 1800, overlap_chars: int = 150):
        """
        Args:
            max_chars: Approx char limit per chunk (~512 tokens at ~3.5 chars/token)
            overlap_chars: Characters of overlap between adjacent sub-chunks
        """
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def chunk(self, doc: ParsedDocument) -> list[ContractChunk]:
        """
        Chunk a ParsedDocument into ContractChunks.

        Returns:
            List of ContractChunk objects sorted by position in document.
        """
        text = doc.full_text
        if not text.strip():
            raise ValueError(f"Document '{doc.filename}' appears to be empty.")

        sections = self._split_into_sections(text)
        chunks: list[ContractChunk] = []
        global_index = 0

        for section in sections:
            section_text = section["text"].strip()
            if not section_text:
                continue

            section_title = section["title"]
            clause_number = self._extract_clause_number(section_title)

            # Estimate page number from character offset
            char_pos = section["char_start"]
            page_num = self._estimate_page(char_pos, doc)

            if len(section_text) <= self.max_chars:
                # Section fits in one chunk
                chunks.append(ContractChunk(
                    chunk_id=f"{doc.filename}__chunk_{global_index}",
                    text=section_text,
                    section_title=section_title,
                    clause_number=clause_number,
                    page_number=page_num,
                    chunk_index=global_index,
                    char_start=section["char_start"],
                    char_end=section["char_start"] + len(section_text),
                    word_count=len(section_text.split()),
                    source_filename=doc.filename,
                ))
                global_index += 1
            else:
                # Split large section on sentence boundaries
                sub_chunks = self._split_on_sentences(
                    section_text,
                    base_char_offset=section["char_start"],
                )
                for sub in sub_chunks:
                    chunks.append(ContractChunk(
                        chunk_id=f"{doc.filename}__chunk_{global_index}",
                        text=sub["text"],
                        section_title=section_title,
                        clause_number=clause_number,
                        page_number=page_num,
                        chunk_index=global_index,
                        char_start=sub["char_start"],
                        char_end=sub["char_end"],
                        word_count=len(sub["text"].split()),
                        source_filename=doc.filename,
                    ))
                    global_index += 1

        logger.info(f"Chunked '{doc.filename}' into {len(chunks)} chunks")
        return chunks

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _split_into_sections(self, text: str) -> list[dict]:
        """
        Find all section boundary positions and split text there.
        Returns list of {title, text, char_start} dicts.
        """
        # Find all matches of any section pattern
        boundary_positions: list[tuple[int, str]] = []

        for pattern in _SECTION_PATTERNS:
            for m in pattern.finditer(text):
                boundary_positions.append((m.start(), m.group().strip()))

        # Deduplicate and sort by position
        seen: set[int] = set()
        unique_boundaries: list[tuple[int, str]] = []
        for pos, title in sorted(boundary_positions):
            if pos not in seen:
                seen.add(pos)
                unique_boundaries.append((pos, title))

        if not unique_boundaries:
            # No section boundaries found — treat whole doc as one section
            return [{"title": "Document", "text": text, "char_start": 0}]

        sections: list[dict] = []

        # Text before first boundary
        if unique_boundaries[0][0] > 0:
            preamble = text[: unique_boundaries[0][0]].strip()
            if preamble:
                sections.append({"title": "Preamble", "text": preamble, "char_start": 0})

        for i, (pos, title) in enumerate(unique_boundaries):
            end = unique_boundaries[i + 1][0] if i + 1 < len(unique_boundaries) else len(text)
            section_text = text[pos:end].strip()
            # Extract section heading (first line) from text body
            lines = section_text.splitlines()
            heading = lines[0].strip() if lines else title
            body = "\n".join(lines[1:]).strip() if len(lines) > 1 else section_text

            sections.append({
                "title": heading or title,
                "text": section_text,
                "char_start": pos,
            })

        return sections

    def _split_on_sentences(
        self, text: str, base_char_offset: int = 0
    ) -> list[dict]:
        """
        Split a large text block into sentence-boundary-respecting sub-chunks.
        Uses simple sentence boundary detection (avoids NLTK dependency).
        """
        # Split on sentence endings, keeping the delimiter
        sentence_re = re.compile(r"(?<=[.!?])\s+(?=[A-Z\(\"])")
        sentences = sentence_re.split(text)

        sub_chunks: list[dict] = []
        current_text = ""
        current_start = base_char_offset
        running_offset = base_char_offset

        for sent in sentences:
            if len(current_text) + len(sent) + 1 > self.max_chars and current_text:
                sub_chunks.append({
                    "text": current_text.strip(),
                    "char_start": current_start,
                    "char_end": current_start + len(current_text),
                })
                # Overlap: carry last `overlap_chars` into next chunk
                overlap = current_text[-self.overlap_chars:] if len(current_text) > self.overlap_chars else current_text
                current_text = overlap + " " + sent
                current_start = running_offset - len(overlap)
            else:
                current_text += (" " if current_text else "") + sent

            running_offset += len(sent) + 1  # +1 for the space we split on

        if current_text.strip():
            sub_chunks.append({
                "text": current_text.strip(),
                "char_start": current_start,
                "char_end": current_start + len(current_text),
            })

        return sub_chunks

    def _extract_clause_number(self, section_title: str) -> str:
        """Extract numeric clause identifier from section title."""
        m = _CLAUSE_NUMBER_RE.match(section_title.strip())
        return m.group(0).strip() if m else ""

    def _estimate_page(self, char_offset: int, doc: ParsedDocument) -> int:
        """
        Estimate which page a character offset falls on.
        Uses cumulative character count of pages.
        """
        if not doc.pages:
            return 1
        cumulative = 0
        for page in doc.pages:
            cumulative += len(page.text)
            if char_offset <= cumulative:
                return page.page_number
        return doc.total_pages
