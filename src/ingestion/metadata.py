"""
Clause type classifier and metadata extractor.

Assigns each chunk a clause_type from the canonical list using
keyword-based heuristics. Fast and CPU-free — no ML inference needed
for this step since legal clause vocabulary is highly specialized.
"""

import re
from dataclasses import dataclass
import logging

from src.ingestion.chunker import ContractChunk

logger = logging.getLogger(__name__)


# Keyword signals per clause type — tuples of (pattern, weight)
_CLAUSE_SIGNALS: dict[str, list[tuple[str, float]]] = {
    "indemnification": [
        (r"\bindemnif", 3.0),
        (r"\bhold harmless\b", 2.5),
        (r"\bdefend\b.*\bclaim", 1.5),
        (r"\blosses.*arising", 1.0),
    ],
    "limitation_of_liability": [
        (r"\blimit.*liabilit", 3.0),
        (r"\bin no event.*liable", 3.0),
        (r"\bexclude.*consequential", 2.0),
        (r"\bcap.*liabilit", 2.5),
        (r"\bmaximum.*liabilit", 2.0),
        (r"\bliabilit.*shall not exceed", 3.0),
        (r"\btotal.*liabilit.*exceed", 2.5),
        (r"\bliabilit.*not exceed", 2.5),
        (r"\bliabilit.*unlimited\b", 2.5),
        (r"\bconsequential.*damages", 2.0),
        (r"\bindirect.*damages", 2.0),
        (r"\bpunitive.*damages", 1.5),
        (r"\bsole remedy.*terminat", 1.5),
    ],
    "termination": [
        (r"\bterminat", 2.5),
        (r"\bexpir", 1.5),
        (r"\bcancel", 1.5),
        (r"\bending.*agreement", 1.5),
        (r"\bnotice.*terminat", 2.0),
        (r"\bminimum.*term\b", 2.0),
        (r"\binitial.*term\b", 2.0),
        (r"\bterm.*of.*this.*agreement", 2.0),
        (r"\bauto.*renew", 2.0),
        (r"\bcure.*period", 2.0),
        (r"\nearly.*terminat", 2.5),
        (r"\btermination.*fee", 2.5),
        (r"\btermination.*notice", 2.0),
        (r"\bsuspend.*service", 1.5),
    ],
    "ip_assignment": [
        (r"\bintellectual property\b", 2.0),
        (r"\bassign.*right", 2.0),
        (r"\bwork.*for.*hire\b", 3.0),
        (r"\bowner.*inventions\b", 2.5),
        (r"\bip\s+right", 2.0),
        (r"\bpatent.*assign", 2.0),
        (r"\birrevocably.*assign", 3.0),
        (r"\bassign.*inventions", 3.0),
        (r"\binventions.*discoveries", 2.5),
        (r"\bworks.*of.*authorship", 2.5),
        (r"\bip.*assignment\b", 3.0),
        (r"\bmoral.*right", 2.0),
        (r"\bbackground.*ip\b", 2.5),
        (r"\bwork.*product\b", 2.0),
    ],
    "non_compete": [
        (r"\bnon.?compet", 3.0),
        (r"\brestrictive covenant\b", 2.5),
        (r"\bsolicit.*employee", 2.0),
        (r"\bnon.?solicit", 2.5),
        (r"\brestraint.*trade", 2.0),
    ],
    "confidentiality": [
        (r"\bconfidential", 2.5),
        (r"\bnon.?disclos", 3.0),
        (r"\bproprietary.*information\b", 2.0),
        (r"\btrade secret", 2.5),
        (r"\bnda\b", 2.0),
    ],
    "governing_law": [
        (r"\bgoverning law\b", 3.0),
        (r"\bchosen.*law\b", 2.0),
        (r"\bjurisdiction.*govern", 2.5),
        (r"\bapplicable law\b", 2.0),
        (r"\blaw.*state.*california|new york|delaware", 2.0),
    ],
    "dispute_resolution": [
        (r"\barbitrat", 3.0),
        (r"\bmediat", 2.0),
        (r"\bdispute.*resolut", 2.5),
        (r"\bjurisdiction.*court", 2.0),
        (r"\bforum.*selection\b", 2.0),
        (r"\bclass.*action.*waiv", 2.5),
    ],
    "data_protection": [
        (r"\bpersonal.*data\b", 2.5),
        (r"\bgdpr\b", 3.0),
        (r"\bccpa\b", 3.0),
        (r"\bdata.*processing", 2.5),
        (r"\bprivacy", 2.0),
        (r"\bdata.*breach", 2.5),
        (r"\bcustomer.*data\b", 2.0),
        (r"\bsell.*customer.*data", 3.0),
        (r"\buse.*data.*any.*purpose", 2.5),
        (r"\bdata.*third.?party", 2.0),
        (r"\bdata.*license\b", 2.0),
        (r"\bdata.*retain", 1.5),
        (r"\bdata.*delete\b", 2.0),
        (r"\bdata.*security", 2.0),
        (r"\btechnical.*organizational.*measures", 2.5),
    ],
    "force_majeure": [
        (r"\bforce majeure\b", 3.0),
        (r"\bact of god\b", 3.0),
        (r"\bbeyond.*control\b", 2.0),
        (r"\bnatural disaster\b", 2.0),
        (r"\bpandemic.*performance", 1.5),
    ],
    "payment_terms": [
        (r"\bpayment\b", 1.5),
        (r"\binvoice\b", 2.0),
        (r"\bfee.*due\b", 2.0),
        (r"\bnet [36]0\b", 2.5),
        (r"\bsubscription.*fee\b", 2.0),
        (r"\bpricing\b", 1.5),
        (r"\brefund\b", 1.5),
    ],
    "warranty": [
        (r"\bwarrant", 2.5),
        (r"\brepresent.*warrant", 2.5),
        (r"\bas.?is\b", 3.0),
        (r"\bno.*warrant", 2.5),
        (r"\bfit.*purpose", 2.0),
    ],
    "representations": [
        (r"\brepresent", 2.0),
        (r"\bcovenant", 1.5),
        (r"\backnowledg", 1.5),
        (r"\bcertif", 1.5),
    ],
}

# Compiled patterns for speed
_COMPILED_SIGNALS: dict[str, list[tuple[re.Pattern, float]]] = {
    clause_type: [(re.compile(pattern, re.IGNORECASE), weight) for pattern, weight in signals]
    for clause_type, signals in _CLAUSE_SIGNALS.items()
}


@dataclass
class EnrichedChunk:
    """ContractChunk + clause_type and risk signals."""
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
    clause_type: str
    clause_type_confidence: float
    detected_signals: list[str]

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "section_title": self.section_title,
            "clause_number": self.clause_number,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "word_count": self.word_count,
            "source_filename": self.source_filename,
            "clause_type": self.clause_type,
            "clause_type_confidence": self.clause_type_confidence,
            "detected_signals": self.detected_signals,
        }


class MetadataExtractor:
    """
    Enriches ContractChunks with clause_type classification
    and additional metadata signals.
    """

    def enrich(self, chunk: ContractChunk) -> EnrichedChunk:
        """
        Classify a chunk's clause type and extract metadata.

        Returns:
            EnrichedChunk with clause_type and confidence score.
        """
        clause_type, confidence, signals = self._classify_clause_type(
            chunk.section_title + " " + chunk.text
        )

        return EnrichedChunk(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            section_title=chunk.section_title,
            clause_number=chunk.clause_number,
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
            word_count=chunk.word_count,
            source_filename=chunk.source_filename,
            clause_type=clause_type,
            clause_type_confidence=confidence,
            detected_signals=signals,
        )

    def enrich_all(self, chunks: list[ContractChunk]) -> list[EnrichedChunk]:
        """Enrich a list of chunks."""
        return [self.enrich(c) for c in chunks]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _classify_clause_type(
        self, text: str
    ) -> tuple[str, float, list[str]]:
        """
        Score text against all clause type signals and return best match.

        Returns:
            (clause_type, confidence_0_to_1, matched_signal_descriptions)
        """
        scores: dict[str, float] = {}
        matched_signals: dict[str, list[str]] = {}

        for clause_type, compiled in _COMPILED_SIGNALS.items():
            score = 0.0
            signals_found: list[str] = []
            for pattern, weight in compiled:
                if pattern.search(text):
                    score += weight
                    signals_found.append(pattern.pattern)
            scores[clause_type] = score
            matched_signals[clause_type] = signals_found

        best_type = max(scores, key=lambda k: scores[k])
        best_score = scores[best_type]

        if best_score == 0.0:
            return "general", 0.0, []

        # Normalize to 0-1 by treating 5.0 as "very confident"
        confidence = min(best_score / 5.0, 1.0)

        return best_type, round(confidence, 3), matched_signals[best_type]
