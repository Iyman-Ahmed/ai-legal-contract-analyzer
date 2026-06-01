"""
Obligation extractor agent.

Runs a single targeted LLM pass over all contract chunks to extract every
concrete obligation, deadline, notice period, payment term, and renewal
trigger into a structured table.

Why a dedicated agent instead of bundling this into risk analysis?
The risk analysis prompt is already complex (clause comparison + scoring).
Adding obligation extraction there degrades both outputs. A focused prompt
with a clear schema produces a complete, accurate table that lawyers can
action immediately — this is one of the highest-value outputs for daily use.
"""

import json
import logging
import re
from typing import TYPE_CHECKING

from src.analysis.schemas import ObligationItem, ObligationTable
from src.analysis.prompts import OBLIGATION_SYSTEM, OBLIGATION_USER

if TYPE_CHECKING:
    from src.analysis.risk_engine import LLMClient
    from src.ingestion.metadata import EnrichedChunk

logger = logging.getLogger(__name__)

# Process this many chars of contract text per LLM call to stay within context limits
_CHUNK_CHAR_LIMIT = 3000
# Maximum total chars to analyze (avoids runaway cost on huge contracts)
_MAX_TOTAL_CHARS = 15000


class ObligationAgent:
    """
    Extracts all actionable obligations from a contract.

    Processes contract text in batches, merges results, and returns an
    ObligationTable with deduplication on identical obligation text.

    Usage:
        agent = ObligationAgent(llm_client)
        table = agent.extract(enriched_chunks)
    """

    def __init__(self, llm: "LLMClient"):
        self.llm = llm

    def extract(self, chunks: list["EnrichedChunk"]) -> ObligationTable:
        """
        Extract all obligations from contract chunks.

        Batches chunks into context-window-sized groups, calls the LLM
        for each batch, and merges results.
        """
        # Build ordered text from chunks, stopping at the char limit
        full_text = "\n\n".join(
            f"[{c.section_title}]\n{c.text}" for c in chunks
        )[:_MAX_TOTAL_CHARS]

        if not full_text.strip():
            return ObligationTable(
                obligations=[],
                extraction_note="No contract text available.",
            )

        batches = _split_into_batches(full_text, _CHUNK_CHAR_LIMIT)
        all_items: list[ObligationItem] = []

        for i, batch in enumerate(batches):
            items = self._extract_batch(batch, batch_index=i, total=len(batches))
            all_items.extend(items)

        deduped = _deduplicate(all_items)
        table = ObligationTable.from_items(deduped)

        if len(full_text) >= _MAX_TOTAL_CHARS:
            table.extraction_note = (
                f"Contract truncated to {_MAX_TOTAL_CHARS} chars for extraction. "
                "Review full document for additional obligations."
            )

        logger.info(
            f"Obligation extraction complete: {len(deduped)} obligations "
            f"({table.high_priority_count} high-priority)"
        )
        return table

    def _extract_batch(
        self,
        text: str,
        batch_index: int,
        total: int,
    ) -> list[ObligationItem]:
        try:
            prompt = OBLIGATION_USER.format(contract_text=text)
            raw = self.llm.chat(
                system=OBLIGATION_SYSTEM,
                user=prompt,
                max_tokens=1500,
            )
            data = _parse_json_array(raw)
            items = []
            for entry in data:
                try:
                    items.append(ObligationItem(**entry))
                except Exception as e:
                    logger.debug(f"Skipping malformed obligation entry: {e}")
            return items

        except Exception as e:
            logger.warning(
                f"Obligation extraction batch {batch_index+1}/{total} failed: {e}"
            )
            return []


def _split_into_batches(text: str, chunk_size: int) -> list[str]:
    """Split text into overlapping batches that respect paragraph boundaries."""
    if len(text) <= chunk_size:
        return [text]

    batches = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Try to break at a paragraph boundary
        if end < len(text):
            newline = text.rfind("\n\n", start, end)
            if newline > start + chunk_size // 2:
                end = newline
        batches.append(text[start:end])
        start = end
    return batches


def _deduplicate(items: list[ObligationItem]) -> list[ObligationItem]:
    """Remove obligations with identical obligation text (case-insensitive)."""
    seen: set[str] = set()
    result: list[ObligationItem] = []
    for item in items:
        key = item.obligation.lower().strip()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _parse_json_array(text: str) -> list[dict]:
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    try:
        result = json.loads(text)
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end > start:
            try:
                result = json.loads(text[start : end + 1])
                return result if isinstance(result, list) else []
            except json.JSONDecodeError:
                pass
    logger.debug(f"Could not parse obligation JSON: {text[:200]}")
    return []
