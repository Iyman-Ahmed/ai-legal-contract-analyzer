"""
Audit trail logger for legal contract analysis.

Emits one newline-delimited JSON record per analysis event to an in-session
log buffer (and optionally to a file).  ABA Formal Opinion 512 requires law
firms to be able to reconstruct how an AI system arrived at its output;
this log provides the evidence chain:

  clause_text → query → chunks retrieved (text + score)
              → prompt tokens → raw LLM response
              → verification score → final output

Usage (from RiskAnalysisEngine):
    self.audit.log_clause(clause_type, query, retrieved, raw_llm, result, verification)
    self.audit.log_summary(filename, summary)
    log_path = self.audit.flush(output_dir)   # write to disk, returns Path
"""

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class AuditLogger:
    """Thread-safe, in-memory audit log with optional disk flush."""

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self._events: list[dict] = []
        self._lock = threading.Lock()

    # ── Public emit methods ───────────────────────────────────────────────────

    def log_clause(
        self,
        clause_type: str,
        clause_text: str,
        retrieval_query: str,
        retrieved_chunks: list[dict],
        raw_llm_response: str,
        verification_score: Optional[float],
        final_risk_level: str,
        confidence: float,
    ) -> None:
        self._emit("clause_analysis", {
            "clause_type": clause_type,
            "clause_text_preview": clause_text[:200],
            "retrieval_query_preview": retrieval_query[:200],
            "retrieved_chunks": [
                {
                    "text_preview": r.get("text", "")[:150],
                    "rerank_score": round(r.get("rerank_score", 0.0), 4),
                    "source": r.get("metadata", {}).get("source_filename", "unknown"),
                    "clause_type": r.get("metadata", {}).get("clause_type", "general"),
                }
                for r in retrieved_chunks
            ],
            "raw_llm_response_preview": raw_llm_response[:400],
            "verification_score": verification_score,
            "final_risk_level": final_risk_level,
            "confidence": confidence,
        })

    def log_obligation(self, obligation_count: int, high_priority: int, note: str) -> None:
        self._emit("obligation_extraction", {
            "obligations_extracted": obligation_count,
            "high_priority_count": high_priority,
            "note": note,
        })

    def log_contradiction(self, contradictions_found: int, has_critical: bool) -> None:
        self._emit("contradiction_detection", {
            "contradictions_found": contradictions_found,
            "has_critical": has_critical,
        })

    def log_summary(self, filename: str, overall_risk: str, risk_score: float) -> None:
        self._emit("document_summary", {
            "filename": filename,
            "overall_risk_level": overall_risk,
            "overall_risk_score": risk_score,
        })

    def log_chat(
        self,
        question: str,
        routed_clause_type: Optional[str],
        category_filter_applied: bool,
        retrieved_count: int,
        answer_preview: str,
        confidence: float,
    ) -> None:
        self._emit("chat_query", {
            "question_preview": question[:200],
            "routed_clause_type": routed_clause_type,
            "category_filter_applied": category_filter_applied,
            "retrieved_count": retrieved_count,
            "answer_preview": answer_preview[:300],
            "confidence": confidence,
        })

    # ── Flush to disk ─────────────────────────────────────────────────────────

    def flush(self, output_dir: Optional[Path] = None) -> Path:
        """Write all buffered events to a NDJSON file. Returns the file path."""
        output_dir = output_dir or Path("data/audit_logs")
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"audit_{self.session_id}.ndjson"
        with self._lock:
            with open(path, "w", encoding="utf-8") as f:
                for event in self._events:
                    f.write(json.dumps(event, ensure_ascii=False) + "\n")
        logger.info(f"Audit log written: {path} ({len(self._events)} events)")
        return path

    def to_json_string(self) -> str:
        """Return buffered events as a pretty-printed JSON array (for UI download)."""
        with self._lock:
            return json.dumps(self._events, indent=2, ensure_ascii=False)

    def event_count(self) -> int:
        with self._lock:
            return len(self._events)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _emit(self, event_type: str, payload: dict) -> None:
        event = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "session": self.session_id,
            "event": event_type,
            **payload,
        }
        with self._lock:
            self._events.append(event)
