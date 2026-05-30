# Trust Redesign — Phase Tracker

Worktree branch: `feat/trust-redesign`
Base: `main` @ 52d7829

## Why this branch exists
Research showed that lawyer-grade trust requires a self-verifying multi-agent pipeline, not a
single LLM pass per clause. Each phase below is one atomic commit — easy to revert, bisect, or
cherry-pick back to main independently.

---

## Phase 1 — LLM-as-judge Verification Agent ✅ / ⬜
**Commit prefix:** `feat(agents): `
**Files changed:**
- `src/agents/__init__.py` (new)
- `src/agents/verification.py` (new)
- `src/guardrails/faithfulness.py` (replaced token-overlap with LLM judge)
- `src/analysis/risk_engine.py` (call verification after each clause)

**What it does:**
After every clause analysis the verification agent asks the LLM:
"Is every claim in this output directly supported by the retrieved context below?
List any unsupported claims and return a faithfulness score 0–1."
If score < 0.75 → re-retrieve with a rephrased query and re-analyze (max 1 retry).
This is agentic RAG self-check — the pattern that separates production from demo.

**Test:** `pytest tests/test_guardrails.py -v`

---

## Phase 2 — Obligation Extractor Agent ⬜
**Commit prefix:** `feat(agents): `
**Files changed:**
- `src/agents/obligation.py` (new)
- `src/analysis/schemas.py` (add ObligationTable, ObligationItem)
- `src/analysis/risk_engine.py` (run obligation agent post-analysis)
- `app.py` (add obligation table section to UI output)

**What it does:**
Dedicated LLM pass with a targeted schema extracting every:
- Deadline / notice period (what, when, which party, consequence-if-missed)
- Payment obligation (amount, trigger, late penalty)
- Renewal condition (auto-renew date, opt-out window)
- Termination trigger (condition, notice required)

Lawyers use this table daily — it's the highest practical-value output.

**Test:** `pytest tests/test_analysis.py -v`

---

## Phase 3 — Cross-Clause Contradiction Detector ⬜
**Commit prefix:** `feat(agents): `
**Files changed:**
- `src/agents/contradiction.py` (new)
- `src/analysis/schemas.py` (add ContradictionFinding)
- `src/analysis/risk_engine.py` (run contradiction agent over full clause set)
- `app.py` (show contradictions in UI)

**What it does:**
After all clauses are individually analyzed, feeds the full structured output to one
LLM call asking it to find logical contradictions between clauses
(e.g. liability cap in §3 vs. unlimited indemnification carve-out in §9).
Current system is blind to these because it analyzes each clause in isolation.

**Test:** `pytest tests/test_analysis.py -v`

---

## Phase 4 — Audit Trail Logger ⬜
**Commit prefix:** `feat(audit): `
**Files changed:**
- `src/audit/__init__.py` (new)
- `src/audit/logger.py` (new)
- `src/analysis/risk_engine.py` (emit audit events)
- `app.py` (add audit log download button)

**What it does:**
Every analysis emits a structured JSON log:
- clause_text → query sent → chunks retrieved (with scores) → prompt → raw LLM response → validation result → final output
ABA Formal Opinion 512 effectively requires this for firm supervision accountability.
Also critical for debugging when the pipeline produces a wrong answer.

**Test:** manual — run analysis, check audit log file

---

## Phase 5 — Wire Everything + UI Updates ⬜
**Commit prefix:** `feat(ui): `
**Files changed:**
- `app.py` (full wiring, new UI sections)
- `src/analysis/risk_engine.py` (orchestrator refactor)

**What it does:**
- Connect all new agents into the main pipeline
- Add Obligation Table tab to Gradio UI
- Add Contradictions section to analysis output
- Add Audit Log download button
- Add verification badge ("verified by LLM judge") per clause

**Test:** run `python app.py`, manually upload sample contract, verify all sections appear

---

## How to use this file
- Mark each phase ✅ when its commit is made
- If a phase breaks tests, note the error here before reverting
- Each phase can be cherry-picked to main independently: `git cherry-pick <sha>`
