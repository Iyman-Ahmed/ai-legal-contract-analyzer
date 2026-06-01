"""
Build evaluation dataset from CUAD (Contract Understanding Atticus Dataset).

Downloads the 510-contract CUAD dataset from HuggingFace, extracts a curated
subset of QA pairs and contract texts, and writes:
  data/evaluation/golden_set.json   — QA pairs for RAG evaluation
  data/test_contracts/              — raw contract .txt files for upload testing

Usage:
    cd contract-analyzer/.claude/worktrees/trust-redesign
    PYTHONPATH=. python3 scripts/build_eval_dataset.py

Requires: pip install datasets huggingface_hub
"""

import json
import logging
import random
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
EVAL_DIR = ROOT / "data" / "evaluation"
TEST_CONTRACTS_DIR = ROOT / "data" / "test_contracts"
GOLDEN_SET_PATH = ROOT / "src" / "evaluation" / "golden_set.json"

EVAL_DIR.mkdir(parents=True, exist_ok=True)
TEST_CONTRACTS_DIR.mkdir(parents=True, exist_ok=True)

# CUAD defines 41 question types; we map them to our clause taxonomy
CUAD_TO_CLAUSE_TYPE = {
    "Governing Law": "governing_law",
    "Termination For Convenience": "termination",
    "Expiration Date": "termination",
    "Renewal Term": "termination",
    "Notice Period To Terminate Renewal": "termination",
    "Indemnification": "indemnification",
    "Limitation Of Liability": "limitation_of_liability",
    "Warranty Duration": "warranty",
    "IP Ownership Assignment": "ip_assignment",
    "License Grant": "ip_assignment",
    "Non-Compete": "non_compete",
    "Non-Disparagement": "confidentiality",
    "Confidentiality Duration": "confidentiality",
    "Data Breach-Notification Obligations": "data_protection",
    "Third Party Beneficiary": "general",
    "Dispute Resolution": "dispute_resolution",
    "Anti-Assignment": "general",
    "Change Of Control": "general",
    "Audit Rights": "general",
    "Most Favored Nation": "payment_terms",
    "Minimum Commitment": "payment_terms",
    "Volume Restriction": "payment_terms",
    "Price Restrictions": "payment_terms",
    "Revenue/Profit Sharing": "payment_terms",
    "Liquidated Damages": "indemnification",
    "Uncapped Liability": "limitation_of_liability",
    "Cap On Liability": "limitation_of_liability",
    "Exclusivity": "non_compete",
    "Non-Solicitation": "non_compete",
    "Source Code Escrow": "ip_assignment",
    "Post-Agreement Obligations": "general",
    "Insurance": "general",
    "Covenant Not To Sue": "dispute_resolution",
    "ERISA": "general",
    "Rofr/Rofo/Rofn": "general",
    "Affiliate License-Licensor": "ip_assignment",
    "Affiliate License-Licensee": "ip_assignment",
    "Irrevocable Or Perpetual License": "ip_assignment",
    "Sublicense": "ip_assignment",
    "Joint Ip Ownership": "ip_assignment",
    "Warranty Liability": "warranty",
}

# Questions we craft per clause type — used when CUAD answer is a yes/no
CLAUSE_QUESTIONS = {
    "governing_law": [
        "Which state or jurisdiction governs this agreement?",
        "What law applies to disputes under this contract?",
    ],
    "termination": [
        "What notice period is required to terminate this agreement?",
        "Can either party terminate this contract for convenience?",
        "When does this agreement expire or auto-renew?",
    ],
    "indemnification": [
        "Who must indemnify whom under this contract?",
        "Is there a cap on indemnification obligations?",
        "Does the indemnification clause cover third-party claims?",
    ],
    "limitation_of_liability": [
        "What is the liability cap in this contract?",
        "Are consequential damages excluded?",
        "Is there uncapped liability for any specific scenarios?",
    ],
    "ip_assignment": [
        "Who owns IP created under this agreement?",
        "Is there a license grant and if so, is it exclusive?",
        "Are there restrictions on sublicensing?",
    ],
    "confidentiality": [
        "How long does the confidentiality obligation last?",
        "What information is considered confidential under this agreement?",
    ],
    "data_protection": [
        "What are the data breach notification requirements?",
        "Does this agreement include GDPR or privacy obligations?",
    ],
    "dispute_resolution": [
        "How are disputes resolved — arbitration or litigation?",
        "Where must disputes be filed?",
    ],
    "non_compete": [
        "Is there a non-compete clause, and for how long?",
        "Does this agreement restrict solicitation of employees or customers?",
    ],
    "payment_terms": [
        "What are the payment terms and any late payment penalties?",
        "Is there a minimum commitment or volume restriction?",
    ],
    "warranty": [
        "What warranties are provided and for how long?",
        "Is there a disclaimer of warranties?",
    ],
    "force_majeure": [
        "Does this contract include a force majeure clause?",
        "What events qualify as force majeure?",
    ],
    "general": [
        "Does this contract have an anti-assignment clause?",
        "Are there change-of-control provisions?",
    ],
}


def download_cuad() -> list[dict]:
    """Download CUAD from HuggingFace and return raw examples."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.warning("'datasets' not installed — using synthetic fallback. To use CUAD: pip install datasets")
        return []

    logger.info("Downloading CUAD from HuggingFace (theatticusproject/cuad-qa)...")
    try:
        ds = load_dataset("theatticusproject/cuad-qa", split="train", trust_remote_code=True)
    except Exception:
        try:
            ds = load_dataset("cuad", split="train", trust_remote_code=True)
        except Exception as e:
            logger.warning(f"HuggingFace download failed ({e}). Using fallback synthetic dataset.")
            return []

    examples = list(ds)
    logger.info(f"Downloaded {len(examples)} CUAD examples")
    return examples


def cuad_to_golden_entries(examples: list[dict]) -> list[dict]:
    """Convert CUAD QA format to our golden set format."""
    entries: list[dict] = []
    seen_contexts: set[str] = set()

    for ex in examples:
        title = ex.get("title", "unknown")
        question = ex.get("question", "")
        context = ex.get("context", "")
        answers = ex.get("answers", {})

        # CUAD answers are {"text": [...], "answer_start": [...]}
        answer_texts = answers.get("text", [])
        answer_start = answers.get("answer_start", [])

        # Skip unanswered (yes/no absent) entries
        if not answer_texts or not any(a.strip() for a in answer_texts):
            is_present = False
            ground_truth = "Not present in this contract."
            excerpt = ""
        else:
            is_present = True
            ground_truth = answer_texts[0].strip()
            excerpt = ground_truth[:400]

        # Map question to our clause type taxonomy
        clause_type = _infer_clause_type(question)

        # Avoid duplicate contexts
        ctx_key = context[:200]
        if ctx_key in seen_contexts:
            continue
        seen_contexts.add(ctx_key)

        entries.append({
            "id": f"cuad_{len(entries):04d}",
            "source": "CUAD",
            "contract_id": title,
            "question": question,
            "ground_truth": ground_truth,
            "source_text": excerpt,
            "clause_type": clause_type,
            "is_present": is_present,
            "expected_answer_keywords": _extract_keywords(ground_truth),
            "difficulty": _estimate_difficulty(ground_truth),
            "context": context[:1500],
        })

    return entries


def _infer_clause_type(question: str) -> str:
    q_lower = question.lower()
    for label, clause_type in CUAD_TO_CLAUSE_TYPE.items():
        if label.lower() in q_lower:
            return clause_type
    return "general"


def _extract_keywords(text: str) -> list[str]:
    """Pull the 5 most distinctive words from an answer (skip stopwords)."""
    if not text:
        return []
    stopwords = {"the", "a", "an", "in", "of", "to", "and", "or", "for",
                 "is", "are", "was", "were", "this", "that", "it", "with",
                 "by", "from", "shall", "will", "may", "not", "any", "all"}
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text)
    filtered = [w.lower() for w in words if w.lower() not in stopwords]
    # Return unique, preserving order
    seen: set[str] = set()
    result: list[str] = []
    for w in filtered:
        if w not in seen:
            seen.add(w)
            result.append(w)
        if len(result) >= 5:
            break
    return result


def _estimate_difficulty(answer: str) -> str:
    if not answer or answer == "Not present in this contract.":
        return "easy"
    if len(answer) > 200:
        return "hard"
    if len(answer) > 80:
        return "medium"
    return "easy"


def build_synthetic_golden_set() -> list[dict]:
    """
    Fallback golden set built from the sample contracts already in the repo.
    Used when CUAD download is unavailable.
    Covers all major clause types with realistic QA pairs.
    """
    entries = []
    data = [
        # NDA
        {
            "id": "syn_001", "source": "synthetic", "contract_id": "sample_nda.txt",
            "question": "What is the confidentiality period in this NDA?",
            "ground_truth": "The confidentiality obligations survive for 3 years after termination of the agreement.",
            "source_text": "Confidentiality obligations shall survive termination for a period of three (3) years.",
            "clause_type": "confidentiality", "is_present": True,
            "expected_answer_keywords": ["3 years", "three", "survive", "termination"],
            "difficulty": "easy",
        },
        {
            "id": "syn_002", "source": "synthetic", "contract_id": "sample_nda.txt",
            "question": "What law governs this NDA?",
            "ground_truth": "This agreement is governed by the laws of the State of California.",
            "source_text": "This Agreement shall be governed by and construed in accordance with the laws of the State of California.",
            "clause_type": "governing_law", "is_present": True,
            "expected_answer_keywords": ["california", "governing", "laws", "state"],
            "difficulty": "easy",
        },
        {
            "id": "syn_003", "source": "synthetic", "contract_id": "sample_nda.txt",
            "question": "Does this NDA have a non-compete clause?",
            "ground_truth": "Not present in this contract.",
            "source_text": "",
            "clause_type": "non_compete", "is_present": False,
            "expected_answer_keywords": [],
            "difficulty": "easy",
        },
        # SaaS Agreement
        {
            "id": "syn_004", "source": "synthetic", "contract_id": "sample_saas_contract.txt",
            "question": "What is the liability cap in the SaaS agreement?",
            "ground_truth": "Vendor's total liability is capped at the fees paid by Customer in the twelve months preceding the claim.",
            "source_text": "In no event shall Vendor's aggregate liability exceed the total fees paid by Customer in the twelve (12) months prior to the event giving rise to liability.",
            "clause_type": "limitation_of_liability", "is_present": True,
            "expected_answer_keywords": ["twelve months", "fees paid", "aggregate", "liability"],
            "difficulty": "medium",
        },
        {
            "id": "syn_005", "source": "synthetic", "contract_id": "sample_saas_contract.txt",
            "question": "Who is responsible for indemnification and is there a cap?",
            "ground_truth": "Customer indemnifies Vendor for third-party IP infringement claims arising from Customer's use. No cap is specified.",
            "source_text": "Customer shall indemnify, defend and hold harmless Vendor against any third-party claim alleging that Customer's use of the Service infringes any intellectual property right.",
            "clause_type": "indemnification", "is_present": True,
            "expected_answer_keywords": ["customer", "indemnify", "third-party", "intellectual property"],
            "difficulty": "hard",
        },
        {
            "id": "syn_006", "source": "synthetic", "contract_id": "sample_saas_contract.txt",
            "question": "What are the payment terms and late payment penalties?",
            "ground_truth": "Invoices are due Net-30. Late payments accrue interest at 1.5% per month.",
            "source_text": "Customer shall pay all undisputed invoices within thirty (30) days of receipt. Overdue amounts shall accrue interest at 1.5% per month.",
            "clause_type": "payment_terms", "is_present": True,
            "expected_answer_keywords": ["30 days", "net-30", "1.5%", "interest", "overdue"],
            "difficulty": "easy",
        },
        {
            "id": "syn_007", "source": "synthetic", "contract_id": "sample_saas_contract.txt",
            "question": "Can the contract be terminated for convenience and with what notice?",
            "ground_truth": "Either party may terminate with 30 days written notice for convenience.",
            "source_text": "Either party may terminate this Agreement for any reason upon thirty (30) days prior written notice to the other party.",
            "clause_type": "termination", "is_present": True,
            "expected_answer_keywords": ["terminate", "30 days", "written notice", "convenience"],
            "difficulty": "easy",
        },
        {
            "id": "syn_008", "source": "synthetic", "contract_id": "sample_saas_contract.txt",
            "question": "Does the SaaS agreement include a force majeure clause?",
            "ground_truth": "Not present in this contract.",
            "source_text": "",
            "clause_type": "force_majeure", "is_present": False,
            "expected_answer_keywords": [],
            "difficulty": "easy",
        },
        # Employment Contract
        {
            "id": "syn_009", "source": "synthetic", "contract_id": "sample_employment.txt",
            "question": "Who owns IP created by the employee during employment?",
            "ground_truth": "The employer owns all IP created by the employee in the scope of employment, including work done outside business hours using company resources.",
            "source_text": "Employee hereby assigns to Company all right, title and interest in any inventions, works of authorship, or other IP created during employment, whether during or outside business hours, if using Company resources or related to Company business.",
            "clause_type": "ip_assignment", "is_present": True,
            "expected_answer_keywords": ["employer", "company", "assigns", "inventions", "ip"],
            "difficulty": "hard",
        },
        {
            "id": "syn_010", "source": "synthetic", "contract_id": "sample_employment.txt",
            "question": "Is there a non-compete and for how long does it last?",
            "ground_truth": "Employee may not work for a direct competitor for 12 months after termination within the same geographic market.",
            "source_text": "For a period of twelve (12) months following termination of employment, Employee shall not engage in or provide services to any direct competitor within the same geographic market.",
            "clause_type": "non_compete", "is_present": True,
            "expected_answer_keywords": ["12 months", "twelve", "competitor", "non-compete", "termination"],
            "difficulty": "medium",
        },
        {
            "id": "syn_011", "source": "synthetic", "contract_id": "sample_employment.txt",
            "question": "What dispute resolution mechanism applies to this employment contract?",
            "ground_truth": "Disputes must be resolved through binding arbitration under AAA rules in San Francisco, California.",
            "source_text": "Any dispute arising under this Agreement shall be resolved by binding arbitration administered by the American Arbitration Association (AAA) in San Francisco, California.",
            "clause_type": "dispute_resolution", "is_present": True,
            "expected_answer_keywords": ["arbitration", "aaa", "san francisco", "binding"],
            "difficulty": "medium",
        },
        # Cross-clause contradiction test
        {
            "id": "syn_012", "source": "synthetic", "contract_id": "sample_saas_contract.txt",
            "question": "Is there a contradiction between the liability cap and the indemnification clause?",
            "ground_truth": "Yes — Section 3 caps liability at 12 months fees but Section 9 requires unlimited indemnification including for Vendor negligence, which effectively overrides the cap.",
            "source_text": "Liability cap: fees paid in 12 months. Indemnification: unlimited, no carve-outs.",
            "clause_type": "indemnification", "is_present": True,
            "expected_answer_keywords": ["contradiction", "cap", "unlimited", "indemnification", "override"],
            "difficulty": "hard",
        },
        # Obligation extraction tests
        {
            "id": "syn_013", "source": "synthetic", "contract_id": "sample_saas_contract.txt",
            "question": "What must happen 30 days before the contract auto-renews?",
            "ground_truth": "Either party must deliver written notice of non-renewal to prevent automatic renewal.",
            "source_text": "This Agreement will automatically renew unless either party provides written notice of non-renewal at least thirty (30) days prior to the end of the then-current term.",
            "clause_type": "termination", "is_present": True,
            "expected_answer_keywords": ["written notice", "non-renewal", "30 days", "automatic renewal"],
            "difficulty": "medium",
        },
        {
            "id": "syn_014", "source": "synthetic", "contract_id": "sample_nda.txt",
            "question": "Are there any data breach notification obligations?",
            "ground_truth": "Not present in this contract.",
            "source_text": "",
            "clause_type": "data_protection", "is_present": False,
            "expected_answer_keywords": [],
            "difficulty": "easy",
        },
        {
            "id": "syn_015", "source": "synthetic", "contract_id": "sample_saas_contract.txt",
            "question": "What warranties does the vendor provide?",
            "ground_truth": "Vendor warrants the Service will perform materially as described in documentation for 90 days. No other warranties.",
            "source_text": "Vendor warrants that the Service will perform materially in accordance with the applicable Documentation for ninety (90) days following delivery. EXCEPT AS EXPRESSLY SET FORTH HEREIN, VENDOR MAKES NO WARRANTIES.",
            "clause_type": "warranty", "is_present": True,
            "expected_answer_keywords": ["90 days", "materially", "documentation", "warranty"],
            "difficulty": "medium",
        },
    ]
    return data


def write_test_contracts() -> None:
    """Write synthetic test contracts as .txt files for upload testing."""
    contracts = {
        "sample_nda.txt": """MUTUAL NON-DISCLOSURE AGREEMENT

This Mutual Non-Disclosure Agreement ("Agreement") is entered into as of January 1, 2025,
between Acme Corp ("Company A") and Beta Inc ("Company B").

1. CONFIDENTIAL INFORMATION
Each party may disclose to the other certain proprietary and confidential information
("Confidential Information") in connection with evaluating a potential business relationship.

2. OBLIGATIONS
Each party agrees to: (a) hold the other's Confidential Information in strict confidence;
(b) not disclose it to third parties without prior written consent; (c) use it solely
for evaluating the potential business relationship.

3. TERM AND SURVIVAL
This Agreement shall remain in effect for two (2) years from the date hereof.
Confidentiality obligations shall survive termination for a period of three (3) years.

4. GOVERNING LAW
This Agreement shall be governed by and construed in accordance with the laws of
the State of California, without regard to conflict of law principles.

5. RETURN OF INFORMATION
Upon request, each party shall promptly return or destroy all Confidential Information.

6. NO WARRANTY
Nothing in this Agreement grants either party any rights in the other's Confidential
Information except as expressly set forth herein.

IN WITNESS WHEREOF, the parties have executed this Agreement as of the date first written above.
""",
        "sample_saas_contract.txt": """SOFTWARE AS A SERVICE AGREEMENT

This SaaS Agreement ("Agreement") is made as of March 1, 2025, between SaaSCo Inc
("Vendor") and Enterprise LLC ("Customer").

1. SERVICES
Vendor will provide Customer access to the SaaS platform ("Service") during the Term.

2. PAYMENT TERMS
Customer shall pay all undisputed invoices within thirty (30) days of receipt.
Overdue amounts shall accrue interest at 1.5% per month or the maximum rate
permitted by law, whichever is lower.

3. LIMITATION OF LIABILITY
IN NO EVENT SHALL VENDOR'S AGGREGATE LIABILITY EXCEED THE TOTAL FEES PAID BY
CUSTOMER IN THE TWELVE (12) MONTHS PRIOR TO THE EVENT GIVING RISE TO LIABILITY.
NEITHER PARTY SHALL BE LIABLE FOR INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES.

4. INDEMNIFICATION
Customer shall indemnify, defend and hold harmless Vendor and its officers from
any and all third-party claims, damages, losses, and expenses (including reasonable
attorneys' fees) arising out of Customer's use of the Service or breach of this
Agreement, with no limitation on the amount of such indemnification.

5. TERM AND TERMINATION
This Agreement commences on the Effective Date and continues for one (1) year.
It will automatically renew unless either party provides written notice of
non-renewal at least thirty (30) days prior to the end of the then-current term.
Either party may terminate this Agreement for any reason upon thirty (30) days
prior written notice.

6. WARRANTIES
Vendor warrants that the Service will perform materially in accordance with the
applicable Documentation for ninety (90) days following delivery.
EXCEPT AS EXPRESSLY SET FORTH HEREIN, VENDOR MAKES NO WARRANTIES, EXPRESS OR IMPLIED.

7. CONFIDENTIALITY
Each party shall maintain the other's Confidential Information in strict confidence
and not disclose it to third parties. Obligations survive termination for 3 years.

8. GOVERNING LAW
This Agreement is governed by the laws of the State of New York.
Disputes shall be resolved in the state or federal courts located in New York County.

9. ASSIGNMENT
Neither party may assign this Agreement without the other's prior written consent,
except to an affiliate or in connection with a merger or acquisition.
""",
        "sample_employment.txt": """EMPLOYMENT AGREEMENT

This Employment Agreement ("Agreement") is entered into as of June 1, 2025,
between TechStartup Inc ("Company") and Jane Smith ("Employee").

1. POSITION AND DUTIES
Employee shall serve as Senior Software Engineer, reporting to the CTO.

2. COMPENSATION
Base salary: $150,000 per year, payable bi-weekly.
Employee is eligible for an annual discretionary bonus of up to 15% of base salary.

3. IP ASSIGNMENT
Employee hereby assigns to Company all right, title and interest in any inventions,
works of authorship, software, or other intellectual property created during
employment, whether during or outside business hours, if using Company resources
or if related to Company's current or contemplated business.

4. CONFIDENTIALITY
Employee shall maintain the confidentiality of all Company Confidential Information
during and for five (5) years following termination of employment.

5. NON-COMPETE
For a period of twelve (12) months following termination of employment, Employee
shall not engage in or provide services to any direct competitor of Company
within the same geographic market where Company operates.

6. NON-SOLICITATION
For twelve (12) months post-termination, Employee shall not solicit Company's
customers or employees for any competing business.

7. DISPUTE RESOLUTION
Any dispute arising under this Agreement shall be resolved by binding arbitration
administered by the American Arbitration Association (AAA) in San Francisco,
California. The decision of the arbitrator shall be final and binding.

8. GOVERNING LAW
This Agreement is governed by the laws of the State of California.

9. AT-WILL EMPLOYMENT
Employment is at-will and may be terminated by either party at any time with or
without cause, subject to two (2) weeks written notice.
""",
    }

    for filename, content in contracts.items():
        path = TEST_CONTRACTS_DIR / filename
        path.write_text(content, encoding="utf-8")
        logger.info(f"Wrote test contract: {path.name} ({len(content)} chars)")


def main():
    logger.info("Building evaluation dataset...")

    # 1. Write synthetic test contracts
    write_test_contracts()
    logger.info(f"Test contracts written to {TEST_CONTRACTS_DIR}/")

    # 2. Try to download CUAD; fall back to synthetic golden set
    cuad_examples = download_cuad()
    if cuad_examples:
        all_entries = cuad_to_golden_entries(cuad_examples)
        # Sample evenly across clause types — max 5 per type
        by_type: dict[str, list[dict]] = {}
        for e in all_entries:
            by_type.setdefault(e["clause_type"], []).append(e)

        sampled: list[dict] = []
        random.seed(42)
        for clause_type, items in by_type.items():
            sampled.extend(random.sample(items, min(5, len(items))))

        golden = sampled
        logger.info(f"Built {len(golden)} golden entries from CUAD across {len(by_type)} clause types")
    else:
        golden = build_synthetic_golden_set()
        logger.info(f"Built {len(golden)} synthetic golden entries (CUAD unavailable)")

    # 3. Write golden set
    GOLDEN_SET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GOLDEN_SET_PATH, "w") as f:
        json.dump(golden, f, indent=2)
    logger.info(f"Golden set written to {GOLDEN_SET_PATH} ({len(golden)} entries)")

    # Also write a copy to data/evaluation/ for reference
    eval_copy = EVAL_DIR / "golden_set.json"
    with open(eval_copy, "w") as f:
        json.dump(golden, f, indent=2)

    # 4. Print summary
    clause_counts: dict[str, int] = {}
    present_count = sum(1 for e in golden if e.get("is_present", True))
    for e in golden:
        clause_counts[e["clause_type"]] = clause_counts.get(e["clause_type"], 0) + 1

    print("\n── Evaluation Dataset Summary ──────────────────────────")
    print(f"  Total QA pairs  : {len(golden)}")
    print(f"  Clause present  : {present_count}")
    print(f"  Clause absent   : {len(golden) - present_count}")
    print(f"  Clause types    : {len(clause_counts)}")
    print(f"  Difficulty split: {dict((d, sum(1 for e in golden if e.get('difficulty') == d)) for d in ['easy','medium','hard'])}")
    print("\n  By clause type:")
    for ct, n in sorted(clause_counts.items(), key=lambda x: -x[1]):
        print(f"    {ct:<30} {n}")
    print(f"\n  Files written to:")
    print(f"    {GOLDEN_SET_PATH}")
    print(f"    {TEST_CONTRACTS_DIR}/")
    print("─────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
