"""
Test dataset generator — creates 50 realistic legal contracts.

Distribution:
  12 NDAs          (4 LOW-risk mutual, 4 MEDIUM one-sided, 4 HIGH-risk / missing clauses)
  15 SaaS          (5 LOW standard, 5 MEDIUM, 5 HIGH/CRITICAL problematic)
  12 Employment    (4 LOW CA-compliant, 4 MEDIUM aggressive, 4 HIGH illegal non-competes)
   7 Service Agmt  (4 LOW balanced, 3 HIGH risky)
   4 Lease         (2 LOW standard, 2 HIGH tenant-unfriendly)

Each contract is written to data/test_contracts/<id>.txt
Ground truth written to data/test_contracts/manifest.json
"""

import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE / "data" / "test_contracts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# CLAUSE LIBRARY  (LOW / MEDIUM / HIGH variants for each clause type)
# ─────────────────────────────────────────────────────────────────────────────

CLAUSES = {

# ── INDEMNIFICATION ──────────────────────────────────────────────────────────
"indemnification_LOW": """\
INDEMNIFICATION

Each party ("Indemnifying Party") shall defend, indemnify, and hold harmless the other party \
and its officers, directors, and employees from and against any third-party claims arising out \
of the Indemnifying Party's gross negligence, willful misconduct, or material breach of this \
Agreement. Each party's total indemnification obligation shall not exceed the greater of (i) \
the fees paid in the preceding twelve (12) months or (ii) USD $100,000.""",

"indemnification_MEDIUM": """\
INDEMNIFICATION

Customer shall indemnify, defend, and hold harmless Vendor from and against any claims, \
damages, and expenses (including reasonable attorneys' fees) arising out of Customer's use of \
the Services or breach of this Agreement. Vendor shall indemnify Customer only for claims \
arising from Vendor's infringement of a third-party patent, copyright, or trademark. \
Vendor's indemnification obligation is capped at $50,000 total; Customer's is uncapped.""",

"indemnification_HIGH": """\
INDEMNIFICATION

Customer shall indemnify, defend, and hold harmless Vendor and its affiliates, officers, \
directors, employees, agents, and successors from and against any and all claims, damages, \
losses, costs, and expenses (including reasonable attorneys' fees) arising out of or related \
to: (a) Customer's use of the Services; (b) any breach by Customer; (c) any third-party \
claims related to Customer Data; or (d) any violation of applicable law by Customer. \
Vendor has no indemnification obligations to Customer. This indemnification obligation \
is unlimited in amount and shall survive termination of this Agreement indefinitely.""",

# ── LIMITATION OF LIABILITY ──────────────────────────────────────────────────
"limitation_of_liability_LOW": """\
LIMITATION OF LIABILITY

IN NO EVENT SHALL EITHER PARTY BE LIABLE TO THE OTHER FOR ANY INDIRECT, INCIDENTAL, SPECIAL, \
EXEMPLARY, PUNITIVE, OR CONSEQUENTIAL DAMAGES, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH \
DAMAGES. EACH PARTY'S TOTAL CUMULATIVE LIABILITY SHALL NOT EXCEED THE GREATER OF: (A) THE \
FEES PAID IN THE TWELVE (12) MONTH PERIOD PRECEDING THE CLAIM; OR (B) USD $50,000. \
THESE LIMITATIONS SHALL NOT APPLY TO: (I) BREACHES OF CONFIDENTIALITY; (II) DATA \
PROTECTION VIOLATIONS; OR (III) GROSS NEGLIGENCE OR WILLFUL MISCONDUCT.""",

"limitation_of_liability_MEDIUM": """\
LIMITATION OF LIABILITY

VENDOR'S TOTAL LIABILITY TO CUSTOMER FOR ANY AND ALL CLAIMS SHALL NOT EXCEED THE FEES \
PAID IN THE THREE (3) MONTHS PRECEDING THE CLAIM. IN NO EVENT SHALL VENDOR BE LIABLE \
FOR ANY INDIRECT OR CONSEQUENTIAL DAMAGES. CUSTOMER'S LIABILITY TO VENDOR IS UNLIMITED. \
THE CAP APPLIES REGARDLESS OF THE FORM OF ACTION AND WHETHER IN CONTRACT, TORT, OR \
STRICT LIABILITY.""",

"limitation_of_liability_HIGH": """\
LIMITATION OF LIABILITY

VENDOR'S TOTAL LIABILITY TO CUSTOMER SHALL NOT EXCEED ONE HUNDRED DOLLARS ($100.00) \
REGARDLESS OF THE CAUSE OF ACTION. CUSTOMER'S TOTAL LIABILITY TO VENDOR SHALL BE \
UNLIMITED. VENDOR EXCLUDES ALL WARRANTIES, CONDITIONS, OR REPRESENTATIONS OF ANY KIND. \
CUSTOMER WAIVES ALL RIGHTS TO SEEK DAMAGES BEYOND DIRECT LOSSES, AND CUSTOMER'S SOLE \
REMEDY FOR ANY DEFICIENCY IN THE SERVICES SHALL BE TERMINATION OF THE AGREEMENT \
WITHOUT REFUND.""",

"limitation_of_liability_CRITICAL": """\
LIMITATION OF LIABILITY

There is no cap on either party's liability under this Agreement. Both parties may seek \
unlimited damages in any forum of their choosing. Consequential, punitive, and special \
damages are expressly available. Vendor reserves the right to pursue Customer for lost \
future profits in the event of early termination by Customer for any reason.""",

# ── CONFIDENTIALITY ──────────────────────────────────────────────────────────
"confidentiality_LOW": """\
CONFIDENTIALITY

Each party agrees to hold in strict confidence all Confidential Information of the other \
party and not to disclose or use it except as necessary to perform obligations hereunder. \
Standard exceptions apply: information that is (a) publicly known through no breach; (b) \
already known to the Receiving Party; (c) received from a third party without restriction; \
(d) independently developed; or (e) required by law with prompt notice. Standard of care: \
at least the same as the Receiving Party uses for its own confidential information, no \
less than reasonable care. Term: two (2) years; trade secrets protected in perpetuity.""",

"confidentiality_MEDIUM": """\
CONFIDENTIALITY

Recipient shall keep all Confidential Information of Discloser in strict confidence and \
shall not disclose it to any third party without prior written consent. Confidential \
Information means all non-public information disclosed by Discloser. The standard \
exceptions (publicly known, independently developed, required by law) apply. \
This obligation continues for five (5) years after disclosure. There are no carve-outs \
for employees or contractors who need access — all disclosures require prior written consent.""",

"confidentiality_HIGH": """\
CONFIDENTIALITY

Employee/Contractor agrees to keep all information obtained during the engagement strictly \
confidential in perpetuity — there is no time limit on this obligation. Confidential \
Information includes all information the Company considers proprietary, regardless of \
whether it is marked as confidential or is publicly available. There are no exceptions \
for information already in the public domain. Employee may not discuss the existence of \
this Agreement with any third party, including family members, without prior written \
approval. Violation shall result in liquidated damages of $500,000 per disclosure.""",

# ── TERMINATION ──────────────────────────────────────────────────────────────
"termination_LOW": """\
TERM AND TERMINATION

This Agreement commences on the Effective Date and continues for one (1) year, \
automatically renewing for successive one-year terms unless either party provides \
thirty (30) days' written notice of non-renewal. Either party may terminate for \
convenience upon thirty (30) days' written notice. Either party may terminate for \
material breach with a thirty (30) day written cure period. Immediate termination \
rights apply upon insolvency. Upon termination, all licenses terminate and each party \
shall return or destroy the other's Confidential Information within fifteen (15) days.""",

"termination_MEDIUM": """\
TERM AND TERMINATION

This Agreement commences on the Effective Date and has an initial term of two (2) years \
with automatic renewal unless ninety (90) days' notice is given. Customer may not \
terminate for convenience during the Initial Term. Vendor may terminate immediately if \
Customer fails to pay any invoice within five (5) business days of the due date. \
Upon termination by Customer for any reason during the Initial Term, Customer shall pay \
an early termination fee equal to all remaining fees for the unexpired term.""",

"termination_HIGH": """\
TERM AND TERMINATION

This Agreement has a minimum term of three (3) years. Customer may not terminate for \
any reason during the minimum term. Vendor may terminate immediately at any time for \
any reason or no reason. Upon Vendor termination, all fees for the remaining term become \
immediately due and payable. Customer Data will be deleted within 24 hours of termination \
without opportunity for export. There is no cure period for any alleged breach.""",

# ── IP ASSIGNMENT ─────────────────────────────────────────────────────────────
"ip_assignment_LOW": """\
INTELLECTUAL PROPERTY

Contractor assigns to Client all right, title, and interest in Work Product specifically \
created for Client under this Agreement. Contractor retains all pre-existing IP (Background \
IP). Contractor grants Client a non-exclusive, royalty-free, perpetual license to use \
Background IP incorporated into the Work Product. Per California Labor Code Section 2870, \
this assignment does not apply to inventions developed entirely on Contractor's own time \
without Company equipment or trade secrets, and unrelated to Company's business.""",

"ip_assignment_MEDIUM": """\
INTELLECTUAL PROPERTY

All work product, inventions, and developments created by Employee during employment, \
using Company resources or related to Company business, shall be solely owned by the \
Company. Employee assigns all such IP to Company. Employee retains rights to inventions \
created entirely on personal time without Company resources and unrelated to Company's \
business, as required by California Labor Code Section 2870. Background IP created \
before employment is excluded from this assignment.""",

"ip_assignment_HIGH": """\
INTELLECTUAL PROPERTY ASSIGNMENT

Employee hereby irrevocably assigns to Company all inventions, discoveries, developments, \
improvements, trade secrets, and works of authorship conceived, made, developed, or \
reduced to practice by Employee, whether alone or jointly with others, during the term \
of employment, whether or not during working hours, whether or not related to Company's \
business, and whether or not using Company resources. This assignment includes all \
intellectual property rights worldwide and shall survive termination. Employee waives \
all moral rights in such works.""",

# ── NON-COMPETE ──────────────────────────────────────────────────────────────
"non_compete_LOW": """\
NON-SOLICITATION

During employment and for twelve (12) months following termination, Employee agrees not \
to directly solicit, recruit, or hire any Company employee who was employed by Company \
during the last six months of Employee's employment. This non-solicitation obligation \
does not restrict Employee from working at any company or in any industry. No geographic \
restriction applies. This Agreement contains no non-compete covenant.""",

"non_compete_MEDIUM": """\
NON-SOLICITATION OF CUSTOMERS

During employment and for one (1) year following termination, Employee agrees not to \
solicit or divert away any customer or prospective customer of the Company with whom \
Employee had material contact during the last twelve months of employment. Employee \
is not restricted from competing generally or from working for competitors.""",

"non_compete_HIGH": """\
NON-COMPETITION AND NON-SOLICITATION

During employment and for two (2) years following termination for any reason, Employee \
shall not, directly or indirectly, engage in, own, manage, operate, control, be employed \
by, provide services to, participate in, or be connected with any business that competes \
with Company's business anywhere in the world. Employee also agrees not to solicit \
Company's employees or customers for a period of two (2) years post-termination. \
Employee acknowledges these restrictions are reasonable and necessary to protect \
Company's legitimate business interests.""",

# ── GOVERNING LAW ────────────────────────────────────────────────────────────
"governing_law_LOW": """\
GOVERNING LAW

This Agreement shall be governed by and construed in accordance with the laws of the \
State of California, without regard to its conflict of laws principles. The parties \
submit to the exclusive jurisdiction of the state and federal courts located in \
San Francisco County, California.""",

"governing_law_MEDIUM": """\
GOVERNING LAW

This Agreement shall be governed by the laws of the State of New York. Any disputes \
shall be heard exclusively in the courts of New York County, New York. Each party \
waives any objection to such venue based on inconvenience.""",

"governing_law_HIGH": """\
GOVERNING LAW

This Agreement shall be governed by the laws of the Cayman Islands. All disputes \
shall be resolved exclusively in the courts of the Cayman Islands. Each party \
irrevocably waives any right to object to such jurisdiction or to claim that such \
forum is inconvenient. Service of process may be effected by international mail.""",

# ── DISPUTE RESOLUTION ───────────────────────────────────────────────────────
"dispute_resolution_LOW": """\
DISPUTE RESOLUTION

Any dispute shall be resolved by: (1) good-faith negotiation (30 days); (2) mediation \
administered by JAMS (30 days); (3) binding arbitration under AAA Commercial Rules in \
San Francisco, California. The arbitrator may award any remedy available at law. \
Either party may seek injunctive relief in court. Each party waives the right to a \
jury trial. Class actions are permitted.""",

"dispute_resolution_MEDIUM": """\
DISPUTE RESOLUTION

Any dispute shall be resolved by binding arbitration in New York City under AAA rules. \
EACH PARTY WAIVES ITS RIGHT TO A JURY TRIAL AND TO PARTICIPATE IN A CLASS ACTION \
OR CLASS ARBITRATION. The arbitration shall be confidential. The arbitrator's award \
shall be final and not subject to appeal except as required by applicable law.""",

"dispute_resolution_HIGH": """\
DISPUTE RESOLUTION

All disputes shall be resolved exclusively by binding arbitration administered by \
Company's chosen arbitrator in Company's home city. Customer waives all rights to \
a jury trial, class action, or any court proceeding. The arbitrator shall apply \
Company's internal policies as the governing rules. The arbitration shall be \
confidential and Customer may not disclose the existence or outcome of any arbitration \
without Company's prior written consent. Company may seek injunctive relief in any \
court at any time.""",

# ── DATA PROTECTION ──────────────────────────────────────────────────────────
"data_protection_LOW": """\
DATA PROTECTION

Each party shall implement appropriate technical and organizational measures to protect \
Personal Data against unauthorized access, disclosure, or destruction. Vendor shall \
notify Customer within seventy-two (72) hours of discovering a Personal Data breach. \
Vendor shall process Customer Personal Data only as instructed by Customer and solely \
for providing the Services. Vendor shall not sell Customer's Personal Data. Upon \
termination, Vendor shall delete or return all Customer Personal Data within thirty \
(30) days.""",

"data_protection_HIGH": """\
DATA USE

Customer grants Vendor a perpetual, irrevocable, royalty-free license to use, copy, \
reproduce, process, adapt, modify, publish, transmit, display, and distribute all \
Customer Data for any purpose, including developing competing products, training \
machine learning models, and sharing with Vendor's affiliates and third-party \
partners worldwide. Vendor may sell or otherwise transfer Customer Data to third \
parties for marketing purposes. Customer Data may be retained indefinitely even \
after termination of this Agreement.""",

# ── FORCE MAJEURE ────────────────────────────────────────────────────────────
"force_majeure_LOW": """\
FORCE MAJEURE

Neither party shall be liable for delays caused by circumstances beyond its reasonable \
control including acts of God, natural disasters, government action, war, or pandemic. \
The affected party must provide prompt written notice, use commercially reasonable \
efforts to mitigate, and provide regular updates. If the event continues for sixty (60) \
days, either party may terminate without liability, except for fees for services already \
performed.""",

"force_majeure_HIGH": """\
FORCE MAJEURE

Vendor shall not be liable for any failure or delay in performance for any reason, \
including but not limited to: equipment failure, staffing shortages, supply chain \
disruptions, economic conditions, or changes in market conditions. This force majeure \
provision applies exclusively to Vendor. Customer's payment obligations are not \
suspended or excused by any Force Majeure Event, regardless of duration.""",

# ── PAYMENT TERMS ────────────────────────────────────────────────────────────
"payment_terms_LOW": """\
PAYMENT TERMS

Customer shall pay all undisputed invoices within thirty (30) days of invoice date. \
Late payments accrue interest at 1.5% per month or the maximum legal rate. If any \
amount is disputed in good faith, Customer shall provide written notice within fifteen \
(15) days and pay undisputed amounts on time. Parties shall resolve disputes within \
thirty (30) days.""",

"payment_terms_MEDIUM": """\
PAYMENT TERMS

Customer shall pay all invoices within fifteen (15) days of the invoice date. All \
fees are non-refundable once invoiced. Vendor may increase fees at any time upon \
thirty (30) days' notice. Customer is responsible for all taxes. Late payments accrue \
interest at 2% per month. Vendor may suspend services immediately upon non-payment \
without further notice.""",

"payment_terms_HIGH": """\
PAYMENT TERMS

All fees are due upon execution of this Agreement and are fully non-refundable \
regardless of circumstances, including early termination, service outages, or \
data loss. Customer shall pay all fees within five (5) business days of any invoice. \
Vendor may increase fees at any time without notice. Customer waives all rights to \
dispute any invoice. Failure to pay shall result in immediate termination of access \
with no data export opportunity.""",

# ── WARRANTY ─────────────────────────────────────────────────────────────────
"warranty_LOW": """\
WARRANTIES

Vendor warrants that the Services will perform materially in accordance with the \
Documentation. Vendor implements commercially reasonable security measures. Services \
are provided by qualified professionals consistent with industry standards. If \
Services fail to conform, Vendor will re-perform or provide a pro-rata refund. \
EXCEPT AS STATED HEREIN, SERVICES ARE PROVIDED AS-IS WITHOUT OTHER WARRANTIES.""",

"warranty_HIGH": """\
WARRANTY DISCLAIMER

THE SERVICES ARE PROVIDED "AS IS" AND "AS AVAILABLE" WITHOUT WARRANTY OF ANY KIND. \
VENDOR EXPRESSLY DISCLAIMS ALL WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT NOT \
LIMITED TO ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR \
PURPOSE, TITLE, AND NON-INFRINGEMENT. VENDOR DOES NOT WARRANT THAT THE SERVICES \
WILL BE UNINTERRUPTED, ERROR-FREE, SECURE, OR THAT DEFECTS WILL BE CORRECTED. \
VENDOR PROVIDES NO UPTIME GUARANTEE AND SHALL HAVE NO LIABILITY FOR DOWNTIME.""",

# ── REPRESENTATIONS ──────────────────────────────────────────────────────────
"representations_LOW": """\
REPRESENTATIONS AND WARRANTIES

Each party represents and warrants that: (a) it is duly organized and in good standing; \
(b) it has full authority to enter into this Agreement; (c) execution does not violate \
any other agreement; (d) it will comply with all applicable laws; and (e) it has all \
permits necessary for performance.""",

# ── NDA-SPECIFIC ─────────────────────────────────────────────────────────────
"nda_purpose_LOW": """\
PURPOSE

The parties wish to explore a potential business relationship (the "Purpose") and may \
disclose Confidential Information to each other solely in connection with the Purpose. \
Each party's Confidential Information may only be used for the Purpose and for no other \
purpose whatsoever.""",

"nda_purpose_HIGH": """\
PURPOSE AND LICENSE

The Disclosing Party grants the Receiving Party an irrevocable, perpetual, royalty-free \
license to use the Confidential Information for any purpose, including development of \
competing products and services, provided the Receiving Party does not directly attribute \
the information to the Disclosing Party.""",

# ── LEASE-SPECIFIC ────────────────────────────────────────────────────────────
"lease_rent_LOW": """\
RENT

Tenant shall pay monthly rent of $3,500 due on the first day of each month with a \
five (5) day grace period. If rent is not received by the sixth day, a late fee of \
$50 shall apply. Landlord shall provide thirty (30) days' written notice before any \
rent increase. Rent may not be increased more than once in any twelve-month period.""",

"lease_rent_HIGH": """\
RENT

Tenant shall pay monthly rent of $4,200 due on the first day of each month with no \
grace period. A late fee of 10% of monthly rent applies if rent is not received by \
2:00 PM on the first day. Landlord may increase rent at any time upon seven (7) days' \
notice. Tenant waives all rights to dispute rent increases. Failure to pay any amount \
when due is grounds for immediate eviction proceedings.""",

"lease_termination_LOW": """\
TERM AND RENEWAL

This lease commences on the start date and continues for twelve (12) months. Either \
party may terminate with sixty (60) days' written notice at the end of any lease term. \
Landlord may not terminate during the lease term except for material breach with a \
thirty (30) day cure period.""",

"lease_termination_HIGH": """\
TERM AND TERMINATION

Landlord may terminate this lease at any time for any reason with seven (7) days' \
written notice. Tenant may not sublet or assign without Landlord's written consent, \
which may be withheld for any reason. Tenant waives all rights to dispute eviction. \
Upon termination, Tenant must vacate within 24 hours and forfeits the security deposit \
regardless of the condition of the premises.""",

"lease_maintenance_LOW": """\
MAINTENANCE AND REPAIRS

Landlord shall maintain the premises in habitable condition and make all structural \
repairs within a reasonable time after notice. Tenant shall maintain cleanliness and \
promptly notify Landlord of needed repairs. Each party is responsible for repairs \
caused by its own negligence.""",

"lease_maintenance_HIGH": """\
MAINTENANCE

Tenant accepts the premises in "as-is" condition and assumes full responsibility for \
all repairs, maintenance, and improvements during the lease term, including structural, \
plumbing, electrical, and HVAC systems. Landlord has no maintenance obligations. \
Tenant indemnifies Landlord for all claims arising from any condition of the premises.""",

# ── SEVERANCE ─────────────────────────────────────────────────────────────────
"severance_LOW": """\
SEVERANCE

If the Company terminates Employee without Cause, Employee shall receive: (a) \
continuation of base salary for three (3) months; (b) COBRA health benefits for \
the same period; and (c) accelerated vesting of equity awards vesting in the next \
three months. "Cause" means felony conviction, willful fraud, or material breach \
after written notice.""",

"severance_HIGH": """\
AT-WILL EMPLOYMENT

Employment is at-will. The Company may terminate Employee at any time for any reason \
or no reason with or without notice. No severance is payable upon termination for \
any reason, including without Cause. All unvested equity is forfeited immediately \
upon termination. Employee forfeits any unpaid bonus upon termination, regardless \
of the reason for termination.""",

# ── ASSIGNMENT ────────────────────────────────────────────────────────────────
"assignment_LOW": """\
ASSIGNMENT

Neither party may assign this Agreement without the prior written consent of the other \
party, not to be unreasonably withheld. Either party may assign without consent to an \
affiliate or to a successor in a merger or acquisition, provided the assignee assumes \
all obligations. Any purported assignment in violation of this section is void.""",

"assignment_HIGH": """\
ASSIGNMENT

Vendor may assign this Agreement, in whole or in part, at any time and for any reason \
without Customer's consent, including to a competitor of Customer. Customer may not \
assign this Agreement without Vendor's prior written consent, which may be withheld \
in Vendor's sole discretion.""",

}

# ─────────────────────────────────────────────────────────────────────────────
# CONTRACT TEMPLATES (50 total)
# Each entry: id, type, title, risk_profile, sections[], ground_truth{}
# ─────────────────────────────────────────────────────────────────────────────

CONTRACTS = [

# ══ NDAs (12) ════════════════════════════════════════════════════════════════

{"id": "nda_001", "type": "NDA", "risk_profile": "LOW",
 "parties": ("AlphaVentures LLC", "BetaTech Inc."),
 "sections": ["nda_purpose_LOW","confidentiality_LOW","governing_law_LOW","representations_LOW"],
 "expected_clauses": ["confidentiality","governing_law","representations"],
 "high_risk_clauses": [], "missing_flag": []},

{"id": "nda_002", "type": "NDA", "risk_profile": "LOW",
 "parties": ("Meridian Capital", "Solaris AI Corp"),
 "sections": ["nda_purpose_LOW","confidentiality_LOW","dispute_resolution_LOW","governing_law_LOW"],
 "expected_clauses": ["confidentiality","dispute_resolution","governing_law"],
 "high_risk_clauses": [], "missing_flag": ["indemnification"]},

{"id": "nda_003", "type": "NDA", "risk_profile": "LOW",
 "parties": ("Pinebrook Partners", "DataStream Inc."),
 "sections": ["nda_purpose_LOW","confidentiality_LOW","governing_law_LOW","assignment_LOW"],
 "expected_clauses": ["confidentiality","governing_law"],
 "high_risk_clauses": [], "missing_flag": ["indemnification","limitation_of_liability"]},

{"id": "nda_004", "type": "NDA", "risk_profile": "LOW",
 "parties": ("NovaStar Corp", "Apex Consulting"),
 "sections": ["nda_purpose_LOW","confidentiality_LOW","governing_law_LOW","force_majeure_LOW"],
 "expected_clauses": ["confidentiality","governing_law","force_majeure"],
 "high_risk_clauses": [], "missing_flag": []},

{"id": "nda_005", "type": "NDA", "risk_profile": "MEDIUM",
 "parties": ("GlobalTech Ltd", "StartupXYZ"),
 "sections": ["nda_purpose_LOW","confidentiality_MEDIUM","governing_law_MEDIUM","representations_LOW"],
 "expected_clauses": ["confidentiality","governing_law","representations"],
 "high_risk_clauses": [], "missing_flag": []},

{"id": "nda_006", "type": "NDA", "risk_profile": "MEDIUM",
 "parties": ("Venture Capital Partners", "HealthAI Inc."),
 "sections": ["nda_purpose_LOW","confidentiality_MEDIUM","governing_law_LOW","dispute_resolution_MEDIUM"],
 "expected_clauses": ["confidentiality","governing_law","dispute_resolution"],
 "high_risk_clauses": ["dispute_resolution"], "missing_flag": []},

{"id": "nda_007", "type": "NDA", "risk_profile": "MEDIUM",
 "parties": ("Enterprise Holdings", "DevOps Startup"),
 "sections": ["nda_purpose_LOW","confidentiality_MEDIUM","governing_law_MEDIUM"],
 "expected_clauses": ["confidentiality","governing_law"],
 "high_risk_clauses": [], "missing_flag": []},

{"id": "nda_008", "type": "NDA", "risk_profile": "MEDIUM",
 "parties": ("Retail Giant Corp", "AI Solutions LLC"),
 "sections": ["nda_purpose_LOW","confidentiality_MEDIUM","governing_law_LOW"],
 "expected_clauses": ["confidentiality","governing_law"],
 "high_risk_clauses": [], "missing_flag": []},

{"id": "nda_009", "type": "NDA", "risk_profile": "HIGH",
 "parties": ("Dominant Corp", "Small Startup Inc."),
 "sections": ["nda_purpose_HIGH","confidentiality_HIGH","governing_law_HIGH"],
 "expected_clauses": ["confidentiality","governing_law"],
 "high_risk_clauses": ["confidentiality","governing_law"], "missing_flag": ["indemnification"]},

{"id": "nda_010", "type": "NDA", "risk_profile": "HIGH",
 "parties": ("MegaCorp Industries", "Freelancer LLC"),
 "sections": ["nda_purpose_HIGH","confidentiality_HIGH","dispute_resolution_HIGH","governing_law_HIGH"],
 "expected_clauses": ["confidentiality","dispute_resolution","governing_law"],
 "high_risk_clauses": ["confidentiality","dispute_resolution","governing_law"], "missing_flag": []},

{"id": "nda_011", "type": "NDA", "risk_profile": "HIGH",
 "parties": ("Market Leader Inc.", "New Entrant Corp"),
 "sections": ["nda_purpose_HIGH","confidentiality_HIGH","governing_law_HIGH","assignment_HIGH"],
 "expected_clauses": ["confidentiality","governing_law"],
 "high_risk_clauses": ["confidentiality","governing_law"], "missing_flag": []},

{"id": "nda_012", "type": "NDA", "risk_profile": "HIGH",
 "parties": ("Platform Corp", "Dependent Vendor"),
 "sections": ["nda_purpose_HIGH","confidentiality_HIGH","governing_law_HIGH"],
 "expected_clauses": ["confidentiality","governing_law"],
 "high_risk_clauses": ["confidentiality"], "missing_flag": ["force_majeure"]},

# ══ SaaS Agreements (15) ═════════════════════════════════════════════════════

{"id": "saas_001", "type": "SaaS Agreement", "risk_profile": "LOW",
 "parties": ("CloudSoft Inc.", "SMB Customer LLC"),
 "sections": ["indemnification_LOW","limitation_of_liability_LOW","termination_LOW","confidentiality_LOW","data_protection_LOW","payment_terms_LOW","warranty_LOW","governing_law_LOW","dispute_resolution_LOW"],
 "expected_clauses": ["indemnification","limitation_of_liability","termination","confidentiality","data_protection","payment_terms","warranty","governing_law","dispute_resolution"],
 "high_risk_clauses": [], "missing_flag": []},

{"id": "saas_002", "type": "SaaS Agreement", "risk_profile": "LOW",
 "parties": ("PlatformX SaaS", "Enterprise Client Corp"),
 "sections": ["indemnification_LOW","limitation_of_liability_LOW","termination_LOW","confidentiality_LOW","data_protection_LOW","payment_terms_LOW","force_majeure_LOW","governing_law_LOW"],
 "expected_clauses": ["indemnification","limitation_of_liability","termination","confidentiality","data_protection","payment_terms","force_majeure","governing_law"],
 "high_risk_clauses": [], "missing_flag": []},

{"id": "saas_003", "type": "SaaS Agreement", "risk_profile": "LOW",
 "parties": ("DevTools Pro", "Startup Customer"),
 "sections": ["indemnification_LOW","limitation_of_liability_LOW","termination_LOW","confidentiality_LOW","data_protection_LOW","warranty_LOW","governing_law_LOW","assignment_LOW"],
 "expected_clauses": ["indemnification","limitation_of_liability","termination","confidentiality","data_protection","warranty","governing_law"],
 "high_risk_clauses": [], "missing_flag": []},

{"id": "saas_004", "type": "SaaS Agreement", "risk_profile": "LOW",
 "parties": ("Analytics Suite Co.", "Mid-Market Customer"),
 "sections": ["indemnification_LOW","limitation_of_liability_LOW","termination_LOW","confidentiality_LOW","data_protection_LOW","payment_terms_LOW","warranty_LOW","representations_LOW","governing_law_LOW"],
 "expected_clauses": ["indemnification","limitation_of_liability","termination","confidentiality","data_protection","payment_terms","warranty","representations","governing_law"],
 "high_risk_clauses": [], "missing_flag": []},

{"id": "saas_005", "type": "SaaS Agreement", "risk_profile": "LOW",
 "parties": ("Security Platform Inc.", "Healthcare Customer"),
 "sections": ["indemnification_LOW","limitation_of_liability_LOW","termination_LOW","confidentiality_LOW","data_protection_LOW","payment_terms_LOW","warranty_LOW","governing_law_LOW","force_majeure_LOW"],
 "expected_clauses": ["indemnification","limitation_of_liability","termination","confidentiality","data_protection","payment_terms","warranty","governing_law","force_majeure"],
 "high_risk_clauses": [], "missing_flag": []},

{"id": "saas_006", "type": "SaaS Agreement", "risk_profile": "MEDIUM",
 "parties": ("CloudAI Vendor", "Enterprise Corp"),
 "sections": ["indemnification_MEDIUM","limitation_of_liability_MEDIUM","termination_MEDIUM","confidentiality_LOW","payment_terms_MEDIUM","warranty_LOW","governing_law_MEDIUM"],
 "expected_clauses": ["indemnification","limitation_of_liability","termination","confidentiality","payment_terms","warranty","governing_law"],
 "high_risk_clauses": ["indemnification","limitation_of_liability","payment_terms"], "missing_flag": ["data_protection"]},

{"id": "saas_007", "type": "SaaS Agreement", "risk_profile": "MEDIUM",
 "parties": ("B2B Software Corp", "Growing Startup"),
 "sections": ["indemnification_MEDIUM","limitation_of_liability_MEDIUM","termination_MEDIUM","confidentiality_MEDIUM","payment_terms_LOW","warranty_HIGH","governing_law_LOW"],
 "expected_clauses": ["indemnification","limitation_of_liability","termination","confidentiality","payment_terms","warranty","governing_law"],
 "high_risk_clauses": ["indemnification","limitation_of_liability","warranty"], "missing_flag": []},

{"id": "saas_008", "type": "SaaS Agreement", "risk_profile": "MEDIUM",
 "parties": ("Infrastructure SaaS", "Financial Services Firm"),
 "sections": ["indemnification_MEDIUM","limitation_of_liability_MEDIUM","termination_LOW","confidentiality_LOW","data_protection_LOW","payment_terms_MEDIUM","governing_law_MEDIUM","dispute_resolution_MEDIUM"],
 "expected_clauses": ["indemnification","limitation_of_liability","termination","confidentiality","data_protection","payment_terms","governing_law","dispute_resolution"],
 "high_risk_clauses": ["indemnification","limitation_of_liability","dispute_resolution"], "missing_flag": []},

{"id": "saas_009", "type": "SaaS Agreement", "risk_profile": "MEDIUM",
 "parties": ("ML Platform Inc.", "Tech Scale-up"),
 "sections": ["indemnification_MEDIUM","limitation_of_liability_MEDIUM","termination_MEDIUM","confidentiality_MEDIUM","payment_terms_MEDIUM","warranty_HIGH","governing_law_LOW"],
 "expected_clauses": ["indemnification","limitation_of_liability","termination","confidentiality","payment_terms","warranty","governing_law"],
 "high_risk_clauses": ["warranty","limitation_of_liability"], "missing_flag": ["data_protection"]},

{"id": "saas_010", "type": "SaaS Agreement", "risk_profile": "MEDIUM",
 "parties": ("Workflow Automation Co.", "Operations Heavy Corp"),
 "sections": ["indemnification_MEDIUM","limitation_of_liability_MEDIUM","termination_MEDIUM","confidentiality_LOW","data_protection_LOW","payment_terms_MEDIUM","governing_law_MEDIUM"],
 "expected_clauses": ["indemnification","limitation_of_liability","termination","confidentiality","data_protection","payment_terms","governing_law"],
 "high_risk_clauses": ["indemnification","payment_terms"], "missing_flag": []},

{"id": "saas_011", "type": "SaaS Agreement", "risk_profile": "HIGH",
 "parties": ("Dominant Platform", "Captive Customer"),
 "sections": ["indemnification_HIGH","limitation_of_liability_HIGH","termination_HIGH","confidentiality_HIGH","data_protection_HIGH","payment_terms_HIGH","warranty_HIGH","governing_law_HIGH","dispute_resolution_HIGH"],
 "expected_clauses": ["indemnification","limitation_of_liability","termination","confidentiality","data_protection","payment_terms","warranty","governing_law","dispute_resolution"],
 "high_risk_clauses": ["indemnification","limitation_of_liability","termination","confidentiality","data_protection","payment_terms","warranty","governing_law","dispute_resolution"], "missing_flag": []},

{"id": "saas_012", "type": "SaaS Agreement", "risk_profile": "HIGH",
 "parties": ("Monopoly SaaS Corp", "No-Choice Enterprise"),
 "sections": ["indemnification_HIGH","limitation_of_liability_CRITICAL","termination_HIGH","confidentiality_HIGH","payment_terms_HIGH","warranty_HIGH","governing_law_HIGH","dispute_resolution_HIGH"],
 "expected_clauses": ["indemnification","limitation_of_liability","termination","confidentiality","payment_terms","warranty","governing_law","dispute_resolution"],
 "high_risk_clauses": ["indemnification","limitation_of_liability","termination","payment_terms","warranty"], "missing_flag": ["data_protection","force_majeure"]},

{"id": "saas_013", "type": "SaaS Agreement", "risk_profile": "HIGH",
 "parties": ("TechGiant Cloud", "Startup Vendor"),
 "sections": ["indemnification_HIGH","limitation_of_liability_HIGH","termination_HIGH","confidentiality_LOW","data_protection_HIGH","payment_terms_HIGH","warranty_HIGH","governing_law_HIGH"],
 "expected_clauses": ["indemnification","limitation_of_liability","termination","confidentiality","data_protection","payment_terms","warranty","governing_law"],
 "high_risk_clauses": ["indemnification","limitation_of_liability","data_protection","payment_terms","warranty"], "missing_flag": []},

{"id": "saas_014", "type": "SaaS Agreement", "risk_profile": "HIGH",
 "parties": ("Vendor Inc.", "Locked-In Customer LLC"),
 "sections": ["indemnification_HIGH","limitation_of_liability_HIGH","termination_HIGH","data_protection_HIGH","payment_terms_HIGH","warranty_HIGH","governing_law_HIGH","dispute_resolution_HIGH"],
 "expected_clauses": ["indemnification","limitation_of_liability","termination","data_protection","payment_terms","warranty","governing_law","dispute_resolution"],
 "high_risk_clauses": ["indemnification","limitation_of_liability","termination","data_protection","payment_terms"], "missing_flag": ["confidentiality"]},

{"id": "saas_015", "type": "SaaS Agreement", "risk_profile": "CRITICAL",
 "parties": ("Abusive Vendor Corp", "Desperate Customer"),
 "sections": ["indemnification_HIGH","limitation_of_liability_CRITICAL","termination_HIGH","confidentiality_HIGH","data_protection_HIGH","payment_terms_HIGH","warranty_HIGH","governing_law_HIGH","dispute_resolution_HIGH","assignment_HIGH"],
 "expected_clauses": ["indemnification","limitation_of_liability","termination","confidentiality","data_protection","payment_terms","warranty","governing_law","dispute_resolution"],
 "high_risk_clauses": ["indemnification","limitation_of_liability","termination","confidentiality","data_protection","payment_terms","warranty","governing_law","dispute_resolution"], "missing_flag": []},

# ══ Employment Contracts (12) ═════════════════════════════════════════════════

{"id": "emp_001", "type": "Employment Contract", "risk_profile": "LOW",
 "parties": ("GoodEmployer Inc.", "Jane Smith"),
 "sections": ["ip_assignment_LOW","non_compete_LOW","confidentiality_LOW","governing_law_LOW","severance_LOW","representations_LOW"],
 "expected_clauses": ["ip_assignment","non_compete","confidentiality","governing_law"],
 "high_risk_clauses": [], "missing_flag": []},

{"id": "emp_002", "type": "Employment Contract", "risk_profile": "LOW",
 "parties": ("TechCompany CA", "Bob Developer"),
 "sections": ["ip_assignment_LOW","non_compete_LOW","confidentiality_LOW","governing_law_LOW","severance_LOW"],
 "expected_clauses": ["ip_assignment","non_compete","confidentiality","governing_law"],
 "high_risk_clauses": [], "missing_flag": []},

{"id": "emp_003", "type": "Employment Contract", "risk_profile": "LOW",
 "parties": ("Startup Corp", "Alice Engineer"),
 "sections": ["ip_assignment_MEDIUM","non_compete_LOW","confidentiality_LOW","governing_law_LOW","severance_LOW"],
 "expected_clauses": ["ip_assignment","non_compete","confidentiality","governing_law"],
 "high_risk_clauses": [], "missing_flag": []},

{"id": "emp_004", "type": "Employment Contract", "risk_profile": "LOW",
 "parties": ("ScaleUp Inc.", "Carlos Manager"),
 "sections": ["ip_assignment_LOW","non_compete_MEDIUM","confidentiality_LOW","governing_law_LOW","severance_LOW"],
 "expected_clauses": ["ip_assignment","non_compete","confidentiality","governing_law"],
 "high_risk_clauses": [], "missing_flag": []},

{"id": "emp_005", "type": "Employment Contract", "risk_profile": "MEDIUM",
 "parties": ("Tech Firm LLC", "Dana Developer"),
 "sections": ["ip_assignment_MEDIUM","non_compete_MEDIUM","confidentiality_MEDIUM","governing_law_LOW","severance_HIGH"],
 "expected_clauses": ["ip_assignment","non_compete","confidentiality","governing_law"],
 "high_risk_clauses": [], "missing_flag": []},

{"id": "emp_006", "type": "Employment Contract", "risk_profile": "MEDIUM",
 "parties": ("Software House Inc.", "Eric Engineer"),
 "sections": ["ip_assignment_MEDIUM","non_compete_MEDIUM","confidentiality_MEDIUM","governing_law_MEDIUM","severance_HIGH"],
 "expected_clauses": ["ip_assignment","non_compete","confidentiality","governing_law"],
 "high_risk_clauses": ["confidentiality"], "missing_flag": []},

{"id": "emp_007", "type": "Employment Contract", "risk_profile": "MEDIUM",
 "parties": ("Enterprise Co.", "Fiona PM"),
 "sections": ["ip_assignment_HIGH","non_compete_LOW","confidentiality_MEDIUM","governing_law_LOW","severance_HIGH"],
 "expected_clauses": ["ip_assignment","non_compete","confidentiality","governing_law"],
 "high_risk_clauses": ["ip_assignment"], "missing_flag": []},

{"id": "emp_008", "type": "Employment Contract", "risk_profile": "MEDIUM",
 "parties": ("Digital Agency", "George Designer"),
 "sections": ["ip_assignment_HIGH","non_compete_MEDIUM","confidentiality_LOW","governing_law_MEDIUM","severance_HIGH"],
 "expected_clauses": ["ip_assignment","non_compete","confidentiality","governing_law"],
 "high_risk_clauses": ["ip_assignment"], "missing_flag": []},

{"id": "emp_009", "type": "Employment Contract", "risk_profile": "HIGH",
 "parties": ("Aggressive Corp", "Hannah Engineer"),
 "sections": ["ip_assignment_HIGH","non_compete_HIGH","confidentiality_HIGH","governing_law_HIGH","severance_HIGH","dispute_resolution_HIGH"],
 "expected_clauses": ["ip_assignment","non_compete","confidentiality","governing_law","dispute_resolution"],
 "high_risk_clauses": ["ip_assignment","non_compete","confidentiality","dispute_resolution"], "missing_flag": []},

{"id": "emp_010", "type": "Employment Contract", "risk_profile": "HIGH",
 "parties": ("Control Freak Inc.", "Ivan Developer"),
 "sections": ["ip_assignment_HIGH","non_compete_HIGH","confidentiality_HIGH","governing_law_HIGH","severance_HIGH"],
 "expected_clauses": ["ip_assignment","non_compete","confidentiality","governing_law"],
 "high_risk_clauses": ["ip_assignment","non_compete","confidentiality"], "missing_flag": []},

{"id": "emp_011", "type": "Employment Contract", "risk_profile": "HIGH",
 "parties": ("Tech Giant Corp", "Julia Scientist"),
 "sections": ["ip_assignment_HIGH","non_compete_HIGH","confidentiality_HIGH","governing_law_LOW","severance_HIGH","dispute_resolution_HIGH"],
 "expected_clauses": ["ip_assignment","non_compete","confidentiality","governing_law","dispute_resolution"],
 "high_risk_clauses": ["ip_assignment","non_compete","dispute_resolution"], "missing_flag": []},

{"id": "emp_012", "type": "Employment Contract", "risk_profile": "CRITICAL",
 "parties": ("Maximum Control Corp", "Kevin Employee"),
 "sections": ["ip_assignment_HIGH","non_compete_HIGH","confidentiality_HIGH","governing_law_HIGH","severance_HIGH","dispute_resolution_HIGH","assignment_HIGH"],
 "expected_clauses": ["ip_assignment","non_compete","confidentiality","governing_law","dispute_resolution"],
 "high_risk_clauses": ["ip_assignment","non_compete","confidentiality","governing_law","dispute_resolution"], "missing_flag": []},

# ══ Service Agreements (7) ════════════════════════════════════════════════════

{"id": "svc_001", "type": "Service Agreement", "risk_profile": "LOW",
 "parties": ("Professional Services Co.", "Client Corp"),
 "sections": ["indemnification_LOW","limitation_of_liability_LOW","confidentiality_LOW","ip_assignment_LOW","payment_terms_LOW","termination_LOW","warranty_LOW","governing_law_LOW"],
 "expected_clauses": ["indemnification","limitation_of_liability","confidentiality","ip_assignment","payment_terms","termination","warranty","governing_law"],
 "high_risk_clauses": [], "missing_flag": []},

{"id": "svc_002", "type": "Service Agreement", "risk_profile": "LOW",
 "parties": ("Consulting Firm LLP", "Mid-Market Client"),
 "sections": ["indemnification_LOW","limitation_of_liability_LOW","confidentiality_LOW","payment_terms_LOW","termination_LOW","warranty_LOW","governing_law_LOW","force_majeure_LOW"],
 "expected_clauses": ["indemnification","limitation_of_liability","confidentiality","payment_terms","termination","warranty","governing_law","force_majeure"],
 "high_risk_clauses": [], "missing_flag": []},

{"id": "svc_003", "type": "Service Agreement", "risk_profile": "LOW",
 "parties": ("IT Services Provider", "Healthcare Client"),
 "sections": ["indemnification_LOW","limitation_of_liability_LOW","confidentiality_LOW","ip_assignment_LOW","data_protection_LOW","payment_terms_LOW","termination_LOW","governing_law_LOW"],
 "expected_clauses": ["indemnification","limitation_of_liability","confidentiality","ip_assignment","data_protection","payment_terms","termination","governing_law"],
 "high_risk_clauses": [], "missing_flag": []},

{"id": "svc_004", "type": "Service Agreement", "risk_profile": "LOW",
 "parties": ("Design Agency LLC", "E-commerce Client"),
 "sections": ["indemnification_LOW","limitation_of_liability_LOW","confidentiality_LOW","ip_assignment_LOW","payment_terms_LOW","termination_LOW","warranty_LOW","governing_law_LOW","assignment_LOW"],
 "expected_clauses": ["indemnification","limitation_of_liability","confidentiality","ip_assignment","payment_terms","termination","warranty","governing_law"],
 "high_risk_clauses": [], "missing_flag": []},

{"id": "svc_005", "type": "Service Agreement", "risk_profile": "HIGH",
 "parties": ("Dominant Agency", "Dependent Client"),
 "sections": ["indemnification_HIGH","limitation_of_liability_HIGH","confidentiality_MEDIUM","ip_assignment_HIGH","payment_terms_HIGH","termination_HIGH","warranty_HIGH","governing_law_HIGH"],
 "expected_clauses": ["indemnification","limitation_of_liability","confidentiality","ip_assignment","payment_terms","termination","warranty","governing_law"],
 "high_risk_clauses": ["indemnification","limitation_of_liability","ip_assignment","payment_terms","termination","warranty"], "missing_flag": []},

{"id": "svc_006", "type": "Service Agreement", "risk_profile": "HIGH",
 "parties": ("Big Agency Corp", "Small Client Inc."),
 "sections": ["indemnification_HIGH","limitation_of_liability_HIGH","confidentiality_HIGH","ip_assignment_HIGH","payment_terms_HIGH","termination_HIGH","warranty_HIGH","governing_law_HIGH","dispute_resolution_HIGH"],
 "expected_clauses": ["indemnification","limitation_of_liability","confidentiality","ip_assignment","payment_terms","termination","warranty","governing_law","dispute_resolution"],
 "high_risk_clauses": ["indemnification","limitation_of_liability","confidentiality","ip_assignment","payment_terms","termination"], "missing_flag": []},

{"id": "svc_007", "type": "Service Agreement", "risk_profile": "MEDIUM",
 "parties": ("Tech Contractor LLC", "Finance Client"),
 "sections": ["indemnification_MEDIUM","limitation_of_liability_MEDIUM","confidentiality_LOW","ip_assignment_MEDIUM","payment_terms_MEDIUM","termination_MEDIUM","warranty_LOW","governing_law_LOW"],
 "expected_clauses": ["indemnification","limitation_of_liability","confidentiality","ip_assignment","payment_terms","termination","warranty","governing_law"],
 "high_risk_clauses": ["indemnification","limitation_of_liability","payment_terms"], "missing_flag": []},

# ══ Lease Agreements (4) ══════════════════════════════════════════════════════

{"id": "lease_001", "type": "Lease Agreement", "risk_profile": "LOW",
 "parties": ("Responsible Landlord LLC", "Tenant Corp"),
 "sections": ["lease_rent_LOW","lease_termination_LOW","lease_maintenance_LOW","confidentiality_LOW","governing_law_LOW"],
 "expected_clauses": ["governing_law"],
 "high_risk_clauses": [], "missing_flag": []},

{"id": "lease_002", "type": "Lease Agreement", "risk_profile": "LOW",
 "parties": ("Fair Property Management", "Office Tenant Inc."),
 "sections": ["lease_rent_LOW","lease_termination_LOW","lease_maintenance_LOW","governing_law_LOW","force_majeure_LOW"],
 "expected_clauses": ["governing_law","force_majeure"],
 "high_risk_clauses": [], "missing_flag": []},

{"id": "lease_003", "type": "Lease Agreement", "risk_profile": "HIGH",
 "parties": ("Aggressive Landlord Corp", "Small Business Tenant"),
 "sections": ["lease_rent_HIGH","lease_termination_HIGH","lease_maintenance_HIGH","indemnification_HIGH","governing_law_MEDIUM"],
 "expected_clauses": ["indemnification","governing_law"],
 "high_risk_clauses": ["indemnification"], "missing_flag": []},

{"id": "lease_004", "type": "Lease Agreement", "risk_profile": "HIGH",
 "parties": ("Predatory Properties LLC", "Startup Tenant"),
 "sections": ["lease_rent_HIGH","lease_termination_HIGH","lease_maintenance_HIGH","indemnification_HIGH","governing_law_HIGH","dispute_resolution_HIGH"],
 "expected_clauses": ["indemnification","governing_law","dispute_resolution"],
 "high_risk_clauses": ["indemnification","governing_law","dispute_resolution"], "missing_flag": []},

]

# ─────────────────────────────────────────────────────────────────────────────
# CONTRACT TEXT TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────

def build_contract_text(contract: dict) -> str:
    party_a, party_b = contract["parties"]
    ctype = contract["type"]
    cid = contract["id"]

    header = f"""{ctype.upper()}

This {ctype} (this "Agreement") is entered into as of January 1, 2025 (the "Effective Date"), by and between:

{party_a} ("Party A" / "Company" / "Vendor" / "Landlord" / "Employer"), and
{party_b} ("Party B" / "Customer" / "Employee" / "Tenant" / "Contractor").

RECITALS

The parties desire to enter into this Agreement on the terms set forth herein.

"""

    body_sections = []
    for section_key in contract["sections"]:
        body_sections.append(CLAUSES[section_key])

    body = "\n\n".join(body_sections)

    footer = f"""

GENERAL PROVISIONS

This Agreement constitutes the entire agreement between the parties with respect to its \
subject matter and supersedes all prior agreements. This Agreement may only be amended by \
a written instrument signed by both parties. If any provision is held unenforceable, the \
remaining provisions continue in full force. This Agreement may be executed in counterparts.

IN WITNESS WHEREOF, the parties have executed this Agreement as of the Effective Date.

{party_a.upper()}
By: ___________________________
Name:
Title:
Date: January 1, 2025

{party_b.upper()}
By: ___________________________
Name:
Title:
Date: January 1, 2025
"""
    return header + body + footer


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE FILES
# ─────────────────────────────────────────────────────────────────────────────

def generate():
    assert len(CONTRACTS) == 50, f"Expected 50 contracts, got {len(CONTRACTS)}"

    manifest = []

    for contract in CONTRACTS:
        text = build_contract_text(contract)
        filepath = OUTPUT_DIR / f"{contract['id']}.txt"
        with open(filepath, "w") as f:
            f.write(text)

        manifest.append({
            "id": contract["id"],
            "filename": f"{contract['id']}.txt",
            "contract_type": contract["type"],
            "risk_profile": contract["risk_profile"],
            "party_a": contract["parties"][0],
            "party_b": contract["parties"][1],
            "expected_clause_types": contract["expected_clauses"],
            "expected_high_risk_clause_types": contract["high_risk_clauses"],
            "expected_missing_clauses": contract["missing_flag"],
            "sections_used": contract["sections"],
        })

    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Summary
    by_type = {}
    by_risk = {}
    for c in manifest:
        by_type[c["contract_type"]] = by_type.get(c["contract_type"], 0) + 1
        by_risk[c["risk_profile"]] = by_risk.get(c["risk_profile"], 0) + 1

    print(f"\n✅ Generated {len(manifest)} contracts in {OUTPUT_DIR}")
    print(f"\nBy type:   {by_type}")
    print(f"By risk:   {by_risk}")
    print(f"Manifest:  {manifest_path}")

    return manifest


if __name__ == "__main__":
    generate()
