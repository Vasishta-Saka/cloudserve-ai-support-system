"""Billing risk classification engine implementing FR-13.

Distinguishes between:
1. General Informational Billing: High-level questions regarding billing policies,
   cycles, payment methods, proration formulas, or overage mechanics. AUTO_RESPOND
   eligible only when all gating criteria pass.
2. Account-Specific Billing: Inquiries regarding specific customer invoices, line
   items, unexplained charges, or usage spikes. Requires human verification (DRAFT_FOR_REVIEW).
3. Billing Disputes & Financial Decisions: Direct disputes, refund requests, credits,
   fee reversals, chargebacks, or contractual commitments. Requires human specialist (ESCALATE).
4. Ambiguous: Uncertain or conflicting signals. Must NEVER auto-respond.
"""

import re
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from cloudserve.ingestion.models import UnifiedTicket
from cloudserve.routing.models import BillingRiskSubtype

DISPUTE_PATTERNS: Final[list[str]] = [
    r"\bdispute\b",
    r"\bdisputing\b",
    r"\bdisputed\b",
    r"\brefund\b",
    r"\brefunds\b",
    r"\bmoney back\b",
    r"\breimburse\w*",
    r"\bbilling credits?\b",
    r"\bservice credits?\b",
    r"\baccount credits?\b",
    r"\bfee credits?\b",
    r"\bcredit adjustments?\b",
    r"\b(?:issue|request|apply|give|demand|receive|grant|need|want) (?:a |some )?credits?\b",
    r"\bcredits? (?:for|due to) (?:the )?(?:downtime|outage|interruption|sla|inconvenience)\b",
    r"\brefund or credits?\b",
    r"\bcredit (?:our|my) account\b",
    r"\breversal\b",
    r"\breverse charge\b",
    r"\bchargeback\w*",
    r"\bcompensation\b",
    r"\bwaive\b",
    r"\bwaiver\b",
    r"\bovercharg\w*",
    r"\bwrongly charged\b",
    r"\bincorrectly charged\b",
    r"\bcontractual commitment\b",
    r"\bprice match\b",
]

ACCOUNT_SPECIFIC_PATTERNS: Final[list[str]] = [
    r"\bour invoice\b",
    r"\bmy invoice\b",
    r"\bthe invoice\b",
    r"\bthis invoice\b",
    r"\bthis month(?:'s)? (?:charges?|bill|invoice)\b",
    r"\blast month(?:'s)? (?:charges?|bill|invoice)\b",
    r"\bcharges? on (?:our|my|the) invoice\b",
    r"\binvoice showing\b",
    r"\bwhy was i charged\b",
    r"\bwhy are we charged\b",
    r"\bhigher than (?:usual|expected|normal)\b",
    r"\b(?:we do not|do not|not|didn't) believe we use\b",
    r"\bservice we (?:not|didn't) use\b",
    r"\bcheck and tell\b",
    r"\bexplain what these relate to\b",
    r"\binv-\d+\b",
    r"\binvoice #\s*\d+\b",
    r"\bline item\b",
    r"\bunexpected charge\b",
    r"\bcharge of \$?\d+\b",
]

GENERAL_INFORMATIONAL_PATTERNS: Final[list[str]] = [
    r"\bbilling period\b",
    r"\bcalendar month\b",
    r"\bbilling cycle\b",
    r"\bproration\b",
    r"\bhow is proration calculated\b",
    r"\bpayment method\b",
    r"\baccepted payment\b",
    r"\bupdate payment\b",
    r"\bcredit cards?\b",
    r"\bdebit cards?\b",
    r"\bspend control\b",
    r"\busage limit\b",
    r"\boverage policy\b",
    r"\bhow (?:do|does|are) .* (?:work|calculated|billed)\b",
    r"\bwhat (?:is|are) the billing\b",
]


class BillingRiskAssessment(BaseModel):
    """Result of billing risk assessment."""

    model_config = ConfigDict(frozen=True)

    subtype: BillingRiskSubtype
    rationale: str
    matched_patterns: list[str] = Field(default_factory=list)


def assess_billing_risk(ticket: UnifiedTicket) -> BillingRiskAssessment:
    """Assess the billing risk profile of an incoming ticket under FR-13.

    Args:
        ticket: Ingested UnifiedTicket (containing subject and body).

    Returns:
        BillingRiskAssessment with subtype and auditable rationale.
    """
    text = ticket.clean_text.lower()
    if not text:
        return BillingRiskAssessment(
            subtype=BillingRiskSubtype.AMBIGUOUS,
            rationale="Ticket content is empty; cannot determine billing subtype.",
        )

    # 1. Check for Dispute or Financial Decision (Highest risk -> Escalate)
    matched_disputes: list[str] = []
    for pattern in DISPUTE_PATTERNS:
        if re.search(pattern, text):
            matched_disputes.append(pattern)

    if matched_disputes:
        return BillingRiskAssessment(
            subtype=BillingRiskSubtype.DISPUTE_OR_FINANCIAL_DECISION,
            rationale=(
                f"Financial dispute or adjustment request identified matching patterns: "
                f"{matched_disputes}. Mandatory specialist escalation required (FR-13)."
            ),
            matched_patterns=matched_disputes,
        )

    # 2. Check for Account-Specific Invoice/Charge Investigation (Requires human review)
    matched_account: list[str] = []
    for pattern in ACCOUNT_SPECIFIC_PATTERNS:
        if re.search(pattern, text):
            matched_account.append(pattern)

    if matched_account:
        return BillingRiskAssessment(
            subtype=BillingRiskSubtype.ACCOUNT_SPECIFIC,
            rationale=(
                f"Account-specific billing or invoice investigation detected matching patterns: "
                f"{matched_account}. Human review required to verify account facts (FR-13)."
            ),
            matched_patterns=matched_account,
        )

    # 3. Check for General Informational Billing Questions (Auto-respond eligible)
    matched_general: list[str] = []
    for pattern in GENERAL_INFORMATIONAL_PATTERNS:
        if re.search(pattern, text):
            matched_general.append(pattern)

    if matched_general:
        return BillingRiskAssessment(
            subtype=BillingRiskSubtype.GENERAL_INFORMATIONAL,
            rationale=(
                f"General informational billing question identified matching patterns: "
                f"{matched_general}. Auto-response eligible if all other gates pass (FR-13)."
            ),
            matched_patterns=matched_general,
        )

    # 4. Ambiguous or Uncategorized Billing Query (Hold/Review, Never Auto-respond)
    return BillingRiskAssessment(
        subtype=BillingRiskSubtype.AMBIGUOUS,
        rationale=(
            "Billing query wording is ambiguous or non-standard. "
            "Policy strictly prohibits auto-responding to ambiguous billing (FR-13)."
        ),
    )
