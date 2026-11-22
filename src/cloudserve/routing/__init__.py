"""Deterministic routing engine and policy enforcement for CloudServe AI Support System.

Exports:
- RouteOutcome: AUTO_RESPOND, DRAFT_FOR_REVIEW, ESCALATE.
- BillingRiskSubtype: GENERAL_INFORMATIONAL, ACCOUNT_SPECIFIC,
  DISPUTE_OR_FINANCIAL_DECISION, AMBIGUOUS.
- RoutingRequirementID: FR-04, FR-05, FR-06, FR-07, FR-12, FR-13.
- RoutingReasonCode: Auditable machine-readable routing reason codes.
- DeterministicRouter: Core deterministic policy decision engine.
- RoutingDecision: Complete typed routing result with audit trail and payloads.
- EscalationPayload: Structured handoff payload for human specialist escalation.
- ReviewDraftPayload: Structured unsent draft payload for tier-1 human review.
- assess_billing_risk: Billing risk classifier implementing FR-13.
- verify_citations: Strict citation resolver against retrieved passages.
"""

from cloudserve.routing.billing import BillingRiskAssessment, assess_billing_risk
from cloudserve.routing.citations import verify_citations
from cloudserve.routing.models import (
    BillingRiskSubtype,
    CitationValidationResult,
    EscalationPayload,
    GuardrailCheckResult,
    ReviewDraftPayload,
    RouteOutcome,
    RoutingDecision,
    RoutingReasonCode,
    RoutingRequirementID,
)
from cloudserve.routing.router import (
    DeterministicRouter,
    is_high_urgency_production_incident,
)

__all__ = [
    "BillingRiskAssessment",
    "BillingRiskSubtype",
    "CitationValidationResult",
    "DeterministicRouter",
    "EscalationPayload",
    "GuardrailCheckResult",
    "ReviewDraftPayload",
    "RouteOutcome",
    "RoutingDecision",
    "RoutingReasonCode",
    "RoutingRequirementID",
    "assess_billing_risk",
    "is_high_urgency_production_incident",
    "verify_citations",
]
