"""Data models, enums, and schemas for deterministic ticket routing.

Implements requirements:
- FR-04: Deterministic route decision with central configurable thresholds and policy precedence.
- FR-05: Auto-response eligibility (low-risk, verified evidence, valid citations, guardrails).
- FR-06: Human-review draft generation (uncertain/contextual, account-specific billing).
- FR-07: Mandatory specialist escalation (security, compliance, data residency, disputes, fallback).
- FR-12: Operational kill switch representation.
- FR-13: Billing risk classification (general, account-specific, dispute, ambiguous).
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from cloudserve.rag.models import RetrievedPassage


class RouteOutcome(str, Enum):
    """The three approved routing outcomes defined in PRD v2.0."""

    AUTO_RESPOND = "auto_respond"
    DRAFT_FOR_REVIEW = "draft_for_review"
    ESCALATE = "escalate"


class BillingRiskSubtype(str, Enum):
    """Billing risk sub-categories defined in FR-13."""

    GENERAL_INFORMATIONAL = "general_informational"
    ACCOUNT_SPECIFIC = "account_specific"
    DISPUTE_OR_FINANCIAL_DECISION = "dispute_or_financial_decision"
    AMBIGUOUS = "ambiguous"


class RoutingRequirementID(str, Enum):
    """PRD functional requirement identifiers for audit traceability."""

    FR_04 = "FR-04"
    FR_05 = "FR-05"
    FR_06 = "FR-06"
    FR_07 = "FR-07"
    FR_12 = "FR-12"
    FR_13 = "FR-13"


class RoutingReasonCode(str, Enum):
    """Machine-readable, auditable decision reason codes."""

    MANDATORY_SECURITY_INCIDENT = "MANDATORY_SECURITY_INCIDENT"
    MANDATORY_COMPLIANCE_REQUEST = "MANDATORY_COMPLIANCE_REQUEST"
    MANDATORY_DATA_RESIDENCY = "MANDATORY_DATA_RESIDENCY"
    MANDATORY_BILLING_DISPUTE = "MANDATORY_BILLING_DISPUTE"
    MANDATORY_FEATURE_REQUEST = "MANDATORY_FEATURE_REQUEST"
    MANDATORY_UNCLEAR_REQUEST = "MANDATORY_UNCLEAR_REQUEST"
    MANDATORY_CLASSIFIER_FALLBACK = "MANDATORY_CLASSIFIER_FALLBACK"
    HIGH_URGENCY_NO_KB_EVIDENCE = "HIGH_URGENCY_NO_KB_EVIDENCE"
    HIGH_URGENCY_PRODUCTION_INCIDENT = "HIGH_URGENCY_PRODUCTION_INCIDENT"
    BILLING_ACCOUNT_SPECIFIC_REVIEW = "BILLING_ACCOUNT_SPECIFIC_REVIEW"
    BILLING_AMBIGUOUS_RISK = "BILLING_AMBIGUOUS_RISK"
    GUARDRAIL_VIOLATION = "GUARDRAIL_VIOLATION"
    GUARDRAIL_UNAVAILABLE = "GUARDRAIL_UNAVAILABLE"
    GROUNDING_VALIDATION_FAILED = "GROUNDING_VALIDATION_FAILED"
    CITATION_VALIDATION_FAILED = "CITATION_VALIDATION_FAILED"
    KILL_SWITCH_ACTIVE = "KILL_SWITCH_ACTIVE"
    LOW_CONFIDENCE_REVIEW = "LOW_CONFIDENCE_REVIEW"
    LOW_CONFIDENCE_ESCALATE = "LOW_CONFIDENCE_ESCALATE"
    MISSING_OR_WEAK_KB_EVIDENCE = "MISSING_OR_WEAK_KB_EVIDENCE"
    ELIGIBLE_AUTO_RESPOND = "ELIGIBLE_AUTO_RESPOND"


class GuardrailCheckResult(BaseModel):
    """Safety guardrail validation status for routing evaluation."""

    model_config = ConfigDict(frozen=True)

    is_safe: bool = Field(
        ...,
        description="Explicit outcome of safety guardrail evaluation. Required with no default.",
    )
    violation_type: str | None = None
    details: str = ""


class CitationValidationResult(BaseModel):
    """Citation verification result against retrieved knowledge-base passages."""

    model_config = ConfigDict(frozen=True)

    is_valid: bool = Field(
        ...,
        description="Explicit outcome of citation validation. Required with no default.",
    )
    total_cited: int = 0
    valid_citations: list[str] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)
    details: str = ""


class EscalationPayload(BaseModel):
    """Structured handoff payload for human specialist escalation (FR-07)."""

    model_config = ConfigDict(frozen=True)

    ticket_id: str
    issue_summary: str
    risk_reason: str
    predicted_intent: str
    predicted_urgency: str
    confidence: float
    troubleshooting_history: str = Field(
        default="Troubleshooting history unavailable in ticket source.",
        description="Troubleshooting steps already attempted, or explicit unavailable note.",
    )
    relevant_passages: list[RetrievedPassage] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    documentation_available: bool = False
    documentation_notes: str = Field(
        default="Documentation unavailable in knowledge base.",
        description="Explicit note stating documentation availability or absence.",
    )
    documentation_excerpt: str | None = Field(
        default=None,
        description=(
            "Raw excerpt from top retrieved knowledge base passage for human agent context. "
            "Not verified customer-safe guidance."
        ),
    )
    safe_guidance: str | None = Field(
        default=None,
        description=(
            "Verified safe guidance synthesized in generation stage (Task 5); "
            "unverified excerpts must not be labeled safe guidance."
        ),
    )
    target_specialist_group: str = "Tier 2 Engineering"


class ReviewDraftPayload(BaseModel):
    """Structured unsent draft payload for tier-1 human review (FR-06)."""

    model_config = ConfigDict(frozen=True)

    ticket_id: str
    predicted_intent: str
    predicted_urgency: str
    confidence: float
    uncertainty_notes: str
    relevant_passages: list[RetrievedPassage] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    draft_response: str | None = None
    verification_checklist: list[str] = Field(default_factory=list)
    sent_to_customer: bool = False


class RoutingDecision(BaseModel):
    """Comprehensive, strongly-typed routing decision output contract.

    Contains the deterministic route, audit trail, requirement traces,
    applied thresholds, and typed handoff payloads.
    """

    model_config = ConfigDict(frozen=True)

    ticket_id: str
    route: RouteOutcome
    primary_requirement_id: str
    requirement_ids: list[str]
    reason_code: RoutingReasonCode
    reason_description: str
    billing_risk_subtype: BillingRiskSubtype | None = None
    intent_confidence_applied: float = Field(
        default=0.0,
        description="Primary confidence score evaluated against intent thresholds.",
    )
    urgency_confidence_applied: float = Field(
        default=0.0,
        description="Preserved raw urgency tier confidence score.",
    )
    confidence_applied: float = Field(
        default=0.0,
        description="Headline or primary confidence applied for reporting.",
    )
    confidence_threshold_applied: float
    retrieval_similarity_threshold_applied: float
    eligible_for_auto_respond: bool
    citations_verified: bool
    guardrails_passed: bool
    kill_switch_active: bool
    documentation_available: bool = Field(
        default=False,
        description="Whether qualifying approved knowledge-base documentation exists.",
    )
    sources_used: list[str] = Field(
        default_factory=list,
        description="Retrieved passage citation IDs retained across all route outcomes.",
    )
    escalation_payload: EscalationPayload | None = None
    review_payload: ReviewDraftPayload | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_decision_log_entry(self) -> dict[str, Any]:
        """Convert decision to an auditable dictionary for SQLite persistence (A8)."""
        billing_sub = self.billing_risk_subtype.value if self.billing_risk_subtype else None

        return {
            "ticket_id": self.ticket_id,
            "created_at": self.created_at.isoformat(),
            "route": self.route.value,
            "primary_requirement_id": self.primary_requirement_id,
            "requirement_ids": self.requirement_ids,
            "reason_code": self.reason_code.value,
            "reason_description": self.reason_description,
            "billing_risk_subtype": billing_sub,
            "intent_confidence_applied": self.intent_confidence_applied,
            "urgency_confidence_applied": self.urgency_confidence_applied,
            "confidence_applied": self.confidence_applied,
            "confidence_threshold_applied": self.confidence_threshold_applied,
            "retrieval_similarity_threshold_applied": self.retrieval_similarity_threshold_applied,
            "eligible_for_auto_respond": self.eligible_for_auto_respond,
            "citations_verified": self.citations_verified,
            "guardrails_passed": self.guardrails_passed,
            "kill_switch_active": self.kill_switch_active,
            "documentation_available": self.documentation_available,
            "sources_used": self.sources_used,
        }
