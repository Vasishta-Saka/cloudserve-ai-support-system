"""Deterministic routing engine for the CloudServe AI Support System.

Implements authoritative routing requirements from PRD v2.0 (FR-04 to FR-07, FR-12, FR-13).
Completely decoupled from the language model: deterministic business logic governs
the final route decision. Language models classify and draft; software routes.
"""

import logging
import re
from collections.abc import Sequence

from cloudserve.classification.models import (
    ClassificationResult,
    IntentCategory,
    UrgencyLevel,
)
from cloudserve.config import Settings, get_settings
from cloudserve.ingestion.models import UnifiedTicket
from cloudserve.rag.models import RetrievedPassage
from cloudserve.routing.billing import assess_billing_risk
from cloudserve.routing.citations import verify_citations
from cloudserve.routing.models import (
    BillingRiskSubtype,
    EscalationPayload,
    GuardrailCheckResult,
    ReviewDraftPayload,
    RouteOutcome,
    RoutingDecision,
    RoutingReasonCode,
    RoutingRequirementID,
)

logger = logging.getLogger(__name__)


def is_high_urgency_production_incident(
    ticket: UnifiedTicket,
    classification: ClassificationResult,
) -> bool:
    """Evaluate whether ticket qualifies as an active high-urgency production incident.

    Qualifying tickets involve active production outages, disruptions, or emergency rollbacks
    under High urgency and specific operational intents (REC-04, FR-07). These require mandatory
    escalation to Tier 2 Engineering and must never be auto-responded with static KB docs.
    """
    if classification.urgency != UrgencyLevel.HIGH:
        return False

    if classification.intent not in (
        IntentCategory.ROLLBACK_REQUEST,
        IntentCategory.DEPLOYMENT_FAILURE,
        IntentCategory.PERFORMANCE_DEGRADATION,
        IntentCategory.DATABASE_ISSUE,
    ):
        return False

    ticket_text = f"{ticket.subject} {ticket.body}".lower()
    has_production_scope = bool(
        re.search(r"\b(?:production|prod|live environment|customer-facing)\b", ticket_text)
    )
    has_active_outage_or_damage = bool(
        re.search(
            r"\b(?:"
            r"outage|downtime|down|broken checkout|broken payment|"
            r"customers? (?:affected|impacted|cannot|can't)|"
            r"users? (?:affected|impacted|cannot|can't)|"
            r"critical|sev-?1|severity 1|emergency|site down"
            r")\b",
            ticket_text,
        )
    )
    return has_production_scope and has_active_outage_or_damage


class DeterministicRouter:
    """Deterministic routing engine implementing CloudServe PRD v2.0 policy."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize router with centralized application settings."""
        self.settings = settings or get_settings()

    @property
    def confidence_threshold_auto_respond(self) -> float:
        """Active confidence threshold for auto-response eligibility."""
        return float(self.settings.confidence_threshold_auto_respond)

    @property
    def confidence_threshold_review(self) -> float:
        """Active confidence threshold for human-review drafting."""
        return float(self.settings.confidence_threshold_review)

    def route_ticket(
        self,
        ticket: UnifiedTicket,
        classification: ClassificationResult,
        retrieved_passages: Sequence[RetrievedPassage] | None = None,
        guardrail_result: GuardrailCheckResult | None = None,
        cited_ids: Sequence[str] | None = None,
        override_kill_switch: bool | None = None,
        confidence_threshold_auto_respond: float | None = None,
        confidence_threshold_review: float | None = None,
        retrieval_similarity_threshold: float | None = None,
    ) -> RoutingDecision:
        """Compute the deterministic routing decision for an incoming ticket.

        Decision Precedence:
        1. Safety Guardrail Failure -> Escalate / Block customer release (FR-04, FR-05)
        2. Mandatory Policy Escalations -> Escalate regardless of confidence/KB (FR-07)
           (Classifier Fallback, Security, Compliance, Data Residency, Feature Requests,
            Unclear, Billing Disputes & Financial Decisions)
        3. Billing Risk Assessment -> General (Auto-eligible), Account (Review), Ambiguous (FR-13)
        4. Knowledge Base Evidence & Relevance Gate -> Missing/weak KB holds as review draft
           or escalates on low confidence (FR-05, FR-06)
        5. Fail-Closed Guardrails -> Missing/unavailable guardrails block auto-respond (FR-04)
        6. Citation Verification -> Missing, empty, or unretrieved citations block release (FR-05)
        7. Operational Kill Switch -> Disables auto-respond, holds as Review Draft (FR-12)
        8. Intent Confidence Thresholds -> Auto-respond vs Review Draft vs Escalate (FR-04, FR-05)

        Args:
            ticket: Ingested UnifiedTicket (containing clean_text, metadata).
            classification: Structured output from ClassificationEngine.
            retrieved_passages: Ranked list of knowledge-base passages from KBRetriever.
            guardrail_result: Output from active safety guardrails (injection, PII, policy).
            cited_ids: Optional list of citation IDs cited in generated response.
            override_kill_switch: Optional override for operational kill switch.
            confidence_threshold_auto_respond: Central config override for auto-respond threshold.
            confidence_threshold_review: Central config override for review threshold.
            retrieval_similarity_threshold: Central config override for retrieval relevance.

        Returns:
            Strongly-typed RoutingDecision with route, audit reason, requirement IDs, and handoff.
        """
        # Resolve central configurable settings and overrides
        auto_thresh = (
            confidence_threshold_auto_respond
            if confidence_threshold_auto_respond is not None
            else self.settings.confidence_threshold_auto_respond
        )
        review_thresh = (
            confidence_threshold_review
            if confidence_threshold_review is not None
            else self.settings.confidence_threshold_review
        )
        sim_thresh = (
            retrieval_similarity_threshold
            if retrieval_similarity_threshold is not None
            else self.settings.retrieval_similarity_threshold
        )
        kill_switch_active = not (
            override_kill_switch
            if override_kill_switch is not None
            else self.settings.enable_auto_respond
        )

        # Filter retrieved passages by similarity threshold
        raw_passages = retrieved_passages or []
        passages = [p for p in raw_passages if p.similarity_score >= sim_thresh]
        has_docs = len(passages) > 0
        sources_used = [p.citation_id for p in passages]

        # Confidence fields: intent_confidence governs the intent threshold gate,
        # while urgency_confidence is preserved separately for audit and downstream SLA.
        intent_conf = classification.intent_confidence
        urgency_conf = classification.urgency_confidence

        # Billing risk assessment
        billing_risk: BillingRiskSubtype | None = None
        if classification.intent == IntentCategory.BILLING_QUERY:
            billing_assessment = assess_billing_risk(ticket)
            billing_risk = billing_assessment.subtype

        # Helper to construct escalation payload
        def build_escalation(
            risk_reason: str,
            target_group: str = "Tier 2 Engineering",
        ) -> EscalationPayload:
            doc_notes = (
                f"Approved documentation available: {len(passages)} passages attached."
                if has_docs
                else "Relevant documentation unavailable in knowledge base."
            )
            # Label raw chunk content as an unverified excerpt, not verified safe guidance
            doc_excerpt = (
                f"Excerpt from {passages[0].citation_id}: {passages[0].content[:250]}..."
                if has_docs
                else None
            )
            summary = (
                ticket.subject
                if ticket.subject
                else (ticket.body[:150] + "..." if len(ticket.body) > 150 else ticket.body)
            )
            return EscalationPayload(
                ticket_id=ticket.ticket_id,
                issue_summary=summary,
                risk_reason=risk_reason,
                predicted_intent=classification.intent.value,
                predicted_urgency=classification.urgency.value,
                confidence=intent_conf,
                troubleshooting_history="Troubleshooting history unavailable in ticket source.",
                relevant_passages=list(passages),
                citations=[p.citation_id for p in passages],
                documentation_available=has_docs,
                documentation_notes=doc_notes,
                documentation_excerpt=doc_excerpt,
                safe_guidance=None,
                target_specialist_group=target_group,
            )

        # Helper to construct review draft payload
        def build_review_draft(
            uncertainty_notes: str,
            checklist: list[str] | None = None,
        ) -> ReviewDraftPayload:
            default_checklist = [
                "Review predicted intent and urgency against ticket content",
                "Verify factual accuracy of attached KB evidence",
                "Approve and personalize draft response before sending to customer",
            ]
            return ReviewDraftPayload(
                ticket_id=ticket.ticket_id,
                predicted_intent=classification.intent.value,
                predicted_urgency=classification.urgency.value,
                confidence=intent_conf,
                uncertainty_notes=uncertainty_notes,
                relevant_passages=list(passages),
                citations=[p.citation_id for p in passages],
                verification_checklist=checklist or default_checklist,
                sent_to_customer=False,
            )

        # =========================================================================
        # 1. Active Safety Guardrail Failure Gate (FR-04, FR-05)
        # =========================================================================
        # Genuine security violations (prompt injection, PII exfiltration, unauthorized actions)
        # are escalated immediately to Security Operations with highest precedence.
        # Note: Non-malicious output grounding failures are evaluated under Gate 5 (post-policy)
        # to ensure mandatory operational policies (Gate 2) take precedence.
        if (
            guardrail_result is not None
            and not guardrail_result.is_safe
            and guardrail_result.violation_type != "output_grounding_failed"
        ):
            violation = guardrail_result.violation_type or "Adversarial or unsafe input detected"
            reason_desc = f"Safety guardrail triggered: {violation}. {guardrail_result.details}"

            return RoutingDecision(
                ticket_id=ticket.ticket_id,
                route=RouteOutcome.ESCALATE,
                primary_requirement_id=RoutingRequirementID.FR_04.value,
                requirement_ids=[
                    RoutingRequirementID.FR_04.value,
                    RoutingRequirementID.FR_05.value,
                ],
                reason_code=RoutingReasonCode.GUARDRAIL_VIOLATION,
                reason_description=reason_desc,
                billing_risk_subtype=billing_risk,
                intent_confidence_applied=intent_conf,
                urgency_confidence_applied=urgency_conf,
                confidence_applied=intent_conf,
                confidence_threshold_applied=auto_thresh,
                retrieval_similarity_threshold_applied=sim_thresh,
                eligible_for_auto_respond=False,
                citations_verified=False,
                guardrails_passed=False,
                kill_switch_active=kill_switch_active,
                documentation_available=has_docs,
                sources_used=sources_used,
                escalation_payload=build_escalation(
                    reason_desc, target_group="Security Operations"
                ),
            )

        # =========================================================================
        # 2. Mandatory Policy Escalations (FR-07) - Highest Precedence
        # =========================================================================

        # 2a. Classifier Fallback (Prediction Failure or Degradation)
        if classification.is_fallback:
            reason_desc = (
                "Classifier fallback triggered due to prediction failure or uncertainty. "
                "Mandatory escalation; fallback can never auto-respond (FR-07)."
            )
            return RoutingDecision(
                ticket_id=ticket.ticket_id,
                route=RouteOutcome.ESCALATE,
                primary_requirement_id=RoutingRequirementID.FR_07.value,
                requirement_ids=[
                    RoutingRequirementID.FR_04.value,
                    RoutingRequirementID.FR_07.value,
                ],
                reason_code=RoutingReasonCode.MANDATORY_CLASSIFIER_FALLBACK,
                reason_description=reason_desc,
                billing_risk_subtype=billing_risk,
                intent_confidence_applied=intent_conf,
                urgency_confidence_applied=urgency_conf,
                confidence_applied=intent_conf,
                confidence_threshold_applied=auto_thresh,
                retrieval_similarity_threshold_applied=sim_thresh,
                eligible_for_auto_respond=False,
                citations_verified=False,
                guardrails_passed=guardrail_result.is_safe if guardrail_result else False,
                kill_switch_active=kill_switch_active,
                documentation_available=has_docs,
                sources_used=sources_used,
                escalation_payload=build_escalation(
                    reason_desc, target_group="Tier 1 Support Agent"
                ),
            )

        # 2b. Security Incident or Account Compromise
        if classification.intent == IntentCategory.SECURITY_INCIDENT:
            reason_desc = (
                "Security incident or suspected account compromise. "
                "Mandatory immediate escalation to Security Operations (FR-07)."
            )
            return RoutingDecision(
                ticket_id=ticket.ticket_id,
                route=RouteOutcome.ESCALATE,
                primary_requirement_id=RoutingRequirementID.FR_07.value,
                requirement_ids=[
                    RoutingRequirementID.FR_04.value,
                    RoutingRequirementID.FR_07.value,
                ],
                reason_code=RoutingReasonCode.MANDATORY_SECURITY_INCIDENT,
                reason_description=reason_desc,
                billing_risk_subtype=billing_risk,
                intent_confidence_applied=intent_conf,
                urgency_confidence_applied=urgency_conf,
                confidence_applied=intent_conf,
                confidence_threshold_applied=auto_thresh,
                retrieval_similarity_threshold_applied=sim_thresh,
                eligible_for_auto_respond=False,
                citations_verified=False,
                guardrails_passed=guardrail_result.is_safe if guardrail_result else False,
                kill_switch_active=kill_switch_active,
                documentation_available=has_docs,
                sources_used=sources_used,
                escalation_payload=build_escalation(
                    reason_desc, target_group="Security Operations"
                ),
            )

        # 2c. Compliance Request
        if classification.intent == IntentCategory.COMPLIANCE_REQUEST:
            reason_desc = (
                "Compliance, regulatory, or audit inquiry (GDPR/SOC2/HIPAA). "
                "Mandatory escalation to Compliance & Legal specialist (FR-07)."
            )
            return RoutingDecision(
                ticket_id=ticket.ticket_id,
                route=RouteOutcome.ESCALATE,
                primary_requirement_id=RoutingRequirementID.FR_07.value,
                requirement_ids=[
                    RoutingRequirementID.FR_04.value,
                    RoutingRequirementID.FR_07.value,
                ],
                reason_code=RoutingReasonCode.MANDATORY_COMPLIANCE_REQUEST,
                reason_description=reason_desc,
                billing_risk_subtype=billing_risk,
                intent_confidence_applied=intent_conf,
                urgency_confidence_applied=urgency_conf,
                confidence_applied=intent_conf,
                confidence_threshold_applied=auto_thresh,
                retrieval_similarity_threshold_applied=sim_thresh,
                eligible_for_auto_respond=False,
                citations_verified=False,
                guardrails_passed=guardrail_result.is_safe if guardrail_result else False,
                kill_switch_active=kill_switch_active,
                documentation_available=has_docs,
                sources_used=sources_used,
                escalation_payload=build_escalation(reason_desc, target_group="Compliance & Legal"),
            )

        # 2d. Data Residency / Data Location
        if classification.intent == IntentCategory.DATA_RESIDENCY:
            reason_desc = (
                "Data location or residency request. Mandatory escalation; "
                "automated residency assurances strictly prohibited (FR-07, PRD v2.0 §10.2)."
            )
            return RoutingDecision(
                ticket_id=ticket.ticket_id,
                route=RouteOutcome.ESCALATE,
                primary_requirement_id=RoutingRequirementID.FR_07.value,
                requirement_ids=[
                    RoutingRequirementID.FR_04.value,
                    RoutingRequirementID.FR_07.value,
                ],
                reason_code=RoutingReasonCode.MANDATORY_DATA_RESIDENCY,
                reason_description=reason_desc,
                billing_risk_subtype=billing_risk,
                intent_confidence_applied=intent_conf,
                urgency_confidence_applied=urgency_conf,
                confidence_applied=intent_conf,
                confidence_threshold_applied=auto_thresh,
                retrieval_similarity_threshold_applied=sim_thresh,
                eligible_for_auto_respond=False,
                citations_verified=False,
                guardrails_passed=guardrail_result.is_safe if guardrail_result else False,
                kill_switch_active=kill_switch_active,
                documentation_available=has_docs,
                sources_used=sources_used,
                escalation_payload=build_escalation(
                    reason_desc, target_group="Compliance & Infrastructure"
                ),
            )

        # 2e. Feature / Roadmap Request
        if classification.intent == IntentCategory.FEATURE_REQUEST:
            reason_desc = (
                "Feature or product roadmap request. Escalated to Product Management; "
                "automated delivery promises strictly prohibited (FR-07)."
            )
            return RoutingDecision(
                ticket_id=ticket.ticket_id,
                route=RouteOutcome.ESCALATE,
                primary_requirement_id=RoutingRequirementID.FR_07.value,
                requirement_ids=[
                    RoutingRequirementID.FR_04.value,
                    RoutingRequirementID.FR_07.value,
                ],
                reason_code=RoutingReasonCode.MANDATORY_FEATURE_REQUEST,
                reason_description=reason_desc,
                billing_risk_subtype=billing_risk,
                intent_confidence_applied=intent_conf,
                urgency_confidence_applied=urgency_conf,
                confidence_applied=intent_conf,
                confidence_threshold_applied=auto_thresh,
                retrieval_similarity_threshold_applied=sim_thresh,
                eligible_for_auto_respond=False,
                citations_verified=False,
                guardrails_passed=guardrail_result.is_safe if guardrail_result else False,
                kill_switch_active=kill_switch_active,
                documentation_available=has_docs,
                sources_used=sources_used,
                escalation_payload=build_escalation(reason_desc, target_group="Product Management"),
            )

        # 2f. Unclear Request
        if classification.intent == IntentCategory.UNCLEAR_REQUEST:
            reason_desc = (
                "Unclear or unintelligible customer request. Mandatory human escalation "
                "for manual clarification (FR-07)."
            )
            return RoutingDecision(
                ticket_id=ticket.ticket_id,
                route=RouteOutcome.ESCALATE,
                primary_requirement_id=RoutingRequirementID.FR_07.value,
                requirement_ids=[
                    RoutingRequirementID.FR_04.value,
                    RoutingRequirementID.FR_07.value,
                ],
                reason_code=RoutingReasonCode.MANDATORY_UNCLEAR_REQUEST,
                reason_description=reason_desc,
                billing_risk_subtype=billing_risk,
                intent_confidence_applied=intent_conf,
                urgency_confidence_applied=urgency_conf,
                confidence_applied=intent_conf,
                confidence_threshold_applied=auto_thresh,
                retrieval_similarity_threshold_applied=sim_thresh,
                eligible_for_auto_respond=False,
                citations_verified=False,
                guardrails_passed=guardrail_result.is_safe if guardrail_result else False,
                kill_switch_active=kill_switch_active,
                documentation_available=has_docs,
                sources_used=sources_used,
                escalation_payload=build_escalation(
                    reason_desc, target_group="Tier 1 Support Agent"
                ),
            )

        # 2g. Billing Dispute / Financial Decision (FR-07, FR-13)
        # Evaluated as a mandatory policy escalation so a billing dispute always
        # routes to Billing Operations and is never intercepted by generic engineering gates.
        if (
            classification.intent == IntentCategory.BILLING_QUERY
            and billing_risk == BillingRiskSubtype.DISPUTE_OR_FINANCIAL_DECISION
        ):
            reason_desc = (
                "Billing dispute or financial commitment request (refund/credit/reversal). "
                "Mandatory escalation to Billing Operations specialist (FR-07, FR-13)."
            )
            return RoutingDecision(
                ticket_id=ticket.ticket_id,
                route=RouteOutcome.ESCALATE,
                primary_requirement_id=RoutingRequirementID.FR_13.value,
                requirement_ids=[
                    RoutingRequirementID.FR_04.value,
                    RoutingRequirementID.FR_07.value,
                    RoutingRequirementID.FR_13.value,
                ],
                reason_code=RoutingReasonCode.MANDATORY_BILLING_DISPUTE,
                reason_description=reason_desc,
                billing_risk_subtype=billing_risk,
                intent_confidence_applied=intent_conf,
                urgency_confidence_applied=urgency_conf,
                confidence_applied=intent_conf,
                confidence_threshold_applied=auto_thresh,
                retrieval_similarity_threshold_applied=sim_thresh,
                eligible_for_auto_respond=False,
                citations_verified=False,
                guardrails_passed=guardrail_result.is_safe if guardrail_result else False,
                kill_switch_active=kill_switch_active,
                documentation_available=has_docs,
                sources_used=sources_used,
                escalation_payload=build_escalation(reason_desc, target_group="Billing Operations"),
            )

        # 2h. High-Urgency Active Production Outage & Emergency Rollback (FR-07, REC-04)
        # Escalates active production disruptions and emergency rollbacks to Tier 2 Engineering.
        # Active production outages must never be automatically answered with static KB docs.
        if is_high_urgency_production_incident(ticket, classification):
            reason_desc = (
                "Active high-urgency production disruption or emergency rollback situation. "
                "Mandatory immediate escalation to Tier 2 Engineering; automated customer "
                "resolution prohibited during live production incident (FR-07, REC-04)."
            )
            return RoutingDecision(
                ticket_id=ticket.ticket_id,
                route=RouteOutcome.ESCALATE,
                primary_requirement_id=RoutingRequirementID.FR_07.value,
                requirement_ids=[
                    RoutingRequirementID.FR_04.value,
                    RoutingRequirementID.FR_07.value,
                ],
                reason_code=RoutingReasonCode.HIGH_URGENCY_PRODUCTION_INCIDENT,
                reason_description=reason_desc,
                billing_risk_subtype=billing_risk,
                intent_confidence_applied=intent_conf,
                urgency_confidence_applied=urgency_conf,
                confidence_applied=intent_conf,
                confidence_threshold_applied=auto_thresh,
                retrieval_similarity_threshold_applied=sim_thresh,
                eligible_for_auto_respond=False,
                citations_verified=False,
                guardrails_passed=guardrail_result.is_safe if guardrail_result else False,
                kill_switch_active=kill_switch_active,
                documentation_available=has_docs,
                sources_used=sources_used,
                escalation_payload=build_escalation(reason_desc, target_group="Tier 2 Engineering"),
            )

        # =========================================================================
        # 3. Billing Risk Classification (FR-13)
        # =========================================================================
        # Note: Billing disputes & financial decisions are handled with highest
        # precedence under Gate 2g (Mandatory Policy Escalations).
        if classification.intent == IntentCategory.BILLING_QUERY:
            if billing_risk == BillingRiskSubtype.ACCOUNT_SPECIFIC:
                reason_desc = (
                    "Account-specific billing/invoice inquiry. Prepared unsent draft; "
                    "human agent must verify customer invoice facts before sending "
                    "(FR-13, FR-06)."
                )
                checklist = [
                    "Verify invoice number and billing period in Stripe/CRM",
                    "Confirm line-item charges against actual resource usage",
                    "Check for proration adjustments or plan change dates",
                    "Approve and personalize draft response before sending to customer",
                ]
                return RoutingDecision(
                    ticket_id=ticket.ticket_id,
                    route=RouteOutcome.DRAFT_FOR_REVIEW,
                    primary_requirement_id=RoutingRequirementID.FR_13.value,
                    requirement_ids=[
                        RoutingRequirementID.FR_04.value,
                        RoutingRequirementID.FR_06.value,
                        RoutingRequirementID.FR_13.value,
                    ],
                    reason_code=RoutingReasonCode.BILLING_ACCOUNT_SPECIFIC_REVIEW,
                    reason_description=reason_desc,
                    billing_risk_subtype=billing_risk,
                    intent_confidence_applied=intent_conf,
                    urgency_confidence_applied=urgency_conf,
                    confidence_applied=intent_conf,
                    confidence_threshold_applied=auto_thresh,
                    retrieval_similarity_threshold_applied=sim_thresh,
                    eligible_for_auto_respond=False,
                    citations_verified=False,
                    guardrails_passed=guardrail_result.is_safe if guardrail_result else False,
                    kill_switch_active=kill_switch_active,
                    documentation_available=has_docs,
                    sources_used=sources_used,
                    review_payload=build_review_draft(reason_desc, checklist=checklist),
                )

            elif billing_risk == BillingRiskSubtype.AMBIGUOUS:
                reason_desc = (
                    "Ambiguous billing inquiry. Policy strictly prohibits auto-responding "
                    "to ambiguous billing questions (FR-13)."
                )
                return RoutingDecision(
                    ticket_id=ticket.ticket_id,
                    route=RouteOutcome.DRAFT_FOR_REVIEW,
                    primary_requirement_id=RoutingRequirementID.FR_13.value,
                    requirement_ids=[
                        RoutingRequirementID.FR_04.value,
                        RoutingRequirementID.FR_06.value,
                        RoutingRequirementID.FR_13.value,
                    ],
                    reason_code=RoutingReasonCode.BILLING_AMBIGUOUS_RISK,
                    reason_description=reason_desc,
                    billing_risk_subtype=billing_risk,
                    intent_confidence_applied=intent_conf,
                    urgency_confidence_applied=urgency_conf,
                    confidence_applied=intent_conf,
                    confidence_threshold_applied=auto_thresh,
                    retrieval_similarity_threshold_applied=sim_thresh,
                    eligible_for_auto_respond=False,
                    citations_verified=False,
                    guardrails_passed=guardrail_result.is_safe if guardrail_result else False,
                    kill_switch_active=kill_switch_active,
                    documentation_available=has_docs,
                    sources_used=sources_used,
                    review_payload=build_review_draft(reason_desc),
                )

        # =========================================================================
        # 4. Knowledge Base Evidence & Relevance Gate (FR-05, FR-06)
        # =========================================================================
        if not has_docs:
            reason_desc = (
                f"No knowledge-base passages met similarity threshold ({sim_thresh:.2f}). "
                "Technical answers must never be invented without approved documentation (FR-05)."
            )
            if intent_conf >= review_thresh:
                return RoutingDecision(
                    ticket_id=ticket.ticket_id,
                    route=RouteOutcome.DRAFT_FOR_REVIEW,
                    primary_requirement_id=RoutingRequirementID.FR_06.value,
                    requirement_ids=[
                        RoutingRequirementID.FR_04.value,
                        RoutingRequirementID.FR_05.value,
                        RoutingRequirementID.FR_06.value,
                    ],
                    reason_code=RoutingReasonCode.MISSING_OR_WEAK_KB_EVIDENCE,
                    reason_description=reason_desc,
                    billing_risk_subtype=billing_risk,
                    intent_confidence_applied=intent_conf,
                    urgency_confidence_applied=urgency_conf,
                    confidence_applied=intent_conf,
                    confidence_threshold_applied=review_thresh,
                    retrieval_similarity_threshold_applied=sim_thresh,
                    eligible_for_auto_respond=False,
                    citations_verified=False,
                    guardrails_passed=guardrail_result.is_safe if guardrail_result else False,
                    kill_switch_active=kill_switch_active,
                    documentation_available=has_docs,
                    sources_used=sources_used,
                    review_payload=build_review_draft(reason_desc),
                )
            else:
                return RoutingDecision(
                    ticket_id=ticket.ticket_id,
                    route=RouteOutcome.ESCALATE,
                    primary_requirement_id=RoutingRequirementID.FR_07.value,
                    requirement_ids=[
                        RoutingRequirementID.FR_04.value,
                        RoutingRequirementID.FR_05.value,
                        RoutingRequirementID.FR_07.value,
                    ],
                    reason_code=RoutingReasonCode.LOW_CONFIDENCE_ESCALATE,
                    reason_description=(
                        f"{reason_desc} Low confidence ({intent_conf:.2f} < {review_thresh:.2f})."
                    ),
                    billing_risk_subtype=billing_risk,
                    intent_confidence_applied=intent_conf,
                    urgency_confidence_applied=urgency_conf,
                    confidence_applied=intent_conf,
                    confidence_threshold_applied=review_thresh,
                    retrieval_similarity_threshold_applied=sim_thresh,
                    eligible_for_auto_respond=False,
                    citations_verified=False,
                    guardrails_passed=guardrail_result.is_safe if guardrail_result else False,
                    kill_switch_active=kill_switch_active,
                    documentation_available=has_docs,
                    sources_used=sources_used,
                    escalation_payload=build_escalation(reason_desc),
                )

        # =========================================================================
        # 5. Output Grounding and Fail-Closed Safety Guardrails Gate (FR-04, FR-05)
        # =========================================================================
        # Non-malicious output grounding failures and missing/unavailable guardrail
        # results must NEVER permit AUTO_RESPOND.
        # Under fail-closed policy, unverified or ungrounded tickets are held safely
        # as review drafts for human review, unless already escalated by higher policy.
        if guardrail_result is not None and not guardrail_result.is_safe:
            if guardrail_result.violation_type == "output_grounding_failed":
                reason_desc = (
                    f"Safety guardrail triggered: {guardrail_result.violation_type}. "
                    f"{guardrail_result.details}"
                )
                return RoutingDecision(
                    ticket_id=ticket.ticket_id,
                    route=RouteOutcome.DRAFT_FOR_REVIEW,
                    primary_requirement_id=RoutingRequirementID.FR_05.value,
                    requirement_ids=[
                        RoutingRequirementID.FR_04.value,
                        RoutingRequirementID.FR_05.value,
                        RoutingRequirementID.FR_06.value,
                    ],
                    reason_code=RoutingReasonCode.GROUNDING_VALIDATION_FAILED,
                    reason_description=reason_desc,
                    billing_risk_subtype=billing_risk,
                    intent_confidence_applied=intent_conf,
                    urgency_confidence_applied=urgency_conf,
                    confidence_applied=intent_conf,
                    confidence_threshold_applied=auto_thresh,
                    retrieval_similarity_threshold_applied=sim_thresh,
                    eligible_for_auto_respond=False,
                    citations_verified=False,
                    guardrails_passed=False,
                    kill_switch_active=kill_switch_active,
                    documentation_available=has_docs,
                    sources_used=sources_used,
                    review_payload=build_review_draft(
                        reason_desc,
                        checklist=[
                            "Review predicted intent and urgency against customer ticket",
                            "Verify factual claims against retrieved knowledge base passages",
                            "Correct or remove ungrounded statements before releasing response",
                        ],
                    ),
                )

        if guardrail_result is None or not guardrail_result.is_safe:
            reason_desc = (
                "Safety guardrails not evaluated or unavailable. Under fail-closed policy, "
                "unverified tickets cannot be auto-responded (FR-04, FR-05). Ticket held "
                "safely as review draft."
            )
            return RoutingDecision(
                ticket_id=ticket.ticket_id,
                route=RouteOutcome.DRAFT_FOR_REVIEW,
                primary_requirement_id=RoutingRequirementID.FR_04.value,
                requirement_ids=[
                    RoutingRequirementID.FR_04.value,
                    RoutingRequirementID.FR_05.value,
                    RoutingRequirementID.FR_06.value,
                ],
                reason_code=RoutingReasonCode.GUARDRAIL_UNAVAILABLE,
                reason_description=reason_desc,
                billing_risk_subtype=billing_risk,
                intent_confidence_applied=intent_conf,
                urgency_confidence_applied=urgency_conf,
                confidence_applied=intent_conf,
                confidence_threshold_applied=auto_thresh,
                retrieval_similarity_threshold_applied=sim_thresh,
                eligible_for_auto_respond=False,
                citations_verified=False,
                guardrails_passed=False,
                kill_switch_active=kill_switch_active,
                documentation_available=has_docs,
                sources_used=sources_used,
                review_payload=build_review_draft(reason_desc),
            )

        # =========================================================================
        # 6. Mandatory Citation Verification Gate (FR-05, A6)
        # =========================================================================
        # AUTO_RESPOND strictly requires at least one citation resolving to a
        # retrieved knowledge-base passage. Missing, empty, or invalid citations
        # block automatic release.
        citation_check = verify_citations(
            passages=passages,
            cited_ids=cited_ids,
            require_at_least_one=True,
        )
        if not citation_check.is_valid:
            reason_desc = (
                f"Citation verification failed: {citation_check.details}. "
                "Automatic release blocked due to missing or invalid citations (FR-05, A6)."
            )
            return RoutingDecision(
                ticket_id=ticket.ticket_id,
                route=RouteOutcome.DRAFT_FOR_REVIEW,
                primary_requirement_id=RoutingRequirementID.FR_05.value,
                requirement_ids=[
                    RoutingRequirementID.FR_04.value,
                    RoutingRequirementID.FR_05.value,
                    RoutingRequirementID.FR_06.value,
                ],
                reason_code=RoutingReasonCode.CITATION_VALIDATION_FAILED,
                reason_description=reason_desc,
                billing_risk_subtype=billing_risk,
                intent_confidence_applied=intent_conf,
                urgency_confidence_applied=urgency_conf,
                confidence_applied=intent_conf,
                confidence_threshold_applied=auto_thresh,
                retrieval_similarity_threshold_applied=sim_thresh,
                eligible_for_auto_respond=False,
                citations_verified=False,
                guardrails_passed=True,
                kill_switch_active=kill_switch_active,
                documentation_available=has_docs,
                sources_used=sources_used,
                review_payload=build_review_draft(reason_desc),
            )

        # =========================================================================
        # 7. Operational Kill Switch Gate (FR-12)
        # =========================================================================
        if kill_switch_active:
            reason_desc = (
                "Operational kill switch active: automatic responses temporarily disabled. "
                "Ticket held safely as review draft for human agent approval (FR-12)."
            )
            return RoutingDecision(
                ticket_id=ticket.ticket_id,
                route=RouteOutcome.DRAFT_FOR_REVIEW,
                primary_requirement_id=RoutingRequirementID.FR_12.value,
                requirement_ids=[
                    RoutingRequirementID.FR_04.value,
                    RoutingRequirementID.FR_06.value,
                    RoutingRequirementID.FR_12.value,
                ],
                reason_code=RoutingReasonCode.KILL_SWITCH_ACTIVE,
                reason_description=reason_desc,
                billing_risk_subtype=billing_risk,
                intent_confidence_applied=intent_conf,
                urgency_confidence_applied=urgency_conf,
                confidence_applied=intent_conf,
                confidence_threshold_applied=auto_thresh,
                retrieval_similarity_threshold_applied=sim_thresh,
                eligible_for_auto_respond=False,
                citations_verified=True,
                guardrails_passed=True,
                kill_switch_active=True,
                documentation_available=has_docs,
                sources_used=sources_used,
                review_payload=build_review_draft(reason_desc),
            )

        # =========================================================================
        # 8. Intent Confidence Threshold Gates (FR-04, FR-05, FR-06)
        # =========================================================================
        # Intent confidence strictly governs the intent threshold evaluation.
        if intent_conf < review_thresh:
            reason_desc = (
                f"Intent confidence score ({intent_conf:.2f}) is below review threshold "
                f"({review_thresh:.2f}). Mandatory escalation to human support (FR-07)."
            )
            return RoutingDecision(
                ticket_id=ticket.ticket_id,
                route=RouteOutcome.ESCALATE,
                primary_requirement_id=RoutingRequirementID.FR_07.value,
                requirement_ids=[
                    RoutingRequirementID.FR_04.value,
                    RoutingRequirementID.FR_07.value,
                ],
                reason_code=RoutingReasonCode.LOW_CONFIDENCE_ESCALATE,
                reason_description=reason_desc,
                billing_risk_subtype=billing_risk,
                intent_confidence_applied=intent_conf,
                urgency_confidence_applied=urgency_conf,
                confidence_applied=intent_conf,
                confidence_threshold_applied=review_thresh,
                retrieval_similarity_threshold_applied=sim_thresh,
                eligible_for_auto_respond=False,
                citations_verified=True,
                guardrails_passed=True,
                kill_switch_active=False,
                documentation_available=has_docs,
                sources_used=sources_used,
                escalation_payload=build_escalation(reason_desc),
            )

        if intent_conf < auto_thresh:
            reason_desc = (
                f"Intent confidence score ({intent_conf:.2f}) meets review threshold "
                f"({review_thresh:.2f}) but is below auto-respond threshold ({auto_thresh:.2f}). "
                "Prepared unsent draft for human review (FR-06)."
            )
            return RoutingDecision(
                ticket_id=ticket.ticket_id,
                route=RouteOutcome.DRAFT_FOR_REVIEW,
                primary_requirement_id=RoutingRequirementID.FR_06.value,
                requirement_ids=[
                    RoutingRequirementID.FR_04.value,
                    RoutingRequirementID.FR_06.value,
                ],
                reason_code=RoutingReasonCode.LOW_CONFIDENCE_REVIEW,
                reason_description=reason_desc,
                billing_risk_subtype=billing_risk,
                intent_confidence_applied=intent_conf,
                urgency_confidence_applied=urgency_conf,
                confidence_applied=intent_conf,
                confidence_threshold_applied=auto_thresh,
                retrieval_similarity_threshold_applied=sim_thresh,
                eligible_for_auto_respond=False,
                citations_verified=True,
                guardrails_passed=True,
                kill_switch_active=False,
                documentation_available=has_docs,
                sources_used=sources_used,
                review_payload=build_review_draft(reason_desc),
            )

        # =========================================================================
        # 9. Eligible for Automatic Customer Response (AUTO_RESPOND) (FR-05)
        # =========================================================================
        reason_desc = (
            f"All gating criteria satisfied: intent confidence ({intent_conf:.2f} >= "
            f"{auto_thresh:.2f}), approved KB evidence ({len(passages)} passages >= "
            f"{sim_thresh:.2f}), verified citations ({len(citation_check.valid_citations)}), "
            "guardrails passed, kill switch enabled, and low-risk policy verified (FR-05)."
        )
        return RoutingDecision(
            ticket_id=ticket.ticket_id,
            route=RouteOutcome.AUTO_RESPOND,
            primary_requirement_id=RoutingRequirementID.FR_05.value,
            requirement_ids=[
                RoutingRequirementID.FR_04.value,
                RoutingRequirementID.FR_05.value,
            ],
            reason_code=RoutingReasonCode.ELIGIBLE_AUTO_RESPOND,
            reason_description=reason_desc,
            billing_risk_subtype=billing_risk,
            intent_confidence_applied=intent_conf,
            urgency_confidence_applied=urgency_conf,
            confidence_applied=intent_conf,
            confidence_threshold_applied=auto_thresh,
            retrieval_similarity_threshold_applied=sim_thresh,
            eligible_for_auto_respond=True,
            citations_verified=True,
            guardrails_passed=True,
            kill_switch_active=False,
            documentation_available=has_docs,
            sources_used=sources_used,
            escalation_payload=None,
            review_payload=None,
        )
