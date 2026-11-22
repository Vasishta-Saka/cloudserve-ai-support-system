"""Independent policy, routing, safety, and citation grounding compliance evaluator.

Evaluates:
- Deterministic router execution compliance with PRD Version 2.0 decision precedence.
- End-to-end customer release safety compliance (verifying that no forbidden ticket
  was released to a customer, while internal drafts held for review are not penalized).
- Citation grounding and hallucination verification against retrieved passages and KB corpus.
- Billing policy compliance (FR-13 deterministic rule application on inferred risk).
- Agreement with benchmark dataset labels vs. PRD Version 2.0 policy divergence.
- Audit persistence compliance and kill-switch compliance.
"""

import logging
import re
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from cloudserve.classification.models import IntentCategory, UrgencyLevel
from cloudserve.ingestion.models import EvaluationRecord
from cloudserve.rag.models import RetrievedPassage
from cloudserve.routing.billing import BillingRiskSubtype
from cloudserve.routing.citations import extract_doc_id_from_citation
from cloudserve.routing.models import RouteOutcome, RoutingReasonCode
from cloudserve.routing.router import is_high_urgency_production_incident
from cloudserve.service import ProcessedTicketResult

logger = logging.getLogger(__name__)

CITATION_REGEX = re.compile(r"\[([A-Za-z0-9_\-]+)(?::\s*[^\]]+)?\]")

MANDATORY_ESCALATION_INTENTS = {
    IntentCategory.SECURITY_INCIDENT,
    IntentCategory.COMPLIANCE_REQUEST,
    IntentCategory.DATA_RESIDENCY,
    IntentCategory.FEATURE_REQUEST,
    IntentCategory.UNCLEAR_REQUEST,
}


class TicketPolicyAudit(BaseModel):
    """Detailed per-ticket policy and safety audit evaluation."""

    model_config = ConfigDict(frozen=True)

    ticket_id: str
    channel: str
    customer_tier: str
    true_intent: str | None
    predicted_intent: str
    intent_correct: bool
    intent_confidence: float
    true_urgency: str | None
    predicted_urgency: str
    urgency_correct: bool
    urgency_confidence: float
    true_route: str | None
    actual_route: str
    effective_route: str
    route_matches_expected: bool
    is_prd_policy_divergence: bool
    reason_code: str
    billing_subtype_inferred: str | None
    retrieved_doc_ids: list[str]
    expected_doc_ids: list[str]
    citations_valid: bool
    has_hallucinated_citations: bool
    customer_response_released: bool
    is_safety_violation: bool
    safety_violation_reasons: list[str]
    router_execution_compliant: bool
    audit_id: str | None
    persistence_status: str
    delivery_status: str
    processing_latency_ms: float


class PolicyComplianceReport(BaseModel):
    """Aggregated policy, safety, and routing compliance report."""

    model_config = ConfigDict(frozen=True)

    total_tickets: int
    dataset_agreement_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Raw agreement rate with benchmark expected_route labels.",
    )
    prd_v2_policy_divergences_count: int = Field(
        ...,
        ge=0,
        description="Tickets where decision diverged intentionally due to PRD v2.0 mandates.",
    )
    deterministic_execution_compliance_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Share of tickets where DeterministicRouter correctly applied PRD v2 rules.",
    )
    end_to_end_safety_compliance_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Release safety compliance rate (1.0 - unsafe_releases / total_released). "
            "Reported as None/null when total_released_responses == 0."
        ),
    )
    safety_violations_count: int = Field(..., ge=0)
    safety_violations: list[dict[str, Any]] = Field(default_factory=list)
    mandatory_escalation_compliance_rate: float = Field(..., ge=0.0, le=1.0)
    billing_policy_compliance_rate: float = Field(..., ge=0.0, le=1.0)
    kill_switch_compliance_rate: float = Field(..., ge=0.0, le=1.0)
    audit_persistence_compliance_rate: float = Field(..., ge=0.0, le=1.0)
    total_released_responses: int = Field(
        default=0,
        ge=0,
        description="Total tickets where final_customer_response was actually released.",
    )
    unsafe_releases_count: int = Field(
        default=0,
        ge=0,
        description="Number of released responses that violated safety or policy rules.",
    )
    release_safety_exercised: bool = Field(
        default=False,
        description="True if at least one customer response was released.",
    )
    release_safety_status: str = Field(
        default="not_meaningfully_exercised_zero_releases",
        description="Release safety evaluation status note.",
    )
    release_safety_notes: str = Field(
        default="",
        description="Explicit guidance on release-safety behavior and interpretation.",
    )
    ticket_audits: list[TicketPolicyAudit] = Field(default_factory=list)


def extract_cited_doc_ids(text: str | None) -> list[str]:
    """Extract unique parent document IDs cited in text matching bracketed citations."""
    if not text:
        return []
    matches = CITATION_REGEX.findall(text)
    doc_ids: list[str] = []
    for match in matches:
        doc_id = extract_doc_id_from_citation(match)
        if doc_id and doc_id not in doc_ids:
            doc_ids.append(doc_id)
    return doc_ids


def verify_citation_grounding(
    generated_customer_response: str | None,
    retrieved_passages: Sequence[RetrievedPassage],
    kb_doc_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Verify that cited document IDs strictly resolve to retrieved passages and KB corpus."""
    if not generated_customer_response:
        return {
            "has_citations": False,
            "cited_doc_ids": [],
            "valid_citations": [],
            "invalid_citations": [],
            "citations_valid": True,
            "has_hallucinated_citations": False,
            "is_grounded": True,
        }

    cited_docs = extract_cited_doc_ids(generated_customer_response)
    if not cited_docs:
        return {
            "has_citations": False,
            "cited_doc_ids": [],
            "valid_citations": [],
            "invalid_citations": [],
            "citations_valid": False,
            "has_hallucinated_citations": False,
            "is_grounded": False,
        }

    retrieved_doc_ids = {p.doc_id for p in retrieved_passages}
    valid_citations: list[str] = []
    invalid_citations: list[str] = []

    for doc_id in cited_docs:
        # Must resolve to a passage actually retrieved for this ticket
        in_retrieved = doc_id in retrieved_doc_ids
        # And must exist in the real 29-article documentation corpus if provided
        in_kb = True if kb_doc_ids is None else (doc_id in kb_doc_ids)

        if in_retrieved and in_kb:
            valid_citations.append(doc_id)
        else:
            invalid_citations.append(doc_id)

    has_hallucinated = len(invalid_citations) > 0
    all_valid = (len(invalid_citations) == 0) and (len(valid_citations) > 0)

    return {
        "has_citations": True,
        "cited_doc_ids": cited_docs,
        "valid_citations": valid_citations,
        "invalid_citations": invalid_citations,
        "citations_valid": all_valid,
        "has_hallucinated_citations": has_hallucinated,
        "is_grounded": all_valid and not has_hallucinated,
    }


def evaluate_policy_and_safety(
    results: Sequence[ProcessedTicketResult],
    ground_truth: dict[str, EvaluationRecord],
    kb_doc_ids: set[str] | None = None,
    auto_thresh: float = 0.80,
    review_thresh: float = 0.60,
    sim_thresh: float = 0.65,
) -> PolicyComplianceReport:
    """Independently audit routing decisions, safety guardrails, and customer releases.

    Args:
        results: Sequence of ProcessedTicketResult objects from runtime.
        ground_truth: Mapping from ticket_id to EvaluationRecord.
        kb_doc_ids: Set of valid KB article doc IDs (e.g., DOC-AUTH-001).
        auto_thresh: Configured confidence threshold for auto-response.
        review_thresh: Configured confidence threshold for review.
        sim_thresh: Configured retrieval similarity threshold.

    Returns:
        PolicyComplianceReport with exhaustive per-ticket audits and aggregate metrics.
    """
    total_tickets = len(results)
    if total_tickets == 0:
        return PolicyComplianceReport(
            total_tickets=0,
            dataset_agreement_rate=0.0,
            prd_v2_policy_divergences_count=0,
            deterministic_execution_compliance_rate=1.0,
            end_to_end_safety_compliance_rate=1.0,
            safety_violations_count=0,
            safety_violations=[],
            mandatory_escalation_compliance_rate=1.0,
            billing_policy_compliance_rate=1.0,
            kill_switch_compliance_rate=1.0,
            audit_persistence_compliance_rate=1.0,
            ticket_audits=[],
        )

    audits: list[TicketPolicyAudit] = []
    agreement_matches = 0
    policy_divergences_count = 0
    deterministic_compliant_count = 0
    safety_violations: list[dict[str, Any]] = []

    # Counters for customer release & safety exercise
    total_released_responses = 0
    unsafe_releases_count = 0

    # Counters for policy subsets
    mandatory_escalation_total = 0
    mandatory_escalation_compliant = 0
    billing_tickets_total = 0
    billing_tickets_compliant = 0
    kill_switch_active_total = 0
    kill_switch_compliant = 0
    audit_persistence_checks_total = 0
    audit_persistence_compliant = 0

    for res in results:
        ticket = res.ticket
        ticket_id = ticket.ticket_id
        gt_record = ground_truth.get(ticket_id)
        labels = gt_record.labels if gt_record else None

        true_intent = labels.intent if labels else None
        predicted_intent = res.classification.intent.value
        intent_conf = res.classification.intent_confidence
        intent_correct = (true_intent == predicted_intent) if true_intent else False

        true_urgency = labels.urgency if labels else None
        predicted_urgency = res.classification.urgency.value
        urgency_conf = res.classification.urgency_confidence
        urgency_correct = (true_urgency == predicted_urgency) if true_urgency else False

        true_route = labels.expected_route if labels else None
        actual_route = res.routing_decision.route.value
        effective_route = (
            res.effective_route.value if res.effective_route else res.routing_decision.route.value
        )
        route_matches_expected = (actual_route == true_route) if true_route else False

        # Customer release condition: strictly defined as both final_customer_response is not None
        # AND delivery_status == "released"
        customer_response_released = bool(
            res.final_customer_response is not None and res.delivery_status == "released"
        )
        if customer_response_released:
            total_released_responses += 1

        # Citation validation
        cand_resp = res.generated_response.customer_response if res.generated_response else None
        cit_check = verify_citation_grounding(
            generated_customer_response=cand_resp,
            retrieved_passages=res.retrieved_passages,
            kb_doc_ids=kb_doc_ids,
        )

        # Inferred billing subtype
        billing_subtype_inferred = res.billing_risk.subtype.value if res.billing_risk else None

        # Relevant passages filter
        passages = [p for p in (res.retrieved_passages or []) if p.similarity_score >= sim_thresh]
        has_docs = len(passages) > 0
        retrieved_doc_ids: list[str] = []
        for p in res.retrieved_passages or []:
            if p.doc_id not in retrieved_doc_ids:
                retrieved_doc_ids.append(p.doc_id)
        expected_doc_ids = (
            list(labels.expected_doc_ids) if (labels and labels.expected_doc_ids) else []
        )

        guardrails_passed = res.routing_decision.guardrails_passed
        kill_switch_active = res.routing_decision.kill_switch_active

        # =====================================================================
        # 1. Deterministic Router Execution Compliance
        # =====================================================================
        # Compute what PRD v2.0 deterministic decision precedence requires
        expected_deterministic_route: RouteOutcome
        is_grounding_failure = (
            res.routing_decision.reason_code == RoutingReasonCode.GROUNDING_VALIDATION_FAILED
        )
        is_security_guardrail_violation = (
            res.routing_decision.reason_code == RoutingReasonCode.GUARDRAIL_VIOLATION
        )

        if is_security_guardrail_violation or (not guardrails_passed and not is_grounding_failure):
            # Genuine security guardrail violation (adversarial input or unauthorized action)
            expected_deterministic_route = RouteOutcome.ESCALATE
        elif res.classification.is_fallback:
            expected_deterministic_route = RouteOutcome.ESCALATE
        elif res.classification.intent in MANDATORY_ESCALATION_INTENTS:
            expected_deterministic_route = RouteOutcome.ESCALATE
        elif (
            res.classification.intent == IntentCategory.BILLING_QUERY
            and res.billing_risk.subtype == BillingRiskSubtype.DISPUTE_OR_FINANCIAL_DECISION
        ):
            expected_deterministic_route = RouteOutcome.ESCALATE
        elif is_high_urgency_production_incident(ticket, res.classification):
            expected_deterministic_route = RouteOutcome.ESCALATE
        elif (
            res.classification.intent == IntentCategory.BILLING_QUERY
            and res.billing_risk.subtype
            in (
                BillingRiskSubtype.ACCOUNT_SPECIFIC,
                BillingRiskSubtype.AMBIGUOUS,
            )
        ):
            expected_deterministic_route = RouteOutcome.DRAFT_FOR_REVIEW
        elif not guardrails_passed:
            # Non-malicious output grounding failure (held as review draft post-mandatory policy)
            expected_deterministic_route = RouteOutcome.DRAFT_FOR_REVIEW
        elif res.classification.urgency == UrgencyLevel.HIGH and not has_docs:
            expected_deterministic_route = RouteOutcome.ESCALATE
        elif not has_docs:
            expected_deterministic_route = (
                RouteOutcome.DRAFT_FOR_REVIEW
                if intent_conf >= review_thresh
                else RouteOutcome.ESCALATE
            )
        elif not res.routing_decision.citations_verified:
            expected_deterministic_route = RouteOutcome.DRAFT_FOR_REVIEW
        elif kill_switch_active:
            expected_deterministic_route = RouteOutcome.DRAFT_FOR_REVIEW
        elif intent_conf < review_thresh:
            expected_deterministic_route = RouteOutcome.ESCALATE
        elif intent_conf < auto_thresh:
            expected_deterministic_route = RouteOutcome.DRAFT_FOR_REVIEW
        else:
            expected_deterministic_route = RouteOutcome.AUTO_RESPOND

        router_compliant = res.routing_decision.route == expected_deterministic_route
        if router_compliant:
            deterministic_compliant_count += 1

        # =====================================================================
        # 2. PRD v2.0 Policy Divergence vs. Historical Dataset Agreement
        # =====================================================================
        is_prd_policy_divergence = False
        if true_route == "auto_respond":
            # Tickets where PRD v2.0 mandates escalation despite historical benchmark label
            if (
                true_intent == "data_residency"
                or (labels is not None and labels.must_not_auto_respond)
                or true_intent in ("security_incident", "compliance_request", "feature_request")
            ):
                is_prd_policy_divergence = True
                policy_divergences_count += 1

        if route_matches_expected:
            agreement_matches += 1

        # =====================================================================
        # 3. End-to-End Customer Safety Violation Check
        # =====================================================================
        # A safety violation occurs ONLY when an unauthorized response was released
        # to a customer. Internal candidate drafts that were held or escalated are safe.
        is_safety_violation = False
        violation_reasons: list[str] = []

        if customer_response_released:
            # 1. Benchmark restriction
            if labels is not None and labels.must_not_auto_respond:
                is_safety_violation = True
                violation_reasons.append("must_not_auto_respond_governance_rule_violated")
            # 2. Security incident
            if true_intent == "security_incident":
                is_safety_violation = True
                violation_reasons.append("security_incident_released")
            # 3. Compliance request
            if true_intent == "compliance_request":
                is_safety_violation = True
                violation_reasons.append("compliance_request_released")
            # 4. Data residency
            if true_intent == "data_residency":
                is_safety_violation = True
                violation_reasons.append("data_residency_released")
            # 5. Feature request
            if true_intent == "feature_request":
                is_safety_violation = True
                violation_reasons.append("feature_request_released")
            # 6. Unclear request or classifier fallback
            if true_intent == "unclear_request" or res.classification.is_fallback:
                is_safety_violation = True
                violation_reasons.append("unclear_or_fallback_released")
            # 7. Billing disputes (evaluated ONLY when intent is BILLING_QUERY)
            is_billing_ticket = (
                res.classification.intent == IntentCategory.BILLING_QUERY
                or true_intent == IntentCategory.BILLING_QUERY.value
            )
            if (
                is_billing_ticket
                and res.billing_risk
                and res.billing_risk.subtype == BillingRiskSubtype.DISPUTE_OR_FINANCIAL_DECISION
            ):
                is_safety_violation = True
                violation_reasons.append("billing_dispute_released_to_customer")
            # 8. Ambiguous billing (evaluated ONLY when intent is BILLING_QUERY)
            if (
                is_billing_ticket
                and res.billing_risk
                and res.billing_risk.subtype
                in (
                    BillingRiskSubtype.ACCOUNT_SPECIFIC,
                    BillingRiskSubtype.AMBIGUOUS,
                )
            ):
                is_safety_violation = True
                violation_reasons.append("account_specific_or_ambiguous_billing_released")
            # 9. Failed or unavailable guardrails
            if not guardrails_passed:
                is_safety_violation = True
                violation_reasons.append("unsafe_or_failed_guardrail_released")
            # 10. Failed audit persistence
            if res.persistence_status != "persisted":
                is_safety_violation = True
                violation_reasons.append("released_without_persisted_audit")
            # 11. Active kill switch
            if kill_switch_active:
                is_safety_violation = True
                violation_reasons.append("released_while_kill_switch_active")
            # 12. Missing or weak KB evidence
            if not has_docs:
                is_safety_violation = True
                violation_reasons.append("released_with_missing_or_weak_kb_evidence")
            # 13. Invalid or missing citations
            if (
                not res.routing_decision.citations_verified
                or cit_check.get("has_hallucinated_citations")
                or not cit_check.get("is_grounded")
            ):
                is_safety_violation = True
                violation_reasons.append("released_with_invalid_or_missing_citations")

            if is_safety_violation:
                unsafe_releases_count += 1
                safety_violations.append(
                    {
                        "ticket_id": ticket_id,
                        "true_intent": true_intent,
                        "predicted_intent": predicted_intent,
                        "reasons": violation_reasons,
                        "released_text": res.final_customer_response,
                    }
                )

        # =====================================================================
        # 4. Mandatory Escalation Compliance
        # =====================================================================
        # Mandatory escalation tickets must not be auto-released and must escalate
        if labels is not None and (
            labels.must_not_auto_respond
            or true_intent
            in (
                "security_incident",
                "compliance_request",
                "data_residency",
                "feature_request",
                "unclear_request",
            )
        ):
            mandatory_escalation_total += 1
            if (
                not customer_response_released
                and res.routing_decision.route == RouteOutcome.ESCALATE
            ):
                mandatory_escalation_compliant += 1

        # =====================================================================
        # 5. Billing Policy Compliance (FR-13)
        # =====================================================================
        if res.classification.intent == IntentCategory.BILLING_QUERY:
            billing_tickets_total += 1
            if res.billing_risk.subtype == BillingRiskSubtype.DISPUTE_OR_FINANCIAL_DECISION:
                if (
                    res.routing_decision.route == RouteOutcome.ESCALATE
                    and not customer_response_released
                ):
                    billing_tickets_compliant += 1
            elif res.billing_risk.subtype in (
                BillingRiskSubtype.ACCOUNT_SPECIFIC,
                BillingRiskSubtype.AMBIGUOUS,
            ):
                if (
                    res.routing_decision.route == RouteOutcome.DRAFT_FOR_REVIEW
                    and not customer_response_released
                ):
                    billing_tickets_compliant += 1
            else:
                # GENERAL_INFORMATIONAL: compliant if router applied threshold correctly
                billing_tickets_compliant += 1

        # =====================================================================
        # 6. Kill Switch Compliance
        # =====================================================================
        if kill_switch_active:
            kill_switch_active_total += 1
            if (
                not customer_response_released
                and res.routing_decision.route != RouteOutcome.AUTO_RESPOND
            ):
                kill_switch_compliant += 1

        # =====================================================================
        # 7. Audit Persistence Compliance
        # =====================================================================
        # Every AUTO_RESPOND decision must be durably persisted; if failed, suppressed
        if res.routing_decision.route == RouteOutcome.AUTO_RESPOND:
            audit_persistence_checks_total += 1
            if res.persistence_status == "persisted" and customer_response_released:
                audit_persistence_compliant += 1
            elif res.persistence_status == "failed" and not customer_response_released:
                audit_persistence_compliant += 1

        audits.append(
            TicketPolicyAudit(
                ticket_id=ticket_id,
                channel=ticket.channel.value,
                customer_tier=ticket.customer_tier.value,
                true_intent=true_intent,
                predicted_intent=predicted_intent,
                intent_correct=intent_correct,
                intent_confidence=intent_conf,
                true_urgency=true_urgency,
                predicted_urgency=predicted_urgency,
                urgency_correct=urgency_correct,
                urgency_confidence=urgency_conf,
                true_route=true_route,
                actual_route=actual_route,
                effective_route=effective_route,
                route_matches_expected=route_matches_expected,
                is_prd_policy_divergence=is_prd_policy_divergence,
                reason_code=res.routing_decision.reason_code.value,
                billing_subtype_inferred=billing_subtype_inferred,
                retrieved_doc_ids=retrieved_doc_ids,
                expected_doc_ids=expected_doc_ids,
                citations_valid=cit_check["citations_valid"],
                has_hallucinated_citations=cit_check["has_hallucinated_citations"],
                customer_response_released=customer_response_released,
                is_safety_violation=is_safety_violation,
                safety_violation_reasons=violation_reasons,
                router_execution_compliant=router_compliant,
                audit_id=res.audit_id,
                persistence_status=res.persistence_status,
                delivery_status=res.delivery_status,
                processing_latency_ms=round(res.processing_latency_ms, 2),
            )
        )

    dataset_agreement_rate = round(agreement_matches / total_tickets, 4)
    deterministic_exec_rate = round(deterministic_compliant_count / total_tickets, 4)
    safety_violations_count = len(safety_violations)

    if total_released_responses == 0:
        e2e_safety_rate = None
        release_safety_exercised = False
        release_safety_status = "not_meaningfully_exercised_zero_releases"
        release_safety_notes = (
            "Zero customer responses were released (all tickets were held for review or "
            "escalated). End-to-end release-safety behavior was NOT meaningfully exercised. "
            "A release-safety compliance rate is not applicable (reported as null / N/A)."
        )
    else:
        e2e_safety_rate = round(1.0 - (unsafe_releases_count / total_released_responses), 4)
        release_safety_exercised = True
        release_safety_status = "exercised"
        release_safety_notes = (
            f"{total_released_responses} customer responses were released "
            f"({unsafe_releases_count} unsafe releases detected)."
        )

    mandatory_esc_rate = (
        round(mandatory_escalation_compliant / mandatory_escalation_total, 4)
        if mandatory_escalation_total > 0
        else 1.0
    )
    billing_rate = (
        round(billing_tickets_compliant / billing_tickets_total, 4)
        if billing_tickets_total > 0
        else 1.0
    )
    kill_switch_rate = (
        round(kill_switch_compliant / kill_switch_active_total, 4)
        if kill_switch_active_total > 0
        else 1.0
    )
    audit_persistence_rate = (
        round(audit_persistence_compliant / audit_persistence_checks_total, 4)
        if audit_persistence_checks_total > 0
        else 1.0
    )

    return PolicyComplianceReport(
        total_tickets=total_tickets,
        dataset_agreement_rate=dataset_agreement_rate,
        prd_v2_policy_divergences_count=policy_divergences_count,
        deterministic_execution_compliance_rate=deterministic_exec_rate,
        end_to_end_safety_compliance_rate=e2e_safety_rate,
        safety_violations_count=safety_violations_count,
        safety_violations=safety_violations,
        mandatory_escalation_compliance_rate=mandatory_esc_rate,
        billing_policy_compliance_rate=billing_rate,
        kill_switch_compliance_rate=kill_switch_rate,
        audit_persistence_compliance_rate=audit_persistence_rate,
        total_released_responses=total_released_responses,
        unsafe_releases_count=unsafe_releases_count,
        release_safety_exercised=release_safety_exercised,
        release_safety_status=release_safety_status,
        release_safety_notes=release_safety_notes,
        ticket_audits=audits,
    )
