"""Regression tests for Gate 1 Guardrail Violation vs Gate 5 Grounding Failure policy parity.

Workstream 1 of Milestone A3.36.2:
Verifies that unauthorized_action_or_status violations properly map to RouteOutcome.ESCALATE
(Gate 1) in policy evaluation, rather than RouteOutcome.DRAFT_FOR_REVIEW (Gate 5).
"""

from datetime import datetime, timezone

from cloudserve.classification.models import ClassificationResult, IntentCategory, UrgencyLevel
from cloudserve.evaluation.policy_eval import evaluate_policy_and_safety
from cloudserve.guardrails.safety import GuardrailCheckResult
from cloudserve.ingestion.models import ChannelEnum, CustomerTier, UnifiedTicket
from cloudserve.routing.billing import assess_billing_risk
from cloudserve.routing.models import RouteOutcome, RoutingReasonCode
from cloudserve.routing.router import DeterministicRouter
from cloudserve.service import ProcessedTicketResult


def _create_mock_processed_result(
    ticket_id: str,
    guardrail_safe: bool,
    violation_type: str | None,
    intent: IntentCategory = IntentCategory.CONFIGURATION_HELP,
    delivery_status: str = "held_for_review",
) -> ProcessedTicketResult:
    ticket = UnifiedTicket(
        ticket_id=ticket_id,
        customer_tier=CustomerTier.STANDARD,
        channel=ChannelEnum.EMAIL,
        subject=f"Test ticket {ticket_id}",
        body="Sample ticket body for policy parity test.",
        received_at=datetime.now(timezone.utc),
    )
    classification = ClassificationResult(
        ticket_id=ticket_id,
        intent=intent,
        urgency=UrgencyLevel.LOW,
        confidence=0.95,
        intent_confidence=0.95,
        urgency_confidence=0.90,
        rationale="Test classification rationale.",
        model_name="mock-model",
        prompt_version="PR-CLS-v2",
    )
    guardrail_result = GuardrailCheckResult(
        is_safe=guardrail_safe,
        violation_type=violation_type,
        details=f"Violation: {violation_type}" if violation_type else "No violation",
    )
    router = DeterministicRouter()
    routing_decision = router.route_ticket(
        ticket=ticket,
        classification=classification,
        guardrail_result=guardrail_result,
    )
    billing_risk = assess_billing_risk(ticket)
    return ProcessedTicketResult(
        ticket=ticket,
        classification=classification,
        billing_risk=billing_risk,
        retrieved_passages=[],
        generated_response=None,
        routing_decision=routing_decision,
        delivery_status=delivery_status,
        audit_id=f"audit-{ticket_id}",
        persistence_status="persisted",
    )


def test_gate1_guardrail_violation_escalation_parity() -> None:
    """Verify Gate 1 unauthorized_action_or_status maps to ESCALATE in policy evaluator."""
    res = _create_mock_processed_result(
        ticket_id="DEV-0047",
        guardrail_safe=False,
        violation_type="unauthorized_action_or_status",
    )
    assert res.routing_decision.route == RouteOutcome.ESCALATE
    assert res.routing_decision.reason_code == RoutingReasonCode.GUARDRAIL_VIOLATION

    report = evaluate_policy_and_safety([res], ground_truth={})
    assert report.deterministic_execution_compliance_rate == 1.0
    audit = report.ticket_audits[0]
    assert audit.actual_route == "escalate"
    assert audit.router_execution_compliant is True
    assert audit.reason_code == RoutingReasonCode.GUARDRAIL_VIOLATION.value


def test_gate5_grounding_validation_failed_parity() -> None:
    """Verify Gate 5 GROUNDING_VALIDATION_FAILED maps to DRAFT_FOR_REVIEW."""
    ticket = UnifiedTicket(
        ticket_id="DEV-TEST-GATE5",
        customer_tier=CustomerTier.STANDARD,
        channel=ChannelEnum.EMAIL,
        subject="Test ticket DEV-TEST-GATE5",
        body="Sample ticket body for policy parity test.",
        received_at=datetime.now(timezone.utc),
    )
    classification = ClassificationResult(
        ticket_id="DEV-TEST-GATE5",
        intent=IntentCategory.CONFIGURATION_HELP,
        urgency=UrgencyLevel.LOW,
        confidence=0.95,
        intent_confidence=0.95,
        urgency_confidence=0.90,
        rationale="Test classification rationale.",
        model_name="mock-model",
        prompt_version="PR-CLS-v2",
    )
    router = DeterministicRouter()
    guardrail_res = GuardrailCheckResult(is_safe=True, violation_type=None, details="No violation")
    routing_decision = router.route_ticket(
        ticket=ticket,
        classification=classification,
        retrieved_passages=[],
        guardrail_result=guardrail_res,
    )
    res = ProcessedTicketResult(
        ticket=ticket,
        classification=classification,
        billing_risk=assess_billing_risk(ticket),
        retrieved_passages=[],
        generated_response=None,
        routing_decision=routing_decision,
        delivery_status="held_for_review",
        audit_id="audit-DEV-TEST-GATE5",
        persistence_status="persisted",
    )

    report = evaluate_policy_and_safety([res], ground_truth={})
    assert report.deterministic_execution_compliance_rate == 1.0
    audit = report.ticket_audits[0]
    assert audit.actual_route == "draft_for_review"
    assert audit.router_execution_compliant is True
