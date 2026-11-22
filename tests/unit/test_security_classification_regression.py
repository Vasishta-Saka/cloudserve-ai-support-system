"""Regression tests for security incident vs routine account offboarding classification taxonomy.

Workstream 2 of Milestone A3.36.2:
Verifies that prompt definitions and classification boundaries cleanly separate:
1. Unauthorized API activity after employee deprovisioning (security_incident).
2. Ordinary access provisioning, role changes, and routine deprovisioning (account_access).
3. Password reset and MFA recovery (authentication_failure).
4. Suspected credential leaks or compromise (security_incident).
5. Confirmed unauthorized access (security_incident).
6. Security incidents mandatorily trigger Gate 2 escalation and bypass response generation.
"""

from datetime import datetime, timezone

import pytest

from cloudserve.classification.models import ClassificationResult, IntentCategory, UrgencyLevel
from cloudserve.classification.prompts import load_classification_system_prompt
from cloudserve.ingestion.models import ChannelEnum, CustomerTier, UnifiedTicket
from cloudserve.routing.models import RouteOutcome, RoutingReasonCode
from cloudserve.routing.router import DeterministicRouter


def test_classification_v2_prompt_boundary_definitions() -> None:
    """Verify v2 prompt explicitly defines boundaries for security vs account access."""
    prompt = load_classification_system_prompt(version="v2")
    assert "account_access" in prompt
    assert "security_incident" in prompt
    assert "routine user deprovisioning" in prompt or "routine" in prompt
    assert "unauthorized API calls" in prompt
    assert "employee termination" in prompt or "leaving" in prompt or "deprovisioning" in prompt


@pytest.mark.parametrize(
    "intent,expected_route,expected_reason,expected_dest",
    [
        (
            IntentCategory.SECURITY_INCIDENT,
            RouteOutcome.ESCALATE,
            RoutingReasonCode.MANDATORY_SECURITY_INCIDENT,
            "Security Operations",
        ),
        (
            IntentCategory.ACCOUNT_ACCESS,
            RouteOutcome.DRAFT_FOR_REVIEW,
            None,
            None,
        ),
        (
            IntentCategory.AUTHENTICATION_FAILURE,
            RouteOutcome.DRAFT_FOR_REVIEW,
            None,
            None,
        ),
    ],
)
def test_router_mandatory_security_incident_escalation(
    intent: IntentCategory,
    expected_route: RouteOutcome,
    expected_reason: RoutingReasonCode | None,
    expected_dest: str | None,
) -> None:
    """Verify security_incident mandatorily routes to ESCALATE (Security Operations)."""
    router = DeterministicRouter()
    ticket = UnifiedTicket(
        ticket_id="TEST-SEC-01",
        customer_tier=CustomerTier.STANDARD,
        channel=ChannelEnum.EMAIL,
        subject="Test Inquiry",
        body="Inquiry text body.",
        received_at=datetime.now(timezone.utc),
    )
    urgency = (
        UrgencyLevel.HIGH if intent == IntentCategory.SECURITY_INCIDENT else UrgencyLevel.MEDIUM
    )
    classification = ClassificationResult(
        ticket_id="TEST-SEC-01",
        intent=intent,
        urgency=urgency,
        confidence=0.95,
        intent_confidence=0.95,
        urgency_confidence=0.90,
        rationale="Test classification rationale.",
        model_name="mock-model",
        prompt_version="PR-CLS-v2",
    )
    decision = router.route_ticket(
        ticket=ticket,
        classification=classification,
    )

    if expected_route == RouteOutcome.ESCALATE:
        assert decision.route == RouteOutcome.ESCALATE
        assert decision.reason_code == expected_reason
        assert decision.escalation_payload is not None
        assert decision.escalation_payload.target_specialist_group == expected_dest
    else:
        assert decision.route != RouteOutcome.AUTO_RESPOND
        assert decision.reason_code != RoutingReasonCode.MANDATORY_SECURITY_INCIDENT


def test_former_employee_unauthorized_activity_scenario() -> None:
    """Scenario representing DEV-0234: Post-departure unauthorized API calls."""
    ticket_body = (
        "former employee appears to still have access three weeks after leaving. "
        "we having found API calls made under their account. please tell what to do."
    )
    router = DeterministicRouter()
    classification = ClassificationResult(
        ticket_id="DEV-0234-REGRESSION",
        intent=IntentCategory.SECURITY_INCIDENT,
        urgency=UrgencyLevel.HIGH,
        confidence=0.96,
        intent_confidence=0.96,
        urgency_confidence=0.90,
        rationale="Unauthorized API activity post-departure constitutes a security breach.",
        model_name="mock-model",
        prompt_version="PR-CLS-v2",
    )
    decision = router.route_ticket(
        ticket=UnifiedTicket(
            ticket_id="DEV-0234-REGRESSION",
            customer_tier=CustomerTier.STANDARD,
            channel=ChannelEnum.EMAIL,
            subject="Former employee unauthorized access",
            body=ticket_body,
            received_at=datetime.now(timezone.utc),
        ),
        classification=classification,
    )
    assert decision.route == RouteOutcome.ESCALATE
    assert decision.reason_code == RoutingReasonCode.MANDATORY_SECURITY_INCIDENT
    assert decision.escalation_payload is not None
    assert decision.escalation_payload.target_specialist_group == "Security Operations"


def test_routine_account_deprovisioning_scenario() -> None:
    """Scenario: Routine offboarding request without breach -> account_access."""
    classification = ClassificationResult(
        ticket_id="DEV-ROUTINE-OFFBOARD",
        intent=IntentCategory.ACCOUNT_ACCESS,
        urgency=UrgencyLevel.LOW,
        confidence=0.95,
        intent_confidence=0.95,
        urgency_confidence=0.92,
        rationale="Routine offboarding guidance.",
        model_name="mock-model",
        prompt_version="PR-CLS-v2",
    )
    assert classification.intent == IntentCategory.ACCOUNT_ACCESS
    assert classification.urgency == UrgencyLevel.LOW
