"""Unit tests for the Deterministic Routing Engine (Task 4, Acceptance Criterion A5).

Verifies PRD v2.0 routing requirements:
- Deterministic decision-making and idempotency (FR-04)
- Auto-response eligibility and citation verification (FR-05, A6)
- Unsent human-review draft generation (FR-06)
- Mandatory escalation categories (FR-07)
- Operational kill switch (FR-12)
- Three-level billing risk classification (FR-13)
- Handoff payload completeness and audit log compatibility (A8)
- Fail-closed guardrails, mandatory citations, and intent confidence gating
"""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from cloudserve.classification.models import (
    ClassificationResult,
    IntentCategory,
    UrgencyLevel,
)
from cloudserve.config import Settings
from cloudserve.ingestion.models import ChannelEnum, CustomerTier, UnifiedTicket
from cloudserve.rag.models import RetrievedPassage
from cloudserve.routing.billing import assess_billing_risk
from cloudserve.routing.citations import verify_citations
from cloudserve.routing.models import (
    BillingRiskSubtype,
    CitationValidationResult,
    GuardrailCheckResult,
    RouteOutcome,
    RoutingReasonCode,
)
from cloudserve.routing.router import DeterministicRouter


@pytest.fixture
def base_ticket() -> UnifiedTicket:
    """Create a sample ingested UnifiedTicket."""
    return UnifiedTicket(
        ticket_id="DEV-TEST-0001",
        channel=ChannelEnum.EMAIL,
        subject="How do I update my service targetPort?",
        body="Health check fails with connection refused on port 8080. Please advise.",
        received_at=datetime(2026, 11, 18, 10, 0, 0, tzinfo=timezone.utc),
        customer_id="CUST-1001",
        customer_name="Acme Corp",
        customer_tier=CustomerTier.BUSINESS,
    )


@pytest.fixture
def base_classification() -> ClassificationResult:
    """Create a sample high-confidence classification result."""
    return ClassificationResult(
        ticket_id="DEV-TEST-0001",
        intent=IntentCategory.DEPLOYMENT_FAILURE,
        urgency=UrgencyLevel.MEDIUM,
        confidence=0.92,
        intent_confidence=0.94,
        urgency_confidence=0.90,
        alternatives=[],
        rationale="Clear deployment container port mismatch.",
        model_name="mock-classifier",
        prompt_version="v1.0",
        is_fallback=False,
    )


@pytest.fixture
def sample_passages() -> list[RetrievedPassage]:
    """Create sample relevant retrieved knowledge-base passages."""
    return [
        RetrievedPassage(
            chunk_id="DOC-DEPLOY-001#resolution",
            doc_id="DOC-DEPLOY-001",
            category="deployment",
            title="Troubleshooting Container Port Mismatch",
            section_name="Resolution",
            content="Update targetPort in service.yaml to match container EXPOSE port 8080.",
            similarity_score=0.88,
            rank=1,
            citation_id="[DOC-DEPLOY-001: Resolution]",
            applies_to="Container Service, CLI",
        ),
        RetrievedPassage(
            chunk_id="DOC-DEPLOY-001#notes",
            doc_id="DOC-DEPLOY-001",
            category="deployment",
            title="Troubleshooting Container Port Mismatch",
            section_name="Notes",
            content="Port 8080 is the standard port for Spring Boot and Node.js applications.",
            similarity_score=0.82,
            rank=2,
            citation_id="[DOC-DEPLOY-001: Notes]",
            applies_to="Container Service, CLI",
        ),
    ]


@pytest.fixture
def billing_passages() -> list[RetrievedPassage]:
    """Create sample retrieved billing documentation passages."""
    return [
        RetrievedPassage(
            chunk_id="DOC-BILL-001#resolution",
            doc_id="DOC-BILL-001",
            category="billing",
            title="Understanding your invoice and usage breakdown",
            section_name="Resolution",
            content="Billing cycles run on calendar months starting on the 1st day of each month.",
            similarity_score=0.89,
            rank=1,
            citation_id="[DOC-BILL-001: Resolution]",
            applies_to="Console, Billing Portal",
        )
    ]


@pytest.fixture
def safe_guardrail() -> GuardrailCheckResult:
    """Sample passing safety guardrail check result."""
    return GuardrailCheckResult(is_safe=True)


@pytest.fixture
def router() -> DeterministicRouter:
    """Create a DeterministicRouter with test configuration."""
    settings = Settings(
        confidence_threshold_auto_respond=0.80,
        confidence_threshold_review=0.60,
        retrieval_similarity_threshold=0.65,
        enable_auto_respond=True,
    )
    return DeterministicRouter(settings=settings)


# =============================================================================
# 1. Determinism and Idempotency Tests (FR-04)
# =============================================================================


def test_identical_input_produces_identical_decision(
    router: DeterministicRouter,
    base_ticket: UnifiedTicket,
    base_classification: ClassificationResult,
    sample_passages: list[RetrievedPassage],
    safe_guardrail: GuardrailCheckResult,
) -> None:
    """Verify that repeated identical inputs yield identical routing outcomes and reasons."""
    decision_1 = router.route_ticket(
        ticket=base_ticket,
        classification=base_classification,
        retrieved_passages=sample_passages,
        guardrail_result=safe_guardrail,
        cited_ids=["[DOC-DEPLOY-001: Resolution]"],
    )
    decision_2 = router.route_ticket(
        ticket=base_ticket,
        classification=base_classification,
        retrieved_passages=sample_passages,
        guardrail_result=safe_guardrail,
        cited_ids=["[DOC-DEPLOY-001: Resolution]"],
    )

    assert decision_1.route == decision_2.route == RouteOutcome.AUTO_RESPOND
    assert decision_1.reason_code == decision_2.reason_code
    assert decision_1.reason_code == RoutingReasonCode.ELIGIBLE_AUTO_RESPOND
    assert decision_1.primary_requirement_id == decision_2.primary_requirement_id == "FR-05"
    assert decision_1.intent_confidence_applied == decision_2.intent_confidence_applied


# =============================================================================
# 2. Mandatory Escalation Categories (FR-07)
# =============================================================================


def test_mandatory_security_incident_always_escalates(
    router: DeterministicRouter,
    base_ticket: UnifiedTicket,
    sample_passages: list[RetrievedPassage],
) -> None:
    """Security incidents must always escalate regardless of confidence or KB availability."""
    sec_classification = ClassificationResult(
        ticket_id=base_ticket.ticket_id,
        intent=IntentCategory.SECURITY_INCIDENT,
        urgency=UrgencyLevel.HIGH,
        confidence=0.98,
        intent_confidence=0.99,
        urgency_confidence=0.97,
        rationale="Suspected API key compromise in public repository.",
        model_name="mock-model",
        prompt_version="v1.0",
        is_fallback=False,
    )
    decision = router.route_ticket(
        ticket=base_ticket,
        classification=sec_classification,
        retrieved_passages=sample_passages,
    )

    assert decision.route == RouteOutcome.ESCALATE
    assert decision.reason_code == RoutingReasonCode.MANDATORY_SECURITY_INCIDENT
    assert decision.primary_requirement_id == "FR-07"
    assert decision.escalation_payload is not None
    assert decision.escalation_payload.target_specialist_group == "Security Operations"
    assert decision.escalation_payload.documentation_available is True
    assert decision.escalation_payload.documentation_excerpt is not None
    assert decision.escalation_payload.safe_guidance is None


def test_mandatory_compliance_request_always_escalates(
    router: DeterministicRouter,
    base_ticket: UnifiedTicket,
    sample_passages: list[RetrievedPassage],
) -> None:
    """Compliance requests must always escalate even when documentation answers it."""
    comp_classification = ClassificationResult(
        ticket_id=base_ticket.ticket_id,
        intent=IntentCategory.COMPLIANCE_REQUEST,
        urgency=UrgencyLevel.LOW,
        confidence=0.95,
        intent_confidence=0.96,
        urgency_confidence=0.94,
        rationale="SOC 2 Type II audit report request.",
        model_name="mock-model",
        prompt_version="v1.0",
        is_fallback=False,
    )
    decision = router.route_ticket(
        ticket=base_ticket,
        classification=comp_classification,
        retrieved_passages=sample_passages,
    )

    assert decision.route == RouteOutcome.ESCALATE
    assert decision.reason_code == RoutingReasonCode.MANDATORY_COMPLIANCE_REQUEST
    assert decision.primary_requirement_id == "FR-07"
    assert decision.escalation_payload is not None
    assert decision.escalation_payload.target_specialist_group == "Compliance & Legal"


def test_mandatory_data_residency_always_escalates(
    router: DeterministicRouter,
    base_ticket: UnifiedTicket,
    sample_passages: list[RetrievedPassage],
) -> None:
    """All data-residency/data-location requests must escalate under PRD v2.0 §10.2."""
    data_classification = ClassificationResult(
        ticket_id=base_ticket.ticket_id,
        intent=IntentCategory.DATA_RESIDENCY,
        urgency=UrgencyLevel.LOW,
        confidence=0.96,
        intent_confidence=0.97,
        urgency_confidence=0.95,
        rationale="Inquiry regarding EU data storage boundary.",
        model_name="mock-model",
        prompt_version="v1.0",
        is_fallback=False,
    )
    decision = router.route_ticket(
        ticket=base_ticket,
        classification=data_classification,
        retrieved_passages=sample_passages,
    )

    assert decision.route == RouteOutcome.ESCALATE
    assert decision.reason_code == RoutingReasonCode.MANDATORY_DATA_RESIDENCY
    assert decision.primary_requirement_id == "FR-07"
    assert decision.escalation_payload is not None
    assert decision.escalation_payload.target_specialist_group == "Compliance & Infrastructure"


def test_mandatory_feature_request_always_escalates(
    router: DeterministicRouter,
    base_ticket: UnifiedTicket,
) -> None:
    """Feature and roadmap requests must always escalate to Product Management."""
    feat_classification = ClassificationResult(
        ticket_id=base_ticket.ticket_id,
        intent=IntentCategory.FEATURE_REQUEST,
        urgency=UrgencyLevel.LOW,
        confidence=0.90,
        intent_confidence=0.92,
        urgency_confidence=0.88,
        rationale="Customer requested ARM64 architecture support.",
        model_name="mock-model",
        prompt_version="v1.0",
        is_fallback=False,
    )
    decision = router.route_ticket(
        ticket=base_ticket,
        classification=feat_classification,
    )

    assert decision.route == RouteOutcome.ESCALATE
    assert decision.reason_code == RoutingReasonCode.MANDATORY_FEATURE_REQUEST
    assert decision.primary_requirement_id == "FR-07"
    assert decision.escalation_payload is not None
    assert decision.escalation_payload.target_specialist_group == "Product Management"


def test_mandatory_unclear_request_always_escalates(
    router: DeterministicRouter,
    base_ticket: UnifiedTicket,
) -> None:
    """Unclear customer requests must always escalate to a human agent."""
    unclear_classification = ClassificationResult(
        ticket_id=base_ticket.ticket_id,
        intent=IntentCategory.UNCLEAR_REQUEST,
        urgency=UrgencyLevel.LOW,
        confidence=0.75,
        intent_confidence=0.75,
        urgency_confidence=0.75,
        rationale="Unintelligible text.",
        model_name="mock-model",
        prompt_version="v1.0",
        is_fallback=False,
    )
    decision = router.route_ticket(
        ticket=base_ticket,
        classification=unclear_classification,
    )

    assert decision.route == RouteOutcome.ESCALATE
    assert decision.reason_code == RoutingReasonCode.MANDATORY_UNCLEAR_REQUEST
    assert decision.primary_requirement_id == "FR-07"


def test_mandatory_classifier_fallback_never_auto_responds(
    router: DeterministicRouter,
    base_ticket: UnifiedTicket,
    sample_passages: list[RetrievedPassage],
) -> None:
    """Classifier fallback must strictly escalate and never send automated responses."""
    fallback_classification = ClassificationResult(
        ticket_id=base_ticket.ticket_id,
        intent=IntentCategory.UNCLEAR_REQUEST,
        urgency=UrgencyLevel.LOW,
        confidence=0.0,
        intent_confidence=0.0,
        urgency_confidence=0.0,
        rationale="LLM call failed; fallback triggered.",
        model_name="mock-fallback",
        prompt_version="v1.0",
        is_fallback=True,
    )
    decision = router.route_ticket(
        ticket=base_ticket,
        classification=fallback_classification,
        retrieved_passages=sample_passages,
    )

    assert decision.route == RouteOutcome.ESCALATE
    assert decision.reason_code == RoutingReasonCode.MANDATORY_CLASSIFIER_FALLBACK
    assert decision.primary_requirement_id == "FR-07"


# =============================================================================
# 3. Billing Risk Classification (FR-13)
# =============================================================================


def test_billing_general_informational_auto_responds_when_all_gates_pass(
    router: DeterministicRouter,
    billing_passages: list[RetrievedPassage],
    safe_guardrail: GuardrailCheckResult,
) -> None:
    """General informational billing queries can auto-respond if all gates pass."""
    ticket = UnifiedTicket(
        ticket_id="DEV-BILL-0001",
        channel=ChannelEnum.EMAIL,
        subject="Billing period dates",
        body="When does our monthly billing period start and how is calendar month aligned?",
        received_at=datetime(2026, 11, 18, 10, 0, 0, tzinfo=timezone.utc),
    )
    classification = ClassificationResult(
        ticket_id=ticket.ticket_id,
        intent=IntentCategory.BILLING_QUERY,
        urgency=UrgencyLevel.LOW,
        confidence=0.91,
        intent_confidence=0.92,
        urgency_confidence=0.90,
        rationale="General billing cycle inquiry.",
        model_name="mock-model",
        prompt_version="v1.0",
    )
    decision = router.route_ticket(
        ticket=ticket,
        classification=classification,
        retrieved_passages=billing_passages,
        guardrail_result=safe_guardrail,
        cited_ids=["[DOC-BILL-001: Resolution]"],
    )

    assert decision.route == RouteOutcome.AUTO_RESPOND
    assert decision.billing_risk_subtype == BillingRiskSubtype.GENERAL_INFORMATIONAL
    assert decision.reason_code == RoutingReasonCode.ELIGIBLE_AUTO_RESPOND
    assert decision.eligible_for_auto_respond is True


def test_billing_account_specific_creates_unsent_review_draft(
    router: DeterministicRouter,
    billing_passages: list[RetrievedPassage],
) -> None:
    """Account-specific billing questions require human review before sending (FR-13)."""
    ticket = UnifiedTicket(
        ticket_id="DEV-BILL-0002",
        channel=ChannelEnum.EMAIL,
        subject="Invoice higher than expected",
        body=(
            "Our invoice this month is considerably higher than usual. "
            "Could you explain what has driven it?"
        ),
        received_at=datetime(2026, 11, 18, 10, 0, 0, tzinfo=timezone.utc),
    )
    classification = ClassificationResult(
        ticket_id=ticket.ticket_id,
        intent=IntentCategory.BILLING_QUERY,
        urgency=UrgencyLevel.MEDIUM,
        confidence=0.93,
        intent_confidence=0.94,
        urgency_confidence=0.92,
        rationale="Account-specific invoice investigation.",
        model_name="mock-model",
        prompt_version="v1.0",
    )
    decision = router.route_ticket(
        ticket=ticket,
        classification=classification,
        retrieved_passages=billing_passages,
        cited_ids=["[DOC-BILL-001: Resolution]"],
    )

    assert decision.route == RouteOutcome.DRAFT_FOR_REVIEW
    assert decision.billing_risk_subtype == BillingRiskSubtype.ACCOUNT_SPECIFIC
    assert decision.reason_code == RoutingReasonCode.BILLING_ACCOUNT_SPECIFIC_REVIEW
    assert decision.primary_requirement_id == "FR-13"
    assert decision.review_payload is not None
    assert decision.review_payload.sent_to_customer is False
    assert len(decision.review_payload.verification_checklist) > 0


def test_billing_dispute_always_escalates(
    router: DeterministicRouter,
    billing_passages: list[RetrievedPassage],
) -> None:
    """Billing disputes and refund/credit demands must always escalate to a specialist."""
    ticket = UnifiedTicket(
        ticket_id="DEV-BILL-0003",
        channel=ChannelEnum.EMAIL,
        subject="Disputing incorrect charge on invoice",
        body=(
            "I dispute the charge on invoice INV-4921 and demand a full refund and billing credit."
        ),
        received_at=datetime(2026, 11, 18, 10, 0, 0, tzinfo=timezone.utc),
    )
    classification = ClassificationResult(
        ticket_id=ticket.ticket_id,
        intent=IntentCategory.BILLING_QUERY,
        urgency=UrgencyLevel.HIGH,
        confidence=0.95,
        intent_confidence=0.96,
        urgency_confidence=0.94,
        rationale="Financial dispute and refund demand.",
        model_name="mock-model",
        prompt_version="v1.0",
    )
    decision = router.route_ticket(
        ticket=ticket,
        classification=classification,
        retrieved_passages=billing_passages,
    )

    assert decision.route == RouteOutcome.ESCALATE
    assert decision.billing_risk_subtype == BillingRiskSubtype.DISPUTE_OR_FINANCIAL_DECISION
    assert decision.reason_code == RoutingReasonCode.MANDATORY_BILLING_DISPUTE
    assert decision.primary_requirement_id == "FR-13"
    assert decision.escalation_payload is not None
    assert decision.escalation_payload.target_specialist_group == "Billing Operations"


def test_billing_ambiguous_never_auto_responds(
    router: DeterministicRouter,
    billing_passages: list[RetrievedPassage],
) -> None:
    """Ambiguous billing inquiries must never auto-respond."""
    ticket = UnifiedTicket(
        ticket_id="DEV-BILL-0004",
        channel=ChannelEnum.CHAT,
        body="Billing question please check.",
        received_at=datetime(2026, 11, 18, 10, 0, 0, tzinfo=timezone.utc),
    )
    classification = ClassificationResult(
        ticket_id=ticket.ticket_id,
        intent=IntentCategory.BILLING_QUERY,
        urgency=UrgencyLevel.LOW,
        confidence=0.88,
        intent_confidence=0.90,
        urgency_confidence=0.86,
        rationale="Unclear billing question.",
        model_name="mock-model",
        prompt_version="v1.0",
    )
    decision = router.route_ticket(
        ticket=ticket,
        classification=classification,
        retrieved_passages=billing_passages,
    )

    assert decision.route == RouteOutcome.DRAFT_FOR_REVIEW
    assert decision.billing_risk_subtype == BillingRiskSubtype.AMBIGUOUS
    assert decision.reason_code == RoutingReasonCode.BILLING_AMBIGUOUS_RISK
    assert decision.eligible_for_auto_respond is False


# =============================================================================
# 4. Urgency and Severity Scenarios
# =============================================================================


def test_high_urgency_alone_does_not_force_escalation(
    router: DeterministicRouter,
    base_ticket: UnifiedTicket,
    sample_passages: list[RetrievedPassage],
    safe_guardrail: GuardrailCheckResult,
) -> None:
    """High urgency alone does not force escalation if low risk and well documented."""
    high_urgency_classification = ClassificationResult(
        ticket_id=base_ticket.ticket_id,
        intent=IntentCategory.DEPLOYMENT_FAILURE,
        urgency=UrgencyLevel.HIGH,
        confidence=0.94,
        intent_confidence=0.95,
        urgency_confidence=0.93,
        rationale="High-urgency deployment port mismatch with exact documentation.",
        model_name="mock-model",
        prompt_version="v1.0",
    )
    decision = router.route_ticket(
        ticket=base_ticket,
        classification=high_urgency_classification,
        retrieved_passages=sample_passages,
        guardrail_result=safe_guardrail,
        cited_ids=["[DOC-DEPLOY-001: Resolution]"],
    )

    assert decision.route == RouteOutcome.AUTO_RESPOND
    assert decision.reason_code == RoutingReasonCode.ELIGIBLE_AUTO_RESPOND


def test_high_urgency_without_kb_evidence_not_escalated_solely_on_urgency(
    router: DeterministicRouter,
    base_ticket: UnifiedTicket,
    safe_guardrail: GuardrailCheckResult,
) -> None:
    """High urgency alone does not force escalation;
    confident ticket without KB routes to review.
    """
    high_urgency_ticket = ClassificationResult(
        ticket_id=base_ticket.ticket_id,
        intent=IntentCategory.DATABASE_ISSUE,
        urgency=UrgencyLevel.HIGH,
        confidence=0.90,
        intent_confidence=0.91,
        urgency_confidence=0.89,
        rationale="Database performance issue with high classification confidence.",
        model_name="mock-model",
        prompt_version="v1.0",
    )
    decision = router.route_ticket(
        ticket=base_ticket,
        classification=high_urgency_ticket,
        retrieved_passages=[],  # No qualifying KB documentation
        guardrail_result=safe_guardrail,
    )

    # Must route to DRAFT_FOR_REVIEW under missing-KB gate, not generic high-urgency escalation
    assert decision.route == RouteOutcome.DRAFT_FOR_REVIEW
    assert decision.reason_code == RoutingReasonCode.MISSING_OR_WEAK_KB_EVIDENCE
    assert decision.primary_requirement_id == "FR-06"
    assert decision.confidence_threshold_applied == router.settings.confidence_threshold_review


# =============================================================================
# 5. Fail-Closed Guardrails Regression Tests
# =============================================================================


def test_missing_guardrail_result_cannot_reach_auto_respond(
    router: DeterministicRouter,
    base_ticket: UnifiedTicket,
    base_classification: ClassificationResult,
    sample_passages: list[RetrievedPassage],
) -> None:
    """Missing/unavailable guardrail result (None) must strictly block AUTO_RESPOND."""
    decision = router.route_ticket(
        ticket=base_ticket,
        classification=base_classification,
        retrieved_passages=sample_passages,
        guardrail_result=None,  # Unavailable / not evaluated
        cited_ids=["[DOC-DEPLOY-001: Resolution]"],
    )

    assert decision.route == RouteOutcome.DRAFT_FOR_REVIEW
    assert decision.reason_code == RoutingReasonCode.GUARDRAIL_UNAVAILABLE
    assert decision.guardrails_passed is False
    assert decision.eligible_for_auto_respond is False


def test_failed_guardrail_cannot_reach_auto_respond(
    router: DeterministicRouter,
    base_ticket: UnifiedTicket,
    base_classification: ClassificationResult,
    sample_passages: list[RetrievedPassage],
) -> None:
    """Failed safety guardrail (e.g. prompt injection) must strictly block release."""
    guardrail_fail = GuardrailCheckResult(
        is_safe=False,
        violation_type="prompt_injection",
        details="Ignore previous instructions detected in ticket body.",
    )
    decision = router.route_ticket(
        ticket=base_ticket,
        classification=base_classification,
        retrieved_passages=sample_passages,
        guardrail_result=guardrail_fail,
        cited_ids=["[DOC-DEPLOY-001: Resolution]"],
    )

    assert decision.route == RouteOutcome.ESCALATE
    assert decision.reason_code == RoutingReasonCode.GUARDRAIL_VIOLATION
    assert decision.guardrails_passed is False
    assert decision.escalation_payload is not None
    assert decision.escalation_payload.target_specialist_group == "Security Operations"


def test_missing_guardrail_preserves_mandatory_escalation_precedence(
    router: DeterministicRouter,
    base_ticket: UnifiedTicket,
    sample_passages: list[RetrievedPassage],
) -> None:
    """Missing guardrail must not override mandatory specialist escalation categories."""
    sec_classification = ClassificationResult(
        ticket_id=base_ticket.ticket_id,
        intent=IntentCategory.SECURITY_INCIDENT,
        urgency=UrgencyLevel.HIGH,
        confidence=0.98,
        intent_confidence=0.99,
        urgency_confidence=0.97,
        rationale="Account breach.",
        model_name="mock-model",
        prompt_version="v1.0",
    )
    decision = router.route_ticket(
        ticket=base_ticket,
        classification=sec_classification,
        retrieved_passages=sample_passages,
        guardrail_result=None,  # Even if guardrail is None, security precedence holds
    )

    assert decision.route == RouteOutcome.ESCALATE
    assert decision.reason_code == RoutingReasonCode.MANDATORY_SECURITY_INCIDENT
    assert decision.escalation_payload is not None
    assert decision.escalation_payload.target_specialist_group == "Security Operations"


def test_guardrail_check_result_requires_is_safe() -> None:
    """GuardrailCheckResult cannot be constructed without explicitly supplying is_safe."""
    with pytest.raises(ValidationError):
        GuardrailCheckResult()  # type: ignore[call-arg]


def test_guardrail_check_result_explicit_safe_and_unsafe(
    router: DeterministicRouter,
    base_ticket: UnifiedTicket,
    base_classification: ClassificationResult,
    sample_passages: list[RetrievedPassage],
) -> None:
    """GuardrailCheckResult(is_safe=False) is rejected from AUTO_RESPOND; is_safe=True can pass."""
    unsafe_gr = GuardrailCheckResult(is_safe=False, violation_type="pii_leak")
    dec_unsafe = router.route_ticket(
        ticket=base_ticket,
        classification=base_classification,
        retrieved_passages=sample_passages,
        guardrail_result=unsafe_gr,
        cited_ids=["[DOC-DEPLOY-001: Resolution]"],
    )
    assert dec_unsafe.route == RouteOutcome.ESCALATE
    assert dec_unsafe.reason_code == RoutingReasonCode.GUARDRAIL_VIOLATION
    assert dec_unsafe.guardrails_passed is False
    assert dec_unsafe.eligible_for_auto_respond is False

    safe_gr = GuardrailCheckResult(is_safe=True)
    dec_safe = router.route_ticket(
        ticket=base_ticket,
        classification=base_classification,
        retrieved_passages=sample_passages,
        guardrail_result=safe_gr,
        cited_ids=["[DOC-DEPLOY-001: Resolution]"],
    )
    assert dec_safe.route == RouteOutcome.AUTO_RESPOND
    assert dec_safe.guardrails_passed is True
    assert dec_safe.eligible_for_auto_respond is True


# =============================================================================
# 6. Mandatory Citations Regression Tests
# =============================================================================


def test_missing_citations_cannot_reach_auto_respond(
    router: DeterministicRouter,
    base_ticket: UnifiedTicket,
    base_classification: ClassificationResult,
    sample_passages: list[RetrievedPassage],
    safe_guardrail: GuardrailCheckResult,
) -> None:
    """Missing citations (cited_ids=None) must block AUTO_RESPOND."""
    decision = router.route_ticket(
        ticket=base_ticket,
        classification=base_classification,
        retrieved_passages=sample_passages,
        guardrail_result=safe_guardrail,
        cited_ids=None,
    )

    assert decision.route == RouteOutcome.DRAFT_FOR_REVIEW
    assert decision.reason_code == RoutingReasonCode.CITATION_VALIDATION_FAILED
    assert decision.citations_verified is False
    assert decision.eligible_for_auto_respond is False


def test_empty_citations_cannot_reach_auto_respond(
    router: DeterministicRouter,
    base_ticket: UnifiedTicket,
    base_classification: ClassificationResult,
    sample_passages: list[RetrievedPassage],
    safe_guardrail: GuardrailCheckResult,
) -> None:
    """Empty citations (cited_ids=[]) must block AUTO_RESPOND."""
    decision = router.route_ticket(
        ticket=base_ticket,
        classification=base_classification,
        retrieved_passages=sample_passages,
        guardrail_result=safe_guardrail,
        cited_ids=[],
    )

    assert decision.route == RouteOutcome.DRAFT_FOR_REVIEW
    assert decision.reason_code == RoutingReasonCode.CITATION_VALIDATION_FAILED
    assert decision.citations_verified is False


def test_invalid_citations_cannot_reach_auto_respond(
    router: DeterministicRouter,
    base_ticket: UnifiedTicket,
    base_classification: ClassificationResult,
    sample_passages: list[RetrievedPassage],
    safe_guardrail: GuardrailCheckResult,
) -> None:
    """Hallucinated/unretrieved citation IDs must block AUTO_RESPOND."""
    decision = router.route_ticket(
        ticket=base_ticket,
        classification=base_classification,
        retrieved_passages=sample_passages,
        guardrail_result=safe_guardrail,
        cited_ids=["[DOC-FAKE-999: Invented]"],
    )

    assert decision.route == RouteOutcome.DRAFT_FOR_REVIEW
    assert decision.reason_code == RoutingReasonCode.CITATION_VALIDATION_FAILED
    assert decision.citations_verified is False


def test_valid_citations_pass_citation_gate(
    router: DeterministicRouter,
    base_ticket: UnifiedTicket,
    base_classification: ClassificationResult,
    sample_passages: list[RetrievedPassage],
    safe_guardrail: GuardrailCheckResult,
) -> None:
    """Valid citations resolving to retrieved passages pass the citation gate."""
    decision = router.route_ticket(
        ticket=base_ticket,
        classification=base_classification,
        retrieved_passages=sample_passages,
        guardrail_result=safe_guardrail,
        cited_ids=["[DOC-DEPLOY-001: Resolution]"],
    )

    assert decision.route == RouteOutcome.AUTO_RESPOND
    assert decision.citations_verified is True
    assert decision.eligible_for_auto_respond is True


# =============================================================================
# 7. Intent Confidence Gating Tests
# =============================================================================


def test_intent_confidence_controls_threshold_not_urgency_confidence(
    router: DeterministicRouter,
    base_ticket: UnifiedTicket,
    sample_passages: list[RetrievedPassage],
    safe_guardrail: GuardrailCheckResult,
) -> None:
    """Verify that intent_confidence, not urgency_confidence, controls auto-response."""
    # Case A: High urgency confidence (0.98), but low intent confidence (0.50)
    low_intent_high_urgency = ClassificationResult(
        ticket_id=base_ticket.ticket_id,
        intent=IntentCategory.DEPLOYMENT_FAILURE,
        urgency=UrgencyLevel.LOW,
        confidence=0.74,  # Composite
        intent_confidence=0.50,  # Below review threshold 0.60
        urgency_confidence=0.98,
        rationale="Uncertain intent prediction.",
        model_name="mock-model",
        prompt_version="v1.0",
    )
    decision_a = router.route_ticket(
        ticket=base_ticket,
        classification=low_intent_high_urgency,
        retrieved_passages=sample_passages,
        guardrail_result=safe_guardrail,
        cited_ids=["[DOC-DEPLOY-001: Resolution]"],
    )
    assert decision_a.route == RouteOutcome.ESCALATE
    assert decision_a.reason_code == RoutingReasonCode.LOW_CONFIDENCE_ESCALATE
    assert decision_a.intent_confidence_applied == 0.50
    assert decision_a.urgency_confidence_applied == 0.98

    # Case B: Low urgency confidence (0.45), but high intent confidence (0.92)
    high_intent_low_urgency = ClassificationResult(
        ticket_id=base_ticket.ticket_id,
        intent=IntentCategory.DEPLOYMENT_FAILURE,
        urgency=UrgencyLevel.LOW,
        confidence=0.68,  # Composite
        intent_confidence=0.92,  # Above auto threshold 0.80
        urgency_confidence=0.45,
        rationale="Confident intent prediction.",
        model_name="mock-model",
        prompt_version="v1.0",
    )
    decision_b = router.route_ticket(
        ticket=base_ticket,
        classification=high_intent_low_urgency,
        retrieved_passages=sample_passages,
        guardrail_result=safe_guardrail,
        cited_ids=["[DOC-DEPLOY-001: Resolution]"],
    )
    assert decision_b.route == RouteOutcome.AUTO_RESPOND
    assert decision_b.intent_confidence_applied == 0.92
    assert decision_b.urgency_confidence_applied == 0.45


# =============================================================================
# 8. Evidence and Kill Switch Gates
# =============================================================================


def test_weak_or_missing_retrieval_prevents_auto_response(
    router: DeterministicRouter,
    base_ticket: UnifiedTicket,
    base_classification: ClassificationResult,
) -> None:
    """Missing or weak KB retrieval blocks automatic response."""
    weak_passage = RetrievedPassage(
        chunk_id="DOC-DEPLOY-001#general",
        doc_id="DOC-DEPLOY-001",
        category="deployment",
        title="Deployment Overview",
        section_name="General",
        content="General deployment overview.",
        similarity_score=0.40,
        rank=1,
        citation_id="[DOC-DEPLOY-001: General]",
    )
    decision = router.route_ticket(
        ticket=base_ticket,
        classification=base_classification,
        retrieved_passages=[weak_passage],
    )

    assert decision.route == RouteOutcome.DRAFT_FOR_REVIEW
    assert decision.reason_code == RoutingReasonCode.MISSING_OR_WEAK_KB_EVIDENCE
    assert decision.eligible_for_auto_respond is False


def test_kill_switch_disables_auto_response_preserves_safe_draft(
    router: DeterministicRouter,
    base_ticket: UnifiedTicket,
    base_classification: ClassificationResult,
    sample_passages: list[RetrievedPassage],
    safe_guardrail: GuardrailCheckResult,
) -> None:
    """Operational kill switch prevents auto-sending while holding ticket safely as a draft."""
    decision = router.route_ticket(
        ticket=base_ticket,
        classification=base_classification,
        retrieved_passages=sample_passages,
        guardrail_result=safe_guardrail,
        cited_ids=["[DOC-DEPLOY-001: Resolution]"],
        override_kill_switch=False,  # Kill switch triggered
    )

    assert decision.route == RouteOutcome.DRAFT_FOR_REVIEW
    assert decision.reason_code == RoutingReasonCode.KILL_SWITCH_ACTIVE
    assert decision.kill_switch_active is True
    assert decision.review_payload is not None
    assert decision.review_payload.sent_to_customer is False


def test_escalation_payload_completeness_and_missing_history(
    router: DeterministicRouter,
    base_ticket: UnifiedTicket,
) -> None:
    """Escalation payload carries all context and explicitly states missing history."""
    classification = ClassificationResult(
        ticket_id=base_ticket.ticket_id,
        intent=IntentCategory.DEPLOYMENT_FAILURE,
        urgency=UrgencyLevel.LOW,
        confidence=0.45,
        intent_confidence=0.45,  # Below review threshold 0.60
        urgency_confidence=0.45,
        rationale="Low confidence prediction.",
        model_name="mock-model",
        prompt_version="v1.0",
    )
    decision = router.route_ticket(
        ticket=base_ticket,
        classification=classification,
    )

    assert decision.route == RouteOutcome.ESCALATE
    assert decision.reason_code == RoutingReasonCode.LOW_CONFIDENCE_ESCALATE
    payload = decision.escalation_payload
    assert payload is not None
    assert payload.ticket_id == base_ticket.ticket_id
    assert "Troubleshooting history unavailable" in payload.troubleshooting_history
    assert payload.documentation_available is False
    assert "documentation unavailable" in payload.documentation_notes.lower()
    assert payload.safe_guidance is None
    assert payload.documentation_excerpt is None


def test_escalation_evidence_labelled_as_documentation_excerpt(
    router: DeterministicRouter,
    base_ticket: UnifiedTicket,
    sample_passages: list[RetrievedPassage],
) -> None:
    """Escalation evidence must be labelled as documentation excerpt, not verified safe guidance."""
    sec_classification = ClassificationResult(
        ticket_id=base_ticket.ticket_id,
        intent=IntentCategory.SECURITY_INCIDENT,
        urgency=UrgencyLevel.HIGH,
        confidence=0.98,
        intent_confidence=0.99,
        urgency_confidence=0.97,
        rationale="Account breach.",
        model_name="mock-model",
        prompt_version="v1.0",
    )
    decision = router.route_ticket(
        ticket=base_ticket,
        classification=sec_classification,
        retrieved_passages=sample_passages,
    )
    payload = decision.escalation_payload
    assert payload is not None
    assert payload.documentation_excerpt is not None
    assert "Excerpt from [DOC-DEPLOY-001: Resolution]" in payload.documentation_excerpt
    assert payload.safe_guidance is None


def test_threshold_configuration_changes(
    router: DeterministicRouter,
    base_ticket: UnifiedTicket,
    base_classification: ClassificationResult,
    sample_passages: list[RetrievedPassage],
    safe_guardrail: GuardrailCheckResult,
) -> None:
    """Raising auto-respond threshold shifts eligible ticket to human-review draft."""
    decision = router.route_ticket(
        ticket=base_ticket,
        classification=base_classification,
        retrieved_passages=sample_passages,
        guardrail_result=safe_guardrail,
        cited_ids=["[DOC-DEPLOY-001: Resolution]"],
        confidence_threshold_auto_respond=0.98,  # Higher than intent_confidence 0.94
        confidence_threshold_review=0.60,
    )

    assert decision.route == RouteOutcome.DRAFT_FOR_REVIEW
    assert decision.reason_code == RoutingReasonCode.LOW_CONFIDENCE_REVIEW
    assert decision.confidence_threshold_applied == 0.98


def test_to_decision_log_entry_schema(
    router: DeterministicRouter,
    base_ticket: UnifiedTicket,
    base_classification: ClassificationResult,
    sample_passages: list[RetrievedPassage],
    safe_guardrail: GuardrailCheckResult,
) -> None:
    """Audit log entry serializes all required fields per Governance Framework (A8)."""
    decision = router.route_ticket(
        ticket=base_ticket,
        classification=base_classification,
        retrieved_passages=sample_passages,
        guardrail_result=safe_guardrail,
        cited_ids=["[DOC-DEPLOY-001: Resolution]"],
    )
    log_entry = decision.to_decision_log_entry()

    assert log_entry["ticket_id"] == base_ticket.ticket_id
    assert log_entry["route"] == "auto_respond"
    assert log_entry["primary_requirement_id"] == "FR-05"
    assert "FR-04" in log_entry["requirement_ids"]
    assert log_entry["reason_code"] == "ELIGIBLE_AUTO_RESPOND"
    assert log_entry["intent_confidence_applied"] == 0.94
    assert log_entry["urgency_confidence_applied"] == 0.90
    assert log_entry["eligible_for_auto_respond"] is True
    assert log_entry["citations_verified"] is True
    assert log_entry["guardrails_passed"] is True
    assert log_entry["kill_switch_active"] is False
    assert isinstance(log_entry["sources_used"], list)

    # Test serialization of escalation and review draft payloads
    esc_dec = router.route_ticket(
        ticket=base_ticket,
        classification=ClassificationResult(
            ticket_id=base_ticket.ticket_id,
            intent=IntentCategory.SECURITY_INCIDENT,
            urgency=UrgencyLevel.HIGH,
            confidence=0.99,
            intent_confidence=0.99,
            urgency_confidence=0.99,
            rationale="Security incident.",
            model_name="mock-model",
            prompt_version="v1.0",
        ),
        retrieved_passages=sample_passages,
    )
    assert len(esc_dec.to_decision_log_entry()["sources_used"]) > 0

    rev_dec = router.route_ticket(
        ticket=base_ticket,
        classification=base_classification,
        retrieved_passages=sample_passages,
        guardrail_result=None,
    )
    assert len(rev_dec.to_decision_log_entry()["sources_used"]) > 0


def test_assess_billing_risk_unit() -> None:
    """Direct tests for assess_billing_risk pattern classification under FR-13."""
    t_dispute = UnifiedTicket(
        ticket_id="T-1",
        channel=ChannelEnum.EMAIL,
        body="We demand a refund and credit reversal for overcharge.",
        received_at=datetime.now(timezone.utc),
    )
    expected_dispute = BillingRiskSubtype.DISPUTE_OR_FINANCIAL_DECISION
    assert assess_billing_risk(t_dispute).subtype == expected_dispute

    t_account = UnifiedTicket(
        ticket_id="T-2",
        channel=ChannelEnum.EMAIL,
        body="There are charges on our invoice for a service we do not believe we use.",
        received_at=datetime.now(timezone.utc),
    )
    assert assess_billing_risk(t_account).subtype == BillingRiskSubtype.ACCOUNT_SPECIFIC

    t_general = UnifiedTicket(
        ticket_id="T-3",
        channel=ChannelEnum.EMAIL,
        body="How is proration calculated when changing plans mid billing cycle?",
        received_at=datetime.now(timezone.utc),
    )
    assert assess_billing_risk(t_general).subtype == BillingRiskSubtype.GENERAL_INFORMATIONAL

    t_empty = UnifiedTicket(
        ticket_id="T-4",
        channel=ChannelEnum.CHAT,
        body="",
        received_at=datetime.now(timezone.utc),
    )
    assert assess_billing_risk(t_empty).subtype == BillingRiskSubtype.AMBIGUOUS


def test_verify_citations_unit(sample_passages: list[RetrievedPassage]) -> None:
    """Direct tests for citation validation against retrieved passages."""
    # Valid exact citation
    res_valid = verify_citations(sample_passages, ["[DOC-DEPLOY-001: Resolution]"])
    assert res_valid.is_valid is True
    assert len(res_valid.valid_citations) == 1

    # Valid doc ID format
    res_doc = verify_citations(sample_passages, ["DOC-DEPLOY-001"])
    assert res_doc.is_valid is True

    # Hallucinated / unretrieved doc ID
    res_hallucinated = verify_citations(sample_passages, ["[DOC-AUTH-001: Resolution]"])
    assert res_hallucinated.is_valid is False
    assert len(res_hallucinated.invalid_citations) == 1

    # Empty citations with require_at_least_one=False
    res_none = verify_citations(sample_passages, [], require_at_least_one=False)
    assert res_none.is_valid is True

    # Empty citations with require_at_least_one=True (default)
    res_empty_req = verify_citations(sample_passages, [], require_at_least_one=True)
    assert res_empty_req.is_valid is False


def test_strict_citation_matching_rejects_fragments(
    sample_passages: list[RetrievedPassage],
) -> None:
    """Strict exact citation matching rejects partial or fabricated fragments."""
    # Substring of section_name
    res_fragment = verify_citations(sample_passages, ["Resolution"])
    assert res_fragment.is_valid is False
    assert "Resolution" in res_fragment.invalid_citations

    # Incomplete prefix
    res_prefix = verify_citations(sample_passages, ["[DOC-"])
    assert res_prefix.is_valid is False
    assert "[DOC-" in res_prefix.invalid_citations

    # Partial doc ID
    res_partial_doc = verify_citations(sample_passages, ["DOC-DEPLOY"])
    assert res_partial_doc.is_valid is False
    assert "DOC-DEPLOY" in res_partial_doc.invalid_citations

    # Substring of title
    res_title = verify_citations(sample_passages, ["Port Mismatch"])
    assert res_title.is_valid is False
    assert "Port Mismatch" in res_title.invalid_citations

    # Fabricated section on valid doc ID
    res_fake_section = verify_citations(sample_passages, ["[DOC-DEPLOY-001: FakeSection]"])
    assert res_fake_section.is_valid is False
    assert "[DOC-DEPLOY-001: FakeSection]" in res_fake_section.invalid_citations

    # Exact valid citations pass
    res_exact_citation = verify_citations(sample_passages, ["[DOC-DEPLOY-001: Resolution]"])
    assert res_exact_citation.is_valid is True

    res_exact_chunk = verify_citations(sample_passages, ["DOC-DEPLOY-001#resolution"])
    assert res_exact_chunk.is_valid is True

    res_exact_doc = verify_citations(sample_passages, ["DOC-DEPLOY-001"])
    assert res_exact_doc.is_valid is True

    res_exact_doc_bracket = verify_citations(sample_passages, ["[DOC-DEPLOY-001]"])
    assert res_exact_doc_bracket.is_valid is True


def test_citation_validation_result_fail_closed_model() -> None:
    """CitationValidationResult requires is_valid with no default (fail-closed)."""
    with pytest.raises(ValidationError):
        CitationValidationResult()  # type: ignore[call-arg]

    res_safe = CitationValidationResult(is_valid=True)
    assert res_safe.is_valid is True

    res_unsafe = CitationValidationResult(is_valid=False)
    assert res_unsafe.is_valid is False


def test_auto_respond_records_sources_used(
    router: DeterministicRouter,
    base_ticket: UnifiedTicket,
    base_classification: ClassificationResult,
    sample_passages: list[RetrievedPassage],
    safe_guardrail: GuardrailCheckResult,
) -> None:
    """AUTO_RESPOND decision preserves retrieved source IDs in decision and audit log."""
    decision = router.route_ticket(
        ticket=base_ticket,
        classification=base_classification,
        retrieved_passages=sample_passages,
        guardrail_result=safe_guardrail,
        cited_ids=["[DOC-DEPLOY-001: Resolution]"],
    )

    assert decision.route == RouteOutcome.AUTO_RESPOND
    assert decision.documentation_available is True
    assert len(decision.sources_used) == 2
    assert decision.sources_used == [
        "[DOC-DEPLOY-001: Resolution]",
        "[DOC-DEPLOY-001: Notes]",
    ]

    log_entry = decision.to_decision_log_entry()
    assert log_entry["documentation_available"] is True
    assert log_entry["sources_used"] == decision.sources_used


def test_citation_verification_semantics_mandatory_escalation_and_review(
    router: DeterministicRouter,
    base_ticket: UnifiedTicket,
    sample_passages: list[RetrievedPassage],
    safe_guardrail: GuardrailCheckResult,
) -> None:
    """Passages attached to escalations/reviews show documentation_available=True
    while citations_verified remains False until response citations are evaluated.
    """
    # Mandatory Security Escalation with relevant passages
    sec_classification = ClassificationResult(
        ticket_id=base_ticket.ticket_id,
        intent=IntentCategory.SECURITY_INCIDENT,
        urgency=UrgencyLevel.HIGH,
        confidence=0.98,
        intent_confidence=0.98,
        urgency_confidence=0.98,
        rationale="Security alert.",
        model_name="mock-model",
        prompt_version="v1.0",
    )
    decision_sec = router.route_ticket(
        ticket=base_ticket,
        classification=sec_classification,
        retrieved_passages=sample_passages,
        guardrail_result=safe_guardrail,
    )
    assert decision_sec.route == RouteOutcome.ESCALATE
    assert decision_sec.documentation_available is True
    assert decision_sec.citations_verified is False
    assert len(decision_sec.sources_used) == 2

    # Account-specific billing review with relevant passages
    billing_ticket = UnifiedTicket(
        ticket_id="T-BILL-01",
        channel=ChannelEnum.EMAIL,
        body="Why does our invoice INV-4567 show unexpected charges?",
        received_at=datetime.now(timezone.utc),
    )
    bill_classification = ClassificationResult(
        ticket_id=billing_ticket.ticket_id,
        intent=IntentCategory.BILLING_QUERY,
        urgency=UrgencyLevel.MEDIUM,
        confidence=0.95,
        intent_confidence=0.95,
        urgency_confidence=0.95,
        rationale="Account invoice query.",
        model_name="mock-model",
        prompt_version="v1.0",
    )
    decision_bill = router.route_ticket(
        ticket=billing_ticket,
        classification=bill_classification,
        retrieved_passages=sample_passages,
        guardrail_result=safe_guardrail,
    )
    assert decision_bill.route == RouteOutcome.DRAFT_FOR_REVIEW
    assert decision_bill.documentation_available is True
    assert decision_bill.citations_verified is False
    assert len(decision_bill.sources_used) == 2


def test_billing_credit_card_payment_method_not_dispute(
    router: DeterministicRouter,
    sample_passages: list[RetrievedPassage],
    safe_guardrail: GuardrailCheckResult,
) -> None:
    """Inquiry regarding credit card payment methods is general informational, not a dispute."""
    ticket = UnifiedTicket(
        ticket_id="T-CARD-01",
        channel=ChannelEnum.EMAIL,
        subject="Payment method question",
        body=(
            "Do you accept credit card payments for our subscription? Can we pay with credit card?"
        ),
        received_at=datetime.now(timezone.utc),
    )
    assessment = assess_billing_risk(ticket)
    assert assessment.subtype == BillingRiskSubtype.GENERAL_INFORMATIONAL

    classification = ClassificationResult(
        ticket_id=ticket.ticket_id,
        intent=IntentCategory.BILLING_QUERY,
        urgency=UrgencyLevel.LOW,
        confidence=0.95,
        intent_confidence=0.95,
        urgency_confidence=0.95,
        rationale="Payment method inquiry.",
        model_name="mock-model",
        prompt_version="v1.0",
    )

    decision = router.route_ticket(
        ticket=ticket,
        classification=classification,
        retrieved_passages=sample_passages,
        guardrail_result=safe_guardrail,
        cited_ids=["[DOC-DEPLOY-001: Resolution]"],
    )
    assert decision.route == RouteOutcome.AUTO_RESPOND
    assert decision.billing_risk_subtype == BillingRiskSubtype.GENERAL_INFORMATIONAL
    assert decision.reason_code == RoutingReasonCode.ELIGIBLE_AUTO_RESPOND


def test_billing_credit_adjustments_and_compensation_are_disputes(
    router: DeterministicRouter,
    sample_passages: list[RetrievedPassage],
    safe_guardrail: GuardrailCheckResult,
) -> None:
    """Requests for billing credits, downtime credits, or adjustments are financial disputes."""
    ticket = UnifiedTicket(
        ticket_id="T-CREDIT-01",
        channel=ChannelEnum.EMAIL,
        subject="Downtime credit request",
        body="We request a service credit and fee adjustment due to the downtime outage.",
        received_at=datetime.now(timezone.utc),
    )
    assessment = assess_billing_risk(ticket)
    assert assessment.subtype == BillingRiskSubtype.DISPUTE_OR_FINANCIAL_DECISION

    classification = ClassificationResult(
        ticket_id=ticket.ticket_id,
        intent=IntentCategory.BILLING_QUERY,
        urgency=UrgencyLevel.HIGH,
        confidence=0.96,
        intent_confidence=0.96,
        urgency_confidence=0.96,
        rationale="Service credit request.",
        model_name="mock-model",
        prompt_version="v1.0",
    )

    decision = router.route_ticket(
        ticket=ticket,
        classification=classification,
        retrieved_passages=sample_passages,
        guardrail_result=safe_guardrail,
        cited_ids=["[DOC-DEPLOY-001: Resolution]"],
    )
    assert decision.route == RouteOutcome.ESCALATE
    assert decision.billing_risk_subtype == BillingRiskSubtype.DISPUTE_OR_FINANCIAL_DECISION
    assert decision.reason_code == RoutingReasonCode.MANDATORY_BILLING_DISPUTE


def test_billing_dispute_high_urgency_no_kb_routes_to_billing_operations(
    router: DeterministicRouter,
    safe_guardrail: GuardrailCheckResult,
) -> None:
    """A high-urgency billing dispute with no qualifying KB passages must route to
    Billing Operations.

    Generic HIGH_URGENCY_NO_KB_EVIDENCE (Tier 2 Engineering) must NOT intercept a billing dispute.
    """
    ticket = UnifiedTicket(
        ticket_id="T-DISPUTE-HIGH-NO-KB",
        channel=ChannelEnum.EMAIL,
        subject="URGENT: Billing chargeback and dispute notice",
        body=(
            "URGENT: We are disputing the $50,000 unauthorized charge on our account. "
            "Please issue an immediate refund and fee reversal or we will initiate a chargeback."
        ),
        received_at=datetime.now(timezone.utc),
    )
    classification = ClassificationResult(
        ticket_id=ticket.ticket_id,
        intent=IntentCategory.BILLING_QUERY,
        urgency=UrgencyLevel.HIGH,
        confidence=0.97,
        intent_confidence=0.97,
        urgency_confidence=0.97,
        rationale="Severe billing dispute and chargeback warning.",
        model_name="mock-model",
        prompt_version="v1.0",
    )

    decision = router.route_ticket(
        ticket=ticket,
        classification=classification,
        retrieved_passages=[],  # No qualifying KB documentation
        guardrail_result=safe_guardrail,
    )

    assert decision.route == RouteOutcome.ESCALATE
    assert decision.reason_code == RoutingReasonCode.MANDATORY_BILLING_DISPUTE
    assert decision.primary_requirement_id == "FR-13"
    assert decision.documentation_available is False
    assert decision.escalation_payload is not None
    assert decision.escalation_payload.target_specialist_group == "Billing Operations"
    # Ensure it was NOT routed to Tier 2 Engineering
    assert decision.escalation_payload.target_specialist_group != "Tier 2 Engineering"
    assert decision.reason_code != RoutingReasonCode.HIGH_URGENCY_NO_KB_EVIDENCE


def test_confidence_threshold_applied_audit_accuracy(
    router: DeterministicRouter,
    base_ticket: UnifiedTicket,
    sample_passages: list[RetrievedPassage],
    safe_guardrail: GuardrailCheckResult,
) -> None:
    """confidence_threshold_applied records the threshold used in the routing decision."""
    review_t = 0.70
    auto_t = 0.85

    # 1. Gate 9a: Intent confidence below review_thresh (< 0.70) -> records review_thresh
    dec_below_review = router.route_ticket(
        ticket=base_ticket,
        classification=ClassificationResult(
            ticket_id=base_ticket.ticket_id,
            intent=IntentCategory.DEPLOYMENT_FAILURE,
            urgency=UrgencyLevel.LOW,
            confidence=0.62,
            intent_confidence=0.62,
            urgency_confidence=0.80,
            rationale="Uncertain classification.",
            model_name="mock-model",
            prompt_version="v1.0",
        ),
        retrieved_passages=sample_passages,
        guardrail_result=safe_guardrail,
        cited_ids=["[DOC-DEPLOY-001: Resolution]"],
        confidence_threshold_review=review_t,
        confidence_threshold_auto_respond=auto_t,
    )
    assert dec_below_review.route == RouteOutcome.ESCALATE
    assert dec_below_review.reason_code == RoutingReasonCode.LOW_CONFIDENCE_ESCALATE
    assert dec_below_review.confidence_threshold_applied == review_t
    assert dec_below_review.to_decision_log_entry()["confidence_threshold_applied"] == review_t

    # 2. Gate 9b: Intent confidence between review_thresh and auto_thresh (0.75)
    dec_review_tier = router.route_ticket(
        ticket=base_ticket,
        classification=ClassificationResult(
            ticket_id=base_ticket.ticket_id,
            intent=IntentCategory.DEPLOYMENT_FAILURE,
            urgency=UrgencyLevel.LOW,
            confidence=0.75,
            intent_confidence=0.75,
            urgency_confidence=0.80,
            rationale="Moderate confidence classification.",
            model_name="mock-model",
            prompt_version="v1.0",
        ),
        retrieved_passages=sample_passages,
        guardrail_result=safe_guardrail,
        cited_ids=["[DOC-DEPLOY-001: Resolution]"],
        confidence_threshold_review=review_t,
        confidence_threshold_auto_respond=auto_t,
    )
    assert dec_review_tier.route == RouteOutcome.DRAFT_FOR_REVIEW
    assert dec_review_tier.reason_code == RoutingReasonCode.LOW_CONFIDENCE_REVIEW
    assert dec_review_tier.confidence_threshold_applied == auto_t
    assert dec_review_tier.to_decision_log_entry()["confidence_threshold_applied"] == auto_t

    # 3. Gate 5 (Missing KB): intent_confidence >= review_thresh (0.80 >= 0.70) -> review_thresh
    dec_missing_kb_review = router.route_ticket(
        ticket=base_ticket,
        classification=ClassificationResult(
            ticket_id=base_ticket.ticket_id,
            intent=IntentCategory.DEPLOYMENT_FAILURE,
            urgency=UrgencyLevel.LOW,
            confidence=0.80,
            intent_confidence=0.80,
            urgency_confidence=0.80,
            rationale="High confidence but missing KB docs.",
            model_name="mock-model",
            prompt_version="v1.0",
        ),
        retrieved_passages=[],  # No KB passages
        guardrail_result=safe_guardrail,
        confidence_threshold_review=review_t,
        confidence_threshold_auto_respond=auto_t,
    )
    assert dec_missing_kb_review.route == RouteOutcome.DRAFT_FOR_REVIEW
    assert dec_missing_kb_review.reason_code == RoutingReasonCode.MISSING_OR_WEAK_KB_EVIDENCE
    assert dec_missing_kb_review.confidence_threshold_applied == review_t
    assert dec_missing_kb_review.to_decision_log_entry()["confidence_threshold_applied"] == review_t

    # 4. Gate 5 (Missing KB): intent_confidence < review_thresh (0.55 < 0.70) -> review_thresh
    dec_missing_kb_escalate = router.route_ticket(
        ticket=base_ticket,
        classification=ClassificationResult(
            ticket_id=base_ticket.ticket_id,
            intent=IntentCategory.DEPLOYMENT_FAILURE,
            urgency=UrgencyLevel.LOW,
            confidence=0.55,
            intent_confidence=0.55,
            urgency_confidence=0.80,
            rationale="Low confidence and missing KB docs.",
            model_name="mock-model",
            prompt_version="v1.0",
        ),
        retrieved_passages=[],  # No KB passages
        guardrail_result=safe_guardrail,
        confidence_threshold_review=review_t,
        confidence_threshold_auto_respond=auto_t,
    )
    assert dec_missing_kb_escalate.route == RouteOutcome.ESCALATE
    assert dec_missing_kb_escalate.reason_code == RoutingReasonCode.LOW_CONFIDENCE_ESCALATE
    assert dec_missing_kb_escalate.confidence_threshold_applied == review_t
    assert (
        dec_missing_kb_escalate.to_decision_log_entry()["confidence_threshold_applied"] == review_t
    )


class TestRec04ProductionIncidentEscalation:
    """Targeted tests for REC-04 high-urgency production incident override (FR-07).

    Verifies:
    1. Active production outage -> mandatory human escalation (HIGH_URGENCY_PRODUCTION_INCIDENT).
    2. Failed emergency rollback affecting production -> mandatory escalation.
    3. Routine deployment question -> existing behavior preserved (eligible for auto-respond).
    4. Planned, non-emergency rollback question -> existing behavior preserved.
    5. Relevant KB evidence must not bypass the active-incident escalation rule.
    """

    @pytest.fixture
    def safe_guardrail(self) -> GuardrailCheckResult:
        return GuardrailCheckResult(is_safe=True)

    @pytest.fixture
    def sample_kb_passage(self) -> RetrievedPassage:
        return RetrievedPassage(
            chunk_id="DOC-DEPLOY-002#resolution",
            doc_id="DOC-DEPLOY-002",
            category="deployment",
            title="Service Revision Rollbacks",
            section_name="Resolution",
            content="To roll back revision, select 'Roll Back' on the revisions page.",
            similarity_score=0.92,
            rank=1,
            citation_id="[DOC-DEPLOY-002: Resolution]",
        )

    def test_active_production_outage_mandatory_escalation(
        self,
        base_ticket: UnifiedTicket,
        safe_guardrail: GuardrailCheckResult,
        sample_kb_passage: RetrievedPassage,
    ) -> None:
        """Verify active production incident escalates even when valid KB docs exist."""
        router = DeterministicRouter()
        ticket = UnifiedTicket(
            ticket_id="TEST-PROD-001",
            channel=ChannelEnum.EMAIL,
            subject="Production outage on checkout service",
            body="Production is down after latest release. Broken checkout for all users.",
            received_at=base_ticket.received_at,
            customer_id=base_ticket.customer_id,
            customer_name=base_ticket.customer_name,
            customer_tier=CustomerTier.ENTERPRISE,
        )
        classification = ClassificationResult(
            ticket_id=ticket.ticket_id,
            intent=IntentCategory.DEPLOYMENT_FAILURE,
            urgency=UrgencyLevel.HIGH,
            confidence=0.96,
            intent_confidence=0.98,
            urgency_confidence=0.98,
            rationale="Production deployment down.",
            model_name="test-model",
            prompt_version="v1.0",
        )

        decision = router.route_ticket(
            ticket=ticket,
            classification=classification,
            retrieved_passages=[sample_kb_passage],
            guardrail_result=safe_guardrail,
            cited_ids=["[DOC-DEPLOY-002: Resolution]"],
        )

        assert decision.route == RouteOutcome.ESCALATE
        assert decision.reason_code == RoutingReasonCode.HIGH_URGENCY_PRODUCTION_INCIDENT
        assert decision.eligible_for_auto_respond is False
        assert decision.escalation_payload is not None
        assert decision.escalation_payload.target_specialist_group == "Tier 2 Engineering"

    def test_failed_emergency_rollback_affecting_production_escalation(
        self,
        base_ticket: UnifiedTicket,
        safe_guardrail: GuardrailCheckResult,
        sample_kb_passage: RetrievedPassage,
    ) -> None:
        """Verify emergency rollback situation affecting production customers escalates."""
        router = DeterministicRouter()
        ticket = UnifiedTicket(
            ticket_id="TEST-ROLL-001",
            channel=ChannelEnum.EMAIL,
            subject="Bad release in production",
            body=(
                "We shipped a release yesterday that has broken checkout for a portion of "
                "our customers. We need emergency rollback to previous version immediately."
            ),
            received_at=base_ticket.received_at,
            customer_id=base_ticket.customer_id,
            customer_name=base_ticket.customer_name,
            customer_tier=CustomerTier.BUSINESS,
        )
        classification = ClassificationResult(
            ticket_id=ticket.ticket_id,
            intent=IntentCategory.ROLLBACK_REQUEST,
            urgency=UrgencyLevel.HIGH,
            confidence=0.97,
            intent_confidence=0.97,
            urgency_confidence=0.96,
            rationale="Emergency rollback needed for broken checkout in production.",
            model_name="test-model",
            prompt_version="v1.0",
        )

        decision = router.route_ticket(
            ticket=ticket,
            classification=classification,
            retrieved_passages=[sample_kb_passage],
            guardrail_result=safe_guardrail,
            cited_ids=["[DOC-DEPLOY-002: Resolution]"],
        )

        assert decision.route == RouteOutcome.ESCALATE
        assert decision.reason_code == RoutingReasonCode.HIGH_URGENCY_PRODUCTION_INCIDENT
        assert decision.eligible_for_auto_respond is False

    def test_routine_deployment_question_preserves_auto_respond(
        self,
        base_ticket: UnifiedTicket,
        safe_guardrail: GuardrailCheckResult,
        sample_kb_passage: RetrievedPassage,
    ) -> None:
        """Verify normal deployment how-to question is not blocked by production incident rule."""
        router = DeterministicRouter()
        ticket = UnifiedTicket(
            ticket_id="TEST-DEPLOY-001",
            channel=ChannelEnum.CHAT,
            subject="Deployment configuration question",
            body="How do I configure rolling update health check timeouts for staging containers?",
            received_at=base_ticket.received_at,
            customer_id=base_ticket.customer_id,
            customer_name=base_ticket.customer_name,
            customer_tier=CustomerTier.STANDARD,
        )
        classification = ClassificationResult(
            ticket_id=ticket.ticket_id,
            intent=IntentCategory.CONFIGURATION_HELP,
            urgency=UrgencyLevel.LOW,
            confidence=0.95,
            intent_confidence=0.95,
            urgency_confidence=0.90,
            rationale="Routine deployment configuration inquiry.",
            model_name="test-model",
            prompt_version="v1.0",
        )

        decision = router.route_ticket(
            ticket=ticket,
            classification=classification,
            retrieved_passages=[sample_kb_passage],
            guardrail_result=safe_guardrail,
            cited_ids=["[DOC-DEPLOY-002: Resolution]"],
        )

        assert decision.route == RouteOutcome.AUTO_RESPOND
        assert decision.reason_code == RoutingReasonCode.ELIGIBLE_AUTO_RESPOND
        assert decision.eligible_for_auto_respond is True

    def test_planned_non_emergency_rollback_preserves_auto_respond(
        self,
        base_ticket: UnifiedTicket,
        safe_guardrail: GuardrailCheckResult,
        sample_kb_passage: RetrievedPassage,
    ) -> None:
        """Verify planned rollback guidance with low/medium urgency can still auto-respond."""
        router = DeterministicRouter()
        ticket = UnifiedTicket(
            ticket_id="TEST-PLAN-001",
            channel=ChannelEnum.CHAT,
            subject="How to revert revision",
            body="What is the standard procedure to roll back a service revision in the dashboard?",
            received_at=base_ticket.received_at,
            customer_id=base_ticket.customer_id,
            customer_name=base_ticket.customer_name,
            customer_tier=CustomerTier.STANDARD,
        )
        classification = ClassificationResult(
            ticket_id=ticket.ticket_id,
            intent=IntentCategory.ROLLBACK_REQUEST,
            urgency=UrgencyLevel.LOW,
            confidence=0.94,
            intent_confidence=0.95,
            urgency_confidence=0.90,
            rationale="General procedural rollback question.",
            model_name="test-model",
            prompt_version="v1.0",
        )

        decision = router.route_ticket(
            ticket=ticket,
            classification=classification,
            retrieved_passages=[sample_kb_passage],
            guardrail_result=safe_guardrail,
            cited_ids=["[DOC-DEPLOY-002: Resolution]"],
        )

        assert decision.route == RouteOutcome.AUTO_RESPOND
        assert decision.reason_code == RoutingReasonCode.ELIGIBLE_AUTO_RESPOND
        assert decision.eligible_for_auto_respond is True

    def test_relevant_kb_evidence_does_not_bypass_incident_escalation(
        self,
        base_ticket: UnifiedTicket,
        safe_guardrail: GuardrailCheckResult,
    ) -> None:
        """Verify that perfect KB similarity and valid citations cannot override outage rule."""
        router = DeterministicRouter()
        ticket = UnifiedTicket(
            ticket_id="TEST-KB-OVERRIDE-001",
            channel=ChannelEnum.EMAIL,
            subject="Critical database failure in prod",
            body="Our prod database is down. All customer queries failing with connection error.",
            received_at=base_ticket.received_at,
            customer_id=base_ticket.customer_id,
            customer_name=base_ticket.customer_name,
            customer_tier=CustomerTier.ENTERPRISE,
        )
        classification = ClassificationResult(
            ticket_id=ticket.ticket_id,
            intent=IntentCategory.DATABASE_ISSUE,
            urgency=UrgencyLevel.HIGH,
            confidence=0.99,
            intent_confidence=0.99,
            urgency_confidence=0.99,
            rationale="Critical production database down.",
            model_name="test-model",
            prompt_version="v1.0",
        )
        perfect_passage = RetrievedPassage(
            chunk_id="DOC-DB-001#resolution",
            doc_id="DOC-DB-001",
            category="database",
            title="Database Connection Troubleshooting",
            section_name="Resolution",
            content="Check the connection pool settings in the CloudServe database settings tab.",
            similarity_score=1.0,
            rank=1,
            citation_id="[DOC-DB-001: Resolution]",
        )

        decision = router.route_ticket(
            ticket=ticket,
            classification=classification,
            retrieved_passages=[perfect_passage],
            guardrail_result=safe_guardrail,
            cited_ids=["[DOC-DB-001: Resolution]"],
        )

        # Gate 2h must preempt Gate 9 (Auto-Respond) regardless of KB score
        assert decision.route == RouteOutcome.ESCALATE
        assert decision.reason_code == RoutingReasonCode.HIGH_URGENCY_PRODUCTION_INCIDENT
        assert decision.documentation_available is True
        assert decision.eligible_for_auto_respond is False


class TestRoutingPrecedenceRegression:
    """Regression tests for routing gate precedence between mandatory operational escalation

    (Gate 2h) and non-malicious output grounding failures (Gate 5).
    """

    def test_dev_0030_outage_escalates_even_if_grounding_failed(
        self,
        base_ticket: UnifiedTicket,
    ) -> None:
        """Verify that DEV-0030 escalates to Tier 2 Engineering even if grounding failed.

        Gate 2h (mandatory policy escalation) must take precedence over Gate 5 grounding draft.
        """
        router = DeterministicRouter()
        ticket = UnifiedTicket(
            ticket_id="DEV-0030",
            channel=ChannelEnum.EMAIL,
            subject="Urgent: Need to rollback production release immediately",
            body=(
                "Our latest release v2.4.1 broke the checkout flow in production. "
                "Customers cannot complete purchases. We need to rollback to v2.4.0 "
                "immediately. What is the safest way to execute this rollback without "
                "causing downtime?"
            ),
            received_at=base_ticket.received_at,
            customer_id=base_ticket.customer_id,
            customer_name=base_ticket.customer_name,
            customer_tier=CustomerTier.ENTERPRISE,
        )
        classification = ClassificationResult(
            ticket_id=ticket.ticket_id,
            intent=IntentCategory.ROLLBACK_REQUEST,
            urgency=UrgencyLevel.HIGH,
            confidence=0.97,
            intent_confidence=0.97,
            urgency_confidence=0.95,
            rationale="Urgent request to rollback production release breaking checkout.",
            model_name="test-model",
            prompt_version="v1.0",
        )
        # Simulate output grounding failure that was generated
        grounding_failure_guardrail = GuardrailCheckResult(
            is_safe=False,
            violation_type="output_grounding_failed",
            details="Grounding validation failed: 2 unsupported sentences.",
        )

        decision = router.route_ticket(
            ticket=ticket,
            classification=classification,
            retrieved_passages=[],
            guardrail_result=grounding_failure_guardrail,
            cited_ids=[],
        )

        # Must be mandatory escalation to Tier 2 Engineering (Gate 2h), NOT review draft (Gate 5)
        assert decision.route == RouteOutcome.ESCALATE
        assert decision.reason_code == RoutingReasonCode.HIGH_URGENCY_PRODUCTION_INCIDENT
        assert decision.escalation_payload is not None
        assert decision.escalation_payload.target_specialist_group == "Tier 2 Engineering"
        assert decision.eligible_for_auto_respond is False

    def test_ordinary_grounding_failure_routes_to_draft_for_review(
        self,
        base_ticket: UnifiedTicket,
        sample_passages: list[RetrievedPassage],
    ) -> None:
        """Verify routine tickets with grounding failures route to DRAFT_FOR_REVIEW."""
        router = DeterministicRouter()
        classification = ClassificationResult(
            ticket_id=base_ticket.ticket_id,
            intent=IntentCategory.DEPLOYMENT_FAILURE,
            urgency=UrgencyLevel.MEDIUM,  # Routine / non-emergency
            confidence=0.95,
            intent_confidence=0.95,
            urgency_confidence=0.90,
            rationale="Routine deployment port configuration question.",
            model_name="test-model",
            prompt_version="v1.0",
        )
        grounding_failure_guardrail = GuardrailCheckResult(
            is_safe=False,
            violation_type="output_grounding_failed",
            details="Grounding validation failed: factual claim not found in KB.",
        )

        decision = router.route_ticket(
            ticket=base_ticket,
            classification=classification,
            retrieved_passages=sample_passages,
            guardrail_result=grounding_failure_guardrail,
            cited_ids=["[DOC-DEPLOY-001: Resolution]"],
        )

        assert decision.route == RouteOutcome.DRAFT_FOR_REVIEW
        assert decision.reason_code == RoutingReasonCode.GROUNDING_VALIDATION_FAILED
        assert decision.review_payload is not None
        assert decision.eligible_for_auto_respond is False

    def test_prompt_injection_escalates_to_secops_in_gate1(
        self,
        base_ticket: UnifiedTicket,
    ) -> None:
        """Verify that genuine security violations (prompt injection) escalate to SecOps."""
        router = DeterministicRouter()
        classification = ClassificationResult(
            ticket_id=base_ticket.ticket_id,
            intent=IntentCategory.ROLLBACK_REQUEST,
            urgency=UrgencyLevel.HIGH,
            confidence=0.90,
            intent_confidence=0.90,
            urgency_confidence=0.90,
            rationale="Simulated injection attempt.",
            model_name="test-model",
            prompt_version="v1.0",
        )
        security_violation_guardrail = GuardrailCheckResult(
            is_safe=False,
            violation_type="prompt_injection",
            details="Prompt injection signature detected.",
        )

        decision = router.route_ticket(
            ticket=base_ticket,
            classification=classification,
            retrieved_passages=[],
            guardrail_result=security_violation_guardrail,
            cited_ids=[],
        )

        assert decision.route == RouteOutcome.ESCALATE
        assert decision.reason_code == RoutingReasonCode.GUARDRAIL_VIOLATION
        assert decision.escalation_payload is not None
        assert decision.escalation_payload.target_specialist_group == "Security Operations"
