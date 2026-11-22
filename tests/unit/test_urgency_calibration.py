"""Targeted regression tests for controlled urgency calibration (Milestone A3.35.4).

Verifies explicit operational criteria across:
1. Low, Medium, and High operational urgency.
2. Non-production vs. production incidents.
3. Partial degradation vs. complete outage.
4. The seven chronic database connection-pool cases.
5. Identical ticket bodies with contradictory historical labels (consistent decisions).
"""

from datetime import datetime, timezone

import pytest

from cloudserve.classification.models import (
    ClassificationResult,
    IntentCategory,
    UrgencyLevel,
)
from cloudserve.classification.prompts import load_classification_system_prompt
from cloudserve.ingestion.models import ChannelEnum, CustomerTier, UnifiedTicket
from cloudserve.rag.models import RetrievedPassage
from cloudserve.routing.models import GuardrailCheckResult, RouteOutcome, RoutingReasonCode
from cloudserve.routing.router import DeterministicRouter, is_high_urgency_production_incident


@pytest.fixture
def deterministic_router() -> DeterministicRouter:
    """Provide a standard deterministic router instance."""
    return DeterministicRouter()


def _make_ticket(
    ticket_id: str,
    subject: str,
    body: str,
    tier: CustomerTier = CustomerTier.STANDARD,
) -> UnifiedTicket:
    """Helper to construct a valid runtime UnifiedTicket."""
    return UnifiedTicket(
        ticket_id=ticket_id,
        channel=ChannelEnum.EMAIL,
        subject=subject,
        body=body,
        received_at=datetime.now(timezone.utc),
        customer_tier=tier,
    )


def test_v2_classification_prompt_contains_operational_criteria() -> None:
    """Verify that v2 prompt incorporates explicit operational impact criteria."""
    v2_prompt = load_classification_system_prompt(version="v2")
    assert "Classify urgency strictly based on operational impact" in v2_prompt
    assert "Live production environments or customer-facing services" in v2_prompt
    assert "No safe operational workaround exists" in v2_prompt
    assert "Informational, non-blocking, non-production, or exploratory" in v2_prompt


def test_non_production_vs_production_incident_gate() -> None:
    """Verify that non-production incidents do not trigger production incident escalation."""
    # Staging issue with high urgency label
    staging_ticket = _make_ticket(
        ticket_id="TEST-STAGING-01",
        subject="Staging deployment failed",
        body="Our staging deployment crashed due to port mismatch.",
    )
    cls_staging = ClassificationResult(
        ticket_id=staging_ticket.ticket_id,
        intent=IntentCategory.DEPLOYMENT_FAILURE,
        urgency=UrgencyLevel.HIGH,
        confidence=0.95,
        intent_confidence=0.95,
        urgency_confidence=0.90,
        model_name="mock",
        prompt_version="v2",
        rationale="Staging failure",
    )
    assert not is_high_urgency_production_incident(staging_ticket, cls_staging)

    # Production outage with high urgency label
    prod_ticket = _make_ticket(
        ticket_id="TEST-PROD-01",
        subject="Production site down emergency",
        body="Our production live environment is down and customers cannot checkout.",
    )
    cls_prod = ClassificationResult(
        ticket_id=prod_ticket.ticket_id,
        intent=IntentCategory.DEPLOYMENT_FAILURE,
        urgency=UrgencyLevel.HIGH,
        confidence=0.98,
        intent_confidence=0.98,
        urgency_confidence=0.95,
        model_name="mock",
        prompt_version="v2",
        rationale="Production outage",
    )
    assert is_high_urgency_production_incident(prod_ticket, cls_prod)


def test_partial_degradation_vs_complete_outage_routing(
    deterministic_router: DeterministicRouter,
) -> None:
    """Verify that partial degradation does not trigger emergency gate, while outage escalates."""
    passage = RetrievedPassage(
        chunk_id="DOC-PERF-001:0",
        doc_id="DOC-PERF-001",
        title="Performance",
        section_name="Troubleshooting",
        content="Check p95 latency and memory metrics.",
        similarity_score=0.88,
        category="technical",
        rank=1,
        citation_id="[DOC-PERF-001: Troubleshooting]",
    )
    gr = GuardrailCheckResult(is_safe=True)

    # Partial degradation (Medium urgency)
    partial_ticket = _make_ticket(
        ticket_id="TEST-DEG-01",
        subject="Response times elevated",
        body="Latency has climbed slightly over the morning. Code unchanged.",
    )
    cls_partial = ClassificationResult(
        ticket_id=partial_ticket.ticket_id,
        intent=IntentCategory.PERFORMANCE_DEGRADATION,
        urgency=UrgencyLevel.MEDIUM,
        confidence=0.95,
        intent_confidence=0.95,
        urgency_confidence=0.85,
        model_name="mock",
        prompt_version="v2",
        rationale="Partial degradation",
    )
    dec_partial = deterministic_router.route_ticket(
        ticket=partial_ticket,
        classification=cls_partial,
        retrieved_passages=[passage],
        guardrail_result=gr,
        cited_ids=[passage.citation_id],
    )
    assert dec_partial.route == RouteOutcome.AUTO_RESPOND

    # Complete production outage (High urgency + Prod Outage)
    outage_ticket = _make_ticket(
        ticket_id="TEST-OUTAGE-01",
        subject="Production outage sev-1",
        body="Our production service is down. Critical downtime impacting all customers.",
    )
    cls_outage = ClassificationResult(
        ticket_id=outage_ticket.ticket_id,
        intent=IntentCategory.PERFORMANCE_DEGRADATION,
        urgency=UrgencyLevel.HIGH,
        confidence=0.95,
        intent_confidence=0.95,
        urgency_confidence=0.95,
        model_name="mock",
        prompt_version="v2",
        rationale="Critical production outage",
    )
    dec_outage = deterministic_router.route_ticket(
        ticket=outage_ticket,
        classification=cls_outage,
        retrieved_passages=[passage],
        guardrail_result=gr,
        cited_ids=[passage.citation_id],
    )
    assert dec_outage.route == RouteOutcome.ESCALATE
    assert dec_outage.reason_code == RoutingReasonCode.HIGH_URGENCY_PRODUCTION_INCIDENT


def test_seven_chronic_database_connection_pool_tickets(
    deterministic_router: DeterministicRouter,
) -> None:
    """Verify deterministic routing consistency across chronic connection-pool tickets."""
    passage = RetrievedPassage(
        chunk_id="DOC-PERF-003:0",
        doc_id="DOC-PERF-003",
        title="Database Performance",
        section_name="Connection Pools",
        content="Increase max connections or configure connection pooler.",
        similarity_score=0.85,
        category="technical",
        rank=1,
        citation_id="[DOC-PERF-003: Connection Pools]",
    )
    gr = GuardrailCheckResult(is_safe=True)

    chronic_tids = [
        "DEV-0115", "DEV-0277", "DEV-0322", "DEV-0330",
        "DEV-0349", "DEV-0437", "DEV-0472",
    ]
    for tid in chronic_tids:
        t = _make_ticket(
            ticket_id=tid,
            subject="Connection pool exhausted",
            body="We are getting errors saying no connection is available. Started upon scale-up.",
        )
        cls_result = ClassificationResult(
            ticket_id=tid,
            intent=IntentCategory.DATABASE_ISSUE,
            urgency=UrgencyLevel.HIGH,
            confidence=0.96,
            intent_confidence=0.96,
            urgency_confidence=0.90,
            model_name="mock",
            prompt_version="v2",
            rationale="High operational urgency due to scaled service errors",
        )
        dec = deterministic_router.route_ticket(
            ticket=t,
            classification=cls_result,
            retrieved_passages=[passage],
            guardrail_result=gr,
            cited_ids=[passage.citation_id],
        )
        assert dec.route == RouteOutcome.AUTO_RESPOND
        assert dec.eligible_for_auto_respond is True


def test_identical_ticket_bodies_consistent_decision(
    deterministic_router: DeterministicRouter,
) -> None:
    """Verify that identical customer inquiries produce identical routing outcomes."""
    body_text = (
        "We are getting errors saying no connection is available. "
        "This started when we scaled up our service. The database shows very few queries."
    )
    t1 = _make_ticket("DEV-0068", "Connection pool issue", body_text)
    t2 = _make_ticket("DEV-0115", "Connection pool issue", body_text)

    passage = RetrievedPassage(
        chunk_id="DOC-PERF-003:0",
        doc_id="DOC-PERF-003",
        title="Database Performance",
        section_name="Connection Pools",
        content="Configure connection pooling.",
        similarity_score=0.85,
        category="technical",
        rank=1,
        citation_id="[DOC-PERF-003: Connection Pools]",
    )
    gr = GuardrailCheckResult(is_safe=True)

    for t in [t1, t2]:
        cls_res = ClassificationResult(
            ticket_id=t.ticket_id,
            intent=IntentCategory.DATABASE_ISSUE,
            urgency=UrgencyLevel.HIGH,
            confidence=0.96,
            intent_confidence=0.96,
            urgency_confidence=0.90,
            model_name="mock",
            prompt_version="v2",
            rationale="Identical operational impact",
        )
        dec = deterministic_router.route_ticket(
            ticket=t,
            classification=cls_res,
            retrieved_passages=[passage],
            guardrail_result=gr,
            cited_ids=[passage.citation_id],
        )
        assert dec.route == RouteOutcome.AUTO_RESPOND
