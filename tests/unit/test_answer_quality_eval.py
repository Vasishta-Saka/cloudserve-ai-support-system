"""Unit and integration tests for Answer Quality Evaluator (A3.34).

Verifies:
1. Explicit ticket_id mapping against reference response benchmark.
2. must_mention coverage measurement and ratio calculation.
3. must_not_claim violation detection.
4. Diagnostic semantic similarity (TF-IDF cosine).
5. Independent citation validity and grounding checks.
6. Correct escalation protection (correct escalations are NEVER penalized as failed answers).
7. Missing reference responses handled gracefully (unmapped tickets are not penalized).
8. Quota-paused tickets categorized correctly.
9. End-to-end batch evaluation and artifact generation.
"""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from cloudserve.classification.models import (
    ClassificationResult,
    IntentCategory,
    UrgencyLevel,
)
from cloudserve.evaluation.answer_eval import (
    AnswerOutcomeCategory,
    AnswerQualityReport,
    ReferenceResponseRecord,
    check_must_mention,
    check_must_not_claim,
    compute_diagnostic_similarity,
    evaluate_answers_batch,
    evaluate_ticket_answer,
)
from cloudserve.evaluation.runner import EvaluationRunner
from cloudserve.generation.grounding import GroundingValidationResult
from cloudserve.generation.models import GeneratedResponse
from cloudserve.ingestion.models import (
    ChannelEnum,
    CustomerTier,
    EvaluationRecord,
    TicketLabels,
    UnifiedTicket,
)
from cloudserve.routing.billing import assess_billing_risk
from cloudserve.routing.models import (
    RouteOutcome,
    RoutingDecision,
    RoutingReasonCode,
)
from cloudserve.service import ProcessedTicketResult


def _make_dummy_ticket(ticket_id: str = "DEV-0001") -> UnifiedTicket:
    return UnifiedTicket(
        ticket_id=ticket_id,
        channel=ChannelEnum.EMAIL,
        customer_id="CUST-100",
        customer_tier=CustomerTier.ENTERPRISE,
        received_at=datetime.now(timezone.utc),
        subject="Test Ticket",
        body="This is a test support ticket body.",
    )


def _make_dummy_result(
    ticket_id: str = "DEV-0001",
    route: RouteOutcome = RouteOutcome.AUTO_RESPOND,
    reason_code: RoutingReasonCode = RoutingReasonCode.ELIGIBLE_AUTO_RESPOND,
    generated_text: str | None = "Here is the resolution to your query [DOC-TEST-001].",
    delivery_status: str = "released",
    grounding_passed: bool = True,
    persistence_status: str = "persisted",
    error: str | None = None,
) -> ProcessedTicketResult:
    ticket = _make_dummy_ticket(ticket_id)
    classification = ClassificationResult(
        ticket_id=ticket_id,
        intent=IntentCategory.DEPLOYMENT_FAILURE,
        urgency=UrgencyLevel.MEDIUM,
        confidence=0.92,
        intent_confidence=0.92,
        urgency_confidence=0.90,
        alternatives=[],
        rationale="Test classification",
        model_name="mock-model",
        prompt_version="1.0.0",
        is_fallback=False,
    )
    routing_dec = RoutingDecision(
        ticket_id=ticket_id,
        route=route,
        reason_code=reason_code,
        reason_description="Test routing decision",
        primary_requirement_id="REC-01",
        requirement_ids=["REC-01"],
        confidence_threshold_applied=0.80,
        retrieval_similarity_threshold_applied=0.65,
        eligible_for_auto_respond=(route == RouteOutcome.AUTO_RESPOND),
        citations_verified=True,
        guardrails_passed=True,
        kill_switch_active=False,
        documentation_available=True,
        sources_used=["DOC-TEST-001"],
    )
    gen_resp = (
        GeneratedResponse(
            ticket_id=ticket_id,
            customer_response=generated_text,
            agent_note="Test note",
            citation_ids=["DOC-TEST-001"],
            insufficient_evidence=False,
            unsupported_claims=[],
            model_name="mock-model",
            prompt_version="1.0.0",
            is_fallback=False,
            attempt_id="att-1",
            generation_duration_ms=120.0,
        )
        if generated_text
        else None
    )
    grounding_res = (
        GroundingValidationResult(
            is_grounded=grounding_passed,
            verified_citations=["DOC-TEST-001"],
            unsupported_reasons=[] if grounding_passed else ["Unverified claim"],
            details="Grounding check",
        )
        if generated_text
        else None
    )

    return ProcessedTicketResult(
        ticket=ticket,
        classification=classification,
        billing_risk=assess_billing_risk(ticket),
        routing_decision=routing_dec,
        generated_response=gen_resp,
        grounding_result=grounding_res,
        final_customer_response=generated_text if delivery_status == "released" else None,
        audit_id=f"audit-{ticket_id}",
        persistence_status=persistence_status,
        delivery_status=delivery_status,
        processing_latency_ms=150.0,
        error=error,
    )


def _make_dummy_eval_record(
    ticket_id: str = "DEV-0001",
    expected_route: str = "auto_respond",
    answerable_from_docs: bool = True,
) -> EvaluationRecord:
    ticket = _make_dummy_ticket(ticket_id)
    labels = TicketLabels(
        intent="deployment_failure",
        urgency="medium",
        expected_route=expected_route,
        answerable_from_docs=answerable_from_docs,
        expected_doc_ids=["DOC-TEST-001"],
        must_not_auto_respond=(expected_route == "escalate"),
    )
    return EvaluationRecord(ticket=ticket, labels=labels)


# ============================================================================
# Unit Tests for Helper Functions
# ============================================================================

def test_check_must_mention_exact_and_partial() -> None:
    text = "Please check your raw body payload and ensure your endpoint is idempotent."
    reqs = ["raw body", "idempotent", "backoff"]
    total, covered, ratio, missing = check_must_mention(text, reqs)
    assert total == 3
    assert covered == 2
    assert ratio == pytest.approx(2 / 3, 0.01)
    assert missing == ["backoff"]


def test_check_must_mention_empty_requirements() -> None:
    text = "Any text here."
    total, covered, ratio, missing = check_must_mention(text, [])
    assert total == 0
    assert covered == 0
    assert ratio == 1.0
    assert missing == []


def test_check_must_not_claim_clean() -> None:
    text = "We are reviewing your configuration and documentation."
    claims = ["a refund has been issued", "the issue has been fixed on our side"]
    total, viol_count, violated = check_must_not_claim(text, claims)
    assert total == 2
    assert viol_count == 0
    assert violated == []


def test_check_must_not_claim_violation_detected() -> None:
    text = "Good news, the issue has been fixed on our side and we are monitoring."
    claims = ["a refund has been issued", "the issue has been fixed on our side"]
    total, viol_count, violated = check_must_not_claim(text, claims)
    assert total == 2
    assert viol_count == 1
    assert violated == ["the issue has been fixed on our side"]


def test_compute_diagnostic_similarity_informative() -> None:
    cand = "When configuring SSO, verify the SAML certificate expiration in the admin console."
    ref = "For SSO configuration issues, please verify SAML certificate expiration via console."
    sim = compute_diagnostic_similarity(cand, ref)
    assert 0.0 < sim <= 1.0
    assert isinstance(sim, float)

    # Empty text handles cleanly
    assert compute_diagnostic_similarity("", ref) == 0.0
    assert compute_diagnostic_similarity(cand, "") == 0.0


# ============================================================================
# Evaluation Logic & Correct Escalation Protection Tests
# ============================================================================

def test_correct_escalation_is_never_penalized() -> None:
    """Requirement 5: Do not penalize correct escalations as failed answers."""
    result = _make_dummy_result(
        ticket_id="DEV-0015",
        route=RouteOutcome.ESCALATE,
        reason_code=RoutingReasonCode.MANDATORY_SECURITY_INCIDENT,
        generated_text=None,
        delivery_status="held_for_review",
    )
    gt = _make_dummy_eval_record("DEV-0015", expected_route="escalate", answerable_from_docs=False)
    ref = ReferenceResponseRecord(
        ticket_id="DEV-0015",
        intent="security_incident",
        reference_response="Reference response from senior engineer.",
        must_mention=[],
        must_not_claim=["a refund has been issued"],
    )

    eval_res = evaluate_ticket_answer(result, ref, gt)
    assert eval_res.outcome_category == AnswerOutcomeCategory.CORRECT_ESCALATION
    assert eval_res.is_penalized_failure is False
    assert len(eval_res.failure_reasons) == 0


def test_unexpected_escalation_is_penalized() -> None:
    """If expected route was auto-respond but system escalated without justification."""
    result = _make_dummy_result(
        ticket_id="DEV-0001",
        route=RouteOutcome.ESCALATE,
        reason_code=RoutingReasonCode.LOW_CONFIDENCE_REVIEW,
        generated_text=None,
        delivery_status="held_for_review",
    )
    gt = _make_dummy_eval_record(
        "DEV-0001", expected_route="auto_respond", answerable_from_docs=True
    )
    ref = ReferenceResponseRecord(
        ticket_id="DEV-0001",
        intent="deployment_failure",
        reference_response="Reference response.",
        must_mention=[],
        must_not_claim=[],
    )

    eval_res = evaluate_ticket_answer(result, ref, gt)
    assert eval_res.outcome_category == AnswerOutcomeCategory.INCORRECT_ROUTING
    assert eval_res.is_penalized_failure is True
    assert "Escalated unexpectedly" in eval_res.failure_reasons[0]


def test_must_not_claim_violation_penalizes_answer() -> None:
    """Requirement 2: Claim violation must fail the answer."""
    bad_text = (
        "Your ticket is resolved: a refund has been issued to your credit card [DOC-TEST-001]."
    )
    result = _make_dummy_result(
        ticket_id="DEV-0004",
        route=RouteOutcome.AUTO_RESPOND,
        generated_text=bad_text,
        delivery_status="released",
    )
    gt = _make_dummy_eval_record("DEV-0004", expected_route="auto_respond")
    ref = ReferenceResponseRecord(
        ticket_id="DEV-0004",
        intent="billing_query",
        reference_response="Reference guidance on billing.",
        must_mention=[],
        must_not_claim=["a refund has been issued"],
    )

    eval_res = evaluate_ticket_answer(result, ref, gt)
    assert eval_res.outcome_category == AnswerOutcomeCategory.FAILED_ANSWER
    assert eval_res.is_penalized_failure is True
    assert eval_res.has_must_not_claim_violation is True
    assert "a refund has been issued" in eval_res.violated_claims


def test_missing_reference_mapping_handled_safely() -> None:
    """Requirement 1 & 9: Missing reference mapping handled gracefully."""
    result = _make_dummy_result(
        ticket_id="DEV-0099",
        route=RouteOutcome.AUTO_RESPOND,
        generated_text="Valid resolution steps provided here [DOC-TEST-001].",
        delivery_status="released",
    )
    gt = _make_dummy_eval_record("DEV-0099", expected_route="auto_respond")

    # Pass reference=None (one of the 300 tickets without reference responses)
    eval_res = evaluate_ticket_answer(result, reference=None, ground_truth=gt)
    assert eval_res.has_reference is False
    assert eval_res.outcome_category == AnswerOutcomeCategory.CORRECT_AUTO_RESOLVE
    assert eval_res.is_penalized_failure is False
    assert eval_res.must_mention_coverage_ratio is None
    assert eval_res.diagnostic_semantic_similarity is None


def test_quota_paused_ticket_not_penalized() -> None:
    """Requirement 5: Quota paused tickets must not be penalized."""
    result = _make_dummy_result(
        ticket_id="DEV-0250",
        route=RouteOutcome.DRAFT_FOR_REVIEW,
        generated_text=None,
        delivery_status="held_for_review",
        persistence_status="paused_quota",
        error="ProviderQuotaExhaustedError: Daily quota exhausted",
    )
    gt = _make_dummy_eval_record("DEV-0250")
    ref = ReferenceResponseRecord(
        ticket_id="DEV-0250",
        intent="database_issue",
        reference_response="Guidance.",
    )

    eval_res = evaluate_ticket_answer(result, ref, gt)
    assert eval_res.outcome_category == AnswerOutcomeCategory.QUOTA_PAUSED
    assert eval_res.is_penalized_failure is False


def test_hallucinated_citations_in_released_answer_penalized() -> None:
    """Requirement 4: Citation validity and grounding independently evaluated."""
    text_with_fake_doc = "Check docs here [DOC-NONEXISTENT-999]."
    result = _make_dummy_result(
        ticket_id="DEV-0030",
        route=RouteOutcome.AUTO_RESPOND,
        generated_text=text_with_fake_doc,
        delivery_status="released",
    )
    gt = _make_dummy_eval_record("DEV-0030", expected_route="auto_respond")
    ref = ReferenceResponseRecord(
        ticket_id="DEV-0030",
        intent="deployment_failure",
        reference_response="Valid guidance.",
    )

    # Valid KB only has DOC-TEST-001
    eval_res = evaluate_ticket_answer(
        result,
        ref,
        gt,
        kb_doc_ids={"DOC-TEST-001"},
    )
    assert eval_res.citations_valid is False
    assert "DOC-NONEXISTENT-999" in eval_res.hallucinated_citations
    assert eval_res.is_penalized_failure is True
    assert eval_res.outcome_category == AnswerOutcomeCategory.FAILED_ANSWER


# ============================================================================
# Batch Evaluation & Integration Tests
# ============================================================================

def test_evaluate_answers_batch_macro_metrics() -> None:
    """Test batch evaluation aggregates accurately across mapped and unmapped tickets."""
    # 3 tickets: 1 correct auto-resolve, 1 correct escalation, 1 unmapped auto-resolve
    res1 = _make_dummy_result("DEV-0001", route=RouteOutcome.AUTO_RESPOND)
    res2 = _make_dummy_result(
        "DEV-0015",
        route=RouteOutcome.ESCALATE,
        reason_code=RoutingReasonCode.MANDATORY_SECURITY_INCIDENT,
        generated_text=None,
        delivery_status="held_for_review",
    )
    res3 = _make_dummy_result("DEV-0305", route=RouteOutcome.AUTO_RESPOND)

    gt1 = _make_dummy_eval_record("DEV-0001", expected_route="auto_respond")
    gt2 = _make_dummy_eval_record("DEV-0015", expected_route="escalate", answerable_from_docs=False)
    gt3 = _make_dummy_eval_record("DEV-0305", expected_route="auto_respond")

    ref1 = ReferenceResponseRecord(
        ticket_id="DEV-0001",
        intent="deployment_failure",
        reference_response="Here is the resolution to your query [DOC-TEST-001].",
        must_mention=["resolution"],
        must_not_claim=["a refund has been issued"],
    )
    ref2 = ReferenceResponseRecord(
        ticket_id="DEV-0015",
        intent="security_incident",
        reference_response="Security incident escalated to Tier 2.",
        must_mention=[],
        must_not_claim=["a refund has been issued"],
    )
    # DEV-0305 has no reference record

    report: AnswerQualityReport = evaluate_answers_batch(
        results=[res1, res2, res3],
        reference_responses={"DEV-0001": ref1, "DEV-0015": ref2},
        ground_truth={"DEV-0001": gt1, "DEV-0015": gt2, "DEV-0305": gt3},
        kb_doc_ids={"DOC-TEST-001"},
        is_mock=True,
    )

    assert report.total_tickets == 3
    assert report.reference_mapped_tickets == 2
    assert report.reference_coverage_pct == pytest.approx(66.67, 0.01)
    assert report.evaluated_answer_count == 2
    assert report.correct_auto_resolve_count == 2  # res1 and res3
    assert report.correct_escalation_count == 1   # res2
    assert report.must_not_claim_clean_rate == 1.0
    assert report.total_claim_violations_count == 0
    assert report.overall_answer_pass_rate == 1.0
    assert report.is_mock is True


def test_runner_generates_answer_quality_artifacts(tmp_path: Path) -> None:
    """Integration test: Runner generates answer quality report and line-delimited records."""
    import json

    # Create dummy dataset of 2 tickets
    input_file = tmp_path / "dev_tickets.json"
    dataset_records = [
        {
            "ticket_id": "DEV-0001",
            "channel": "chat",
            "subject": "",
            "body": "Dependency resolution error during build.",
            "received_at": "2026-05-23T22:43:00Z",
            "labels": {
                "intent": "deployment_failure",
                "urgency": "medium",
                "expected_route": "auto_respond",
                "answerable_from_docs": True,
                "expected_doc_ids": ["DOC-DEPLOY-001"],
            },
        },
        {
            "ticket_id": "DEV-0015",
            "channel": "email",
            "subject": "Security notification",
            "body": "Potential credential compromise detected on API key.",
            "received_at": "2026-05-23T22:50:00Z",
            "labels": {
                "intent": "security_incident",
                "urgency": "high",
                "expected_route": "escalate",
                "answerable_from_docs": False,
                "expected_doc_ids": [],
            },
        },
    ]
    with open(input_file, "w", encoding="utf-8") as f:
        json.dump(dataset_records, f)

    # Create reference responses file mapping DEV-0001
    ref_file = tmp_path / "ground_truth_responses.json"
    ref_data = [
        {
            "ticket_id": "DEV-0001",
            "intent": "deployment_failure",
            "reference_response": "Guidance on dependency resolution.",
            "must_mention": ["dependency"],
            "must_not_claim": ["a refund has been issued"],
        }
    ]
    with open(ref_file, "w", encoding="utf-8") as f:
        json.dump(ref_data, f)

    out_dir = tmp_path / "eval_run_out"
    runner = EvaluationRunner()
    run_res = runner.run(
        input_path=input_file,
        output_dir=out_dir,
        mode="mock",
        reference_responses_path=ref_file,
    )

    assert run_res.status == "completed"
    assert (out_dir / "answer_quality_report.json").exists()
    assert (out_dir / "answer_quality_results.jsonl").exists()
    assert (out_dir / "summary_metrics.json").exists()
    assert (out_dir / "EVALUATION_REPORT.md").exists()

    with open(out_dir / "summary_metrics.json", encoding="utf-8") as f:
        summary = json.load(f)
    assert "answer_quality" in summary
    aq = summary["answer_quality"]
    assert aq["total_tickets"] == 2
    assert aq["reference_mapped_tickets"] == 1
    assert aq["is_mock"] is True

    with open(out_dir / "EVALUATION_REPORT.md", encoding="utf-8") as f:
        report_text = f.read()
    assert "## 7.1 Answer Quality Evaluation against Reference Responses" in report_text
    assert "Reference-Mapped Tickets Evaluated" in report_text
