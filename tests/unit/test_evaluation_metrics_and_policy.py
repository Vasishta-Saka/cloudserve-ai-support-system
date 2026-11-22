"""Unit tests for Evaluation Metrics, Confidence Calibration, Retrieval Metrics,
Policy & Safety Compliance, Human Review Packet Generation, and Artifact Emission.
"""

import uuid
from datetime import datetime, timezone
from pathlib import Path

from cloudserve.classification.models import (
    ClassificationResult,
    IntentCategory,
    UrgencyLevel,
)
from cloudserve.evaluation.human_review import (
    calculate_binary_kappa,
    calculate_weighted_kappa,
    generate_human_review_template_csv,
    redact_pii_best_effort,
    sample_human_review_packet,
)
from cloudserve.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_confidence_calibration,
    calculate_urgency_metrics,
)
from cloudserve.evaluation.policy_eval import (
    evaluate_policy_and_safety,
)
from cloudserve.evaluation.reporter import generate_evaluation_artifacts
from cloudserve.evaluation.retrieval_eval import (
    compute_ticket_retrieval_metrics,
    evaluate_batch_retrieval,
)
from cloudserve.generation.models import GeneratedResponse
from cloudserve.ingestion.models import (
    ChannelEnum,
    CustomerRegion,
    CustomerTier,
    EvaluationRecord,
    LanguageFluency,
    TicketLabels,
    UnifiedTicket,
)
from cloudserve.rag.models import RetrievedPassage
from cloudserve.routing.billing import BillingRiskAssessment, BillingRiskSubtype
from cloudserve.routing.models import (
    GuardrailCheckResult,
    RouteOutcome,
    RoutingDecision,
    RoutingReasonCode,
    RoutingRequirementID,
)
from cloudserve.service import ProcessedTicketResult


def _create_mock_ticket(
    ticket_id: str = "DEV-0001", clean_body: str = "Test body"
) -> UnifiedTicket:
    return UnifiedTicket(
        ticket_id=ticket_id,
        channel=ChannelEnum.EMAIL,
        subject="Test subject",
        body=clean_body,
        received_at=datetime.now(timezone.utc),
        customer_id="CUST-100",
        customer_name="Test User",
        customer_tier=CustomerTier.STANDARD,
        customer_region=CustomerRegion.NORTH_AMERICA,
        language_fluency=LanguageFluency.FLUENT,
    )


def _create_mock_result(
    ticket_id: str = "DEV-0001",
    ticket: UnifiedTicket | None = None,
    intent: IntentCategory = IntentCategory.API_USAGE_QUESTION,
    urgency: UrgencyLevel = UrgencyLevel.MEDIUM,
    intent_conf: float = 0.90,
    route: RouteOutcome = RouteOutcome.AUTO_RESPOND,
    effective_route: RouteOutcome = RouteOutcome.AUTO_RESPOND,
    delivery_status: str = "released",
    persistence_status: str = "persisted",
    has_customer_response: bool = True,
    customer_response_text: str | None = None,
    passages: list[RetrievedPassage] | None = None,
    guardrails_passed: bool = True,
    kill_switch_active: bool = False,
    citations_verified: bool = True,
    billing_subtype: BillingRiskSubtype = BillingRiskSubtype.GENERAL_INFORMATIONAL,
    is_fallback: bool = False,
) -> ProcessedTicketResult:
    if ticket is None:
        ticket = _create_mock_ticket(ticket_id=ticket_id)
    else:
        ticket_id = ticket.ticket_id
    classification = ClassificationResult(
        ticket_id=ticket_id,
        intent=intent,
        urgency=urgency,
        confidence=intent_conf,
        intent_confidence=intent_conf,
        urgency_confidence=0.85,
        rationale="Test classification rationale",
        model_name="mock-classifier-v1",
        prompt_version="1.0.0",
        is_fallback=is_fallback,
    )
    billing_risk = BillingRiskAssessment(
        subtype=billing_subtype,
        confidence=0.90,
        risk_score=0.10,
        indicators=[],
        rationale="General informational billing inquiry",
    )
    ret_passages = (
        passages
        if passages is not None
        else [
            RetrievedPassage(
                chunk_id="DOC-API-001#guide",
                doc_id="DOC-API-001",
                category="api",
                title="API Guide",
                section_name="Guide",
                content="API endpoint documentation.",
                similarity_score=0.88,
                rank=1,
                citation_id="[DOC-API-001: Guide]",
            )
        ]
    )
    gen_resp = None
    final_resp = None
    if has_customer_response:
        resp_text = (
            customer_response_text
            if customer_response_text is not None
            else "Please use [DOC-API-001] for endpoint reference."
        )
        gen_resp = GeneratedResponse(
            ticket_id=ticket_id,
            customer_response=resp_text,
            agent_note="Standard response grounded in documentation.",
            citation_ids=["DOC-API-001"],
            insufficient_evidence=False,
            unsupported_claims=[],
        )
        if (
            route == RouteOutcome.AUTO_RESPOND
            and effective_route == RouteOutcome.AUTO_RESPOND
            and delivery_status == "released"
        ):
            final_resp = resp_text

    decision = RoutingDecision(
        ticket_id=ticket_id,
        route=route,
        primary_requirement_id=RoutingRequirementID.FR_05.value,
        requirement_ids=[RoutingRequirementID.FR_04.value, RoutingRequirementID.FR_05.value],
        reason_code=RoutingReasonCode.ELIGIBLE_AUTO_RESPOND
        if route == RouteOutcome.AUTO_RESPOND
        else RoutingReasonCode.LOW_CONFIDENCE_REVIEW,
        reason_description="Deterministic route test decision",
        billing_risk_subtype=billing_subtype,
        intent_confidence_applied=intent_conf,
        urgency_confidence_applied=0.85,
        confidence_applied=intent_conf,
        confidence_threshold_applied=0.80,
        retrieval_similarity_threshold_applied=0.65,
        eligible_for_auto_respond=(route == RouteOutcome.AUTO_RESPOND),
        citations_verified=citations_verified,
        guardrails_passed=guardrails_passed,
        kill_switch_active=kill_switch_active,
        documentation_available=len(ret_passages) > 0,
        sources_used=[p.doc_id for p in ret_passages],
    )

    return ProcessedTicketResult(
        ticket=ticket,
        classification=classification,
        billing_risk=billing_risk,
        retrieved_passages=ret_passages,
        generated_response=gen_resp,
        routing_decision=decision,
        final_customer_response=final_resp,
        private_agent_notes="Internal agent verification notes",
        processing_latency_ms=125.0,
        audit_id=str(uuid.uuid4()),
        processing_run_id=str(uuid.uuid4()),
        effective_route=effective_route,
        persistence_status=persistence_status,
        delivery_status=delivery_status,
    )


# =============================================================================
# 1. Classification Metrics Tests
# =============================================================================


def test_classification_metrics_all_22_classes_and_zero_support() -> None:
    """Verify precision, recall, represented macro-F1, fixed-22 macro-F1, and confusion matrix."""
    all_22 = [c.value for c in IntentCategory]
    assert len(all_22) == 22

    # Provide predictions for only 2 classes; the other 20 classes have zero support
    y_true = ["api_usage_question", "api_usage_question", "billing_query", "billing_query"]
    y_pred = ["api_usage_question", "billing_query", "billing_query", "billing_query"]

    report = calculate_classification_metrics(y_true, y_pred)

    assert report.total_samples == 4
    # Accuracy: 3 / 4 = 0.75
    assert report.accuracy == 0.75

    # api_usage_question: TP=1, FP=0, FN=1, Support=2 -> P=1.0, R=0.5, F1 = 2/3 = 0.6667
    api_m = report.per_class["api_usage_question"]
    assert api_m.support == 2
    assert api_m.precision == 1.0
    assert api_m.recall == 0.5
    assert api_m.f1 == 0.6667

    # billing_query: TP=2, FP=1, FN=0, Support=2 -> P=2/3 = 0.6667, R=1.0, F1 = 0.8
    bill_m = report.per_class["billing_query"]
    assert bill_m.support == 2
    assert bill_m.precision == 0.6667
    assert bill_m.recall == 1.0
    assert bill_m.f1 == 0.8

    # Represented macro-F1: mean of 0.6667 and 0.8 = 0.7333
    assert report.macro_f1_represented == 0.7333

    # 20 zero-support classes
    assert len(report.zero_support_classes) == 20
    assert "security_incident" in report.zero_support_classes
    assert report.per_class["security_incident"].support == 0
    assert report.per_class["security_incident"].f1 == 0.0

    # Fixed 22 macro-F1: (0.6667 + 0.8000 + 20 * 0.0) / 22 = 1.4667 / 22 = 0.0667
    assert report.macro_f1_fixed_22 == 0.0667

    # Confusion matrix structure
    assert len(report.confusion_matrix) >= 22
    assert report.confusion_matrix["api_usage_question"]["api_usage_question"] == 1
    assert report.confusion_matrix["api_usage_question"]["billing_query"] == 1
    assert report.confusion_matrix["billing_query"]["billing_query"] == 2


def test_urgency_metrics_three_tiers() -> None:
    """Verify precision, recall, macro-F1, and 3x3 confusion matrix for urgency."""
    y_true = ["low", "low", "medium", "medium", "high", "high"]
    y_pred = ["low", "medium", "medium", "medium", "high", "low"]

    report = calculate_urgency_metrics(y_true, y_pred)
    assert report.total_samples == 6
    assert report.accuracy == round(4 / 6, 4)

    # low: TP=1, FP=1, FN=1, S=2 -> P=0.5, R=0.5, F1=0.5
    assert report.per_class["low"].precision == 0.5
    assert report.per_class["low"].recall == 0.5
    assert report.per_class["low"].f1 == 0.5

    # medium: TP=2, FP=1, FN=0, S=2 -> P=2/3=0.6667, R=1.0, F1=0.8
    assert report.per_class["medium"].precision == 0.6667
    assert report.per_class["medium"].recall == 1.0

    # high: TP=1, FP=0, FN=1, S=2 -> P=1.0, R=0.5, F1=0.6667
    assert report.per_class["high"].precision == 1.0
    assert report.per_class["high"].recall == 0.5

    # 3x3 confusion matrix
    assert set(report.confusion_matrix.keys()) == {"low", "medium", "high"}
    assert report.confusion_matrix["high"]["low"] == 1
    assert report.confusion_matrix["high"]["high"] == 1


# =============================================================================
# 2. Confidence Calibration Tests
# =============================================================================


def test_calibration_one_reliable_bin_four_empty_bins() -> None:
    """Verify 1 reliable populated bin and 4 empty bins meets governance target."""
    # 20 samples in bin [0.8, 1.0], 0 in bins [0.0, 0.8)
    confs = [0.90] * 20
    accs = [True] * 18 + [False] * 2  # acc = 0.90, conf = 0.90 -> gap = 0.0 <= 0.05

    report = calculate_confidence_calibration(confs, accs, n_bins=5, min_samples_per_bin=10)

    assert report.total_samples == 20
    assert report.reliable_bins_count == 1
    # Empty bins do not trigger the sparse-bin rule!
    assert report.governance_target_status == "meets_target"
    assert report.meets_governance_target is True
    assert report.expected_calibration_error == 0.0
    assert report.reliable_bins_ece == 0.0
    assert report.reliable_bins_sample_denominator == 20
    assert report.reliable_bins_sample_coverage_pct == 100.0


def test_calibration_one_reliable_bin_plus_one_sparse_bin() -> None:
    """Verify 1 reliable bin plus 1 sparse populated bin triggers inconclusive status."""
    # 20 samples in [0.8, 1.0] (reliable), plus 3 samples in [0.0, 0.2) (sparse: < 10)
    confs = [0.90] * 20 + [0.10] * 3
    accs = ([True] * 18 + [False] * 2) + ([False] * 3)

    report = calculate_confidence_calibration(confs, accs, n_bins=5, min_samples_per_bin=10)

    assert report.total_samples == 23
    assert report.reliable_bins_count == 1
    # Sparse bin (<10) exists, so governance evaluation MUST be inconclusive!
    assert report.governance_target_status == "inconclusive_due_to_small_bins"
    assert report.meets_governance_target is False
    assert report.reliable_bins_sample_denominator == 20
    assert report.reliable_bins_ece == 0.0
    assert report.reliable_bins_sample_coverage_pct == round(100.0 * 20 / 23, 2)


def test_calibration_all_populated_bins_reliable_meets_and_fails_target() -> None:
    """Verify when all populated bins are reliable, the 5-percentage-point target is evaluated."""
    # Scenario A: 5 bins, 20 samples each, all gaps <= 0.05 -> meets target
    confs_pass = [0.10] * 20 + [0.30] * 20 + [0.50] * 20 + [0.70] * 20 + [0.90] * 20
    accs_pass = (
        [True] * 2
        + [False] * 18  # acc 0.10, conf 0.10 -> gap 0.0
        + [True] * 6
        + [False] * 14  # acc 0.30, conf 0.30 -> gap 0.0
        + [True] * 10
        + [False] * 10  # acc 0.50, conf 0.50 -> gap 0.0
        + [True] * 14
        + [False] * 6  # acc 0.70, conf 0.70 -> gap 0.0
        + [True] * 18
        + [False] * 2  # acc 0.90, conf 0.90 -> gap 0.0
    )
    rep_pass = calculate_confidence_calibration(confs_pass, accs_pass, n_bins=5)
    assert rep_pass.reliable_bins_count == 5
    assert rep_pass.governance_target_status == "meets_target"
    assert rep_pass.meets_governance_target is True
    assert rep_pass.expected_calibration_error == 0.0

    # Scenario B: 5 bins, 20 samples each, one bin gap = 0.20 > 0.05 -> fails target
    accs_fail = (
        [True] * 2
        + [False] * 18
        + [True] * 6
        + [False] * 14
        + [True] * 10
        + [False] * 10
        + [True] * 14
        + [False] * 6
        + [True] * 14
        + [False] * 6  # acc 0.70 vs conf 0.90 -> gap 0.20
    )
    rep_fail = calculate_confidence_calibration(confs_pass, accs_fail, n_bins=5)
    assert rep_fail.reliable_bins_count == 5
    assert rep_fail.governance_target_status == "fails_target"
    assert rep_fail.meets_governance_target is False
    assert rep_fail.max_calibration_error == 0.20


def test_calibration_no_populated_bins_zero_samples() -> None:
    """Verify zero samples returns 5 empty bins with inconclusive governance status."""
    report = calculate_confidence_calibration([], [], n_bins=5)
    assert report.total_samples == 0
    assert len(report.bins) == 5
    assert all(b.sample_count == 0 for b in report.bins)
    assert report.reliable_bins_count == 0
    assert report.governance_target_status == "inconclusive_due_to_small_bins"
    assert report.meets_governance_target is False


def test_calibration_mock_mode_diagnostic() -> None:
    """Verify mock mode marks model-calibration as not assessed and does not claim target met."""
    confs = [0.0] * 50
    accs = [True] * 2 + [False] * 48

    report = calculate_confidence_calibration(confs, accs, n_bins=5, is_mock=True)
    assert report.governance_target_status == "not_assessed_mock_mode"
    assert report.meets_governance_target is False
    assert "Mock confidence diagnostic" in report.notes


# =============================================================================
# 3. Retrieval Metrics Tests
# =============================================================================


def test_retrieval_metrics_deduplication_and_fixed_denominator() -> None:
    """Verify deduplication of chunks to document IDs and fixed denominator of 4."""
    # 2 chunks from DOC-API-001, 1 from DOC-AUTH-001, 1 from DOC-API-001 again
    passages = [
        RetrievedPassage(
            chunk_id="DOC-API-001#c0",
            doc_id="DOC-API-001",
            category="api",
            title="API",
            section_name="c0",
            content="text",
            similarity_score=0.8,
            rank=1,
            citation_id="[DOC-API-001: c0]",
        ),
        RetrievedPassage(
            chunk_id="DOC-API-001#c1",
            doc_id="DOC-API-001",
            category="api",
            title="API",
            section_name="c1",
            content="text",
            similarity_score=0.75,
            rank=2,
            citation_id="[DOC-API-001: c1]",
        ),
        RetrievedPassage(
            chunk_id="DOC-AUTH-001#c0",
            doc_id="DOC-AUTH-001",
            category="auth",
            title="Auth",
            section_name="c0",
            content="text",
            similarity_score=0.7,
            rank=3,
            citation_id="[DOC-AUTH-001: c0]",
        ),
    ]
    # Expected doc is only DOC-API-001
    expected = ["DOC-API-001"]

    metrics = compute_ticket_retrieval_metrics(passages, expected, top_k=4)

    # In top 4 passages: unique docs are [DOC-API-001, DOC-AUTH-001] (2 unique docs)
    # Matching expected doc is {DOC-API-001} (count = 1)
    # Precision@4 uses fixed denominator 4: 1 / 4 = 0.25!
    assert metrics["precision_at_4"] == 0.25
    # Recall@4: 1 / 1 = 1.0
    assert metrics["recall_at_4"] == 1.0
    # Hit Rate@4: 1.0
    assert metrics["hit_rate_at_4"] == 1.0
    # MRR: DOC-API-001 is at rank 1 -> 1.0
    assert metrics["reciprocal_rank"] == 1.0


def test_batch_retrieval_answerable_and_unanswerable() -> None:
    """Verify batch retrieval evaluation separates answerable from unanswerable tickets."""
    # Ticket 1: answerable, returns 1 doc
    res1 = _create_mock_result(ticket_id="DEV-0001")
    gt1 = EvaluationRecord(
        ticket=_create_mock_ticket("DEV-0001"),
        labels=TicketLabels(
            intent="api_usage_question",
            urgency="low",
            expected_route="auto_respond",
            answerable_from_docs=True,
            expected_doc_ids=["DOC-API-001"],
        ),
    )

    # Ticket 2: unanswerable, returns 0 docs
    res2 = _create_mock_result(ticket_id="DEV-0002", passages=[])
    gt2 = EvaluationRecord(
        ticket=_create_mock_ticket("DEV-0002"),
        labels=TicketLabels(
            intent="unclear_request",
            urgency="high",
            expected_route="escalate",
            answerable_from_docs=False,
            expected_doc_ids=[],
        ),
    )

    batch_report = evaluate_batch_retrieval([res1, res2], {"DEV-0001": gt1, "DEV-0002": gt2})

    ans = batch_report["answerable_tickets"]
    assert ans["total"] == 1
    assert ans["precision_at_4"] == 0.25
    assert ans["recall_at_4"] == 1.0
    assert ans["hit_rate_at_4"] == 1.0

    unans = batch_report["unanswerable_tickets"]
    assert unans["total"] == 1
    assert unans["empty_count"] == 1
    assert unans["empty_retrieval_rate"] == 1.0
    assert unans["non_empty_retrieval_rate"] == 0.0


# =============================================================================
# 4. Policy and Safety Compliance Tests
# =============================================================================


def test_policy_mandatory_escalation_intents() -> None:
    """Verify all 5 mandatory escalation intents require ESCALATE route."""
    for mandatory_intent in [
        IntentCategory.SECURITY_INCIDENT,
        IntentCategory.COMPLIANCE_REQUEST,
        IntentCategory.DATA_RESIDENCY,
        IntentCategory.FEATURE_REQUEST,
        IntentCategory.UNCLEAR_REQUEST,
    ]:
        res = _create_mock_result(
            ticket_id="DEV-0001",
            intent=mandatory_intent,
            route=RouteOutcome.ESCALATE,
            effective_route=RouteOutcome.ESCALATE,
            delivery_status="escalated",
        )
        gt = EvaluationRecord(
            ticket=_create_mock_ticket("DEV-0001"),
            labels=TicketLabels(
                intent=mandatory_intent.value,
                urgency="high",
                expected_route="escalate",
                must_not_auto_respond=True,
            ),
        )

        rep = evaluate_policy_and_safety([res], {"DEV-0001": gt})
        assert rep.deterministic_execution_compliance_rate == 1.0
        assert rep.end_to_end_safety_compliance_rate is None
        assert rep.release_safety_exercised is False
        assert rep.safety_violations_count == 0


def test_safety_violation_only_on_customer_release() -> None:
    """Verify unreleased internal drafts are NOT safety violations, while released ones are."""
    gt = EvaluationRecord(
        ticket=_create_mock_ticket("DEV-0001"),
        labels=TicketLabels(
            intent="security_incident",
            urgency="high",
            expected_route="escalate",
            must_not_auto_respond=True,
        ),
    )

    # Case A: Candidate response generated internally, but held for review (unreleased)
    res_held = _create_mock_result(
        ticket_id="DEV-0001",
        intent=IntentCategory.SECURITY_INCIDENT,
        route=RouteOutcome.DRAFT_FOR_REVIEW,
        effective_route=RouteOutcome.DRAFT_FOR_REVIEW,
        delivery_status="held_for_review",
        has_customer_response=True,  # Internal draft exists
    )
    rep_held = evaluate_policy_and_safety([res_held], {"DEV-0001": gt})
    # Not released -> 0 safety violations, release-safety not exercised (None)
    assert rep_held.safety_violations_count == 0
    assert rep_held.total_released_responses == 0
    assert rep_held.end_to_end_safety_compliance_rate is None
    assert rep_held.release_safety_exercised is False

    # Case B: Actually released to customer -> Critical safety violation!
    res_released = _create_mock_result(
        ticket_id="DEV-0001",
        intent=IntentCategory.SECURITY_INCIDENT,
        route=RouteOutcome.AUTO_RESPOND,
        effective_route=RouteOutcome.AUTO_RESPOND,
        delivery_status="released",
        has_customer_response=True,
    )
    rep_released = evaluate_policy_and_safety([res_released], {"DEV-0001": gt})
    assert rep_released.safety_violations_count == 1
    assert rep_released.total_released_responses == 1
    assert rep_released.end_to_end_safety_compliance_rate == 0.0
    assert rep_released.release_safety_exercised is True


def test_all_13_mandatory_policy_restrictions_independent_safety() -> None:
    """Verify each of the 13 approved policy restrictions is independently audited."""
    # (name, result_kwargs, gt_labels_kwargs, expected_violation_keyword)
    test_cases = [
        (
            "must_not_auto_respond",
            {"intent": IntentCategory.API_USAGE_QUESTION},
            {"intent": "api_usage_question", "must_not_auto_respond": True},
            "must_not_auto_respond",
        ),
        (
            "security_incident",
            {"intent": IntentCategory.SECURITY_INCIDENT},
            {"intent": "security_incident"},
            "security_incident",
        ),
        (
            "compliance_request",
            {"intent": IntentCategory.COMPLIANCE_REQUEST},
            {"intent": "compliance_request"},
            "compliance_request",
        ),
        (
            "data_residency",
            {"intent": IntentCategory.DATA_RESIDENCY},
            {"intent": "data_residency"},
            "data_residency",
        ),
        (
            "feature_request",
            {"intent": IntentCategory.FEATURE_REQUEST},
            {"intent": "feature_request"},
            "feature_request",
        ),
        (
            "unclear_or_fallback",
            {"intent": IntentCategory.UNCLEAR_REQUEST, "is_fallback": True},
            {"intent": "unclear_request"},
            "unclear_or_fallback",
        ),
        (
            "billing_dispute",
            {
                "intent": IntentCategory.BILLING_QUERY,
                "billing_subtype": BillingRiskSubtype.DISPUTE_OR_FINANCIAL_DECISION,
            },
            {"intent": "billing_query"},
            "billing_dispute",
        ),
        (
            "billing_account_specific",
            {
                "intent": IntentCategory.BILLING_QUERY,
                "billing_subtype": BillingRiskSubtype.ACCOUNT_SPECIFIC,
            },
            {"intent": "billing_query"},
            "account_specific_or_ambiguous_billing",
        ),
        (
            "failed_guardrail",
            {"intent": IntentCategory.API_USAGE_QUESTION, "guardrails_passed": False},
            {"intent": "api_usage_question"},
            "guardrail",
        ),
        (
            "failed_audit_persistence",
            {"intent": IntentCategory.API_USAGE_QUESTION, "persistence_status": "failed"},
            {"intent": "api_usage_question"},
            "persisted_audit",
        ),
        (
            "kill_switch_active",
            {"intent": IntentCategory.API_USAGE_QUESTION, "kill_switch_active": True},
            {"intent": "api_usage_question"},
            "kill_switch",
        ),
        (
            "missing_or_weak_kb_evidence",
            {"intent": IntentCategory.API_USAGE_QUESTION, "passages": []},
            {"intent": "api_usage_question"},
            "missing_or_weak_kb_evidence",
        ),
        (
            "invalid_or_missing_citations",
            {
                "intent": IntentCategory.API_USAGE_QUESTION,
                "customer_response_text": "Refer to hallucinated citation [DOC-NONEXISTENT-999].",
                "citations_verified": False,
                "passages": [
                    RetrievedPassage(
                        chunk_id="DOC-API-001#c0",
                        doc_id="DOC-API-001",
                        category="api",
                        title="API Guide",
                        section_name="c0",
                        content="Content",
                        similarity_score=0.90,
                        rank=1,
                        citation_id="[DOC-API-001: c0]",
                    )
                ],
            },
            {"intent": "api_usage_question"},
            "invalid_or_missing_citations",
        ),
    ]

    for name, r_kwargs, gt_kwargs, reason_kw in test_cases:
        ticket = _create_mock_ticket(ticket_id="DEV-0001")
        labels = TicketLabels(urgency="medium", expected_route="auto_respond", **gt_kwargs)
        gt = EvaluationRecord(ticket=ticket, labels=labels)

        # 1. Released to customer -> MUST trigger safety violation!
        init_r_kwargs = {k: v for k, v in r_kwargs.items() if k != "persistence_status"}
        res_released = _create_mock_result(
            ticket_id="DEV-0001",
            route=RouteOutcome.AUTO_RESPOND,
            effective_route=RouteOutcome.AUTO_RESPOND,
            delivery_status="released",
            persistence_status="persisted",
            has_customer_response=True,
            **init_r_kwargs,
        )
        if name == "failed_audit_persistence":
            object.__setattr__(res_released, "persistence_status", "failed")

        rep_released = evaluate_policy_and_safety(
            results=[res_released],
            ground_truth={"DEV-0001": gt},
            kb_doc_ids={"DOC-API-001"},
        )
        assert rep_released.ticket_audits[0].customer_response_released is True
        assert rep_released.total_released_responses == 1
        assert rep_released.release_safety_exercised is True
        assert rep_released.end_to_end_safety_compliance_rate == 0.0
        assert rep_released.safety_violations_count >= 1, (
            f"Expected safety violation for {name} when released."
        )
        reasons_text = " ".join(rep_released.safety_violations[0]["reasons"])
        assert reason_kw in reasons_text, (
            f"Expected keyword '{reason_kw}' in reasons for {name}: {reasons_text}"
        )

        # 2. Internal candidate draft exists, but held for review -> NEVER a customer violation!
        res_held = _create_mock_result(
            ticket_id="DEV-0001",
            route=RouteOutcome.DRAFT_FOR_REVIEW,
            effective_route=RouteOutcome.DRAFT_FOR_REVIEW,
            delivery_status="held_for_review",
            has_customer_response=True,  # Internal draft exists
            **r_kwargs,
        )
        rep_held = evaluate_policy_and_safety(
            results=[res_held],
            ground_truth={"DEV-0001": gt},
            kb_doc_ids={"DOC-API-001"},
        )
        assert rep_held.total_released_responses == 0
        assert rep_held.end_to_end_safety_compliance_rate is None
        assert rep_held.release_safety_exercised is False
        assert rep_held.safety_violations_count == 0, (
            f"Unreleased internal draft for {name} must NOT be a safety violation."
        )


def test_zero_customer_releases_reports_not_meaningfully_exercised() -> None:
    """Verify when 0 responses are released, release-safety is marked not meaningfully exercised."""
    res = _create_mock_result(
        ticket_id="DEV-0001",
        intent=IntentCategory.UNCLEAR_REQUEST,
        route=RouteOutcome.ESCALATE,
        delivery_status="escalated",
        has_customer_response=False,
    )
    gt = EvaluationRecord(
        ticket=_create_mock_ticket("DEV-0001"),
        labels=TicketLabels(intent="unclear_request", urgency="high", expected_route="escalate"),
    )
    rep = evaluate_policy_and_safety([res], {"DEV-0001": gt})
    assert rep.total_released_responses == 0
    assert rep.unsafe_releases_count == 0
    assert rep.end_to_end_safety_compliance_rate is None
    assert rep.release_safety_exercised is False
    assert rep.release_safety_status == "not_meaningfully_exercised_zero_releases"
    assert "not meaningfully exercised" in rep.release_safety_notes.lower()


def test_safe_customer_release_reports_100_percent_safety() -> None:
    """Verify released responses that satisfy all policies yield 1.0 (100%) safety rate."""
    gt1 = EvaluationRecord(
        ticket=_create_mock_ticket("DEV-0001"),
        labels=TicketLabels(
            intent="api_usage_question",
            urgency="low",
            expected_route="auto_respond",
            must_not_auto_respond=False,
        ),
    )
    res1 = _create_mock_result(
        ticket_id="DEV-0001",
        intent=IntentCategory.API_USAGE_QUESTION,
        route=RouteOutcome.AUTO_RESPOND,
        effective_route=RouteOutcome.AUTO_RESPOND,
        delivery_status="released",
        has_customer_response=True,
        customer_response_text="Use endpoint /v1/api [DOC-API-001: Guide].",
        citations_verified=True,
    )
    rep = evaluate_policy_and_safety([res1], {"DEV-0001": gt1}, kb_doc_ids={"DOC-API-001"})
    assert rep.total_released_responses == 1
    assert rep.unsafe_releases_count == 0
    assert rep.end_to_end_safety_compliance_rate == 1.0
    assert rep.release_safety_exercised is True
    assert rep.release_safety_status == "exercised"


def test_feature_request_cannot_route_to_auto_respond_or_draft_for_review() -> None:
    """Feature and roadmap requests must strictly route to ESCALATE in policy evaluator."""
    from cloudserve.routing.router import DeterministicRouter

    router = DeterministicRouter()
    ticket = _create_mock_ticket("DEV-FEAT-001")
    classification = ClassificationResult(
        ticket_id=ticket.ticket_id,
        intent=IntentCategory.FEATURE_REQUEST,
        urgency=UrgencyLevel.LOW,
        confidence=0.98,
        intent_confidence=0.99,
        urgency_confidence=0.95,
        rationale="Customer wants multi-region active-active database clustering.",
        model_name="mock-model",
        prompt_version="1.0.0",
        is_fallback=False,
    )
    # 1. DeterministicRouter check under ideal conditions (passages, guardrail, citations)
    passages = [
        RetrievedPassage(
            doc_id="DOC-FEAT-001",
            chunk_id="chunk_001",
            category="roadmap",
            title="Roadmap Policy",
            content="Roadmap items require PM approval.",
            similarity_score=0.95,
            citation_id="[DOC-FEAT-001: Roadmap]",
            section_name="Roadmap",
            rank=1,
        )
    ]
    guardrail = GuardrailCheckResult(is_safe=True, rationale="Safe")
    decision = router.route_ticket(
        ticket=ticket,
        classification=classification,
        retrieved_passages=passages,
        guardrail_result=guardrail,
        cited_ids=["[DOC-FEAT-001: Roadmap]"],
    )
    # Must NEVER be AUTO_RESPOND or DRAFT_FOR_REVIEW
    assert decision.route == RouteOutcome.ESCALATE
    assert decision.route != RouteOutcome.AUTO_RESPOND
    assert decision.route != RouteOutcome.DRAFT_FOR_REVIEW
    assert decision.reason_code == RoutingReasonCode.MANDATORY_FEATURE_REQUEST

    # 2. Policy evaluator: verify route == DRAFT_FOR_REVIEW violates router compliance
    gt = EvaluationRecord(
        ticket=ticket,
        labels=TicketLabels(
            intent="feature_request",
            urgency="low",
            expected_route="escalate",
            must_not_auto_respond=True,
        ),
    )
    res_draft = _create_mock_result(
        ticket_id="DEV-FEAT-001",
        intent=IntentCategory.FEATURE_REQUEST,
        route=RouteOutcome.DRAFT_FOR_REVIEW,
        effective_route=RouteOutcome.DRAFT_FOR_REVIEW,
        delivery_status="held_for_review",
    )
    rep_draft = evaluate_policy_and_safety([res_draft], {"DEV-FEAT-001": gt})
    # Router compliance must fail because feature requests must ESCALATE, not DRAFT_FOR_REVIEW
    assert rep_draft.deterministic_execution_compliance_rate == 0.0
    assert rep_draft.ticket_audits[0].router_execution_compliant is False

    # 3. Policy evaluator: verify route == AUTO_RESPOND released causes safety violation
    res_released = _create_mock_result(
        ticket_id="DEV-FEAT-001",
        intent=IntentCategory.FEATURE_REQUEST,
        route=RouteOutcome.AUTO_RESPOND,
        effective_route=RouteOutcome.AUTO_RESPOND,
        delivery_status="released",
        has_customer_response=True,
    )
    rep_released = evaluate_policy_and_safety([res_released], {"DEV-FEAT-001": gt})
    assert rep_released.deterministic_execution_compliance_rate == 0.0
    assert rep_released.safety_violations_count == 1
    assert "feature_request_released" in rep_released.safety_violations[0]["reasons"]

    # 4. Policy evaluator: verify route == ESCALATE is fully compliant
    res_esc = _create_mock_result(
        ticket_id="DEV-FEAT-001",
        intent=IntentCategory.FEATURE_REQUEST,
        route=RouteOutcome.ESCALATE,
        effective_route=RouteOutcome.ESCALATE,
        delivery_status="escalated",
        has_customer_response=False,
    )
    rep_esc = evaluate_policy_and_safety([res_esc], {"DEV-FEAT-001": gt})
    assert rep_esc.deterministic_execution_compliance_rate == 1.0
    assert rep_esc.mandatory_escalation_compliance_rate == 1.0
    assert rep_esc.safety_violations_count == 0


# =============================================================================
# 5. Human Review Sampling and Rubric Tests
# =============================================================================


def test_human_review_sampling_strata_and_backfill() -> None:
    """Verify 25/15/10 targets, customer-facing candidate requirement, and backfill."""
    # Create 30 auto-respond candidates, 10 draft candidates, 0 escalate candidates
    results: list[ProcessedTicketResult] = []
    for i in range(30):
        results.append(_create_mock_result(ticket_id=f"AUTO-{i}", route=RouteOutcome.AUTO_RESPOND))
    for i in range(10):
        results.append(
            _create_mock_result(
                ticket_id=f"DRAFT-{i}",
                route=RouteOutcome.DRAFT_FOR_REVIEW,
                effective_route=RouteOutcome.DRAFT_FOR_REVIEW,
                delivery_status="held_for_review",
            )
        )

    items, meta = sample_human_review_packet(
        results, target_auto=25, target_review=15, target_escalate=10
    )

    # Total candidates available: 40
    assert meta.candidate_pool_size == 40
    # Target was 50, so actual sample size is 40 and shortfall is 10
    assert meta.actual_sample_size == 40
    assert meta.overall_shortfall == 10
    assert len(items) == 40

    # Verify template CSV generation
    csv_text = generate_human_review_template_csv(items)
    assert "review_id,ticket_id" in csv_text
    assert "r1_factual_correctness_1_to_5" in csv_text
    assert "r1_contains_hallucination_true_false" in csv_text
    assert "r2_factual_correctness_1_to_5" in csv_text


def test_pii_redaction_best_effort() -> None:
    """Verify best-effort scrubbing of email, phone, SSN, card, secret, customer ID, IP, UUID."""
    raw = (
        "Contact alice@example.com or call +1 555-123-4567. "
        "SSN: 000-12-3456, Card: 4111-2222-3333-4444. "
        "Secret: gsk_secrettoken12345678, Account: CUST-999. "
        "Server: 192.168.1.1, Request ID: a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d."
    )
    scrubbed = redact_pii_best_effort(raw)
    assert "[REDACTED_EMAIL]" in scrubbed
    assert "[REDACTED_PHONE]" in scrubbed
    assert "[REDACTED_SSN]" in scrubbed
    assert "[REDACTED_CARD]" in scrubbed
    assert "[REDACTED_SECRET]" in scrubbed
    assert "[REDACTED_CUSTOMER_ID]" in scrubbed
    assert "[REDACTED_IP]" in scrubbed
    assert "[REDACTED_UUID]" in scrubbed
    assert "alice@example.com" not in scrubbed
    assert "192.168.1.1" not in scrubbed
    assert "a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d" not in scrubbed


def test_calculate_linear_weighted_kappa() -> None:
    """Verify linear weighted Cohen's Kappa for ordinal 1-5 ratings."""
    import pytest

    # 1. Perfect agreement
    r1 = [1, 2, 3, 4, 5]
    r2 = [1, 2, 3, 4, 5]
    assert calculate_weighted_kappa(r1, r2) == 1.0

    # 2. Complete disagreement
    r1 = [1, 1, 1]
    r2 = [5, 5, 5]
    assert calculate_weighted_kappa(r1, r2) <= 0.0

    # 3. Partial agreement with linear weighting formula w_ij = |i - j| / (k - 1)
    r1 = [1, 2, 3, 4]
    r2 = [1, 3, 3, 2]
    kappa = calculate_weighted_kappa(r1, r2)
    assert -1.0 <= kappa <= 1.0
    assert isinstance(kappa, float)

    # 4. Zero reviewer scores raises ValueError (no synthetic scores allowed)
    with pytest.raises(ValueError, match="two actual reviewer score sets"):
        calculate_weighted_kappa([], [])


def test_calculate_binary_kappa_hallucination_flag() -> None:
    """Verify unweighted Cohen's Kappa for the binary hallucination flag."""
    import pytest

    # Perfect agreement
    assert calculate_binary_kappa([False, True], [False, True]) == 1.0

    # Disagreement
    assert calculate_binary_kappa([False, True], [True, False]) <= 0.0

    # Zero scores raises ValueError
    with pytest.raises(ValueError, match="two actual reviewer score sets"):
        calculate_binary_kappa([], [])


# =============================================================================
# 6. Artifact Emission and Data Minimization Tests
# =============================================================================


def test_generate_evaluation_artifacts_data_minimization(tmp_path: Path) -> None:
    """Verify artifact generation and strict data minimization in diagnostic exports."""
    ticket_id = "DEV-0001"
    raw_body = "Customer inquiry with PII: contact alice@example.com or call 555-123-4567."
    ticket = _create_mock_ticket(ticket_id=ticket_id, clean_body=raw_body)
    res = _create_mock_result(ticket_id=ticket_id, ticket=ticket)
    gt = EvaluationRecord(
        ticket=ticket,
        labels=TicketLabels(
            intent="api_usage_question",
            urgency="medium",
            expected_route="auto_respond",
            expected_doc_ids=["DOC-API-001"],
        ),
    )

    reconciliation = {
        "processing_run_id": "test-run",
        "total_processed": 1,
        "persisted_records": 1,
        "failed_or_missing_records": 0,
        "reconciled": True,
        "gap": 0,
        "status": "completed",
    }
    metadata = {
        "run_id": "test-run",
        "timestamp": "2026-11-19T10:00:00Z",
        "mode": "mock",
        "confidence_threshold_auto_respond": 0.80,
        "confidence_threshold_review": 0.60,
        "retrieval_similarity_threshold": 0.65,
    }

    summary = generate_evaluation_artifacts(
        run_id="test-run",
        metadata=metadata,
        results=[res],
        ground_truth={ticket_id: gt},
        reconciliation=reconciliation,
        output_dir=tmp_path,
        kb_doc_ids={"DOC-API-001"},
    )

    # 1. Check summary_metrics.json
    assert (tmp_path / "summary_metrics.json").exists()
    assert summary["total_records"] == 1
    assert summary["classification"]["intent_accuracy"] == 1.0

    # 2. Check per_ticket_results.jsonl (Strict Data Minimization: Zero raw bodies)
    jsonl_path = tmp_path / "per_ticket_results.jsonl"
    assert jsonl_path.exists()
    jsonl_content = jsonl_path.read_text(encoding="utf-8")
    assert raw_body not in jsonl_content
    assert "alice@example.com" not in jsonl_content
    assert "Test User" not in jsonl_content
    assert ticket_id in jsonl_content

    # 3. Check per_ticket_results.csv (Strict Data Minimization: Zero raw bodies)
    csv_path = tmp_path / "per_ticket_results.csv"
    assert csv_path.exists()
    csv_content = csv_path.read_text(encoding="utf-8")
    assert raw_body not in csv_content
    assert "alice@example.com" not in csv_content
    assert "Test User" not in csv_content

    # 4. Check human review artifacts (Best-effort PII scrubbing applied in packet & template)
    packet_path = tmp_path / "human_review_packet.json"
    template_path = tmp_path / "human_review_template.csv"
    assert packet_path.exists()
    assert template_path.exists()
    assert (tmp_path / "human_review_metadata.json").exists()

    packet_content = packet_path.read_text(encoding="utf-8")
    assert "[REDACTED_EMAIL]" in packet_content
    assert "alice@example.com" not in packet_content

    template_content = template_path.read_text(encoding="utf-8")
    assert "[REDACTED_EMAIL]" in template_content
    assert "alice@example.com" not in template_content

    # 5. Check EVALUATION_REPORT.md
    report_path = tmp_path / "EVALUATION_REPORT.md"
    assert report_path.exists()
    report_md = report_path.read_text(encoding="utf-8")
    assert "# CloudServe AI Support System — Unattended Evaluation Report" in report_md
    assert "Executive Summary & Governance Scorecard" in report_md


def test_policy_eval_non_billing_technical_ticket_not_flagged_unsafe() -> None:
    """Regression test: Non-billing technical tickets defaulting to AMBIGUOUS billing risk

    must NOT be falsely flagged as unsafe billing releases.
    """
    res = _create_mock_result(
        ticket_id="VAL-0001",
        intent=IntentCategory.API_KEY_ISSUE,
        route=RouteOutcome.AUTO_RESPOND,
        effective_route=RouteOutcome.AUTO_RESPOND,
        delivery_status="released",
        billing_subtype=BillingRiskSubtype.AMBIGUOUS,
        citations_verified=True,
    )
    gt = EvaluationRecord(
        ticket=_create_mock_ticket("VAL-0001"),
        labels=TicketLabels(
            intent="api_key_issue",
            urgency="medium",
            expected_route="auto_respond",
            answerable_from_docs=True,
            expected_doc_ids=["DOC-AUTH-004"],
        ),
    )

    report = evaluate_policy_and_safety([res], {"VAL-0001": gt})
    assert report.total_released_responses == 1
    assert report.safety_violations_count == 0
    assert report.end_to_end_safety_compliance_rate == 1.0
    assert report.safety_violations == []


def test_policy_eval_ambiguous_billing_ticket_correctly_flagged() -> None:
    """Regression test: An ambiguous billing response released to customer

    MUST still be correctly flagged as a safety violation.
    """
    res = _create_mock_result(
        ticket_id="VAL-0041",
        intent=IntentCategory.BILLING_QUERY,
        route=RouteOutcome.AUTO_RESPOND,
        effective_route=RouteOutcome.AUTO_RESPOND,
        delivery_status="released",
        billing_subtype=BillingRiskSubtype.AMBIGUOUS,
        citations_verified=True,
    )
    gt = EvaluationRecord(
        ticket=_create_mock_ticket("VAL-0041"),
        labels=TicketLabels(
            intent="billing_query",
            urgency="medium",
            expected_route="draft_for_review",
            answerable_from_docs=True,
            expected_doc_ids=["DOC-BILL-001"],
        ),
    )

    report = evaluate_policy_and_safety([res], {"VAL-0041": gt})
    assert report.total_released_responses == 1
    assert report.safety_violations_count == 1
    assert report.end_to_end_safety_compliance_rate == 0.0
    assert len(report.safety_violations) == 1
    reasons = report.safety_violations[0]["reasons"]
    assert "account_specific_or_ambiguous_billing_released" in reasons


def test_evaluation_reporter_live_mode_metadata(tmp_path: Path) -> None:
    """Regression test: A live evaluation run must report execution_mode as 'live',

    display 'LIVE' in EVALUATION_REPORT.md, and correctly report the configured
    model and embedding implementations instead of MockLLMClient.
    """
    ticket_id = "VAL-0001"
    gt = EvaluationRecord(
        ticket=_create_mock_ticket(ticket_id),
        labels=TicketLabels(
            intent="api_key_issue",
            urgency="medium",
            expected_route="auto_respond",
            answerable_from_docs=True,
            expected_doc_ids=["DOC-AUTH-004"],
        ),
    )
    res = _create_mock_result(
        ticket_id=ticket_id,
        intent=IntentCategory.API_KEY_ISSUE,
        route=RouteOutcome.AUTO_RESPOND,
        effective_route=RouteOutcome.AUTO_RESPOND,
        delivery_status="released",
        citations_verified=True,
    )

    live_metadata = {
        "run_id": "live-test-run-1234",
        "timestamp": "2026-11-20T19:04:05Z",
        "execution_mode": "live",
        "classifier_class": "TicketClassifier",
        "classifier_model": "openai/gpt-oss-120b",
        "classifier_prompt_version": "v1",
        "generator_class": "ResponseGenerator",
        "generator_model": "openai/gpt-oss-120b",
        "generator_prompt_version": "v1",
        "embedding_class": "SentenceTransformerEmbeddingAdapter",
        "embedding_model": "sentence_transformer_adapter",
        "retriever_similarity_threshold": 0.4,
        "confidence_threshold_auto_respond": 0.85,
        "confidence_threshold_review": 0.60,
    }

    reconciliation = {
        "processing_run_id": "live-test-run-1234",
        "total_processed": 1,
        "persisted_records": 1,
        "failed_or_missing_records": 0,
        "reconciled": True,
        "gap": 0,
        "status": "completed",
    }

    summary = generate_evaluation_artifacts(
        run_id="live-test-run-1234",
        metadata=live_metadata,
        results=[res],
        ground_truth={ticket_id: gt},
        reconciliation=reconciliation,
        output_dir=tmp_path,
        kb_doc_ids={"DOC-AUTH-004"},
    )

    # 1. summary_metrics.json has execution_mode == "live"
    assert summary["execution_mode"] == "live"

    # 2. EVALUATION_REPORT.md displays LIVE and accurate component details
    report_md = (tmp_path / "EVALUATION_REPORT.md").read_text(encoding="utf-8")
    assert "**Execution Mode:** `LIVE`" in report_md
    assert "- **LLM Classifier Provider:** `TicketClassifier (openai/gpt-oss-120b)`" in report_md
    expected_gen = "- **Response Generator Provider:** `ResponseGenerator (openai/gpt-oss-120b)`"
    assert expected_gen in report_md
    expected_emb = (
        "- **Embedding Provider:** `SentenceTransformerEmbeddingAdapter "
        "(sentence_transformer_adapter)`"
    )
    assert expected_emb in report_md
    # 3. Mock confidence diagnostic warning is NOT present
    assert "Mock Confidence Diagnostic" not in report_md
