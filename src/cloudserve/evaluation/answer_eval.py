"""Answer quality evaluation module for generated support responses.

Evaluates generated answers against human-authored reference responses according to
Acceptance Criteria A3.34:
1. Evaluates generated answers against reference records using explicit ticket_id mapping.
2. Measures must_mention coverage and must_not_claim violations separately.
3. Computes semantic similarity as a diagnostic metric without treating an uncalibrated
   cosine threshold as a definitive pass/fail barrier.
4. Evaluates citation validity and grounding independently.
5. Distinguishes correct human escalation, correct auto-resolution, pending/quota-paused,
   and genuinely incorrect outcomes; ensures correct escalations are not penalized.
6. Strictly isolated from runtime classification, retrieval, generation, and routing.
7. Keeps mock and live evaluation modes strictly separated.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from cloudserve.ingestion.models import EvaluationRecord
from cloudserve.routing.citations import extract_doc_id_from_citation
from cloudserve.routing.models import RouteOutcome
from cloudserve.service import ProcessedTicketResult

logger = logging.getLogger(__name__)

CITATION_REGEX = re.compile(r"\[([A-Za-z0-9_\-]+)(?::\s*[^\]]+)?\]")


class AnswerOutcomeCategory(str, Enum):
    """Evaluation categorization for answer outcomes."""

    CORRECT_AUTO_RESOLVE = "correct_auto_resolve"
    CORRECT_ESCALATION = "correct_escalation"
    CORRECT_DRAFT_REVIEW = "correct_draft_review"
    QUOTA_PAUSED = "quota_paused"
    INCORRECT_ROUTING = "incorrect_routing"
    FAILED_ANSWER = "failed_answer"
    UNANSWERABLE_NO_EVAL = "unanswerable_no_eval"


class ReferenceResponseRecord(BaseModel):
    """Pydantic model representing a reference response record."""

    model_config = ConfigDict(frozen=True)

    ticket_id: str
    intent: str
    expected_doc_ids: list[str] = Field(default_factory=list)
    reference_response: str
    must_mention: list[str] = Field(default_factory=list)
    must_not_claim: list[str] = Field(default_factory=list)
    written_by: str = ""


class TicketAnswerEvaluation(BaseModel):
    """Per-ticket answer quality and compliance evaluation record."""

    model_config = ConfigDict(frozen=True)

    ticket_id: str
    has_reference: bool = Field(
        ...,
        description="Whether this ticket has a matching reference response in the benchmark key.",
    )
    outcome_category: AnswerOutcomeCategory = Field(
        ...,
        description="Categorized evaluation outcome (e.g. correct_escalation).",
    )
    is_penalized_failure: bool = Field(
        ...,
        description="True if ticket is classified as a genuine answer or routing quality failure.",
    )
    failure_reasons: list[str] = Field(
        default_factory=list,
        description="Traceable failure reasons if is_penalized_failure is True.",
    )

    # Content quality metrics (applicable when an answer was generated and reference exists)
    must_mention_total: int = 0
    must_mention_covered: int = 0
    must_mention_coverage_ratio: float | None = None
    missing_mentions: list[str] = Field(default_factory=list)

    must_not_claim_total: int = 0
    must_not_claim_violated: int = 0
    violated_claims: list[str] = Field(default_factory=list)
    has_must_not_claim_violation: bool = False

    # Diagnostic semantic similarity
    diagnostic_semantic_similarity: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Diagnostic TF-IDF cosine similarity against human reference response.",
    )

    # Citation & grounding metrics
    has_citations: bool = False
    cited_doc_ids: list[str] = Field(default_factory=list)
    citations_valid: bool = True
    grounding_passed: bool = True
    hallucinated_citations: list[str] = Field(default_factory=list)

    # Route and delivery context
    actual_route: str
    expected_route: str | None = None
    customer_response_released: bool = False
    response_length_chars: int = 0


class AnswerQualityReport(BaseModel):
    """Aggregated answer quality evaluation report."""

    model_config = ConfigDict(frozen=True)

    total_tickets: int
    reference_mapped_tickets: int = Field(
        ...,
        description="Number of tickets evaluated with an explicit reference response mapping.",
    )
    reference_coverage_pct: float = Field(
        ...,
        description="Percentage of total processed tickets that have reference answers.",
    )

    # Outcome breakdown
    correct_auto_resolve_count: int = 0
    correct_escalation_count: int = 0
    correct_draft_review_count: int = 0
    quota_paused_count: int = 0
    incorrect_routing_count: int = 0
    failed_answer_count: int = 0
    unanswerable_no_eval_count: int = 0

    # Answer Quality Metrics (computed strictly over evaluated candidates with references)
    evaluated_answer_count: int = Field(
        ...,
        description="Number of tickets evaluated against a reference response.",
    )
    must_mention_evaluated_tickets: int = Field(
        ...,
        description="Number of evaluated tickets with >= 1 required must_mention phrases.",
    )
    mean_must_mention_coverage: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Mean must_mention coverage ratio across tickets with required mentions.",
    )
    must_not_claim_clean_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Proportion of evaluated tickets completely free of must_not_claim violations.",
    )
    total_claim_violations_count: int = 0

    # Diagnostic Similarity Summary
    mean_diagnostic_similarity: float | None = Field(
        default=None,
        description="Average diagnostic cosine similarity against reference responses.",
    )
    median_diagnostic_similarity: float | None = None

    # Independent Citation & Grounding
    citation_validity_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Proportion of cited responses whose citations are corpus valid.",
    )
    grounding_pass_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Proportion of generated responses passing grounding validation.",
    )

    # Overall Quality Assessment
    overall_answer_pass_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Proportion of reference-evaluated tickets achieving non-penalized status.",
    )
    is_mock: bool = Field(
        ...,
        description="Whether this evaluation was conducted in mock mode.",
    )
    ticket_evaluations: list[TicketAnswerEvaluation] = Field(default_factory=list)


def normalize_for_matching(text: str) -> str:
    """Normalize text for substring and token phrase matching."""
    return re.sub(r"\s+", " ", text.lower().strip())


def check_must_mention(
    generated_text: str,
    required_mentions: Sequence[str],
) -> tuple[int, int, float, list[str]]:
    """Measure must_mention coverage.

    Args:
        generated_text: Text generated by the model.
        required_mentions: Phrases that must be covered.

    Returns:
        tuple of (total_required, covered_count, coverage_ratio, missing_mentions).
    """
    if not required_mentions:
        return 0, 0, 1.0, []

    norm_gen = normalize_for_matching(generated_text)
    covered = 0
    missing: list[str] = []

    for phrase in required_mentions:
        norm_phrase = normalize_for_matching(phrase)
        if norm_phrase in norm_gen:
            covered += 1
        else:
            missing.append(phrase)

    ratio = covered / len(required_mentions)
    return len(required_mentions), covered, ratio, missing


def check_must_not_claim(
    generated_text: str,
    forbidden_claims: Sequence[str],
) -> tuple[int, int, list[str]]:
    """Detect must_not_claim violations.

    Args:
        generated_text: Text generated by the model.
        forbidden_claims: Claims that must not be made.

    Returns:
        tuple of (total_forbidden, violation_count, violated_claims).
    """
    if not forbidden_claims:
        return 0, 0, []

    norm_gen = normalize_for_matching(generated_text)
    violated: list[str] = []

    for claim in forbidden_claims:
        norm_claim = normalize_for_matching(claim)
        if norm_claim in norm_gen:
            violated.append(claim)

    return len(forbidden_claims), len(violated), violated


def compute_diagnostic_similarity(
    candidate: str,
    reference: str,
) -> float:
    """Compute diagnostic TF-IDF unigram/bigram cosine similarity.

    NOTE: This is strictly an informational diagnostic metric. It must NEVER
    be used as a definitive pass/fail barrier.
    """
    if not candidate.strip() or not reference.strip():
        return 0.0

    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        tfidf = vectorizer.fit_transform([candidate, reference])
        sim = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
        return max(0.0, min(1.0, round(sim, 4)))
    except Exception as exc:
        logger.debug("Diagnostic similarity computation failed: %s", exc)
        return 0.0


def evaluate_ticket_answer(
    result: ProcessedTicketResult,
    reference: ReferenceResponseRecord | None,
    ground_truth: EvaluationRecord | None,
    kb_doc_ids: set[str] | None = None,
) -> TicketAnswerEvaluation:
    """Evaluate a single ticket's generated answer against ground truth and reference.

    Correct escalations, reviews, and quota pauses are distinguished and protected
    from being scored as failed answers.
    """
    tid = result.ticket.ticket_id
    actual_route = result.routing_decision.route.value
    expected_route = (
        ground_truth.labels.expected_route
        if (ground_truth and ground_truth.labels)
        else None
    )
    ans_from_docs = (
        ground_truth.labels.answerable_from_docs
        if (ground_truth and ground_truth.labels)
        else True
    )

    gen_text = (
        (result.generated_response.customer_response or "")
        if result.generated_response
        else (result.final_customer_response or "")
    )
    released = result.delivery_status == "released"
    has_ref = reference is not None

    failure_reasons: list[str] = []
    is_penalized = False

    # 1. Evaluate Quota / Process Pause
    if result.persistence_status == "paused_quota" or "quota" in (result.error or "").lower():
        return TicketAnswerEvaluation(
            ticket_id=tid,
            has_reference=has_ref,
            outcome_category=AnswerOutcomeCategory.QUOTA_PAUSED,
            is_penalized_failure=False,
            actual_route=actual_route,
            expected_route=expected_route,
            customer_response_released=False,
            response_length_chars=len(gen_text),
        )

    # 2. Evaluate Citation and Grounding Independently
    cited_doc_ids: list[str] = []
    citations_valid = True
    grounding_passed = True
    hallucinated: list[str] = []

    if gen_text:
        raw_citations = CITATION_REGEX.findall(gen_text)
        for raw_c in raw_citations:
            doc_id = extract_doc_id_from_citation(f"[{raw_c}]")
            if doc_id:
                cited_doc_ids.append(doc_id)

        # Check citations against valid KB doc IDs if provided
        if kb_doc_ids is not None:
            for c_id in cited_doc_ids:
                if c_id not in kb_doc_ids:
                    citations_valid = False
                    hallucinated.append(c_id)

        # Check grounding result if present
        if result.grounding_result is not None:
            grounding_passed = result.grounding_result.is_grounded
        elif result.routing_decision.reason_code.value == "GROUNDING_VALIDATION_FAILED":
            grounding_passed = False

    # 3. Evaluate Outcome Category and Protection of Correct Escalations
    # A. Mandatory or Legitimate Escalation
    if actual_route == RouteOutcome.ESCALATE.value:
        if expected_route == RouteOutcome.ESCALATE.value or not ans_from_docs:
            category = AnswerOutcomeCategory.CORRECT_ESCALATION
            is_penalized = False
        elif result.routing_decision.reason_code.value in {
            "MANDATORY_SECURITY_INCIDENT",
            "MANDATORY_COMPLIANCE_REQUEST",
            "MANDATORY_DATA_RESIDENCY",
            "MANDATORY_FEATURE_REQUEST",
            "MANDATORY_UNCLEAR_REQUEST",
            "MANDATORY_CLASSIFIER_FALLBACK",
            "MANDATORY_BILLING_DISPUTE",
            "HIGH_URGENCY_NO_KB_EVIDENCE",
            "HIGH_URGENCY_PRODUCTION_INCIDENT",
            "GUARDRAIL_VIOLATION",
            "GUARDRAIL_UNAVAILABLE",
        }:
            category = AnswerOutcomeCategory.CORRECT_ESCALATION
            is_penalized = False
        else:
            category = AnswerOutcomeCategory.INCORRECT_ROUTING
            is_penalized = True
            reason_code_val = result.routing_decision.reason_code.value
            failure_reasons.append(
                f"Escalated unexpectedly (expected {expected_route}, reason={reason_code_val})"
            )

        return TicketAnswerEvaluation(
            ticket_id=tid,
            has_reference=has_ref,
            outcome_category=category,
            is_penalized_failure=is_penalized,
            failure_reasons=failure_reasons,
            actual_route=actual_route,
            expected_route=expected_route,
            customer_response_released=released,
            response_length_chars=len(gen_text),
            has_citations=bool(cited_doc_ids),
            cited_doc_ids=cited_doc_ids,
            citations_valid=citations_valid,
            grounding_passed=grounding_passed,
            hallucinated_citations=hallucinated,
        )

    # B. Draft For Review (Gate Failures / Low Confidence)
    if actual_route == RouteOutcome.DRAFT_FOR_REVIEW.value:
        if expected_route == RouteOutcome.DRAFT_FOR_REVIEW.value:
            category = AnswerOutcomeCategory.CORRECT_DRAFT_REVIEW
            is_penalized = False
        elif not grounding_passed:
            # Correctly held internal draft due to grounding check
            category = AnswerOutcomeCategory.CORRECT_DRAFT_REVIEW
            is_penalized = False
        else:
            category = AnswerOutcomeCategory.CORRECT_DRAFT_REVIEW
            is_penalized = False

    # C. Auto-Respond (Customer Released)
    elif actual_route == RouteOutcome.AUTO_RESPOND.value:
        if expected_route == RouteOutcome.ESCALATE.value:
            category = AnswerOutcomeCategory.INCORRECT_ROUTING
            is_penalized = True
            failure_reasons.append(
                f"Auto-responded when expected route was mandatory ESCALATE ({expected_route})"
            )
        else:
            category = AnswerOutcomeCategory.CORRECT_AUTO_RESOLVE
    else:
        category = AnswerOutcomeCategory.UNANSWERABLE_NO_EVAL

    # 4. Content Quality Verification (must_mention & must_not_claim)
    mention_tot = 0
    mention_cov = 0
    mention_ratio: float | None = None
    missing_m: list[str] = []

    claim_tot = 0
    claim_viol = 0
    viol_c: list[str] = []
    has_viol = False
    diag_sim: float | None = None

    if has_ref and reference:
        # Check forbidden claims
        claim_tot, claim_viol, viol_c = check_must_not_claim(
            gen_text, reference.must_not_claim
        )
        if claim_viol > 0:
            has_viol = True
            is_penalized = True
            category = AnswerOutcomeCategory.FAILED_ANSWER
            failure_reasons.append(
                f"Violated must_not_claim ({len(viol_c)} claims): {', '.join(viol_c)}"
            )

        # Check required mentions
        if reference.must_mention:
            mention_tot, mention_cov, mention_ratio, missing_m = check_must_mention(
                gen_text, reference.must_mention
            )
            # If answer was auto-responded but completely missed required mentions (< 0.5)
            if actual_route == RouteOutcome.AUTO_RESPOND.value and mention_ratio < 0.5:
                is_penalized = True
                category = AnswerOutcomeCategory.FAILED_ANSWER
                failure_reasons.append(
                    f"Low must_mention coverage ({mention_cov}/{mention_tot}): missing {missing_m}"
                )

        # Compute diagnostic similarity (informative only)
        diag_sim = compute_diagnostic_similarity(
            gen_text, reference.reference_response
        )

    # 5. Citation Validity Enforcement on Released Responses
    if released and not citations_valid:
        is_penalized = True
        category = AnswerOutcomeCategory.FAILED_ANSWER
        failure_reasons.append(
            f"Released response contained invalid/hallucinated citations: {hallucinated}"
        )

    return TicketAnswerEvaluation(
        ticket_id=tid,
        has_reference=has_ref,
        outcome_category=category,
        is_penalized_failure=is_penalized,
        failure_reasons=failure_reasons,
        must_mention_total=mention_tot,
        must_mention_covered=mention_cov,
        must_mention_coverage_ratio=mention_ratio,
        missing_mentions=missing_m,
        must_not_claim_total=claim_tot,
        must_not_claim_violated=claim_viol,
        violated_claims=viol_c,
        has_must_not_claim_violation=has_viol,
        diagnostic_semantic_similarity=diag_sim,
        has_citations=bool(cited_doc_ids),
        cited_doc_ids=cited_doc_ids,
        citations_valid=citations_valid,
        grounding_passed=grounding_passed,
        hallucinated_citations=hallucinated,
        actual_route=actual_route,
        expected_route=expected_route,
        customer_response_released=released,
        response_length_chars=len(gen_text),
    )


def evaluate_answers_batch(
    results: Sequence[ProcessedTicketResult],
    reference_responses: dict[str, ReferenceResponseRecord] | Sequence[dict[str, Any]],
    ground_truth: dict[str, EvaluationRecord] | None = None,
    kb_doc_ids: set[str] | None = None,
    is_mock: bool = False,
) -> AnswerQualityReport:
    """Evaluate batch of processed tickets against reference response benchmarks.

    Args:
        results: Sequence of ProcessedTicketResult records from evaluation run.
        reference_responses: Dict mapping ticket_id to ReferenceResponseRecord or list of dicts.
        ground_truth: Optional mapping of ticket_id to EvaluationRecord.
        kb_doc_ids: Optional set of known doc IDs for citation verification.
        is_mock: Whether the evaluation was executed in mock mode.

    Returns:
        AnswerQualityReport with per-ticket evaluations and macro quality indicators.
    """
    # 1. Normalize reference responses lookup
    ref_map: dict[str, ReferenceResponseRecord] = {}
    if isinstance(reference_responses, dict):
        for k, v in reference_responses.items():
            if isinstance(v, ReferenceResponseRecord):
                ref_map[k] = v
            elif isinstance(v, dict):
                ref_map[k] = ReferenceResponseRecord.model_validate(v)
    elif isinstance(reference_responses, Sequence):
        for item in reference_responses:
            if isinstance(item, ReferenceResponseRecord):
                ref_map[item.ticket_id] = item
            elif isinstance(item, dict) and "ticket_id" in item:
                ref_map[item["ticket_id"]] = ReferenceResponseRecord.model_validate(item)

    # 2. Evaluate individual tickets
    evaluations: list[TicketAnswerEvaluation] = []
    gt_map = ground_truth or {}

    for res in results:
        tid = res.ticket.ticket_id
        ref = ref_map.get(tid)
        gt = gt_map.get(tid)
        ticket_eval = evaluate_ticket_answer(
            result=res,
            reference=ref,
            ground_truth=gt,
            kb_doc_ids=kb_doc_ids,
        )
        evaluations.append(ticket_eval)

    # 3. Aggregate metrics
    total_tickets = len(evaluations)
    ref_mapped_count = sum(1 for e in evaluations if e.has_reference)
    ref_cov_pct = (
        round((ref_mapped_count / total_tickets * 100.0), 2) if total_tickets > 0 else 0.0
    )

    # Categorization counts
    cat_counts: dict[AnswerOutcomeCategory, int] = {cat: 0 for cat in AnswerOutcomeCategory}
    for e in evaluations:
        cat_counts[e.outcome_category] += 1

    # Answer Quality Metrics strictly over tickets with references
    ref_evals = [e for e in evaluations if e.has_reference]
    evaluated_answer_count = len(ref_evals)

    mention_evals = [e for e in ref_evals if e.must_mention_total > 0]
    mention_ratios = [
        e.must_mention_coverage_ratio
        for e in mention_evals
        if e.must_mention_coverage_ratio is not None
    ]
    mean_mention_cov = (
        round(sum(mention_ratios) / len(mention_ratios), 4) if mention_ratios else 1.0
    )

    clean_claims_count = sum(1 for e in ref_evals if not e.has_must_not_claim_violation)
    claim_clean_rate = (
        round(clean_claims_count / len(ref_evals), 4) if ref_evals else 1.0
    )
    total_violations = sum(e.must_not_claim_violated for e in ref_evals)

    # Diagnostic similarities
    sim_scores = [
        e.diagnostic_semantic_similarity
        for e in ref_evals
        if e.diagnostic_semantic_similarity is not None
    ]
    mean_sim = round(sum(sim_scores) / len(sim_scores), 4) if sim_scores else None
    med_sim: float | None = None
    if sim_scores:
        sorted_sim = sorted(sim_scores)
        n = len(sorted_sim)
        med_sim = (
            sorted_sim[n // 2]
            if n % 2 == 1
            else round((sorted_sim[n // 2 - 1] + sorted_sim[n // 2]) / 2.0, 4)
        )

    # Citation & Grounding
    cited_responses = [e for e in evaluations if e.has_citations]
    valid_citations_count = sum(1 for e in cited_responses if e.citations_valid)
    cit_valid_rate = (
        round(valid_citations_count / len(cited_responses), 4) if cited_responses else 1.0
    )

    generated_responses = [e for e in evaluations if e.response_length_chars > 0]
    grounding_passed_count = sum(1 for e in generated_responses if e.grounding_passed)
    ground_pass_rate = (
        round(grounding_passed_count / len(generated_responses), 4)
        if generated_responses
        else 1.0
    )

    # Overall non-penalized pass rate over reference-mapped tickets
    non_penalized_count = sum(1 for e in ref_evals if not e.is_penalized_failure)
    overall_pass_rate = (
        round(non_penalized_count / len(ref_evals), 4) if ref_evals else 1.0
    )

    return AnswerQualityReport(
        total_tickets=total_tickets,
        reference_mapped_tickets=ref_mapped_count,
        reference_coverage_pct=ref_cov_pct,
        correct_auto_resolve_count=cat_counts[AnswerOutcomeCategory.CORRECT_AUTO_RESOLVE],
        correct_escalation_count=cat_counts[AnswerOutcomeCategory.CORRECT_ESCALATION],
        correct_draft_review_count=cat_counts[AnswerOutcomeCategory.CORRECT_DRAFT_REVIEW],
        quota_paused_count=cat_counts[AnswerOutcomeCategory.QUOTA_PAUSED],
        incorrect_routing_count=cat_counts[AnswerOutcomeCategory.INCORRECT_ROUTING],
        failed_answer_count=cat_counts[AnswerOutcomeCategory.FAILED_ANSWER],
        unanswerable_no_eval_count=cat_counts[AnswerOutcomeCategory.UNANSWERABLE_NO_EVAL],
        evaluated_answer_count=evaluated_answer_count,
        must_mention_evaluated_tickets=len(mention_evals),
        mean_must_mention_coverage=mean_mention_cov,
        must_not_claim_clean_rate=claim_clean_rate,
        total_claim_violations_count=total_violations,
        mean_diagnostic_similarity=mean_sim,
        median_diagnostic_similarity=med_sim,
        citation_validity_rate=cit_valid_rate,
        grounding_pass_rate=ground_pass_rate,
        overall_answer_pass_rate=overall_pass_rate,
        is_mock=is_mock,
        ticket_evaluations=evaluations,
    )
