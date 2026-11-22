"""Human-review packet generation, stratified sampling, PII redaction, and scoring rubric.

Implements:
- Best-effort regex PII redaction (emails, phones, cards, SSNs, secrets, customer IDs).
- Stratified sampling targeting 25 AUTO_RESPOND, 15 DRAFT_FOR_REVIEW, 10 ESCALATE responses.
- Sourced strictly from eligible customer-facing candidate responses.
- Strictly prevents substituting internal escalation notes for customer-facing responses.
- Bidirectional backfill across all three strata with transparent metadata logging.
- Blinded, unpopulated two-reviewer scoring template with 4 dimensions + hallucination flag.
- Linear-weighted Cohen's Kappa agreement metric calculator.
"""

import csv
import io
import logging
import re
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict

from cloudserve.routing.models import RouteOutcome
from cloudserve.service import ProcessedTicketResult

logger = logging.getLogger(__name__)

# Best-effort PII scrub patterns
RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
RE_PHONE = re.compile(r"(?:\+?1[-. ]?)?\(?[0-9]{3}\)?[-. ]?[0-9]{3}[-. ]?[0-9]{4}")
RE_CARD = re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b|\b\d{15,16}\b")
RE_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
RE_SECRETS = re.compile(r"\b(?:gsk_|sk-|key-|Bearer\s+)[a-zA-Z0-9_\-\.]{8,}\b")
RE_CUSTOMER_ID = re.compile(r"\bCUST-[A-Z0-9-]+\b")
RE_IP = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
RE_UUID = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)


def redact_pii_best_effort(text: str) -> str:
    """Apply best-effort regex scrubbing to redact potential PII and secrets.

    Supported patterns:
        - Email addresses ([REDACTED_EMAIL])
        - Phone numbers ([REDACTED_PHONE])
        - Payment card numbers ([REDACTED_CARD])
        - US Social Security Numbers ([REDACTED_SSN])
        - API keys and bearer tokens ([REDACTED_SECRET])
        - Customer identifiers ([REDACTED_CUSTOMER_ID])
        - IPv4 addresses ([REDACTED_IP])
        - UUIDs ([REDACTED_UUID])

    Note:
        This is a best-effort heuristic scrubber, not a guaranteed complete mathematical
        proof of zero PII.
    """
    if not text:
        return ""
    scrubbed = RE_EMAIL.sub("[REDACTED_EMAIL]", text)
    scrubbed = RE_PHONE.sub("[REDACTED_PHONE]", scrubbed)
    scrubbed = RE_CARD.sub("[REDACTED_CARD]", scrubbed)
    scrubbed = RE_SSN.sub("[REDACTED_SSN]", scrubbed)
    scrubbed = RE_SECRETS.sub("[REDACTED_SECRET]", scrubbed)
    scrubbed = RE_CUSTOMER_ID.sub("[REDACTED_CUSTOMER_ID]", scrubbed)
    scrubbed = RE_IP.sub("[REDACTED_IP]", scrubbed)
    scrubbed = RE_UUID.sub("[REDACTED_UUID]", scrubbed)
    return scrubbed


class HumanReviewSampleItem(BaseModel):
    """An individual ticket response selected for human qualitative review."""

    model_config = ConfigDict(frozen=True)

    review_id: str
    ticket_id: str
    channel: str
    customer_tier: str
    route_outcome: str
    redacted_inquiry: str
    candidate_customer_response: str
    retrieved_passages: list[dict[str, str]]
    cited_doc_ids: list[str]


class HumanReviewSamplingMetadata(BaseModel):
    """Metadata tracking sampling execution, stratification targets, backfill, and shortfalls."""

    model_config = ConfigDict(frozen=True)

    target_sample_size: int = 50
    actual_sample_size: int
    strata_targets: dict[str, int]
    strata_actuals: dict[str, int]
    backfill_details: list[dict[str, Any]]
    overall_shortfall: int
    candidate_pool_size: int
    notes: str


def sample_human_review_packet(
    results: Sequence[ProcessedTicketResult],
    target_auto: int = 25,
    target_review: int = 15,
    target_escalate: int = 10,
) -> tuple[list[HumanReviewSampleItem], HumanReviewSamplingMetadata]:
    """Sample up to 50 eligible candidate responses with documented backfill.

    Eligibility Rule:
        Candidate pool is strictly restricted to tickets where a customer-facing
        candidate response was synthesized (generated_response.customer_response is not None).
        Internal escalation notes (agent_note, private_agent_notes) are NEVER substituted.
    """
    # 1. Filter eligible tickets with valid, non-empty customer responses
    eligible_by_route: dict[RouteOutcome, list[ProcessedTicketResult]] = {
        RouteOutcome.AUTO_RESPOND: [],
        RouteOutcome.DRAFT_FOR_REVIEW: [],
        RouteOutcome.ESCALATE: [],
    }

    for res in results:
        # Check if customer-facing candidate response exists and is non-empty
        cand = res.generated_response.customer_response if res.generated_response else None
        if cand and cand.strip():
            route = res.routing_decision.route
            if route in eligible_by_route:
                eligible_by_route[route].append(res)

    total_eligible = sum(len(pool) for pool in eligible_by_route.values())

    # 2. Determine initial allocation from target pools
    sampled_by_route: dict[RouteOutcome, list[ProcessedTicketResult]] = {
        RouteOutcome.AUTO_RESPOND: list(eligible_by_route[RouteOutcome.AUTO_RESPOND][:target_auto]),
        RouteOutcome.DRAFT_FOR_REVIEW: list(
            eligible_by_route[RouteOutcome.DRAFT_FOR_REVIEW][:target_review]
        ),
        RouteOutcome.ESCALATE: list(eligible_by_route[RouteOutcome.ESCALATE][:target_escalate]),
    }

    remaining_pools: dict[RouteOutcome, list[ProcessedTicketResult]] = {
        RouteOutcome.AUTO_RESPOND: list(eligible_by_route[RouteOutcome.AUTO_RESPOND][target_auto:]),
        RouteOutcome.DRAFT_FOR_REVIEW: list(
            eligible_by_route[RouteOutcome.DRAFT_FOR_REVIEW][target_review:]
        ),
        RouteOutcome.ESCALATE: list(eligible_by_route[RouteOutcome.ESCALATE][target_escalate:]),
    }

    backfill_details: list[dict[str, Any]] = []

    # 3. Apply backfill for deficits in each stratum
    targets = {
        RouteOutcome.AUTO_RESPOND: target_auto,
        RouteOutcome.DRAFT_FOR_REVIEW: target_review,
        RouteOutcome.ESCALATE: target_escalate,
    }

    backfill_order = {
        RouteOutcome.AUTO_RESPOND: [RouteOutcome.DRAFT_FOR_REVIEW, RouteOutcome.ESCALATE],
        RouteOutcome.DRAFT_FOR_REVIEW: [RouteOutcome.AUTO_RESPOND, RouteOutcome.ESCALATE],
        RouteOutcome.ESCALATE: [RouteOutcome.DRAFT_FOR_REVIEW, RouteOutcome.AUTO_RESPOND],
    }

    for stratum, target_count in targets.items():
        current_count = len(sampled_by_route[stratum])
        deficit = target_count - current_count
        if deficit > 0:
            for donor in backfill_order[stratum]:
                donor_pool = remaining_pools[donor]
                available = len(donor_pool)
                to_take = min(deficit, available)
                if to_take > 0:
                    taken = donor_pool[:to_take]
                    sampled_by_route[stratum].extend(taken)
                    remaining_pools[donor] = donor_pool[to_take:]
                    deficit -= to_take
                    backfill_details.append(
                        {
                            "stratum_receiving": stratum.value,
                            "donor_stratum": donor.value,
                            "count": to_take,
                            "reason": f"Target was {target_count}, available was {current_count}",
                        }
                    )
                if deficit == 0:
                    break

    # 4. Flatten selected items in stable order
    all_sampled: list[ProcessedTicketResult] = []
    strata_actuals: dict[str, int] = {}
    for stratum in [
        RouteOutcome.AUTO_RESPOND,
        RouteOutcome.DRAFT_FOR_REVIEW,
        RouteOutcome.ESCALATE,
    ]:
        items = sampled_by_route[stratum]
        all_sampled.extend(items)
        strata_actuals[stratum.value] = len(items)

    target_total = target_auto + target_review + target_escalate
    actual_sample_size = len(all_sampled)
    overall_shortfall = max(0, target_total - actual_sample_size)

    # 5. Convert to HumanReviewSampleItem with best-effort PII scrubbing
    sample_items: list[HumanReviewSampleItem] = []
    for idx, res in enumerate(all_sampled, start=1):
        review_id = f"REV-{idx:03d}"
        ticket = res.ticket
        inquiry_text = ticket.clean_text
        redacted_inquiry = redact_pii_best_effort(inquiry_text)
        cand_resp = (
            res.generated_response.customer_response
            if res.generated_response and res.generated_response.customer_response
            else ""
        )
        redacted_cand_resp = redact_pii_best_effort(cand_resp)

        passages_summary: list[dict[str, str]] = []
        for p in res.retrieved_passages or []:
            passages_summary.append(
                {
                    "doc_id": p.doc_id,
                    "citation_id": p.citation_id,
                    "title": p.title,
                    "excerpt": redact_pii_best_effort(
                        p.content[:200] + "..." if len(p.content) > 200 else p.content
                    ),
                }
            )

        cited_docs: list[str] = []
        if res.generated_response and res.generated_response.citation_ids:
            cited_docs = list(res.generated_response.citation_ids)
        elif res.routing_decision and res.routing_decision.sources_used:
            cited_docs = list(res.routing_decision.sources_used)

        sample_items.append(
            HumanReviewSampleItem(
                review_id=review_id,
                ticket_id=ticket.ticket_id,
                channel=ticket.channel.value,
                customer_tier=ticket.customer_tier.value,
                route_outcome=res.routing_decision.route.value,
                redacted_inquiry=redacted_inquiry,
                candidate_customer_response=redacted_cand_resp,
                retrieved_passages=passages_summary,
                cited_doc_ids=cited_docs,
            )
        )

    notes = (
        f"Sampled {actual_sample_size} customer candidate responses (target: {target_total}). "
        f"Shortfall: {overall_shortfall}. Internal escalation notes were strictly excluded."
    )

    metadata = HumanReviewSamplingMetadata(
        target_sample_size=target_total,
        actual_sample_size=actual_sample_size,
        strata_targets={
            "auto_respond": target_auto,
            "draft_for_review": target_review,
            "escalate": target_escalate,
        },
        strata_actuals=strata_actuals,
        backfill_details=backfill_details,
        overall_shortfall=overall_shortfall,
        candidate_pool_size=total_eligible,
        notes=notes,
    )

    return sample_items, metadata


def generate_human_review_template_csv(items: Sequence[HumanReviewSampleItem]) -> str:
    """Generate an unpopulated, blinded 2-reviewer evaluation sheet in CSV format.

    Dimensions scored on 1-5 scale:
    - factual_correctness_1_to_5
    - citation_faithfulness_1_to_5
    - completeness_actionability_1_to_5
    - tone_compliance_1_to_5
    Plus binary hallucination flag:
    - contains_hallucination_true_false
    """
    output = io.StringIO()
    writer = csv.writer(output)

    headers = [
        "review_id",
        "ticket_id",
        "channel",
        "customer_tier",
        "route_outcome",
        "redacted_inquiry",
        "candidate_customer_response",
        "cited_doc_ids",
        # Reviewer 1 Scoring Columns (Unpopulated)
        "r1_factual_correctness_1_to_5",
        "r1_citation_faithfulness_1_to_5",
        "r1_completeness_actionability_1_to_5",
        "r1_tone_compliance_1_to_5",
        "r1_contains_hallucination_true_false",
        "r1_notes",
        # Reviewer 2 Scoring Columns (Unpopulated)
        "r2_factual_correctness_1_to_5",
        "r2_citation_faithfulness_1_to_5",
        "r2_completeness_actionability_1_to_5",
        "r2_tone_compliance_1_to_5",
        "r2_contains_hallucination_true_false",
        "r2_notes",
    ]
    writer.writerow(headers)

    for it in items:
        writer.writerow(
            [
                it.review_id,
                it.ticket_id,
                it.channel,
                it.customer_tier,
                it.route_outcome,
                it.redacted_inquiry,
                it.candidate_customer_response,
                ";".join(it.cited_doc_ids),
                # Empty reviewer fields for Reviewer 1
                "",
                "",
                "",
                "",
                "",
                "",
                # Empty reviewer fields for Reviewer 2
                "",
                "",
                "",
                "",
                "",
                "",
            ]
        )

    return output.getvalue()


def calculate_weighted_kappa(
    r1_scores: Sequence[int],
    r2_scores: Sequence[int],
    min_score: int = 1,
    max_score: int = 5,
) -> float:
    """Calculate linear-weighted Cohen's Kappa for ordinal 1-5 ratings.

    Formula:
        kappa = 1.0 - (sum(w_ij * O_ij) / sum(w_ij * E_ij))
        where linear weights w_ij = |i - j| / (k - 1).

    Args:
        r1_scores: Integer ratings from Reviewer 1 (e.g. 1 to 5).
        r2_scores: Integer ratings from Reviewer 2 (e.g. 1 to 5).
        min_score: Minimum Likert score (default 1).
        max_score: Maximum Likert score (default 5).

    Returns:
        Linear-weighted Cohen's Kappa in [-1.0, 1.0].

    Raises:
        ValueError: If sequence lengths mismatch or if zero reviewer scores exist.
    """
    if len(r1_scores) != len(r2_scores):
        raise ValueError("Length mismatch between reviewer score sequences.")

    n = len(r1_scores)
    if n == 0:
        raise ValueError("Cannot calculate agreement without two actual reviewer score sets.")

    k = max_score - min_score + 1
    # Build k x k observed matrix
    observed: list[list[int]] = [[0] * k for _ in range(k)]
    for s1, s2 in zip(r1_scores, r2_scores, strict=True):
        idx1 = max(0, min(k - 1, s1 - min_score))
        idx2 = max(0, min(k - 1, s2 - min_score))
        observed[idx1][idx2] += 1

    # Marginal totals
    r1_marginals = [sum(observed[i][j] for j in range(k)) for i in range(k)]
    r2_marginals = [sum(observed[i][j] for i in range(k)) for j in range(k)]

    # Expected matrix under chance agreement
    expected: list[list[float]] = [[0.0] * k for _ in range(k)]
    for i in range(k):
        for j in range(k):
            expected[i][j] = (r1_marginals[i] * r2_marginals[j]) / float(n)

    # Linear weights: w_ij = |i - j| / (k - 1)
    weight_denom = float(k - 1) if k > 1 else 1.0
    weighted_obs = 0.0
    weighted_exp = 0.0

    for i in range(k):
        for j in range(k):
            w = abs(i - j) / weight_denom
            weighted_obs += w * observed[i][j]
            weighted_exp += w * expected[i][j]

    if weighted_exp == 0.0:
        return 1.0 if weighted_obs == 0.0 else 0.0

    kappa = 1.0 - (weighted_obs / weighted_exp)
    return round(max(-1.0, min(1.0, kappa)), 4)


def calculate_binary_kappa(
    r1_flags: Sequence[bool],
    r2_flags: Sequence[bool],
) -> float:
    """Calculate unweighted Cohen's Kappa for the binary hallucination flag.

    Args:
        r1_flags: Boolean indicators from Reviewer 1 (e.g. contains_hallucination).
        r2_flags: Boolean indicators from Reviewer 2.

    Returns:
        Unweighted Cohen's Kappa in [-1.0, 1.0].

    Raises:
        ValueError: If sequence lengths mismatch or if zero reviewer scores exist.
    """
    if len(r1_flags) != len(r2_flags):
        raise ValueError("Length mismatch between reviewer flag sequences.")

    n = len(r1_flags)
    if n == 0:
        raise ValueError("Cannot calculate agreement without two actual reviewer score sets.")

    # 2x2 contingency matrix: [[both_false, r1_false_r2_true], [r1_true_r2_false, both_true]]
    c00 = sum(1 for f1, f2 in zip(r1_flags, r2_flags, strict=True) if not f1 and not f2)
    c01 = sum(1 for f1, f2 in zip(r1_flags, r2_flags, strict=True) if not f1 and f2)
    c10 = sum(1 for f1, f2 in zip(r1_flags, r2_flags, strict=True) if f1 and not f2)
    c11 = sum(1 for f1, f2 in zip(r1_flags, r2_flags, strict=True) if f1 and f2)

    po = (c00 + c11) / float(n)
    r1_true = (c10 + c11) / float(n)
    r1_false = (c00 + c01) / float(n)
    r2_true = (c01 + c11) / float(n)
    r2_false = (c00 + c10) / float(n)
    pe = (r1_true * r2_true) + (r1_false * r2_false)

    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0

    kappa = (po - pe) / (1.0 - pe)
    return round(max(-1.0, min(1.0, kappa)), 4)
