"""Evaluation reporter and artifact generator.

Coordinates generation of:
- summary_metrics.json (High-level aggregated JSON metrics across all subsystems)
- per_ticket_results.jsonl (Sanitized line-delimited diagnostics with zero raw inquiry PII)
- per_ticket_results.csv (Tabular diagnostic summary for spreadsheet import)
- human_review_packet.json (Stratified candidate responses with best-effort PII scrubbing)
- human_review_template.csv (Blinded 2-reviewer scoring sheet ready for human judges)
- human_review_metadata.json (Sampling targets, actuals, backfill counts, shortfall)
- EVALUATION_REPORT.md (Comprehensive human-readable governance and metrics report)
"""

import csv
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from cloudserve.evaluation.answer_eval import (
    AnswerQualityReport,
    ReferenceResponseRecord,
    evaluate_answers_batch,
)
from cloudserve.evaluation.human_review import (
    generate_human_review_template_csv,
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
from cloudserve.evaluation.retrieval_eval import (
    compute_ticket_retrieval_metrics,
    evaluate_batch_retrieval,
)
from cloudserve.ingestion.models import EvaluationRecord
from cloudserve.service import ProcessedTicketResult

logger = logging.getLogger(__name__)


def generate_evaluation_artifacts(
    run_id: str,
    metadata: dict[str, Any],
    results: Sequence[ProcessedTicketResult],
    ground_truth: dict[str, EvaluationRecord],
    reconciliation: dict[str, Any],
    output_dir: Path | str,
    kb_doc_ids: set[str] | None = None,
    reference_responses: (
        dict[str, ReferenceResponseRecord] | Sequence[dict[str, Any]] | None
    ) = None,
) -> dict[str, Any]:
    """Calculate all evaluation metrics, policy compliance, and export standardized artifacts.

    Enforces data minimization: raw ticket bodies, customer names, and emails are never
    written to per_ticket_results.jsonl, per_ticket_results.csv, or summary_metrics.json.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    auto_thresh = float(metadata.get("confidence_threshold_auto_respond", 0.80))
    review_thresh = float(metadata.get("confidence_threshold_review", 0.60))
    sim_thresh = float(metadata.get("retrieval_similarity_threshold", 0.65))
    exec_mode = str(metadata.get("execution_mode") or metadata.get("mode") or "mock").lower()

    # 1. Classification Metrics
    y_true_intent: list[str] = []
    y_pred_intent: list[str] = []
    y_true_urgency: list[str] = []
    y_pred_urgency: list[str] = []
    intent_confs: list[float] = []
    intent_accs: list[bool] = []
    urgency_confs: list[float] = []
    urgency_accs: list[bool] = []

    for res in results:
        tid = res.ticket.ticket_id
        gt = ground_truth.get(tid)
        labels = gt.labels if gt else None

        pred_i = res.classification.intent.value
        pred_u = res.classification.urgency.value
        conf_i = float(res.classification.intent_confidence)
        conf_u = float(res.classification.urgency_confidence)

        if labels:
            true_i = labels.intent
            true_u = labels.urgency
            y_true_intent.append(true_i)
            y_pred_intent.append(pred_i)
            y_true_urgency.append(true_u)
            y_pred_urgency.append(pred_u)

            is_i_correct = true_i == pred_i
            is_u_correct = true_u == pred_u
            intent_confs.append(conf_i)
            intent_accs.append(is_i_correct)
            urgency_confs.append(conf_u)
            urgency_accs.append(is_u_correct)

    class_report = calculate_classification_metrics(y_true_intent, y_pred_intent)
    urgency_report = calculate_urgency_metrics(y_true_urgency, y_pred_urgency)

    # 2. Calibration Metrics (flag mock diagnostic if mock execution mode)
    is_mock_run = exec_mode == "mock"
    intent_calib = calculate_confidence_calibration(intent_confs, intent_accs, is_mock=is_mock_run)
    urgency_calib = calculate_confidence_calibration(
        urgency_confs, urgency_accs, is_mock=is_mock_run
    )

    # 3. Retrieval Metrics
    retrieval_report = evaluate_batch_retrieval(results, ground_truth, top_k=4)

    # 4. Policy, Safety, and Routing Compliance Evaluation
    policy_report = evaluate_policy_and_safety(
        results=results,
        ground_truth=ground_truth,
        kb_doc_ids=kb_doc_ids,
        auto_thresh=auto_thresh,
        review_thresh=review_thresh,
        sim_thresh=sim_thresh,
    )

    # 5. Citations and Grounding Summary
    total_responses_with_citations = 0
    valid_citations_count = 0
    hallucinated_citations_count = 0

    for res in results:
        cand = res.generated_response.customer_response if res.generated_response else None
        if cand:
            audit_item = next(
                (a for a in policy_report.ticket_audits if a.ticket_id == res.ticket.ticket_id),
                None,
            )
            if audit_item and audit_item.retrieved_doc_ids:
                total_responses_with_citations += 1
                if audit_item.citations_valid:
                    valid_citations_count += 1
                if audit_item.has_hallucinated_citations:
                    hallucinated_citations_count += 1

    valid_citations_rate = (
        round(valid_citations_count / total_responses_with_citations, 4)
        if total_responses_with_citations > 0
        else 1.0
    )

    # 6. Human Review Sampling
    sample_items, hr_metadata = sample_human_review_packet(
        results=results,
        target_auto=25,
        target_review=15,
        target_escalate=10,
    )
    hr_template_csv = generate_human_review_template_csv(sample_items)

    # 6b. Answer Quality Evaluation against Reference Responses
    answer_quality_report = evaluate_answers_batch(
        results=results,
        reference_responses=reference_responses or {},
        ground_truth=ground_truth,
        kb_doc_ids=kb_doc_ids,
        is_mock=(exec_mode == "mock"),
    )

    # 7. Assemble summary_metrics.json
    summary_metrics_payload: dict[str, Any] = {
        "run_id": run_id,
        "execution_mode": exec_mode,
        "timestamp": metadata.get("timestamp") or metadata.get("created_at"),
        "total_records": len(results),
        "successful_records": sum(1 for r in results if r.error is None),
        "failed_records": sum(1 for r in results if r.error is not None),
        "classification": {
            "intent_accuracy": class_report.accuracy,
            "intent_macro_f1_represented": class_report.macro_f1_represented,
            "intent_macro_f1_fixed_22": class_report.macro_f1_fixed_22,
            "zero_support_classes": class_report.zero_support_classes,
            "per_class_metrics": {
                c: {
                    "precision": m.precision,
                    "recall": m.recall,
                    "f1": m.f1,
                    "support": m.support,
                }
                for c, m in class_report.per_class.items()
            },
            "urgency": {
                "accuracy": urgency_report.accuracy,
                "macro_f1": urgency_report.macro_f1,
                "per_class_metrics": {
                    c: {
                        "precision": m.precision,
                        "recall": m.recall,
                        "f1": m.f1,
                        "support": m.support,
                    }
                    for c, m in urgency_report.per_class.items()
                },
            },
        },
        "confidence_calibration": {
            "intent": {
                "full_expected_calibration_error": intent_calib.expected_calibration_error,
                "reliable_bins_ece": intent_calib.reliable_bins_ece,
                "reliable_bins_sample_denominator": (intent_calib.reliable_bins_sample_denominator),
                "reliable_bins_sample_coverage_pct": (
                    intent_calib.reliable_bins_sample_coverage_pct
                ),
                "max_calibration_error": intent_calib.max_calibration_error,
                "reliable_bins_count": intent_calib.reliable_bins_count,
                "meets_governance_target": intent_calib.meets_governance_target,
                "governance_target_status": intent_calib.governance_target_status,
                "bins": [
                    {
                        "bin_lower": b.bin_lower,
                        "bin_upper": b.bin_upper,
                        "sample_count": b.sample_count,
                        "average_confidence": b.average_confidence,
                        "observed_accuracy": b.observed_accuracy,
                        "calibration_gap": b.calibration_gap,
                        "is_statistically_reliable": b.is_statistically_reliable,
                    }
                    for b in intent_calib.bins
                ],
                "notes": intent_calib.notes,
            },
            "urgency": {
                "full_expected_calibration_error": urgency_calib.expected_calibration_error,
                "reliable_bins_ece": urgency_calib.reliable_bins_ece,
                "reliable_bins_sample_denominator": (
                    urgency_calib.reliable_bins_sample_denominator
                ),
                "reliable_bins_sample_coverage_pct": (
                    urgency_calib.reliable_bins_sample_coverage_pct
                ),
                "max_calibration_error": urgency_calib.max_calibration_error,
                "reliable_bins_count": urgency_calib.reliable_bins_count,
                "meets_governance_target": urgency_calib.meets_governance_target,
                "governance_target_status": urgency_calib.governance_target_status,
                "bins": [
                    {
                        "bin_lower": b.bin_lower,
                        "bin_upper": b.bin_upper,
                        "sample_count": b.sample_count,
                        "average_confidence": b.average_confidence,
                        "observed_accuracy": b.observed_accuracy,
                        "calibration_gap": b.calibration_gap,
                        "is_statistically_reliable": b.is_statistically_reliable,
                    }
                    for b in urgency_calib.bins
                ],
                "notes": urgency_calib.notes,
            },
        },
        "retrieval": retrieval_report,
        "routing_and_policy_compliance": {
            "dataset_agreement_rate": policy_report.dataset_agreement_rate,
            "prd_v2_policy_divergences_count": (policy_report.prd_v2_policy_divergences_count),
            "deterministic_execution_compliance_rate": (
                policy_report.deterministic_execution_compliance_rate
            ),
            "end_to_end_safety_compliance_rate": (policy_report.end_to_end_safety_compliance_rate),
            "safety_violations_count": policy_report.safety_violations_count,
            "total_released_responses": policy_report.total_released_responses,
            "unsafe_releases_count": policy_report.unsafe_releases_count,
            "release_safety_exercised": policy_report.release_safety_exercised,
            "release_safety_status": policy_report.release_safety_status,
            "release_safety_notes": policy_report.release_safety_notes,
            "mandatory_escalation_compliance_rate": (
                policy_report.mandatory_escalation_compliance_rate
            ),
            "billing_policy_compliance_rate": (policy_report.billing_policy_compliance_rate),
            "kill_switch_compliance_rate": (policy_report.kill_switch_compliance_rate),
            "audit_persistence_compliance_rate": (policy_report.audit_persistence_compliance_rate),
        },
        "citations_and_grounding": {
            "total_responses_with_citations": total_responses_with_citations,
            "valid_citations_rate": valid_citations_rate,
            "hallucinated_citations_count": hallucinated_citations_count,
        },
        "answer_quality": {
            "total_tickets": answer_quality_report.total_tickets,
            "reference_mapped_tickets": answer_quality_report.reference_mapped_tickets,
            "reference_coverage_pct": answer_quality_report.reference_coverage_pct,
            "evaluated_answer_count": answer_quality_report.evaluated_answer_count,
            "overall_answer_pass_rate": answer_quality_report.overall_answer_pass_rate,
            "must_mention": {
                "evaluated_tickets": answer_quality_report.must_mention_evaluated_tickets,
                "mean_coverage_ratio": answer_quality_report.mean_must_mention_coverage,
            },
            "must_not_claim": {
                "clean_rate": answer_quality_report.must_not_claim_clean_rate,
                "total_violations_count": answer_quality_report.total_claim_violations_count,
            },
            "diagnostic_semantic_similarity": {
                "mean": answer_quality_report.mean_diagnostic_similarity,
                "median": answer_quality_report.median_diagnostic_similarity,
                "is_diagnostic_only": True,
            },
            "citations_and_grounding": {
                "citation_validity_rate": answer_quality_report.citation_validity_rate,
                "grounding_pass_rate": answer_quality_report.grounding_pass_rate,
            },
            "outcomes": {
                "correct_auto_resolve": answer_quality_report.correct_auto_resolve_count,
                "correct_escalation": answer_quality_report.correct_escalation_count,
                "correct_draft_review": answer_quality_report.correct_draft_review_count,
                "quota_paused": answer_quality_report.quota_paused_count,
                "incorrect_routing": answer_quality_report.incorrect_routing_count,
                "failed_answer": answer_quality_report.failed_answer_count,
                "unanswerable_no_eval": answer_quality_report.unanswerable_no_eval_count,
            },
            "is_mock": answer_quality_report.is_mock,
        },
        "human_review_sampling": {
            "target_sample_size": hr_metadata.target_sample_size,
            "actual_sample_size": hr_metadata.actual_sample_size,
            "strata_counts": hr_metadata.strata_actuals,
            "backfill_details": hr_metadata.backfill_details,
            "overall_shortfall": hr_metadata.overall_shortfall,
            "candidate_pool_size": hr_metadata.candidate_pool_size,
        },
        "audit_reconciliation": reconciliation,
    }

    # Compute individual ticket processing latency statistics
    latencies = [
        r.processing_latency_ms
        for r in results
        if getattr(r, "processing_latency_ms", None) is not None
    ]
    if latencies:
        arr_lat = np.array(latencies, dtype=float)
        p95_val = round(float(np.percentile(arr_lat, 95)), 2)
        summary_metrics_payload["latency"] = {
            "observation_count": len(arr_lat),
            "min_ms": round(float(np.min(arr_lat)), 2),
            "max_ms": round(float(np.max(arr_lat)), 2),
            "mean_ms": round(float(np.mean(arr_lat)), 2),
            "median_ms": round(float(np.median(arr_lat)), 2),
            "p90_ms": round(float(np.percentile(arr_lat, 90)), 2),
            "p95_ms": p95_val,
            "p99_ms": round(float(np.percentile(arr_lat, 99)), 2),
            "p95_sla_target_ms": 3000.0,
            "p95_sla_met": bool(p95_val < 3000.0),
            "methodology_note": (
                "Computed directly from individual ticket observation records; "
                "not an aggregate of aggregates."
            ),
        }
    else:
        summary_metrics_payload["latency"] = None

    # Write summary_metrics.json
    with open(out_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary_metrics_payload, f, indent=2)

    # 8. Write per_ticket_results.jsonl and per_ticket_results.csv (Data Minimization Enforced)
    ticket_audit_lookup = {a.ticket_id: a for a in policy_report.ticket_audits}

    jsonl_lines: list[str] = []
    csv_rows: list[dict[str, Any]] = []

    for res in results:
        tid = res.ticket.ticket_id
        audit = ticket_audit_lookup.get(tid)
        gt = ground_truth.get(tid)
        labels = gt.labels if gt else None

        # Compute retrieval metrics for this specific ticket
        expected_docs = labels.expected_doc_ids if (labels and labels.expected_doc_ids) else []
        ret_metrics = compute_ticket_retrieval_metrics(
            passages=res.retrieved_passages,
            expected_doc_ids=expected_docs,
            top_k=4,
        )

        pred_intent_val = audit.predicted_intent if audit else res.classification.intent.value
        row: dict[str, Any] = {
            "ticket_id": tid,
            "channel": res.ticket.channel.value,
            "customer_tier": res.ticket.customer_tier.value,
            "true_intent": audit.true_intent if audit else None,
            "predicted_intent": pred_intent_val,
            "intent_correct": audit.intent_correct if audit else False,
            "intent_confidence": (
                audit.intent_confidence if audit else res.classification.intent_confidence
            ),
            "true_urgency": audit.true_urgency if audit else None,
            "predicted_urgency": (
                audit.predicted_urgency if audit else res.classification.urgency.value
            ),
            "urgency_correct": audit.urgency_correct if audit else False,
            "urgency_confidence": (
                audit.urgency_confidence if audit else res.classification.urgency_confidence
            ),
            "true_route": audit.true_route if audit else None,
            "actual_route": audit.actual_route if audit else res.routing_decision.route.value,
            "effective_route": (
                audit.effective_route if audit else res.routing_decision.route.value
            ),
            "route_matches_expected": audit.route_matches_expected if audit else False,
            "is_prd_policy_divergence": audit.is_prd_policy_divergence if audit else False,
            "reason_code": audit.reason_code if audit else res.routing_decision.reason_code.value,
            "billing_subtype_inferred": audit.billing_subtype_inferred if audit else None,
            "retrieved_doc_ids": audit.retrieved_doc_ids if audit else [],
            "expected_doc_ids": audit.expected_doc_ids if audit else expected_docs,
            "precision_at_4": ret_metrics["precision_at_4"],
            "recall_at_4": ret_metrics["recall_at_4"],
            "hit_at_4": ret_metrics["hit_rate_at_4"],
            "reciprocal_rank": ret_metrics["reciprocal_rank"],
            "citations_valid": audit.citations_valid if audit else False,
            "safety_compliant": not audit.is_safety_violation if audit else True,
            "customer_response_released": (audit.customer_response_released if audit else False),
            "audit_id": res.audit_id,
            "persistence_status": res.persistence_status,
            "delivery_status": res.delivery_status,
            "processing_latency_ms": round(res.processing_latency_ms, 2),
        }

        jsonl_lines.append(json.dumps(row))

        # Flatten list fields for CSV representation
        csv_row = dict(row)
        csv_row["retrieved_doc_ids"] = ";".join(row["retrieved_doc_ids"])
        csv_row["expected_doc_ids"] = ";".join(row["expected_doc_ids"])
        csv_rows.append(csv_row)

    # Write per_ticket_results.jsonl
    with open(out_dir / "per_ticket_results.jsonl", "w", encoding="utf-8") as f:
        f.write("\n".join(jsonl_lines) + "\n")

    # Write per_ticket_results.csv
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with open(out_dir / "per_ticket_results.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

    # 9. Write Human Review Artifacts
    with open(out_dir / "human_review_packet.json", "w", encoding="utf-8") as f:
        json.dump([item.model_dump() for item in sample_items], f, indent=2)

    with open(out_dir / "human_review_template.csv", "w", encoding="utf-8", newline="") as f:
        f.write(hr_template_csv)

    with open(out_dir / "human_review_metadata.json", "w", encoding="utf-8") as f:
        json.dump(hr_metadata.model_dump(), f, indent=2)

    # 9b. Write Answer Quality Artifacts
    with open(out_dir / "answer_quality_report.json", "w", encoding="utf-8") as f:
        json.dump(answer_quality_report.model_dump(), f, indent=2)

    with open(out_dir / "answer_quality_results.jsonl", "w", encoding="utf-8") as f:
        for te in answer_quality_report.ticket_evaluations:
            f.write(te.model_dump_json() + "\n")

    # 10. Write EVALUATION_REPORT.md
    report_md = _build_evaluation_markdown_report(
        metadata=metadata,
        summary=summary_metrics_payload,
        class_report=class_report,
        urgency_report=urgency_report,
        intent_calib=intent_calib,
        urgency_calib=urgency_calib,
        policy_report=policy_report,
        retrieval_report=retrieval_report,
        hr_metadata=hr_metadata,
        reconciliation=reconciliation,
        answer_quality_report=answer_quality_report,
    )
    with open(out_dir / "EVALUATION_REPORT.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    logger.info("Successfully exported all evaluation artifacts to %s", out_dir)
    return summary_metrics_payload


def _build_evaluation_markdown_report(
    metadata: dict[str, Any],
    summary: dict[str, Any],
    class_report: Any,
    urgency_report: Any,
    intent_calib: Any,
    urgency_calib: Any,
    policy_report: Any,
    retrieval_report: dict[str, Any],
    hr_metadata: Any,
    reconciliation: dict[str, Any],
    answer_quality_report: AnswerQualityReport | None = None,
) -> str:
    """Assemble GitHub-Flavored Markdown evaluation report for stakeholders."""
    run_id = metadata.get("run_id", "N/A")
    mode = str(metadata.get("execution_mode") or metadata.get("mode") or "mock").upper()
    timestamp = metadata.get("timestamp") or metadata.get("created_at") or "N/A"
    total_records = summary.get("total_records", 0)

    # Build per-class intent markdown table
    class_table_rows: list[str] = []
    for cls_name, m in class_report.per_class.items():
        class_table_rows.append(
            f"| `{cls_name}` | {m.precision:.3f} | {m.recall:.3f} | {m.f1:.3f} | {m.support} |"
        )
    class_table_str = "\n".join(class_table_rows)

    # Build urgency markdown table
    urgency_rows: list[str] = []
    for u_name, m in urgency_report.per_class.items():
        urgency_rows.append(
            f"| `{u_name}` | {m.precision:.3f} | {m.recall:.3f} | {m.f1:.3f} | {m.support} |"
        )
    urgency_table_str = "\n".join(urgency_rows)

    # Build calibration bins table
    calib_rows: list[str] = []
    for b in intent_calib.bins:
        if b.sample_count == 0:
            reliable_mark = "Empty (0)"
        elif b.is_statistically_reliable:
            reliable_mark = "Yes"
        else:
            reliable_mark = "Sparse (<10)"
        calib_rows.append(
            f"| [{b.bin_lower:.1f}, {b.bin_upper:.1f}) | {b.sample_count} | "
            f"{b.average_confidence:.3f} | {b.observed_accuracy:.3f} | "
            f"{b.calibration_gap:.3f} | {reliable_mark} |"
        )
    calib_table_str = "\n".join(calib_rows)

    # Retrieval stats
    ans_ret = retrieval_report.get("answerable_tickets", {})
    unans_ret = retrieval_report.get("unanswerable_tickets", {})

    reconciled_str = "PASSED (1:1)" if reconciliation.get("reconciled") else "FAILED GAP"
    exec_rate = policy_report.deterministic_execution_compliance_rate * 100
    exec_stat = "PASS" if policy_report.deterministic_execution_compliance_rate == 1.0 else "WARN"
    if policy_report.end_to_end_safety_compliance_rate is not None:
        safe_rate_str = f"{policy_report.end_to_end_safety_compliance_rate * 100:.1f}%"
        safe_stat = "PASS" if policy_report.end_to_end_safety_compliance_rate == 1.0 else "CRITICAL"
    else:
        safe_rate_str = "N/A (0 releases)"
        safe_stat = "NOT EXERCISED"
    esc_rate = policy_report.mandatory_escalation_compliance_rate * 100
    esc_stat = "PASS" if policy_report.mandatory_escalation_compliance_rate == 1.0 else "FAIL"

    p4 = ans_ret.get("precision_at_4", 0.0)
    r4 = ans_ret.get("recall_at_4", 0.0)
    hit4 = ans_ret.get("hit_rate_at_4", 0.0)
    mrr = ans_ret.get("mrr", 0.0)
    mean_p = ans_ret.get("mean_precision", 0.0)
    mean_r = ans_ret.get("mean_recall", 0.0)

    unans_empty = unans_ret.get("empty_retrieval_rate", 0.0)
    unans_non_empty = unans_ret.get("non_empty_retrieval_rate", 0.0)
    unans_cnt_e = unans_ret.get("empty_count", 0)
    unans_cnt_ne = unans_ret.get("non_empty_count", 0)

    total_ans = ans_ret.get("total", 0)
    total_unans = unans_ret.get("total", 0)

    cls_class = metadata.get("classifier_class") or metadata.get("classifier_llm_client_class")
    cls_model = metadata.get("classifier_model")
    if cls_class and cls_model and cls_model != cls_class and cls_model not in cls_class:
        cls_llm = f"{cls_class} ({cls_model})"
    elif cls_model:
        cls_llm = str(cls_model)
    elif cls_class:
        cls_llm = str(cls_class)
    else:
        cls_llm = "MockLLMClient"

    gen_class = metadata.get("generator_class") or metadata.get("generator_llm_client_class")
    gen_model = metadata.get("generator_model")
    if gen_class and gen_model and gen_model != gen_class and gen_model not in gen_class:
        gen_llm = f"{gen_class} ({gen_model})"
    elif gen_model:
        gen_llm = str(gen_model)
    elif gen_class:
        gen_llm = str(gen_class)
    else:
        gen_llm = "MockLLMClient"

    emb_class = metadata.get("embedding_class") or metadata.get("embedding_function_class")
    emb_model = metadata.get("embedding_model")
    if emb_class and emb_model and emb_model != emb_class and emb_model not in emb_class:
        emb_fn = f"{emb_class} ({emb_model})"
    elif emb_model:
        emb_fn = str(emb_model)
    elif emb_class:
        emb_fn = str(emb_class)
    else:
        emb_fn = "MockEmbeddingFunction"

    p_ver = (
        metadata.get("prompt_version")
        or metadata.get("classifier_prompt_version")
        or metadata.get("generator_prompt_version")
        or "1.0.0"
    )
    t_auto = metadata.get("confidence_threshold_auto_respond", 0.80)
    t_rev = metadata.get("confidence_threshold_review", 0.60)
    t_sim = metadata.get("retrieval_similarity_threshold", 0.65)

    tot_cit_resps = summary.get("citations_and_grounding", {}).get(
        "total_responses_with_citations", 0
    )
    val_cit_rate = summary.get("citations_and_grounding", {}).get("valid_citations_rate", 1.0) * 100
    halluc_cits = summary.get("citations_and_grounding", {}).get("hallucinated_citations_count", 0)

    backfill_json = (
        json.dumps(hr_metadata.backfill_details)
        if hr_metadata.backfill_details
        else "None required"
    )

    persisted_cnt = reconciliation.get("persisted_records", 0)
    processed_cnt = reconciliation.get("total_processed", 0)

    calib_ece = intent_calib.expected_calibration_error
    if mode.lower() == "mock":
        calib_stat = "Mock Diagnostic (Not Assessed)"
    else:
        calib_stat = intent_calib.governance_target_status
    rel_ece_str = (
        f"{intent_calib.reliable_bins_ece:.4f}"
        if intent_calib.reliable_bins_ece is not None
        else "N/A"
    )
    rel_cov = intent_calib.reliable_bins_sample_coverage_pct
    rel_denom = intent_calib.reliable_bins_sample_denominator
    tot_samp = intent_calib.total_samples
    mce_val = intent_calib.max_calibration_error

    zero_supp_str = (
        class_report.zero_support_classes if class_report.zero_support_classes else "None"
    )
    hr_auto_cnt = hr_metadata.strata_actuals.get("auto_respond", 0)
    hr_draft_cnt = hr_metadata.strata_actuals.get("draft_for_review", 0)
    hr_esc_cnt = hr_metadata.strata_actuals.get("escalate", 0)
    bill_comp_pct = policy_report.billing_policy_compliance_rate * 100
    kill_comp_pct = policy_report.kill_switch_compliance_rate * 100
    audit_comp_pct = policy_report.audit_persistence_compliance_rate * 100
    agr_pct = policy_report.dataset_agreement_rate * 100
    div_cnt = policy_report.prd_v2_policy_divergences_count

    doc_lines = [
        "# CloudServe AI Support System — Unattended Evaluation Report",
        "",
        f"**Run Identifier:** `{run_id}`",
        f"**Execution Mode:** `{mode}`",
        f"**Evaluation Timestamp:** `{timestamp}`",
        f"**Total Records Evaluated:** `{total_records}`",
        f"**Persistence Reconciliation:** `{reconciled_str}`",
        "",
        "---",
        "",
        "## 1. Executive Summary & Governance Scorecard",
        "",
        "| Governance Area | Observed Metric | Target | Status |",
        "| :--- | :--- | :--- | :--- |",
        f"| **Intent Accuracy** | `{class_report.accuracy:.3f}` | Baseline | Evaluated |",
        (
            f"| **Represented Macro-F1** | `{class_report.macro_f1_represented:.3f}` | "
            "Baseline | Evaluated |"
        ),
        (
            f"| **Fixed 22 Macro-F1** | `{class_report.macro_f1_fixed_22:.3f}` | "
            "Fixed Taxonomy | Evaluated |"
        ),
        (f"| **Urgency Accuracy** | `{urgency_report.accuracy:.3f}` | Baseline | Evaluated |"),
        f"| **Intent Calibration (ECE)** | `{calib_ece:.3f}` | $\\le 0.05$ | `{calib_stat}` |",
        f"| **Precision@4 (Fixed /4)** | `{p4:.3f}` | Relevance | Evaluated |",
        f"| **Recall@4** | `{r4:.3f}` | Grounding | Evaluated |",
        f"| **Hit Rate@4** | `{hit4:.3f}` | Grounding | Evaluated |",
        *(
            [
                (
                    f"| **Answer Quality Pass Rate** | "
                    f"`{answer_quality_report.overall_answer_pass_rate * 100:.1f}%` | "
                    f"Ref Mapped ({answer_quality_report.reference_mapped_tickets}) | "
                    "Evaluated |"
                ),
                (
                    f"| **Forbidden Claims Clean Rate** | "
                    f"`{answer_quality_report.must_not_claim_clean_rate * 100:.1f}%` | "
                    "100.0% | "
                    + (
                        "PASS |"
                        if answer_quality_report.must_not_claim_clean_rate == 1.0
                        else "FAIL |"
                    )
                ),
            ]
            if answer_quality_report is not None
            else []
        ),
        (f"| **Deterministic Router Execution** | `{exec_rate:.1f}%` | 100.0% | `{exec_stat}` |"),
        (
            f"| **Released Customer Responses** | "
            f"`{policy_report.total_released_responses}` | "
            f"Exercised: `{policy_report.release_safety_exercised}` | "
            f"`{policy_report.release_safety_status}` |"
        ),
        (f"| **End-to-End Customer Safety** | `{safe_rate_str}` | 100.0% | `{safe_stat}` |"),
        (f"| **Mandatory Escalation Compliance**| `{esc_rate:.1f}%` | 100.0% | `{esc_stat}` |"),
        (
            f"| **Decision Audit Persistence** | `{persisted_cnt} / {processed_cnt}` | "
            f"100% | `{reconciled_str}` |"
        ),
        "",
        "---",
        "",
        "## 2. Component Configuration & Active Thresholds",
        "",
        f"- **LLM Classifier Provider:** `{cls_llm}`",
        f"- **Response Generator Provider:** `{gen_llm}`",
        f"- **Embedding Provider:** `{emb_fn}`",
        f"- **Active Prompt Version:** `{p_ver}`",
        f"- **Auto-Respond Confidence Threshold:** `{t_auto}`",
        f"- **Review Draft Confidence Threshold:** `{t_rev}`",
        f"- **Retrieval Similarity Threshold:** `{t_sim}`",
        "",
        "---",
        "",
        "## 3. Classification Performance",
        "",
        "### 3.1 Intent Classification across the 22-Class Taxonomy",
        f"- **Overall Intent Accuracy:** `{class_report.accuracy:.3f}`",
        (
            f"- **Represented Macro-F1:** `{class_report.macro_f1_represented:.3f}` "
            "*(represented classes)*"
        ),
        (
            f"- **Fixed 22-Taxonomy Macro-F1:** `{class_report.macro_f1_fixed_22:.3f}` "
            "*(zero-support scores 0.0)*"
        ),
        f"- **Zero-Support Classes:** `{zero_supp_str}`",
        "",
        "| Intent Class | Precision | Recall | F1 Score | Support |",
        "| :--- | :--- | :--- | :--- | :--- |",
        class_table_str,
        "",
        "### 3.2 Urgency Classification (3 Tiers)",
        f"- **Overall Urgency Accuracy:** `{urgency_report.accuracy:.3f}`",
        f"- **Urgency Macro-F1:** `{urgency_report.macro_f1:.3f}`",
        "",
        "| Urgency Level | Precision | Recall | F1 Score | Support |",
        "| :--- | :--- | :--- | :--- | :--- |",
        urgency_table_str,
        "",
        "---",
        "",
        "## 4. Confidence Calibration Analysis",
        "",
        "### 4.1 Intent Confidence Reliability Table",
        f"- **Full ECE (All Samples):** `{calib_ece:.4f}`",
        f"- **Reliable Bins ECE:** `{rel_ece_str}`",
        f"- **Reliable Bins Sample Coverage:** `{rel_cov:.1f}% ({rel_denom}/{tot_samp} samples)`",
        f"- **Maximum Calibration Error (MCE):** `{mce_val:.4f}`",
        f"- **Governance Target Status:** `{calib_stat}`",
        "",
        (
            "| Confidence Bin | Sample Count | Mean Confidence | Observed Accuracy | "
            "Calibration Gap | Reliable? |"
        ),
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
        calib_table_str,
        "",
        "> [!NOTE]",
        "> **LLM Verbalized Confidence Disclaimer:**",
        (
            "> The confidence values analyzed above are verbalized numeric ratings emitted by "
            "the prompt."
        ),
        (
            "> They represent an uncalibrated heuristic score rather than true Bayesian posterior "
            "probabilities."
        ),
        (
            "> They have not been adjusted through post-hoc calibration methods (such as Platt "
            "scaling)."
        ),
        *(
            [
                ">",
                "> [!NOTE]",
                "> **Mock Confidence Diagnostic:**",
                (
                    "> This run executed in mock mode using MockLLMClient. "
                    "Actual LLM calibration is not assessed."
                ),
            ]
            if mode.lower() == "mock"
            else []
        ),
        "",
        "---",
        "",
        "## 5. Knowledge Base Retrieval Performance",
        "",
        f"### 5.1 Documentation-Answerable Tickets ({total_ans} tickets)",
        f"- **Precision@4 (Fixed Denominator of 4):** `{p4:.4f}`",
        (
            "  *(Relevant unique documents in top 4 divided by 4; missing positions "
            "count as non-relevant).*"
        ),
        f"- **Recall@4:** `{r4:.4f}`",
        f"- **Hit Rate@4:** `{hit4:.4f}`",
        f"- **Mean Reciprocal Rank (MRR):** `{mrr:.4f}`",
        f"- **Mean Precision (All Retrieved Passages):** `{mean_p:.4f}`",
        f"- **Mean Recall (All Retrieved Passages):** `{mean_r:.4f}`",
        "",
        f"### 5.2 Documentation-Unanswerable Tickets ({total_unans} tickets)",
        f"- **Empty Retrieval Rate:** `{unans_empty:.4f} ({unans_cnt_e} tickets returned 0)`",
        f"- **Non-Empty Retrieval Rate:** `{unans_non_empty:.4f} ({unans_cnt_ne} tickets)`",
        "",
        "> [!NOTE]",
        "> **Embedding Representation Transparency:**",
        (
            f"> This evaluation ran in **`{mode}`** mode using `{emb_fn}`. Mock hashing uses "
            "random token"
        ),
        (
            "> projection and does not model semantics. Neural vector evaluation requires "
            "live adapter."
        ),
        "",
        "---",
        "",
        "## 6. Policy, Safety, and Routing Compliance",
        "",
        "### 6.1 Three-Layer Disposition Breakdown",
        "1. **Deterministic Router Disposition:**",
        "   - Evaluates whether router obeyed PRD v2.0 decision precedence on runtime inputs.",
        f"   - **Compliance Rate:** `{exec_rate:.1f}%`",
        "2. **Internal Candidate Response Generation:**",
        "   - Candidate responses synthesized for support review or audit handoffs.",
        "   - Unreleased candidate responses held for review are not customer violations.",
        "3. **Actual Customer Release:**",
        (
            "   - Defined strictly as `final_customer_response is not None AND "
            'delivery_status == "released"`.'
        ),
        (f"   - **Total Customer Responses Released:** `{policy_report.total_released_responses}`"),
        f"   - **Unsafe Customer Releases:** `{policy_report.unsafe_releases_count}`",
        f"   - **Release-Safety Exercised:** `{policy_report.release_safety_exercised}`",
        f"   - **Release-Safety Status:** `{policy_report.release_safety_status}`",
        f"   - **End-to-End Safety Compliance Rate:** `{safe_rate_str}`",
        f"   - **Safety Violations Count:** `{policy_report.safety_violations_count}`",
        "",
        "> [!IMPORTANT]",
        "> **Release-Safety Governance Note:**",
        f"> {policy_report.release_safety_notes}",
        ("> Internal candidate responses synthesized for agent review or audit notes are strictly"),
        ("> excluded from customer release and do NOT constitute customer safety violations."),
        "",
        "### 6.2 Policy Subsystem Compliance Rates",
        f"- **Mandatory Escalation Compliance:** `{esc_rate:.1f}%`",
        f"- **Billing Policy Compliance (FR-13):** `{bill_comp_pct:.1f}%`",
        "  *(Audits deterministic execution on inferred billing risk; lacks ground truth).*",
        f"- **Operational Kill Switch Compliance:** `{kill_comp_pct:.1f}%`",
        f"- **Audit Persistence Gate Compliance:** `{audit_comp_pct:.1f}%`",
        "",
        "### 6.3 PRD v2.0 Policy Divergence vs Historical Dataset Agreement",
        f"- **Historical Benchmark Agreement Rate:** `{agr_pct:.1f}%`",
        f"- **PRD v2.0 Policy Overrides:** `{div_cnt} tickets`",
        (
            "  *(Tickets where dataset labeled `auto_respond`, but PRD v2.0 mandates "
            "escalation, e.g. Data Residency).*"
        ),
        "",
        "---",
        "",
        "## 7. Citation Grounding & Guardrails Verification",
        "",
        f"- **Candidate Responses Checked for Citations:** `{tot_cit_resps}`",
        f"- **Valid Citations Rate:** `{val_cit_rate:.1f}%`",
        f"- **Hallucinated Citations Count:** `{halluc_cits}`",
        *(
            [
                "",
                "---",
                "",
                "## 7.1 Answer Quality Evaluation against Reference Responses",
                "",
                (
                    f"- **Reference-Mapped Tickets Evaluated:** "
                    f"`{answer_quality_report.reference_mapped_tickets} / "
                    f"{answer_quality_report.total_tickets} "
                    f"({answer_quality_report.reference_coverage_pct:.1f}%)`"
                ),
                (
                    f"- **Overall Answer Pass Rate:** "
                    f"`{answer_quality_report.overall_answer_pass_rate * 100:.1f}%`"
                ),
                (
                    f"- **Must-Not-Claim Clean Rate:** "
                    f"`{answer_quality_report.must_not_claim_clean_rate * 100:.1f}%` "
                    f"({answer_quality_report.total_claim_violations_count} violations detected)"
                ),
                (
                    f"- **Must-Mention Mean Coverage:** "
                    f"`{answer_quality_report.mean_must_mention_coverage * 100:.1f}%` "
                    f"across {answer_quality_report.must_mention_evaluated_tickets} "
                    "applicable tickets"
                ),
                (
                    f"- **Diagnostic Semantic Similarity (TF-IDF Cosine):** "
                    f"Mean: `{answer_quality_report.mean_diagnostic_similarity or 'N/A'}` | "
                    f"Median: `{answer_quality_report.median_diagnostic_similarity or 'N/A'}`"
                ),
                (
                    "  *(Diagnostic similarity is an informative metric only; "
                    "no uncalibrated threshold is used for pass/fail decisions).*"
                ),
                "",
                "### Outcome Categorization Breakdown:",
                (
                    f"- **Correct Auto-Resolve:** "
                    f"`{answer_quality_report.correct_auto_resolve_count}`"
                ),
                (
                    f"- **Correct Escalation (Protected):** "
                    f"`{answer_quality_report.correct_escalation_count}`"
                ),
                (
                    f"- **Correct Draft Review:** "
                    f"`{answer_quality_report.correct_draft_review_count}`"
                ),
                (
                    f"- **Quota Paused:** "
                    f"`{answer_quality_report.quota_paused_count}`"
                ),
                (
                    f"- **Incorrect Routing:** "
                    f"`{answer_quality_report.incorrect_routing_count}`"
                ),
                (
                    f"- **Failed Answer (Claims/Mentions/Citations):** "
                    f"`{answer_quality_report.failed_answer_count}`"
                ),
            ]
            if answer_quality_report is not None
            else []
        ),
        "",
        "---",
        "",
        "## 8. Human Review Sampling Protocol",
        "",
        f"- **Target Sample Size:** `{hr_metadata.target_sample_size}`",
        f"- **Actual Sample Size:** `{hr_metadata.actual_sample_size}`",
        f"- **Overall Shortfall:** `{hr_metadata.overall_shortfall}`",
        (f"- **Candidate Pool Size (Customer drafts only):** `{hr_metadata.candidate_pool_size}`"),
        "- **Strata Breakdown:**",
        f"  - `AUTO_RESPOND`: `{hr_auto_cnt}` (Target: 25)",
        f"  - `DRAFT_FOR_REVIEW`: `{hr_draft_cnt}` (Target: 15)",
        f"  - `ESCALATE` (Customer drafts only): `{hr_esc_cnt}` (Target: 10)",
        f"- **Backfill Log:** `{backfill_json}`",
        "- **Blinded Template:** `human_review_template.csv` generated with empty scoring columns.",
        "",
        "---",
        "",
        "## 9. Decision Audit Reconciliation (A8 Compliance)",
        "",
        f"- **Total Processed Attempts:** `{processed_cnt}`",
        f"- **Persisted Decision Records in SQLite:** `{persisted_cnt}`",
        f"- **Failed or Missing Records:** `{reconciliation.get('failed_or_missing_records', 0)}`",
        f"- **Audit Reconciliation Gap:** `{reconciliation.get('gap', 0)}`",
        f"- **Reconciliation Status:** `{reconciled_str}`",
        "",
        "---",
        "",
        "## 10. Processing Latency Telemetry (Individual Observations)",
        "",
        *(
            [
                f"- **Total Latency Observations:** `{summary['latency']['observation_count']}`",
                (
                    f"- **Mean Latency:** `{summary['latency']['mean_ms']:.2f} ms` "
                    f"({summary['latency']['mean_ms'] / 1000.0:.2f} s)"
                ),
                (
                    f"- **Median Latency (P50):** `{summary['latency']['median_ms']:.2f} ms` "
                    f"({summary['latency']['median_ms'] / 1000.0:.2f} s)"
                ),
                (
                    f"- **P90 Latency:** `{summary['latency']['p90_ms']:.2f} ms` "
                    f"({summary['latency']['p90_ms'] / 1000.0:.2f} s)"
                ),
                (
                    f"- **P95 Latency:** `{summary['latency']['p95_ms']:.2f} ms` "
                    f"({summary['latency']['p95_ms'] / 1000.0:.2f} s) | "
                    f"**Target SLA:** `< {summary['latency']['p95_sla_target_ms']:.0f} ms` "
                    f"({'PASS' if summary['latency']['p95_sla_met'] else 'SLA MISSED'})"
                ),
                (
                    f"- **P99 Latency:** `{summary['latency']['p99_ms']:.2f} ms` "
                    f"({summary['latency']['p99_ms'] / 1000.0:.2f} s)"
                ),
                (
                    f"- **Min / Max Latency:** `{summary['latency']['min_ms']:.2f} ms` / "
                    f"`{summary['latency']['max_ms']:.2f} ms`"
                ),
                "",
                "| Metric | Observed (ms) | Observed (s) | Target SLA | Compliance Status |",
                "| :--- | :--- | :--- | :--- | :--- |",
                (
                    f"| **P95 Processing Latency** | `{summary['latency']['p95_ms']:.2f} ms` | "
                    f"`{summary['latency']['p95_ms'] / 1000.0:.2f} s` | `< 3000 ms` (< 3.0 s) | "
                    f"`{'PASS' if summary['latency']['p95_sla_met'] else 'FAIL (SLA MISSED)'}` |"
                ),
                (
                    f"| **Mean Processing Latency** | `{summary['latency']['mean_ms']:.2f} ms` | "
                    f"`{summary['latency']['mean_ms'] / 1000.0:.2f} s` | "
                    "Operational Reference | Evaluated |"
                ),
                (
                    f"| **Median (P50) Latency** | `{summary['latency']['median_ms']:.2f} ms` | "
                    f"`{summary['latency']['median_ms'] / 1000.0:.2f} s` | "
                    "Operational Reference | Evaluated |"
                ),
                "",
                "> [!NOTE]",
                "> **Latency Telemetry Methodology Disclaimer:**",
                (
                    f"> {summary['latency']['methodology_note']} "
                    "These statistics represent exact end-to-end execution times per ticket "
                    "(intake -> classify -> retrieve -> generate -> safety -> persist) "
                    "and are not derived from pre-aggregated batch percentiles."
                ),
            ]
            if summary.get("latency")
            else [
                "- Latency telemetry was not recorded for this execution run.",
            ]
        ),
        "",
        "---",
        "",
        "## 11. Governance Boundaries & Operational Disclaimers",
        "",
        "1. **No Live CSAT or First-Response Time Claims:**",
        (
            "   The benchmark dataset contains historical metrics (`csat_rating`, "
            "`resolution_time_minutes`, `first_contact_resolution`) achieved by humans."
        ),
        (
            "   Offline batch evaluation measures latency, accuracy, and routing. "
            "It cannot measure live customer satisfaction."
        ),
        ("   Offline processing latency does not equal live first-response time."),
        "2. **Policy Divergence vs Defect:**",
        (
            "   PRD Version 2.0 policy mandates escalation for all Data Residency inquiries. "
            "Where the benchmark dataset historically labeled such tickets as `auto_respond`,"
        ),
        ("   system deviation is an intentional governance compliance override, not a defect."),
        "3. **Mock Mode Disclaimer:**",
        (
            "   Mock evaluation runs reflect deterministic pipeline mechanics, routing logic, "
            "and software contracts, not generative performance of live models."
        ),
        "4. **Zero Validation-Set Tuning:**",
        (
            "   All routing thresholds and prompt configurations are fixed in application "
            "settings; no parameters are tuned against evaluation or validation datasets."
        ),
    ]
    return "\n".join(doc_lines) + "\n"
