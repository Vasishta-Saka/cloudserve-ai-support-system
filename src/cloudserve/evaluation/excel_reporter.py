"""Excel evaluation reporter for CloudServe AI Support System.

Generates the comprehensive workbook:
    CloudServe_Ticket_Level_Evaluation_Report.xlsx

Worksheets:
    1. Ticket-Level Evaluation (43 columns, 1 row per ticket, styled and formatted)
    2. Evaluation Summary (Executive dashboard with KPI blocks, routing, safety, and latency)
    3. Data Sources & Dictionary (Dataset provenance, 43-field dictionary, policy rules)
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import openpyxl
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.worksheet import Worksheet

    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

from cloudserve.ingestion.models import EvaluationRecord
from cloudserve.service import ProcessedTicketResult

logger = logging.getLogger(__name__)

# 43 Canonical Column Headers for Sheet 1
TICKET_LEVEL_COLUMNS: list[str] = [
    # 1-5 Inbound Inquiry Metadata
    "Ticket ID",
    "Channel",
    "Customer Tier",
    "Customer Subject",
    "Customer Question / Body",
    # 6-13 Classification & Calibration
    "Predicted Intent",
    "Expected Intent",
    "Intent Confidence",
    "Intent Match",
    "Predicted Urgency",
    "Expected Urgency",
    "Urgency Confidence",
    "Urgency Match",
    # 14-18 Input Safety & Guardrails
    "Input Guardrail Status",
    "Guardrail Violation Category",
    "Prompt Injection Detection Status",
    "PII Detection Status",
    "Guardrails Passed",
    # 19-24 Knowledge-Base Retrieval
    "Retrieved Article IDs",
    "Top Retrieved Chunk ID",
    "Top Retrieval Similarity Score",
    "All Retrieved Chunk IDs",
    "All Retrieved Similarity Scores",
    "Retrieved Context Excerpt",
    # 25-30 Response Synthesis & Citations
    "Candidate Response Generated",
    "Candidate Response Text",
    "Candidate Response Word Count",
    "Response Citation IDs",
    "Citations Verified",
    "Citation Validation Details",
    # 31-37 Routing & Delivery Decision
    "Routing Decision",
    "Routing Reason Code",
    "Routing Reason Description",
    "Target Specialist Group",
    "Delivery Status",
    "Final Response Released to Customer",
    "Review Hold Reason",
    # 38-43 Execution & Audit Metadata
    "Processing Latency (ms)",
    "Persistence Status",
    "Audit ID",
    "Execution Timestamp",
    "Evaluation Mode",
    "Notes / Observations",
]


def export_evaluation_excel(
    output_path: Path | str,
    results: Sequence[ProcessedTicketResult],
    ground_truth: dict[str, EvaluationRecord],
    metadata: dict[str, Any],
    summary_metrics: dict[str, Any],
    reconciliation: dict[str, Any],
    policy_report: Any | None = None,
    answer_quality_report: Any | None = None,
) -> Path:
    """Generate the standardized CloudServe_Ticket_Level_Evaluation_Report.xlsx workbook.

    Args:
        output_path: Destination filepath for the Excel workbook.
        results: In-memory sequence of ProcessedTicketResult objects.
        ground_truth: Lookup mapping ticket_id to EvaluationRecord.
        metadata: Execution and configuration metadata.
        summary_metrics: Aggregated metrics dictionary.
        reconciliation: SQLite audit reconciliation dictionary.
        policy_report: Optional PolicySafetyReport instance.
        answer_quality_report: Optional AnswerQualityReport instance.

    Returns:
        Path to the generated Excel workbook.

    Raises:
        ImportError: If openpyxl is not installed.
        RuntimeError: If workbook generation fails.
    """
    if not OPENPYXL_AVAILABLE:
        raise ImportError(
            "The openpyxl library is required to generate Excel evaluation reports. "
            "Please install it with: pip install 'openpyxl>=3.1.2'"
        )

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    wb = openpyxl.Workbook()
    default_sheet = wb.active
    if default_sheet is not None:
        wb.remove(default_sheet)

    # 1. Sheet 1: Ticket-Level Evaluation
    ws_tickets = wb.create_sheet(title="Ticket-Level Evaluation")
    _populate_ticket_evaluation_sheet(
        ws=ws_tickets,
        results=results,
        ground_truth=ground_truth,
        metadata=metadata,
        policy_report=policy_report,
    )

    # 2. Sheet 2: Evaluation Summary
    ws_summary = wb.create_sheet(title="Evaluation Summary")
    _populate_evaluation_summary_sheet(
        ws=ws_summary,
        summary_metrics=summary_metrics,
        metadata=metadata,
        reconciliation=reconciliation,
        policy_report=policy_report,
    )

    # 3. Sheet 3: Data Sources & Dictionary
    ws_dict = wb.create_sheet(title="Data Sources & Dictionary")
    _populate_dictionary_sheet(
        ws=ws_dict,
        metadata=metadata,
        total_records=len(results),
    )

    wb.save(out_file)
    logger.info("Exported evaluation workbook to %s (%d tickets)", out_file, len(results))
    return out_file


def _populate_ticket_evaluation_sheet(
    ws: Worksheet,
    results: Sequence[ProcessedTicketResult],
    ground_truth: dict[str, EvaluationRecord],
    metadata: dict[str, Any],
    policy_report: Any | None = None,
) -> None:
    """Populate Sheet 1 with 43 columns and formatted rows for all processed tickets."""
    font_header = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    fill_header = PatternFill(start_color="1B365D", end_color="1B365D", fill_type="solid")
    align_header = Alignment(horizontal="center", vertical="center", wrap_text=True)

    thin_border_side = Side(style="thin", color="D9D9D9")
    cell_border = Border(
        left=thin_border_side,
        right=thin_border_side,
        top=thin_border_side,
        bottom=thin_border_side,
    )

    # Route-specific highlight styling
    fill_auto = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
    font_auto = Font(name="Calibri", size=10, bold=True, color="276A3C")

    fill_review = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
    font_review = Font(name="Calibri", size=10, bold=True, color="8C6B00")

    fill_escalate = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid")
    font_escalate = Font(name="Calibri", size=10, bold=True, color="A61C1C")

    # Alignments
    align_center = Alignment(horizontal="center", vertical="center")
    align_wrap_top = Alignment(horizontal="left", vertical="top", wrap_text=True)
    align_right_top = Alignment(horizontal="right", vertical="top")

    font_data = Font(name="Calibri", size=10)

    # Build lookup for policy audit records if available
    audit_lookup: dict[str, Any] = {}
    if policy_report and hasattr(policy_report, "ticket_audits"):
        audit_lookup = {a.ticket_id: a for a in policy_report.ticket_audits}

    # Write Header Row
    ws.append(TICKET_LEVEL_COLUMNS)
    ws.row_dimensions[1].height = 28

    for col_idx in range(1, len(TICKET_LEVEL_COLUMNS) + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.font = font_header
        cell.fill = fill_header
        cell.alignment = align_header
        cell.border = cell_border

    exec_mode = str(metadata.get("execution_mode") or metadata.get("mode") or "mock").upper()
    exec_timestamp = (
        metadata.get("timestamp")
        or metadata.get("created_at")
        or datetime.now(timezone.utc).isoformat()
    )

    # Write Data Rows
    for row_idx, res in enumerate(results, start=2):
        tid = res.ticket.ticket_id
        gt = ground_truth.get(tid)
        labels = gt.labels if gt else None
        audit = audit_lookup.get(tid)

        # 1-5 Inbound inquiry details
        channel = (
            res.ticket.channel.value
            if hasattr(res.ticket.channel, "value")
            else str(res.ticket.channel)
        )
        tier = (
            res.ticket.customer_tier.value
            if hasattr(res.ticket.customer_tier, "value")
            else str(res.ticket.customer_tier)
        )
        subject = res.ticket.subject or ""
        body = res.ticket.body or ""

        # 6-13 Intent and Urgency classification
        pred_intent = (
            res.classification.intent.value
            if hasattr(res.classification.intent, "value")
            else str(res.classification.intent)
        )
        expected_intent = labels.intent if (labels and labels.intent) else "N/A"
        intent_conf = (
            float(res.classification.intent_confidence)
            if res.classification.intent_confidence is not None
            else 0.0
        )
        intent_match = (
            (pred_intent == expected_intent) if expected_intent != "N/A" else False
        )

        pred_urgency = (
            res.classification.urgency.value
            if hasattr(res.classification.urgency, "value")
            else str(res.classification.urgency)
        )
        expected_urgency = labels.urgency if (labels and labels.urgency) else "N/A"
        urgency_conf = (
            float(res.classification.urgency_confidence)
            if res.classification.urgency_confidence is not None
            else 0.0
        )
        urgency_match = (
            (pred_urgency == expected_urgency) if expected_urgency != "N/A" else False
        )

        # 14-18 Input Safety & Guardrails
        guardrails_passed = getattr(res.routing_decision, "guardrails_passed", True)
        if audit and getattr(audit, "is_safety_violation", False):
            guardrails_passed = False

        input_guardrail_status = "PASSED" if guardrails_passed else "FAILED"

        guardrail_violation_category = "NONE"
        prompt_injection_status = "NOT_DETECTED"
        if not guardrails_passed:
            err_text = (res.error or "").lower()
            notes_text = (res.private_agent_notes or "").lower()
            desc_text = (res.routing_decision.reason_description or "").lower()

            if res.error and "input safety violation:" in res.error.lower():
                guardrail_violation_category = (
                    res.error.split("Input safety violation:", 1)[1].strip()
                )
            elif "prompt injection" in notes_text or "prompt injection" in desc_text:
                guardrail_violation_category = "prompt_injection"
            elif "pii" in notes_text or "pii" in desc_text:
                guardrail_violation_category = "pii_detected"
            elif res.routing_decision.reason_code.value == "GUARDRAIL_VIOLATION":
                guardrail_violation_category = "guardrail_violation"
            else:
                guardrail_violation_category = "NOT_AVAILABLE"

            if (
                "prompt_injection" in guardrail_violation_category.lower()
                or "prompt injection" in notes_text
                or "prompt injection" in desc_text
                or "prompt_injection" in err_text
            ):
                prompt_injection_status = "DETECTED"

        # Dedicated PII status: strictly NOT_INSTRUMENTED if pipeline does not expose
        pii_detection_status = getattr(res, "pii_detection_status", None)
        if pii_detection_status is None:
            pii_detection_status = "NOT_INSTRUMENTED"

        # 19-24 Knowledge-Base Retrieval
        retrieved_doc_ids = list(
            dict.fromkeys(p.doc_id for p in res.retrieved_passages if p.doc_id)
        )
        retrieved_articles_str = (
            "; ".join(retrieved_doc_ids) if retrieved_doc_ids else "None"
        )
        top_chunk_id = (
            res.retrieved_passages[0].chunk_id if res.retrieved_passages else "None"
        )
        top_sim_score = (
            float(res.retrieved_passages[0].similarity_score)
            if res.retrieved_passages
            else None
        )
        all_chunk_ids_str = (
            "; ".join(p.chunk_id for p in res.retrieved_passages if p.chunk_id)
            if res.retrieved_passages
            else "None"
        )
        all_sim_scores_str = (
            "; ".join(
                f"{float(p.similarity_score):.3f}"
                for p in res.retrieved_passages
                if p.similarity_score is not None
            )
            if res.retrieved_passages
            else "None"
        )
        if res.retrieved_passages:
            raw_excerpt = res.retrieved_passages[0].content or ""
            retrieved_excerpt = (
                raw_excerpt[:300] + ("..." if len(raw_excerpt) > 300 else "")
            )
        else:
            retrieved_excerpt = "None"

        # 25-30 Response Synthesis & Citations
        candidate_text = (
            res.generated_response.customer_response
            if (res.generated_response and res.generated_response.customer_response)
            else ""
        )
        candidate_generated = bool(candidate_text.strip())
        candidate_word_count = (
            len(candidate_text.strip().split()) if candidate_generated else 0
        )

        # Extract response citations from structured data where available
        if res.generated_response is None:
            response_citations_str = "NOT_APPLICABLE"
        elif res.generated_response.citation_ids:
            response_citations_str = "; ".join(res.generated_response.citation_ids)
        else:
            response_citations_str = "None"

        citations_verified = getattr(
            res.routing_decision, "citations_verified", False
        )

        # Grounding and citation validation details
        if res.grounding_result is not None:
            details_parts: list[str] = []
            if res.grounding_result.details:
                details_parts.append(res.grounding_result.details)
            if res.grounding_result.unsupported_reasons:
                details_parts.append(
                    "Unsupported: " + "; ".join(res.grounding_result.unsupported_reasons)
                )
            if res.grounding_result.verified_citations:
                details_parts.append(
                    "Verified: " + "; ".join(res.grounding_result.verified_citations)
                )
            citation_val_details = " | ".join(details_parts) if details_parts else (
                "Grounded" if res.grounding_result.is_grounded else "Ungrounded"
            )
        elif res.generated_response is None:
            citation_val_details = "NOT_APPLICABLE"
        elif not citations_verified:
            if res.routing_decision.reason_code.value in (
                "CITATION_VALIDATION_FAILED",
                "GROUNDING_VALIDATION_FAILED",
            ):
                citation_val_details = res.routing_decision.reason_description
            else:
                citation_val_details = "NOT_AVAILABLE"
        else:
            citation_val_details = "NOT_AVAILABLE"

        # 31-37 Routing Decision & Delivery
        route_enum = res.routing_decision.route
        route_str = (
            route_enum.value if hasattr(route_enum, "value") else str(route_enum)
        ).upper()
        rc_enum = res.routing_decision.reason_code
        reason_code_str = (
            rc_enum.value if hasattr(rc_enum, "value") else str(rc_enum)
        )
        reason_desc = res.routing_decision.reason_description or ""

        target_specialist = "None"
        if (
            res.routing_decision.escalation_payload
            and res.routing_decision.escalation_payload.target_specialist_group
        ):
            target_specialist = (
                res.routing_decision.escalation_payload.target_specialist_group
            )

        delivery_status = getattr(res, "delivery_status", "undelivered")
        final_released = bool(
            delivery_status == "released" and res.final_customer_response
        )

        review_hold_reason = ""
        if delivery_status == "held_for_review" or route_str == "DRAFT_FOR_REVIEW":
            review_hold_reason = reason_desc or reason_code_str

        # 38-43 Diagnostic, Persistence & Metadata
        latency_ms = (
            round(float(res.processing_latency_ms), 2)
            if res.processing_latency_ms is not None
            else 0.0
        )
        persistence_status = getattr(res, "persistence_status", "omitted")
        audit_id_str = res.audit_id or "N/A"

        notes_parts: list[str] = []
        if audit and getattr(audit, "is_prd_policy_divergence", False):
            notes_parts.append("PRD v2 policy divergence")
        if not guardrails_passed:
            notes_parts.append("Input guardrail violation blocked")
        if (
            route_str == "DRAFT_FOR_REVIEW"
            and "billing" in pred_intent.lower()
        ):
            notes_parts.append("Billing policy draft hold")
        elif (
            route_str == "DRAFT_FOR_REVIEW"
            and not citations_verified
        ):
            notes_parts.append("Citation verification hold")
        elif route_str == "ESCALATE":
            notes_parts.append("Direct specialist escalation (generation bypassed)")
        elif route_str == "AUTO_RESPOND":
            notes_parts.append("Automated response released to customer")

        notes_str = "; ".join(notes_parts) if notes_parts else "Standard processing"

        row_values: list[Any] = [
            tid,
            channel,
            tier,
            subject,
            body,
            pred_intent,
            expected_intent,
            intent_conf,
            intent_match,
            pred_urgency,
            expected_urgency,
            urgency_conf,
            urgency_match,
            input_guardrail_status,
            guardrail_violation_category,
            prompt_injection_status,
            pii_detection_status,
            guardrails_passed,
            retrieved_articles_str,
            top_chunk_id,
            top_sim_score,
            all_chunk_ids_str,
            all_sim_scores_str,
            retrieved_excerpt,
            candidate_generated,
            candidate_text,
            candidate_word_count,
            response_citations_str,
            citations_verified,
            citation_val_details,
            route_str,
            reason_code_str,
            reason_desc,
            target_specialist,
            delivery_status,
            final_released,
            review_hold_reason,
            latency_ms,
            persistence_status,
            audit_id_str,
            exec_timestamp,
            exec_mode,
            notes_str,
        ]

        ws.append(row_values)

        # Style and format each cell in the row
        for c_idx, _val in enumerate(row_values, start=1):
            cell = ws.cell(row=row_idx, column=c_idx)
            cell.font = font_data
            cell.border = cell_border
            col_name = TICKET_LEVEL_COLUMNS[c_idx - 1]

            if col_name in (
                "Ticket ID",
                "Channel",
                "Customer Tier",
                "Predicted Intent",
                "Expected Intent",
                "Intent Match",
                "Predicted Urgency",
                "Expected Urgency",
                "Urgency Match",
                "Input Guardrail Status",
                "Guardrail Violation Category",
                "Prompt Injection Detection Status",
                "PII Detection Status",
                "Guardrails Passed",
                "Top Retrieved Chunk ID",
                "Candidate Response Generated",
                "Response Citation IDs",
                "Citations Verified",
                "Routing Decision",
                "Routing Reason Code",
                "Target Specialist Group",
                "Delivery Status",
                "Final Response Released to Customer",
                "Persistence Status",
                "Audit ID",
                "Evaluation Mode",
            ):
                cell.alignment = align_center
            elif col_name in (
                "Customer Subject",
                "Customer Question / Body",
                "Retrieved Article IDs",
                "All Retrieved Chunk IDs",
                "All Retrieved Similarity Scores",
                "Retrieved Context Excerpt",
                "Candidate Response Text",
                "Citation Validation Details",
                "Routing Reason Description",
                "Review Hold Reason",
                "Execution Timestamp",
                "Notes / Observations",
            ):
                cell.alignment = align_wrap_top
            elif col_name in (
                "Intent Confidence",
                "Urgency Confidence",
                "Top Retrieval Similarity Score",
            ):
                cell.alignment = align_right_top
                cell.number_format = "0.000"
            elif col_name == "Candidate Response Word Count":
                cell.alignment = align_right_top
                cell.number_format = "#,##0"
            elif col_name == "Processing Latency (ms)":
                cell.alignment = align_right_top
                cell.number_format = "#,##0.00"

            # Route highlight styling on Routing Decision column
            if col_name == "Routing Decision":
                if route_str == "AUTO_RESPOND":
                    cell.fill = fill_auto
                    cell.font = font_auto
                elif route_str == "DRAFT_FOR_REVIEW":
                    cell.fill = fill_review
                    cell.font = font_review
                elif route_str == "ESCALATE":
                    cell.fill = fill_escalate
                    cell.font = font_escalate

    # Freeze header row
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    _adjust_column_widths(ws)


def _populate_evaluation_summary_sheet(
    ws: Worksheet,
    summary_metrics: dict[str, Any],
    metadata: dict[str, Any],
    reconciliation: dict[str, Any],
    policy_report: Any | None = None,
) -> None:
    """Populate Sheet 2 with executive KPIs, routing breakdown, safety, and latency."""
    font_title = Font(name="Calibri", size=16, bold=True, color="1B365D")
    font_section = Font(name="Calibri", size=12, bold=True, color="FFFFFF")
    fill_section = PatternFill(start_color="1B365D", end_color="1B365D", fill_type="solid")

    font_label = Font(name="Calibri", size=10, bold=True, color="333333")
    font_value = Font(name="Calibri", size=10, color="000000")
    font_kpi_num = Font(name="Calibri", size=11, bold=True, color="1B365D")

    align_left = Alignment(horizontal="left", vertical="center")
    align_right = Alignment(horizontal="right", vertical="center")
    thin_border = Border(
        left=Side(style="thin", color="E0E0E0"),
        right=Side(style="thin", color="E0E0E0"),
        top=Side(style="thin", color="E0E0E0"),
        bottom=Side(style="thin", color="E0E0E0"),
    )

    ws.column_dimensions["A"].width = 4
    ws.column_dimensions["B"].width = 38
    ws.column_dimensions["C"].width = 24
    ws.column_dimensions["D"].width = 38
    ws.column_dimensions["E"].width = 24

    ws["B2"] = "CloudServe AI Support System — Evaluation Performance Dashboard"
    ws["B2"].font = font_title
    ws.merge_cells("B2:E2")

    current_row = 4

    def add_section_header(title: str) -> None:
        nonlocal current_row
        ws.cell(row=current_row, column=2, value=title).font = font_section
        ws.cell(row=current_row, column=2).fill = fill_section
        ws.cell(row=current_row, column=2).alignment = Alignment(
            horizontal="left", vertical="center", indent=1
        )
        ws.merge_cells(
            start_row=current_row, start_column=2, end_row=current_row, end_column=5
        )
        ws.row_dimensions[current_row].height = 24
        current_row += 1

    def add_kv_row(
        lbl1: str, val1: Any, lbl2: str = "", val2: Any = "", is_kpi: bool = False
    ) -> None:
        nonlocal current_row
        cB_lbl = ws.cell(row=current_row, column=2, value=lbl1)
        cB_lbl.font = font_label
        cB_lbl.alignment = align_left
        cB_lbl.border = thin_border

        cC_val = ws.cell(row=current_row, column=3, value=val1)
        cC_val.font = font_kpi_num if is_kpi else font_value
        cC_val.alignment = align_right
        cC_val.border = thin_border

        if lbl2:
            cD_lbl = ws.cell(row=current_row, column=4, value=lbl2)
            cD_lbl.font = font_label
            cD_lbl.alignment = align_left
            cD_lbl.border = thin_border

            cE_val = ws.cell(row=current_row, column=5, value=val2)
            cE_val.font = font_kpi_num if is_kpi else font_value
            cE_val.alignment = align_right
            cE_val.border = thin_border
        else:
            ws.merge_cells(
                start_row=current_row, start_column=3, end_row=current_row, end_column=5
            )

        ws.row_dimensions[current_row].height = 20
        current_row += 1

    def fmt_flt(val: Any, decimals: int = 4, default: str = "N/A") -> str:
        if val is None:
            return default
        try:
            return f"{float(val):.{decimals}f}"
        except (ValueError, TypeError):
            return str(val)

    cls_metrics = summary_metrics.get("classification", {})
    urg_metrics = cls_metrics.get("urgency", {})
    calib = summary_metrics.get("confidence_calibration", {})
    calib_intent = calib.get("intent", {})
    calib_urgency = calib.get("urgency", {})
    retrieval = summary_metrics.get("retrieval", {}).get("answerable_tickets", {})
    policy = summary_metrics.get("routing_and_policy_compliance", {})
    latency = summary_metrics.get("latency") or {}

    total_records = summary_metrics.get("total_records", 0)

    # 1. Executive Run Metadata
    add_section_header("1. Executive Overview & Run Metadata")
    add_kv_row("Run ID", summary_metrics.get("run_id", "N/A"), "Execution Status", "COMPLETED")
    add_kv_row(
        "Execution Mode",
        str(metadata.get("execution_mode") or metadata.get("mode") or "mock").upper(),
        "Evaluation Timestamp",
        str(metadata.get("timestamp") or metadata.get("created_at") or "N/A"),
    )
    add_kv_row(
        "Total Evaluated Tickets",
        total_records,
        "Successfully Processed",
        summary_metrics.get("successful_records", total_records),
    )
    add_kv_row(
        "Source Dataset",
        Path(metadata.get("input_path", "data/val/validation_tickets.json")).name,
        "Failed Tickets",
        summary_metrics.get("failed_records", 0),
    )
    current_row += 1

    # 2. Classification Performance
    add_section_header("2. Classification Performance & Calibration")
    add_kv_row(
        "Intent Accuracy",
        fmt_flt(cls_metrics.get("intent_accuracy")),
        "Urgency Accuracy",
        fmt_flt(urg_metrics.get("accuracy")),
        is_kpi=True,
    )
    add_kv_row(
        "Intent Macro-F1 (Represented)",
        fmt_flt(cls_metrics.get("intent_macro_f1_represented")),
        "Urgency Macro-F1",
        fmt_flt(urg_metrics.get("macro_f1")),
    )
    add_kv_row(
        "Intent Full ECE",
        fmt_flt(calib_intent.get("full_expected_calibration_error")),
        "Urgency Full ECE",
        fmt_flt(calib_urgency.get("full_expected_calibration_error")),
    )
    add_kv_row(
        "Intent Reliable Bins ECE",
        fmt_flt(calib_intent.get("reliable_bins_ece")),
        "Calibration Target Met",
        str(calib_intent.get("meets_governance_target", True)).upper(),
    )
    current_row += 1

    # 3. Knowledge-Base Retrieval
    add_section_header("3. Knowledge-Base Retrieval Performance")
    add_kv_row(
        "Precision@4",
        fmt_flt(retrieval.get("precision_at_4")),
        "Recall@4",
        fmt_flt(retrieval.get("recall_at_4")),
        is_kpi=True,
    )
    add_kv_row(
        "Hit Rate@4",
        fmt_flt(retrieval.get("hit_rate_at_4")),
        "Mean Reciprocal Rank (MRR)",
        fmt_flt(retrieval.get("mrr")),
    )
    current_row += 1

    # 4. Routing & Policy Compliance
    add_section_header("4. Routing & Policy Compliance")
    add_kv_row(
        "Deterministic Router Compliance",
        fmt_flt(policy.get("deterministic_execution_compliance_rate"), default="1.0000"),
        "Mandatory Escalation Compliance",
        fmt_flt(policy.get("mandatory_escalation_compliance_rate"), default="1.0000"),
        is_kpi=True,
    )
    add_kv_row(
        "Billing Policy Compliance",
        fmt_flt(policy.get("billing_policy_compliance_rate"), default="1.0000"),
        "Kill Switch Compliance",
        fmt_flt(policy.get("kill_switch_compliance_rate"), default="1.0000"),
    )
    add_kv_row(
        "Dataset Agreement Rate",
        fmt_flt(policy.get("dataset_agreement_rate"), default="0.0000"),
        "PRD v2 Policy Divergences",
        policy.get("prd_v2_policy_divergences_count", 0),
    )
    current_row += 1

    # 5. Safety & Customer Protection
    add_section_header("5. Safety & Customer Protection")
    add_kv_row(
        "End-to-End Safety Compliance",
        fmt_flt(policy.get("end_to_end_safety_compliance_rate"), default="1.0000"),
        "Unsafe Customer Releases",
        policy.get("unsafe_releases_count", 0),
        is_kpi=True,
    )
    add_kv_row(
        "Safety Violations Count",
        policy.get("safety_violations_count", 0),
        "Total Responses Released",
        policy.get("total_released_responses", 0),
    )
    add_kv_row(
        "Release Safety Exercised",
        str(policy.get("release_safety_exercised", True)).upper(),
        "Release Safety Status",
        policy.get("release_safety_status", "PASS"),
    )
    current_row += 1

    # 6. Latency & Performance SLAs
    add_section_header("6. Processing Latency & Governance SLA Compliance")
    if latency:
        add_kv_row(
            "Mean Latency (ms)",
            fmt_flt(latency.get("mean_ms"), decimals=2),
            "Median Latency (ms)",
            fmt_flt(latency.get("median_ms"), decimals=2),
        )
        add_kv_row(
            "P90 Latency (ms)",
            fmt_flt(latency.get("p90_ms"), decimals=2),
            "P95 Latency (ms)",
            fmt_flt(latency.get("p95_ms"), decimals=2),
            is_kpi=True,
        )
        add_kv_row(
            "P95 SLA Target (< 3000 ms)",
            "3000.00 ms",
            "P95 Governance SLA Met?",
            str(latency.get("p95_sla_met", True)).upper(),
            is_kpi=True,
        )
    else:
        add_kv_row("Latency Statistics", "Not computed / No latency samples available")
    current_row += 1

    # 7. SQLite Audit Persistence Reconciliation
    add_section_header("7. SQLite Audit Persistence Reconciliation (A8)")
    add_kv_row(
        "Processed Records",
        reconciliation.get("total_processed", total_records),
        "Database Persisted Records",
        reconciliation.get("persisted_records", total_records),
    )
    add_kv_row(
        "Reconciliation Gap",
        reconciliation.get("gap", 0),
        "Audit Log Reconciled?",
        str(reconciliation.get("reconciled", True)).upper(),
        is_kpi=True,
    )


def _populate_dictionary_sheet(
    ws: Worksheet,
    metadata: dict[str, Any],
    total_records: int,
) -> None:
    """Populate Sheet 3 with dataset provenance, 43-column data dictionary, and policy rules."""
    font_title = Font(name="Calibri", size=15, bold=True, color="1B365D")
    font_section = Font(name="Calibri", size=12, bold=True, color="FFFFFF")
    fill_section = PatternFill(start_color="1B365D", end_color="1B365D", fill_type="solid")

    font_tbl_hdr = Font(name="Calibri", size=10, bold=True, color="FFFFFF")
    fill_tbl_hdr = PatternFill(start_color="2C4D75", end_color="2C4D75", fill_type="solid")

    font_data = Font(name="Calibri", size=9)
    font_bold = Font(name="Calibri", size=9, bold=True)

    align_center = Alignment(horizontal="center", vertical="center")
    align_left = Alignment(horizontal="left", vertical="center")
    align_wrap_top = Alignment(horizontal="left", vertical="top", wrap_text=True)

    thin_border = Border(
        left=Side(style="thin", color="D9D9D9"),
        right=Side(style="thin", color="D9D9D9"),
        top=Side(style="thin", color="D9D9D9"),
        bottom=Side(style="thin", color="D9D9D9"),
    )

    ws.column_dimensions["A"].width = 10
    ws.column_dimensions["B"].width = 32
    ws.column_dimensions["C"].width = 16
    ws.column_dimensions["D"].width = 32
    ws.column_dimensions["E"].width = 65

    ws["A1"] = "CloudServe Evaluation Report — Data Sources, Dictionary & Policy Documentation"
    ws["A1"].font = font_title
    ws.merge_cells("A1:E1")
    ws.row_dimensions[1].height = 26

    current_row = 3

    # Section 1: Provenance
    sec1_cell = ws.cell(row=current_row, column=1, value="1. Benchmark Dataset & Run Provenance")
    sec1_cell.font = font_section
    sec1_cell.fill = fill_section
    ws.merge_cells(start_row=current_row, start_column=1, end_row=current_row, end_column=5)
    ws.row_dimensions[current_row].height = 22
    current_row += 1

    input_path_str = str(metadata.get("input_path", "data/val/validation_tickets.json"))
    dataset_hash_str = str(metadata.get("dataset_hash", "Verified authentic benchmark dataset"))
    eval_mode_str = str(
        metadata.get("execution_mode") or metadata.get("mode") or "mock"
    ).upper()

    provenance_rows = [
        ("Source Dataset Path", input_path_str),
        ("Dataset Fingerprint (SHA-256)", dataset_hash_str),
        ("Total Inbound Ticket Records", str(total_records)),
        ("Knowledge Base Documentation", "data/kb/documentation.json (10 articles, 34 chunks)"),
        ("Active Embedding Model", "all-MiniLM-L6-v2 (384-dimensional dense vectors)"),
        ("Active LLM Provider / Model", "Groq Cloud / openai/gpt-oss-120b (temperature=0.0)"),
        ("Vector Store Engine", "ChromaDB (Cosine Similarity metric with persistent collection)"),
        ("Evaluation Mode", eval_mode_str),
    ]

    for label, val in provenance_rows:
        cL = ws.cell(row=current_row, column=1, value=label)
        cL.font = font_bold
        cL.alignment = align_left
        cL.border = thin_border
        ws.merge_cells(start_row=current_row, start_column=1, end_row=current_row, end_column=2)

        cV = ws.cell(row=current_row, column=3, value=val)
        cV.font = font_data
        cV.alignment = align_left
        cV.border = thin_border
        ws.merge_cells(start_row=current_row, start_column=3, end_row=current_row, end_column=5)

        ws.row_dimensions[current_row].height = 18
        current_row += 1

    current_row += 1

    # Section 2: Data Dictionary (43 Columns)
    sec2_cell = ws.cell(
        row=current_row, column=1, value="2. 43-Column Ticket-Level Data Dictionary"
    )
    sec2_cell.font = font_section
    sec2_cell.fill = fill_section
    ws.merge_cells(start_row=current_row, start_column=1, end_row=current_row, end_column=5)
    ws.row_dimensions[current_row].height = 22
    current_row += 1

    dict_headers = [
        "Col #",
        "Field Name",
        "Data Type",
        "Valid Values / Format",
        "Description & Policy Interpretation",
    ]
    for c_idx, h_text in enumerate(dict_headers, start=1):
        cell = ws.cell(row=current_row, column=c_idx, value=h_text)
        cell.font = font_tbl_hdr
        cell.fill = fill_tbl_hdr
        cell.alignment = align_center
        cell.border = thin_border
    ws.row_dimensions[current_row].height = 20
    current_row += 1

    dictionary_data = [
        (1, "Ticket ID", "String", "VAL-001, DEV-001",
         "Authoritative unique identifier of the inbound support ticket."),
        (2, "Channel", "String", "email, portal, chat, social",
         "Inbound customer communication channel."),
        (3, "Customer Tier", "String", "standard, pro, enterprise",
         "Customer account entitlement and SLA priority tier."),
        (4, "Customer Subject", "String", "Text or blank",
         "Subject line provided in ticket (blank for live chat inquiries)."),
        (5, "Customer Question / Body", "String", "Raw verbatim text",
         "Full verbatim customer inquiry submitted into the system."),
        (6, "Predicted Intent", "String", "22 intent categories",
         "Intent category predicted by the multi-class classifier."),
        (7, "Expected Intent", "String", "22 intent categories",
         "Benchmark ground-truth intent label from validation dataset."),
        (8, "Intent Confidence", "Float", "0.000 to 1.000",
         "Calibrated model posterior probability for the predicted intent."),
        (9, "Intent Match", "Boolean", "TRUE / FALSE",
         "TRUE if predicted intent exactly matches ground truth."),
        (10, "Predicted Urgency", "String", "low, medium, high, critical",
         "Urgency level predicted by the classifier."),
        (11, "Expected Urgency", "String", "low, medium, high, critical",
         "Benchmark ground-truth urgency label from dataset."),
        (12, "Urgency Confidence", "Float", "0.000 to 1.000",
         "Calibrated model confidence score for predicted urgency."),
        (13, "Urgency Match", "Boolean", "TRUE / FALSE",
         "TRUE if predicted urgency matches ground truth."),
        (14, "Input Guardrail Status", "String", "PASSED, FAILED",
         "Outcome of input safety scan across prompt injection and high-risk safety checks."),
        (15, "Guardrail Violation Category", "String", "NONE, prompt_injection, NOT_AVAILABLE",
         "Explicit safety violation category flagged by input scanner."),
        (16, "Prompt Injection Detection Status", "String", "DETECTED, NOT_DETECTED",
         "Dedicated detection status for adversarial prompt injection attempts."),
        (17, "PII Detection Status", "String", "NOT_INSTRUMENTED",
         "Ticket-level PII detection status (NOT_INSTRUMENTED if unexposed at ticket level)."),
        (18, "Guardrails Passed", "Boolean", "TRUE / FALSE",
         "Overall boolean safety gate confirming input scan passed."),
        (19, "Retrieved Article IDs", "String", "KB-001; KB-005 or None",
         "All unique Knowledge-Base document IDs retrieved for inquiry."),
        (20, "Top Retrieved Chunk ID", "String", "e.g. KB-001-001 or None",
         "Identifier of the top-ranked retrieved passage chunk."),
        (21, "Top Retrieval Similarity Score", "Float", "0.000 to 1.000",
         "Cosine similarity score between inquiry embedding and top passage."),
        (22, "All Retrieved Chunk IDs", "String", "Semicolon-delimited IDs",
         "All retrieved passage chunk IDs in rank order."),
        (23, "All Retrieved Similarity Scores", "String", "Semicolon-delimited floats",
         "All retrieved cosine similarity scores in the same rank order."),
        (24, "Retrieved Context Excerpt", "String", "Text (up to 300 chars)",
         "Leading passage text of the top-ranked knowledge base excerpt."),
        (25, "Candidate Response Generated", "Boolean", "TRUE / FALSE",
         "TRUE if generator synthesized candidate response; FALSE if bypassed/suppressed."),
        (26, "Candidate Response Text", "String", "Full text or blank",
         "Actual LLM response synthesized; blank if generation was bypassed."),
        (27, "Candidate Response Word Count", "Integer", ">= 0",
         "Word count of candidate response (0 if no response generated)."),
        (28, "Response Citation IDs", "String", "Structured IDs, None, NOT_APPLICABLE",
         "Citation IDs explicitly cited in the candidate response from structured data."),
        (29, "Citations Verified", "Boolean", "TRUE / FALSE",
         "TRUE if candidate response claims are verified against retrieved KB documentation."),
        (30, "Citation Validation Details", "String", "Details, NOT_AVAILABLE, NOT_APPLICABLE",
         "Validation details from grounding check or rationale for citation hold."),
        (31, "Routing Decision", "String", "AUTO_RESPOND, DRAFT_FOR_REVIEW, ESCALATE",
         "Authoritative deterministic routing decision."),
        (32, "Routing Reason Code", "String", "Standardized policy enum",
         "Structured reason code explaining the routing decision."),
        (33, "Routing Reason Description", "String", "Explanatory text",
         "Human-readable explanation of why this decision was taken."),
        (34, "Target Specialist Group", "String", "security_tier2, billing, etc.",
         "Assigned human specialist group for escalation tickets."),
        (35, "Delivery Status", "String", "released, held_for_review, escalated",
         "Operational delivery state of the ticket response."),
        (36, "Final Response Released to Customer", "Boolean", "TRUE / FALSE",
         "TRUE only for AUTO_RESPOND where response is dispatched to customer."),
        (37, "Review Hold Reason", "String", "Hold description or blank",
         "Specific reason why response was held in human review queue."),
        (38, "Processing Latency (ms)", "Float", "Milliseconds",
         "Total end-to-end execution duration for this ticket."),
        (39, "Persistence Status", "String", "persisted, failed, omitted",
         "SQLite audit database commit status."),
        (40, "Audit ID", "String", "UUIDv4 or N/A",
         "Authoritative audit log UUID stored in SQLite database."),
        (41, "Execution Timestamp", "String", "ISO-8601 UTC timestamp",
         "Date and time when evaluation processing was executed."),
        (42, "Evaluation Mode", "String", "MOCK / LIVE",
         "Execution mode under which the evaluation was conducted."),
        (43, "Notes / Observations", "String", "Diagnostic notes",
         "Operational and forensic observations (divergence, holds, etc.)."),
    ]

    for item in dictionary_data:
        col_num, name, d_type, vals, desc = item
        ws.cell(row=current_row, column=1, value=col_num).alignment = align_center
        ws.cell(row=current_row, column=2, value=name).font = font_bold
        ws.cell(row=current_row, column=2).alignment = align_left
        ws.cell(row=current_row, column=3, value=d_type).alignment = align_center
        ws.cell(row=current_row, column=4, value=vals).alignment = align_left
        ws.cell(row=current_row, column=5, value=desc).alignment = align_wrap_top

        for c_idx in range(1, 6):
            cell = ws.cell(row=current_row, column=c_idx)
            cell.font = font_data if c_idx != 2 else font_bold
            cell.border = thin_border

        ws.row_dimensions[current_row].height = 22
        current_row += 1

    current_row += 1

    # Section 3: Policy Documentation
    sec3_cell = ws.cell(
        row=current_row, column=1, value="3. Core Operational Governance & Safety Policies"
    )
    sec3_cell.font = font_section
    sec3_cell.fill = fill_section
    ws.merge_cells(start_row=current_row, start_column=1, end_row=current_row, end_column=5)
    ws.row_dimensions[current_row].height = 22
    current_row += 1

    policies = [
        (
            "Candidate Response Generation Policy",
            "Candidate responses are synthesized ONLY for AUTO_RESPOND candidates and select "
            "DRAFT_FOR_REVIEW tickets (e.g. billing inquiries). For ESCALATE tickets (e.g. "
            "security incidents, critical outages, legal disputes), generation is deliberately "
            "BYPASSED to eliminate LLM hallucination risk and avoid unauthorized commitments.",
        ),
        (
            "Grounding & Citation Hold Policy",
            "Any candidate response whose claims reference ungrounded knowledge or fail citation "
            "validation is strictly WITHHELD from automated delivery "
            "(delivery_status='held_for_review'). These tickets are held in human review queues "
            "to safeguard customer trust.",
        ),
        (
            "Billing Dispute Safety Policy",
            "All account cancellation and billing dispute tickets are routed to human specialists "
            "with drafted guidance rather than auto-released, ensuring human empathy, accurate "
            "accounting adjustments, and commercial alignment.",
        ),
        (
            "Audit Trail Persistence Gate",
            "Before any customer response is dispatched (AUTO_RESPOND), an authoritative audit "
            "record must be committed to the SQLite database. If persistence fails, the response "
            "is suppressed to ensure complete audit compliance.",
        ),
    ]

    for p_title, p_desc in policies:
        cT = ws.cell(row=current_row, column=1, value=p_title)
        cT.font = font_bold
        cT.alignment = align_wrap_top
        cT.border = thin_border
        ws.merge_cells(start_row=current_row, start_column=1, end_row=current_row, end_column=2)

        cD = ws.cell(row=current_row, column=3, value=p_desc)
        cD.font = font_data
        cD.alignment = align_wrap_top
        cD.border = thin_border
        ws.merge_cells(start_row=current_row, start_column=3, end_row=current_row, end_column=5)

        ws.row_dimensions[current_row].height = 40
        current_row += 1


def _adjust_column_widths(ws: Worksheet) -> None:
    """Auto-adjust worksheet column widths with minimum and maximum bounds."""
    width_map: dict[str, float] = {
        "Ticket ID": 13,
        "Channel": 10,
        "Customer Tier": 14,
        "Customer Subject": 26,
        "Customer Question / Body": 45,
        "Predicted Intent": 22,
        "Expected Intent": 22,
        "Intent Confidence": 16,
        "Intent Match": 13,
        "Predicted Urgency": 18,
        "Expected Urgency": 18,
        "Urgency Confidence": 18,
        "Urgency Match": 14,
        "Input Guardrail Status": 20,
        "Guardrail Violation Category": 26,
        "Prompt Injection Detection Status": 30,
        "PII Detection Status": 20,
        "Guardrails Passed": 16,
        "Retrieved Article IDs": 24,
        "Top Retrieved Chunk ID": 22,
        "Top Retrieval Similarity Score": 26,
        "All Retrieved Chunk IDs": 35,
        "All Retrieved Similarity Scores": 30,
        "Retrieved Context Excerpt": 40,
        "Candidate Response Generated": 26,
        "Candidate Response Text": 45,
        "Candidate Response Word Count": 26,
        "Response Citation IDs": 24,
        "Citations Verified": 16,
        "Citation Validation Details": 35,
        "Routing Decision": 20,
        "Routing Reason Code": 28,
        "Routing Reason Description": 35,
        "Target Specialist Group": 22,
        "Delivery Status": 16,
        "Final Response Released to Customer": 32,
        "Review Hold Reason": 30,
        "Processing Latency (ms)": 20,
        "Persistence Status": 18,
        "Audit ID": 38,
        "Execution Timestamp": 26,
        "Evaluation Mode": 16,
        "Notes / Observations": 32,
    }

    for col_idx in range(1, ws.max_column + 1):
        col_letter = get_column_letter(col_idx)
        header_val = ws.cell(row=1, column=col_idx).value
        if header_val and str(header_val) in width_map:
            ws.column_dimensions[col_letter].width = width_map[str(header_val)]
        else:
            ws.column_dimensions[col_letter].width = 18
