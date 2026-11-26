"""Command-line interface for the CloudServe unattended evaluation runner.

Usage:
    python scripts/evaluate.py --input data/dev/development_tickets.json \
        --output-dir reports/eval_run --mode mock
    python scripts/evaluate.py --input data/dev/development_tickets.json \
        --output-dir reports/eval_run --mode live --overwrite
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from cloudserve.evaluation.runner import EvaluationRunner
from cloudserve.generation.llm_client import ConfigurationError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("cloudserve.evaluate")


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run unattended evaluation of CloudServe support system."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input benchmark dataset JSON file (e.g. development_tickets.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write run artifacts (run_metadata.json, audit_reconciliation.json).",
    )
    parser.add_argument(
        "--mode",
        choices=["mock", "live"],
        default="mock",
        help=(
            "Evaluation mode: 'mock' (deterministic, offline) or 'live' (Groq + MiniLM). "
            "Default: mock."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Allow writing into an existing non-empty output directory.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help=(
            "Optional authoritative UUID for this evaluation run. "
            "If omitted, a UUIDv4 is generated."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume an existing or interrupted evaluation run from its checkpoints.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        default=False,
        help="In resume mode, re-attempt tickets that previously ended with a failure.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        default=False,
        help="Display checkpoint status summary for output-dir and exit without processing.",
    )
    parser.add_argument(
        "--ground-truth-responses",
        "--reference-responses",
        dest="reference_responses",
        type=Path,
        default=None,
        help=(
            "Optional path to human reference responses JSON (default: "
            "data/val/ground_truth_responses.json if exists)."
        ),
    )
    parser.add_argument(
        "--no-excel",
        action="store_true",
        default=False,
        help="Skip automatic generation of CloudServe_Ticket_Level_Evaluation_Report.xlsx.",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parsed = parse_args(args)

    if not parsed.input.exists():
        logger.error("Input dataset file not found: %s", parsed.input)
        return 1

    # Check if --status was requested
    if parsed.status:
        out_meta = parsed.output_dir / "run_metadata.json"
        if not out_meta.exists():
            print(f"No evaluation metadata found at: {out_meta}")
            return 1
        with open(out_meta, encoding="utf-8") as f:
            meta = json.load(f)
        run_id = meta.get("run_id", "unknown")
        print("\n" + "=" * 68)
        print(" Evaluation Run Status Inspection")
        print("=" * 68)
        print(f" Run ID:             {run_id}")
        print(f" Status:             {meta.get('status', 'unknown').upper()}")
        print(f" Execution Mode:     {meta.get('execution_mode', 'unknown')}")
        print(f" Total Records:      {meta.get('total_records', 'unknown')}")
        print(f" Processed:          {meta.get('processed_records', 'unknown')}")
        print(f" Failed Tickets:     {meta.get('failed_tickets', 'unknown')}")
        print(f" Paused for Quota:   {meta.get('paused_for_quota', False)}")
        if meta.get("quota_error"):
            print(f" Quota Error:        {meta.get('quota_error')}")
        print("=" * 68 + "\n")
        return 0

    # Configure optional Excel generation flag
    if parsed.no_excel:
        os.environ["CLOUDSERVE_NO_EXCEL"] = "1"
    else:
        os.environ.pop("CLOUDSERVE_NO_EXCEL", None)

    logger.info("Initializing EvaluationRunner (mode=%s)...", parsed.mode)
    runner = EvaluationRunner()

    try:
        result = runner.run(
            input_path=parsed.input,
            output_dir=parsed.output_dir,
            mode=parsed.mode,
            overwrite=parsed.overwrite,
            run_id=parsed.run_id,
            resume=parsed.resume,
            retry_failed=parsed.retry_failed,
            reference_responses_path=parsed.reference_responses,
        )
    except FileExistsError as exc:
        logger.error("Output directory protection triggered: %s", exc)
        return 1
    except ConfigurationError as exc:
        logger.error("Live mode configuration error: %s", exc)
        return 1
    except RuntimeError as exc:
        logger.error("Runtime initialization error: %s", exc)
        return 1
    except Exception as exc:
        logger.exception("Evaluation runner failed unexpectedly: %s", exc)
        return 1

    # Log summary table of the evaluation run
    sm = getattr(result, "summary_metrics", {})
    cls_sm = sm.get("classification", {})
    cal_sm = sm.get("confidence_calibration", {}).get("intent", {})
    ret_sm = sm.get("retrieval", {}).get("answerable_tickets", {})
    pol_sm = sm.get("routing_and_policy_compliance", {})

    print("\n" + "=" * 68)
    print(" CloudServe Unattended Evaluation Run Summary")
    print("=" * 68)
    print(f" Run ID:             {result.run_id}")
    print(f" Execution Mode:     {result.metadata.get('execution_mode')}")
    print(f" Run Status:         {result.status.upper()}")
    print(f" Total Processed:    {len(result.results)}")
    print(f" Persisted (A8):     {result.reconciliation.get('persisted_records')}")
    print(f" Reconciled:         {result.reconciliation.get('reconciled')}")
    print("-" * 68)
    print(f" Intent Accuracy:    {cls_sm.get('intent_accuracy', 'N/A')}")
    print(f" Macro-F1 (Rep):     {cls_sm.get('intent_macro_f1_represented', 'N/A')}")
    print(f" Macro-F1 (Fixed):   {cls_sm.get('intent_macro_f1_fixed_22', 'N/A')}")
    print(f" Intent Full ECE:    {cal_sm.get('full_expected_calibration_error', 'N/A')}")
    print(f" Target Status:      {cal_sm.get('governance_target_status', 'N/A')}")
    print(f" Retrieval P@4:      {ret_sm.get('precision_at_4', 'N/A')}")
    print(f" Retrieval R@4:      {ret_sm.get('recall_at_4', 'N/A')}")
    print(f" Retrieval Hit@4:    {ret_sm.get('hit_rate_at_4', 'N/A')}")
    print(f" Retrieval MRR:      {ret_sm.get('mrr', 'N/A')}")
    print(f" Router Compliance:  {pol_sm.get('deterministic_execution_compliance_rate', 'N/A')}")
    print(f" Customer Releases:  {pol_sm.get('total_released_responses', 'N/A')}")
    print(f" Unsafe Releases:    {pol_sm.get('unsafe_releases_count', 'N/A')}")
    print(f" Release Exercised:  {pol_sm.get('release_safety_exercised', 'N/A')}")
    safe_rate_val = pol_sm.get("end_to_end_safety_compliance_rate")
    safe_rate_disp = f"{safe_rate_val:.4f}" if safe_rate_val is not None else "N/A (0 releases)"
    print(f" E2E Safety Rate:    {safe_rate_disp}")
    print(f" Safety Violations:  {pol_sm.get('safety_violations_count', 'N/A')}")
    print("-" * 68)
    print(f" Output Directory:   {result.output_dir}")
    excel_path = result.output_dir / "CloudServe_Ticket_Level_Evaluation_Report.xlsx"
    if excel_path.exists():
        print(f" Excel Report:       {excel_path}")
    print(" Generated Artifacts:")
    print("   - run_metadata.json")
    print("   - audit_reconciliation.json")
    print("   - summary_metrics.json")
    print("   - per_ticket_results.jsonl")
    print("   - per_ticket_results.csv")
    print("   - human_review_packet.json")
    print("   - human_review_template.csv")
    print("   - human_review_metadata.json")
    print("   - EVALUATION_REPORT.md")
    if excel_path.exists():
        print("   - CloudServe_Ticket_Level_Evaluation_Report.xlsx")
    print("=" * 68 + "\n")

    if result.status == "failed":
        logger.error("Evaluation run finished with status FAILED.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
