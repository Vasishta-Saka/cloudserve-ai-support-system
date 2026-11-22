"""Evaluation runner core for unattended evaluation runs.

Implements Acceptance Criteria A9, A10, and A11:
- Loads arbitrary evaluation datasets with benchmark labels and historical outcomes.
- Enforces strict data isolation: extracts runtime UnifiedTicket objects and passes ONLY
  runtime tickets to TicketProcessor.
- Keeps ground-truth labels outside the runtime execution pipeline.
- Coordinates batch processing across arbitrary output directories with overwrite protection.
- Emits run_metadata.json and audit_reconciliation.json with strict data minimization.
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from cloudserve.classification.models import (
    ClassificationResult,
    IntentCategory,
    UrgencyLevel,
)
from cloudserve.config import Settings, get_settings
from cloudserve.evaluation.factory import EvaluationComponentFactory
from cloudserve.evaluation.reporter import generate_evaluation_artifacts
from cloudserve.generation.llm_client import BaseLLMClient, ProviderQuotaExhaustedError
from cloudserve.ingestion.models import ChannelEnum, EvaluationRecord, UnifiedTicket
from cloudserve.ingestion.parsers import parse_evaluation_record
from cloudserve.routing.billing import assess_billing_risk
from cloudserve.routing.models import GuardrailCheckResult, RouteOutcome
from cloudserve.service import ProcessedTicketResult
from cloudserve.storage.repository import DecisionAuditRepository

logger = logging.getLogger(__name__)


class EvaluationRunResult(BaseModel):
    """Complete structured result envelope of an evaluation run.

    Provides both attribute and mapping access to the run results, metadata,
    reconciliation metrics, and paired ground-truth records.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    run_id: str = Field(..., description="Authoritative UUID for this evaluation run.")
    status: str = Field(..., description="Final run status (completed, partially_failed, failed).")
    metadata: dict[str, Any] = Field(..., description="Configuration and component metadata.")
    reconciliation: dict[str, Any] = Field(..., description="A8 SQLite reconciliation status.")
    results: list[ProcessedTicketResult] = Field(
        ...,
        description="Sequential processed ticket results in exact input order.",
    )
    ground_truth: dict[str, EvaluationRecord] = Field(
        ...,
        description="Lookup mapping ticket_id to its ground-truth EvaluationRecord.",
    )
    output_dir: Path = Field(..., description="Path to directory containing generated artifacts.")
    summary_metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Comprehensive summary metrics calculated across all subsystems.",
    )

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


class EvaluationRunner:
    """Core runner orchestrating batch evaluation against arbitrary datasets.

    Enforces:
    - Strict boundary between runtime tickets and evaluation answer key.
    - One result per input record preserving original dataset ordering.
    - Protection against accidental overwrite of existing non-empty directories.
    - Data minimization: never writes raw ticket bodies, customer names, emails,
      credentials, or raw exceptions to runner artifacts.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        audit_repository: DecisionAuditRepository | None = None,
        kb_path: Path | str | None = None,
        similarity_threshold_override: float | None = None,
        reference_responses_path: Path | str | None = None,
    ) -> None:
        """Initialize EvaluationRunner with optional overrides."""
        self.settings = settings or get_settings()
        self.audit_repository = audit_repository
        self.kb_path = kb_path
        self.similarity_threshold_override = similarity_threshold_override
        self.reference_responses_path = reference_responses_path

    def run(
        self,
        input_path: Path | str,
        output_dir: Path | str,
        mode: Literal["mock", "live"] | str = "mock",
        overwrite: bool = False,
        run_id: str | None = None,
        mock_llm_responses: list[str] | str | None = None,
        llm_client_override: BaseLLMClient | None = None,
        resume: bool = False,
        retry_failed: bool = False,
        reference_responses_path: Path | str | None = None,
    ) -> EvaluationRunResult:
        """Execute unattended evaluation against an input dataset with resumability.

        Args:
            input_path: Path to dataset JSON file (e.g. development_tickets.json).
            output_dir: Path to directory for persisting run metadata and reconciliation.
            mode: "mock" for isolated deterministic execution; "live" for production models.
            overwrite: If False, raises FileExistsError if output_dir is non-empty.
            run_id: Optional processing_run_id override or resume target.
            mock_llm_responses: Optional canned responses for mock LLM client.
            llm_client_override: Optional LLM client injection for testing.
            resume: If True, resumes an existing or interrupted evaluation run.
            retry_failed: If True in resume mode, retries previously failed tickets.

        Returns:
            EvaluationRunResult: Encapsulating run ID, status, metadata, results, and ground truth.

        Raises:
            FileNotFoundError: If input_path does not exist.
            FileExistsError: If output_dir exists and contains files without overwrite/resume.
            ValueError: If dataset is empty or malformed, or if configuration mismatch on resume.
        """
        # 1. Validate input dataset path
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Input dataset file not found: {input_file}")

        # 2. Validate output directory and enforce overwrite protection
        out_dir = Path(output_dir)
        metadata_file = out_dir / "run_metadata.json"
        existing_run_metadata: dict[str, Any] | None = None
        if metadata_file.exists():
            try:
                with open(metadata_file, encoding="utf-8") as f:
                    existing_run_metadata = json.load(f)
            except Exception:
                existing_run_metadata = None

        if out_dir.exists() and any(out_dir.iterdir()):
            if not overwrite and not resume:
                raise FileExistsError(
                    f"Output directory '{out_dir}' exists and is not empty. "
                    "Specify overwrite=True (or --overwrite via CLI) to permit writing into it, "
                    "or resume=True (or --resume via CLI) to resume an interrupted run."
                )
        out_dir.mkdir(parents=True, exist_ok=True)

        # 3. Parse evaluation dataset records and calculate dataset fingerprint
        with open(input_file, encoding="utf-8") as f:
            raw_content = f.read()

        dataset_hash = hashlib.sha256(raw_content.encode("utf-8")).hexdigest()

        try:
            raw_items = json.loads(raw_content)
        except json.JSONDecodeError as err:
            raise ValueError(f"Malformed JSON in dataset file '{input_file}': {err}") from err

        if not isinstance(raw_items, list):
            raise ValueError(
                f"Expected JSON array of ticket records in '{input_file}', "
                f"got {type(raw_items).__name__}"
            )

        if not raw_items:
            raise ValueError(f"Input dataset at '{input_file}' is empty or contains no records.")

        # 4. Strict isolation: Separate runtime tickets from ground-truth answer key
        valid_tickets: list[UnifiedTicket] = []
        valid_indices: list[int] = []
        malformed_entries: dict[int, tuple[str, str]] = {}
        ground_truth_map: dict[str, EvaluationRecord] = {}

        for idx, item in enumerate(raw_items):
            try:
                rec = parse_evaluation_record(item)
                valid_tickets.append(rec.ticket)
                valid_indices.append(idx)
                ground_truth_map[rec.ticket.ticket_id] = rec
            except Exception as parse_err:
                ticket_id = (
                    item.get("ticket_id")
                    if isinstance(item, dict) and item.get("ticket_id")
                    else f"MALFORMED-RECORD-{idx + 1:04d}"
                )
                logger.warning(
                    "Skipping malformed dataset record at index %d ('%s'): %s",
                    idx,
                    ticket_id,
                    parse_err,
                )
                malformed_entries[idx] = (str(ticket_id), str(parse_err))

        # 5. Bind authoritative execution run ID and validate resume configuration
        if resume:
            if run_id:
                active_run_id = run_id
            elif existing_run_metadata and "run_id" in existing_run_metadata:
                active_run_id = existing_run_metadata["run_id"]
            else:
                raise ValueError(
                    "Cannot resume: run_id not specified and run_metadata.json not found."
                )

            # Validate dataset integrity on resume
            if existing_run_metadata:
                saved_hash = existing_run_metadata.get("dataset_hash")
                if saved_hash and saved_hash != dataset_hash:
                    raise ValueError(
                        f"Resume rejected: dataset hash mismatch. Run had hash "
                        f"{saved_hash[:12]}..., current dataset has hash {dataset_hash[:12]}..."
                    )
                saved_mode = existing_run_metadata.get("execution_mode")
                if saved_mode and saved_mode != mode:
                    raise ValueError(
                        f"Resume rejected: mode mismatch. Run was '{saved_mode}', "
                        f"resume requested '{mode}'."
                    )
        else:
            active_run_id = run_id or str(uuid.uuid4())

        # 6. Build isolated processor via EvaluationComponentFactory
        processor, metadata = EvaluationComponentFactory.build_processor(
            mode=mode,
            settings=self.settings,
            audit_repository=self.audit_repository,
            processing_run_id=active_run_id,
            kb_path=self.kb_path,
            similarity_threshold_override=self.similarity_threshold_override,
            mock_llm_responses=mock_llm_responses,
            llm_client_override=llm_client_override,
        )
        metadata["dataset_hash"] = dataset_hash

        # Initialize checkpoints in SQLite if repository is available
        existing_checkpoints: dict[str, Any] = {}
        if processor.audit_repository is not None:
            all_ticket_ids = [t.ticket_id for t in valid_tickets]
            processor.audit_repository.init_checkpoints(active_run_id, all_ticket_ids)
            cp_list = processor.audit_repository.get_checkpoints(active_run_id)
            existing_checkpoints = {cp.ticket_id: cp for cp in cp_list}

        # 7. Execute batch processing with checkpointing and quota safety
        logger.info(
            "Starting evaluation run '%s' (mode=%s, resume=%s, retry_failed=%s) "
            "with %d valid tickets (%d malformed).",
            active_run_id,
            mode,
            resume,
            retry_failed,
            len(valid_tickets),
            len(malformed_entries),
        )

        valid_results: list[ProcessedTicketResult] = []
        paused_for_quota = False
        quota_error_msg: str | None = None

        if resume:
            # Resumed run: iterate ticket-by-ticket respecting previous checkpoints
            for ticket in valid_tickets:
                cp = existing_checkpoints.get(ticket.ticket_id)
                if cp is not None and cp.status == "completed" and cp.result_json:
                    try:
                        loaded_res = ProcessedTicketResult.model_validate_json(cp.result_json)
                        valid_results.append(loaded_res)
                        continue
                    except Exception as load_err:
                        logger.warning(
                            "Failed to deserialize checkpoint for ticket '%s': %s",
                            ticket.ticket_id,
                            load_err,
                        )

                if not retry_failed and cp is not None and cp.status == "failed" and cp.result_json:
                    try:
                        loaded_res = ProcessedTicketResult.model_validate_json(cp.result_json)
                        valid_results.append(loaded_res)
                        continue
                    except Exception:
                        pass

                if paused_for_quota:
                    continue

                if processor.audit_repository is not None:
                    processor.audit_repository.update_checkpoint(
                        run_id=active_run_id,
                        ticket_id=ticket.ticket_id,
                        status="processing",
                        increment_attempt=True,
                    )

                try:
                    ticket_res = processor.process_ticket(ticket, run_id=active_run_id)
                    valid_results.append(ticket_res)
                    res_status = "completed" if ticket_res.error is None else "failed"
                    if processor.audit_repository is not None:
                        processor.audit_repository.update_checkpoint(
                            run_id=active_run_id,
                            ticket_id=ticket.ticket_id,
                            status=res_status,
                            audit_id=ticket_res.audit_id,
                            result_json=ticket_res.model_dump_json(),
                            error=ticket_res.error,
                        )
                except ProviderQuotaExhaustedError as quota_exc:
                    logger.error(
                        "EVALUATION_PAUSED_QUOTA: Daily quota exhausted on ticket '%s': %s",
                        ticket.ticket_id,
                        quota_exc,
                    )
                    paused_for_quota = True
                    quota_error_msg = str(quota_exc)
                    if processor.audit_repository is not None:
                        processor.audit_repository.update_checkpoint(
                            run_id=active_run_id,
                            ticket_id=ticket.ticket_id,
                            status="paused_quota",
                            error=str(quota_exc),
                        )
                    break
                except Exception as ticket_exc:
                    logger.error(
                        "Unexpected ticket processing failure for ticket '%s': %s",
                        ticket.ticket_id,
                        ticket_exc,
                    )
                    dummy_fail_route = processor.router.route_ticket(
                        ticket=ticket,
                        classification=ClassificationResult(
                            ticket_id=ticket.ticket_id,
                            intent=IntentCategory.UNCLEAR_REQUEST,
                            urgency=UrgencyLevel.HIGH,
                            confidence=0.0,
                            intent_confidence=0.0,
                            urgency_confidence=0.0,
                            alternatives=[],
                            rationale=f"Processing exception: {ticket_exc}",
                            model_name="error-fallback",
                            prompt_version="v1",
                            is_fallback=True,
                        ),
                        retrieved_passages=[],
                        guardrail_result=GuardrailCheckResult(
                            is_safe=False,
                            violation_type="processing_exception",
                            details=f"Processing exception: {ticket_exc}",
                        ),
                        cited_ids=[],
                    )
                    failed_res = ProcessedTicketResult(
                        ticket=ticket,
                        classification=ClassificationResult(
                            ticket_id=ticket.ticket_id,
                            intent=IntentCategory.UNCLEAR_REQUEST,
                            urgency=UrgencyLevel.HIGH,
                            confidence=0.0,
                            intent_confidence=0.0,
                            urgency_confidence=0.0,
                            alternatives=[],
                            rationale=f"Processing exception: {ticket_exc}",
                            model_name="error-fallback",
                            prompt_version="v1",
                            is_fallback=True,
                        ),
                        billing_risk=assess_billing_risk(ticket),
                        retrieved_passages=[],
                        generated_response=None,
                        grounding_result=None,
                        routing_decision=dummy_fail_route,
                        final_customer_response=None,
                        private_agent_notes=f"Processing failed: {ticket_exc}",
                        processing_latency_ms=0.0,
                        error=f"Processing failed: {ticket_exc}",
                        audit_id=None,
                        processing_run_id=active_run_id,
                        effective_route=RouteOutcome.ESCALATE,
                        persistence_status="failed",
                        delivery_status="held_for_review",
                    )
                    valid_results.append(failed_res)
                    if processor.audit_repository is not None:
                        processor.audit_repository.update_checkpoint(
                            run_id=active_run_id,
                            ticket_id=ticket.ticket_id,
                            status="failed",
                            result_json=failed_res.model_dump_json(),
                            error=str(ticket_exc),
                        )
        else:
            # Fresh execution: calls processor.process_batch(valid_tickets)
            def _on_single_complete(
                completed_ticket: UnifiedTicket, outcome: ProcessedTicketResult
            ) -> None:
                valid_results.append(outcome)
                res_status = "completed" if outcome.error is None else "failed"
                if processor.audit_repository is not None:
                    processor.audit_repository.update_checkpoint(
                        run_id=active_run_id,
                        ticket_id=completed_ticket.ticket_id,
                        status=res_status,
                        audit_id=outcome.audit_id,
                        result_json=outcome.model_dump_json(),
                        error=outcome.error,
                        increment_attempt=True,
                    )

            try:
                processor.process_batch(
                    valid_tickets,
                    run_id=active_run_id,
                    on_ticket_complete=_on_single_complete,
                )
            except ProviderQuotaExhaustedError as quota_exc:
                logger.error(
                    "EVALUATION_PAUSED_QUOTA: Daily quota exhausted during batch: %s",
                    quota_exc,
                )
                paused_for_quota = True
                quota_error_msg = str(quota_exc)
                if processor.audit_repository is not None:
                    # Mark the current in-flight ticket as paused_quota, leave remaining as pending
                    for ticket in valid_tickets:
                        ticket_processed = any(
                            r.ticket.ticket_id == ticket.ticket_id for r in valid_results
                        )
                        if not ticket_processed:
                            processor.audit_repository.update_checkpoint(
                                run_id=active_run_id,
                                ticket_id=ticket.ticket_id,
                                status="paused_quota",
                                error=str(quota_exc),
                            )
                            break
            except TypeError:
                # Fallback for custom mocks without on_ticket_complete
                batch_res = processor.process_batch(valid_tickets, run_id=active_run_id)
                for ticket, outcome in zip(valid_tickets, batch_res, strict=True):
                    valid_results.append(outcome)
                    res_status = "completed" if outcome.error is None else "failed"
                    if processor.audit_repository is not None:
                        processor.audit_repository.update_checkpoint(
                            run_id=active_run_id,
                            ticket_id=ticket.ticket_id,
                            status=res_status,
                            audit_id=outcome.audit_id,
                            result_json=outcome.model_dump_json(),
                            error=outcome.error,
                            increment_attempt=True,
                        )

        # Record audit attempts and failures for malformed records to maintain A8 ledger integrity
        malformed_results: dict[int, ProcessedTicketResult] = {}
        if malformed_entries and processor.audit_repository is not None:
            for malformed_id, _ in malformed_entries.values():
                try:
                    processor.audit_repository.record_run_attempt(active_run_id)
                    processor.audit_repository.record_run_failure(active_run_id)
                except Exception as audit_exc:
                    logger.warning(
                        "Failed to register audit failure for malformed ticket '%s': %s",
                        malformed_id,
                        audit_exc,
                    )

        for idx, (malformed_id, err_msg) in malformed_entries.items():
            dummy_ticket = UnifiedTicket(
                ticket_id=malformed_id,
                channel=ChannelEnum.EMAIL,
                received_at=datetime.now(timezone.utc),
                subject="",
                body="[MALFORMED_RECORD_PARSE_FAILED]",
            )
            fallback_classification = ClassificationResult(
                ticket_id=dummy_ticket.ticket_id,
                intent=IntentCategory.UNCLEAR_REQUEST,
                urgency=UrgencyLevel.HIGH,
                confidence=0.0,
                intent_confidence=0.0,
                urgency_confidence=0.0,
                alternatives=[],
                rationale=f"Malformed record at index {idx}: {err_msg}",
                model_name="error-fallback",
                prompt_version="v1",
                is_fallback=True,
            )
            guardrail = GuardrailCheckResult(
                is_safe=False,
                violation_type="malformed_record_parse_error",
                details=f"Malformed record parse failure: {err_msg}",
            )
            routing_decision = processor.router.route_ticket(
                ticket=dummy_ticket,
                classification=fallback_classification,
                retrieved_passages=[],
                guardrail_result=guardrail,
                cited_ids=[],
            )
            malformed_results[idx] = ProcessedTicketResult(
                ticket=dummy_ticket,
                classification=fallback_classification,
                billing_risk=assess_billing_risk(dummy_ticket),
                retrieved_passages=[],
                generated_response=None,
                grounding_result=None,
                routing_decision=routing_decision,
                final_customer_response=None,
                private_agent_notes=f"Dataset parsing failure at record index {idx}: {err_msg}",
                processing_latency_ms=0.0,
                error=f"Malformed record parse failure: {err_msg}",
                audit_id=None,
                processing_run_id=active_run_id,
                effective_route=RouteOutcome.ESCALATE,
                persistence_status="failed",
                delivery_status="held_for_review",
            )

        # Assemble results preserving exact input sequence
        if paused_for_quota:
            # During paused quota, results contains whatever valid results were processed so far
            results = list(valid_results)
        else:
            results = [None] * len(raw_items)  # type: ignore[list-item]
            for v_idx, res in zip(valid_indices, valid_results, strict=True):
                results[v_idx] = res
            for m_idx, res in malformed_results.items():
                results[m_idx] = res

            # Ensure exact 1-to-1 match and order preservation
            if len(results) != len(raw_items):
                raise RuntimeError(
                    f"Batch result count mismatch: expected {len(raw_items)}, got {len(results)}"
                )

        # 8. Capture audit reconciliation status from SQLite
        if processor.audit_repository is not None:
            reconciliation = processor.audit_repository.reconciliation_status(
                run_id=active_run_id,
                total_processed=len(results),
            )
        else:
            reconciliation = {
                "processing_run_id": active_run_id,
                "total_processed": len(results),
                "persisted_records": 0,
                "failed_or_missing_records": 0,
                "reconciled": False,
                "gap": 0,
                "failed_in_memory_records": 0,
                "failed_attempts": 0,
                "status": "omitted",
                "completed_at": None,
            }

        # 9. Classify run status
        failed_tickets = sum(
            1 for r in results if r.error is not None or r.persistence_status == "failed"
        )
        is_reconciled = reconciliation.get("reconciled", False)

        if paused_for_quota:
            run_status = "paused_quota"
            if processor.audit_repository is not None:
                processor.audit_repository.complete_run(active_run_id, status="paused_quota")
        elif len(results) == len(raw_items) and failed_tickets == 0 and is_reconciled:
            run_status = "completed"
            if processor.audit_repository is not None:
                processor.audit_repository.complete_run(active_run_id, status="completed")
        elif len(results) == len(raw_items) and failed_tickets < len(results):
            run_status = "partially_failed"
            if processor.audit_repository is not None:
                processor.audit_repository.complete_run(active_run_id, status="partially_failed")
        else:
            run_status = "failed"
            if processor.audit_repository is not None:
                processor.audit_repository.complete_run(active_run_id, status="failed")

        # 10. Persist runner artifacts
        # run_metadata.json
        run_metadata_payload: dict[str, Any] = {
            "run_id": active_run_id,
            "status": run_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_path": str(input_file.resolve()),
            "output_dir": str(out_dir.resolve()),
            "total_records": len(raw_items),
            "processed_records": len(results),
            "failed_tickets": failed_tickets,
            "paused_for_quota": paused_for_quota,
            "quota_error": quota_error_msg,
            **metadata,
        }
        with open(out_dir / "run_metadata.json", "w", encoding="utf-8") as f:
            json.dump(run_metadata_payload, f, indent=2)

        # audit_reconciliation.json
        with open(out_dir / "audit_reconciliation.json", "w", encoding="utf-8") as f:
            json.dump(reconciliation, f, indent=2)

        # 11. Extract KB doc IDs and generate comprehensive evaluation artifacts if not paused
        summary_metrics: dict[str, Any] = {}
        if not paused_for_quota or len(results) > 0:
            kb_doc_ids: set[str] | None = None
            resolved_kb_path = Path(
                self.kb_path or self.settings.kb_directory / "documentation.json"
            )
            if resolved_kb_path.exists():
                try:
                    with open(resolved_kb_path, encoding="utf-8") as f:
                        kb_raw = json.load(f)
                    kb_doc_ids = {
                        item["doc_id"]
                        for item in kb_raw
                        if isinstance(item, dict) and "doc_id" in item
                    }
                except Exception as exc:
                    logger.warning("Could not read KB doc IDs from %s: %s", resolved_kb_path, exc)

            # Load reference responses if available (for answer quality evaluation)
            ref_responses_data: list[dict[str, Any]] | None = None
            ref_path = Path(
                reference_responses_path
                or self.reference_responses_path
                or "data/val/ground_truth_responses.json"
            )
            if ref_path.exists():
                try:
                    with open(ref_path, encoding="utf-8") as f:
                        ref_responses_data = json.load(f)
                except Exception as exc:
                    logger.warning(
                        "Could not load reference responses from %s: %s", ref_path, exc
                    )

            summary_metrics = generate_evaluation_artifacts(
                run_id=active_run_id,
                metadata=run_metadata_payload,
                results=results,
                ground_truth=ground_truth_map,
                reconciliation=reconciliation,
                output_dir=out_dir,
                kb_doc_ids=kb_doc_ids,
                reference_responses=ref_responses_data,
            )

        logger.info(
            "Evaluation run '%s' finished with status '%s' (%d/%d records processed). "
            "Artifacts written to '%s'.",
            active_run_id,
            run_status,
            len(results),
            len(raw_items),
            out_dir,
        )

        return EvaluationRunResult(
            run_id=active_run_id,
            status=run_status,
            metadata=run_metadata_payload,
            reconciliation=reconciliation,
            results=results,
            ground_truth=ground_truth_map,
            output_dir=out_dir,
            summary_metrics=summary_metrics,
        )
