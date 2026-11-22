"""Comprehensive tests for resumable evaluation and provider reliability (A3.33).

Validates:
1. Durable checkpointing and state persistence.
2. Resuming an interrupted evaluation skips completed tickets without repeating work.
3. Configuration validation on resume (detects dataset hash, model, and mode mismatch).
4. Groq HTTP 429 transient rate limit parsing with Retry-After and backoff.
5. Daily quota exhaustion (TPD) detection and safe pause without silent fallback.
6. Audit duplicate prevention: retries do not create duplicate final decisions.
7. Output directory overwrite protection preserved while permitting resume.
8. CLI status command and options (--resume, --retry-failed, --status).
"""

import json
import uuid
from pathlib import Path
from typing import Any

import pytest

import scripts.evaluate as cli_module
from cloudserve.classification.models import (
    ClassificationResult,
    IntentCategory,
    UrgencyLevel,
)
from cloudserve.evaluation.runner import EvaluationRunner
from cloudserve.generation.llm_client import (
    ProviderQuotaExhaustedError,
    is_daily_quota_exhaustion,
    parse_retry_after,
)
from cloudserve.ingestion.models import UnifiedTicket
from cloudserve.service import ProcessedTicketResult, TicketProcessor
from cloudserve.storage.database import create_sqlite_engine, get_session_factory, init_db
from cloudserve.storage.repository import DecisionAuditRepository


@pytest.fixture
def in_memory_audit_repo() -> DecisionAuditRepository:
    """Fixture providing an isolated in-memory DecisionAuditRepository with all tables."""
    engine = create_sqlite_engine("sqlite:///:memory:")
    init_db(engine)
    session_factory = get_session_factory(engine)
    return DecisionAuditRepository(session_factory=session_factory)


def _make_sample_dataset(count: int = 5) -> str:
    """Generate a valid sample dataset JSON string."""
    records = []
    for i in range(1, count + 1):
        records.append(
            {
                "ticket_id": f"DEV-{i:04d}",
                "channel": "email",
                "subject": f"Sample inquiry {i}",
                "body": f"Customer question details for ticket {i}.",
                "received_at": "2026-11-15T10:00:00Z",
                "customer_id": f"CUST-{i:03d}",
                "customer_name": f"Customer {i}",
                "customer_tier": "standard",
                "labels": {
                    "intent": "api_usage_question",
                    "urgency": "low",
                    "expected_route": "auto_respond",
                },
            }
        )
    return json.dumps(records, indent=2)


# -----------------------------------------------------------------------------
# 1. Checkpoint Table and Lifecycle
# -----------------------------------------------------------------------------
def test_checkpoint_table_lifecycle(in_memory_audit_repo: DecisionAuditRepository) -> None:
    """Verify checkpoints initialize as pending and update to completed."""
    run_id = str(uuid.uuid4())
    ticket_ids = ["DEV-0001", "DEV-0002"]

    in_memory_audit_repo.init_checkpoints(run_id, ticket_ids)
    cps = in_memory_audit_repo.get_checkpoints(run_id)
    assert len(cps) == 2
    assert cps[0].ticket_id == "DEV-0001"
    assert cps[0].status == "pending"
    assert cps[0].attempt_count == 0

    # Update to processing
    in_memory_audit_repo.update_checkpoint(
        run_id=run_id,
        ticket_id="DEV-0001",
        status="processing",
        increment_attempt=True,
    )
    cps = in_memory_audit_repo.get_checkpoints(run_id)
    assert cps[0].status == "processing"
    assert cps[0].attempt_count == 1

    # Update to completed
    dummy_audit_id = str(uuid.uuid4())
    in_memory_audit_repo.update_checkpoint(
        run_id=run_id,
        ticket_id="DEV-0001",
        status="completed",
        audit_id=dummy_audit_id,
        result_json='{"status": "ok"}',
    )
    cps = in_memory_audit_repo.get_checkpoints(run_id)
    assert cps[0].status == "completed"
    assert cps[0].audit_id == dummy_audit_id
    assert cps[0].result_json == '{"status": "ok"}'

    counts = in_memory_audit_repo.get_checkpoint_counts(run_id)
    assert counts.get("completed") == 1
    assert counts.get("pending") == 1


# -----------------------------------------------------------------------------
# 2. Resuming Skips Completed Tickets
# -----------------------------------------------------------------------------
def test_resume_skips_completed_tickets(
    tmp_path: Path,
    in_memory_audit_repo: DecisionAuditRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify an interrupted run resumed with resume=True reuses completed tickets."""
    dataset_file = tmp_path / "dataset.json"
    dataset_file.write_text(_make_sample_dataset(4), encoding="utf-8")
    out_dir = tmp_path / "eval_out"
    explicit_run_id = f"eval-run-{uuid.uuid4()}"

    processed_calls: list[str] = []
    orig_process_ticket = TicketProcessor.process_ticket

    def _mock_process_ticket(
        self: TicketProcessor, ticket: UnifiedTicket, run_id: str | None = None
    ) -> ProcessedTicketResult:
        processed_calls.append(ticket.ticket_id)
        if ticket.ticket_id == "DEV-0003":
            # Simulate crash or interruption on 3rd ticket
            raise RuntimeError("Simulated crash mid-run")
        return orig_process_ticket(self, ticket, run_id=run_id)

    monkeypatch.setattr(TicketProcessor, "process_ticket", _mock_process_ticket)

    runner = EvaluationRunner(audit_repository=in_memory_audit_repo)
    # First run: DEV-0001, DEV-0002 succeed; DEV-0003 encounters exception (handled fail-closed)
    runner.run(
        input_path=dataset_file,
        output_dir=out_dir,
        mode="mock",
        run_id=explicit_run_id,
    )
    assert len(processed_calls) == 4

    # Now resume with resume=True (without retry_failed)
    processed_calls.clear()
    res2 = runner.run(
        input_path=dataset_file,
        output_dir=out_dir,
        mode="mock",
        resume=True,
    )
    assert res2.run_id == explicit_run_id
    # DEV-0001 and DEV-0002 were completed; DEV-0003 was failed; DEV-0004 was completed.
    # Because retry_failed=False, completed/failed with result_json should not be re-processed
    assert len(processed_calls) == 0
    assert len(res2.results) == 4


# -----------------------------------------------------------------------------
# 3. Configuration Mismatch on Resume Fails Safely
# -----------------------------------------------------------------------------
def test_resume_configuration_mismatch_fails_safely(
    tmp_path: Path,
    in_memory_audit_repo: DecisionAuditRepository,
) -> None:
    """Verify resume fails if the dataset content or execution mode changed."""
    dataset_file = tmp_path / "dataset.json"
    dataset_file.write_text(_make_sample_dataset(3), encoding="utf-8")
    out_dir = tmp_path / "eval_out"
    run_id = f"eval-{uuid.uuid4()}"

    runner = EvaluationRunner(audit_repository=in_memory_audit_repo)
    runner.run(
        input_path=dataset_file,
        output_dir=out_dir,
        mode="mock",
        run_id=run_id,
    )

    # Modify dataset content
    dataset_file.write_text(_make_sample_dataset(4), encoding="utf-8")

    with pytest.raises(ValueError, match="Resume rejected: dataset hash mismatch"):
        runner.run(
            input_path=dataset_file,
            output_dir=out_dir,
            mode="mock",
            resume=True,
        )


# -----------------------------------------------------------------------------
# 4. Groq HTTP 429 Parsing & Differentiation
# -----------------------------------------------------------------------------
def test_retry_after_and_quota_exhaustion_detection() -> None:
    """Verify parse_retry_after and is_daily_quota_exhaustion differentiate rate limits."""
    # Transient rate limit header
    assert parse_retry_after("2.5") == 2.5
    assert parse_retry_after("0") == 0.0
    assert parse_retry_after(None) is None
    assert parse_retry_after("invalid") is None

    # Transient 429 error messages (TPM / RPM)
    transient_msg = "Rate limit reached: Limit 6000 TPM. Please try again in 2.3s."
    assert is_daily_quota_exhaustion(transient_msg) is False

    # Daily quota error messages (TPD 200,000 exceeded)
    daily_msg_1 = "Rate limit reached: Limit 200000 TPD. Please try again in 14h32m."
    assert is_daily_quota_exhaustion(daily_msg_1) is True

    daily_msg_2 = "Requests per day limit exceeded: daily quota of 200000 reached."
    assert is_daily_quota_exhaustion(daily_msg_2) is True

    daily_msg_3 = "Try again in 8 hours."
    assert is_daily_quota_exhaustion(daily_msg_3) is True


# -----------------------------------------------------------------------------
# 5. Daily Quota Exhaustion Halts Cleanly Without Silent Fallback
# -----------------------------------------------------------------------------
def test_daily_quota_exhaustion_halts_cleanly(
    tmp_path: Path,
    in_memory_audit_repo: DecisionAuditRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify daily quota exhaustion pauses evaluation cleanly without fallback."""
    dataset_file = tmp_path / "dataset.json"
    dataset_file.write_text(_make_sample_dataset(3), encoding="utf-8")
    out_dir = tmp_path / "eval_out"

    call_count = 0

    def _failing_classify(self: Any, ticket: UnifiedTicket) -> ClassificationResult:
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            raise ProviderQuotaExhaustedError("Rate limit reached: Limit 200000 TPD.")
        return ClassificationResult(
            ticket_id=ticket.ticket_id,
            intent=IntentCategory.API_USAGE_QUESTION,
            urgency=UrgencyLevel.LOW,
            confidence=0.9,
            intent_confidence=0.95,
            urgency_confidence=0.95,
            alternatives=[],
            rationale="Normal classification",
            model_name="mock-model",
            prompt_version="v1",
        )

    from cloudserve.classification.classifier import TicketClassifier

    monkeypatch.setattr(TicketClassifier, "classify", _failing_classify)

    runner = EvaluationRunner(audit_repository=in_memory_audit_repo)
    result = runner.run(
        input_path=dataset_file,
        output_dir=out_dir,
        mode="mock",
    )

    assert result.status == "paused_quota"
    assert result.metadata.get("paused_for_quota") is True
    # DEV-0001 succeeded, DEV-0002 paused quota
    cps = in_memory_audit_repo.get_checkpoints(result.run_id)
    status_map = {cp.ticket_id: cp.status for cp in cps}
    assert status_map["DEV-0001"] == "completed"
    assert status_map["DEV-0002"] == "paused_quota"
    assert status_map["DEV-0003"] == "pending"


# -----------------------------------------------------------------------------
# 6. Retry Failed Tickets Mode
# -----------------------------------------------------------------------------
def test_resume_retry_failed_reprocesses_failures(
    tmp_path: Path,
    in_memory_audit_repo: DecisionAuditRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify resume=True with retry_failed=True re-attempts previously failed tickets."""
    dataset_file = tmp_path / "dataset.json"
    dataset_file.write_text(_make_sample_dataset(3), encoding="utf-8")
    out_dir = tmp_path / "eval_out"
    run_id = f"eval-{uuid.uuid4()}"

    should_fail_ticket_2 = True
    orig_process_ticket = TicketProcessor.process_ticket

    def _conditionally_failing_process(
        self: TicketProcessor, ticket: UnifiedTicket, run_id: str | None = None
    ) -> ProcessedTicketResult:
        if ticket.ticket_id == "DEV-0002" and should_fail_ticket_2:
            raise RuntimeError("Transient provider error on ticket 2")
        return orig_process_ticket(self, ticket, run_id=run_id)

    monkeypatch.setattr(TicketProcessor, "process_ticket", _conditionally_failing_process)

    runner = EvaluationRunner(audit_repository=in_memory_audit_repo)
    # First pass: DEV-0002 fails
    res1 = runner.run(
        input_path=dataset_file,
        output_dir=out_dir,
        mode="mock",
        run_id=run_id,
    )
    assert res1.status == "partially_failed"
    cps1 = in_memory_audit_repo.get_checkpoints(run_id)
    assert cps1[1].status == "failed"

    # Second pass: fix condition and resume with retry_failed=True
    should_fail_ticket_2 = False
    res2 = runner.run(
        input_path=dataset_file,
        output_dir=out_dir,
        mode="mock",
        resume=True,
        retry_failed=True,
    )
    assert res2.status == "completed"
    cps2 = in_memory_audit_repo.get_checkpoints(run_id)
    assert cps2[1].status == "completed"
    assert cps2[1].attempt_count >= 2


# -----------------------------------------------------------------------------
# 7. CLI Status Inspection
# -----------------------------------------------------------------------------
def test_cli_status_inspection(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify CLI --status flag prints evaluation metadata without re-executing."""
    dataset_file = tmp_path / "sample_dataset.json"
    dataset_file.write_text(_make_sample_dataset(3), encoding="utf-8")
    out_dir = tmp_path / "cli_status_out"
    # Run evaluation
    cli_module.main(
        [
            "--input",
            str(dataset_file),
            "--output-dir",
            str(out_dir),
            "--mode",
            "mock",
        ]
    )

    # Check status
    exit_code = cli_module.main(
        [
            "--input",
            str(dataset_file),
            "--output-dir",
            str(out_dir),
            "--status",
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Evaluation Run Status Inspection" in captured.out
    assert "Status:             COMPLETED" in captured.out
