"""Unit tests for EvaluationComponentFactory, EvaluationRunner, and CLI.

Verifies:
- Runtime/evaluation label separation and UnifiedTicket strict isolation.
- Factory mock mode and explicit component injection.
- Live mode missing credentials fails without mock fallback.
- Live mode embedding failure fails without mock fallback.
- Arbitrary input and output paths.
- Existing non-empty output directory protection.
- Stable ticket ordering and one result per input.
- Ground-truth labels never passed to TicketProcessor.
- Run ID and audit reconciliation handling.
- Per-ticket processing failure does not lose the rest of the batch.
- Retriever similarity threshold consistency and verification.
- Data minimization and privacy in exported runner artifacts.
- CLI argument parsing, execution, and exit codes.
"""

import json
import uuid
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr, ValidationError

import scripts.evaluate as cli_module
from cloudserve.classification.classifier import TicketClassifier
from cloudserve.config import Settings
from cloudserve.evaluation.factory import EvaluationComponentFactory
from cloudserve.evaluation.runner import EvaluationRunner
from cloudserve.generation.llm_client import (
    ConfigurationError,
    MockLLMClient,
)
from cloudserve.generation.response_generator import ResponseGenerator
from cloudserve.ingestion.models import EvaluationRecord, UnifiedTicket
from cloudserve.ingestion.parsers import parse_evaluation_dataset
from cloudserve.rag.retriever import KBRetriever
from cloudserve.routing.models import RouteOutcome
from cloudserve.service import ProcessedTicketResult, TicketProcessor
from cloudserve.storage.database import create_sqlite_engine, get_session_factory, init_db
from cloudserve.storage.repository import DecisionAuditRepository


def _make_sample_dataset(num_tickets: int = 3) -> str:
    """Generate a valid JSON string of benchmark EvaluationRecords."""
    records = []
    for i in range(1, num_tickets + 1):
        rec = {
            "ticket_id": f"DEV-{i:04d}",
            "channel": "email",
            "subject": f"Inquiry about service feature {i}",
            "body": f"Hello, I have a question regarding feature {i}. Thank you.",
            "received_at": "2026-11-15T10:00:00Z",
            "customer_id": f"CUST-{i:03d}",
            "customer_name": f"Alice Customer {i}",
            "customer_tier": "standard",
            "customer_region": "north_america",
            "language_fluency": "fluent",
            "labels": {
                "intent": "api_inquiry",
                "urgency": "medium",
                "expected_route": "auto_respond",
                "answerable_from_docs": True,
                "expected_doc_ids": ["DOC-API-001"],
                "must_not_auto_respond": False,
            },
            "history": {
                "first_contact_resolution": True,
                "resolution_time_minutes": 30,
                "csat_rating": 5,
                "escalated": False,
                "repeat_contact": False,
            },
        }
        records.append(rec)
    return json.dumps(records)


@pytest.fixture
def in_memory_audit_repo() -> DecisionAuditRepository:
    """Provide an isolated in-memory DecisionAuditRepository for tests."""
    engine = create_sqlite_engine("sqlite:///:memory:")
    init_db(engine)
    session_factory = get_session_factory(engine)
    return DecisionAuditRepository(session_factory=session_factory, engine=engine)


@pytest.fixture
def sample_dataset_path(tmp_path: Path) -> Path:
    """Write sample dataset JSON to disk and return path."""
    dataset_file = tmp_path / "test_dataset.json"
    dataset_file.write_text(_make_sample_dataset(3), encoding="utf-8")
    return dataset_file


# -----------------------------------------------------------------------------
# 1. Label Separation & UnifiedTicket Strict Isolation
# -----------------------------------------------------------------------------
def test_runtime_evaluation_label_separation(sample_dataset_path: Path) -> None:
    """Verify that runtime UnifiedTicket is strictly isolated from evaluation labels."""
    records = parse_evaluation_dataset(sample_dataset_path.read_text(encoding="utf-8"))
    assert len(records) == 3

    for rec in records:
        assert isinstance(rec, EvaluationRecord)
        ticket = rec.ticket
        assert isinstance(ticket, UnifiedTicket)
        # UnifiedTicket has extra='forbid' and must not contain label fields
        assert not hasattr(ticket, "expected_intent")
        assert not hasattr(ticket, "expected_route")
        assert not hasattr(ticket, "expected_urgency")
        assert not hasattr(ticket, "expected_doc_ids")
        assert not hasattr(ticket, "labels")
        assert not hasattr(ticket, "history")

        # Confirm labels and history reside solely on EvaluationRecord
        assert rec.labels is not None
        assert rec.labels.intent == "api_inquiry"
        assert rec.history is not None
        assert rec.history.csat_rating == 5

    # Attempting to assign or construct UnifiedTicket with labels must fail
    with pytest.raises(ValidationError):
        UnifiedTicket(
            ticket_id="DEV-9999",
            channel="email",
            received_at="2026-11-15T10:00:00Z",
            labels={"expected_intent": "api_inquiry"},  # Forbidden extra field
        )


# -----------------------------------------------------------------------------
# 2. Factory Mock Mode and Explicit Component Injection
# -----------------------------------------------------------------------------
def test_factory_mock_mode_injection(
    in_memory_audit_repo: DecisionAuditRepository,
    default_settings: Settings,
) -> None:
    """Verify mock mode injects offline components and configured baseline threshold."""
    processor, metadata = EvaluationComponentFactory.build_processor(
        mode="mock",
        settings=default_settings,
        audit_repository=in_memory_audit_repo,
    )

    assert isinstance(processor, TicketProcessor)
    assert isinstance(processor.classifier, TicketClassifier)
    assert isinstance(processor.classifier.llm_client, MockLLMClient)
    assert isinstance(processor.generator, ResponseGenerator)
    assert isinstance(processor.generator.llm_client, MockLLMClient)
    assert isinstance(processor.retriever, KBRetriever)
    assert processor.retriever.similarity_threshold == 0.15

    # Check recorded metadata
    assert metadata["execution_mode"] == "mock"
    assert metadata["classifier_class"] == "TicketClassifier"
    assert metadata["embedding_class"] == "MockEmbeddingFunction"
    assert metadata["retriever_class"] == "KBRetriever"
    assert metadata["retriever_similarity_threshold"] == 0.15
    assert metadata["retrieval_top_k"] == 4
    assert metadata["confidence_threshold_auto_respond"] == 0.80
    assert metadata["confidence_threshold_review"] == 0.60


def test_factory_configured_threshold_override(
    in_memory_audit_repo: DecisionAuditRepository,
) -> None:
    """Verify factory records custom/environment threshold override when provided."""
    override_settings = Settings(confidence_threshold_auto_respond=0.85, _env_file=None)
    processor, metadata = EvaluationComponentFactory.build_processor(
        mode="mock",
        settings=override_settings,
        audit_repository=in_memory_audit_repo,
    )
    assert metadata["confidence_threshold_auto_respond"] == 0.85
    assert processor.router.confidence_threshold_auto_respond == 0.85


def test_factory_invalid_mode_raises() -> None:
    """Verify unknown evaluation mode raises ValueError."""
    with pytest.raises(ValueError, match="Unknown evaluation mode 'invalid'"):
        EvaluationComponentFactory.build_processor(mode="invalid")


# -----------------------------------------------------------------------------
# 3. Live Mode Missing Credentials Fails Without Mock Fallback
# -----------------------------------------------------------------------------
def test_factory_live_mode_missing_credentials_fails(
    in_memory_audit_repo: DecisionAuditRepository,
) -> None:
    """Verify live mode raises ConfigurationError if GROQ_API_KEY is missing/empty."""
    settings = Settings(groq_api_key=None)

    with pytest.raises(ConfigurationError, match="Missing Groq API key"):
        EvaluationComponentFactory.build_processor(
            mode="live",
            settings=settings,
            audit_repository=in_memory_audit_repo,
        )


# -----------------------------------------------------------------------------
# 4. Live Mode Embedding Failure Fails Without Mock Fallback
# -----------------------------------------------------------------------------
def test_factory_live_mode_embedding_failure_fails(
    monkeypatch: pytest.MonkeyPatch,
    in_memory_audit_repo: DecisionAuditRepository,
) -> None:
    """Verify live mode raises RuntimeError if embedding model weights fail to load."""
    from cloudserve.rag.indexer import SentenceTransformerEmbeddingAdapter

    settings = Settings(groq_api_key=SecretStr("gsk_valid_mock_key_for_testing"))

    def _failing_load(self: Any) -> Any:
        raise RuntimeError("Weights not downloaded and network offline")

    monkeypatch.setattr(SentenceTransformerEmbeddingAdapter, "_load_model", _failing_load)

    with pytest.raises(RuntimeError, match="Weights not downloaded"):
        EvaluationComponentFactory.build_processor(
            mode="live",
            settings=settings,
            audit_repository=in_memory_audit_repo,
        )


# -----------------------------------------------------------------------------
# 4b. Live Mode Injects Real LangChainGroqClient Into Both Components
# -----------------------------------------------------------------------------
def test_factory_live_mode_injects_real_groq_client_into_both_components(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    in_memory_audit_repo: DecisionAuditRepository,
) -> None:
    """Verify live mode injects the same LangChainGroqClient into both components."""
    from cloudserve.generation.llm_client import LangChainGroqClient, MockLLMClient
    from cloudserve.rag.indexer import SentenceTransformerEmbeddingAdapter

    settings = Settings(
        groq_api_key=SecretStr("gsk_valid_mock_key_for_testing"),
        groq_model="openai/gpt-oss-120b",
        chroma_persist_directory=tmp_path / "test_chroma",
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

    # Mock sentence-transformers weight loading so test runs offline without network calls
    monkeypatch.setattr(SentenceTransformerEmbeddingAdapter, "_load_model", lambda self: None)

    processor, metadata = EvaluationComponentFactory.build_processor(
        mode="live",
        settings=settings,
        audit_repository=in_memory_audit_repo,
    )

    # 1. Real LangChainGroqClient is instantiated
    assert isinstance(processor.classifier.llm_client, LangChainGroqClient)
    assert isinstance(processor.generator.llm_client, LangChainGroqClient)

    # 2. The exact same client instance reaches both classifier and generator
    assert processor.classifier.llm_client is processor.generator.llm_client

    # 3. Neither component silently instantiates MockLLMClient
    assert not isinstance(processor.classifier.llm_client, MockLLMClient)
    assert not isinstance(processor.generator.llm_client, MockLLMClient)

    # 4. Correct model name and configuration applied in metadata
    assert metadata["execution_mode"] == "live"
    assert metadata["classifier_model"] == "openai/gpt-oss-120b"
    assert metadata["generator_model"] == "openai/gpt-oss-120b"
    assert metadata["embedding_class"] == "SentenceTransformerEmbeddingAdapter"
    assert metadata["retriever_similarity_threshold"] == 0.40


# -----------------------------------------------------------------------------
# 5. Arbitrary Input and Output Paths
# -----------------------------------------------------------------------------
def test_arbitrary_input_and_output_paths(
    tmp_path: Path,
    in_memory_audit_repo: DecisionAuditRepository,
) -> None:
    """Verify runner functions cleanly with arbitrary input and output paths."""
    input_path = tmp_path / "deeply" / "nested" / "tickets.json"
    input_path.parent.mkdir(parents=True)
    input_path.write_text(_make_sample_dataset(2), encoding="utf-8")

    out_dir = tmp_path / "custom_reports" / "run_01"

    runner = EvaluationRunner(audit_repository=in_memory_audit_repo)
    result = runner.run(input_path=str(input_path), output_dir=str(out_dir), mode="mock")

    assert result.status == "completed"
    assert (out_dir / "run_metadata.json").exists()
    assert (out_dir / "audit_reconciliation.json").exists()


# -----------------------------------------------------------------------------
# 6. Existing Non-Empty Output Directory Protection
# -----------------------------------------------------------------------------
def test_existing_non_empty_output_directory_protection(
    tmp_path: Path,
    sample_dataset_path: Path,
    in_memory_audit_repo: DecisionAuditRepository,
) -> None:
    """Verify FileExistsError unless overwrite=True."""
    out_dir = tmp_path / "eval_output"
    out_dir.mkdir(parents=True)
    (out_dir / "existing_file.txt").write_text("historical", encoding="utf-8")

    runner = EvaluationRunner(audit_repository=in_memory_audit_repo)

    # Must raise FileExistsError without overwrite
    with pytest.raises(FileExistsError, match="exists and is not empty"):
        runner.run(
            input_path=sample_dataset_path,
            output_dir=out_dir,
            mode="mock",
            overwrite=False,
        )

    # Must succeed with overwrite=True
    result = runner.run(
        input_path=sample_dataset_path,
        output_dir=out_dir,
        mode="mock",
        overwrite=True,
    )
    assert result.status == "completed"


# -----------------------------------------------------------------------------
# 7. Stable Ticket Ordering and One Result Per Input
# -----------------------------------------------------------------------------
def test_stable_ticket_ordering_and_one_result_per_input(
    tmp_path: Path,
    in_memory_audit_repo: DecisionAuditRepository,
) -> None:
    """Verify result order matches input order exactly, 1-to-1."""
    dataset_file = tmp_path / "five_tickets.json"
    dataset_file.write_text(_make_sample_dataset(5), encoding="utf-8")

    out_dir = tmp_path / "order_test_out"

    runner = EvaluationRunner(audit_repository=in_memory_audit_repo)
    result = runner.run(input_path=dataset_file, output_dir=out_dir, mode="mock")

    assert len(result.results) == 5
    expected_order = [f"DEV-{i:04d}" for i in range(1, 6)]
    actual_order = [r.ticket.ticket_id for r in result.results]
    assert actual_order == expected_order


# -----------------------------------------------------------------------------
# 8. Ground-Truth Labels Never Passed to TicketProcessor
# -----------------------------------------------------------------------------
def test_ground_truth_never_passed_to_processor(
    tmp_path: Path,
    sample_dataset_path: Path,
    in_memory_audit_repo: DecisionAuditRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify TicketProcessor.process_batch receives only runtime UnifiedTicket objects."""
    received_batch: list[Any] = []

    orig_process_batch = TicketProcessor.process_batch

    def _spied_process_batch(
        self: TicketProcessor,
        tickets: Any,
        run_id: str | None = None,
    ) -> list[ProcessedTicketResult]:
        received_batch.extend(tickets)
        batch_results: list[ProcessedTicketResult] = orig_process_batch(
            self, tickets, run_id=run_id
        )
        return batch_results

    monkeypatch.setattr(TicketProcessor, "process_batch", _spied_process_batch)

    out_dir = tmp_path / "leak_check_out"
    runner = EvaluationRunner(audit_repository=in_memory_audit_repo)
    result = runner.run(input_path=sample_dataset_path, output_dir=out_dir, mode="mock")

    assert len(received_batch) == 3
    for t in received_batch:
        assert isinstance(t, UnifiedTicket)
        # None of the tickets received by TicketProcessor may contain label fields
        assert not hasattr(t, "expected_intent")
        assert not hasattr(t, "expected_route")
        assert not hasattr(t, "labels")
        assert not hasattr(t, "history")

    # Post-hoc matching verifies result ground-truth map
    for res in result.results:
        assert res.ticket.ticket_id in result.ground_truth
        gt = result.ground_truth[res.ticket.ticket_id]
        assert gt.labels is not None
        assert gt.labels.intent == "api_inquiry"


# -----------------------------------------------------------------------------
# 9. Run ID and Audit Reconciliation Handling
# -----------------------------------------------------------------------------
def test_run_id_and_audit_reconciliation_handling(
    tmp_path: Path,
    sample_dataset_path: Path,
    in_memory_audit_repo: DecisionAuditRepository,
) -> None:
    """Verify run ID propagation, A8 persistence, and reconciliation metrics."""
    explicit_run_id = f"eval-run-{uuid.uuid4()}"
    out_dir = tmp_path / "recon_test_out"

    runner = EvaluationRunner(audit_repository=in_memory_audit_repo)
    result = runner.run(
        input_path=sample_dataset_path,
        output_dir=out_dir,
        mode="mock",
        run_id=explicit_run_id,
    )

    assert result.run_id == explicit_run_id
    for res in result.results:
        assert res.processing_run_id == explicit_run_id

    # Check reconciliation status
    recon = result.reconciliation
    assert recon["processing_run_id"] == explicit_run_id
    assert recon["total_processed"] == 3
    assert recon["persisted_records"] == 3
    assert recon["reconciled"] is True
    assert recon["gap"] == 0

    # Read written audit_reconciliation.json
    recon_file = out_dir / "audit_reconciliation.json"
    assert recon_file.exists()
    saved_recon = json.loads(recon_file.read_text(encoding="utf-8"))
    assert saved_recon["processing_run_id"] == explicit_run_id
    assert saved_recon["persisted_records"] == 3


# -----------------------------------------------------------------------------
# 10. Per-Ticket Processing Failure Containment
# -----------------------------------------------------------------------------
def test_per_ticket_processing_failure_containment(
    tmp_path: Path,
    sample_dataset_path: Path,
    in_memory_audit_repo: DecisionAuditRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify an unexpected crash on one ticket produces a safe fallback result."""
    orig_process_ticket = TicketProcessor.process_ticket

    def _partially_failing_process_ticket(
        self: TicketProcessor,
        ticket: UnifiedTicket,
        run_id: str | None = None,
    ) -> ProcessedTicketResult:
        if ticket.ticket_id == "DEV-0002":
            raise ValueError("Simulated unexpected processing crash on ticket 2")
        return orig_process_ticket(self, ticket, run_id=run_id)

    monkeypatch.setattr(TicketProcessor, "process_ticket", _partially_failing_process_ticket)

    out_dir = tmp_path / "partial_fail_out"
    runner = EvaluationRunner(audit_repository=in_memory_audit_repo)
    result = runner.run(input_path=sample_dataset_path, output_dir=out_dir, mode="mock")

    # All 3 tickets must have results
    assert len(result.results) == 3

    # Ticket 1 and 3 processed normally
    assert result.results[0].ticket.ticket_id == "DEV-0001"
    assert result.results[0].error is None

    assert result.results[2].ticket.ticket_id == "DEV-0003"
    assert result.results[2].error is None

    # Ticket 2 produced fallback error result
    failed_res = result.results[1]
    assert failed_res.ticket.ticket_id == "DEV-0002"
    assert failed_res.final_customer_response is None
    assert failed_res.effective_route == RouteOutcome.ESCALATE
    assert failed_res.error is not None
    assert "Simulated unexpected processing crash on ticket 2" in failed_res.error

    # Overall run status is partially_failed
    assert result.status == "partially_failed"


# -----------------------------------------------------------------------------
# 11. Retriever Similarity Threshold Consistency
# -----------------------------------------------------------------------------
def test_retriever_similarity_threshold_consistency(
    in_memory_audit_repo: DecisionAuditRepository,
) -> None:
    """Verify retriever actively uses the threshold recorded in metadata."""
    # 1. Default mock mode threshold: 0.15
    proc1, meta1 = EvaluationComponentFactory.build_processor(
        mode="mock",
        audit_repository=in_memory_audit_repo,
    )
    assert proc1.retriever is not None
    assert proc1.retriever.similarity_threshold == 0.15
    assert meta1["retriever_similarity_threshold"] == 0.15

    # 2. Explicit threshold override: 0.35
    proc2, meta2 = EvaluationComponentFactory.build_processor(
        mode="mock",
        audit_repository=in_memory_audit_repo,
        similarity_threshold_override=0.35,
    )
    assert proc2.retriever is not None
    assert proc2.retriever.similarity_threshold == 0.35
    assert meta2["retriever_similarity_threshold"] == 0.35


# -----------------------------------------------------------------------------
# 12. Privacy and Data Minimization in Artifacts
# -----------------------------------------------------------------------------
def test_privacy_data_minimization_in_artifacts(
    tmp_path: Path,
    sample_dataset_path: Path,
    in_memory_audit_repo: DecisionAuditRepository,
) -> None:
    """Verify artifacts do not contain customer PII, bodies, or credentials."""
    out_dir = tmp_path / "privacy_out"
    runner = EvaluationRunner(audit_repository=in_memory_audit_repo)
    runner.run(input_path=sample_dataset_path, output_dir=out_dir, mode="mock")

    meta_text = (out_dir / "run_metadata.json").read_text(encoding="utf-8")
    recon_text = (out_dir / "audit_reconciliation.json").read_text(encoding="utf-8")

    for raw in (meta_text, recon_text):
        assert "Alice Customer" not in raw
        assert "Hello, I have a question regarding feature" not in raw
        assert "gsk_" not in raw
        assert "sk-" not in raw


# -----------------------------------------------------------------------------
# 13. CLI Invocation Tests
# -----------------------------------------------------------------------------
def test_cli_mock_mode_success(tmp_path: Path, sample_dataset_path: Path) -> None:
    """Verify CLI succeeds with exit code 0 in mock mode."""
    out_dir = tmp_path / "cli_out"
    exit_code = cli_module.main(
        [
            "--input",
            str(sample_dataset_path),
            "--output-dir",
            str(out_dir),
            "--mode",
            "mock",
        ]
    )
    assert exit_code == 0
    assert (out_dir / "run_metadata.json").exists()
    assert (out_dir / "audit_reconciliation.json").exists()


def test_cli_non_empty_dir_fails_without_overwrite(
    tmp_path: Path,
    sample_dataset_path: Path,
) -> None:
    """Verify CLI fails with exit code 1 if output directory is non-empty without --overwrite."""
    out_dir = tmp_path / "cli_non_empty"
    out_dir.mkdir(parents=True)
    (out_dir / "existing.txt").write_text("old", encoding="utf-8")

    exit_code = cli_module.main(
        [
            "--input",
            str(sample_dataset_path),
            "--output-dir",
            str(out_dir),
            "--mode",
            "mock",
        ]
    )
    assert exit_code == 1

    # Succeeds with --overwrite
    exit_code_overwrite = cli_module.main(
        [
            "--input",
            str(sample_dataset_path),
            "--output-dir",
            str(out_dir),
            "--mode",
            "mock",
            "--overwrite",
        ]
    )
    assert exit_code_overwrite == 0


def test_cli_missing_input_file(tmp_path: Path) -> None:
    """Verify CLI returns exit code 1 if input dataset does not exist."""
    exit_code = cli_module.main(
        [
            "--input",
            str(tmp_path / "missing.json"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert exit_code == 1


# -----------------------------------------------------------------------------
# 14. Targeted Verification: Malformed Record Isolation
# -----------------------------------------------------------------------------
def test_malformed_record_isolation_in_dataset(
    tmp_path: Path,
    in_memory_audit_repo: DecisionAuditRepository,
) -> None:
    """Verify [valid A, malformed B, valid C] dataset processes A and C in order.

    Ensures the malformed record receives an explicit failure outcome without crashing the
    runner or leaking invalid data / ground truth into TicketProcessor.
    """
    mixed_dataset = [
        {
            "ticket_id": "TICKET-A",
            "channel": "email",
            "subject": "Valid ticket A",
            "body": "First valid customer inquiry.",
            "received_at": "2026-11-15T10:00:00Z",
            "customer_id": "CUST-001",
            "customer_name": "Valid Customer A",
            "customer_tier": "standard",
            "labels": {
                "intent": "api_inquiry",
                "urgency": "low",
                "expected_route": "auto_respond",
            },
        },
        {
            "ticket_id": "TICKET-MALFORMED",
            # Missing channel, missing received_at, invalid fields
            "invalid_field": True,
        },
        {
            "ticket_id": "TICKET-C",
            "channel": "email",
            "subject": "Valid ticket C",
            "body": "Second valid customer inquiry.",
            "received_at": "2026-11-15T10:02:00Z",
            "customer_id": "CUST-003",
            "customer_name": "Valid Customer C",
            "customer_tier": "standard",
            "labels": {
                "intent": "api_inquiry",
                "urgency": "low",
                "expected_route": "auto_respond",
            },
        },
    ]

    dataset_file = tmp_path / "mixed_dataset.json"
    dataset_file.write_text(json.dumps(mixed_dataset), encoding="utf-8")

    out_dir = tmp_path / "malformed_test_out"
    runner = EvaluationRunner(audit_repository=in_memory_audit_repo)
    result = runner.run(input_path=dataset_file, output_dir=out_dir, mode="mock")

    # Order must be strictly preserved: 3 records
    assert len(result.results) == 3

    # Ticket A was processed cleanly
    assert result.results[0].ticket.ticket_id == "TICKET-A"
    assert result.results[0].error is None
    assert result.results[0].persistence_status == "persisted"

    # Ticket B received explicit failure outcome
    malformed_res = result.results[1]
    assert malformed_res.ticket.ticket_id == "TICKET-MALFORMED"
    assert malformed_res.error is not None
    assert "Malformed record" in malformed_res.error
    assert malformed_res.effective_route == RouteOutcome.ESCALATE
    assert malformed_res.final_customer_response is None
    assert malformed_res.persistence_status == "failed"

    # Ticket C was processed cleanly
    assert result.results[2].ticket.ticket_id == "TICKET-C"
    assert result.results[2].error is None
    assert result.results[2].persistence_status == "persisted"

    # Run status and audit reconciliation reflect the failure gap
    assert result.status == "partially_failed"
    recon = result.reconciliation
    assert recon["total_processed"] == 3
    assert recon["persisted_records"] == 2
    assert recon["failed_or_missing_records"] == 1
    assert recon["gap"] == 1
    assert recon["reconciled"] is False


# -----------------------------------------------------------------------------
# 15. Targeted Verification: Router Confidence Threshold Consistency
# -----------------------------------------------------------------------------
def test_router_confidence_threshold_consistency(
    in_memory_audit_repo: DecisionAuditRepository,
    default_settings: Settings,
) -> None:
    """Verify active router threshold matches recorded metadata with no hidden fallbacks."""
    # 1. Default settings threshold (0.80)
    proc_default, meta_default = EvaluationComponentFactory.build_processor(
        mode="mock",
        settings=default_settings,
        audit_repository=in_memory_audit_repo,
    )
    router_active_thresh = proc_default.router.confidence_threshold_auto_respond
    recorded_thresh = meta_default["confidence_threshold_auto_respond"]
    assert router_active_thresh == 0.80
    assert recorded_thresh == 0.80
    assert router_active_thresh == recorded_thresh

    # 2. Custom settings threshold (0.88)
    custom_settings = Settings(confidence_threshold_auto_respond=0.88, _env_file=None)
    proc_custom, meta_custom = EvaluationComponentFactory.build_processor(
        mode="mock",
        settings=custom_settings,
        audit_repository=in_memory_audit_repo,
    )
    router_custom_thresh = proc_custom.router.confidence_threshold_auto_respond
    recorded_custom_thresh = meta_custom["confidence_threshold_auto_respond"]
    assert router_custom_thresh == 0.88
    assert recorded_custom_thresh == 0.88
    assert router_custom_thresh == recorded_custom_thresh


# -----------------------------------------------------------------------------
# 16. Targeted Verification: Mock KB Loading and In-Memory Retrieval
# -----------------------------------------------------------------------------
def test_mock_kb_loading_and_offline_retrieval(
    in_memory_audit_repo: DecisionAuditRepository,
) -> None:
    """Verify mock mode loads all 29 KB articles and retrieves without network dependencies."""
    kb_path = Path("data/kb/documentation.json")
    if not kb_path.exists():
        pytest.skip(
            "data/kb/documentation.json not present in workspace (excluded benchmark dataset)"
        )
    proc, meta = EvaluationComponentFactory.build_processor(
        mode="mock",
        audit_repository=in_memory_audit_repo,
    )
    assert proc.retriever is not None

    # Confirm collection count reflects all 29 documentation articles (116 section chunks)
    collection = proc.retriever.collection
    chunk_count = collection.count()
    assert chunk_count == 116

    # Verify querying returns identifiable passages
    passages = proc.retriever.retrieve(
        query="resolving authentication and credential errors",
        top_k=2,
    )
    assert len(passages) > 0
    assert passages[0].citation_id.startswith("[DOC-AUTH-")
    assert passages[0].similarity_score >= proc.retriever.similarity_threshold


def test_mock_kb_loading_with_custom_articles(
    in_memory_audit_repo: DecisionAuditRepository,
) -> None:
    """Verify mock mode can index custom articles in-memory without benchmark datasets."""
    from cloudserve.rag.models import KBArticle

    custom_articles = [
        KBArticle(
            doc_id="DOC-CUSTOM-01",
            title="Custom Troubleshooting",
            category="custom",
            content=(
                "Symptoms: High latency.\n"
                "Common causes: Network congestion.\n"
                "Resolution: Restart service.\n"
                "Notes: Monitored."
            ),
        )
    ]
    proc, meta = EvaluationComponentFactory.build_processor(
        mode="mock",
        audit_repository=in_memory_audit_repo,
        articles=custom_articles,
    )
    assert proc.retriever is not None
    assert proc.retriever.collection.count() > 0
