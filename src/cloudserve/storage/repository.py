"""Decision audit repository for durable, persistent recording of automated decisions.

Implements acceptance criterion A8:
- Durable SQLite persistence of every automated decision using SQLAlchemy 2.0.
- Application-level append-only design: authoritative UUIDv4 event IDs, no destructive updates.
- Data minimization and privacy: zero raw inquiries, customer responses,
  private notes, or credentials.
- Observable reconciliation: tracks processed vs persisted records and highlights audit gaps.
- Supports dependency injection and in-memory test databases.
"""

import hashlib
import json
import logging
import re
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from cloudserve.classification.models import IntentCategory, UrgencyLevel
from cloudserve.config import Settings, get_settings
from cloudserve.generation.grounding import GroundingValidationResult
from cloudserve.generation.models import GeneratedResponse
from cloudserve.ingestion.models import UnifiedTicket
from cloudserve.rag.models import RetrievedPassage
from cloudserve.routing.billing import BillingRiskAssessment
from cloudserve.routing.models import (
    RouteOutcome,
    RoutingDecision,
    RoutingReasonCode,
)
from cloudserve.storage.database import (
    create_sqlite_engine,
    get_session_factory,
    init_db,
)
from cloudserve.storage.models import (
    DecisionAuditLog,
    EvaluationCheckpoint,
    GenerationDiagnosticLog,
    ProcessingRun,
)

logger = logging.getLogger(__name__)


def sanitize_diagnostic_message(msg: str | None, max_length: int = 500) -> str | None:
    """Sanitize error messages and diagnostics to prevent secret leakage or raw stack traces.

    Redacts API keys, tokens, credit card numbers, and SSNs. Truncates to max_length.
    """
    if not msg:
        return None

    # Redact common credential patterns (e.g. Groq, OpenAI, bearer tokens)
    redacted = re.sub(
        r"(gsk_|sk-|key-|Bearer\s+)[a-zA-Z0-9_\-\.]{8,}",
        r"\1[REDACTED_SECRET]",
        msg,
    )
    # Redact full credit card numbers
    redacted = re.sub(
        r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",
        "[REDACTED_CARD]",
        redacted,
    )
    # Redact SSNs
    redacted = re.sub(
        r"\b\d{3}[- ]?\d{2}[- ]?\d{4}\b",
        "[REDACTED_SSN]",
        redacted,
    )

    if len(redacted) > max_length:
        return redacted[: max_length - 3] + "..."
    return redacted


def compute_customer_response_hash(response_text: str | None) -> str | None:
    """Compute SHA-256 hash of released customer response text.

    Populated ONLY when customer response is officially released to the user.
    """
    if not response_text:
        return None
    return hashlib.sha256(response_text.encode("utf-8")).hexdigest()


def build_audit_entry(
    ticket: UnifiedTicket,
    routing_decision: RoutingDecision,
    original_route: RouteOutcome,
    effective_route: RouteOutcome,
    persistence_status: str,
    delivery_status: str,
    classification_predicted_intent: IntentCategory | str,
    classification_predicted_urgency: UrgencyLevel | str,
    classification_headline_confidence: float,
    classification_intent_confidence: float,
    classification_urgency_confidence: float,
    classification_model: str,
    classification_prompt_version: str,
    is_classifier_fallback: bool = False,
    alternatives: list[Any] | None = None,
    billing_risk: BillingRiskAssessment | None = None,
    retrieved_passages: Sequence[RetrievedPassage] | None = None,
    generated_response: GeneratedResponse | None = None,
    grounding_result: GroundingValidationResult | None = None,
    final_customer_response: str | None = None,
    processing_latency_ms: float = 0.0,
    processing_error: str | None = None,
    audit_id: str | None = None,
    processing_run_id: str | None = None,
) -> DecisionAuditLog:
    """Construct an append-only DecisionAuditLog ORM entity adhering to data minimization.

    Never stores raw ticket text, subject, body, private notes, or unredacted credentials.
    """
    event_id = audit_id or str(uuid.uuid4())
    run_id = processing_run_id or str(uuid.uuid4())
    passages = list(retrieved_passages or [])

    # Minimal safe metadata
    input_char_len = len(getattr(ticket, "subject", "") or "") + len(
        getattr(ticket, "body", "") or ""
    )
    channel_str = (
        ticket.channel.value if hasattr(ticket.channel, "value") else str(ticket.channel)
    )
    tier_str = (
        ticket.customer_tier.value
        if hasattr(ticket.customer_tier, "value")
        else str(ticket.customer_tier)
    )
    fluency_str = (
        ticket.language_fluency.value
        if hasattr(ticket.language_fluency, "value")
        else str(ticket.language_fluency)
    )

    # Classification fields
    intent_str = (
        classification_predicted_intent.value
        if hasattr(classification_predicted_intent, "value")
        else str(classification_predicted_intent)
    )
    urgency_str = (
        classification_predicted_urgency.value
        if hasattr(classification_predicted_urgency, "value")
        else str(classification_predicted_urgency)
    )
    alts_json = None
    if alternatives:
        alts_json = json.dumps(
            [a.model_dump() if hasattr(a, "model_dump") else a for a in alternatives]
        )

    # Billing risk fields
    billing_subtype_str = None
    billing_rationale_str = None
    if billing_risk:
        if billing_risk.subtype:
            billing_subtype_str = (
                billing_risk.subtype.value
                if hasattr(billing_risk.subtype, "value")
                else str(billing_risk.subtype)
            )
        billing_rationale_str = billing_risk.rationale

    # Retrieval evidence fields
    retrieval_count = len(passages)
    sources_data = [
        {
            "chunk_id": p.chunk_id,
            "doc_id": p.doc_id,
            "citation_id": p.citation_id,
            "similarity_score": round(p.similarity_score, 4),
        }
        for p in passages
    ]
    top_score = (
        max((p.similarity_score for p in passages), default=None)
        if passages
        else None
    )

    input_violation = None
    if processing_error and "Input safety violation:" in processing_error:
        input_violation = processing_error.split("Input safety violation:")[-1].strip()

    input_guardrail_passed = input_violation is None

    # Generation & Grounding fields
    if generated_response is not None:
        if grounding_result and not grounding_result.is_grounded:
            generation_status = "suppressed_grounding_failure"
        else:
            generation_status = "draft_produced"
    else:
        if not input_guardrail_passed:
            generation_status = "skipped_safety_violation"
        elif routing_decision.reason_code in (
            RoutingReasonCode.MANDATORY_SECURITY_INCIDENT,
            RoutingReasonCode.MANDATORY_COMPLIANCE_REQUEST,
            RoutingReasonCode.MANDATORY_DATA_RESIDENCY,
            RoutingReasonCode.MANDATORY_BILLING_DISPUTE,
            RoutingReasonCode.MANDATORY_FEATURE_REQUEST,
            RoutingReasonCode.MANDATORY_UNCLEAR_REQUEST,
            RoutingReasonCode.MANDATORY_CLASSIFIER_FALLBACK,
            RoutingReasonCode.HIGH_URGENCY_PRODUCTION_INCIDENT,
        ):
            generation_status = "skipped_mandatory"
        else:
            generation_status = "not_generated"

    generation_model = generated_response.model_name if generated_response else None
    generation_prompt_ver = (
        generated_response.prompt_version if generated_response else None
    )

    output_grounding_passed = (
        grounding_result.is_grounded if grounding_result else None
    )
    output_grounding_reasons = (
        json.dumps(grounding_result.unsupported_reasons)
        if grounding_result and grounding_result.unsupported_reasons
        else None
    )
    citations_verified = (
        routing_decision.citations_verified
        if hasattr(routing_decision, "citations_verified")
        else (bool(grounding_result.verified_citations) if grounding_result else False)
    )
    verified_citations = (
        json.dumps(grounding_result.verified_citations)
        if grounding_result and grounding_result.verified_citations
        else None
    )

    # Delivery & response hash
    resp_hash = (
        compute_customer_response_hash(final_customer_response)
        if (delivery_status == "released" and final_customer_response)
        else None
    )

    return DecisionAuditLog(
        audit_id=event_id,
        processing_run_id=run_id,
        ticket_id=ticket.ticket_id,
        processed_at=datetime.now(timezone.utc),
        processing_latency_ms=processing_latency_ms,
        channel=channel_str,
        customer_tier=tier_str,
        customer_region=ticket.customer_region,
        language_fluency=fluency_str,
        input_char_length=input_char_len,
        predicted_intent=intent_str,
        predicted_urgency=urgency_str,
        headline_confidence=classification_headline_confidence,
        intent_confidence=classification_intent_confidence,
        urgency_confidence=classification_urgency_confidence,
        is_classifier_fallback=is_classifier_fallback,
        classification_model=classification_model,
        classification_prompt_version=classification_prompt_version,
        alternatives_json=alts_json,
        billing_risk_subtype=billing_subtype_str,
        billing_risk_rationale=billing_rationale_str,
        retrieval_passages_count=retrieval_count,
        retrieved_sources_json=json.dumps(sources_data),
        top_similarity_score=top_score,
        documentation_available=retrieval_count > 0,
        generation_status=generation_status,
        generation_model=generation_model,
        generation_prompt_version=generation_prompt_ver,
        input_guardrail_passed=input_guardrail_passed,
        input_guardrail_violation=input_violation,
        output_grounding_passed=output_grounding_passed,
        output_grounding_reasons_json=output_grounding_reasons,
        citations_verified=citations_verified,
        verified_citations_json=verified_citations,
        original_route=original_route.value,
        effective_route=effective_route.value,
        reason_code=routing_decision.reason_code.value,
        reason_description=routing_decision.reason_description,
        target_specialist_group=(
            routing_decision.escalation_payload.target_specialist_group
            if routing_decision.escalation_payload
            else None
        ),
        primary_requirement_id=routing_decision.primary_requirement_id,
        requirement_ids_json=json.dumps(routing_decision.requirement_ids),
        auto_respond_threshold_applied=float(
            getattr(routing_decision, "confidence_threshold_applied", 0.80)
        ),
        review_threshold_applied=float(
            getattr(routing_decision, "threshold_review", 0.60)
        ),
        similarity_threshold_applied=float(
            getattr(routing_decision, "retrieval_similarity_threshold_applied", 0.65)
        ),
        kill_switch_active=routing_decision.kill_switch_active,
        eligible_for_auto_respond=routing_decision.eligible_for_auto_respond,
        persistence_status=persistence_status,
        delivery_status=delivery_status,
        customer_response_hash=resp_hash,
        processing_error=sanitize_diagnostic_message(processing_error),
    )


class DecisionAuditRepository:
    """Repository handling atomic persistence and retrieval of decision audit logs.

    Provides application-level append-only recording, observable reconciliation status,
    and isolation for unit and integration testing.
    """

    def __init__(
        self,
        session_factory: sessionmaker[Session] | None = None,
        engine: Any = None,
    ) -> None:
        """Initialize the repository with a configured sessionmaker."""
        self._session_factory = session_factory
        self._engine = engine
        self._persisted_count = 0
        self._failed_count = 0
        self._run_attempts: dict[str, int] = {}
        self._run_failures: dict[str, int] = {}
        self._last_active_run_id: str | None = None

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
    ) -> "DecisionAuditRepository":
        """Instantiate repository using application Settings and SQLite engine."""
        cfg = settings or get_settings()
        engine = create_sqlite_engine(cfg.database_url)
        init_db(engine)
        factory = get_session_factory(engine)
        return cls(session_factory=factory, engine=engine)

    @classmethod
    def in_memory(cls) -> "DecisionAuditRepository":
        """Instantiate an isolated, in-memory SQLite repository for testing."""
        engine = create_sqlite_engine("sqlite:///:memory:")
        init_db(engine)
        factory = get_session_factory(engine)
        return cls(session_factory=factory, engine=engine)

    @property
    def persisted_count(self) -> int:
        """Total number of successful persistent commits during this instance lifecycle."""
        return self._persisted_count

    @property
    def failed_count(self) -> int:
        """Total number of failed persistent commit attempts during this instance lifecycle."""
        return self._failed_count

    def record_run_attempt(self, run_id: str) -> int:
        """Durably record a processing attempt for run_id in the processing_runs table (A8).

        Creates the run record if it does not exist, increments total_attempted,
        and commits immediately. Returns the updated total_attempted count.
        """
        if not hasattr(self, "_run_attempts"):
            self._run_attempts = {}
        self._last_active_run_id = run_id
        self._run_attempts[run_id] = self._run_attempts.get(run_id, 0) + 1

        if not hasattr(self, "_session_factory") or self._session_factory is None:
            raise RuntimeError(
                f"Durable audit storage unavailable: session factory is not configured "
                f"for run '{run_id}'."
            )

        session = self._session_factory()
        try:
            run = session.get(ProcessingRun, run_id)
            if run is None:
                run = ProcessingRun(
                    run_id=run_id,
                    started_at=datetime.now(timezone.utc),
                    total_attempted=1,
                    failed_attempts=0,
                    status="active",
                )
                session.add(run)
            else:
                run.total_attempted += 1
            session.commit()
            return int(run.total_attempted)
        except Exception as exc:
            session.rollback()
            logger.error(
                "AUDIT_RUN_REGISTRATION_FAILED: operation=record_run_attempt "
                "run_id=%s exception_type=%s",
                run_id,
                type(exc).__name__,
            )
            raise
        finally:
            session.close()

    def record_run_failure(self, run_id: str) -> None:
        """Durably record a failed audit attempt for run_id in the processing_runs table."""
        if not hasattr(self, "_run_failures"):
            self._run_failures = {}
        self._run_failures[run_id] = self._run_failures.get(run_id, 0) + 1

        if not hasattr(self, "_session_factory") or self._session_factory is None:
            raise RuntimeError(
                f"Durable audit storage unavailable: session factory is not configured "
                f"for run '{run_id}'."
            )

        session = self._session_factory()
        try:
            run = session.get(ProcessingRun, run_id)
            if run is not None:
                run.failed_attempts += 1
                session.commit()
            else:
                run = ProcessingRun(
                    run_id=run_id,
                    started_at=datetime.now(timezone.utc),
                    total_attempted=1,
                    failed_attempts=1,
                    status="failed",
                )
                session.add(run)
                session.commit()
        except Exception as exc:
            session.rollback()
            logger.error(
                "AUDIT_RUN_REGISTRATION_FAILED: operation=record_run_failure "
                "run_id=%s exception_type=%s",
                run_id,
                type(exc).__name__,
            )
            raise
        finally:
            session.close()

    def complete_run(self, run_id: str, status: str = "completed") -> None:
        """Mark a processing run as completed with final status in processing_runs."""
        if not hasattr(self, "_run_statuses"):
            self._run_statuses = {}
        self._run_statuses[run_id] = status

        if not hasattr(self, "_session_factory") or self._session_factory is None:
            return

        session = self._session_factory()
        try:
            run = session.get(ProcessingRun, run_id)
            if run is not None:
                run.completed_at = datetime.now(timezone.utc)
                run.status = status
                session.commit()
        except Exception as exc:
            session.rollback()
            logger.error(
                "AUDIT_RUN_COMPLETION_FAILED: operation=complete_run "
                "run_id=%s exception_type=%s",
                run_id,
                type(exc).__name__,
            )
        finally:
            session.close()

    def get_run(self, run_id: str) -> ProcessingRun | None:
        """Retrieve a processing run record by its UUIDv4 run_id."""
        if not hasattr(self, "_session_factory") or self._session_factory is None:
            return None

        session = self._session_factory()
        try:
            stmt = select(ProcessingRun).where(ProcessingRun.run_id == run_id)
            return session.execute(stmt).scalar_one_or_none()
        finally:
            session.close()

    def list_runs(self, limit: int = 50) -> list[ProcessingRun]:
        """Retrieve recent processing runs in reverse chronological order."""
        if not hasattr(self, "_session_factory") or self._session_factory is None:
            return []

        session = self._session_factory()
        try:
            stmt = select(ProcessingRun).order_by(ProcessingRun.started_at.desc()).limit(limit)
            return list(session.execute(stmt).scalars().all())
        finally:
            session.close()

    def record_decision(self, log_entry: DecisionAuditLog) -> DecisionAuditLog:
        """Atomically persist a single DecisionAuditLog record.

        Increments internal success or failure counters. If the database transaction fails,
        rolls back and raises the exception to trigger the upstream fail-closed gate.
        """
        if not hasattr(self, "_session_factory") or self._session_factory is None:
            self._failed_count += 1
            raise RuntimeError(
                "Durable audit storage unavailable: session factory is not configured."
            )

        session = self._session_factory()
        try:
            session.add(log_entry)
            if log_entry.processing_run_id:
                self._last_active_run_id = log_entry.processing_run_id
                run = session.get(ProcessingRun, log_entry.processing_run_id)
                if run is None:
                    run = ProcessingRun(
                        run_id=log_entry.processing_run_id,
                        started_at=datetime.now(timezone.utc),
                        total_attempted=1,
                        failed_attempts=0,
                        status="active",
                    )
                    session.add(run)
            session.commit()
            self._persisted_count += 1
            logger.debug(
                "Persisted audit log entry %s for ticket %s.",
                log_entry.audit_id,
                log_entry.ticket_id,
            )
            return log_entry
        except Exception as exc:
            self._failed_count += 1
            session.rollback()
            logger.error(
                "AUDIT_PERSISTENCE_FAILED: operation=record_decision "
                "ticket_id=%s exception_type=%s",
                log_entry.ticket_id,
                type(exc).__name__,
            )
            raise
        finally:
            session.close()

    def get_by_audit_id(self, audit_id: str) -> DecisionAuditLog | None:
        """Retrieve a specific audit entry by its authoritative UUIDv4 event ID."""
        if not hasattr(self, "_session_factory") or self._session_factory is None:
            return None

        session = self._session_factory()
        try:
            stmt = select(DecisionAuditLog).where(DecisionAuditLog.audit_id == audit_id)
            return session.execute(stmt).scalar_one_or_none()
        finally:
            session.close()

    def record_generation_diagnostic(
        self,
        diagnostic: GenerationDiagnosticLog,
    ) -> GenerationDiagnosticLog | None:
        """Safely persist a generation diagnostic record without breaking pipeline execution.

        Logging failures are caught, logged, and never allowed to crash ticket processing (REC-06).
        """
        if not hasattr(self, "_session_factory") or self._session_factory is None:
            logger.warning(
                "DIAGNOSTIC_STORAGE_SKIPPED: Session factory not configured. "
                "Diagnostic for ticket '%s' not persisted.",
                diagnostic.ticket_id,
            )
            return None

        session = self._session_factory()
        try:
            session.add(diagnostic)
            session.commit()
            logger.debug(
                "Persisted generation diagnostic %s for ticket %s.",
                diagnostic.diagnostic_id,
                diagnostic.ticket_id,
            )
            return diagnostic
        except Exception as exc:
            session.rollback()
            logger.error(
                "DIAGNOSTIC_PERSISTENCE_FAILED: operation=record_generation_diagnostic "
                "ticket_id=%s exception_type=%s. Diagnostic write safely suppressed.",
                diagnostic.ticket_id,
                type(exc).__name__,
            )
            return None
        finally:
            session.close()

    def get_generation_diagnostics_by_ticket_id(
        self,
        ticket_id: str,
    ) -> list[GenerationDiagnosticLog]:
        """Retrieve all generation diagnostic logs for a given ticket ID in chronological order."""
        if not hasattr(self, "_session_factory") or self._session_factory is None:
            return []

        session = self._session_factory()
        try:
            stmt = (
                select(GenerationDiagnosticLog)
                .where(GenerationDiagnosticLog.ticket_id == ticket_id)
                .order_by(GenerationDiagnosticLog.created_at.asc())
            )
            return list(session.execute(stmt).scalars().all())
        finally:
            session.close()

    def get_generation_diagnostics_by_run_id(
        self,
        run_id: str,
    ) -> list[GenerationDiagnosticLog]:
        """Retrieve all generation diagnostic logs for a given processing run ID."""
        if not hasattr(self, "_session_factory") or self._session_factory is None:
            return []

        session = self._session_factory()
        try:
            stmt = (
                select(GenerationDiagnosticLog)
                .where(GenerationDiagnosticLog.processing_run_id == run_id)
                .order_by(GenerationDiagnosticLog.created_at.asc())
            )
            return list(session.execute(stmt).scalars().all())
        finally:
            session.close()

    def get_by_ticket_id(self, ticket_id: str) -> list[DecisionAuditLog]:
        """Retrieve all audit entries associated with a ticket in chronological order.

        Demonstrates application-level append-only multi-attempt history.
        """
        if not hasattr(self, "_session_factory") or self._session_factory is None:
            return []

        session = self._session_factory()
        try:
            stmt = (
                select(DecisionAuditLog)
                .where(DecisionAuditLog.ticket_id == ticket_id)
                .order_by(DecisionAuditLog.processed_at.asc())
            )
            return list(session.execute(stmt).scalars().all())
        finally:
            session.close()

    def list_decisions(
        self,
        limit: int = 100,
        offset: int = 0,
        effective_route: str | None = None,
        reason_code: str | None = None,
    ) -> list[DecisionAuditLog]:
        """Query audit log entries with optional route/reason filters and pagination."""
        if not hasattr(self, "_session_factory") or self._session_factory is None:
            return []

        session = self._session_factory()
        try:
            stmt = select(DecisionAuditLog).order_by(
                DecisionAuditLog.processed_at.desc()
            )
            if effective_route:
                stmt = stmt.where(DecisionAuditLog.effective_route == effective_route)
            if reason_code:
                stmt = stmt.where(DecisionAuditLog.reason_code == reason_code)
            stmt = stmt.offset(offset).limit(limit)
            return list(session.execute(stmt).scalars().all())
        finally:
            session.close()

    def count_persisted_decisions(self, run_id: str | None = None) -> int:
        """Count persisted decision rows, optionally scoped to a specific processing_run_id."""
        if not hasattr(self, "_session_factory") or self._session_factory is None:
            return getattr(self, "_persisted_count", 0)
        session = self._session_factory()
        try:
            stmt = select(func.count()).select_from(DecisionAuditLog)
            if run_id is not None:
                stmt = stmt.where(DecisionAuditLog.processing_run_id == run_id)
            result = session.execute(stmt).scalar_one()
            return int(result)
        finally:
            session.close()

    def reconciliation_status(
        self,
        run_id: str | None = None,
        total_processed: int | None = None,
    ) -> dict[str, Any]:
        """Produce observable reconciliation metrics matching A8 requirements.

        Performs run-scoped reconciliation comparing attempts vs persisted records.
        Durably tracks state across restarts via the processing_runs table.
        """
        # Handle legacy positional total_processed call e.g. reconciliation_status(5)
        if isinstance(run_id, int) and total_processed is None:
            total_processed = run_id
            run_id = None

        active_run = run_id or getattr(self, "_last_active_run_id", None) or "default"
        run_rec = self.get_run(active_run) if run_id else None
        if run_id:
            try:
                persisted = self.count_persisted_decisions(run_id=active_run)
            except TypeError:
                persisted = self.count_persisted_decisions()
        else:
            persisted = self.count_persisted_decisions()

        failed_attempts = 0
        if run_id:
            if run_rec is not None:
                failed_attempts = run_rec.failed_attempts
            else:
                failures_map = getattr(self, "_run_failures", {})
                failed_attempts = failures_map.get(active_run, getattr(self, "_failed_count", 0))
        else:
            failed_attempts = getattr(self, "_failed_count", 0)

        # Determine total processing attempts for this run
        if total_processed is None:
            if run_rec is not None:
                total_processed = run_rec.total_attempted
            else:
                attempts_map = getattr(self, "_run_attempts", {})
                total_processed = attempts_map.get(active_run, max(persisted, failed_attempts))

        gap = max(0, total_processed - persisted)
        failed_or_missing = max(gap, failed_attempts)
        # In multi-attempt or retry evaluations, persisted may exceed unique tickets processed
        reconciled = (persisted >= total_processed) and (gap == 0) and (failed_attempts == 0)

        if run_rec is not None:
            run_status = run_rec.status
        else:
            statuses_map = getattr(self, "_run_statuses", {})
            if active_run in statuses_map:
                run_status = statuses_map[active_run]
            elif failed_or_missing > 0:
                run_status = "failed"
            else:
                run_status = "active"
        completed_at = (
            run_rec.completed_at.isoformat()
            if (run_rec is not None and run_rec.completed_at)
            else None
        )

        return {
            "processing_run_id": active_run,
            "total_processed": total_processed,
            "persisted_records": persisted,
            "failed_or_missing_records": failed_or_missing,
            "reconciled": reconciled,
            "gap": gap,
            "failed_in_memory_records": getattr(self, "_failed_count", 0),
            "failed_attempts": failed_attempts,
            "status": run_status,
            "completed_at": completed_at,
        }

    # -------------------------------------------------------------------------
    # Evaluation Checkpoint Management (A3.33)
    # -------------------------------------------------------------------------

    def init_checkpoints(
        self,
        run_id: str,
        ticket_ids: Sequence[str],
    ) -> None:
        """Atomically initialize checkpoints for a batch of tickets as 'pending'.

        If a checkpoint already exists for a ticket under this run_id, its existing state
        is preserved.
        """
        if not hasattr(self, "_session_factory") or self._session_factory is None:
            return

        session = self._session_factory()
        try:
            now = datetime.now(timezone.utc)
            for idx, tid in enumerate(ticket_ids):
                existing = session.get(EvaluationCheckpoint, (run_id, str(tid)))
                if existing is None:
                    cp = EvaluationCheckpoint(
                        run_id=run_id,
                        ticket_id=str(tid),
                        sequence_index=idx,
                        status="pending",
                        attempt_count=0,
                        created_at=now,
                        updated_at=now,
                    )
                    session.add(cp)
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.error("Failed to initialize checkpoints for run '%s': %s", run_id, exc)
            raise
        finally:
            session.close()

    def update_checkpoint(
        self,
        run_id: str,
        ticket_id: str,
        status: str,
        audit_id: str | None = None,
        result_json: str | None = None,
        error: str | None = None,
        increment_attempt: bool = False,
    ) -> None:
        """Atomically update the checkpoint status and payload for a single ticket."""
        if not hasattr(self, "_session_factory") or self._session_factory is None:
            return

        session = self._session_factory()
        try:
            cp = session.get(EvaluationCheckpoint, (run_id, str(ticket_id)))
            now = datetime.now(timezone.utc)
            if cp is None:
                cp = EvaluationCheckpoint(
                    run_id=run_id,
                    ticket_id=str(ticket_id),
                    sequence_index=0,
                    status=status,
                    audit_id=audit_id,
                    result_json=result_json,
                    attempt_count=1 if increment_attempt else 0,
                    last_error=sanitize_diagnostic_message(error) if error else None,
                    created_at=now,
                    updated_at=now,
                )
                session.add(cp)
            else:
                cp.status = status
                if audit_id is not None:
                    cp.audit_id = audit_id
                if result_json is not None:
                    cp.result_json = result_json
                if error is not None:
                    cp.last_error = sanitize_diagnostic_message(error)
                if increment_attempt:
                    cp.attempt_count += 1
                cp.updated_at = now
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.error(
                "Failed to update checkpoint for ticket '%s' in run '%s': %s",
                ticket_id,
                run_id,
                exc,
            )
            raise
        finally:
            session.close()

    def get_checkpoints(self, run_id: str) -> list[EvaluationCheckpoint]:
        """Retrieve all checkpoints for an evaluation run ordered by sequence_index."""
        if not hasattr(self, "_session_factory") or self._session_factory is None:
            return []

        session = self._session_factory()
        try:
            stmt = (
                select(EvaluationCheckpoint)
                .where(EvaluationCheckpoint.run_id == run_id)
                .order_by(EvaluationCheckpoint.sequence_index.asc())
            )
            return list(session.execute(stmt).scalars().all())
        finally:
            session.close()

    def get_checkpoint_counts(self, run_id: str) -> dict[str, int]:
        """Return counts of checkpoints grouped by status for run_id."""
        if not hasattr(self, "_session_factory") or self._session_factory is None:
            return {}

        session = self._session_factory()
        try:
            stmt = (
                select(EvaluationCheckpoint.status, func.count())
                .where(EvaluationCheckpoint.run_id == run_id)
                .group_by(EvaluationCheckpoint.status)
            )
            rows = session.execute(stmt).all()
            return {status: count for status, count in rows}
        finally:
            session.close()

