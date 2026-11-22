"""SQLAlchemy 2.0 ORM models for decision audit logging.

Implements acceptance criterion A8:
- Durable, persistent recording of every automated decision.
- Captures all fields necessary to explain and reconstruct decisions months later.
- Application-level append-only design: rows are inserted once and never updated or deleted.
- Data minimization: never stores raw ticket bodies, subjects, credentials, customer
  responses, or private agent notes.
"""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base declarative class for CloudServe storage entities."""

    pass


class ProcessingRun(Base):
    """Persistent ledger for tracking processing runs and reconciliation state (A8).

    Enables run-scoped reconciliation across application restarts without relying
    exclusively on volatile in-memory counters.
    """

    __tablename__ = "processing_runs"

    run_id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        doc="Authoritative UUIDv4 identifier for the processing run.",
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
        doc="UTC timestamp when the processing run began.",
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="UTC timestamp when the processing run concluded.",
    )
    total_attempted: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        doc="Total number of ticket processing attempts registered for this run.",
    )
    failed_attempts: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        doc="Total number of failed audit write attempts recorded for this run.",
    )
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="active",
        doc="Lifecycle status of this run: active, completed, failed.",
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert run record to dictionary."""
        return {
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_attempted": self.total_attempted,
            "failed_attempts": self.failed_attempts,
            "status": self.status,
        }


class DecisionAuditLog(Base):
    """Application-level append-only persistent record of an automated support decision.

    Records the exact inputs, classifier predictions, KB retrieval evidence,
    guardrail outcomes, deterministic routing decisions, and delivery statuses.
    """

    __tablename__ = "decision_audit_logs"

    # 1. Authoritative Event Identity (UUIDv4)
    audit_id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        doc="Authoritative globally-unique UUIDv4 identifier for this audit event.",
    )
    processing_run_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        index=True,
        doc="Authoritative UUIDv4 identifier of the processing/evaluation run.",
    )
    ticket_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        doc="Identifier of the support ticket evaluated.",
    )
    processed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
        doc="UTC timestamp when processing concluded.",
    )
    processing_latency_ms: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        doc="Total pipeline latency in milliseconds.",
    )

    # 2. Minimal Safe Ticket Metadata (No raw subject or body text!)
    channel: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        doc="Inbound channel: email, chat, docs_comment, forum.",
    )
    customer_tier: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        index=True,
        doc="Customer subscription tier for fairness auditing.",
    )
    customer_region: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
        doc="Customer geographic region if identified.",
    )
    language_fluency: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        index=True,
        doc="Customer language fluency tier for fairness auditing.",
    )
    input_char_length: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        doc="Character length of clean inquiry text for auditing.",
    )

    # 3. Classification Outputs
    predicted_intent: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        doc="Predicted primary intent category.",
    )
    predicted_urgency: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        doc="Predicted urgency level (low, medium, high).",
    )
    headline_confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        doc="Composite headline confidence score.",
    )
    intent_confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        doc="Raw intent confidence evaluated against auto-respond threshold.",
    )
    urgency_confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        doc="Raw urgency confidence score.",
    )
    is_classifier_fallback: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        doc="True if classification used fallback due to provider failure or degradation.",
    )
    classification_model: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        doc="Model identifier that produced the classification.",
    )
    classification_prompt_version: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        doc="Prompt template version used for classification.",
    )
    alternatives_json: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc="JSON list of candidate runner-up intents and likelihoods.",
    )

    # 4. Billing Risk Assessment
    billing_risk_subtype: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        index=True,
        doc="Billing risk subtype (general_informational, account_specific, dispute, ambiguous).",
    )
    billing_risk_rationale: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc="Deterministic rationale for billing risk categorization.",
    )

    # 5. Knowledge-Base Retrieval Evidence
    retrieval_passages_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        doc="Count of knowledge-base passages retrieved above similarity threshold.",
    )
    retrieved_sources_json: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="[]",
        doc=(
            "JSON list of retrieved passage references "
            "(chunk_id, doc_id, citation_id, score, rank)."
        ),
    )
    top_similarity_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        doc="Highest similarity score among retrieved passages.",
    )
    documentation_available: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        doc="Whether qualifying documentation was retrieved for the inquiry.",
    )

    # 6. Response Generation & Grounding Validation
    generation_status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        doc=(
            "Generation status: draft_produced, suppressed_grounding_failure, "
            "skipped_mandatory, etc."
        ),
    )
    generation_model: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        doc="Model identifier used for candidate draft generation.",
    )
    generation_prompt_version: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
        doc="Prompt template version used for response generation.",
    )
    input_guardrail_passed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        doc="True if input safety scanner passed without prompt injection or PII violations.",
    )
    input_guardrail_violation: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        doc="Violation type if input safety check failed (prompt_injection, pii_detected).",
    )
    output_grounding_passed: Mapped[bool | None] = mapped_column(
        Boolean,
        nullable=True,
        doc="True if candidate claims were grounded in retrieved documentation; None if skipped.",
    )
    output_grounding_reasons_json: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc="JSON list of specific grounding failure reasons if output grounding failed.",
    )
    citations_verified: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        doc="True if cited passages in the draft match retrieved documentation.",
    )
    verified_citations_json: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc="JSON list of verified citation IDs cited in the response.",
    )

    # 7. Authoritative Routing Decisions & Delivery
    original_route: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        index=True,
        doc=(
            "Original deterministic route decided by router "
            "(auto_respond, draft_for_review, escalate)."
        ),
    )
    effective_route: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        index=True,
        doc=(
            "Effective route executed after applying persistence gates "
            "(e.g. escalate on audit failure)."
        ),
    )
    reason_code: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        doc="Standardized RoutingReasonCode enum string.",
    )
    reason_description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="Detailed audit description explaining why this action was assigned.",
    )
    target_specialist_group: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        doc="Target human specialist group if escalated.",
    )
    primary_requirement_id: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        doc="Primary PRD requirement governing the decision (FR-04, FR-05, FR-07, etc.).",
    )
    requirement_ids_json: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="[]",
        doc="JSON list of all PRD requirement IDs evaluated.",
    )

    # 8. Thresholds & System State
    auto_respond_threshold_applied: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        doc="Configured auto-respond confidence threshold applied.",
    )
    review_threshold_applied: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        doc="Configured review confidence threshold applied.",
    )
    similarity_threshold_applied: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        doc="Configured retrieval similarity threshold applied.",
    )
    kill_switch_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        doc="Whether the operational kill switch was active at decision time.",
    )
    eligible_for_auto_respond: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        doc="Whether ticket met all gating criteria for automatic response.",
    )

    # 9. Audit & Delivery Status
    persistence_status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="persisted",
        doc="Audit persistence status: persisted, failed, omitted.",
    )
    delivery_status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        doc="Delivery status: released, suppressed_audit_failure, held_for_review, escalated.",
    )
    customer_response_hash: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        doc="SHA-256 hash of released customer response (populated ONLY on released AUTO_RESPOND).",
    )
    processing_error: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc="Sanitized processing or persistence error diagnostic.",
    )

    # Composite indexes for high-frequency audit queries
    __table_args__ = (
        Index("ix_audit_ticket_processed", "ticket_id", "processed_at"),
        Index("ix_audit_original_effective", "original_route", "effective_route"),
        Index("ix_audit_route_reason", "effective_route", "reason_code"),
        Index("ix_audit_run_id", "processing_run_id"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert audit log entity to a plain dictionary for reporting and inspection."""
        return {
            "audit_id": self.audit_id,
            "processing_run_id": self.processing_run_id,
            "ticket_id": self.ticket_id,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "processing_latency_ms": self.processing_latency_ms,
            "channel": self.channel,
            "customer_tier": self.customer_tier,
            "customer_region": self.customer_region,
            "language_fluency": self.language_fluency,
            "input_char_length": self.input_char_length,
            "predicted_intent": self.predicted_intent,
            "predicted_urgency": self.predicted_urgency,
            "headline_confidence": self.headline_confidence,
            "intent_confidence": self.intent_confidence,
            "urgency_confidence": self.urgency_confidence,
            "is_classifier_fallback": self.is_classifier_fallback,
            "classification_model": self.classification_model,
            "classification_prompt_version": self.classification_prompt_version,
            "alternatives_json": self.alternatives_json,
            "billing_risk_subtype": self.billing_risk_subtype,
            "billing_risk_rationale": self.billing_risk_rationale,
            "retrieval_passages_count": self.retrieval_passages_count,
            "retrieved_sources_json": self.retrieved_sources_json,
            "top_similarity_score": self.top_similarity_score,
            "documentation_available": self.documentation_available,
            "generation_status": self.generation_status,
            "generation_model": self.generation_model,
            "generation_prompt_version": self.generation_prompt_version,
            "input_guardrail_passed": self.input_guardrail_passed,
            "input_guardrail_violation": self.input_guardrail_violation,
            "output_grounding_passed": self.output_grounding_passed,
            "output_grounding_reasons_json": self.output_grounding_reasons_json,
            "citations_verified": self.citations_verified,
            "verified_citations_json": self.verified_citations_json,
            "original_route": self.original_route,
            "effective_route": self.effective_route,
            "reason_code": self.reason_code,
            "reason_description": self.reason_description,
            "target_specialist_group": self.target_specialist_group,
            "primary_requirement_id": self.primary_requirement_id,
            "requirement_ids_json": self.requirement_ids_json,
            "auto_respond_threshold_applied": self.auto_respond_threshold_applied,
            "review_threshold_applied": self.review_threshold_applied,
            "similarity_threshold_applied": self.similarity_threshold_applied,
            "kill_switch_active": self.kill_switch_active,
            "eligible_for_auto_respond": self.eligible_for_auto_respond,
            "persistence_status": self.persistence_status,
            "delivery_status": self.delivery_status,
            "customer_response_hash": self.customer_response_hash,
            "processing_error": self.processing_error,
        }


class GenerationDiagnosticLog(Base):
    """Detailed diagnostic record for an individual LLM response generation attempt (REC-06).

    Captures model identifiers, prompt versions, timing, validation details, failure reasons,
    and stage-level latency without persisting credentials or unredacted customer PII.
    Linked to the primary decision audit log via ticket_id, processing_run_id, and audit_id.
    """

    __tablename__ = "generation_diagnostic_logs"

    # Identity and Correlation
    diagnostic_id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        doc="Authoritative UUIDv4 identifier for this generation diagnostic record.",
    )
    attempt_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        index=True,
        doc="Authoritative UUIDv4 identifier for this generation attempt.",
    )
    ticket_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        doc="Identifier of the support ticket evaluated.",
    )
    processing_run_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        index=True,
        doc="Identifier of the processing/evaluation run.",
    )
    audit_id: Mapped[str | None] = mapped_column(
        String(36),
        nullable=True,
        index=True,
        doc="Correlation link to the primary DecisionAuditLog event ID, if persisted.",
    )

    # Model & Template Metadata
    model_name: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        doc="Model identifier that executed the generation attempt.",
    )
    prompt_version: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        doc="Prompt template version used for response generation.",
    )

    # Timing & Latency Observability
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
        doc="UTC timestamp when the generation attempt began.",
    )
    generation_duration_ms: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        doc="Latency in milliseconds for the LLM completion call.",
    )
    validation_duration_ms: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        doc="Latency in milliseconds for citation and grounding validation.",
    )
    total_stage_duration_ms: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        doc="Total generation stage duration including LLM call and post-validation.",
    )

    # Provider Execution Status & Categorized Errors
    provider_status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        doc=(
            "Provider call outcome: success, timeout, rate_limit, "
            "api_error, client_error, skipped."
        ),
    )
    provider_error_category: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        doc=(
            "Normalized error category: TIMEOUT, RATE_LIMIT_429, AUTH_FAILURE, "
            "BAD_REQUEST, MALFORMED_OUTPUT."
        ),
    )
    sanitized_error_details: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc="Sanitized error description with redacted credentials and sensitive tokens.",
    )

    # Parsing & Candidate Production Status
    response_candidate_produced: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        doc="True if the model generated a parseable non-empty customer response candidate.",
    )
    raw_response_token_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        doc="Completion tokens consumed if reported by provider usage metadata.",
    )
    raw_prompt_token_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        doc="Prompt tokens consumed if reported by provider usage metadata.",
    )

    # Verification Outcomes
    citations_attempted_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        doc="Number of citations asserted in the candidate response.",
    )
    citations_verified_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        doc="Number of citations confirmed against retrieved documentation passages.",
    )
    citation_validation_passed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        doc="True if all candidate citations strictly resolved to supplied retrieved passages.",
    )
    grounding_validation_passed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        doc="True if factual statements were fully grounded in retrieved evidence.",
    )

    # Final Stage Disposition and Stable Reason Code
    final_disposition: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        index=True,
        doc="Final disposition: released, draft, suppressed, rejected, failed, or skipped.",
    )
    disposition_reason_code: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        doc=(
            "Stable reason code: SUCCESS, CITATION_MISMATCH, UNGROUNDED_CLAIM, "
            "PROVIDER_TIMEOUT, etc."
        ),
    )
    rejection_details_json: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc="JSON list of specific validation failures, unsupported claims, or diagnostic reasons.",
    )

    # Data Minimization & Opt-In Safe Snippet
    redacted_response_snippet: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc="Optional redacted snippet stored ONLY if explicit opt-in diagnostics flag is enabled.",
    )

    __table_args__ = (
        Index("ix_diag_ticket_created", "ticket_id", "created_at"),
        Index("ix_diag_run_attempt", "processing_run_id", "attempt_id"),
        Index("ix_diag_disposition_code", "final_disposition", "disposition_reason_code"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert diagnostic record to a plain dictionary for reporting and inspection."""
        return {
            "diagnostic_id": self.diagnostic_id,
            "attempt_id": self.attempt_id,
            "ticket_id": self.ticket_id,
            "processing_run_id": self.processing_run_id,
            "audit_id": self.audit_id,
            "model_name": self.model_name,
            "prompt_version": self.prompt_version,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "generation_duration_ms": self.generation_duration_ms,
            "validation_duration_ms": self.validation_duration_ms,
            "total_stage_duration_ms": self.total_stage_duration_ms,
            "provider_status": self.provider_status,
            "provider_error_category": self.provider_error_category,
            "sanitized_error_details": self.sanitized_error_details,
            "response_candidate_produced": self.response_candidate_produced,
            "raw_response_token_count": self.raw_response_token_count,
            "raw_prompt_token_count": self.raw_prompt_token_count,
            "citations_attempted_count": self.citations_attempted_count,
            "citations_verified_count": self.citations_verified_count,
            "citation_validation_passed": self.citation_validation_passed,
            "grounding_validation_passed": self.grounding_validation_passed,
            "final_disposition": self.final_disposition,
            "disposition_reason_code": self.disposition_reason_code,
            "rejection_details_json": self.rejection_details_json,
            "redacted_response_snippet": self.redacted_response_snippet,
        }


class EvaluationCheckpoint(Base):
    """Durable checkpoint tracking evaluation progress per ticket within a processing run.

    Enables crash-safe resumption of large benchmark evaluations (e.g., 500-ticket dev dataset)
    without re-executing successfully completed tickets or creating duplicate audit entries.
    """

    __tablename__ = "evaluation_checkpoints"

    run_id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        doc="Authoritative UUIDv4 identifier of the evaluation run.",
    )
    ticket_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        doc="Identifier of the support ticket being evaluated.",
    )
    sequence_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        doc="Zero-based sequence index preserving exact input order in the dataset.",
    )
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="pending",
        index=True,
        doc="Checkpoint status: pending, processing, completed, failed, paused_quota.",
    )
    audit_id: Mapped[str | None] = mapped_column(
        String(36),
        nullable=True,
        doc="UUIDv4 identifier of the final decision audit record if completed.",
    )
    result_json: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc="Serialized ProcessedTicketResult JSON for completed tickets.",
    )
    attempt_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        doc="Number of processing attempts for this ticket.",
    )
    last_error: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc="Sanitized description of the last error encountered.",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        doc="UTC timestamp when the checkpoint entry was created.",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        doc="UTC timestamp when the checkpoint entry was last updated.",
    )

    __table_args__ = (
        Index("ix_checkpoint_run_status", "run_id", "status"),
        Index("ix_checkpoint_run_seq", "run_id", "sequence_index"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert checkpoint record to dictionary."""
        return {
            "run_id": self.run_id,
            "ticket_id": self.ticket_id,
            "sequence_index": self.sequence_index,
            "status": self.status,
            "audit_id": self.audit_id,
            "attempt_count": self.attempt_count,
            "last_error": self.last_error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

