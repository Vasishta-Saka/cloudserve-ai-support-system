"""End-to-end orchestration service for the CloudServe AI Support System.

Connects input safety scanning, multi-channel classification, billing risk assessment,
knowledge-base retrieval, evidence-grounded response generation, and deterministic routing
into a unified ticket processing pipeline.
"""

import json
import logging
import time
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cloudserve.classification.classifier import TicketClassifier
from cloudserve.classification.models import (
    ClassificationResult,
    IntentCategory,
    UrgencyLevel,
)
from cloudserve.config import Settings, get_settings
from cloudserve.generation.grounding import GroundingValidationResult, validate_grounding
from cloudserve.generation.llm_client import ProviderQuotaExhaustedError
from cloudserve.generation.models import GeneratedResponse
from cloudserve.generation.response_generator import ResponseGenerator
from cloudserve.guardrails.safety import InputSafetyScanner
from cloudserve.ingestion.models import ChannelEnum, UnifiedTicket
from cloudserve.rag.models import RetrievedPassage
from cloudserve.rag.retriever import KBRetriever
from cloudserve.routing.billing import BillingRiskAssessment, assess_billing_risk
from cloudserve.routing.models import (
    BillingRiskSubtype,
    GuardrailCheckResult,
    RouteOutcome,
    RoutingDecision,
)
from cloudserve.routing.router import (
    DeterministicRouter,
    is_high_urgency_production_incident,
)
from cloudserve.storage.models import GenerationDiagnosticLog
from cloudserve.storage.repository import (
    DecisionAuditRepository,
    build_audit_entry,
    sanitize_diagnostic_message,
)

logger = logging.getLogger(__name__)


class ProcessedTicketResult(BaseModel):
    """Complete end-to-end processing result for a support ticket.

    Explicitly separates customer-facing responses from internal agent/specialist notes.
    Guarantees that final_customer_response is strictly None for non-AUTO_RESPOND routes.
    """

    model_config = ConfigDict(frozen=True)

    ticket: UnifiedTicket = Field(
        ...,
        description="The original normalized inbound support ticket.",
    )
    classification: ClassificationResult = Field(
        ...,
        description="Intent and urgency classification result.",
    )
    billing_risk: BillingRiskAssessment = Field(
        ...,
        description="Billing risk categorization (general, account, dispute, ambiguous).",
    )
    retrieved_passages: list[RetrievedPassage] = Field(
        default_factory=list,
        description="Ranked knowledge-base passages retrieved for the inquiry.",
    )
    generated_response: GeneratedResponse | None = Field(
        default=None,
        description="Candidate customer response and agent note synthesized by generator.",
    )
    grounding_result: GroundingValidationResult | None = Field(
        default=None,
        description="Grounding and citation verification outcome.",
    )
    routing_decision: RoutingDecision = Field(
        ...,
        description="The final authoritative deterministic routing decision.",
    )
    final_customer_response: str | None = Field(
        default=None,
        description=(
            "Customer-facing response. Populated ONLY when routing decision is "
            "AUTO_RESPOND; strictly None for DRAFT_FOR_REVIEW and ESCALATE."
        ),
    )
    private_agent_notes: str = Field(
        default="",
        description="Internal-only diagnostic and verification notes for support agents.",
    )
    processing_latency_ms: float = Field(
        default=0.0,
        description="Total end-to-end processing duration for this ticket in milliseconds.",
    )
    error: str | None = Field(
        default=None,
        description="Error details if processing encountered an issue or degradation.",
    )
    audit_id: str | None = Field(
        default=None,
        description="Authoritative UUIDv4 identifier of the persisted audit record.",
    )
    processing_run_id: str | None = Field(
        default=None,
        description="Authoritative UUIDv4 identifier of the processing/evaluation run.",
    )
    effective_route: RouteOutcome | None = Field(
        default=None,
        description=(
            "Effective operational route executed after applying persistence gates. "
            "For example, ESCALATE if audit commit failed on an AUTO_RESPOND decision."
        ),
    )
    persistence_status: str = Field(
        default="omitted",
        description="Audit persistence status: persisted, failed, or omitted.",
    )
    delivery_status: str = Field(
        default="undelivered",
        description=(
            "Customer delivery status: released, suppressed_audit_failure, held_for_review, "
            "escalated, or failed_audit."
        ),
    )

    @model_validator(mode="after")
    def enforce_customer_response_isolation(self) -> "ProcessedTicketResult":
        """Strictly enforce that customer responses are never released on non-AUTO_RESPOND."""
        effective = self.effective_route or self.routing_decision.route
        if (
            self.routing_decision.route != RouteOutcome.AUTO_RESPOND
            or effective != RouteOutcome.AUTO_RESPOND
        ) and self.final_customer_response is not None:
            raise ValueError(
                f"final_customer_response must be None when route is "
                f"{self.routing_decision.route.value} (effective: {effective.value}), "
                f"got non-None response."
            )
        if self.persistence_status == "failed" and self.final_customer_response is not None:
            raise ValueError(
                "final_customer_response must be None when persistence_status is 'failed'."
            )
        return self


_UNSET: Any = object()


class TicketProcessor:
    """Orchestrates end-to-end ticket processing through the CloudServe support pipeline.

    Coordinates input safety scanning, classification, billing risk assessment,
    knowledge-base retrieval, grounded response generation, and deterministic routing.
    Enforces that the router remains the sole authority for routing outcomes.
    """

    def __init__(
        self,
        classifier: TicketClassifier | None = None,
        retriever: KBRetriever | None = None,
        generator: ResponseGenerator | None = None,
        router: DeterministicRouter | None = None,
        safety_scanner: InputSafetyScanner | None = None,
        audit_repository: DecisionAuditRepository | None = _UNSET,
        processing_run_id: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize the processor with injected components or sensible defaults.

        Mandatory repository injection:
        - In normal evaluation and production paths (when audit_repository is left unset),
          automatically instantiates DecisionAuditRepository from application Settings.
        - To explicitly omit persistence, caller must explicitly pass audit_repository=None
          (reserved exclusively for isolated unit testing).
        """
        self.settings = settings or get_settings()
        self._explicit_run_id = processing_run_id
        self.processing_run_id = processing_run_id or str(uuid.uuid4())
        self.safety_scanner = safety_scanner or InputSafetyScanner(
            enable_injection_shield=self.settings.enable_prompt_injection_shield,
        )
        self.classifier = classifier or TicketClassifier(settings=self.settings)
        self.generator = generator or ResponseGenerator(settings=self.settings)
        self.router = router or DeterministicRouter(settings=self.settings)

        # Mandatory repository handling
        if audit_repository is _UNSET:
            self.audit_repository: DecisionAuditRepository | None = (
                DecisionAuditRepository.from_settings(self.settings)
            )
        else:
            self.audit_repository = audit_repository

        self._retriever_init_error: str | None = None
        if retriever is not None:
            self.retriever: KBRetriever | None = retriever
        else:
            try:
                self.retriever = KBRetriever()
            except Exception as exc:
                logger.warning(
                    "Could not initialize default KBRetriever: %s. Using empty retriever.",
                    exc,
                )
                self.retriever = None
                self._retriever_init_error = (
                    f"Retriever initialization failed: {type(exc).__name__}"
                )

    @staticmethod
    def _is_mandatory_escalation(
        classification: ClassificationResult,
        billing_risk: BillingRiskAssessment,
        ticket: UnifiedTicket | None = None,
    ) -> bool:
        """Evaluate if ticket satisfies mandatory specialist escalation policy (FR-07, REC-04)."""
        if classification.is_fallback:
            return True
        if classification.intent in (
            IntentCategory.SECURITY_INCIDENT,
            IntentCategory.COMPLIANCE_REQUEST,
            IntentCategory.DATA_RESIDENCY,
            IntentCategory.FEATURE_REQUEST,
            IntentCategory.UNCLEAR_REQUEST,
        ):
            return True
        if (
            classification.intent == IntentCategory.BILLING_QUERY
            and billing_risk.subtype == BillingRiskSubtype.DISPUTE_OR_FINANCIAL_DECISION
        ):
            return True
        if ticket is not None and is_high_urgency_production_incident(ticket, classification):
            return True
        return False

    @staticmethod
    def _construct_final_guardrail_result(
        input_guardrail: GuardrailCheckResult,
        grounding_result: GroundingValidationResult | None,
    ) -> GuardrailCheckResult:
        """Construct the combined GuardrailCheckResult from input and output validations."""
        if not input_guardrail.is_safe:
            return input_guardrail

        if grounding_result is not None:
            if not grounding_result.is_grounded:
                # Check if output failure included malicious unauthorized action claims or
                # unverified operational-status assertions (investigating or reviewing account)
                # Genuine security attacks/fraud trigger SecOps escalation (FR-04).
                unauthorized = [
                    r
                    for r in grounding_result.unsupported_reasons
                    if "Unauthorized action" in r or "Unsupported operational-status" in r
                ]
                if unauthorized:
                    return GuardrailCheckResult(
                        is_safe=False,
                        violation_type="unauthorized_action_or_status",
                        details="; ".join(unauthorized),
                    )

                # Non-malicious grounding discrepancies (including unsupported technical
                # recommendations such as undocumented 'restart' or ungrounded operational
                # parameters) route to Tier 2 Support/Engineering review draft (FR-05).
                reasons = (
                    "; ".join(grounding_result.unsupported_reasons)
                    if grounding_result.unsupported_reasons
                    else (grounding_result.details or "Output grounding validation failed.")
                )
                return GuardrailCheckResult(
                    is_safe=False,
                    violation_type="output_grounding_failed",
                    details=f"Grounding validation failed: {reasons}",
                )

            return GuardrailCheckResult(
                is_safe=True,
                violation_type=None,
                details="Input safety and output grounding checks passed.",
            )

        return GuardrailCheckResult(
            is_safe=True,
            violation_type=None,
            details="Input safety checks passed; output generation skipped or docs unavailable.",
        )

    def process_ticket(
        self,
        ticket: UnifiedTicket,
        run_id: str | None = None,
    ) -> ProcessedTicketResult:
        """Execute end-to-end processing for a single UnifiedTicket.

        Args:
            ticket: Ingested and normalized UnifiedTicket.
            run_id: Optional processing_run_id overriding self.processing_run_id.

        Returns:
            Strongly-typed ProcessedTicketResult containing all intermediate outputs,
            the authoritative routing decision, and properly isolated responses.

        Raises:
            TypeError: If input is not a UnifiedTicket.
        """
        start_time = time.perf_counter()
        is_standalone = run_id is None
        active_run = run_id or (
            self._explicit_run_id if self._explicit_run_id else str(uuid.uuid4())
        )
        self.processing_run_id = active_run

        if not isinstance(ticket, UnifiedTicket):
            raise TypeError(
                f"TicketProcessor.process_ticket requires a UnifiedTicket instance, "
                f"got {type(ticket).__name__}."
            )

        # Durably record processing attempt for this run (A8)
        if self.audit_repository is not None:
            try:
                self.audit_repository.record_run_attempt(active_run)
            except Exception as exc:
                logger.critical(
                    "AUDIT_RUN_REGISTRATION_FAILED: operation=record_run_attempt "
                    "run_id=%s ticket_id=%s exception_type=%s. "
                    "Halting normal ticket processing fail-closed.",
                    active_run,
                    getattr(ticket, "ticket_id", "unknown"),
                    type(exc).__name__,
                )
                if hasattr(self.audit_repository, "_failed_count"):
                    self.audit_repository._failed_count += 1

                durable_failure_recorded = False
                try:
                    self.audit_repository.record_run_failure(active_run)
                    durable_failure_recorded = True
                except Exception as secondary_exc:
                    logger.critical(
                        "AUDIT_RUN_REGISTRATION_FAILED: operation=record_run_failure "
                        "run_id=%s ticket_id=%s exception_type=%s. "
                        "Durable storage rejected all writes; failure could not be recorded.",
                        active_run,
                        getattr(ticket, "ticket_id", "unknown"),
                        type(secondary_exc).__name__,
                    )

                latency_ms = (time.perf_counter() - start_time) * 1000.0
                return self._create_audit_registration_error_result(
                    ticket=ticket,
                    run_id=active_run,
                    latency_ms=latency_ms,
                    durable_failure_recorded=durable_failure_recorded,
                )

        # ---------------------------------------------------------------------
        # 1. Validate and scan input text for prompt injection and PII
        # ---------------------------------------------------------------------
        input_guardrail = self.safety_scanner.scan_ticket(ticket)
        if not input_guardrail.is_safe:
            logger.warning(
                "Ticket '%s' flagged as unsafe by input scanner: %s. Halting LLM calls.",
                ticket.ticket_id,
                input_guardrail.violation_type,
            )
            # Route immediately to Security Operations; never invoke LLM on unsafe input
            fallback_classification = ClassificationResult(
                ticket_id=ticket.ticket_id,
                intent=IntentCategory.SECURITY_INCIDENT,
                urgency=UrgencyLevel.HIGH,
                confidence=0.0,
                intent_confidence=0.0,
                urgency_confidence=0.0,
                alternatives=[],
                rationale="Input safety guardrail violation; routing to Security Operations.",
                model_name="safety-scanner",
                prompt_version="v1",
                is_fallback=True,
            )
            billing_risk = assess_billing_risk(ticket)
            routing_decision = self.router.route_ticket(
                ticket=ticket,
                classification=fallback_classification,
                retrieved_passages=[],
                guardrail_result=input_guardrail,
                cited_ids=[],
            )
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            result = self._apply_persistence_gate(
                ticket=ticket,
                classification=fallback_classification,
                billing_risk=billing_risk,
                retrieved_passages=[],
                generated_response=None,
                grounding_result=None,
                routing_decision=routing_decision,
                candidate_customer_response=None,
                private_agent_notes=(
                    f"SECURITY ALERT: Input blocked by safety scanner. {input_guardrail.details}"
                ),
                latency_ms=latency_ms,
                error=f"Input safety violation: {input_guardrail.violation_type}",
                processing_run_id=active_run,
            )
            if is_standalone and self.audit_repository is not None:
                try:
                    final_status = (
                        "completed" if result.persistence_status == "persisted" else "failed"
                    )
                    self.audit_repository.complete_run(active_run, status=final_status)
                except Exception as exc:
                    logger.warning("Failed to complete standalone run %s: %s", active_run, exc)
            return result

        # ---------------------------------------------------------------------
        # 2. Classify ticket for intent and urgency
        # ---------------------------------------------------------------------
        try:
            classification = self.classifier.classify(ticket)
        except ProviderQuotaExhaustedError:
            # Re-raise daily quota exhaustion immediately to prevent silent fallbacks
            raise
        except Exception as exc:
            logger.error(
                "Classification failed unexpectedly for ticket '%s': %s",
                ticket.ticket_id,
                exc,
            )
            classification = ClassificationResult(
                ticket_id=ticket.ticket_id,
                intent=IntentCategory.UNCLEAR_REQUEST,
                urgency=UrgencyLevel.MEDIUM,
                confidence=0.0,
                intent_confidence=0.0,
                urgency_confidence=0.0,
                alternatives=[],
                rationale=f"Classification exception: {exc}",
                model_name="error-fallback",
                prompt_version="v1",
                is_fallback=True,
            )

        # ---------------------------------------------------------------------
        # 3. Assess billing risk
        # ---------------------------------------------------------------------
        billing_risk = assess_billing_risk(ticket)

        # ---------------------------------------------------------------------
        # 4. Evaluate mandatory escalation conditions
        # ---------------------------------------------------------------------
        is_mandatory = self._is_mandatory_escalation(
            classification=classification,
            billing_risk=billing_risk,
            ticket=ticket,
        )

        # ---------------------------------------------------------------------
        # 5. Retrieve relevant KB passages (including for mandatory escalation)
        # ---------------------------------------------------------------------
        retrieved_passages: list[RetrievedPassage] = []
        retrieval_error: str | None = None
        if self.retriever is not None:
            try:
                query = f"{ticket.subject} {ticket.body}".strip()
                intent_val = (
                    classification.intent.value
                    if classification and classification.intent
                    else None
                )
                try:
                    retrieved_passages = self.retriever.retrieve(
                        query=query,
                        top_k=self.settings.retrieval_top_k,
                        intent_hint=intent_val,
                    )
                except TypeError:
                    # Backward compatibility for mock/custom retrievers not accepting intent_hint
                    retrieved_passages = self.retriever.retrieve(
                        query=query,
                        top_k=self.settings.retrieval_top_k,
                    )
            except Exception as exc:
                logger.error("Retrieval failed for ticket '%s': %s", ticket.ticket_id, exc)
                retrieval_error = f"Retrieval exception: {exc}"
                retrieved_passages = []
        elif self._retriever_init_error is not None:
            retrieval_error = self._retriever_init_error

        # ---------------------------------------------------------------------
        # 6. Response generation and grounding validation
        # ---------------------------------------------------------------------
        generated_response: GeneratedResponse | None = None
        grounding_result: GroundingValidationResult | None = None

        if is_mandatory:
            # Skip ordinary customer-response generation for mandatory escalation categories
            logger.info(
                "Ticket '%s' is a mandatory escalation (%s); skipping response generation.",
                ticket.ticket_id,
                classification.intent.value,
            )
            agent_notes = (
                f"Mandatory escalation: {classification.intent.value}. "
                f"Billing risk subtype: {billing_risk.subtype.value}. "
                f"Ordinary customer response skipped to prevent unverified advice."
            )
        else:
            # Eligible non-mandatory inquiry: generate candidate response
            try:
                generated_response = self.generator.generate_response(
                    ticket=ticket,
                    retrieved_passages=retrieved_passages,
                    classification=classification,
                )
            except ProviderQuotaExhaustedError:
                # Re-raise daily quota exhaustion immediately
                raise
            except Exception as exc:
                logger.error(
                    "Response generation failed for ticket '%s': %s",
                    ticket.ticket_id,
                    exc,
                )
                generated_response = GeneratedResponse(
                    ticket_id=ticket.ticket_id,
                    customer_response=None,
                    agent_note=f"Generation failed due to provider error: {exc}",
                    citation_ids=[],
                    insufficient_evidence=True,
                    model_name="error-fallback",
                    is_fallback=True,
                )

            # Grounding and citation validation
            t_val_start = time.perf_counter()
            if generated_response.customer_response is not None:
                grounding_result = validate_grounding(
                    customer_response=generated_response.customer_response,
                    citation_ids=generated_response.citation_ids,
                    passages=retrieved_passages,
                    model_unsupported_claims=generated_response.unsupported_claims,
                )
            elif (
                retrieved_passages
                and generated_response.unsupported_claims
                and not generated_response.is_fallback
            ):
                # Customer draft was evaluated against passages and suppressed by generator
                grounding_result = GroundingValidationResult(
                    is_grounded=False,
                    verified_citations=[],
                    unsupported_reasons=list(generated_response.unsupported_claims),
                    details="Customer response draft was suppressed due to grounding failure.",
                )
            else:
                grounding_result = None
            val_duration_ms = (time.perf_counter() - t_val_start) * 1000.0

            agent_notes = generated_response.agent_note

            # 6b. Record structured generation diagnostic (REC-06)
            if self.settings.enable_generation_diagnostics and self.audit_repository is not None:
                try:
                    self._record_generation_diagnostic_safe(
                        ticket=ticket,
                        run_id=active_run,
                        generated_response=generated_response,
                        grounding_result=grounding_result,
                        val_duration_ms=val_duration_ms,
                    )
                except Exception as diag_exc:
                    logger.warning(
                        "Generation diagnostic logging failed safely for ticket '%s': %s",
                        ticket.ticket_id,
                        diag_exc,
                    )

        # ---------------------------------------------------------------------
        # 7. Construct final guardrail check result and verified citation IDs
        # ---------------------------------------------------------------------
        final_guardrail = self._construct_final_guardrail_result(
            input_guardrail=input_guardrail,
            grounding_result=grounding_result,
        )

        if grounding_result is not None and grounding_result.is_grounded:
            effective_cited_ids = grounding_result.verified_citations
        else:
            effective_cited_ids = []

        # ---------------------------------------------------------------------
        # 8. Deterministic Routing (Sole authority for the final route)
        # ---------------------------------------------------------------------
        routing_decision = self.router.route_ticket(
            ticket=ticket,
            classification=classification,
            retrieved_passages=retrieved_passages,
            guardrail_result=final_guardrail,
            cited_ids=effective_cited_ids,
        )

        # ---------------------------------------------------------------------
        # 9. Isolate candidate customer-facing response vs internal agent notes
        # ---------------------------------------------------------------------
        if routing_decision.route == RouteOutcome.AUTO_RESPOND:
            candidate_customer_response = (
                generated_response.customer_response if generated_response is not None else None
            )
        else:
            candidate_customer_response = None

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        result = self._apply_persistence_gate(
            ticket=ticket,
            classification=classification,
            billing_risk=billing_risk,
            retrieved_passages=retrieved_passages,
            generated_response=generated_response,
            grounding_result=grounding_result,
            routing_decision=routing_decision,
            candidate_customer_response=candidate_customer_response,
            private_agent_notes=agent_notes,
            latency_ms=latency_ms,
            error=retrieval_error,
            processing_run_id=active_run,
        )
        if is_standalone and self.audit_repository is not None:
            try:
                final_status = "completed" if result.persistence_status == "persisted" else "failed"
                self.audit_repository.complete_run(active_run, status=final_status)
            except Exception as exc:
                logger.warning("Failed to complete standalone run %s: %s", active_run, exc)
        return result

    def _apply_persistence_gate(
        self,
        ticket: UnifiedTicket,
        classification: ClassificationResult,
        billing_risk: BillingRiskAssessment,
        retrieved_passages: list[RetrievedPassage],
        generated_response: GeneratedResponse | None,
        grounding_result: GroundingValidationResult | None,
        routing_decision: RoutingDecision,
        candidate_customer_response: str | None,
        private_agent_notes: str,
        latency_ms: float,
        error: str | None,
        processing_run_id: str | None = None,
    ) -> ProcessedTicketResult:
        """Apply fail-closed persistence gate and record decision audit log (A8).

        Guarantees:
        - If self.audit_repository is None, passes through without persistent logging.
        - If persistent logging succeeds, releases customer response for AUTO_RESPOND.
        - If persistent logging fails, strictly suppresses customer response (None),
          sets effective_route=ESCALATE, delivery_status="suppressed_audit_failure",
          preserves the original routing_decision intact, and records the persistence error.
        - Non-AUTO_RESPOND routes never release customer responses.
        """
        original_route = routing_decision.route

        if self.audit_repository is None:
            effective_route = original_route
            delivery_status = (
                "released"
                if effective_route == RouteOutcome.AUTO_RESPOND
                else (
                    "held_for_review"
                    if effective_route == RouteOutcome.DRAFT_FOR_REVIEW
                    else "escalated"
                )
            )
            return ProcessedTicketResult(
                ticket=ticket,
                classification=classification,
                billing_risk=billing_risk,
                retrieved_passages=retrieved_passages,
                generated_response=generated_response,
                grounding_result=grounding_result,
                routing_decision=routing_decision,
                final_customer_response=candidate_customer_response,
                private_agent_notes=private_agent_notes,
                processing_latency_ms=latency_ms,
                error=error,
                audit_id=None,
                processing_run_id=processing_run_id,
                effective_route=effective_route,
                persistence_status="omitted",
                delivery_status=delivery_status,
            )

        audit_id = str(uuid.uuid4())
        if original_route == RouteOutcome.AUTO_RESPOND:
            tentative_effective = RouteOutcome.AUTO_RESPOND
            tentative_delivery = "released"
            tentative_response = candidate_customer_response
        elif original_route == RouteOutcome.DRAFT_FOR_REVIEW:
            tentative_effective = RouteOutcome.DRAFT_FOR_REVIEW
            tentative_delivery = "held_for_review"
            tentative_response = None
        else:
            tentative_effective = RouteOutcome.ESCALATE
            tentative_delivery = "escalated"
            tentative_response = None

        log_entry = build_audit_entry(
            ticket=ticket,
            routing_decision=routing_decision,
            original_route=original_route,
            effective_route=tentative_effective,
            persistence_status="persisted",
            delivery_status=tentative_delivery,
            classification_predicted_intent=classification.intent,
            classification_predicted_urgency=classification.urgency,
            classification_headline_confidence=classification.confidence,
            classification_intent_confidence=classification.intent_confidence,
            classification_urgency_confidence=classification.urgency_confidence,
            classification_model=classification.model_name,
            classification_prompt_version=classification.prompt_version,
            is_classifier_fallback=classification.is_fallback,
            alternatives=classification.alternatives,
            billing_risk=billing_risk,
            retrieved_passages=retrieved_passages,
            generated_response=generated_response,
            grounding_result=grounding_result,
            final_customer_response=tentative_response,
            processing_latency_ms=latency_ms,
            processing_error=error,
            audit_id=audit_id,
            processing_run_id=processing_run_id,
        )

        try:
            self.audit_repository.record_decision(log_entry)
            return ProcessedTicketResult(
                ticket=ticket,
                classification=classification,
                billing_risk=billing_risk,
                retrieved_passages=retrieved_passages,
                generated_response=generated_response,
                grounding_result=grounding_result,
                routing_decision=routing_decision,
                final_customer_response=tentative_response,
                private_agent_notes=private_agent_notes,
                processing_latency_ms=latency_ms,
                error=error,
                audit_id=audit_id,
                processing_run_id=processing_run_id,
                effective_route=tentative_effective,
                persistence_status="persisted",
                delivery_status=tentative_delivery,
            )
        except Exception as exc:
            logger.critical(
                "AUDIT_PERSISTENCE_FAILED: operation=record_decision "
                "ticket_id=%s exception_type=%s. Suppressing response.",
                ticket.ticket_id,
                type(exc).__name__,
            )
            pers_err = f"Audit persistence failed: {type(exc).__name__}: {exc}"
            combined_err = f"{error}; {pers_err}" if error else pers_err

            if original_route == RouteOutcome.AUTO_RESPOND:
                failed_effective = RouteOutcome.ESCALATE
                failed_delivery = "suppressed_audit_failure"
            else:
                failed_effective = original_route
                failed_delivery = "failed_audit"

            notes = (
                f"{private_agent_notes}\n[AUDIT FAILURE]: Response release blocked "
                f"due to persistent audit failure: {type(exc).__name__}."
            ).strip()

            try:
                if processing_run_id and self.audit_repository is not None:
                    self.audit_repository.record_run_failure(processing_run_id)
            except Exception:
                pass

            return ProcessedTicketResult(
                ticket=ticket,
                classification=classification,
                billing_risk=billing_risk,
                retrieved_passages=retrieved_passages,
                generated_response=generated_response,
                grounding_result=grounding_result,
                routing_decision=routing_decision,
                final_customer_response=None,
                private_agent_notes=notes,
                processing_latency_ms=latency_ms,
                error=combined_err,
                audit_id=audit_id,
                processing_run_id=processing_run_id,
                effective_route=failed_effective,
                persistence_status="failed",
                delivery_status=failed_delivery,
            )

    def _record_generation_diagnostic_safe(
        self,
        ticket: UnifiedTicket,
        run_id: str,
        generated_response: GeneratedResponse,
        grounding_result: GroundingValidationResult | None,
        val_duration_ms: float,
    ) -> None:
        """Construct and record a generation diagnostic record fail-safely (REC-06)."""
        if self.audit_repository is None:
            return

        attempt_id = generated_response.attempt_id or str(uuid.uuid4())
        diag_id = str(uuid.uuid4())
        gen_duration_ms = generated_response.generation_duration_ms
        total_duration_ms = gen_duration_ms + val_duration_ms

        # Determine validation outcomes
        citation_passed = False
        grounding_passed = False
        rejection_reasons: list[str] = []

        if grounding_result is not None:
            citation_passed = bool(grounding_result.verified_citations)
            grounding_passed = grounding_result.is_grounded
            if not grounding_passed:
                rejection_reasons.extend(grounding_result.unsupported_reasons)
        elif generated_response.unsupported_claims:
            rejection_reasons.extend(generated_response.unsupported_claims)

        # Determine final disposition and stable reason code
        prov_status = generated_response.provider_status
        prov_err_cat = generated_response.provider_error_category

        if generated_response.is_fallback:
            final_disp = "failed"
            if prov_err_cat == "TIMEOUT":
                disp_code = "PROVIDER_TIMEOUT"
            elif prov_err_cat == "RATE_LIMIT_429":
                disp_code = "PROVIDER_RATE_LIMIT"
            elif prov_err_cat in ("MALFORMED_JSON", "MALFORMED_OBJECT", "SCHEMA_VALIDATION_ERROR"):
                disp_code = "MALFORMED_MODEL_OUTPUT"
            else:
                disp_code = "PROVIDER_ERROR"
        elif not generated_response.has_customer_response:
            if generated_response.insufficient_evidence:
                final_disp = "suppressed"
                has_grounding_err = any(
                    "Grounding validation failed" in r for r in rejection_reasons
                )
                if has_grounding_err or (not grounding_passed and grounding_result is not None):
                    disp_code = "GROUNDING_VALIDATION_FAILURE"
                elif any("Citation" in r or "citation" in r for r in rejection_reasons):
                    disp_code = "CITATION_VALIDATION_FAILURE"
                else:
                    disp_code = "INSUFFICIENT_EVIDENCE"
            else:
                final_disp = "rejected"
                disp_code = "EMPTY_MODEL_OUTPUT"
        else:
            if grounding_passed:
                final_disp = "released"
                disp_code = "SUCCESS"
            else:
                final_disp = "suppressed"
                disp_code = "GROUNDING_VALIDATION_FAILURE"

        # Safe redacted response snippet (opt-in ONLY)
        redacted_snippet: str | None = None
        should_retain = self.settings.diagnostics_retain_redacted_response and bool(
            generated_response.customer_response
        )
        if should_retain and generated_response.customer_response:
            redacted_snippet = sanitize_diagnostic_message(
                generated_response.customer_response,
                max_length=200,
            )

        verified_count = len(grounding_result.verified_citations) if grounding_result else 0
        sanitized_details = sanitize_diagnostic_message(generated_response.sanitized_error_details)

        diag_entry = GenerationDiagnosticLog(
            diagnostic_id=diag_id,
            attempt_id=attempt_id,
            ticket_id=ticket.ticket_id,
            processing_run_id=run_id,
            audit_id=None,  # Primary audit entry is minted in persistence gate
            model_name=generated_response.model_name,
            prompt_version=generated_response.prompt_version,
            created_at=datetime.now(timezone.utc),
            generation_duration_ms=gen_duration_ms,
            validation_duration_ms=val_duration_ms,
            total_stage_duration_ms=total_duration_ms,
            provider_status=prov_status,
            provider_error_category=prov_err_cat,
            sanitized_error_details=sanitized_details,
            response_candidate_produced=generated_response.has_customer_response,
            raw_response_token_count=generated_response.raw_completion_tokens,
            raw_prompt_token_count=generated_response.raw_prompt_tokens,
            citations_attempted_count=len(generated_response.citation_ids),
            citations_verified_count=verified_count,
            citation_validation_passed=citation_passed,
            grounding_validation_passed=grounding_passed,
            final_disposition=final_disp,
            disposition_reason_code=disp_code,
            rejection_details_json=json.dumps(rejection_reasons) if rejection_reasons else None,
            redacted_response_snippet=redacted_snippet,
        )

        self.audit_repository.record_generation_diagnostic(diag_entry)

    def process_batch(
        self,
        tickets: Sequence[UnifiedTicket],
        run_id: str | None = None,
        on_ticket_complete: Any = None,
    ) -> list[ProcessedTicketResult]:
        """Process a sequence of tickets independently, preserving input order.

        Ensures that all tickets in one batch share the exact same run ID.
        Updates completed_at and final run status (completed, partially_failed, failed)
        upon batch conclusion.
        """
        active_run = run_id or (
            self._explicit_run_id if self._explicit_run_id else str(uuid.uuid4())
        )
        self.processing_run_id = active_run
        results: list[ProcessedTicketResult] = []
        for ticket in tickets:
            try:
                result = self.process_ticket(ticket, run_id=active_run)
            except ProviderQuotaExhaustedError:
                # Never catch daily quota exhaustion inside process_batch as a generic ticket error!
                raise
            except Exception as exc:
                logger.error(
                    "Unexpected processing exception on ticket '%s': %s",
                    getattr(ticket, "ticket_id", "unknown"),
                    exc,
                )
                result = self._create_error_result(ticket, exc, run_id=active_run)
            results.append(result)
            if on_ticket_complete is not None:
                try:
                    on_ticket_complete(ticket, result)
                except Exception:
                    pass

        if self.audit_repository is not None:
            try:
                failed_count = sum(1 for r in results if r.persistence_status == "failed")
                if failed_count == 0:
                    status = "completed"
                elif failed_count == len(results):
                    status = "failed"
                else:
                    status = "partially_failed"
                self.audit_repository.complete_run(active_run, status=status)
            except Exception as exc:
                logger.warning("Failed to complete batch run %s: %s", active_run, exc)

        return results

    def _create_audit_registration_error_result(
        self,
        ticket: Any,
        run_id: str,
        latency_ms: float,
        durable_failure_recorded: bool = False,
    ) -> ProcessedTicketResult:
        """Create a safe fallback ProcessedTicketResult when audit run attempt registration fails.

        Guarantees that normal ticket processing is halted, no LLM calls are made,
        no automated customer response is released (None), and a clear sanitized operational
        failure is surfaced without leaking raw exception messages or stack traces.
        """
        diagnostic_code = "AUDIT_RUN_REGISTRATION_FAILED"
        safe_ticket = (
            ticket
            if isinstance(ticket, UnifiedTicket)
            else UnifiedTicket(
                ticket_id=getattr(ticket, "ticket_id", "ERR-UNKNOWN"),
                channel=getattr(ticket, "channel", ChannelEnum.EMAIL),
                subject="Audit Failure Ticket",
                body=str(getattr(ticket, "body", "")),
                received_at=datetime.now(timezone.utc),
            )
        )
        fallback_classification = ClassificationResult(
            ticket_id=safe_ticket.ticket_id,
            intent=IntentCategory.UNCLEAR_REQUEST,
            urgency=UrgencyLevel.HIGH,
            confidence=0.0,
            intent_confidence=0.0,
            urgency_confidence=0.0,
            alternatives=[],
            rationale=f"Execution halted fail-closed due to {diagnostic_code}.",
            model_name="error-fallback",
            prompt_version="v1",
            is_fallback=True,
        )
        billing_risk = assess_billing_risk(safe_ticket)
        guardrail = GuardrailCheckResult(
            is_safe=False,
            violation_type="audit_registration_failed",
            details=f"Operational audit registration failed: {diagnostic_code}.",
        )
        routing_decision = self.router.route_ticket(
            ticket=safe_ticket,
            classification=fallback_classification,
            retrieved_passages=[],
            guardrail_result=guardrail,
            cited_ids=[],
        )

        if durable_failure_recorded:
            notes = (
                f"[{diagnostic_code}]: Durable run attempt registration failed. "
                f"Failure state durably recorded. Customer response suppressed fail-closed."
            )
            error_msg = f"{diagnostic_code}: Run attempt registration failed in storage."
        else:
            notes = (
                f"[{diagnostic_code}]: Durable run attempt registration and failure "
                f"recording both failed (storage unavailable). Attempt unpersisted. "
                f"Customer response suppressed fail-closed."
            )
            error_msg = (
                f"{diagnostic_code}: Storage rejected both attempt registration "
                f"and failure recording."
            )

        return ProcessedTicketResult(
            ticket=safe_ticket,
            classification=fallback_classification,
            billing_risk=billing_risk,
            retrieved_passages=[],
            generated_response=None,
            grounding_result=None,
            routing_decision=routing_decision,
            final_customer_response=None,
            private_agent_notes=notes,
            processing_latency_ms=latency_ms,
            error=error_msg,
            audit_id=str(uuid.uuid4()),
            processing_run_id=run_id,
            effective_route=RouteOutcome.ESCALATE,
            persistence_status="failed",
            delivery_status="suppressed_audit_failure",
        )

    def _create_error_result(
        self,
        ticket: Any,
        exc: Exception,
        run_id: str | None = None,
    ) -> ProcessedTicketResult:
        """Create a safe fallback ProcessedTicketResult when an unhandled exception occurs."""
        safe_ticket = (
            ticket
            if isinstance(ticket, UnifiedTicket)
            else UnifiedTicket(
                ticket_id=getattr(ticket, "ticket_id", "ERR-UNKNOWN"),
                channel=getattr(ticket, "channel", ChannelEnum.EMAIL),
                subject="Error Ticket",
                body=str(getattr(ticket, "body", "")),
                received_at=datetime.now(timezone.utc),
            )
        )
        fallback_classification = ClassificationResult(
            ticket_id=safe_ticket.ticket_id,
            intent=IntentCategory.UNCLEAR_REQUEST,
            urgency=UrgencyLevel.HIGH,
            confidence=0.0,
            intent_confidence=0.0,
            urgency_confidence=0.0,
            alternatives=[],
            rationale=f"Processing failed with unexpected exception: {exc}",
            model_name="error-fallback",
            prompt_version="v1",
            is_fallback=True,
        )
        billing_risk = assess_billing_risk(safe_ticket)
        guardrail = GuardrailCheckResult(
            is_safe=False,
            violation_type="system_processing_error",
            details=f"Unexpected processing exception: {exc}",
        )
        routing_decision = self.router.route_ticket(
            ticket=safe_ticket,
            classification=fallback_classification,
            retrieved_passages=[],
            guardrail_result=guardrail,
            cited_ids=[],
        )
        return self._apply_persistence_gate(
            ticket=safe_ticket,
            classification=fallback_classification,
            billing_risk=billing_risk,
            retrieved_passages=[],
            generated_response=None,
            grounding_result=None,
            routing_decision=routing_decision,
            candidate_customer_response=None,
            private_agent_notes=f"CRITICAL ERROR: Processing failed with exception: {exc}",
            latency_ms=0.0,
            error=str(exc),
            processing_run_id=run_id or self.processing_run_id,
        )


def create_ticket_processor(
    settings: Settings | None = None,
    audit_repository: DecisionAuditRepository | None = None,
    classifier: TicketClassifier | None = None,
    retriever: KBRetriever | None = None,
    generator: ResponseGenerator | None = None,
    router: DeterministicRouter | None = None,
    safety_scanner: InputSafetyScanner | None = None,
    processing_run_id: str | None = None,
) -> TicketProcessor:
    """Factory creating a production-ready TicketProcessor with all default services wired.

    Ensures that DecisionAuditRepository is constructed and injected from application Settings
    if not explicitly provided, guaranteeing that all automated decisions are persistently
    audited (A8).
    """
    cfg = settings or get_settings()
    repo = audit_repository or DecisionAuditRepository.from_settings(cfg)
    return TicketProcessor(
        classifier=classifier,
        retriever=retriever,
        generator=generator,
        router=router,
        safety_scanner=safety_scanner,
        audit_repository=repo,
        processing_run_id=processing_run_id,
        settings=cfg,
    )
