"""Evidence-grounded response generation engine.

Implements Task 5.1:
- Produces customer-facing response draft grounded strictly in retrieved KB evidence.
- Produces separate internal agent notes for human agents and specialists.
- Fails closed when evidence is absent, insufficient, or citations are invalid.
- Treats ticket content as untrusted data with strict prompt-injection defenses.
- Reuses existing BaseLLMClient and MockLLMClient abstractions.
"""

import json
import logging
import re
import time
import uuid
from collections.abc import Sequence

from pydantic import ValidationError

from cloudserve.classification.models import ClassificationResult
from cloudserve.config import Settings, get_settings
from cloudserve.generation.grounding import validate_grounding
from cloudserve.generation.llm_client import (
    BaseLLMClient,
    MockLLMClient,
    ProviderQuotaExhaustedError,
)
from cloudserve.generation.models import GeneratedResponse, RawLLMResponsePayload
from cloudserve.generation.prompts import (
    format_generation_prompt,
    load_generation_system_prompt,
)
from cloudserve.ingestion.models import UnifiedTicket
from cloudserve.rag.models import RetrievedPassage
from cloudserve.storage.repository import sanitize_diagnostic_message

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """Generates evidence-grounded customer responses and internal agent notes.

    Completely decoupled from the routing engine: produces drafts and diagnostic notes,
    while DeterministicRouter remains the sole authority on routing decisions.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient | None = None,
        prompt_version: str = "v1",
        settings: Settings | None = None,
    ) -> None:
        """Initialize response generator with injectable LLM client and prompt version."""
        self.prompt_version = prompt_version
        self.settings = settings or get_settings()
        self._system_prompt = load_generation_system_prompt(version=prompt_version)

        if llm_client is not None:
            self.llm_client = llm_client
        else:
            self.llm_client = MockLLMClient(model_name=self.settings.groq_model)

    def generate_response(
        self,
        ticket: UnifiedTicket,
        retrieved_passages: Sequence[RetrievedPassage] | None = None,
        classification: ClassificationResult | None = None,
    ) -> GeneratedResponse:
        """Generate a grounded customer response draft and internal agent note.

        Args:
            ticket: Ingested and normalized UnifiedTicket.
            retrieved_passages: Ranked list of RetrievedPassage objects from vector retriever.
            classification: Optional classification result providing intent/urgency context.

        Returns:
            Strongly-typed GeneratedResponse with customer draft, agent note, and citations.

        Raises:
            TypeError: If input ticket is not a UnifiedTicket.
        """
        if not isinstance(ticket, UnifiedTicket):
            raise TypeError(
                f"ResponseGenerator requires a UnifiedTicket instance, got {type(ticket).__name__}."
            )

        passages = list(retrieved_passages or [])
        attempt_id = str(uuid.uuid4())

        # Fail-closed guard: Empty retrieved passages must NEVER produce
        # a factual customer response.
        if not passages:
            return self._build_empty_passages_response(
                ticket=ticket,
                classification=classification,
                attempt_id=attempt_id,
            )

        user_prompt = format_generation_prompt(
            ticket=ticket,
            passages=passages,
            classification=classification,
        )

        t_gen_start = time.perf_counter()
        try:
            llm_res = self.llm_client.complete(
                prompt=user_prompt,
                system_prompt=self._system_prompt,
                temperature=0.0,
                max_tokens=self.settings.max_response_tokens,
            )
            gen_duration_ms = (time.perf_counter() - t_gen_start) * 1000.0
            return self._parse_llm_response(
                raw_content=llm_res.content,
                ticket=ticket,
                passages=passages,
                model_name=llm_res.model_name,
                attempt_id=attempt_id,
                gen_duration_ms=gen_duration_ms,
                usage=llm_res.usage,
            )
        except ProviderQuotaExhaustedError:
            # Propagate daily quota exhaustion to avoid generating fallbacks or proceeding
            logger.error(
                "Response generation halted for ticket '%s': provider daily quota exhausted.",
                ticket.ticket_id,
            )
            raise
        except Exception as exc:
            gen_duration_ms = (time.perf_counter() - t_gen_start) * 1000.0
            logger.warning(
                "Response generation failed for ticket '%s': %s. Applying safe fallback.",
                ticket.ticket_id,
                exc,
            )
            # Categorize error
            exc_str = str(exc).lower()
            if "timeout" in exc_str or "timed out" in exc_str:
                error_category = "TIMEOUT"
                prov_status = "timeout"
            elif "429" in exc_str or "rate limit" in exc_str or "quota" in exc_str:
                error_category = "RATE_LIMIT_429"
                prov_status = "rate_limit"
            elif "401" in exc_str or "403" in exc_str or "auth" in exc_str or "api key" in exc_str:
                error_category = "AUTH_FAILURE"
                prov_status = "api_error"
            else:
                error_category = "PROVIDER_ERROR"
                prov_status = "api_error"

            return self._create_fallback_response(
                ticket_id=ticket.ticket_id,
                reason=f"Response generation failed due to provider error: {exc}",
                attempt_id=attempt_id,
                gen_duration_ms=gen_duration_ms,
                provider_status=prov_status,
                provider_error_category=error_category,
                sanitized_error_details=sanitize_diagnostic_message(str(exc)),
            )

    def _build_empty_passages_response(
        self,
        ticket: UnifiedTicket,
        classification: ClassificationResult | None,
        attempt_id: str | None = None,
    ) -> GeneratedResponse:
        """Construct a deterministic fail-closed response when no KB passages exist."""
        intent_desc = (
            f" (predicted intent: {classification.intent.value})" if classification else ""
        )
        agent_note = (
            f"No qualifying knowledge-base documentation was retrieved for ticket "
            f"{ticket.ticket_id}{intent_desc}. Automatic customer response is strictly blocked. "
            f"Issue summary: {ticket.clean_text[:200]}... Human specialist must investigate "
            f"and formulate appropriate manual guidance."
        )
        return GeneratedResponse(
            ticket_id=ticket.ticket_id,
            customer_response=None,
            agent_note=agent_note,
            citation_ids=[],
            insufficient_evidence=True,
            unsupported_claims=["No knowledge-base documentation retrieved for ticket."],
            model_name=self.llm_client.model_name,
            prompt_version=self.prompt_version,
            is_fallback=False,
            attempt_id=attempt_id or str(uuid.uuid4()),
            generation_duration_ms=0.0,
            provider_status="skipped",
            provider_error_category=None,
            sanitized_error_details=None,
        )

    def _parse_llm_response(
        self,
        raw_content: str,
        ticket: UnifiedTicket,
        passages: Sequence[RetrievedPassage],
        model_name: str,
        attempt_id: str | None = None,
        gen_duration_ms: float = 0.0,
        usage: dict[str, int] | None = None,
    ) -> GeneratedResponse:
        """Parse, validate, and verify the model's raw response JSON against retrieved passages."""
        aid = attempt_id or str(uuid.uuid4())
        prompt_tokens = usage.get("prompt_tokens") if usage else None
        completion_tokens = usage.get("completion_tokens") if usage else None

        cleaned = self._extract_json_block(raw_content)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.warning(
                "Malformed JSON from LLM for ticket '%s': %s. Raw: %s",
                ticket.ticket_id,
                exc,
                raw_content[:200],
            )
            return self._create_fallback_response(
                ticket_id=ticket.ticket_id,
                reason=f"Model output was not valid JSON: {exc}",
                raw_content=raw_content,
                attempt_id=aid,
                gen_duration_ms=gen_duration_ms,
                provider_status="client_error",
                provider_error_category="MALFORMED_JSON",
                sanitized_error_details=f"JSONDecodeError: {exc}",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        if not isinstance(data, dict):
            return self._create_fallback_response(
                ticket_id=ticket.ticket_id,
                reason="Model output did not produce a JSON dictionary object.",
                raw_content=raw_content,
                attempt_id=aid,
                gen_duration_ms=gen_duration_ms,
                provider_status="client_error",
                provider_error_category="MALFORMED_OBJECT",
                sanitized_error_details="Model output was not a JSON dictionary.",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        # Fix 4: Strict output schema validation using Pydantic (rejects type coercion)
        try:
            payload = RawLLMResponsePayload.model_validate(data)
        except ValidationError as exc:
            logger.warning(
                "Strict schema validation failed for ticket '%s': %s. Raw: %s",
                ticket.ticket_id,
                exc,
                raw_content[:200],
            )
            return self._create_fallback_response(
                ticket_id=ticket.ticket_id,
                reason=f"Strict schema validation failed: {exc}",
                raw_content=raw_content,
                attempt_id=aid,
                gen_duration_ms=gen_duration_ms,
                provider_status="client_error",
                provider_error_category="SCHEMA_VALIDATION_ERROR",
                sanitized_error_details=f"ValidationError: {exc}",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        raw_customer_response = payload.customer_response
        agent_note = payload.agent_note.strip()
        insufficient_evidence = payload.insufficient_evidence
        unsupported_claims = list(payload.unsupported_claims)
        raw_citations = list(payload.citation_ids)

        # Normalize customer response
        customer_response: str | None = None
        if not insufficient_evidence and raw_customer_response is not None:
            if raw_customer_response.strip():
                customer_response = raw_customer_response.strip()

        # Fix 1 & Fix 2: Conservative grounding and citation validation
        grounding_result = validate_grounding(
            customer_response=customer_response,
            citation_ids=raw_citations,
            passages=passages,
            model_unsupported_claims=unsupported_claims,
        )

        if not grounding_result.is_grounded:
            for reason in grounding_result.unsupported_reasons:
                unsupported_claims.append(reason)
            agent_note = (
                f"{agent_note}\n\nWARNING: Grounding validation failed "
                f"({grounding_result.details}). Customer draft was suppressed "
                f"and flagged for human review."
            )
            customer_response = None
            insufficient_evidence = True
            valid_citations: list[str] = []
        else:
            valid_citations = list(grounding_result.verified_citations)

        return GeneratedResponse(
            ticket_id=ticket.ticket_id,
            customer_response=customer_response,
            agent_note=agent_note,
            citation_ids=valid_citations,
            insufficient_evidence=insufficient_evidence,
            unsupported_claims=unsupported_claims,
            model_name=model_name,
            prompt_version=self.prompt_version,
            is_fallback=False,
            attempt_id=aid,
            generation_duration_ms=gen_duration_ms,
            provider_status="success",
            provider_error_category=None,
            sanitized_error_details=None,
            raw_prompt_tokens=prompt_tokens,
            raw_completion_tokens=completion_tokens,
        )

    def _extract_json_block(self, text: str) -> str:
        """Extract JSON substring, stripping markdown code block fences if present."""
        # Strip reasoning tags (e.g. <think>...</think> from reasoning models like GPT-OSS)
        text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
        # Look for markdown code fence e.g. ```json ... ```
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if fence_match:
            return fence_match.group(1).strip()

        # Look for raw outermost {...}
        brace_match = re.search(r"(\{[\s\S]*\})", text)
        if brace_match:
            return brace_match.group(1).strip()

        return text

    def _create_fallback_response(
        self,
        ticket_id: str,
        reason: str,
        raw_content: str | None = None,
        attempt_id: str | None = None,
        gen_duration_ms: float = 0.0,
        provider_status: str = "api_error",
        provider_error_category: str | None = "PROVIDER_ERROR",
        sanitized_error_details: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> GeneratedResponse:
        """Create a safe, ungrounded fallback response when generation fails."""
        raw_sample = f" Raw excerpt: {raw_content[:150]}..." if raw_content else ""
        agent_note = (
            f"Response generation fallback triggered for ticket {ticket_id}. "
            f"Reason: {reason}.{raw_sample} Customer release is strictly blocked; "
            f"human specialist review is required."
        )
        return GeneratedResponse(
            ticket_id=ticket_id,
            customer_response=None,
            agent_note=agent_note,
            citation_ids=[],
            insufficient_evidence=True,
            unsupported_claims=["Generation failed due to provider or parse error."],
            model_name=self.llm_client.model_name,
            prompt_version=self.prompt_version,
            is_fallback=True,
            attempt_id=attempt_id or str(uuid.uuid4()),
            generation_duration_ms=gen_duration_ms,
            provider_status=provider_status,
            provider_error_category=provider_error_category,
            sanitized_error_details=sanitized_error_details or sanitize_diagnostic_message(reason),
            raw_prompt_tokens=prompt_tokens,
            raw_completion_tokens=completion_tokens,
        )
