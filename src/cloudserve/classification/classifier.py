"""Ticket classification engine implementation.

Classifies incoming UnifiedTicket instances across 22 intent categories and 3 urgency
levels with candidate alternatives and calibrated self-reported confidence.
Decoupled from routing decisions and isolated from ground-truth evaluation labels.
"""

import json
import logging
import re

from pydantic import ValidationError

from cloudserve.classification.models import (
    AlternativeClassification,
    ClassificationResult,
    IntentCategory,
    UrgencyLevel,
)
from cloudserve.classification.prompts import (
    format_ticket_prompt,
    load_classification_system_prompt,
)
from cloudserve.config import Settings, get_settings
from cloudserve.generation.llm_client import (
    BaseLLMClient,
    MockLLMClient,
    ProviderQuotaExhaustedError,
)
from cloudserve.ingestion.models import UnifiedTicket

logger = logging.getLogger(__name__)


class TicketClassifier:
    """Classifies support tickets for intent, urgency, and confidence using an LLM.

    Accepts an injectable LLMClient abstraction for testability.
    Enforces deterministic fallbacks when encountering malformed outputs or provider errors.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient | None = None,
        prompt_version: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize classifier with an injectable LLM client and prompt version."""
        cfg = settings or get_settings()
        self.prompt_version = prompt_version or cfg.classification_prompt_version
        self._system_prompt = load_classification_system_prompt(version=self.prompt_version)

        if llm_client is not None:
            self.llm_client = llm_client
        else:
            # Default to MockLLMClient if no real client is provided during bootstrap
            self.llm_client = MockLLMClient(model_name=cfg.groq_model)

    def classify(self, ticket: UnifiedTicket) -> ClassificationResult:
        """Classify an inbound UnifiedTicket.

        Args:
            ticket: Ingested and normalized UnifiedTicket.

        Returns:
            ClassificationResult: Strongly-typed classification prediction.

        Raises:
            TypeError: If the input is not a UnifiedTicket.
        """
        if not isinstance(ticket, UnifiedTicket):
            raise TypeError(
                f"TicketClassifier.classify() requires a UnifiedTicket instance, "
                f"got {type(ticket).__name__}. Ground-truth evaluation records are forbidden."
            )

        user_prompt = format_ticket_prompt(ticket)

        try:
            response = self.llm_client.complete(
                prompt=user_prompt,
                system_prompt=self._system_prompt,
                temperature=0.0,
            )
            return self._parse_llm_response(
                raw_content=response.content,
                ticket_id=ticket.ticket_id,
                model_name=response.model_name,
            )
        except ProviderQuotaExhaustedError:
            # Never silently convert daily quota exhaustion into an unclear_request fallback!
            logger.error(
                "Classification halted for ticket '%s': provider daily quota exhausted.",
                ticket.ticket_id,
            )
            raise
        except Exception as exc:
            logger.warning(
                "Classification failed for ticket '%s': %s. Applying fallback.",
                ticket.ticket_id,
                exc,
            )
            return self._create_fallback_result(
                ticket_id=ticket.ticket_id,
                reason=f"Classification failed due to provider error: {exc}",
            )

    def _parse_llm_response(
        self,
        raw_content: str,
        ticket_id: str,
        model_name: str,
    ) -> ClassificationResult:
        """Parse, validate, and normalize structured JSON response from LLM."""
        clean_json_str = _extract_json_string(raw_content)

        try:
            data = json.loads(clean_json_str)
        except json.JSONDecodeError as err:
            return self._create_fallback_result(
                ticket_id=ticket_id,
                reason=f"Malformed JSON in model output: {err}",
            )

        if not isinstance(data, dict):
            return self._create_fallback_result(
                ticket_id=ticket_id,
                reason="Model output did not parse into a dictionary.",
            )

        # Validate intent
        raw_intent = data.get("intent")
        try:
            intent = IntentCategory(raw_intent)
        except ValueError:
            return self._create_fallback_result(
                ticket_id=ticket_id,
                reason=f"Unknown intent category '{raw_intent}'.",
            )

        # Validate urgency
        raw_urgency = data.get("urgency")
        try:
            urgency = UrgencyLevel(raw_urgency)
        except ValueError:
            return self._create_fallback_result(
                ticket_id=ticket_id,
                reason=f"Unknown urgency level '{raw_urgency}'.",
            )

        # Validate confidences
        try:
            intent_conf = float(data.get("intent_confidence", 0.0))
            intent_conf = max(0.0, min(1.0, intent_conf))
        except (ValueError, TypeError):
            intent_conf = 0.0

        try:
            urgency_conf = float(data.get("urgency_confidence", 0.0))
            urgency_conf = max(0.0, min(1.0, urgency_conf))
        except (ValueError, TypeError):
            urgency_conf = 0.0

        # Composite headline confidence
        headline_conf = round(intent_conf * urgency_conf, 4)

        # Parse alternatives
        alternatives: list[AlternativeClassification] = []
        raw_alts = data.get("alternatives", [])
        if isinstance(raw_alts, list):
            for item in raw_alts:
                if isinstance(item, dict):
                    alt_intent_str = item.get("intent")
                    try:
                        alt_intent = IntentCategory(alt_intent_str)
                        alt_conf = float(item.get("confidence", 0.0))
                        alt_conf = max(0.0, min(1.0, alt_conf))
                        alternatives.append(
                            AlternativeClassification(
                                intent=alt_intent,
                                confidence=alt_conf,
                                reason=str(item.get("reason", "")),
                            )
                        )
                    except (ValueError, TypeError, ValidationError):
                        continue

        rationale = str(data.get("rationale", "No rationale provided by model.")).strip()

        return ClassificationResult(
            ticket_id=ticket_id,
            intent=intent,
            urgency=urgency,
            confidence=headline_conf,
            intent_confidence=intent_conf,
            urgency_confidence=urgency_conf,
            alternatives=alternatives,
            rationale=rationale,
            model_name=model_name,
            prompt_version=f"PR-CLS-{self.prompt_version}",
            is_fallback=False,
        )

    def _create_fallback_result(self, ticket_id: str, reason: str) -> ClassificationResult:
        """Create a deterministic fallback result with zero confidence and is_fallback=True.

        Ensures downstream routing will never auto-respond to failed classifications.
        """
        return ClassificationResult(
            ticket_id=ticket_id,
            intent=IntentCategory.UNCLEAR_REQUEST,
            urgency=UrgencyLevel.MEDIUM,
            confidence=0.0,
            intent_confidence=0.0,
            urgency_confidence=0.0,
            alternatives=[],
            rationale=reason,
            model_name=self.llm_client.model_name,
            prompt_version=f"PR-CLS-{self.prompt_version}",
            is_fallback=True,
        )


def _extract_json_string(text: str) -> str:
    """Extract raw JSON text from markdown fences or surrounding commentary."""
    # Strip reasoning tags (e.g. <think>...</think> from reasoning models like GPT-OSS)
    text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
    # Match markdown code block ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Match raw JSON object { ... }
    match_raw = re.search(r"(\{.*\})", text, re.DOTALL)
    if match_raw:
        return match_raw.group(1).strip()

    return text
