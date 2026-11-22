"""Data models and schemas for response generation.

Supports:
- FR-05: Customer-facing factual draft with grounded document citations.
- FR-06: Internal human agent note with context, missing information, and verification steps.
- Fail-closed handling when retrieved evidence is absent or insufficient.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr, field_validator


class RawLLMResponsePayload(BaseModel):
    """Strict schema for parsing and validating the raw LLM response JSON.

    Rejects type coercion (e.g., string "false" for boolean, string for list)
    and enforces non-empty internal agent note.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    customer_response: StrictStr | None = None
    agent_note: StrictStr
    citation_ids: list[StrictStr]
    insufficient_evidence: StrictBool
    unsupported_claims: list[StrictStr]

    @field_validator("agent_note")
    @classmethod
    def validate_agent_note_not_blank(cls, v: str) -> str:
        """Enforce that agent_note is a non-empty string."""
        if not v.strip():
            raise ValueError("agent_note must not be empty or whitespace only.")
        return v


class GeneratedResponse(BaseModel):
    """Structured output contract for response generation.

    Separates customer-facing draft responses from internal agent/specialist notes.
    Grounds all factual claims strictly in retrieved knowledge base passages.
    """

    model_config = ConfigDict(frozen=True)

    ticket_id: str = Field(
        ...,
        description="Unique identifier of the ticket being addressed.",
    )
    customer_response: str | None = Field(
        default=None,
        description=(
            "Professional, concise customer-facing response draft grounded strictly in "
            "retrieved KB documentation. None if evidence is absent or insufficient."
        ),
    )
    agent_note: str = Field(
        ...,
        description=(
            "Internal-only diagnostic note for human agents/specialists summarizing the "
            "issue, documentation used, evidence support, missing information, and next steps."
        ),
    )
    citation_ids: list[str] = Field(
        default_factory=list,
        description="Identifiable document citation IDs referenced in customer response.",
    )
    insufficient_evidence: bool = Field(
        default=False,
        description="True if retrieved passages do not contain sufficient evidence to answer.",
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Unsupported customer claims or ungrounded statements identified/rejected.",
    )
    model_name: str = Field(
        default="unknown",
        description="Name of the model provider or mock engine that generated the response.",
    )
    prompt_version: str = Field(
        default="v1",
        description="Version identifier of the generation prompt template used.",
    )
    is_fallback: bool = Field(
        default=False,
        description="True if generation failed due to model/provider error and used fallback.",
    )
    # Stage Observability Diagnostics (REC-06)
    attempt_id: str | None = Field(
        default=None,
        description="Authoritative UUIDv4 attempt identifier for generation observability.",
    )
    generation_duration_ms: float = Field(
        default=0.0,
        description="Duration of the LLM completion call in milliseconds.",
    )
    provider_status: str = Field(
        default="success",
        description="Provider status: success, timeout, rate_limit, api_error, client_error.",
    )
    provider_error_category: str | None = Field(
        default=None,
        description="Normalized category of provider failure if applicable.",
    )
    sanitized_error_details: str | None = Field(
        default=None,
        description="Sanitized error description with redacted credentials.",
    )
    raw_prompt_tokens: int | None = Field(
        default=None,
        description="Prompt tokens consumed reported by provider usage metadata.",
    )
    raw_completion_tokens: int | None = Field(
        default=None,
        description="Completion tokens consumed reported by provider usage metadata.",
    )

    @property
    def has_customer_response(self) -> bool:
        """Return True if a non-empty customer-facing response is present."""
        return self.customer_response is not None and bool(self.customer_response.strip())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation for logging or downstream storage."""
        return {
            "ticket_id": self.ticket_id,
            "customer_response": self.customer_response,
            "agent_note": self.agent_note,
            "citation_ids": self.citation_ids,
            "insufficient_evidence": self.insufficient_evidence,
            "unsupported_claims": self.unsupported_claims,
            "model_name": self.model_name,
            "prompt_version": self.prompt_version,
            "is_fallback": self.is_fallback,
            "attempt_id": self.attempt_id,
            "generation_duration_ms": self.generation_duration_ms,
            "provider_status": self.provider_status,
            "provider_error_category": self.provider_error_category,
            "sanitized_error_details": self.sanitized_error_details,
            "raw_prompt_tokens": self.raw_prompt_tokens,
            "raw_completion_tokens": self.raw_completion_tokens,
        }
