"""Data models for multi-channel ticket ingestion, normalization, and evaluation.

Implements strongly-typed Pydantic v2 schemas:
- UnifiedTicket: Runtime ticket payload containing only live operational data.
- TicketLabels, TicketHistory, EvaluationRecord: Evaluation ground truth, strictly
  isolated from live support decisioning.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChannelEnum(str, Enum):
    """Supported inbound ticket channels."""

    EMAIL = "email"
    CHAT = "chat"
    DOCS_COMMENT = "docs_comment"
    FORUM = "forum"


class CustomerTier(str, Enum):
    """Customer account subscription tiers."""

    ENTERPRISE = "enterprise"
    BUSINESS = "business"
    STANDARD = "standard"
    UNKNOWN = "unknown"


class CustomerRegion(str, Enum):
    """Geographic operating region of the customer."""

    NORTH_AMERICA = "north_america"
    EUROPE = "europe"
    ASIA_PACIFIC = "asia_pacific"
    LATIN_AMERICA = "latin_america"
    UNKNOWN = "unknown"


class LanguageFluency(str, Enum):
    """Customer language fluency level."""

    FLUENT = "fluent"
    NON_FLUENT = "non_fluent"
    UNKNOWN = "unknown"


class UnifiedTicket(BaseModel):
    """Normalized internal representation of an ingested support ticket for live runtime.

    Contains ONLY data available to the live operational support system.
    Strictly isolated from ground-truth evaluation labels and historical metrics.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    ticket_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for the ticket (e.g. DEV-0001, VAL-0001).",
    )
    channel: ChannelEnum = Field(
        ...,
        description="Originating channel (email, chat, docs_comment, or forum).",
    )
    subject: str = Field(
        default="",
        description="Ticket subject line or heading. Empty string for live chat tickets.",
    )
    body: str = Field(
        default="",
        description="Original unmodified customer inquiry body text.",
    )
    received_at: datetime = Field(
        ...,
        description="Timestamp when the ticket was received by the system.",
    )
    customer_id: str = Field(
        default="CUST-UNKNOWN",
        min_length=1,
        description="Customer account or organization identifier.",
    )
    customer_name: str = Field(
        default="Unknown Customer",
        min_length=1,
        description="Customer display name or contact name.",
    )
    customer_tier: CustomerTier = Field(
        default=CustomerTier.UNKNOWN,
        description="Subscription tier of customer account, or UNKNOWN if unspecified.",
    )
    customer_region: CustomerRegion | None = Field(
        default=None,
        description="Customer geographic region if identified.",
    )
    language_fluency: LanguageFluency = Field(
        default=LanguageFluency.UNKNOWN,
        description="Language fluency assessment, or UNKNOWN if unspecified.",
    )
    channel_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Channel-specific metadata (e.g., URL path, user reputation, thread depth).",
    )

    @field_validator("ticket_id")
    @classmethod
    def validate_ticket_id(cls, v: str) -> str:
        """Ensure ticket_id is not empty or whitespace-only."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("ticket_id cannot be empty or whitespace-only.")
        return stripped

    @field_validator("received_at")
    @classmethod
    def ensure_timezone_aware(cls, v: datetime) -> datetime:
        """Ensure datetime has timezone information attached (defaults to UTC)."""
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

    @property
    def clean_text(self) -> str:
        """Return combined subject and body text suitable for classification and retrieval."""
        if self.subject:
            return f"{self.subject}\n\n{self.body}".strip()
        return self.body.strip()


class TicketLabels(BaseModel):
    """Ground-truth evaluation labels for benchmark datasets.

    Used EXCLUSIVELY by evaluation tooling, never by live runtime response generation.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="ignore",
    )

    intent: str = Field(..., description="Expected intent class from 22 dataset categories.")
    urgency: str = Field(..., description="Expected urgency level (low, medium, high).")
    expected_route: str = Field(
        ...,
        description="Expected routing action (auto_respond, escalate).",
    )
    answerable_from_docs: bool = Field(
        default=True,
        description="Whether the ticket can be grounded using the documentation corpus.",
    )
    expected_doc_ids: list[str] = Field(
        default_factory=list,
        description="List of document IDs that a grounded answer should cite.",
    )
    must_not_auto_respond: bool = Field(
        default=False,
        description="True for classes that must never be answered automatically.",
    )


class TicketHistory(BaseModel):
    """Historical baseline metrics for benchmark datasets.

    Records what human support achieved historically. Never exposed to runtime components.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="ignore",
    )

    first_contact_resolution: bool = Field(
        ...,
        description="Whether the ticket was historically resolved on first contact.",
    )
    resolution_time_minutes: float | int = Field(
        ...,
        description="Total historical time to resolution in minutes.",
    )
    csat_rating: int = Field(
        ...,
        description="Historical customer satisfaction score (1 to 5).",
    )
    escalated: bool = Field(
        ...,
        description="Whether the ticket was historically escalated to higher-tier human support.",
    )
    repeat_contact: bool = Field(
        ...,
        description="Whether the customer contacted support again for the same issue.",
    )


class EvaluationRecord(BaseModel):
    """Composite record pairing a live-compatible UnifiedTicket with evaluation ground truth.

    Used by evaluation pipelines, benchmarks, and dataset loaders.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    ticket: UnifiedTicket = Field(
        ...,
        description="The live-compatible runtime ticket representation.",
    )
    labels: TicketLabels | None = Field(
        default=None,
        description="Expected ground-truth labels if present in the benchmark dataset.",
    )
    history: TicketHistory | None = Field(
        default=None,
        description="Historical support outcomes if present in the benchmark dataset.",
    )
