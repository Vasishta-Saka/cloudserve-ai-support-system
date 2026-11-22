"""Data models and taxonomy for ticket classification.

Defines all 22 intent categories, 3 urgency levels, candidate alternatives,
and the strongly-typed ClassificationResult contract.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class IntentCategory(str, Enum):
    """The 22 official intent categories from the CloudServe dataset taxonomy."""

    ACCOUNT_ACCESS = "account_access"
    API_KEY_ISSUE = "api_key_issue"
    API_USAGE_QUESTION = "api_usage_question"
    AUTHENTICATION_FAILURE = "authentication_failure"
    BILLING_QUERY = "billing_query"
    COMPLIANCE_REQUEST = "compliance_request"
    CONFIGURATION_HELP = "configuration_help"
    DATA_EXPORT = "data_export"
    DATA_RESIDENCY = "data_residency"
    DATABASE_ISSUE = "database_issue"
    DEPLOYMENT_FAILURE = "deployment_failure"
    FEATURE_REQUEST = "feature_request"
    INTEGRATION_HELP = "integration_help"
    ONBOARDING = "onboarding"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    QUOTA_OR_OVERAGE = "quota_or_overage"
    RATE_LIMIT = "rate_limit"
    ROLLBACK_REQUEST = "rollback_request"
    SECURITY_INCIDENT = "security_incident"
    SSO_CONFIGURATION = "sso_configuration"
    UNCLEAR_REQUEST = "unclear_request"
    WEBHOOK_ISSUE = "webhook_issue"


class UrgencyLevel(str, Enum):
    """The 3 official urgency tiers from the CloudServe dataset taxonomy."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AlternativeClassification(BaseModel):
    """Candidate alternative considered during classification.

    Preserves runner-up intent predictions and their self-reported likelihood
    to satisfy governance audit requirements.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    intent: IntentCategory = Field(
        ...,
        description="Candidate alternative intent considered.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Self-reported confidence or likelihood for this alternative.",
    )
    reason: str = Field(
        default="",
        description="Brief explanation of why this alternative was considered but not chosen.",
    )


class ClassificationResult(BaseModel):
    """Structured output contract of the Classification Engine.

    Carries predicted intent, urgency, self-reported confidences, candidate alternatives,
    and a concise auditable rationale. Decoupled from routing decisions.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ticket_id: str = Field(
        ...,
        description="Identifier of the ticket classified.",
    )
    intent: IntentCategory = Field(
        ...,
        description="Predicted primary intent from the 22 categories.",
    )
    urgency: UrgencyLevel = Field(
        ...,
        description="Predicted urgency level (low, medium, high).",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Headline confidence score (composite of intent and urgency certainty).",
    )
    intent_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Raw self-reported confidence for the primary intent prediction.",
    )
    urgency_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Raw self-reported confidence for the urgency tier prediction.",
    )
    alternatives: list[AlternativeClassification] = Field(
        default_factory=list,
        description="List of candidate alternative intents evaluated by the classifier.",
    )
    rationale: str = Field(
        ...,
        description="Concise, auditable explanation of why this classification was assigned.",
    )
    model_name: str = Field(
        ...,
        description="Model identifier that produced this classification.",
    )
    prompt_version: str = Field(
        ...,
        description="Version-controlled prompt template ID used for this prediction.",
    )
    is_fallback: bool = False
