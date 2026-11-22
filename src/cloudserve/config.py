"""Application configuration management using Pydantic Settings.

Provides strongly typed, validated settings with environment variable overrides
and secret protection (preventing API keys from appearing in logs or error messages).
"""

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    GROQ = "groq"
    OPENROUTER = "openrouter"


class Settings(BaseSettings):
    """Core application settings for CloudServe AI Support System."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Runtime Environment
    environment: Literal["development", "testing", "staging", "production"] = Field(
        default="development",
        description="Execution environment mode.",
    )
    log_level: str = Field(default="INFO", description="Logging verbosity level.")

    # LLM Provider Configuration
    llm_provider: LLMProvider = Field(
        default=LLMProvider.GROQ,
        description="Active LLM provider (groq or openrouter).",
    )
    groq_api_key: SecretStr | None = Field(
        default=None,
        description="API key for Groq Cloud.",
    )
    groq_model: str = Field(
        default="openai/gpt-oss-120b",
        description="Target model identifier on Groq.",
    )
    groq_include_reasoning: bool = Field(
        default=False,
        description=(
            "Whether to include reasoning tokens from Groq reasoning models (e.g. GPT-OSS). "
            "Set to False to exclude reasoning output from model responses."
        ),
    )
    groq_timeout: float = Field(
        default=60.0,
        gt=0.0,
        description="HTTP request timeout in seconds for Groq API calls.",
    )
    groq_max_retries: int = Field(
        default=2,
        ge=0,
        description="Maximum retry attempts on transient Groq provider errors.",
    )
    openrouter_api_key: SecretStr | None = Field(
        default=None,
        description="API key for OpenRouter.",
    )
    openrouter_model: str = Field(
        default="meta-llama/llama-3.3-70b-instruct",
        description="Target model identifier on OpenRouter.",
    )

    # Embedding & Vector Store (Chroma)
    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace embedding model for passage retrieval.",
    )
    chroma_persist_directory: Path = Field(
        default=Path("./data/runtime/chroma_db"),
        description="Filesystem path for persistent Chroma vector database.",
    )
    chroma_collection_name: str = Field(
        default="cloudserve_kb",
        description="Chroma collection name for documentation passages.",
    )

    # Persistence Database (SQLite)
    database_url: str = Field(
        default="sqlite:///./data/runtime/cloudserve.db",
        description="SQLAlchemy synchronous connection URL for decision logging.",
    )

    # Knowledge Base & Dataset Directories
    kb_directory: Path = Field(
        default=Path("./data/kb"),
        description="Directory containing documentation.json.",
    )
    dev_data_directory: Path = Field(
        default=Path("./data/dev"),
        description="Directory containing development tickets.",
    )
    val_data_directory: Path = Field(
        default=Path("./data/val"),
        description="Directory containing validation tickets.",
    )

    # Decision Engine Thresholds
    confidence_threshold_auto_respond: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Minimum numeric confidence required to auto-respond.",
    )
    confidence_threshold_review: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Minimum numeric confidence required for human-review drafting.",
    )
    retrieval_top_k: int = Field(
        default=4,
        ge=1,
        description="Number of knowledge-base passages to retrieve per ticket.",
    )
    retrieval_similarity_threshold: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum similarity score required to keep a retrieved passage. "
            "Calibrated to 0.40 based on empirical MiniLM benchmark on development tickets."
        ),
    )
    enable_auto_respond: bool = Field(
        default=True,
        description=(
            "Operational kill switch to immediately disable automatic customer responses "
            "without redeployment (FR-12). When False, eligible auto-responses are held as drafts."
        ),
    )

    # Safety Guardrails
    enable_prompt_injection_shield: bool = Field(
        default=True,
        description="Enable active blocking guardrail against prompt injection.",
    )
    classification_prompt_version: str = Field(
        default="v2",
        description="Version of system prompt template used for ticket classification.",
    )
    enable_citation_verifier: bool = Field(
        default=True,
        description="Enable post-generation citation validation against retrieved passages.",
    )
    max_response_tokens: int = Field(
        default=2048,
        gt=0,
        description="Max generation token limit for LLM calls.",
    )

    # Generation Observability & Diagnostics (REC-06)
    enable_generation_diagnostics: bool = Field(
        default=True,
        description="Enable structured diagnostic persistence for generation attempts.",
    )
    diagnostics_retain_redacted_response: bool = Field(
        default=False,
        description=(
            "Opt-in security flag to retain redacted response snippet in generation diagnostics. "
            "Default is False to enforce strict zero-content data minimization."
        ),
    )

    @field_validator("confidence_threshold_review")
    @classmethod
    def validate_threshold_hierarchy(cls, v: float, info: object) -> float:
        """Ensure review threshold is not higher than auto-respond threshold."""
        # Note: Validation when both fields are available
        data = getattr(info, "data", {})
        auto_thresh = data.get("confidence_threshold_auto_respond", 0.80)
        if v > auto_thresh:
            raise ValueError(
                f"confidence_threshold_review ({v}) cannot be greater than "
                f"confidence_threshold_auto_respond ({auto_thresh})"
            )
        return v

    def get_active_api_key(self) -> SecretStr | None:
        """Safely retrieve the API key for the configured LLM provider."""
        if self.llm_provider == LLMProvider.GROQ:
            return self.groq_api_key
        elif self.llm_provider == LLMProvider.OPENROUTER:
            return self.openrouter_api_key
        return None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton instance of Settings."""
    return Settings()
