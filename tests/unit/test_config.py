"""Unit tests for application configuration management.

Tests default settings, environment overrides, threshold validations,
and secret masking.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from cloudserve.config import LLMProvider, Settings


def test_default_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify default settings provide sensible local development defaults."""
    monkeypatch.delenv("CONFIDENCE_THRESHOLD_AUTO_RESPOND", raising=False)
    monkeypatch.delenv("CONFIDENCE_THRESHOLD_REVIEW", raising=False)
    monkeypatch.delenv("RETRIEVAL_SIMILARITY_THRESHOLD", raising=False)
    settings = Settings(_env_file=None)

    assert settings.environment == "development"
    assert settings.log_level == "INFO"
    assert settings.llm_provider == LLMProvider.GROQ
    assert settings.groq_model == "openai/gpt-oss-120b"
    assert settings.groq_include_reasoning is False
    assert settings.groq_timeout == 60.0
    assert settings.groq_max_retries == 2
    assert settings.openrouter_model == "meta-llama/llama-3.3-70b-instruct"
    assert settings.confidence_threshold_auto_respond == 0.80
    assert settings.confidence_threshold_review == 0.60
    assert settings.retrieval_top_k == 4
    assert settings.retrieval_similarity_threshold == 0.40
    assert settings.enable_auto_respond is True
    assert settings.enable_prompt_injection_shield is True
    assert settings.enable_citation_verifier is True


def test_env_file_override(tmp_path: Path) -> None:
    """Explicitly verify that .env file values override default settings."""
    env_file = tmp_path / ".env"
    env_file.write_text("CONFIDENCE_THRESHOLD_AUTO_RESPOND=0.85\n", encoding="utf-8")
    settings = Settings(_env_file=str(env_file))
    assert settings.confidence_threshold_auto_respond == 0.85


def test_environment_variable_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicitly verify that environment variables override default settings."""
    monkeypatch.setenv("CONFIDENCE_THRESHOLD_AUTO_RESPOND", "0.88")
    settings = Settings(_env_file=None)
    assert settings.confidence_threshold_auto_respond == 0.88


def test_secrets_not_exposed_in_repr() -> None:
    """Verify API keys are wrapped in SecretStr and masked in string representation."""
    settings = Settings(groq_api_key="super_secret_groq_key")

    assert "super_secret_groq_key" not in repr(settings)
    assert settings.groq_api_key is not None
    assert settings.groq_api_key.get_secret_value() == "super_secret_groq_key"


def test_get_active_api_key() -> None:
    """Verify get_active_api_key returns the key matching the active provider."""
    groq_settings = Settings(
        llm_provider=LLMProvider.GROQ,
        groq_api_key="groq-test-key",
        openrouter_api_key="openrouter-test-key",
    )
    active_key = groq_settings.get_active_api_key()
    assert active_key is not None
    assert active_key.get_secret_value() == "groq-test-key"

    openrouter_settings = Settings(
        llm_provider=LLMProvider.OPENROUTER,
        groq_api_key="groq-test-key",
        openrouter_api_key="openrouter-test-key",
    )
    active_key = openrouter_settings.get_active_api_key()
    assert active_key is not None
    assert active_key.get_secret_value() == "openrouter-test-key"


def test_invalid_threshold_hierarchy() -> None:
    """Verify review threshold cannot exceed auto-respond threshold."""
    with pytest.raises(ValidationError, match="cannot be greater than"):
        Settings(
            confidence_threshold_auto_respond=0.50,
            confidence_threshold_review=0.75,
        )


def test_threshold_out_of_bounds() -> None:
    """Verify confidence thresholds must be between 0.0 and 1.0."""
    with pytest.raises(ValidationError):
        Settings(confidence_threshold_auto_respond=1.5)

    with pytest.raises(ValidationError):
        Settings(confidence_threshold_review=-0.1)
