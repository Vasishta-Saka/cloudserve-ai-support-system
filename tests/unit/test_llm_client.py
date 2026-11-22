"""Unit tests for injectable LLM clients and LangChain Groq integration.

Verifies:
1. Successful LCEL completion using FakeListChatModel and mock chat models.
2. System and user prompts correctly passed through prompt templates into message contents.
3. Prompt safety: literal braces in ticket content remain literal without template errors.
4. Missing Groq API key in live mode raises ConfigurationError.
5. Provider timeout/error raises LLMProviderError and does not return fabricated text.
6. Model metadata extraction extracts model name from provider response_metadata.
7. Token usage extraction extracts prompt_tokens, completion_tokens, total_tokens when present.
8. Unknown token usage returns an empty dict without inventing token counts.
9. MockLLMClient remains completely usable offline without network access.
10. get_llm_client factory returns appropriate client based on environment and credentials.
11. ChatGroq receives the configured model name without network access.
12. Reasoning models' think tags are stripped before JSON parsing.
13. ChatGroq receives include_reasoning=False, temperature, and max_completion_tokens.
14. Response generator strict schema rejects extraneous reasoning fields and suppresses release.
15. Outgoing request contains max_completion_tokens and omits deprecated max_tokens.
"""

import json
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from cloudserve.config import Settings
from cloudserve.generation.llm_client import (
    BaseLLMClient,
    ConfigurationError,
    LangChainGroqClient,
    LLMClientError,
    LLMProviderError,
    LLMResponse,
    MockLLMClient,
    get_llm_client,
)


class RecordingChatModel(BaseChatModel):
    """Test chat model recording received messages and kwargs for assertion."""

    captured_messages: list[list[BaseMessage]] = Field(default_factory=list)
    captured_kwargs: list[dict[str, Any]] = Field(default_factory=list)
    response_msg: AIMessage = Field(
        default_factory=lambda: AIMessage(content="Default recorded response")
    )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.captured_messages.append(messages)
        self.captured_kwargs.append(kwargs)
        return ChatResult(generations=[ChatGeneration(message=self.response_msg)])

    @property
    def _llm_type(self) -> str:
        return "recording-test-chat-model"


class ErrorChatModel(BaseChatModel):
    """Test chat model that raises a simulated provider exception."""

    error_message: str = "Simulated Groq timeout error"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise TimeoutError(self.error_message)

    @property
    def _llm_type(self) -> str:
        return "error-test-chat-model"


# -------------------------------------------------------------------------
# Scenario 1: Successful LCEL completion using FakeListChatModel
# -------------------------------------------------------------------------


def test_langchain_groq_client_lcel_completion_success() -> None:
    """Verify LangChainGroqClient successfully executes completion via LCEL."""
    fake_model = FakeListChatModel(responses=["Hello from LangChain Groq!"])
    client = LangChainGroqClient(chat_model=fake_model)

    assert isinstance(client, BaseLLMClient)
    assert client.model_name == "openai/gpt-oss-120b"

    response = client.complete(
        prompt="Explain cloud storage tiers",
        system_prompt="You are a CloudServe support agent.",
    )

    assert isinstance(response, LLMResponse)
    assert response.content == "Hello from LangChain Groq!"
    assert response.model_name == "openai/gpt-oss-120b"
    assert response.usage == {}


def test_langchain_groq_client_default_generation_parameters_passed() -> None:
    """Verify default temperature=0.0 and max_completion_tokens=2048 are passed to the model."""
    recording_model = RecordingChatModel()
    client = LangChainGroqClient(chat_model=recording_model)

    response = client.complete(prompt="Test prompt")
    assert isinstance(response, LLMResponse)
    assert len(recording_model.captured_kwargs) == 1
    passed_kwargs = recording_model.captured_kwargs[0]
    assert passed_kwargs.get("temperature") == 0.0
    assert passed_kwargs.get("max_completion_tokens") == 2048
    assert "max_tokens" not in passed_kwargs


def test_langchain_groq_client_custom_generation_parameters_passed() -> None:
    """Verify custom temperature and max_completion_tokens parameters are passed to the model."""
    recording_model = RecordingChatModel()
    client = LangChainGroqClient(chat_model=recording_model)

    response = client.complete(
        prompt="Test prompt",
        temperature=0.7,
        max_tokens=256,
    )
    assert isinstance(response, LLMResponse)
    assert len(recording_model.captured_kwargs) == 1
    passed_kwargs = recording_model.captured_kwargs[0]
    assert passed_kwargs.get("temperature") == 0.7
    assert passed_kwargs.get("max_completion_tokens") == 256
    assert "max_tokens" not in passed_kwargs


def test_langchain_groq_client_pipeline_execution_consistency() -> None:
    """Verify stored default pipeline is executed when parameters match defaults."""
    recording_model = RecordingChatModel()
    client = LangChainGroqClient(chat_model=recording_model)

    assert hasattr(client, "_pipeline")
    # First call with default parameters executes stored pipeline
    r1 = client.complete(prompt="Default call")
    assert r1.content == "Default recorded response"
    assert recording_model.captured_kwargs[0]["max_completion_tokens"] == 2048
    assert "max_tokens" not in recording_model.captured_kwargs[0]
    assert recording_model.captured_kwargs[0]["temperature"] == 0.0

    # Second call with custom parameters builds parameterized chain
    r2 = client.complete(prompt="Custom call", temperature=0.3, max_tokens=500)
    assert r2.content == "Default recorded response"
    assert recording_model.captured_kwargs[1]["max_completion_tokens"] == 500
    assert "max_tokens" not in recording_model.captured_kwargs[1]
    assert recording_model.captured_kwargs[1]["temperature"] == 0.3


# -------------------------------------------------------------------------
# Scenario 2: System and user prompts correctly passed into messages
# -------------------------------------------------------------------------


def test_langchain_groq_client_passes_prompts_to_messages() -> None:
    """Verify system and user prompts are passed as distinct template variables."""
    recording_model = RecordingChatModel(response_msg=AIMessage(content="Verified prompt delivery"))
    client = LangChainGroqClient(chat_model=recording_model)

    sys_text = "You are a specialized CloudServe security reviewer."
    usr_text = "Customer ticket: Cannot authenticate via SSO SAML endpoint."

    response = client.complete(prompt=usr_text, system_prompt=sys_text)

    assert response.content == "Verified prompt delivery"
    assert len(recording_model.captured_messages) == 1

    messages = recording_model.captured_messages[0]
    assert len(messages) == 2

    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == sys_text

    assert isinstance(messages[1], HumanMessage)
    assert messages[1].content == usr_text


def test_langchain_groq_client_default_system_prompt() -> None:
    """Verify a default system prompt is applied when system_prompt is None."""
    recording_model = RecordingChatModel()
    client = LangChainGroqClient(chat_model=recording_model)

    client.complete(prompt="How do I reset my password?", system_prompt=None)

    assert len(recording_model.captured_messages) == 1
    messages = recording_model.captured_messages[0]
    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == "You are a helpful AI assistant."


# -------------------------------------------------------------------------
# Scenario 3: Prompt safety - literal braces remain literal
# -------------------------------------------------------------------------


def test_langchain_groq_client_literal_braces_in_ticket_content() -> None:
    """Verify literal braces in ticket content do not trigger template parsing errors."""
    recording_model = RecordingChatModel()
    client = LangChainGroqClient(chat_model=recording_model)

    ticket_with_braces = (
        "Server logs:\n"
        '{"timestamp": "2026-04-01T12:00:00Z", "level": "ERROR", "meta": {"code": 500}}\n'
        "Variable template: ${USER} in {unmatched_brace and {other_var}."
    )

    response = client.complete(prompt=ticket_with_braces)
    assert isinstance(response, LLMResponse)

    captured = recording_model.captured_messages[0]
    assert captured[1].content == ticket_with_braces


# -------------------------------------------------------------------------
# Scenario 4: Missing Groq API key in live mode raises ConfigurationError
# -------------------------------------------------------------------------


def test_langchain_groq_client_missing_api_key_raises_configuration_error() -> None:
    """Verify live mode without a valid Groq API key raises ConfigurationError."""
    settings_no_key = Settings(groq_api_key=None)

    with pytest.raises(ConfigurationError, match="Missing Groq API key"):
        LangChainGroqClient(settings=settings_no_key)

    with pytest.raises(ConfigurationError, match="Missing Groq API key"):
        LangChainGroqClient(api_key="", settings=settings_no_key)

    with pytest.raises(ConfigurationError, match="Missing Groq API key"):
        LangChainGroqClient(api_key="   ", settings=settings_no_key)


def test_langchain_groq_client_accepts_secretstr_and_str_api_key() -> None:
    """Verify LangChainGroqClient accepts explicit string or SecretStr api_key overrides."""
    from pydantic import SecretStr

    client_str = LangChainGroqClient(api_key="gsk_explicit_str_key")
    assert client_str.model_name == "openai/gpt-oss-120b"

    client_secret = LangChainGroqClient(api_key=SecretStr("gsk_explicit_secret_str_key"))
    assert client_secret.model_name == "openai/gpt-oss-120b"


# -------------------------------------------------------------------------
# Scenario 5: Provider timeout/error raises LLMProviderError
# -------------------------------------------------------------------------


def test_langchain_groq_client_provider_error_raises_llm_provider_error() -> None:
    """Verify provider failures raise LLMProviderError and do not fabricate text."""
    error_model = ErrorChatModel(error_message="Groq upstream HTTP 504 Gateway Timeout")
    client = LangChainGroqClient(chat_model=error_model)

    with pytest.raises(LLMProviderError) as exc_info:
        client.complete("Evaluate this ticket")

    assert issubclass(LLMProviderError, LLMClientError)
    assert "Groq upstream HTTP 504 Gateway Timeout" in str(exc_info.value)


# -------------------------------------------------------------------------
# Scenario 6: Model metadata extraction
# -------------------------------------------------------------------------


def test_langchain_groq_client_model_name_extraction_from_response_metadata() -> None:
    """Verify model name identifier is extracted from provider response_metadata."""
    msg = AIMessage(
        content="Response text",
        response_metadata={"model_name": "openai/gpt-oss-120b-specdec"},
    )
    recording_model = RecordingChatModel(response_msg=msg)
    client = LangChainGroqClient(chat_model=recording_model)

    response = client.complete("Hello")
    assert response.model_name == "openai/gpt-oss-120b-specdec"


def test_langchain_groq_client_model_name_fallback_to_configured() -> None:
    """Verify fallback to configured model name when response_metadata lacks model info."""
    msg = AIMessage(content="Response text", response_metadata={})
    recording_model = RecordingChatModel(response_msg=msg)
    client = LangChainGroqClient(
        chat_model=recording_model,
        model_name="custom-fallback-model",
    )

    response = client.complete("Hello")
    assert response.model_name == "custom-fallback-model"


# -------------------------------------------------------------------------
# Scenario 7: Token usage extraction
# -------------------------------------------------------------------------


def test_langchain_groq_client_token_usage_from_response_metadata() -> None:
    """Verify token counts are extracted from response_metadata['token_usage']."""
    msg = AIMessage(
        content="Classification JSON",
        response_metadata={
            "token_usage": {
                "prompt_tokens": 150,
                "completion_tokens": 42,
                "total_tokens": 192,
            }
        },
    )
    recording_model = RecordingChatModel(response_msg=msg)
    client = LangChainGroqClient(chat_model=recording_model)

    response = client.complete("Classify ticket")
    assert response.usage == {
        "prompt_tokens": 150,
        "completion_tokens": 42,
        "total_tokens": 192,
    }


def test_langchain_groq_client_token_usage_from_usage_metadata() -> None:
    """Verify token counts are extracted from usage_metadata when response_metadata is absent."""
    msg = AIMessage(
        content="Grounding text",
        usage_metadata={
            "input_tokens": 200,
            "output_tokens": 80,
            "total_tokens": 280,
        },
    )
    recording_model = RecordingChatModel(response_msg=msg)
    client = LangChainGroqClient(chat_model=recording_model)

    response = client.complete("Ground check")
    assert response.usage == {
        "prompt_tokens": 200,
        "completion_tokens": 80,
        "total_tokens": 280,
    }


# -------------------------------------------------------------------------
# Scenario 8: Unknown token usage returns empty dict (no invented counts)
# -------------------------------------------------------------------------


def test_langchain_groq_client_missing_token_usage_returns_empty_dict() -> None:
    """Verify missing usage metadata returns empty dict, never fabricating tokens."""
    msg = AIMessage(content="Plain answer")
    recording_model = RecordingChatModel(response_msg=msg)
    client = LangChainGroqClient(chat_model=recording_model)

    response = client.complete("Question")
    assert response.usage == {}


# -------------------------------------------------------------------------
# Scenario 9: MockLLMClient offline usability
# -------------------------------------------------------------------------


def test_mock_llm_client_offline_usability() -> None:
    """Verify MockLLMClient operates offline with empty usage by default."""
    mock_client = MockLLMClient(responses=["First response", "Second response"])
    assert mock_client.model_name == "mock-llm-v1"

    r1 = mock_client.complete("Prompt 1", system_prompt="System 1", temperature=0.5)
    assert r1.content == "First response"
    assert r1.usage == {}  # Empty by default, not fabricated tokens
    assert len(mock_client.call_history) == 1
    assert mock_client.call_history[0]["prompt"] == "Prompt 1"
    assert mock_client.call_history[0]["system_prompt"] == "System 1"
    assert mock_client.call_history[0]["temperature"] == 0.5

    r2 = mock_client.complete("Prompt 2")
    assert r2.content == "Second response"
    assert r2.usage == {}

    # Queue exhaustion returns default empty JSON
    r3 = mock_client.complete("Prompt 3")
    assert r3.content == "{}"
    assert r3.usage == {}


def test_mock_llm_client_explicit_synthetic_usage() -> None:
    """Verify MockLLMClient reports explicit synthetic usage when supplied for a test."""
    mock_client = MockLLMClient(
        responses=["Canned response"],
        default_usage={"prompt_tokens": 42, "completion_tokens": 18, "total_tokens": 60},
    )
    res = mock_client.complete("Prompt")
    assert res.content == "Canned response"
    assert res.usage == {"prompt_tokens": 42, "completion_tokens": 18, "total_tokens": 60}


# -------------------------------------------------------------------------
# Scenario 10: get_llm_client factory selection
# -------------------------------------------------------------------------


def test_get_llm_client_factory_selection() -> None:
    """Verify get_llm_client selects client based on environment and credentials."""
    # Development without key -> MockLLMClient
    dev_settings = Settings(environment="development", groq_api_key=None)
    dev_client = get_llm_client(dev_settings)
    assert isinstance(dev_client, MockLLMClient)

    # Testing without key -> MockLLMClient
    test_settings = Settings(environment="testing", groq_api_key=None)
    test_client = get_llm_client(test_settings)
    assert isinstance(test_client, MockLLMClient)

    # Production without key -> ConfigurationError
    prod_no_key = Settings(environment="production", groq_api_key=None)
    with pytest.raises(ConfigurationError):
        get_llm_client(prod_no_key)

    # Staging without key -> ConfigurationError
    staging_no_key = Settings(environment="staging", groq_api_key=None)
    with pytest.raises(ConfigurationError):
        get_llm_client(staging_no_key)

    # With key provided -> LangChainGroqClient
    live_settings = Settings(
        environment="development",
        groq_api_key="gsk_test_mock_key_for_client_factory",
    )
    live_client = get_llm_client(live_settings)
    assert isinstance(live_client, LangChainGroqClient)


# -------------------------------------------------------------------------
# Scenario 11: ChatGroq model configuration without network access
# -------------------------------------------------------------------------


def test_chatgroq_receives_configured_model_id_without_network_call() -> None:
    """Verify ChatGroq receives configured model id without network call."""
    from langchain_groq import ChatGroq

    settings_default = Settings(groq_api_key="gsk_mock_test_key_123")
    client_default = LangChainGroqClient(settings=settings_default)
    assert isinstance(client_default._chat_model, ChatGroq)
    assert client_default._chat_model.model_name == "openai/gpt-oss-120b"
    assert client_default.model_kwargs == {"include_reasoning": False}
    assert client_default._chat_model.model_kwargs == {"include_reasoning": False}

    settings_custom = Settings(
        groq_api_key="gsk_mock_test_key_123",
        groq_model="custom/custom-model",
        groq_include_reasoning=True,
    )
    client_custom = LangChainGroqClient(settings=settings_custom)
    assert isinstance(client_custom._chat_model, ChatGroq)
    assert client_custom._chat_model.model_name == "custom/custom-model"
    assert client_custom.model_kwargs == {"include_reasoning": True}
    assert client_custom._chat_model.model_kwargs == {"include_reasoning": True}


# -------------------------------------------------------------------------
# Scenario 12: Reasoning model <think> tag handling in JSON extraction
# -------------------------------------------------------------------------


def test_reasoning_models_think_tags_stripped_for_json_parsing() -> None:
    """Verify <think> tags from reasoning models are stripped before JSON extraction."""
    from cloudserve.classification.classifier import _extract_json_string
    from cloudserve.generation.response_generator import ResponseGenerator

    raw_with_think = (
        "<think>\n"
        "Analyzing user inquiry...\n"
        "Intent seems to be api_usage_question.\n"
        "</think>\n"
        "```json\n"
        '{"intent": "api_usage_question"}\n'
        "```"
    )
    extracted_classifier = _extract_json_string(raw_with_think)
    assert extracted_classifier == '{"intent": "api_usage_question"}'

    gen = ResponseGenerator()
    extracted_generator = gen._extract_json_block(raw_with_think)
    assert extracted_generator == '{"intent": "api_usage_question"}'


# -------------------------------------------------------------------------
# Scenario 13: ChatGroq GPT-OSS request parameter verification (offline mock)
# -------------------------------------------------------------------------


def test_chatgroq_gpt_oss_request_parameters_without_network_call() -> None:
    """Verify include_reasoning=False, temperature, and max_completion_tokens reach create()."""
    from unittest.mock import MagicMock

    from langchain_groq import ChatGroq

    settings = Settings(groq_api_key="gsk_mock_test_key_123")
    client = LangChainGroqClient(settings=settings)
    assert isinstance(client._chat_model, ChatGroq)

    # Mock the internal Groq SDK client to verify exact request payload kwargs
    mock_groq_client = MagicMock()
    mock_create = MagicMock()
    mock_groq_client.create = mock_create
    mock_create.return_value = {
        "id": "chatcmpl-mock-123",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Grounded answer text",
                    "reasoning": "Internal reasoning that must not leak",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 8,
            "total_tokens": 23,
        },
    }
    client._chat_model.client = mock_groq_client

    response = client.complete("Test inquiry prompt")

    # Verify request payload passed to Groq API
    assert mock_create.called
    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["model"] == "openai/gpt-oss-120b"
    assert call_kwargs["include_reasoning"] is False
    assert call_kwargs["temperature"] == 0.0
    assert call_kwargs["max_completion_tokens"] == 2048
    assert "max_tokens" not in call_kwargs
    assert call_kwargs.get("reasoning_format") is None

    # Verify reasoning content did NOT leak into the returned LLMResponse.content
    assert response.content == "Grounded answer text"
    assert "Internal reasoning" not in response.content


# -------------------------------------------------------------------------
# Scenario 14: Reasoning field rejection in response generator schema
# -------------------------------------------------------------------------


def test_response_generator_schema_forbids_reasoning_field() -> None:
    """Verify RawLLMResponsePayload rejects payloads containing extraneous reasoning fields."""
    from datetime import datetime, timezone

    from pydantic import ValidationError

    from cloudserve.generation.models import RawLLMResponsePayload
    from cloudserve.generation.response_generator import ResponseGenerator
    from cloudserve.ingestion.models import (
        ChannelEnum,
        CustomerRegion,
        CustomerTier,
        LanguageFluency,
        UnifiedTicket,
    )

    # Raw payload with reasoning field present
    payload_with_reasoning = {
        "reasoning": "Step-by-step internal model reasoning...",
        "customer_response": "Here is your solution.",
        "agent_note": "Verified customer issue.",
        "citation_ids": [],
        "insufficient_evidence": False,
        "unsupported_claims": [],
    }

    # Strict schema validation forbids extra fields
    with pytest.raises(ValidationError):
        RawLLMResponsePayload.model_validate(payload_with_reasoning)

    # ResponseGenerator gracefully falls back and strictly suppresses customer response
    ticket = UnifiedTicket(
        ticket_id="TCK-REASONING-001",
        customer_id="CUST-001",
        received_at=datetime.now(timezone.utc),
        channel=ChannelEnum.EMAIL,
        subject="Cloud storage setup",
        body="How do I configure my cloud storage?",
        customer_tier=CustomerTier.ENTERPRISE,
        customer_region=CustomerRegion.NORTH_AMERICA,
        language_fluency=LanguageFluency.FLUENT,
    )
    gen = ResponseGenerator()
    resp = gen._parse_llm_response(
        raw_content=json.dumps(payload_with_reasoning),
        ticket=ticket,
        passages=[],
        model_name="openai/gpt-oss-120b",
    )
    assert resp.customer_response is None
    assert resp.is_fallback is True
    assert resp.citation_ids == []
    assert resp.insufficient_evidence is True
    assert "Strict schema validation failed" in resp.agent_note


# -------------------------------------------------------------------------
# Scenario 15: Outgoing request uses max_completion_tokens and omits max_tokens
# -------------------------------------------------------------------------


def test_outgoing_request_uses_max_completion_tokens_and_omits_deprecated_max_tokens() -> None:
    """Verify max_completion_tokens is sent exactly once and max_tokens is never sent."""
    from unittest.mock import MagicMock

    from langchain_groq import ChatGroq

    # 1. Central setting configuration (e.g. max_response_tokens=2048)
    settings = Settings(groq_api_key="gsk_mock_test_key_123", max_response_tokens=2048)
    client = LangChainGroqClient(settings=settings)
    assert isinstance(client._chat_model, ChatGroq)

    mock_groq_client = MagicMock()
    mock_create = MagicMock()
    mock_groq_client.create = mock_create
    mock_create.return_value = {
        "id": "chatcmpl-mock-456",
        "choices": [
            {
                "message": {"role": "assistant", "content": "Sample completion"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    client._chat_model.client = mock_groq_client

    # Default call resolves from settings.max_response_tokens (2048)
    client.complete("Hello prompt")
    kwargs1 = mock_create.call_args.kwargs
    assert "max_completion_tokens" in kwargs1
    assert kwargs1["max_completion_tokens"] == 2048
    assert "max_tokens" not in kwargs1

    # 2. Call-time override using max_completion_tokens (512)
    client.complete("Hello prompt", max_completion_tokens=512)
    kwargs2 = mock_create.call_args.kwargs
    assert "max_completion_tokens" in kwargs2
    assert kwargs2["max_completion_tokens"] == 512
    assert "max_tokens" not in kwargs2

    # 3. Call-time override using legacy max_tokens argument (256) maps to max_completion_tokens
    client.complete("Hello prompt", max_tokens=256)
    kwargs3 = mock_create.call_args.kwargs
    assert "max_completion_tokens" in kwargs3
    assert kwargs3["max_completion_tokens"] == 256
    assert "max_tokens" not in kwargs3
