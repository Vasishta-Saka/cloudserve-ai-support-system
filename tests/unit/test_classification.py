"""Unit tests for the Ticket Classification Engine.

Verifies:
- All 22 intent categories and 3 urgency tiers.
- Separate intent_confidence and urgency_confidence fields.
- Fallback behavior on malformed responses, unknown categories, and provider exceptions.
- Zero ground-truth leakage (ensures classifier only accepts UnifiedTicket).
- Binned calibration reporting and ECE calculations.
"""

import json
from datetime import datetime, timezone

import pytest

from cloudserve.classification.classifier import TicketClassifier
from cloudserve.classification.models import (
    ClassificationResult,
    IntentCategory,
    UrgencyLevel,
)
from cloudserve.classification.prompts import (
    format_ticket_prompt,
    load_classification_system_prompt,
)
from cloudserve.evaluation.metrics import (
    calculate_confidence_calibration,
)
from cloudserve.generation.llm_client import BaseLLMClient, LLMResponse, MockLLMClient
from cloudserve.ingestion.models import (
    ChannelEnum,
    CustomerRegion,
    CustomerTier,
    EvaluationRecord,
    LanguageFluency,
    TicketLabels,
    UnifiedTicket,
)


@pytest.fixture
def sample_ticket() -> UnifiedTicket:
    """Fixture returning a normalized live UnifiedTicket."""
    return UnifiedTicket(
        ticket_id="TEST-CLS-001",
        channel=ChannelEnum.CHAT,
        subject="",
        body="Builds that worked last week are failing during dependency resolution.",
        received_at=datetime(2026, 5, 23, 22, 43, tzinfo=timezone.utc),
        customer_id="CUST-1131",
        customer_name="Xin Kulkarni",
        customer_tier=CustomerTier.STANDARD,
        customer_region=CustomerRegion.LATIN_AMERICA,
        language_fluency=LanguageFluency.NON_FLUENT,
    )


class TestTaxonomyIntegrity:
    """Verify that all 22 intents and 3 urgencies match the official dataset taxonomy."""

    def test_all_22_intent_categories_present(self) -> None:
        """Verify exactly 22 intent categories exist in IntentCategory enum."""
        expected_intents = {
            "account_access",
            "api_key_issue",
            "api_usage_question",
            "authentication_failure",
            "billing_query",
            "compliance_request",
            "configuration_help",
            "data_export",
            "data_residency",
            "database_issue",
            "deployment_failure",
            "feature_request",
            "integration_help",
            "onboarding",
            "performance_degradation",
            "quota_or_overage",
            "rate_limit",
            "rollback_request",
            "security_incident",
            "sso_configuration",
            "unclear_request",
            "webhook_issue",
        }
        actual_intents = {item.value for item in IntentCategory}
        assert actual_intents == expected_intents
        assert len(IntentCategory) == 22

    def test_all_3_urgency_levels_present(self) -> None:
        """Verify exactly 3 urgency levels exist in UrgencyLevel enum."""
        expected_urgencies = {"low", "medium", "high"}
        actual_urgencies = {item.value for item in UrgencyLevel}
        assert actual_urgencies == expected_urgencies
        assert len(UrgencyLevel) == 3


class TestClassifierParsingAndConfidence:
    """Tests for structured LLM response parsing and confidence handling."""

    def test_successful_classification_with_separated_confidences(
        self, sample_ticket: UnifiedTicket
    ) -> None:
        """Verify intent_confidence and urgency_confidence are stored separately with composite."""
        mock_response = json.dumps(
            {
                "intent": "deployment_failure",
                "urgency": "high",
                "intent_confidence": 0.95,
                "urgency_confidence": 0.90,
                "alternatives": [
                    {
                        "intent": "configuration_help",
                        "confidence": 0.05,
                        "reason": "Could be environment config.",
                    }
                ],
                "rationale": "Dependency resolution failures indicate a build/deployment issue.",
            }
        )
        client = MockLLMClient(responses=mock_response)
        classifier = TicketClassifier(llm_client=client)

        result = classifier.classify(sample_ticket)

        assert isinstance(result, ClassificationResult)
        assert result.ticket_id == "TEST-CLS-001"
        assert result.intent == IntentCategory.DEPLOYMENT_FAILURE
        assert result.urgency == UrgencyLevel.HIGH
        assert result.intent_confidence == 0.95
        assert result.urgency_confidence == 0.90
        # Composite headline confidence
        assert result.confidence == 0.855  # round(0.95 * 0.90, 4)
        assert len(result.alternatives) == 1
        assert result.alternatives[0].intent == IntentCategory.CONFIGURATION_HELP
        assert result.alternatives[0].confidence == 0.05
        assert result.rationale == (
            "Dependency resolution failures indicate a build/deployment issue."
        )
        assert result.is_fallback is False

    def test_markdown_wrapped_json_parsing(self, sample_ticket: UnifiedTicket) -> None:
        """Verify model responses wrapped in markdown ```json ... ``` blocks are cleanly peeled."""
        mock_response = (
            "Here is the classification result:\n"
            "```json\n"
            "{\n"
            '  "intent": "rate_limit",\n'
            '  "urgency": "medium",\n'
            '  "intent_confidence": 0.88,\n'
            '  "urgency_confidence": 0.80,\n'
            '  "alternatives": [],\n'
            '  "rationale": "HTTP 429 indicates rate limiting."\n'
            "}\n"
            "```"
        )
        client = MockLLMClient(responses=mock_response)
        classifier = TicketClassifier(llm_client=client)

        result = classifier.classify(sample_ticket)

        assert result.intent == IntentCategory.RATE_LIMIT
        assert result.urgency == UrgencyLevel.MEDIUM
        assert result.intent_confidence == 0.88
        assert result.urgency_confidence == 0.80
        assert result.is_fallback is False


class TestDeterministicFallbackBehavior:
    """Verify errors yield is_fallback=True and confidence=0.0 so router never auto-responds."""

    def test_malformed_json_triggers_deterministic_fallback(
        self, sample_ticket: UnifiedTicket
    ) -> None:
        """Verify non-JSON response triggers fallback with zero confidence."""
        client = MockLLMClient(responses="I think this is a deployment problem, but no JSON.")
        classifier = TicketClassifier(llm_client=client)

        result = classifier.classify(sample_ticket)

        assert result.is_fallback is True
        assert result.intent == IntentCategory.UNCLEAR_REQUEST
        assert result.urgency == UrgencyLevel.MEDIUM
        assert result.confidence == 0.0
        assert result.intent_confidence == 0.0
        assert result.urgency_confidence == 0.0
        assert "Malformed JSON" in result.rationale

    def test_unknown_intent_category_triggers_fallback(self, sample_ticket: UnifiedTicket) -> None:
        """Verify unknown category like 'random_invented_intent' triggers fallback."""
        mock_response = json.dumps(
            {
                "intent": "random_invented_intent",
                "urgency": "low",
                "intent_confidence": 0.90,
                "urgency_confidence": 0.90,
                "rationale": "Made up category.",
            }
        )
        client = MockLLMClient(responses=mock_response)
        classifier = TicketClassifier(llm_client=client)

        result = classifier.classify(sample_ticket)

        assert result.is_fallback is True
        assert result.intent == IntentCategory.UNCLEAR_REQUEST
        assert result.confidence == 0.0
        assert "Unknown intent category" in result.rationale

    def test_unknown_urgency_triggers_fallback(self, sample_ticket: UnifiedTicket) -> None:
        """Verify invalid urgency string like 'emergency' triggers fallback."""
        mock_response = json.dumps(
            {
                "intent": "billing_query",
                "urgency": "emergency",
                "intent_confidence": 0.90,
                "urgency_confidence": 0.90,
                "rationale": "High priority issue.",
            }
        )
        client = MockLLMClient(responses=mock_response)
        classifier = TicketClassifier(llm_client=client)

        result = classifier.classify(sample_ticket)

        assert result.is_fallback is True
        assert result.intent == IntentCategory.UNCLEAR_REQUEST
        assert result.confidence == 0.0
        assert "Unknown urgency level" in result.rationale

    def test_provider_exception_triggers_fallback_without_crashing(
        self, sample_ticket: UnifiedTicket
    ) -> None:
        """Verify network or provider exceptions cleanly yield fallback conforming to AC-11."""

        class FailingClient(BaseLLMClient):
            def complete(self, *args: object, **kwargs: object) -> LLMResponse:
                raise ConnectionError("503 Service Unavailable: Provider down")

            @property
            def model_name(self) -> str:
                return "failing-provider"

        classifier = TicketClassifier(llm_client=FailingClient())
        result = classifier.classify(sample_ticket)

        assert result.is_fallback is True
        assert result.confidence == 0.0
        assert result.intent == IntentCategory.UNCLEAR_REQUEST
        assert "503 Service Unavailable" in result.rationale


class TestGroundTruthIsolation:
    """Verify that the classifier strictly accepts UnifiedTicket and rejects EvaluationRecord."""

    def test_classify_rejects_evaluation_record(self, sample_ticket: UnifiedTicket) -> None:
        """Verify TypeError if someone mistakenly passes an EvaluationRecord."""
        eval_record = EvaluationRecord(
            ticket=sample_ticket,
            labels=TicketLabels(
                intent="deployment_failure",
                urgency="high",
                expected_route="auto_respond",
            ),
        )
        classifier = TicketClassifier(llm_client=MockLLMClient())

        with pytest.raises(TypeError, match="requires a UnifiedTicket instance"):
            classifier.classify(eval_record)  # type: ignore

    def test_classify_rejects_raw_dict(self) -> None:
        """Verify TypeError if someone passes a raw dictionary instead of UnifiedTicket."""
        classifier = TicketClassifier(llm_client=MockLLMClient())

        with pytest.raises(TypeError, match="requires a UnifiedTicket instance"):
            classifier.classify({"ticket_id": "RAW-001", "body": "test"})  # type: ignore


class TestConfidenceCalibrationCalculation:
    """Tests for binned confidence calibration and governance target verification."""

    def test_well_calibrated_predictions_meet_target(self) -> None:
        """Verify calibration metrics when stated confidence matches observed accuracy."""
        # 100 predictions across 5 bins, each bin perfectly aligned
        confidences = [0.10] * 20 + [0.30] * 20 + [0.50] * 20 + [0.70] * 20 + [0.90] * 20
        accuracies = (
            [True] * 2
            + [False] * 18  # 10% accurate
            + [True] * 6
            + [False] * 14  # 30% accurate
            + [True] * 10
            + [False] * 10  # 50% accurate
            + [True] * 14
            + [False] * 6  # 70% accurate
            + [True] * 18
            + [False] * 2  # 90% accurate
        )

        report = calculate_confidence_calibration(confidences, accuracies, n_bins=5)

        assert report.total_samples == 100
        assert report.reliable_bins_count == 5
        assert report.expected_calibration_error == 0.0
        assert report.meets_governance_target is True
        assert "Governance target met" in report.notes

    def test_poorly_calibrated_predictions_fail_target(self) -> None:
        """Verify calibration metrics when stated confidence is far from observed accuracy."""
        # Model is 90% confident, but only 50% accurate
        confidences = [0.90] * 50
        accuracies = [True] * 25 + [False] * 25  # 50% accurate

        report = calculate_confidence_calibration(confidences, accuracies, n_bins=5)

        assert report.expected_calibration_error > 0.05
        assert report.meets_governance_target is False
        assert "NOT met" in report.notes

    def test_insufficient_samples_does_not_claim_target_met(self) -> None:
        """Verify that small sample sizes do not falsely claim governance target is met."""
        # Only 3 samples, fewer than min_samples_per_bin (default: 10)
        confidences = [0.90, 0.90, 0.90]
        accuracies = [True, True, True]

        report = calculate_confidence_calibration(
            confidences, accuracies, n_bins=5, min_samples_per_bin=10
        )

        assert report.total_samples == 3
        assert report.reliable_bins_count == 0
        # Must not claim target met when sample size is insufficient!
        assert report.meets_governance_target is False
        assert "insufficient" in report.notes.lower()


class TestPromptLoadingAndFormatting:
    """Verify version-controlled prompt loading and ticket text rendering."""

    def test_load_classification_system_prompt_v1(self) -> None:
        """Verify prompt template v1 loads and contains all 22 category keywords."""
        prompt = load_classification_system_prompt("v1")
        assert "CloudServe Systems" in prompt
        assert "account_access" in prompt
        assert "deployment_failure" in prompt
        assert "webhook_issue" in prompt
        assert "intent_confidence" in prompt
        assert "urgency_confidence" in prompt
        assert "rationale" in prompt

    def test_format_ticket_prompt(self, sample_ticket: UnifiedTicket) -> None:
        """Verify user prompt properly interpolates ticket fields."""
        prompt_text = format_ticket_prompt(sample_ticket)
        assert sample_ticket.ticket_id in prompt_text
        assert sample_ticket.body in prompt_text
        assert sample_ticket.channel.value in prompt_text
        assert sample_ticket.customer_tier.value in prompt_text


class TestRec01IntegrationClassificationDistinctions:
    """Targeted regression tests for REC-01 intent classification improvements.

    Verifies unambiguous boundaries and contrasting calibration examples between:
    - integration_help
    - webhook_issue
    - api_key_issue
    """

    def test_prompt_contains_expanded_integration_help_definition(self) -> None:
        """Verify integration_help definition includes log forwarding, sinks, and CI/CD."""
        prompt = load_classification_system_prompt("v1")
        assert "log forwarding pipelines" in prompt
        assert "external logging destinations" in prompt
        assert "streaming sinks" in prompt
        assert "CI/CD automation design" in prompt
        # Ensure confusing "custom webhooks" is not bundled under integration_help definition
        lines = prompt.splitlines()
        integration_lines = [ln for ln in lines if ln.strip().startswith("13. integration_help:")]
        assert len(integration_lines) == 1
        assert "custom webhooks" not in integration_lines[0]

    def test_prompt_contains_discriminating_definitions_for_related_intents(self) -> None:
        """Verify api_key_issue and webhook_issue have explicit boundary disclaimers."""
        prompt = load_classification_system_prompt("v1")
        lines = prompt.splitlines()

        api_key_lines = [ln for ln in lines if ln.strip().startswith("2. api_key_issue:")]
        assert len(api_key_lines) == 1
        assert "Excludes architectural choices for CI/CD" in api_key_lines[0]

        webhook_lines = [ln for ln in lines if ln.strip().startswith("22. webhook_issue:")]
        assert len(webhook_lines) == 1
        assert "CloudServe outbound webhook notifications" in webhook_lines[0]

    def test_prompt_contains_contrasting_few_shot_examples(self) -> None:
        """Verify prompt contains calibration examples for all three contrasting intents."""
        prompt = load_classification_system_prompt("v1")

        # 1. integration_help (log forwarding)
        assert "Log forwarding to our external system stopped" in prompt
        # 2. integration_help (CI/CD authentication choice)
        assert (
            "We are connecting our CI pipeline and are not sure whether to use a personal key"
            in prompt
        )
        # 3. webhook_issue (outbound webhook delivery)
        assert "Our webhook endpoint has stopped receiving events" in prompt
        # 4. api_key_issue (401 on existing key)
        assert "One of our production keys started returning 401" in prompt

    def test_classify_log_forwarding_ticket_as_integration_help(
        self, sample_ticket: UnifiedTicket
    ) -> None:
        """Verify log forwarding ticket is parsed as integration_help with webhook alternative."""
        log_ticket = UnifiedTicket(
            ticket_id="TEST-LOG-001",
            channel=ChannelEnum.EMAIL,
            subject="Log forwarding stopped working",
            body=(
                "Log forwarding to our external system stopped around 14:00 yesterday and has "
                "not resumed by itself. Destination is healthy as far as we can tell."
            ),
            received_at=sample_ticket.received_at,
            customer_id=sample_ticket.customer_id,
            customer_name=sample_ticket.customer_name,
            customer_tier=sample_ticket.customer_tier,
            customer_region=sample_ticket.customer_region,
            language_fluency=sample_ticket.language_fluency,
        )
        mock_response = json.dumps(
            {
                "intent": "integration_help",
                "urgency": "medium",
                "intent_confidence": 0.95,
                "urgency_confidence": 0.88,
                "alternatives": [
                    {
                        "intent": "webhook_issue",
                        "confidence": 0.05,
                        "reason": "External logging sinks differ from outbound webhook endpoints.",
                    }
                ],
                "rationale": "Inquiry concerns external log forwarding pipeline delivery.",
            }
        )
        client = MockLLMClient(responses=mock_response)
        classifier = TicketClassifier(llm_client=client)

        result = classifier.classify(log_ticket)

        assert result.intent == IntentCategory.INTEGRATION_HELP
        assert result.urgency == UrgencyLevel.MEDIUM
        assert result.intent_confidence == 0.95
        assert result.is_fallback is False
        assert len(result.alternatives) == 1
        assert result.alternatives[0].intent == IntentCategory.WEBHOOK_ISSUE

    def test_classify_ci_cd_auth_choice_as_integration_help_not_api_key(
        self, sample_ticket: UnifiedTicket
    ) -> None:
        """Verify CI/CD credential design inquiry parses as integration_help, not api_key_issue."""
        ci_ticket = UnifiedTicket(
            ticket_id="TEST-CI-001",
            channel=ChannelEnum.CHAT,
            subject="Connecting our CI pipeline",
            body=(
                "We are connecting our CI pipeline and are not sure whether to use a personal key "
                "or something else. What is the recommended approach for automation?"
            ),
            received_at=sample_ticket.received_at,
            customer_id=sample_ticket.customer_id,
            customer_name=sample_ticket.customer_name,
            customer_tier=sample_ticket.customer_tier,
            customer_region=sample_ticket.customer_region,
            language_fluency=sample_ticket.language_fluency,
        )
        mock_response = json.dumps(
            {
                "intent": "integration_help",
                "urgency": "low",
                "intent_confidence": 0.94,
                "urgency_confidence": 0.90,
                "alternatives": [
                    {
                        "intent": "api_key_issue",
                        "confidence": 0.06,
                        "reason": (
                            "Mention of personal key vs service account relates to CI design, "
                            "not a malfunctioning or broken key."
                        ),
                    }
                ],
                "rationale": (
                    "Guidance on authentication architecture for CI/CD pipeline integration is "
                    "an integration setup inquiry, not an API key failure."
                ),
            }
        )
        client = MockLLMClient(responses=mock_response)
        classifier = TicketClassifier(llm_client=client)

        result = classifier.classify(ci_ticket)

        assert result.intent == IntentCategory.INTEGRATION_HELP
        assert result.urgency == UrgencyLevel.LOW
        assert result.intent_confidence == 0.94
        assert result.is_fallback is False
        assert len(result.alternatives) == 1
        assert result.alternatives[0].intent == IntentCategory.API_KEY_ISSUE

    def test_classify_webhook_delivery_failure_as_webhook_issue(
        self, sample_ticket: UnifiedTicket
    ) -> None:
        """Verify webhook delivery failure is parsed as webhook_issue, not integration_help."""
        webhook_ticket = UnifiedTicket(
            ticket_id="TEST-WH-001",
            channel=ChannelEnum.EMAIL,
            subject="Webhook delivery failures",
            body=(
                "Our webhook endpoint has stopped receiving events since late last night. "
                "The endpoint is up and responding to health checks from our monitor."
            ),
            received_at=sample_ticket.received_at,
            customer_id=sample_ticket.customer_id,
            customer_name=sample_ticket.customer_name,
            customer_tier=sample_ticket.customer_tier,
            customer_region=sample_ticket.customer_region,
            language_fluency=sample_ticket.language_fluency,
        )
        mock_response = json.dumps(
            {
                "intent": "webhook_issue",
                "urgency": "high",
                "intent_confidence": 0.95,
                "urgency_confidence": 0.90,
                "alternatives": [
                    {
                        "intent": "integration_help",
                        "confidence": 0.05,
                        "reason": "Direct failure of outbound webhook notifications.",
                    }
                ],
                "rationale": "Failure of CloudServe webhook notifications dispatched to endpoint.",
            }
        )
        client = MockLLMClient(responses=mock_response)
        classifier = TicketClassifier(llm_client=client)

        result = classifier.classify(webhook_ticket)

        assert result.intent == IntentCategory.WEBHOOK_ISSUE
        assert result.urgency == UrgencyLevel.HIGH
        assert result.intent_confidence == 0.95
        assert result.is_fallback is False
        assert len(result.alternatives) == 1
        assert result.alternatives[0].intent == IntentCategory.INTEGRATION_HELP


class TestRec02UnclearRequestClassificationFix:
    """Targeted regression tests for REC-02 unclear_request precision improvements.

    Verifies:
    - unclear_request is reserved for truly unparseable, context-free tickets.
    - Short or broken-English tickets with identifiable problem domains are not marked unclear.
    - Prompt contains contrasting examples from the development dataset.
    - Classifier correctly parses both short-but-classifiable tickets and truly unclear tickets.
    """

    def test_prompt_contains_clarified_unclear_request_definition(self) -> None:
        """Verify prompt explicitly bounds unclear_request to genuinely unparseable tickets."""
        prompt = load_classification_system_prompt("v1")
        lines = prompt.splitlines()
        unclear_lines = [ln for ln in lines if ln.strip().startswith("21. unclear_request:")]
        assert len(unclear_lines) == 1
        definition = unclear_lines[0]
        assert "genuinely unparseable, incomplete, context-free, or non-actionable" in definition
        assert "Do NOT use unclear_request merely because a ticket is short" in definition

    def test_prompt_contains_contrasting_unclear_few_shot_examples(self) -> None:
        """Verify prompt contains calibration examples for short-clear vs truly-unclear tickets."""
        prompt = load_classification_system_prompt("v1")
        # Short but identifiable billing inquiry
        assert "invoice showing service we not use. can you check and tell?" in prompt
        # Truly unparseable request
        assert "it not working. Please help. sorry for my english." in prompt

    def test_short_broken_english_ticket_classified_into_domain_not_unclear(
        self, sample_ticket: UnifiedTicket
    ) -> None:
        """Verify short ticket with broken English parses into domain intent (billing_query)."""
        short_billing_ticket = UnifiedTicket(
            ticket_id="TEST-SHORT-001",
            channel=ChannelEnum.CHAT,
            subject="",
            body="invoice showing service we not use. can you check and tell?",
            received_at=sample_ticket.received_at,
            customer_id=sample_ticket.customer_id,
            customer_name=sample_ticket.customer_name,
            customer_tier=sample_ticket.customer_tier,
            customer_region=sample_ticket.customer_region,
            language_fluency=LanguageFluency.NON_FLUENT,
        )
        mock_response = json.dumps(
            {
                "intent": "billing_query",
                "urgency": "medium",
                "intent_confidence": 0.92,
                "urgency_confidence": 0.85,
                "alternatives": [
                    {
                        "intent": "unclear_request",
                        "confidence": 0.08,
                        "reason": "Brief text but core domain is an unexpected charge.",
                    }
                ],
                "rationale": "Customer specifically asks about unexpected invoice charges.",
            }
        )
        client = MockLLMClient(responses=mock_response)
        classifier = TicketClassifier(llm_client=client)

        result = classifier.classify(short_billing_ticket)

        assert result.intent == IntentCategory.BILLING_QUERY
        assert result.urgency == UrgencyLevel.MEDIUM
        assert result.intent_confidence == 0.92
        assert result.is_fallback is False
        assert len(result.alternatives) == 1
        assert result.alternatives[0].intent == IntentCategory.UNCLEAR_REQUEST

    def test_truly_unparseable_ticket_classified_as_unclear_request(
        self, sample_ticket: UnifiedTicket
    ) -> None:
        """Verify ticket with zero technical context correctly classifies as unclear_request."""
        vague_ticket = UnifiedTicket(
            ticket_id="TEST-VAGUE-001",
            channel=ChannelEnum.CHAT,
            subject="Problem with the service",
            body="it not working. Please help. sorry for my english.",
            received_at=sample_ticket.received_at,
            customer_id=sample_ticket.customer_id,
            customer_name=sample_ticket.customer_name,
            customer_tier=sample_ticket.customer_tier,
            customer_region=sample_ticket.customer_region,
            language_fluency=LanguageFluency.NON_FLUENT,
        )
        mock_response = json.dumps(
            {
                "intent": "unclear_request",
                "urgency": "medium",
                "intent_confidence": 0.96,
                "urgency_confidence": 0.80,
                "alternatives": [],
                "rationale": "Customer states only that something is not working without context.",
            }
        )
        client = MockLLMClient(responses=mock_response)
        classifier = TicketClassifier(llm_client=client)

        result = classifier.classify(vague_ticket)

        assert result.intent == IntentCategory.UNCLEAR_REQUEST
        assert result.urgency == UrgencyLevel.MEDIUM
        assert result.intent_confidence == 0.96
        assert result.is_fallback is False
        assert len(result.alternatives) == 0

    def test_short_technical_inquiry_with_unfamiliar_terms_not_unclear(
        self, sample_ticket: UnifiedTicket
    ) -> None:
        """Verify brief ticket describing recognizable technical problem is not unclear."""
        short_tech_ticket = UnifiedTicket(
            ticket_id="TEST-TECH-001",
            channel=ChannelEnum.CHAT,
            subject="how to get all data?",
            body="page size big is reject. small is very slow.",
            received_at=sample_ticket.received_at,
            customer_id=sample_ticket.customer_id,
            customer_name=sample_ticket.customer_name,
            customer_tier=sample_ticket.customer_tier,
            customer_region=sample_ticket.customer_region,
            language_fluency=LanguageFluency.NON_FLUENT,
        )
        mock_response = json.dumps(
            {
                "intent": "api_usage_question",
                "urgency": "low",
                "intent_confidence": 0.91,
                "urgency_confidence": 0.85,
                "alternatives": [
                    {
                        "intent": "performance_degradation",
                        "confidence": 0.09,
                        "reason": "Mention of slowness relates to pagination query strategy.",
                    }
                ],
                "rationale": "Customer inquires about API data retrieval and page size limits.",
            }
        )
        client = MockLLMClient(responses=mock_response)
        classifier = TicketClassifier(llm_client=client)

        result = classifier.classify(short_tech_ticket)

        assert result.intent == IntentCategory.API_USAGE_QUESTION
        assert result.urgency == UrgencyLevel.LOW
        assert result.intent_confidence == 0.91
        assert result.is_fallback is False

    def test_runtime_classifier_loads_v2_prompt_by_default(self) -> None:
        """Verify that runtime TicketClassifier loads version-controlled v2 prompt by default."""
        classifier = TicketClassifier()
        assert classifier.prompt_version == "v2"
        assert "Classify urgency strictly based on operational impact" in classifier._system_prompt
        assert (
            "Live production environments or customer-facing services"
            in classifier._system_prompt
        )
