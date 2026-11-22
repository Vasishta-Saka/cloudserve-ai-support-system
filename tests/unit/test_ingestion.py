"""Unit tests for multi-channel ticket ingestion and normalization.

Tests email, live chat, documentation comments, and community forum parsers,
verifying deterministic schema enforcement, edge case handling, evaluation ground-truth
isolation, and error behavior.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from cloudserve.ingestion.models import (
    ChannelEnum,
    CustomerRegion,
    CustomerTier,
    EvaluationRecord,
    LanguageFluency,
    UnifiedTicket,
)
from cloudserve.ingestion.parsers import (
    ChatParser,
    DocsCommentParser,
    EmailParser,
    ForumParser,
    TicketParsingError,
    parse_evaluation_dataset,
    parse_evaluation_record,
    parse_ticket,
    parse_ticket_batch,
)


@pytest.fixture
def sample_email_raw() -> dict:
    """Fixture representing a raw email ticket."""
    return {
        "ticket_id": "TEST-EMAIL-001",
        "channel": "email",
        "subject": "Billing discrepancy on invoice INV-2026-04",
        "body": "We noticed an unexpected charge on our latest invoice. Could you please clarify?",
        "received_at": "2026-05-10T14:30:00Z",
        "customer_id": "CUST-2001",
        "customer_name": "Elena Rostova",
        "customer_tier": "business",
        "customer_region": "europe",
        "language_fluency": "fluent",
    }


@pytest.fixture
def sample_chat_raw() -> dict:
    """Fixture representing a raw live chat ticket."""
    return {
        "ticket_id": "TEST-CHAT-001",
        "channel": "chat",
        "subject": "",
        "body": "Hello, our API requests are returning 429 suddenly. Are there outages?",
        "received_at": "2026-05-11T09:15:00Z",
        "customer_id": "CUST-2002",
        "customer_name": "Marcus Vance",
        "customer_tier": "enterprise",
        "customer_region": "north_america",
        "language_fluency": "fluent",
    }


@pytest.fixture
def sample_docs_comment_raw() -> dict:
    """Fixture representing a raw documentation comment ticket."""
    return {
        "ticket_id": "TEST-DOC-001",
        "channel": "docs_comment",
        "subject": "Missing step in SSO setup guide",
        "body": (
            "The Okta SAML configuration section mentions step 4 but does not define the ACS URL."
        ),
        "received_at": "2026-05-12T11:45:00Z",
        "customer_id": "CUST-2003",
        "customer_name": "Akira Tanaka",
        "customer_tier": "standard",
        "customer_region": "asia_pacific",
        "language_fluency": "non_fluent",
    }


@pytest.fixture
def sample_forum_raw() -> dict:
    """Fixture representing a raw community forum ticket."""
    return {
        "ticket_id": "TEST-FORUM-001",
        "channel": "forum",
        "subject": "Best practices for multi-tenant VPC peering",
        "body": "Has anyone configured peering across isolated VPCs with shared transit gateways?",
        "received_at": "2026-05-13T16:20:00Z",
        "customer_id": "CUST-2004",
        "customer_name": "Carlos Gomez",
        "customer_tier": "standard",
        "customer_region": "latin_america",
        "language_fluency": "fluent",
    }


@pytest.fixture
def sample_labeled_raw() -> dict:
    """Fixture representing a raw dataset ticket containing labels and history."""
    return {
        "ticket_id": "DEV-0001",
        "channel": "chat",
        "subject": "",
        "body": "builds that worked last week are failing during dependency resolution.",
        "received_at": "2026-05-23T22:43:00Z",
        "customer_id": "CUST-1131",
        "customer_name": "Xin Kulkarni",
        "customer_tier": "standard",
        "customer_region": "latin_america",
        "language_fluency": "non_fluent",
        "labels": {
            "intent": "deployment_failure",
            "urgency": "high",
            "expected_route": "auto_respond",
            "answerable_from_docs": True,
            "expected_doc_ids": ["DOC-DEPLOY-001"],
            "must_not_auto_respond": False,
        },
        "history": {
            "first_contact_resolution": True,
            "resolution_time_minutes": 30,
            "csat_rating": 5,
            "escalated": False,
            "repeat_contact": False,
        },
    }


class TestChannelParsers:
    """Tests for individual channel parsers and normalizers."""

    def test_email_parser_normalization(self, sample_email_raw: dict) -> None:
        """Verify email parser correctly populates subject, body, and customer info."""
        ticket = EmailParser().parse(sample_email_raw)

        assert isinstance(ticket, UnifiedTicket)
        assert ticket.ticket_id == "TEST-EMAIL-001"
        assert ticket.channel == ChannelEnum.EMAIL
        assert ticket.subject == "Billing discrepancy on invoice INV-2026-04"
        assert "unexpected charge" in ticket.body
        assert ticket.customer_tier == CustomerTier.BUSINESS
        assert ticket.customer_region == CustomerRegion.EUROPE
        assert ticket.language_fluency == LanguageFluency.FLUENT
        assert ticket.received_at == datetime(2026, 5, 10, 14, 30, tzinfo=timezone.utc)

    def test_chat_parser_normalizes_empty_subject(self, sample_chat_raw: dict) -> None:
        """Verify live chat parser enforces an empty subject per specification."""
        ticket = ChatParser().parse(sample_chat_raw)

        assert ticket.channel == ChannelEnum.CHAT
        assert ticket.subject == ""
        assert "returning 429" in ticket.body
        assert ticket.clean_text == ticket.body

    def test_chat_parser_captures_provided_subject_in_metadata(self) -> None:
        """Verify if chat payload provides a non-empty subject, it is preserved in metadata."""
        raw = {
            "ticket_id": "TEST-CHAT-002",
            "channel": "chat",
            "subject": "Quick question",
            "body": "How do I reset my API secret?",
            "received_at": "2026-05-11T10:00:00Z",
        }
        ticket = ChatParser().parse(raw)
        assert ticket.subject == ""
        assert ticket.channel_metadata.get("original_subject") == "Quick question"

    def test_docs_comment_parser(self, sample_docs_comment_raw: dict) -> None:
        """Verify docs comment parser preserves subject and non-fluent fluency."""
        ticket = DocsCommentParser().parse(sample_docs_comment_raw)

        assert ticket.channel == ChannelEnum.DOCS_COMMENT
        assert ticket.subject == "Missing step in SSO setup guide"
        assert ticket.language_fluency == LanguageFluency.NON_FLUENT
        assert ticket.customer_region == CustomerRegion.ASIA_PACIFIC

    def test_forum_parser(self, sample_forum_raw: dict) -> None:
        """Verify forum parser handles forum inquiry properly."""
        ticket = ForumParser().parse(sample_forum_raw)

        assert ticket.channel == ChannelEnum.FORUM
        assert ticket.subject == "Best practices for multi-tenant VPC peering"
        assert ticket.customer_region == CustomerRegion.LATIN_AMERICA


class TestMasterParserDispatch:
    """Tests for top-level parse_ticket and parse_ticket_batch functions."""

    def test_parse_from_dict(self, sample_email_raw: dict) -> None:
        """Verify parse_ticket handles dictionary input."""
        ticket = parse_ticket(sample_email_raw)
        assert ticket.ticket_id == sample_email_raw["ticket_id"]
        assert ticket.channel == ChannelEnum.EMAIL

    def test_parse_from_json_string(self, sample_email_raw: dict) -> None:
        """Verify parse_ticket handles serialized JSON string."""
        json_str = json.dumps(sample_email_raw)
        ticket = parse_ticket(json_str)
        assert ticket.ticket_id == sample_email_raw["ticket_id"]

    def test_parse_from_bytes(self, sample_chat_raw: dict) -> None:
        """Verify parse_ticket handles serialized UTF-8 bytes."""
        json_bytes = json.dumps(sample_chat_raw).encode("utf-8")
        ticket = parse_ticket(json_bytes)
        assert ticket.ticket_id == sample_chat_raw["ticket_id"]
        assert ticket.channel == ChannelEnum.CHAT

    def test_parse_ticket_batch(
        self,
        sample_email_raw: dict,
        sample_chat_raw: dict,
        sample_docs_comment_raw: dict,
        sample_forum_raw: dict,
    ) -> None:
        """Verify batch parsing of mixed channels."""
        batch = [sample_email_raw, sample_chat_raw, sample_docs_comment_raw, sample_forum_raw]
        results = parse_ticket_batch(batch)

        assert len(results) == 4
        assert [t.channel for t in results] == [
            ChannelEnum.EMAIL,
            ChannelEnum.CHAT,
            ChannelEnum.DOCS_COMMENT,
            ChannelEnum.FORUM,
        ]


class TestGroundTruthIsolation:
    """Tests verifying complete separation between live runtime tickets and evaluation data."""

    def test_unified_ticket_does_not_expose_labels_or_history(
        self, sample_labeled_raw: dict
    ) -> None:
        """Verify parse_ticket strips ground-truth labels and history from UnifiedTicket."""
        ticket = parse_ticket(sample_labeled_raw)

        assert isinstance(ticket, UnifiedTicket)
        # Ensure no label or history attributes exist on live UnifiedTicket
        assert not hasattr(ticket, "labels")
        assert not hasattr(ticket, "history")
        assert not hasattr(ticket, "raw_labels")
        assert not hasattr(ticket, "raw_history")

        # Verify model dump excludes any labels or history keys
        dump = ticket.model_dump()
        assert "labels" not in dump
        assert "history" not in dump
        assert "raw_labels" not in dump
        assert "raw_history" not in dump

    def test_evaluation_record_preserves_source_labels_and_history(
        self, sample_labeled_raw: dict
    ) -> None:
        """Verify parse_evaluation_record preserves ground truth for evaluation tooling."""
        eval_rec = parse_evaluation_record(sample_labeled_raw)

        assert isinstance(eval_rec, EvaluationRecord)
        assert eval_rec.ticket.ticket_id == "DEV-0001"
        assert eval_rec.labels is not None
        assert eval_rec.labels.intent == "deployment_failure"
        assert eval_rec.labels.urgency == "high"
        assert eval_rec.labels.expected_route == "auto_respond"
        assert eval_rec.labels.answerable_from_docs is True
        assert eval_rec.labels.expected_doc_ids == ["DOC-DEPLOY-001"]
        assert eval_rec.labels.must_not_auto_respond is False

        assert eval_rec.history is not None
        assert eval_rec.history.first_contact_resolution is True
        assert eval_rec.history.resolution_time_minutes == 30
        assert eval_rec.history.csat_rating == 5
        assert eval_rec.history.escalated is False
        assert eval_rec.history.repeat_contact is False


class TestValidationAndEdgeCases:
    """Tests for edge cases, missing fields, and error handling (AC-11)."""

    def test_missing_ticket_id_raises_error(self) -> None:
        """Verify missing ticket_id raises TicketParsingError."""
        with pytest.raises(TicketParsingError, match="ticket_id"):
            parse_ticket(
                {
                    "channel": "email",
                    "body": "Hello",
                    "received_at": "2026-05-10T12:00:00Z",
                }
            )

    def test_empty_ticket_id_raises_error(self) -> None:
        """Verify whitespace-only ticket_id raises TicketParsingError."""
        with pytest.raises(TicketParsingError, match="ticket_id"):
            parse_ticket(
                {
                    "ticket_id": "   ",
                    "channel": "email",
                    "body": "Hello",
                    "received_at": "2026-05-10T12:00:00Z",
                }
            )

    def test_missing_channel_raises_error(self) -> None:
        """Verify missing channel raises TicketParsingError."""
        with pytest.raises(TicketParsingError, match="Missing required field 'channel'"):
            parse_ticket(
                {
                    "ticket_id": "TEST-001",
                    "body": "Hello",
                    "received_at": "2026-05-10T12:00:00Z",
                }
            )

    def test_unsupported_channel_raises_error(self) -> None:
        """Verify unsupported channel like 'sms' raises descriptive TicketParsingError."""
        with pytest.raises(TicketParsingError, match="Unsupported channel 'sms'"):
            parse_ticket(
                {
                    "ticket_id": "TEST-002",
                    "channel": "sms",
                    "body": "Hello",
                    "received_at": "2026-05-10T12:00:00Z",
                }
            )

    def test_missing_received_at_raises_error(self) -> None:
        """Verify missing received_at timestamp raises TicketParsingError."""
        with pytest.raises(TicketParsingError, match="received_at"):
            parse_ticket({"ticket_id": "TEST-003", "channel": "email", "body": "Hello"})

    def test_malformed_timestamp_raises_error(self) -> None:
        """Verify non-ISO or invalid datetime raises TicketParsingError."""
        with pytest.raises(TicketParsingError, match="Schema validation failed"):
            parse_ticket(
                {
                    "ticket_id": "TEST-004",
                    "channel": "email",
                    "body": "Hello",
                    "received_at": "invalid-datetime-string",
                }
            )

    def test_malformed_json_syntax_raises_error(self) -> None:
        """Verify invalid JSON string syntax raises TicketParsingError."""
        with pytest.raises(TicketParsingError, match="Malformed JSON payload"):
            parse_ticket("{invalid json...")

    def test_invalid_payload_type_raises_error(self) -> None:
        """Verify invalid input type raises TicketParsingError."""
        with pytest.raises(TicketParsingError, match="Expected dict, str, or bytes"):
            parse_ticket(12345)  # type: ignore

    def test_empty_body_handled_without_failure(self) -> None:
        """Verify empty body does not crash parser, conforming to AC-11."""
        ticket = parse_ticket(
            {
                "ticket_id": "TEST-EMPTY-001",
                "channel": "chat",
                "body": "",
                "received_at": "2026-05-10T12:00:00Z",
            }
        )
        assert ticket.body == ""
        assert ticket.clean_text == ""

    def test_unusual_characters_and_emojis(self) -> None:
        """Verify unusual characters (Unicode, emojis, tabs) are preserved properly."""
        unusual_text = "Deploy failed: ⚠️ Error 500 in namespace 'prod-🚀' \n\t\u00a9 2026"
        ticket = parse_ticket(
            {
                "ticket_id": "TEST-UNICODE-001",
                "channel": "forum",
                "subject": "🚨 Critical issue",
                "body": unusual_text,
                "received_at": "2026-05-10T12:00:00Z",
            }
        )
        assert "⚠️" in ticket.body
        assert "🚀" in ticket.body
        assert "🚨" in ticket.subject

    def test_missing_customer_tier_and_fluency_remain_unknown(self) -> None:
        """Verify missing customer metadata explicitly defaults to UNKNOWN rather than assuming."""
        minimal_ticket = parse_ticket(
            {
                "ticket_id": "TEST-MINIMAL-001",
                "channel": "chat",
                "body": "Need quick help.",
                "received_at": "2026-05-10T12:00:00Z",
            }
        )
        assert minimal_ticket.customer_id == "CUST-UNKNOWN"
        assert minimal_ticket.customer_name == "Unknown Customer"
        assert minimal_ticket.customer_tier == CustomerTier.UNKNOWN
        assert minimal_ticket.customer_region is None
        assert minimal_ticket.language_fluency == LanguageFluency.UNKNOWN

    def test_invalid_customer_tier_raises_error(self) -> None:
        """Verify invalid customer tier raises TicketParsingError."""
        with pytest.raises(TicketParsingError, match="customer_tier"):
            parse_ticket(
                {
                    "ticket_id": "TEST-TIER-001",
                    "channel": "email",
                    "body": "Help",
                    "received_at": "2026-05-10T12:00:00Z",
                    "customer_tier": "unlimited_platinum",
                }
            )

    def test_invalid_customer_region_raises_error(self) -> None:
        """Verify invalid customer region raises TicketParsingError."""
        with pytest.raises(TicketParsingError, match="customer_region"):
            parse_ticket(
                {
                    "ticket_id": "TEST-REGION-001",
                    "channel": "email",
                    "body": "Help",
                    "received_at": "2026-05-10T12:00:00Z",
                    "customer_region": "antarctica",
                }
            )

    def test_clean_text_property(self) -> None:
        """Verify clean_text combines subject and body with clean spacing."""
        ticket_with_sub = parse_ticket(
            {
                "ticket_id": "TEST-CLEAN-001",
                "channel": "email",
                "subject": "Header Title",
                "body": "Detailed paragraph.",
                "received_at": "2026-05-10T12:00:00Z",
            }
        )
        assert ticket_with_sub.clean_text == "Header Title\n\nDetailed paragraph."

        ticket_no_sub = parse_ticket(
            {
                "ticket_id": "TEST-CLEAN-002",
                "channel": "chat",
                "body": "Chat line only.",
                "received_at": "2026-05-10T12:00:00Z",
            }
        )
        assert ticket_no_sub.clean_text == "Chat line only."


class TestRealDevelopmentDatasetSamples:
    """Integration checks against real development tickets (DEV-0001 through DEV-0005)."""

    def test_parse_sample_tickets_from_dev_file_as_live_tickets(self) -> None:
        """Verify parser converts raw records to live UnifiedTicket without labels/history."""
        dev_file = Path("data/dev/development_tickets.json")
        if not dev_file.exists():
            pytest.skip("development_tickets.json not present in workspace")

        with open(dev_file, encoding="utf-8") as f:
            tickets_raw = json.load(f)

        # Take first 20 records spanning different channels
        sample_batch = tickets_raw[:20]
        parsed = parse_ticket_batch(sample_batch)

        assert len(parsed) == 20
        for original, normalized in zip(sample_batch, parsed, strict=True):
            assert normalized.ticket_id == original["ticket_id"]
            assert normalized.channel.value == original["channel"]
            assert normalized.customer_id == original["customer_id"]
            # Assert live ticket does not expose labels or history
            assert not hasattr(normalized, "labels")
            assert not hasattr(normalized, "history")
            assert not hasattr(normalized, "raw_labels")
            assert not hasattr(normalized, "raw_history")
            if original["channel"] == "chat":
                assert normalized.subject == ""
            else:
                assert normalized.subject == original["subject"]

    def test_parse_entire_development_dataset_as_evaluation_records(self) -> None:
        """Verify all 500 records can be parsed into EvaluationRecords via dataset adapter."""
        dev_file = Path("data/dev/development_tickets.json")
        if not dev_file.exists():
            pytest.skip("development_tickets.json not present in workspace")

        with open(dev_file, encoding="utf-8") as f:
            tickets_raw = json.load(f)

        eval_records = parse_evaluation_dataset(tickets_raw)
        assert len(eval_records) == 500

        for original, record in zip(tickets_raw, eval_records, strict=True):
            assert isinstance(record, EvaluationRecord)
            assert record.ticket.ticket_id == original["ticket_id"]
            assert record.labels is not None
            assert record.labels.intent == original["labels"]["intent"]
            assert record.labels.urgency == original["labels"]["urgency"]
            assert record.history is not None
            assert (
                record.history.first_contact_resolution
                == original["history"]["first_contact_resolution"]
            )
