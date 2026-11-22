"""Parsers for multi-channel ticket normalization into UnifiedTicket models.

Supports Email, Live Chat, Documentation Comments, and Community Forum channels.
Preserves all original inquiry text, timestamps, customer details, and metadata
while strictly enforcing that ground-truth labels and history are never leaked
into runtime UnifiedTicket instances.

Provides dataset adapter functions (`parse_evaluation_record`, `parse_evaluation_dataset`)
for evaluation tooling requiring access to benchmark labels.
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from pydantic import ValidationError

from cloudserve.ingestion.models import (
    ChannelEnum,
    CustomerRegion,
    CustomerTier,
    EvaluationRecord,
    LanguageFluency,
    TicketHistory,
    TicketLabels,
    UnifiedTicket,
)


class TicketParsingError(ValueError):
    """Raised when an inbound ticket payload is malformed or cannot be normalized."""


class BaseChannelParser(ABC):
    """Abstract base class for channel-specific ticket parsers."""

    channel: ChannelEnum

    @abstractmethod
    def parse(self, raw_data: dict[str, Any]) -> UnifiedTicket:
        """Parse raw channel payload into a UnifiedTicket."""


class EmailParser(BaseChannelParser):
    """Parser for email channel tickets."""

    channel = ChannelEnum.EMAIL

    def parse(self, raw_data: dict[str, Any]) -> UnifiedTicket:
        """Normalize email ticket payload."""
        subject = raw_data.get("subject", "")
        normalized_subject = str(subject).strip() if subject is not None else ""

        return _build_unified_ticket(
            raw_data=raw_data,
            channel=self.channel,
            subject=normalized_subject,
            channel_metadata=raw_data.get("channel_metadata", {}),
        )


class ChatParser(BaseChannelParser):
    """Parser for live chat channel tickets."""

    channel = ChannelEnum.CHAT

    def parse(self, raw_data: dict[str, Any]) -> UnifiedTicket:
        """Normalize live chat ticket payload.

        Per dataset specification, live chat inquiries do not have subject lines;
        any provided subject is normalized to an empty string.
        """
        metadata = dict(raw_data.get("channel_metadata", {}))
        original_subject = raw_data.get("subject")
        if original_subject:
            metadata["original_subject"] = str(original_subject).strip()

        return _build_unified_ticket(
            raw_data=raw_data,
            channel=self.channel,
            subject="",  # Live chat always has an empty subject
            channel_metadata=metadata,
        )


class DocsCommentParser(BaseChannelParser):
    """Parser for documentation comment channel tickets."""

    channel = ChannelEnum.DOCS_COMMENT

    def parse(self, raw_data: dict[str, Any]) -> UnifiedTicket:
        """Normalize documentation comment ticket payload."""
        subject = raw_data.get("subject", "")
        normalized_subject = str(subject).strip() if subject is not None else ""

        return _build_unified_ticket(
            raw_data=raw_data,
            channel=self.channel,
            subject=normalized_subject,
            channel_metadata=raw_data.get("channel_metadata", {}),
        )


class ForumParser(BaseChannelParser):
    """Parser for community forum channel tickets."""

    channel = ChannelEnum.FORUM

    def parse(self, raw_data: dict[str, Any]) -> UnifiedTicket:
        """Normalize community forum ticket payload."""
        subject = raw_data.get("subject", "")
        normalized_subject = str(subject).strip() if subject is not None else ""

        return _build_unified_ticket(
            raw_data=raw_data,
            channel=self.channel,
            subject=normalized_subject,
            channel_metadata=raw_data.get("channel_metadata", {}),
        )


_PARSERS: dict[ChannelEnum, BaseChannelParser] = {
    ChannelEnum.EMAIL: EmailParser(),
    ChannelEnum.CHAT: ChatParser(),
    ChannelEnum.DOCS_COMMENT: DocsCommentParser(),
    ChannelEnum.FORUM: ForumParser(),
}


def _build_unified_ticket(
    raw_data: dict[str, Any],
    channel: ChannelEnum,
    subject: str,
    channel_metadata: dict[str, Any],
) -> UnifiedTicket:
    """Helper to build and validate a runtime UnifiedTicket from raw data and normalized fields.

    Crucial: Labels and history are deliberately NOT attached to UnifiedTicket.
    """
    ticket_id = raw_data.get("ticket_id")
    if not ticket_id or not str(ticket_id).strip():
        raise TicketParsingError("Missing or empty required field 'ticket_id'.")

    body = raw_data.get("body")
    normalized_body = str(body) if body is not None else ""

    received_at = raw_data.get("received_at")
    if not received_at:
        raise TicketParsingError(f"Missing required field 'received_at' in ticket '{ticket_id}'.")

    # Customer identifiers with explicit defaults
    customer_id = str(raw_data.get("customer_id", "CUST-UNKNOWN"))
    customer_name = str(raw_data.get("customer_name", "Unknown Customer"))

    # Explicit unknown handling: do not silently assume STANDARD or FLUENT
    raw_tier = raw_data.get("customer_tier")
    if raw_tier:
        try:
            customer_tier = CustomerTier(raw_tier)
        except ValueError as e:
            raise TicketParsingError(
                f"Invalid customer_tier '{raw_tier}' in ticket '{ticket_id}'."
            ) from e
    else:
        customer_tier = CustomerTier.UNKNOWN

    raw_region = raw_data.get("customer_region")
    customer_region = None
    if raw_region:
        try:
            customer_region = CustomerRegion(raw_region)
        except ValueError as e:
            raise TicketParsingError(
                f"Invalid customer_region '{raw_region}' in ticket '{ticket_id}'."
            ) from e

    raw_fluency = raw_data.get("language_fluency")
    if raw_fluency:
        try:
            language_fluency = LanguageFluency(raw_fluency)
        except ValueError as e:
            raise TicketParsingError(
                f"Invalid language_fluency '{raw_fluency}' in ticket '{ticket_id}'."
            ) from e
    else:
        language_fluency = LanguageFluency.UNKNOWN

    try:
        return UnifiedTicket(
            ticket_id=str(ticket_id).strip(),
            channel=channel,
            subject=subject,
            body=normalized_body,
            received_at=received_at,  # Pydantic validates ISO datetime strings
            customer_id=customer_id,
            customer_name=customer_name,
            customer_tier=customer_tier,
            customer_region=customer_region,
            language_fluency=language_fluency,
            channel_metadata=channel_metadata,
        )
    except ValidationError as err:
        raise TicketParsingError(
            f"Schema validation failed for ticket '{ticket_id}': {err}"
        ) from err


def _extract_dict_payload(payload: dict[str, Any] | str | bytes) -> dict[str, Any]:
    """Helper to convert str/bytes/dict to dict safely."""
    if isinstance(payload, (str, bytes)):
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as err:
            raise TicketParsingError(f"Malformed JSON payload: {err}") from err
    elif isinstance(payload, dict):
        data = payload
    else:
        raise TicketParsingError(
            f"Expected dict, str, or bytes payload, got {type(payload).__name__}."
        )
    if not isinstance(data, dict):
        raise TicketParsingError(f"Expected JSON object dictionary, got {type(data).__name__}.")
    return data


def parse_ticket(payload: dict[str, Any] | str | bytes) -> UnifiedTicket:
    """Parse and normalize a raw ticket dictionary or JSON string into a live UnifiedTicket.

    Ground-truth labels and history are excluded from the returned object to ensure
    runtime isolation.

    Args:
        payload: Inbound ticket dictionary or serialized JSON string/bytes.

    Returns:
        UnifiedTicket: Normalized live runtime ticket instance.

    Raises:
        TicketParsingError: If payload cannot be parsed, has an unsupported channel,
                            or fails schema validation.
    """
    data = _extract_dict_payload(payload)

    raw_channel = data.get("channel")
    if not raw_channel:
        raise TicketParsingError(
            "Missing required field 'channel'. Expected one of: email, chat, docs_comment, forum."
        )

    try:
        channel_enum = ChannelEnum(raw_channel)
    except ValueError as err:
        supported = ", ".join(c.value for c in ChannelEnum)
        raise TicketParsingError(
            f"Unsupported channel '{raw_channel}'. Expected one of: {supported}."
        ) from err

    parser = _PARSERS.get(channel_enum)
    if not parser:
        raise TicketParsingError(f"No registered parser for channel '{channel_enum}'.")

    return parser.parse(data)


def parse_ticket_batch(
    payloads: list[dict[str, Any]] | str | bytes,
) -> list[UnifiedTicket]:
    """Parse a list of raw ticket payloads deterministically into live UnifiedTickets.

    Args:
        payloads: List of ticket dictionaries or serialized JSON array string/bytes.

    Returns:
        list[UnifiedTicket]: List of successfully normalized runtime tickets.

    Raises:
        TicketParsingError: If batch decoding fails or any individual ticket is malformed.
    """
    if isinstance(payloads, (str, bytes)):
        try:
            records = json.loads(payloads)
        except json.JSONDecodeError as err:
            raise TicketParsingError(f"Malformed JSON array batch: {err}") from err
    elif isinstance(payloads, list):
        records = payloads
    else:
        raise TicketParsingError(
            f"Expected list, str, or bytes batch, got {type(payloads).__name__}."
        )

    return [parse_ticket(rec) for rec in records]


def parse_evaluation_record(payload: dict[str, Any] | str | bytes) -> EvaluationRecord:
    """Parse a development or validation ticket into a composite EvaluationRecord.

    Extracts ground-truth labels and history exclusively for evaluation pipelines,
    while creating a live-compatible UnifiedTicket.

    Args:
        payload: Raw dataset ticket dictionary or JSON string.

    Returns:
        EvaluationRecord: Containing live UnifiedTicket alongside validated labels and history.
    """
    data = _extract_dict_payload(payload)
    ticket = parse_ticket(data)

    labels = None
    raw_labels = data.get("labels")
    if isinstance(raw_labels, dict):
        try:
            labels = TicketLabels(**raw_labels)
        except ValidationError as err:
            raise TicketParsingError(
                f"Invalid labels in ticket '{ticket.ticket_id}': {err}"
            ) from err

    history = None
    raw_history = data.get("history")
    if isinstance(raw_history, dict):
        try:
            history = TicketHistory(**raw_history)
        except ValidationError as err:
            raise TicketParsingError(
                f"Invalid history in ticket '{ticket.ticket_id}': {err}"
            ) from err

    return EvaluationRecord(ticket=ticket, labels=labels, history=history)


logger = logging.getLogger(__name__)


def parse_evaluation_dataset(
    payloads: list[dict[str, Any]] | str | bytes,
    strict: bool = True,
) -> list[EvaluationRecord]:
    """Parse an entire dataset file (e.g. development_tickets.json) into EvaluationRecords.

    Args:
        payloads: List of dataset dictionaries or serialized JSON array string/bytes.
        strict: If True, raises TicketParsingError on the first invalid record.
            If False, logs a warning and skips invalid records.

    Returns:
        list[EvaluationRecord]: Parsed evaluation records.
    """
    if isinstance(payloads, (str, bytes)):
        try:
            records = json.loads(payloads)
        except json.JSONDecodeError as err:
            raise TicketParsingError(f"Malformed JSON array batch: {err}") from err
    elif isinstance(payloads, list):
        records = payloads
    else:
        raise TicketParsingError(
            f"Expected list, str, or bytes batch, got {type(payloads).__name__}."
        )

    if strict:
        return [parse_evaluation_record(rec) for rec in records]

    parsed: list[EvaluationRecord] = []
    for rec in records:
        try:
            parsed.append(parse_evaluation_record(rec))
        except Exception as exc:
            logger.warning("Skipping malformed evaluation record: %s", exc)
    return parsed
