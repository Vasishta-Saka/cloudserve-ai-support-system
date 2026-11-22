"""Multi-channel ticket ingestion, normalization, and evaluation dataset adapters."""

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
from cloudserve.ingestion.parsers import (
    BaseChannelParser,
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

__all__ = [
    "ChannelEnum",
    "CustomerRegion",
    "CustomerTier",
    "LanguageFluency",
    "UnifiedTicket",
    "TicketLabels",
    "TicketHistory",
    "EvaluationRecord",
    "BaseChannelParser",
    "EmailParser",
    "ChatParser",
    "DocsCommentParser",
    "ForumParser",
    "TicketParsingError",
    "parse_ticket",
    "parse_ticket_batch",
    "parse_evaluation_record",
    "parse_evaluation_dataset",
]
