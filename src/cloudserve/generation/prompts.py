"""Prompt loader and template formatter for evidence-grounded response generation."""

import re
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

from cloudserve.classification.models import ClassificationResult
from cloudserve.ingestion.models import UnifiedTicket
from cloudserve.rag.models import RetrievedPassage

_CANONICAL_PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts" / "response_generation"

_BOUNDARY_DELIMITER_PATTERN = re.compile(
    r"---+(\s*(?:BEGIN|END)\s+[A-Za-z0-9_ -]+)---+",
    re.IGNORECASE,
)


def escape_prompt_delimiters(text: str) -> str:
    """Safely escape delimiter-like sequences within untrusted text.

    Replaces occurrences of '--- BEGIN ... ---' or '--- END ... ---' within
    ticket subjects, bodies, or retrieved passages with escaped versions
    (e.g., r'\\-\\-\\- ... \\-\\-\\-') so that untrusted content cannot fake an end
    marker or prematurely terminate boundary blocks.

    Preserves original data models unmodified (only affects prompt formatting).
    """
    if not text:
        return text

    def _replace_delimiter(match: re.Match[str]) -> str:
        inner = match.group(1)
        return rf"\-\-\-{inner}\-\-\-"

    escaped = _BOUNDARY_DELIMITER_PATTERN.sub(_replace_delimiter, text)

    exact_delimiters = [
        "--- BEGIN UNTRUSTED TICKET SUBJECT ---",
        "--- END UNTRUSTED TICKET SUBJECT ---",
        "--- BEGIN UNTRUSTED TICKET BODY ---",
        "--- END UNTRUSTED TICKET BODY ---",
        "--- BEGIN RETRIEVED PASSAGE",
        "--- END RETRIEVED PASSAGE",
    ]
    for delim in exact_delimiters:
        if delim in escaped:
            escaped = escaped.replace(delim, delim.replace("---", r"\-\-\-"))

    return escaped


def get_canonical_prompt_path(version: str = "v1") -> Path:
    """Return the authoritative canonical path to the response generation prompt template."""
    return _CANONICAL_PROMPTS_DIR / f"{version}.txt"


@lru_cache(maxsize=4)
def load_generation_system_prompt(version: str = "v1") -> str:
    """Load the version-controlled system prompt template from prompts/response_generation/."""
    canonical_path = get_canonical_prompt_path(version)
    if not canonical_path.is_file():
        raise FileNotFoundError(
            f"Canonical response generation prompt template version '{version}' "
            f"not found at: {canonical_path}"
        )
    return canonical_path.read_text(encoding="utf-8")


def format_generation_prompt(
    ticket: UnifiedTicket,
    passages: Sequence[RetrievedPassage] | None = None,
    classification: ClassificationResult | None = None,
) -> str:
    """Format ticket and retrieved passages into the user prompt for response generation."""
    raw_subject = ticket.subject if ticket.subject else "(No subject provided)"
    safe_subject = escape_prompt_delimiters(raw_subject)
    safe_body = escape_prompt_delimiters(ticket.body.strip())

    sections = [
        "### CUSTOMER SUPPORT TICKET (UNTRUSTED EXTERNAL DATA):",
        f"Ticket ID: {ticket.ticket_id}",
        f"Channel: {ticket.channel.value}",
        f"Customer Tier: {ticket.customer_tier.value}",
        "--- BEGIN UNTRUSTED TICKET SUBJECT ---",
        safe_subject,
        "--- END UNTRUSTED TICKET SUBJECT ---",
        "--- BEGIN UNTRUSTED TICKET BODY ---",
        safe_body,
        "--- END UNTRUSTED TICKET BODY ---",
    ]

    if classification is not None:
        sections.extend(
            [
                "",
                "### TICKET CLASSIFICATION CONTEXT:",
                f"Predicted Intent: {classification.intent.value}",
                f"Predicted Urgency: {classification.urgency.value}",
                f"Intent Confidence: {classification.intent_confidence:.2f}",
            ]
        )

    sections.extend(["", "### RETRIEVED KNOWLEDGE BASE PASSAGES (REFERENCE DATA):"])

    if not passages:
        sections.append("(No approved knowledge-base passages were retrieved for this ticket.)")
    else:
        for idx, passage in enumerate(passages, start=1):
            safe_title = escape_prompt_delimiters(passage.title)
            safe_section = escape_prompt_delimiters(passage.section_name)
            safe_content = escape_prompt_delimiters(passage.content.strip())
            sections.extend(
                [
                    f"--- BEGIN RETRIEVED PASSAGE {idx} ---",
                    f"Citation ID: {passage.citation_id}",
                    f"Document ID: {passage.doc_id}",
                    f"Title: {safe_title}",
                    f"Section: {safe_section}",
                    "Content:",
                    safe_content,
                    f"--- END RETRIEVED PASSAGE {idx} ---",
                    "",
                ]
            )

    sections.extend(
        [
            "### INSTRUCTIONS:",
            "Review the ticket and retrieved passages.",
            (
                "1. Every factual statement and troubleshooting bullet in customer_response "
                "must end with its exact supporting Citation ID inline "
                "(e.g. [DOC-XXX-001: Section])."
            ),
            (
                "2. The citation_ids array must contain exactly the unique Citation IDs "
                "appearing inline in customer_response (exact correspondence required; "
                "citations in agent_note alone do not count)."
            ),
            (
                "3. Align troubleshooting instructions with terminology used in retrieved "
                "documentation; prefer diagnostic checks explicitly described in retrieved "
                "passages and do NOT invent operational fixes, port modifications, or redeployment "
                "actions absent from the documentation."
            ),
            (
                "4. Do NOT invent, truncate, or fabricate citation identifiers. "
                "Citations must match the retrieved Citation IDs exactly."
            ),
            (
                "5. If retrieved passages do not contain sufficient evidence to answer "
                "the ticket, set customer_response to null, citation_ids to [], and "
                "insufficient_evidence to true. Do NOT invent an answer."
            ),
            (
                "6. Keep agent_note concise (strictly 1-2 sentences) summarizing the core inquiry "
                "and documentation support."
            ),
            (
                "7. Return strictly the requested JSON object containing customer_response, "
                "agent_note, citation_ids, insufficient_evidence, and unsupported_claims."
            ),
        ]
    )

    return "\n".join(sections)
