"""Prompt loader and template formatter for ticket classification."""

from functools import lru_cache
from pathlib import Path

from cloudserve.ingestion.models import UnifiedTicket

_PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts" / "classification"


@lru_cache(maxsize=4)
def load_classification_system_prompt(version: str = "v1") -> str:
    """Load the version-controlled system prompt template from prompts/classification/."""
    prompt_file = _PROMPTS_DIR / f"{version}.txt"
    if not prompt_file.exists():
        # Fallback to local default if prompts directory path differs
        local_path = Path("prompts/classification") / f"{version}.txt"
        if local_path.exists():
            return local_path.read_text(encoding="utf-8")
        raise FileNotFoundError(f"Classification prompt template not found at {prompt_file}")
    return prompt_file.read_text(encoding="utf-8")


def format_ticket_prompt(ticket: UnifiedTicket) -> str:
    """Format an ingested UnifiedTicket into user prompt text for the classifier."""
    subject_display = ticket.subject if ticket.subject else "(None - Live Chat)"
    return (
        f"Please classify the following support ticket:\n\n"
        f"Ticket ID: {ticket.ticket_id}\n"
        f"Channel: {ticket.channel.value}\n"
        f"Customer Tier: {ticket.customer_tier.value}\n"
        f"Subject: {subject_display}\n"
        f"Body:\n{ticket.body}\n\n"
        f"Return strictly the requested JSON object."
    )
