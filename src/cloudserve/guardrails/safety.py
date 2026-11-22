"""Deterministic input safety scanner for PII and adversarial prompt injection.

Provides heuristic, pattern-based detection to protect downstream LLM components
from prompt injection and prevent automated exposure of sensitive customer data.

Limitations:
- Regex and pattern matching are heuristic defense-in-depth measures and cannot
  detect all adversarial jailbreaks, semantic evasions, obfuscated encodings, or
  atypical PII formats.
- This scanner does not replace dedicated identity verification or full compliance
  DLP engines.
- Fail-closed design: if safety evaluation cannot be completed with certainty,
  the check reports unsafe to prevent automated exposure.
"""

import logging
import re
from typing import Final

from cloudserve.ingestion.models import UnifiedTicket
from cloudserve.routing.models import GuardrailCheckResult

logger = logging.getLogger(__name__)

# Heuristic prompt-injection indicators (instruction overrides, jailbreak phrases, role hijacking)
PROMPT_INJECTION_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    (
        "ignore_instructions",
        re.compile(
            r"\bignore\s+(?:all\s+)?(?:previous|prior|above|former)?\s*(?:safety|operational|support|system|security)?\s*(?:instructions|prompts|rules|commands|guidelines|directives|policies)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "disregard_instructions",
        re.compile(
            r"\bdisregard\s+(?:all\s+)?(?:previous|prior|above|former)?\s*(?:safety|operational|support|system|security)?\s*(?:instructions|prompts|rules|commands|guidelines|directives|policies)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "forget_instructions",
        re.compile(
            r"\bforget\s+(?:all\s+)?(?:previous|prior|above|former)?\s*(?:safety|operational|support|system|security)?\s*(?:instructions|prompts|rules|commands|guidelines|directives|policies)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "override_prompt",
        re.compile(
            r"\b(?:(?:system|safety|security|operational)\s+(?:prompt|directive|instruction|rule|guideline)s?\s+override|(?:emergency\s+)?override\s+(?:all\s+|of\s+(?:all\s+)?)?(?:system|safety|security|operational)\s+(?:prompts?|instructions?|rules?|guidelines?|directives?))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "jailbreak_roleplay",
        re.compile(
            r"\b(?:you\s+are\s+now\s+(?:operating\s+under\s+|in\s+)?|operating\s+under\s+)(?:dan|developer\s+mode|unrestricted|jailbroken|an\s+ai\s+without\s+rules|(?:administrative\s+|system\s+|root\s+)?(?:root\s+mode|admin\s+mode|developer\s+mode))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "system_instruction_injection",
        re.compile(
            r"(?:\b(?:critical|emergency|new|mandatory)?\s*system\s+instruction\s*:|\bsystem\s*:\s*you\s+must\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "system_prompt_exfiltration",
        re.compile(
            r"\b(?:output|print|reveal|show|display)\s+(?:the\s+)?(?:full\s+)?(?:system\s+(?:prompt|instructions?|rules?)|initial\s+instructions?)\b",
            re.IGNORECASE,
        ),
    ),
]

# Credit-card candidate patterns (subject to mandatory Luhn checksum validation)
CREDIT_CARD_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    (
        "credit_card_standard",
        re.compile(
            r"\b(?:4[0-9]{3}|5[1-5][0-9]{2}|6(?:011|5[0-9]{2}))[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}\b",
        ),
    ),
    (
        "credit_card_amex",
        re.compile(
            r"\b3[47][0-9]{2}[\s\-]?[0-9]{6}[\s\-]?[0-9]{5}\b",
        ),
    ),
]

# Other high-risk sensitive data patterns (SSNs, private keys, plaintext passwords)
OTHER_PII_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    (
        "social_security_number",
        re.compile(
            r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b",
        ),
    ),
    (
        "private_key_block",
        re.compile(
            r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
            re.IGNORECASE,
        ),
    ),
    (
        "plaintext_credential",
        re.compile(
            r"\b(?:password|passwd|api[_-]?key|secret[_-]?token)\s*[:=]\s*[^\s,;]{8,}\b",
            re.IGNORECASE,
        ),
    ),
]

# Combined PII patterns for export and backwards compatibility
PII_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = CREDIT_CARD_PATTERNS + OTHER_PII_PATTERNS


def validate_luhn(candidate: str) -> bool:
    """Validate a candidate credit card number string using the Luhn checksum algorithm.

    Args:
        candidate: Candidate credit card number string (digits with optional spaces/hyphens).

    Returns:
        True if the candidate contains 13-19 digits and passes the Luhn checksum.
    """
    digits = [int(c) for c in candidate if c.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False

    checksum = 0
    # Process digits right to left; double every second digit starting from the second to last
    for i, digit in enumerate(reversed(digits)):
        if i % 2 == 1:
            doubled = digit * 2
            checksum += doubled - 9 if doubled > 9 else doubled
        else:
            checksum += digit

    return checksum % 10 == 0


class InputSafetyScanner:
    """Deterministic, pattern-based input safety scanner for customer inquiries.

    Scans inbound ticket content for prompt injection and sensitive credentials/PII.
    Never includes raw matched sensitive strings in logs or diagnostics.
    """

    def __init__(
        self,
        enable_injection_shield: bool = True,
        enable_pii_shield: bool = True,
    ) -> None:
        self.enable_injection_shield = enable_injection_shield
        self.enable_pii_shield = enable_pii_shield

    def scan_text(self, text: str) -> GuardrailCheckResult:
        """Scan raw input text for prompt injection and PII violations.

        Args:
            text: Customer inquiry text to evaluate.

        Returns:
            GuardrailCheckResult indicating whether the text is safe to process.
        """
        if not isinstance(text, str):
            logger.warning("Input safety scanner received non-string input. Failing closed.")
            return GuardrailCheckResult(
                is_safe=False,
                violation_type="invalid_input_type",
                details="Input text was not a valid string.",
            )

        # 1. Prompt Injection Scanning
        if self.enable_injection_shield:
            for rule_name, pattern in PROMPT_INJECTION_PATTERNS:
                if pattern.search(text):
                    logger.warning("Prompt injection pattern detected: %s", rule_name)
                    return GuardrailCheckResult(
                        is_safe=False,
                        violation_type="prompt_injection",
                        details=f"Potential prompt injection detected (pattern: {rule_name}).",
                    )

        # 2. PII / High-Risk Sensitive Data Scanning
        if self.enable_pii_shield:
            # 2a. Credit-card candidate validation: requires regex match AND valid Luhn checksum
            for cc_type, cc_pattern in CREDIT_CARD_PATTERNS:
                for match in cc_pattern.finditer(text):
                    candidate = match.group(0)
                    if validate_luhn(candidate):
                        # NEVER log the matched sensitive data; report only the category
                        logger.warning("Sensitive PII/credential pattern detected: %s", cc_type)
                        return GuardrailCheckResult(
                            is_safe=False,
                            violation_type="pii_detected",
                            details=(
                                f"Sensitive customer data detected (type: {cc_type}). "
                                "Inquiry must not be processed via automated LLM responses."
                            ),
                        )

            # 2b. Other sensitive patterns (SSNs, private keys, plaintext passwords)
            for pii_type, pattern in OTHER_PII_PATTERNS:
                if pattern.search(text):
                    # NEVER log the matched sensitive data; report only the category
                    logger.warning("Sensitive PII/credential pattern detected: %s", pii_type)
                    return GuardrailCheckResult(
                        is_safe=False,
                        violation_type="pii_detected",
                        details=(
                            f"Sensitive customer data detected (type: {pii_type}). "
                            "Inquiry must not be processed via automated LLM responses."
                        ),
                    )

        return GuardrailCheckResult(
            is_safe=True,
            violation_type=None,
            details="Input passed deterministic safety checks.",
        )

    def scan_ticket(self, ticket: UnifiedTicket) -> GuardrailCheckResult:
        """Scan a UnifiedTicket for safety violations across subject and body.

        Args:
            ticket: Ingested UnifiedTicket instance.

        Returns:
            GuardrailCheckResult indicating whether the ticket is safe to process.
        """
        combined_text = f"{ticket.subject}\n{ticket.body}".strip()
        return self.scan_text(combined_text)
