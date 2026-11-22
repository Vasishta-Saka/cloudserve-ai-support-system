"""Conservative grounding and citation validation engine for generated responses.

Implements Fix 1 and Fix 2 for Task 5.1:
- Extracts bracketed citation references from customer response text.
- Enforces strict agreement between response citations and citation_ids payload.
- Validates citations against supplied retrieved knowledge-base passages.
- Rejects ungrounded or uncited factual statements.
- Prevents unauthorized completed account or financial action claims.
- Evaluates claim-level lexical and technical concept support.
- Fails closed when claim support cannot be established reliably.
"""

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from cloudserve.rag.models import RetrievedPassage
from cloudserve.routing.citations import extract_doc_id_from_citation, normalize_citation_id

logger = logging.getLogger(__name__)

CITATION_REGEX = re.compile(r"(\[[A-Za-z0-9_\-#]+(?::\s*[^\]]+)?\])")

COMMON_STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "aren't",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "cannot",
    "could",
    "couldn't",
    "did",
    "didn't",
    "do",
    "does",
    "doesn't",
    "doing",
    "don't",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "hadn't",
    "has",
    "hasn't",
    "have",
    "haven't",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "isn't",
    "it",
    "it's",
    "its",
    "itself",
    "let's",
    "me",
    "more",
    "most",
    "mustn't",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "ought",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "shan't",
    "she",
    "should",
    "shouldn't",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "wasn't",
    "we",
    "were",
    "weren't",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "with",
    "won't",
    "would",
    "wouldn't",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "please",
    "cloudserve",
    "support",
    "system",
    "customer",
    "team",
    "hello",
    "thank",
    "thanks",
    "regards",
    "existing",
    "current",
    "reflect",
    "reflects",
    "specifically",
    "per",
    "actually",
    "currently",
    "described",
    "documentation",
    "pattern",
    "matches",
    "aligns",
    "corresponds",
    "outlined",
    "noted",
    "guide",
    "docs",
}

CONVERSATIONAL_PATTERNS = [
    # 1. Standalone greetings
    re.compile(
        r"^(?:hello|hi|dear|greetings)(?:[,\s]+(?:there|team|customer|all|[A-Za-z]+))?[,!.]*$",
        re.IGNORECASE,
    ),
    # 2. Courtesy thanks and acknowledgments
    re.compile(
        r"^(?:(?:hello|hi|dear|greetings)[,\s]+)?(?:thank\s+you|thanks)"
        r"(?:\s+(?:for\s+(?:contacting(?:\s+(?:us|(?:cloudserve\s+|customer\s+)?support))?|"
        r"reaching(\s+out)?(?:\s+to\s+(?:us|(?:cloudserve\s+|customer\s+)?support))?|"
        r"your\s+(?:patience|ticket|inquiry|report|request)|"
        r"(?:cloudserve\s+|customer\s+)?support)|very\s+much))?[.!]*$",
        re.IGNORECASE,
    ),
    # 3. Formal sign-offs
    re.compile(
        r"^(?:[-*]\s*)?(best\s+regards|warm\s+regards|kind\s+regards|regards|sincerely|"
        r"cheers|thanks|thank\s+you)"
        r"(?:,?\s*(?:the\s+)?(?:[A-Za-z0-9_-]+\s+)*support\s+team)?,?\s*[.!]*$",
        re.IGNORECASE,
    ),
    # 4. Standalone support team signatures
    re.compile(
        r"^(?:[-*]\s*)?(?:(?:the\s+)?(?:cloudserve\s+|customer\s+)?support\s+team|"
        r"cloudserve\s+support|customer\s+support)[.!]*$",
        re.IGNORECASE,
    ),
    # 5. Structural list introductions
    re.compile(
        r"^(?:to\s+(?:troubleshoot|diagnose|resolve|address)(?:\s+(?:this|the))?(?:\s+issue)?(?:[,\s]+(?:please|you\s+can))?|"
        r"you\s+can\s+(?:either|also)|"
        r"here\s+(?:are|is)\s+(?:the\s+)?(?:options|steps|recommendations)|"
        r"(?:please\s+)?follow\s+these\s+steps|to\s+resolve\s+this(?:\s+issue)?|"
        r"options(?:\s+to\s+consider)?|recommended\s+actions?):?$",
        re.IGNORECASE,
    ),
    # 6. Courteous closings and follow-up offers
    re.compile(
        r"^(?:please\s+)?(?:let\s+us\s+know|feel\s+free\s+to\s+reach\s+out)"
        r"(?:\s+(?:the\s+results(?:\s+(?:or|and))?\s+)?(?:if|whether)\s+(?:you\s+(?:have|need)\s+(?:any\s+)?(?:further\s+)?"
        r"(?:questions?|assistance|clarification|help)|"
        r"(?:the\s+)?(?:issue|problem)\s+persists(?:\s+after\s+making\s+these\s+adjustments)?|"
        r"there\s+is\s+anything\s+else)|"
        r"\s+the\s+results)?[.!]*$|"
        r"^(?:if\s+you\s+have\s+(?:any\s+)?(?:further\s+)?questions?[,\s]+)?"
        r"(?:please\s+)?(?:let\s+us\s+know|feel\s+free\s+to\s+reach\s+out)[.!]*$|"
        r"^hope\s+this\s+helps(?:\s+resolve\s+(?:the\s+|this\s+)?(?:issue|problem))?[.!]*$",
        re.IGNORECASE,
    ),
    # 7. Empathy and ticket receipt acknowledgments
    re.compile(
        r"^(?:(?:hello|hi|dear|greetings)[,\s]+)?"
        r"(?:we\s+understand\s+your\s+(?:urgency|frustration|concern)"
        r"(?:\s+(?:with\s+this\s+issue|and\s+(?:are\s+happy\s+to\s+assist|here\s+to\s+help)))?|"
        r"(?:i\s*['’]?m|we\s*['’]?re)\s+sorry\s+"
        r"(?:to\s+hear(?:\s+that)?(?:\s+you\s*['’]?re\s+(?:experiencing|facing|having)\s+"
        r"(?:this|an?|any)?\s*(?:issue|problem|trouble|difficulties))?|"
        r"for\s+(?:the|any)\s+inconvenience(?:\s+caused)?|"
        r"you\s*['’]?re\s+(?:experiencing|facing|having)\s+(?:this|an?|any)?\s*"
        r"(?:issue|problem|trouble|difficulties))|"
        r"we\s+apologize\s+for\s+(?:the\s+inconvenience|any\s+inconvenience(?:\s+caused)?)|"
        r"we\s+have\s+received\s+your\s+(?:ticket|request|inquiry|report)"
        r"(?:\s+and\s+are\s+happy\s+to\s+assist)?)[.!]*$",
        re.IGNORECASE,
    ),
    # 8. Attribution and citation-explanation clauses
    re.compile(
        r"^(?:(?:based\s+on|according\s+to|as\s+detailed\s+in|per)\s+(?:our\s+)?(?:documentation|docs|knowledge\s*base)|"
        r"(?:this|that)\s+(?:matches|aligns\s+with|corresponds\s+to|reflects)\s+(?:the\s+)?(?:symptom\s+|issue\s+|error\s+)?(?:pattern|description|details|behavior|scenario)\s+(?:described|outlined|detailed|documented|noted)\s+(?:in|under)\s+(?:our\s+)?(?:documentation|docs|guide|knowledge\s*base)?)\s*[.!]*$",
        re.IGNORECASE,
    ),
]

TECHNICAL_INDICATOR_PATTERNS = [
    re.compile(
        r"(\bcsctl\b|\bkubectl\b|\bapi\b|\bpeering\b|\btimeout\b|\bconfig\b|"
        r"\bport\b|\brefund\b|\$\d+|"
        r"\bdeployment\b|\broll\s*back\b|\brollback\b|\bhealth[-_\s]*check\b|"
        r"\bpods?\b|\bcontainers?\b|\bclusters?\b|\bingress\b|\bservice\.yaml\b|"
        r"\bconnection\s+pool\b|\bdatabase\b|\bdb\b|\breplicas?\b|\btargetport\b|"
        r"\bnodes?\b|\bcpu\b|\bmemory\b|\bstorage\b|\bsubnets?\b|\bvpc\b|\bgateway\b|"
        r"\blatency\b|\bprorat(?:ed|ion)\b)",
        re.IGNORECASE,
    ),
]

UNSUPPORTED_OPERATIONAL_STATUS_PATTERNS = [
    re.compile(
        r"\b(?:our\s+|the\s+)?(?:[A-Za-z0-9_-]+\s+)*team\s+(?:is|will)\s+(?:actively\s+|currently\s+)?"
        r"(reviewing|investigating|looking\s+into|handling|working\s+on|analyzing|"
        r"review|investigate|look\s+into)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(we\s+are|we're|i\s+am|i'm)\s+(?:actively\s+|currently\s+)?"
        r"(reviewing|investigating|looking\s+into|working\s+on)\s+(?:your|the)\s+"
        r"(?:account|incident|ticket|issue|system|logs)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(is|are)\s+(?:currently\s+|actively\s+)?(?:being\s+)?"
        r"(reviewed|investigated|analyzed|handled)\s+by\s+(?:our|the)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:our\s+team|we|i)\s*(?:['’]ll|\s+will|\s+can)\s+(?:actively\s+|further\s+|manually\s+)?"
        r"(?:investigate|look\s+into|review|fix|resolve|handle|contact\s+you|reach\s+out\s+to\s+you)\b",
        re.IGNORECASE,
    ),
]

UNAUTHORIZED_ACTION_PATTERNS = [
    re.compile(
        r"\b(we\s+have|i\s+have|has\s+been|was)\s+(issued|processed|refunded|credited|waived|"
        r"cancelled)\b.*?\b(refund|credit|\$\d+|fee|charge|invoice)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(refund|credit|\$\d+|fee\s+waiver)\s+(has\s+been|was|is)\s+"
        r"(issued|processed|credited|approved|granted)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(we\s+have|i\s+have)\s+(granted|approved|credited|refunded)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(we\s+have|i\s+have|has\s+been)\s+(deleted|purged|rolled\s+back|terminated|"
        r"reset|unlocked)\s+(your|the)\b",
        re.IGNORECASE,
    ),
]

OPERATIONAL_ACTION_PATTERNS = [
    re.compile(r"\b(redeploy|restart|reboot|reconfigure)\b", re.IGNORECASE),
    re.compile(
        r"\b(align|modify|update|change|set|edit|patch|replace|switch|bind|adjust|tweak|alter)\b"
        r"(?:\s+\w+){0,6}?\s+"
        r"\b(targetport|containerport|port|dockerfile|manifest|service\.yaml|service\s+definition|configuration|config|settings?|parameters?|env(?:ironment)?\s+variables?|revision|dns|ingress|route|cluster|node|pod)\b",
        re.IGNORECASE,
    ),
]

OPERATIONAL_TARGET_RESOURCE_PATTERN = re.compile(
    r"\b(targetport|containerport|dockerfile|manifest|service\.yaml|service\s+definition|dns|ingress|route|cluster|node|pod)\b",
    re.IGNORECASE,
)

OPERATIONAL_WARNING_PATTERN = re.compile(
    r"\b(warning|caution|danger|alert|risk)\b",
    re.IGNORECASE,
)

OPERATIONAL_NEGATION_PREFIXES = (
    r"\b(do\s+not|don'?t|never|avoid|not\s+recommended\s+to|should\s+not|"
    r"must\s+not|cannot|prohibited\s+from)\b"
)


def get_action_stems(verb: str) -> set[str]:
    """Generate inflected morphological forms and common derivations of an action verb."""
    v = verb.lower()
    stems = {v}
    if v.endswith("e"):
        stems.add(v[:-1] + "ing")
        stems.add(v + "d")
    else:
        stems.add(v + "ing")
        stems.add(v + "ed")
    stems.add(v + "s")
    if v == "redeploy":
        stems.update({"redeployment", "redeployments", "re-deploy", "re-deployment"})
    elif v == "restart":
        stems.update({"restarting", "restarts"})
    elif v == "reconfigure":
        stems.update({"reconfiguration", "reconfigurations"})
    elif v == "align":
        stems.update({"alignment", "aligning"})
    elif v == "modify":
        stems.update({"modification", "modifications", "modifying", "modified"})
    return stems


def validate_operational_claim(claim: str, passage_text: str) -> str | None:
    """Validate whether an operational recommendation is strictly authorized by cited documentation.

    Returns an error message describing the violation if the claim prescribes an operational
    action, resource mutation, or parameter unsupported or contradicted by the cited passages.
    Returns None if the claim is diagnostic, non-operational, or fully supported.
    """
    p_lower = passage_text.lower()
    cleaned_claim = CITATION_REGEX.sub("", claim)
    found_actions: list[tuple[str, str | None]] = []

    m1 = re.search(r"\b(redeploy|restart|reboot|reconfigure)\b", cleaned_claim, re.IGNORECASE)
    if m1:
        found_actions.append((m1.group(1).lower(), None))

    for m2 in re.finditer(
        r"\b(align|modify|update|change|set|edit|patch|replace|switch|bind|adjust|tweak|alter)\b"
        r"(?:\s+\w+){0,6}?\s+"
        r"\b(targetport|containerport|port|dockerfile|manifest|service\.yaml|service\s+definition|configuration|config|settings?|parameters?|env(?:ironment)?\s+variables?|revision|dns|ingress|route|cluster|node|pod)\b",
        cleaned_claim,
        re.IGNORECASE,
    ):
        v = m2.group(1).lower()
        t = m2.group(2).lower()
        found_actions.append((v, t))

    if not found_actions:
        return None

    for verb, target in found_actions:
        stems = get_action_stems(verb)

        # 1. Contradiction / Negation detection: e.g. 'do not redeploy yet'
        for stem in stems:
            neg_pat = rf"{OPERATIONAL_NEGATION_PREFIXES}\s+(?:\w+\s+){{0,3}}?{re.escape(stem)}\b"
            neg_match = re.search(neg_pat, p_lower)
            if neg_match:
                return (
                    f"Contradictory operational recommendation: cited documentation cautions "
                    f"against '{verb}' ('{neg_match.group(0)}')."
                )

        # 2. Warning-only context detection: e.g. 'Warning: Redeployment will cause downtime'
        verb_sentences = [
            s.strip()
            for s in re.split(r"[.\n;]+", p_lower)
            if any(re.search(r"\b" + re.escape(st) + r"\b", s) for st in stems)
        ]
        if verb_sentences:
            all_warning = all(OPERATIONAL_WARNING_PATTERN.search(s) for s in verb_sentences)
            has_affirmative = any(
                re.search(rf"\b{re.escape(st)}\b\s+(after|the|your|when|to)\b", s)
                or re.search(rf"\b\d+\.\s+.*?\b{re.escape(st)}\b", s)
                for s in verb_sentences
                for st in stems
            )
            if all_warning and not has_affirmative:
                return (
                    f"Unsupported operational recommendation: action '{verb}' is only mentioned "
                    "in a warning/risk context in cited documentation without affirmative "
                    "guidance."
                )

        # 3. Action presence in passage
        action_in_passage = any(re.search(r"\b" + re.escape(st) + r"\b", p_lower) for st in stems)
        if not action_in_passage:
            return (
                f"Unsupported operational recommendation: action '{verb}' is not authorized "
                f"by cited documentation."
            )

        # 4. Target resource presence in passage
        if target:
            target_norm = target.replace(" ", "")
            t_in_passage = target in p_lower or target_norm in p_lower.replace(" ", "")
            if not t_in_passage:
                return (
                    f"Unsupported operational recommendation: target resource '{target}' for "
                    f"action '{verb}' is not supported by cited documentation."
                )

        # 5. Check all target resources mentioned in the operational claim
        for t_match in OPERATIONAL_TARGET_RESOURCE_PATTERN.finditer(cleaned_claim):
            res_term = t_match.group(0).lower()
            res_norm = res_term.replace(" ", "")
            if res_term not in p_lower and res_norm not in p_lower.replace(" ", ""):
                return (
                    f"Unsupported operational recommendation: target resource '{res_term}' "
                    f"is not supported by cited documentation."
                )

        # 6. Parameter validation (e.g. port 8080 vs 9090)
        claim_nums = re.findall(r"\b\d{2,5}\b", cleaned_claim)
        for num in claim_nums:
            if num not in p_lower:
                return (
                    f"Unsupported operational recommendation: parameter '{num}' in claim "
                    f"is not supported by cited documentation."
                )

    return None


@dataclass(frozen=True)
class GroundingValidationResult:
    """Outcome of conservative grounding and citation validation."""

    is_grounded: bool
    verified_citations: list[str] = field(default_factory=list)
    unsupported_reasons: list[str] = field(default_factory=list)
    details: str = ""


def extract_citations_from_text(text: str) -> list[str]:
    """Extract all bracketed citations from text preserving order of appearance."""
    raw_citations = CITATION_REGEX.findall(text)
    seen: set[str] = set()
    extracted: list[str] = []
    for c in raw_citations:
        norm = normalize_citation_id(c)
        if norm not in seen:
            seen.add(norm)
            extracted.append(norm)
    return extracted


SHORT_CONVERSATIONAL_EXPRESSIONS = frozenset(
    {
        "hi",
        "hey",
        "hello",
        "ok",
        "okay",
        "thanks",
        "thank you",
        "bye",
        "cheers",
        "yes",
        "no",
    }
)

UNICODE_HYPHENS = re.compile(r"[\u2010\u2011\u2012\u2013\u2014]")

SENSITIVE_CREDENTIAL_PATTERN = re.compile(
    r"\b(passwords?|secrets?|tokens?|api[_-]?keys?|credentials?|auth[_-]?tokens?|private[_-]?keys?)\b",
    re.IGNORECASE,
)

SAFE_DIAGNOSTIC_LOG_REQUEST_PATTERN = re.compile(
    r"^(?:if\s+(?:the\s+)?(?:issue|problem)\s+persists(?:\s+after\s+these\s+steps)?[,\s]+)?"
    r"(?:please\s+)?(?:share|provide|send)\s+"
    r"(?:the\s+|us\s+)?(?:relevant\s+)?"
    r"(?:build-log|build\s+log|error\s+log|diagnostic\s+log|build|error|diagnostic|service|container)?\s*"
    r"(?:excerpts?|logs?|outputs?|details?|messages?)"
    r"(?:\s+so\s+we\s+can\s+(?:help|assist)(?:\s+further)?)?[.!]*$",
    re.IGNORECASE,
)

LIST_INTRO_PATTERN = re.compile(
    r"^(?:to\s+(?:troubleshoot|diagnose|resolve|address)(?:\s+(?:this|the))?(?:\s+issue)?(?:[,\s]+(?:please|you\s+can))?|"
    r"(?:please\s+)?follow\s+these\s+steps|"
    r"you\s+can\s+(?:either|also)|"
    r"here\s+(?:are|is)\s+(?:the\s+)?(?:options?|steps?|recommendations?)|"
    r"options(?:\s+to\s+consider)?|"
    r"recommended\s+actions?|"
    r"(?:try|consider)\s+the\s+following(?:\s+steps|\s+options)?):?$",
    re.IGNORECASE,
)


def is_conversational_sentence(sentence: str) -> bool:
    """Return True if sentence is purely boilerplate greeting, closing, or empathy.

    Short utterances are not given blanket exemption; only recognized conversational
    expressions (such as 'Hi', 'Thanks', 'OK') are permitted without citation.
    """
    s = sentence.strip()
    if not s:
        return True

    # Pure punctuation / whitespace / list bullets have no factual claims
    if not re.search(r"[A-Za-z0-9]", s):
        return True

    # Operational status assertions or unauthorized actions can NEVER be conversational
    for op_pat in UNSUPPORTED_OPERATIONAL_STATUS_PATTERNS:
        if op_pat.search(s):
            return False
    for act_pat in UNAUTHORIZED_ACTION_PATTERNS:
        if act_pat.search(s):
            return False

    # Technical commands, parameters, identifiers, currency, or technical domain concepts
    # can NEVER be conversational, UNLESS part of a recognized safe diagnostic log request.
    s_norm = UNICODE_HYPHENS.sub("-", s)
    is_safe_diag = bool(SAFE_DIAGNOSTIC_LOG_REQUEST_PATTERN.match(s_norm)) and not bool(
        SENSITIVE_CREDENTIAL_PATTERN.search(s_norm)
    )
    if not is_safe_diag:
        for tech_pat in TECHNICAL_INDICATOR_PATTERNS:
            if tech_pat.search(s):
                return False
    else:
        return True

    # Check for explicitly recognized short conversational expressions
    normalized_short = re.sub(r"[^\w\s]", "", s.lower()).strip()
    if normalized_short in SHORT_CONVERSATIONAL_EXPRESSIONS:
        return True

    # Strip bracketed citations and attribution prefixes so forms like
    # 'This matches the symptom pattern described in our documentation [DOC-DEPLOY-003: Symptoms].'
    # match attribution and courtesy patterns cleanly without requiring literal passage overlap
    cleaned_conv = CITATION_REGEX.sub("", s).strip()

    for pat in CONVERSATIONAL_PATTERNS:
        if pat.search(s) or (cleaned_conv and pat.search(cleaned_conv)):
            return True

    return False


def split_sentences(text: str) -> list[str]:
    """Split text into individual sentences and compound claims preserving citations."""
    raw_parts = re.split(
        r"(?<=[.!?])\s+(?=[A-Z0-9\[])|;\s+|,\s*(?:then|and\s+then)\s+|\n+",
        text,
        flags=re.IGNORECASE,
    )
    return [p.strip() for p in raw_parts if p.strip()]


def extract_informative_terms(sentence: str) -> list[str]:
    """Extract domain terms and technical identifiers from a sentence."""
    # Strip bracketed citations first so they don't count as sentence terms
    cleaned = CITATION_REGEX.sub("", sentence)
    # Extract words or hyphenated identifiers of length >= 3
    words = re.findall(r"[A-Za-z0-9_-]{3,}", cleaned.lower())
    terms = [w for w in words if w not in COMMON_STOPWORDS and not w.isdigit()]
    return terms


def is_list_item(line: str) -> bool:
    """Return True if line begins with a list bullet or numbering."""
    return bool(re.match(r"^\s*([-*+]|\d+[.)])\s+\S", line))


def is_list_intro(line: str) -> bool:
    """Return True only if line matches an explicitly recognized structural list introduction."""
    s = line.strip()
    if not s:
        return False
    # Strip any bracketed citations and attribution prefixes so forms like
    # 'Based on [DOC-1: Res], please follow these steps:' or
    # 'To troubleshoot, please [DOC-INT-002: Resolution]:' or
    # 'Per [DOC-1], try the following:' match.
    cleaned = CITATION_REGEX.sub("", s).strip()
    cleaned = re.sub(
        r"^(?:based\s+on|according\s+to|per)\s*,?\s*", "", cleaned, flags=re.IGNORECASE
    ).strip()
    cleaned = re.sub(r"\s*:\s*$", ":", cleaned)
    return bool(LIST_INTRO_PATTERN.match(cleaned))


def is_citation_only_line(line: str) -> bool:
    """Return True if line contains only citation tags."""
    s = line.strip()
    return bool(re.match(r"^(\[[A-Za-z0-9_\-#]+(?::\s*[^\]]+)?\]\s*)+$", s))


def extract_inline_numbered_items(line: str) -> list[str] | None:
    """Extract inline numbered list items from a line when formatted as an ordered list.

    Recognizes sequences starting at 1 (e.g. '1. ... 2. ...' or '1) ... 2) ...') while
    strictly preserving ordinary numeric references such as port 80, version 2.0, or quantities.
    """
    marker_re = re.compile(r"(?:^|(?<=[.!?]|\s))(\d+)[.)]\s+")
    matches = list(marker_re.finditer(line))
    if not matches:
        return None

    # Must start with item '1'
    start_match = None
    for m in matches:
        if m.group(1) == "1":
            start_match = m
            break

    if not start_match:
        return None

    expected_num = 2
    matched_sequence = [start_match]
    curr_pos = start_match.end()

    while True:
        next_pat = re.compile(rf"(?:^|(?<=[.!?]|\s)){expected_num}[.)]\s+")
        m_next = next_pat.search(line, curr_pos)
        if m_next:
            matched_sequence.append(m_next)
            expected_num += 1
            curr_pos = m_next.end()
        else:
            break

    # Must contain at least two sequential numbered items to be an inline list
    if len(matched_sequence) < 2:
        return None

    items: list[str] = []
    prefix = line[: matched_sequence[0].start()].strip()
    if prefix:
        items.append(prefix)

    for i in range(len(matched_sequence)):
        start = matched_sequence[i].start()
        end = matched_sequence[i + 1].start() if i + 1 < len(matched_sequence) else len(line)
        item_text = line[start:end].strip()
        if item_text:
            items.append(item_text)

    # Detach any trailing conversational or distinct sentence from last item after citation
    if items and len(items) > 1:
        last_item = items[-1]
        m_trail = re.search(r"(\[[A-Za-z0-9_\-#]+(?::\s*[^\]]+)?\][.]?)\s+([A-Z].+)$", last_item)
        if m_trail:
            items[-1] = last_item[: m_trail.start(2)].strip()
            items.append(m_trail.group(2).strip())

    return items


def parse_response_blocks(text: str) -> list[dict[str, Any]]:
    """Parse response text into structural blocks: paragraphs and list blocks."""
    raw_lines = [line.strip() for line in text.split("\n")]
    lines: list[str] = []
    for raw_line in raw_lines:
        if not raw_line:
            lines.append("")
            continue
        inline_items = extract_inline_numbered_items(raw_line)
        if inline_items:
            lines.extend(inline_items)
        else:
            lines.append(raw_line)

    blocks: list[dict[str, Any]] = []
    current_list_items: list[str] = []
    current_list_intro: str | None = None
    current_para_lines: list[str] = []

    def flush_para() -> None:
        nonlocal current_para_lines
        if current_para_lines:
            para_text = "\n".join(current_para_lines).strip()
            if para_text:
                blocks.append({"type": "paragraph", "text": para_text})
            current_para_lines = []

    def flush_list() -> None:
        nonlocal current_list_items, current_list_intro
        if current_list_items:
            blocks.append(
                {
                    "type": "list",
                    "intro": current_list_intro,
                    "items": list(current_list_items),
                }
            )
            current_list_items = []
            current_list_intro = None

    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if not line:
            peek_idx = idx + 1
            while peek_idx < len(lines) and not lines[peek_idx]:
                peek_idx += 1
            if peek_idx < len(lines):
                next_line = lines[peek_idx]
                if is_list_item(next_line) or (
                    current_list_items and is_citation_only_line(next_line)
                ):
                    idx += 1
                    continue
            flush_para()
            flush_list()
            idx += 1
            continue

        if is_list_item(line):
            if current_para_lines and is_list_intro(current_para_lines[-1]):
                current_list_intro = current_para_lines.pop()
            flush_para()
            current_list_items.append(line)
            idx += 1
            continue

        if current_list_items and is_citation_only_line(line):
            current_list_items.append(line)
            flush_list()
            idx += 1
            continue

        if is_list_intro(line):
            peek_idx = idx + 1
            while peek_idx < len(lines) and not lines[peek_idx]:
                peek_idx += 1
            if peek_idx < len(lines) and is_list_item(lines[peek_idx]):
                flush_para()
                flush_list()
                current_list_intro = line
                idx += 1
                continue

        flush_list()
        current_para_lines.append(line)
        idx += 1

    flush_para()
    flush_list()
    return blocks


def resolve_passages_for_citations(
    cids: Sequence[str],
    passages: Sequence[RetrievedPassage],
) -> list[RetrievedPassage]:
    """Resolve citations strictly to their matching retrieved passages.

    If citations are section-specific (e.g. '[DOC-DEPLOY-001: Resolution]' or
    'DOC-DEPLOY-001#resolution'), evidence is strictly isolated to the passages
    matching those specific sections. Other sections of the same document (such as
    'Notes' or 'Symptoms') are NOT included.

    Multi-section citations (e.g. citing both '[DOC-DEPLOY-001: Resolution]' and
    '[DOC-DEPLOY-001: Notes]') combine evidence from all explicitly cited sections.

    Unsectioned document-level citations (e.g. '[DOC-DEPLOY-001]') expand to all
    retrieved passages for that document, but ONLY if no section-scoped citation
    for that document is already present.
    """
    resolved: list[RetrievedPassage] = []
    seen_chunk_ids: set[str] = set()
    section_scoped_doc_ids: set[str] = set()

    # Index passages by exact and canonical section identifiers
    section_map: dict[str, list[RetrievedPassage]] = {}
    for p in passages:
        norm_cid = normalize_citation_id(p.citation_id)
        section_map.setdefault(norm_cid, []).append(p)
        section_map.setdefault(p.chunk_id, []).append(p)
        section_map.setdefault(f"[{p.chunk_id}]", []).append(p)
        clean_norm = norm_cid.strip("[] \t\n\r")
        if ":" in clean_norm:
            d_part, s_part = clean_norm.split(":", 1)
            canon_sec = f"{d_part.strip().upper()}:{s_part.strip().lower()}"
            section_map.setdefault(canon_sec, []).append(p)
            section_map.setdefault(f"[{canon_sec}]", []).append(p)
        if "#" in p.chunk_id:
            d_part, s_part = p.chunk_id.split("#", 1)
            canon_hash = f"{d_part.strip().upper()}#{s_part.strip().lower()}"
            section_map.setdefault(canon_hash, []).append(p)

    # 1. Resolve all explicitly section-scoped citations
    for c in cids:
        norm_c = normalize_citation_id(c)
        if not norm_c:
            continue
        clean_c = norm_c.strip("[] \t\n\r")
        matched_passages: list[RetrievedPassage] = []
        if norm_c in section_map:
            matched_passages = section_map[norm_c]
        elif ":" in clean_c:
            d_part, s_part = clean_c.split(":", 1)
            canon_key = f"{d_part.strip().upper()}:{s_part.strip().lower()}"
            matched_passages = section_map.get(canon_key, [])
        elif "#" in clean_c:
            d_part, s_part = clean_c.split("#", 1)
            canon_key = f"{d_part.strip().upper()}#{s_part.strip().lower()}"
            matched_passages = section_map.get(canon_key, [])

        if matched_passages:
            for p in matched_passages:
                section_scoped_doc_ids.add(p.doc_id.upper())
                if p.chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(p.chunk_id)
                    resolved.append(p)

    # 2. Resolve unsectioned document-level citations
    for c in cids:
        norm_c = normalize_citation_id(c)
        if not norm_c:
            continue
        clean_c = norm_c.strip("[] \t\n\r")
        # An unsectioned citation contains no ':' or '#'
        if ":" in clean_c or "#" in clean_c:
            continue
        doc_id = clean_c
        # If this document already had an explicitly section-scoped citation,
        # document-level fallback must NEVER override/expand it.
        if doc_id.upper() in section_scoped_doc_ids:
            continue
        for p in passages:
            if p.doc_id.upper() == doc_id.upper() and p.chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(p.chunk_id)
                resolved.append(p)

    return resolved


def validate_grounding(
    customer_response: str | None,
    citation_ids: Sequence[str],
    passages: Sequence[RetrievedPassage],
    model_unsupported_claims: Sequence[str] | None = None,
) -> GroundingValidationResult:
    """Perform conservative grounding and citation validation.

    Checks:
    1. If customer_response is None or empty, returns grounded.
    2. Delimited citation extraction from text.
    3. Agreement between extracted citations and citation_ids payload.
    4. Exact resolution against supplied retrieved passages.
    5. Detection of unauthorized completed action claims (e.g., unauthorized refunds).
    6. Identification of uncited factual statements.
    7. Claim-level lexical and technical concept support against cited passages.

    Args:
        customer_response: The customer-facing response draft (or None).
        citation_ids: Citation identifiers reported by the model.
        passages: The retrieved knowledge-base passages provided as context.
        model_unsupported_claims: Claims flagged by the model (not treated as proof of correctness).

    Returns:
        GroundingValidationResult indicating whether customer response may be released.
    """
    if customer_response is None or not customer_response.strip():
        return GroundingValidationResult(
            is_grounded=True,
            verified_citations=[],
            unsupported_reasons=[],
            details="No customer-facing draft present; grounding validation passed.",
        )

    response_text = customer_response.strip()
    unsupported_reasons: list[str] = []

    # 1. Check for unauthorized completed actions (Prompt injection / Fraud / Hallucination)
    for pat in UNAUTHORIZED_ACTION_PATTERNS:
        match = pat.search(response_text)
        if match:
            reason = (
                f"Unauthorized action claim detected: response claims a completed "
                f"administrative or financial action ('{match.group(0)}'), which is not "
                f"supported by static documentation."
            )
            unsupported_reasons.append(reason)
            logger.warning(reason)

    # 1b. Check for unsupported operational-status claims (Fix 1)
    for pat in UNSUPPORTED_OPERATIONAL_STATUS_PATTERNS:
        match = pat.search(response_text)
        if match:
            reason = (
                f"Unsupported operational-status claim detected: response asserts an operational "
                f"event ('{match.group(0)}'), which cannot be verified by static documentation."
            )
            unsupported_reasons.append(reason)
            logger.warning(reason)

    # 2. Extract citations directly from response text
    extracted_citations = extract_citations_from_text(response_text)
    payload_citations = [normalize_citation_id(c) for c in citation_ids if str(c).strip()]

    if not extracted_citations:
        unsupported_reasons.append(
            "Customer response contains factual claims but includes zero identifiable "
            "citations in the response text."
        )

    # 3. Ensure citation_ids payload agrees with citations in response text
    extracted_set = set(extracted_citations)
    payload_set = set(payload_citations)

    if extracted_set != payload_set:
        unsupported_reasons.append(
            f"Citation mismatch: citations in text ({sorted(extracted_set)}) do not match "
            f"citation_ids in payload ({sorted(payload_set)})."
        )

    # 4. Ensure every citation matches a supplied retrieved passage exactly
    valid_citation_ids: set[str] = set()
    for p in passages:
        norm_cid = normalize_citation_id(p.citation_id)
        valid_citation_ids.add(norm_cid)
        valid_citation_ids.add(p.chunk_id)
        valid_citation_ids.add(p.doc_id)
        valid_citation_ids.add(f"[{p.doc_id}]")
        valid_citation_ids.add(f"[{p.chunk_id}]")

    invalid_citations: list[str] = []
    verified_citations: list[str] = []
    for c in extracted_citations:
        if c in valid_citation_ids:
            verified_citations.append(c)
        else:
            invalid_citations.append(c)

    if invalid_citations:
        unsupported_reasons.append(
            f"Customer response references citations not found in retrieved passages: "
            f"{invalid_citations}."
        )

    # 5. Block-level and sentence-level analysis for uncited factual statements and claim grounding
    blocks = parse_response_blocks(response_text)

    for block in blocks:
        if block["type"] == "list":
            intro = block.get("intro")
            items = block.get("items", [])
            full_list_text = (intro + "\n" if intro else "") + "\n".join(items)
            list_citations = extract_citations_from_text(full_list_text)

            # Check intro if present
            if intro:
                if not is_list_intro(intro) and not is_conversational_sentence(intro):
                    intro_citations = extract_citations_from_text(intro)
                    cids = intro_citations if intro_citations else list_citations
                    if not cids:
                        unsupported_reasons.append(
                            f"Uncited factual statement detected: '{intro[:80]}...' "
                            f"requires a supporting citation."
                        )
                    else:
                        op_passages = resolve_passages_for_citations(cids, passages)
                        op_comb = " ".join(
                            [
                                f"{p.doc_id} {p.title} {p.section_name} {p.content}"
                                for p in op_passages
                            ]
                        ).lower()
                        op_err = validate_operational_claim(intro, op_comb)
                        if op_err:
                            unsupported_reasons.append(op_err)

                        cited_doc_ids = {p.doc_id for p in op_passages}
                        if not cited_doc_ids:
                            cited_doc_ids = {extract_doc_id_from_citation(c) for c in cids if c}
                        rel_passages = [p for p in passages if p.doc_id in cited_doc_ids]
                        comb_text = " ".join(
                            [
                                f"{p.doc_id} {p.title} {p.section_name} {p.content}"
                                for p in rel_passages
                            ]
                        ).lower()
                        terms = extract_informative_terms(intro)
                        if terms:
                            supp = [t for t in terms if t in comb_text]
                            unsupp = [t for t in terms if t not in comb_text]
                            overlap = len(supp) / len(terms)
                            if overlap < 0.40:
                                unsupported_reasons.append(
                                    f"Claim in '{intro[:80]}...' contains terms unsupported by "
                                    f"cited passages: {unsupp[:5]} (overlap: {overlap:.1%})."
                                )

            if not list_citations:
                for line in items:
                    cleaned_line = re.sub(r"^\s*([-*+]|\d+[.)])\s+", "", line).strip()
                    if not is_conversational_sentence(cleaned_line) and not is_list_intro(
                        cleaned_line
                    ):
                        unsupported_reasons.append(
                            f"Uncited factual statement detected: '{cleaned_line[:80]}...' "
                            f"requires a supporting citation."
                        )
                continue

            # List has citations: verify each list item independently
            for line in items:
                cleaned_line = re.sub(r"^\s*([-*+]|\d+[.)])\s+", "", line).strip()
                if is_citation_only_line(cleaned_line):
                    continue
                if is_conversational_sentence(cleaned_line) or is_list_intro(cleaned_line):
                    continue

                item_cids = extract_citations_from_text(cleaned_line)
                target_cids = item_cids if item_cids else list_citations

                sub_claims = split_sentences(cleaned_line)
                for sub_claim in sub_claims:
                    if (
                        is_citation_only_line(sub_claim)
                        or is_conversational_sentence(sub_claim)
                        or is_list_intro(sub_claim)
                    ):
                        continue
                    item_terms = extract_informative_terms(sub_claim)
                    if not item_terms:
                        continue

                    claim_cids = extract_citations_from_text(sub_claim)
                    active_cids = claim_cids if claim_cids else target_cids
                    op_passages = resolve_passages_for_citations(active_cids, passages)
                    op_comb = " ".join(
                        [f"{p.doc_id} {p.title} {p.section_name} {p.content}" for p in op_passages]
                    ).lower()
                    op_err = validate_operational_claim(sub_claim, op_comb)
                    if op_err:
                        unsupported_reasons.append(op_err)

                    cited_doc_ids = {p.doc_id for p in op_passages}
                    if not cited_doc_ids:
                        cited_doc_ids = {extract_doc_id_from_citation(c) for c in active_cids if c}
                    rel_passages = [p for p in passages if p.doc_id in cited_doc_ids]
                    comb_text = " ".join(
                        [f"{p.doc_id} {p.title} {p.section_name} {p.content}" for p in rel_passages]
                    ).lower()

                    supp = [t for t in item_terms if t in comb_text]
                    unsupp = [t for t in item_terms if t not in comb_text]
                    overlap = len(supp) / len(item_terms)
                    if overlap < 0.40:
                        unsupported_reasons.append(
                            f"Claim in '{sub_claim[:80]}...' contains terms unsupported by "
                            f"cited passages: {unsupp[:5]} (overlap: {overlap:.1%})."
                        )

        else:
            # Paragraph block
            para = str(block["text"])
            para_citations = extract_citations_from_text(para)
            sentences = split_sentences(para)

            if not para_citations:
                # Paragraph has no citations: ensure all sentences are purely conversational
                for sent in sentences:
                    if not is_conversational_sentence(sent):
                        unsupported_reasons.append(
                            f"Uncited factual statement detected: '{sent[:80]}...' "
                            f"requires a supporting citation."
                        )
                continue

            # Paragraph has citations
            para_op_passages = resolve_passages_for_citations(para_citations, passages)
            para_doc_ids = {p.doc_id for p in para_op_passages}
            if not para_doc_ids:
                para_doc_ids = {extract_doc_id_from_citation(c) for c in para_citations if c}
            relevant_passages = [p for p in passages if p.doc_id in para_doc_ids]
            combined_passage_text = " ".join(
                [f"{p.doc_id} {p.title} {p.section_name} {p.content}" for p in relevant_passages]
            ).lower()

            for sent in sentences:
                if is_conversational_sentence(sent):
                    continue
                sent_terms = extract_informative_terms(sent)
                if not sent_terms:
                    continue

                sent_cids = extract_citations_from_text(sent)
                active_s_cids = sent_cids if sent_cids else para_citations
                s_op_passages = resolve_passages_for_citations(active_s_cids, passages)
                s_op_comb = " ".join(
                    [f"{p.doc_id} {p.title} {p.section_name} {p.content}" for p in s_op_passages]
                ).lower()
                op_err = validate_operational_claim(sent, s_op_comb)
                if op_err:
                    unsupported_reasons.append(op_err)

                if sent_cids:
                    s_doc_ids = {p.doc_id for p in s_op_passages}
                    if not s_doc_ids:
                        s_doc_ids = {extract_doc_id_from_citation(c) for c in sent_cids if c}
                    s_passages = [p for p in passages if p.doc_id in s_doc_ids]
                    s_comb = " ".join(
                        [f"{p.doc_id} {p.title} {p.section_name} {p.content}" for p in s_passages]
                    ).lower()
                else:
                    s_comb = combined_passage_text

                supported_terms = [t for t in sent_terms if t in s_comb]
                unsupported_terms = [t for t in sent_terms if t not in s_comb]
                overlap = len(supported_terms) / len(sent_terms)
                if overlap < 0.40:
                    unsupported_reasons.append(
                        f"Claim in '{sent[:80]}...' contains terms unsupported by cited passages: "
                        f"{unsupported_terms[:5]} (overlap: {overlap:.1%})."
                    )

    if unsupported_reasons:
        details_str = "; ".join(unsupported_reasons)
        if invalid_citations:
            details_str = f"Citation verification failed: {details_str}"
        return GroundingValidationResult(
            is_grounded=False,
            verified_citations=[],
            unsupported_reasons=unsupported_reasons,
            details=details_str,
        )

    return GroundingValidationResult(
        is_grounded=True,
        verified_citations=verified_citations,
        unsupported_reasons=[],
        details="Grounding validation passed; all claims grounded in retrieved passages.",
    )
