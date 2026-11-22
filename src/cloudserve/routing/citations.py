"""Citation verification engine for deterministic routing.

Verifies that all citations referenced in a response or candidate draft resolve
strictly to actually retrieved knowledge base passages. Blocks automatic release
if citations are missing, invalid, or hallucinated (FR-05, A6).
"""

from collections.abc import Sequence

from cloudserve.rag.models import RetrievedPassage
from cloudserve.routing.models import CitationValidationResult


def normalize_citation_id(citation: str) -> str:
    """Normalize citation string by stripping surrounding whitespace."""
    return citation.strip()


def extract_doc_id_from_citation(citation: str) -> str:
    """Extract parent doc_id from citation formatted as [DOC-XXX-001: Section] or raw ID."""
    cleaned = citation.strip("[] \t\n\r")
    if ":" in cleaned:
        return cleaned.split(":", 1)[0].strip()
    return cleaned


def verify_citations(
    passages: Sequence[RetrievedPassage],
    cited_ids: Sequence[str] | None = None,
    require_at_least_one: bool = True,
) -> CitationValidationResult:
    """Verify that cited identifiers strictly resolve to retrieved knowledge base passages.

    Accepts only exact matches against retrieved citation IDs, chunk IDs, or document IDs.
    Partial fragments or fabricated citations are strictly rejected (no substring matching).

    Args:
        passages: Ranked list of RetrievedPassage objects returned by retriever.
        cited_ids: Optional list of citation IDs or document IDs cited in response.
        require_at_least_one: If True, requires at least one valid citation when passages exist.

    Returns:
        CitationValidationResult with validity flag and detailed validation diagnostics.
    """
    if not cited_ids:
        if require_at_least_one:
            return CitationValidationResult(
                is_valid=False,
                total_cited=0,
                valid_citations=[],
                invalid_citations=[],
                details=(
                    "No citations provided; auto-response requires at least one verified citation."
                ),
            )
        return CitationValidationResult(
            is_valid=True,
            total_cited=0,
            valid_citations=[],
            invalid_citations=[],
            details="No citations cited and none required.",
        )

    # Collect valid identifiers from the retrieved passages
    valid_citation_ids = {p.citation_id for p in passages}
    valid_chunk_ids = {p.chunk_id for p in passages}
    valid_doc_ids = {p.doc_id for p in passages}

    valid_cited: list[str] = []
    invalid_cited: list[str] = []

    for raw_citation in cited_ids:
        norm = normalize_citation_id(raw_citation)

        # Strict exact matching: must match exact citation_id, chunk_id, or doc_id.
        # Substring matching is strictly prohibited to reject partial fragments.
        is_matched = (
            norm in valid_citation_ids
            or norm in valid_chunk_ids
            or norm in valid_doc_ids
            or f"[{norm}]" in valid_citation_ids
            or (norm.startswith("[") and norm.endswith("]") and norm.strip("[]") in valid_doc_ids)
        )

        if is_matched:
            valid_cited.append(norm)
        else:
            invalid_cited.append(norm)

    is_valid = len(invalid_cited) == 0 and (len(valid_cited) > 0 or not require_at_least_one)

    if is_valid:
        details = (
            f"Verified {len(valid_cited)} of {len(cited_ids)} citations "
            f"against {len(passages)} passages."
        )
    else:
        details = (
            f"Citation verification failed: {len(invalid_cited)} invalid or "
            f"unretrieved citations: {invalid_cited}."
        )

    return CitationValidationResult(
        is_valid=is_valid,
        total_cited=len(cited_ids),
        valid_citations=valid_cited,
        invalid_citations=invalid_cited,
        details=details,
    )
