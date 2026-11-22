"""Pydantic data models for Knowledge Base articles, chunks, and retrieved passages.

Supports acceptance criterion A4 (identifiable KB passages) and A6 (grounded citations).
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class KBSection(BaseModel):
    """An individual semantic section parsed from a markdown documentation article."""

    model_config = ConfigDict(frozen=True)

    section_name: str = Field(description="Display title of the section (e.g. 'Resolution').")
    section_slug: str = Field(description="Normalized slug for identifiers (e.g. 'resolution').")
    content: str = Field(description="Raw markdown body of the section without heading prefix.")
    word_count: int = Field(default=0, ge=0, description="Word count of the section content.")


class KBArticle(BaseModel):
    """A full documentation article loaded from documentation.json."""

    model_config = ConfigDict(frozen=True)

    doc_id: str = Field(description="Unique document identifier, e.g. 'DOC-AUTH-001'.")
    title: str = Field(description="Full title of the documentation article.")
    category: str = Field(description="Knowledge base category (e.g. 'authentication').")
    applies_to: str = Field(
        default="",
        description="Target products or platforms (e.g. 'Console, CLI, SDK').",
    )
    content: str = Field(description="Complete raw markdown content of the article.")
    related_docs: list[str] = Field(
        default_factory=list,
        description="IDs of related documentation articles.",
    )
    last_reviewed_days_ago: int = Field(default=0, ge=0, description="Recency indicator in days.")


class KBChunk(BaseModel):
    """An indexed chunk of knowledge-base content prepared for vector storage."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(
        description="Deterministic chunk identifier, e.g. 'DOC-AUTH-001#resolution'."
    )
    doc_id: str = Field(description="Parent document ID, e.g. 'DOC-AUTH-001'.")
    category: str = Field(description="Article category.")
    title: str = Field(description="Article title.")
    section_name: str = Field(description="Section heading name.")
    section_slug: str = Field(description="Section normalized slug.")
    content: str = Field(description="Body content of the section.")
    contextual_text: str = Field(
        description="Enriched text including document context prefix for dense vector embedding."
    )
    applies_to: str = Field(default="", description="Platform applicability.")
    related_docs: list[str] = Field(default_factory=list, description="Related document IDs.")
    strategy: str = Field(default="section_aware", description="Chunking strategy name.")

    @property
    def citation_id(self) -> str:
        """Return standardized citation identifier, e.g. '[DOC-AUTH-001: Resolution]'."""
        return f"[{self.doc_id}: {self.section_name}]"

    def to_chroma_metadata(self) -> dict[str, Any]:
        """Convert chunk metadata to primitive types accepted by ChromaDB."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "category": self.category,
            "title": self.title,
            "section_name": self.section_name,
            "section_slug": self.section_slug,
            "applies_to": self.applies_to,
            "related_docs": ",".join(self.related_docs),
            "strategy": self.strategy,
            "citation_id": self.citation_id,
        }


class RetrievedPassage(BaseModel):
    """A ranked knowledge-base passage returned by the vector retriever."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(
        description="Identifier of the matched chunk, e.g. 'DOC-AUTH-001#resolution'."
    )
    doc_id: str = Field(description="Parent article ID, e.g. 'DOC-AUTH-001'.")
    category: str = Field(description="Category of the article.")
    title: str = Field(description="Title of the article.")
    section_name: str = Field(description="Name of the section.")
    content: str = Field(description="Passage content text.")
    similarity_score: float = Field(
        ge=-1.0,
        le=1.0,
        description="Cosine similarity score (1.0 - cosine_distance).",
    )
    rank: int = Field(ge=1, description="1-based retrieval rank.")
    citation_id: str = Field(
        description="Verifiable citation string, e.g. '[DOC-AUTH-001: Resolution]'."
    )
    applies_to: str = Field(default="", description="Applies to platforms or components.")
