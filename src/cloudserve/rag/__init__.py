"""Knowledge Base indexing, vector retrieval with Chroma, and citation verification.

Implements acceptance criterion A4 (traceable KB passage retrieval) and supports
criterion A6 (verifiable citations).
"""

from cloudserve.rag.chunker import (
    chunk_article_fixed_size,
    chunk_article_section_aware,
    chunk_kb,
    load_kb_articles,
    parse_article_sections,
)
from cloudserve.rag.indexer import (
    KBIndexer,
    MockEmbeddingFunction,
    SentenceTransformerEmbeddingAdapter,
)
from cloudserve.rag.models import (
    KBArticle,
    KBChunk,
    KBSection,
    RetrievedPassage,
)
from cloudserve.rag.retriever import KBRetriever

__all__ = [
    "KBArticle",
    "KBChunk",
    "KBIndexer",
    "KBRetriever",
    "KBSection",
    "MockEmbeddingFunction",
    "RetrievedPassage",
    "SentenceTransformerEmbeddingAdapter",
    "chunk_article_fixed_size",
    "chunk_article_section_aware",
    "chunk_kb",
    "load_kb_articles",
    "parse_article_sections",
]
