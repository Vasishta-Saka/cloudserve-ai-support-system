"""Knowledge base vector retrieval engine.

Implements acceptance criterion A4:
- Queries the supplied 29-article documentation corpus.
- Returns ranked passages with cosine similarity scores.
- Applies relevance threshold filtering (returns [] if below threshold).
- Returns verifiable citations ([doc_id: Section-Name]) resolving to real articles.
- Strictly isolated from evaluation ground-truth labels.
"""

import logging
from pathlib import Path
from typing import Any

from chromadb.api.models.Collection import Collection
from chromadb.api.types import Documents, EmbeddingFunction

from cloudserve.config import get_settings
from cloudserve.rag.indexer import KBIndexer
from cloudserve.rag.models import RetrievedPassage

logger = logging.getLogger(__name__)


class KBRetriever:
    """Retriever for querying knowledge-base passages from ChromaDB."""

    def __init__(
        self,
        collection: Collection | None = None,
        indexer: KBIndexer | None = None,
        persist_directory: Path | str | None = None,
        collection_name: str | None = None,
        embedding_function: EmbeddingFunction[Documents] | None = None,
        in_memory: bool = False,
        similarity_threshold: float | None = None,
    ) -> None:
        """Initialize retriever from an existing Collection, KBIndexer, or configuration."""
        self.similarity_threshold = similarity_threshold
        if collection is not None:
            self.collection = collection
        elif indexer is not None:
            self.collection = indexer.collection
        else:
            idx = KBIndexer(
                persist_directory=persist_directory,
                collection_name=collection_name,
                embedding_function=embedding_function,
                in_memory=in_memory,
            )
            self.collection = idx.collection

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        similarity_threshold: float | None = None,
        intent_hint: str | None = None,
    ) -> list[RetrievedPassage]:
        """Retrieve and rank relevant knowledge base passages for a text query.

        Args:
            query: Customer query string (e.g. ticket clean_text).
            top_k: Maximum number of passages to return.
            similarity_threshold: Minimum cosine similarity score required (1.0 - distance).
                Defaults to None, resolving to self.similarity_threshold or
                Settings.retrieval_similarity_threshold.
                (Note: 0.65 is an unvalidated configuration placeholder; 0.15 is the
                provisional mock-embedding development baseline. Real embedding threshold
                calibration is pending successful MiniLM model evaluation). Explicit caller
                overrides are fully respected.
            intent_hint: Optional domain or intent context string (e.g. predicted intent
                category like 'onboarding' or 'authentication_failure'). Used to enrich
                sparse ticket queries without altering underlying evidence or citations.

        Returns:
            Ranked list of RetrievedPassage objects meeting the threshold, or [] if none meet it.
        """
        # Resolve threshold from explicit argument, configured instance attribute, or settings
        effective_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else (
                self.similarity_threshold
                if self.similarity_threshold is not None
                else get_settings().retrieval_similarity_threshold
            )
        )

        # Rule: Graceful handling of empty or whitespace-only queries
        cleaned_query = query.strip() if query else ""
        if not cleaned_query:
            return []

        if top_k <= 0:
            return []

        # Intent-guided query enrichment for sparse/short ticket queries
        effective_query = cleaned_query
        if intent_hint:
            norm_hint = intent_hint.strip().lower().replace("_", " ")
            if norm_hint and norm_hint not in ("unclear request", "unknown", "none"):
                effective_query = f"{norm_hint}: {cleaned_query}"

        # Check if collection has documents
        try:
            total_docs = self.collection.count()
            if total_docs == 0:
                logger.warning("Knowledge base collection is empty. Retrieval returned 0 passages.")
                return []
        except Exception as e:
            logger.error(f"Error checking collection count during retrieval: {e}")
            return []

        n_results = min(top_k, total_docs)

        # Query ChromaDB collection
        try:
            results: Any = self.collection.query(
                query_texts=[effective_query],
                n_results=n_results,
            )
        except Exception as e:
            # Rule: Degrade gracefully on retrieval errors rather than crashing
            short_q = effective_query[:50]
            logger.error(f"Vector search failed unexpectedly for query '{short_q}...': {e}")
            return []

        raw_ids = results.get("ids", [[]])[0]
        raw_distances = results.get("distances", [[]])[0]
        raw_metadatas = results.get("metadatas", [[]])[0]
        raw_documents = results.get("documents", [[]])[0]

        passages: list[RetrievedPassage] = []

        for idx, chunk_id in enumerate(raw_ids):
            distance = float(raw_distances[idx])
            # Conversion: similarity = 1.0 - cosine_distance
            similarity = 1.0 - distance

            # Rule: Threshold filtering - discard passages below effective similarity threshold
            if similarity < effective_threshold:
                continue

            metadata = raw_metadatas[idx] or {}
            doc_text = raw_documents[idx] or ""

            doc_id = str(metadata.get("doc_id", ""))
            category = str(metadata.get("category", ""))
            title = str(metadata.get("title", ""))
            section_name = str(metadata.get("section_name", "General"))
            applies_to = str(metadata.get("applies_to", ""))

            # Citation format: [DOC-AUTH-001: Resolution]
            citation_id = str(metadata.get("citation_id") or f"[{doc_id}: {section_name}]")

            # Raw section body (content), stripping contextual prefix if stored in doc text
            content = str(metadata.get("raw_content", ""))
            if not content:
                # If raw_content was not stored in metadata, extract from contextual document text
                content = doc_text.split("\n\n", 1)[-1] if "\n\n" in doc_text else doc_text

            passages.append(
                RetrievedPassage(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    category=category,
                    title=title,
                    section_name=section_name,
                    content=content,
                    similarity_score=round(similarity, 4),
                    rank=1,  # Assigned after sorting
                    citation_id=citation_id,
                    applies_to=applies_to,
                )
            )

        # Sort by similarity descending
        passages.sort(key=lambda p: p.similarity_score, reverse=True)

        # Assign 1-based ranks
        ranked_passages: list[RetrievedPassage] = []
        for rank, p in enumerate(passages, start=1):
            ranked_passages.append(
                RetrievedPassage(
                    chunk_id=p.chunk_id,
                    doc_id=p.doc_id,
                    category=p.category,
                    title=p.title,
                    section_name=p.section_name,
                    content=p.content,
                    similarity_score=p.similarity_score,
                    rank=rank,
                    citation_id=p.citation_id,
                    applies_to=p.applies_to,
                )
            )

        return ranked_passages
