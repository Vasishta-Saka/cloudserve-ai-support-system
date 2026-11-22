"""Knowledge base indexing pipeline and ChromaDB vector store manager.

Supports acceptance criterion A4 (retrieval against supplied documentation corpus)
with cosine distance configuration, deterministic identifiers, idempotent updates,
and automatic pruning of stale chunks.
"""

import hashlib
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, cast

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from chromadb.config import Settings as ChromaSettings

from cloudserve.config import get_settings
from cloudserve.rag.chunker import chunk_kb, load_kb_articles
from cloudserve.rag.models import KBArticle, KBChunk


class MockEmbeddingFunction(EmbeddingFunction[Documents]):
    """Deterministic, 100% offline 384-dimensional embedding function for tests and CI.

    Uses feature hashing (token-level random projection) normalized to unit length.
    Ensures that identical texts have similarity 1.0, texts sharing vocabulary have
    high cosine similarity, and unrelated texts have near-zero similarity, without
    requiring any external network requests or heavyweight neural weights.
    """

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def __call__(self, input: Documents) -> Embeddings:
        embeddings: list[list[float]] = []
        for text in input:
            embeddings.append(self._embed_single(text))
        return cast(Embeddings, embeddings)

    def _embed_single(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        tokens = re.findall(r"\b[a-zA-Z0-9_\-\.]+\b", text.lower())

        if not tokens:
            vec[0] = 1.0
            return vec

        for token in tokens:
            # Deterministic hash to dimension index and sign
            h = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dimension
            sign = 1.0 if (h >> 16) & 1 else -1.0
            vec[idx] += sign

        # Normalize to unit length (L2 norm)
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 1e-12:
            return [v / norm for v in vec]
        vec[0] = 1.0
        return vec

    @staticmethod
    def name() -> str:
        return "mock_feature_hash_embedding"

    def get_config(self) -> dict[str, Any]:
        return {"dimension": self.dimension}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "MockEmbeddingFunction":
        return MockEmbeddingFunction(dimension=config.get("dimension", 384))


class SentenceTransformerEmbeddingAdapter(EmbeddingFunction[Documents]):
    """Production embedding adapter using sentence-transformers (all-MiniLM-L6-v2).

    Loads the model lazily and outputs 384-dimensional unit-normalized embeddings.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
        local_files_only: bool | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.local_files_only = local_files_only
        self._model: Any = None

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                if self.local_files_only is not None:
                    self._model = SentenceTransformer(
                        self.model_name,
                        device=self.device,
                        local_files_only=self.local_files_only,
                    )
                else:
                    # Attempt local_files_only=True first to avoid HF Hub network timeouts
                    try:
                        self._model = SentenceTransformer(
                            self.model_name,
                            device=self.device,
                            local_files_only=True,
                        )
                    except Exception:
                        self._model = SentenceTransformer(
                            self.model_name,
                            device=self.device,
                        )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load sentence-transformers model '{self.model_name}'. "
                    f"Ensure weights are cached or network access is permitted: {e}"
                ) from e
        return self._model

    def __call__(self, input: Documents) -> Embeddings:
        model = self._load_model()
        embeddings = model.encode(
            list(input),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return cast(Embeddings, embeddings.tolist())

    @staticmethod
    def name() -> str:
        return "sentence_transformer_adapter"

    def get_config(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "local_files_only": self.local_files_only,
        }

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "SentenceTransformerEmbeddingAdapter":
        return SentenceTransformerEmbeddingAdapter(
            model_name=config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
            device=config.get("device"),
            local_files_only=config.get("local_files_only"),
        )


class KBIndexer:
    """Manages indexing, updates, and persistence of KB chunks in ChromaDB."""

    def __init__(
        self,
        persist_directory: Path | str | None = None,
        collection_name: str | None = None,
        embedding_function: EmbeddingFunction[Documents] | None = None,
        in_memory: bool = False,
    ) -> None:
        settings = get_settings()
        self.persist_directory = Path(persist_directory or settings.chroma_persist_directory)
        self.collection_name = collection_name or settings.chroma_collection_name
        self.in_memory = in_memory

        # Default embedding function: SentenceTransformer in prod, configurable for tests
        if embedding_function is not None:
            self.embedding_function = embedding_function
        else:
            self.embedding_function = SentenceTransformerEmbeddingAdapter(
                model_name=settings.embedding_model_name
            )

        # Initialize ChromaDB client with telemetry strictly disabled
        if self.in_memory:
            self.client: ClientAPI = chromadb.Client(
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    is_persistent=False,
                )
            )
        else:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    is_persistent=True,
                ),
            )

        # Chroma collection with cosine distance metric
        self.collection: Collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=cast(Any, self.embedding_function),
        )

    def index_chunks(
        self,
        chunks: Sequence[KBChunk],
        prune_stale: bool = True,
        batch_size: int = 100,
    ) -> int:
        """Idempotently index chunks into the Chroma collection.

        - Overwrites existing chunks with matching chunk_ids.
        - Adds new chunks.
        - When prune_stale=True, deletes chunks that exist in the collection
          but are no longer in the input chunks list.
        """
        if not chunks:
            if prune_stale:
                existing_data = self.collection.get()
                if existing_data and existing_data["ids"]:
                    self.collection.delete(ids=existing_data["ids"])
            return 0

        target_ids = {c.chunk_id for c in chunks}

        # Handle stale chunk removal
        if prune_stale:
            existing_data = self.collection.get()
            if existing_data and existing_data["ids"]:
                existing_ids = set(existing_data["ids"])
                stale_ids = existing_ids - target_ids
                if stale_ids:
                    self.collection.delete(ids=list(stale_ids))

        # Batch upsert chunks
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            self.collection.upsert(
                ids=[c.chunk_id for c in batch],
                documents=[c.contextual_text for c in batch],
                metadatas=[c.to_chroma_metadata() for c in batch],
            )

        return len(chunks)

    def index_kb(
        self,
        articles: Sequence[KBArticle] | None = None,
        kb_path: Path | str | None = None,
        strategy: Literal["section_aware", "fixed_size"] = "section_aware",
        prune_stale: bool = True,
    ) -> int:
        """Load, chunk, and index the entire knowledge base."""
        if articles is None:
            settings = get_settings()
            path = Path(kb_path) if kb_path else settings.kb_directory / "documentation.json"
            articles = load_kb_articles(path)

        chunks = chunk_kb(list(articles), strategy=strategy)
        return self.index_chunks(chunks, prune_stale=prune_stale)

    def get_indexed_count(self) -> int:
        """Return the current number of records stored in the collection."""
        return self.collection.count()
