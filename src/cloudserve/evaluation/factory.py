"""Evaluation component factory for assembling deterministic or live pipelines.

Provides EvaluationComponentFactory to build TicketProcessor instances with
explicit mock or live component injection. Never silently falls back to mocks in live mode.
"""

import logging
import subprocess
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from cloudserve.classification.classifier import TicketClassifier
from cloudserve.config import Settings, get_settings
from cloudserve.generation.llm_client import (
    BaseLLMClient,
    ConfigurationError,
    LangChainGroqClient,
    MockLLMClient,
)
from cloudserve.generation.response_generator import ResponseGenerator
from cloudserve.guardrails.safety import InputSafetyScanner
from cloudserve.rag.indexer import (
    KBIndexer,
    MockEmbeddingFunction,
    SentenceTransformerEmbeddingAdapter,
)
from cloudserve.rag.models import KBArticle
from cloudserve.rag.retriever import KBRetriever
from cloudserve.routing.router import DeterministicRouter
from cloudserve.service import TicketProcessor
from cloudserve.storage.repository import DecisionAuditRepository

logger = logging.getLogger(__name__)


def _get_git_commit() -> str:
    """Safely obtain current short git commit hash, or return fallback identifier."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    return "uncommitted"


class EvaluationComponentFactory:
    """Factory for building isolated TicketProcessor instances for evaluation runs.

    Explicitly separates mock and live execution modes:
    - Mock mode: 100% deterministic, offline MockLLMClient, MockEmbeddingFunction,
      in-memory KB index, provisional 0.15 similarity threshold, zero network dependency.
    - Live mode: LangChainGroqClient, SentenceTransformerEmbeddingAdapter, configured
      production threshold (0.65). Fails fast if credentials or model weights are missing.
    """

    MOCK_SIMILARITY_THRESHOLD: float = 0.15

    @classmethod
    def build_processor(
        cls,
        mode: Literal["mock", "live"] | str = "mock",
        settings: Settings | None = None,
        audit_repository: DecisionAuditRepository | None = None,
        processing_run_id: str | None = None,
        kb_path: Path | str | None = None,
        articles: Sequence[KBArticle] | None = None,
        similarity_threshold_override: float | None = None,
        mock_llm_responses: list[str] | str | None = None,
        llm_client_override: BaseLLMClient | None = None,
    ) -> tuple[TicketProcessor, dict[str, Any]]:
        """Construct a TicketProcessor and associated configuration metadata.

        Args:
            mode: "mock" for isolated deterministic execution; "live" for production models.
            settings: Optional application settings override.
            audit_repository: Optional audit repository override. If None, instantiates
                from settings.
            processing_run_id: Optional processing run UUID.
            kb_path: Path to knowledge-base documentation.json for indexing.
            articles: Pre-loaded KB articles for indexing (overrides kb_path).
            similarity_threshold_override: Explicit override for retriever similarity threshold.
            mock_llm_responses: Canned responses for mock LLM client.
            llm_client_override: Injected LLM client (e.g. for testing live mode with fakes).

        Returns:
            tuple[TicketProcessor, dict[str, Any]]: Wired processor and configuration metadata.

        Raises:
            ValueError: If mode is not "mock" or "live".
            ConfigurationError: In live mode if Groq API key is missing or invalid.
            RuntimeError: In live mode if sentence-transformers model weights fail to load.
        """
        cfg = settings or get_settings()

        if mode == "mock":
            return cls._build_mock_processor(
                cfg=cfg,
                audit_repository=audit_repository,
                processing_run_id=processing_run_id,
                kb_path=kb_path,
                articles=articles,
                similarity_threshold_override=similarity_threshold_override,
                mock_llm_responses=mock_llm_responses,
                llm_client_override=llm_client_override,
            )
        elif mode == "live":
            return cls._build_live_processor(
                cfg=cfg,
                audit_repository=audit_repository,
                processing_run_id=processing_run_id,
                kb_path=kb_path,
                articles=articles,
                similarity_threshold_override=similarity_threshold_override,
                llm_client_override=llm_client_override,
            )
        else:
            raise ValueError(f"Unknown evaluation mode '{mode}'. Must be 'mock' or 'live'.")

    @classmethod
    def _build_mock_processor(
        cls,
        cfg: Settings,
        audit_repository: DecisionAuditRepository | None,
        processing_run_id: str | None,
        kb_path: Path | str | None,
        articles: Sequence[KBArticle] | None,
        similarity_threshold_override: float | None,
        mock_llm_responses: list[str] | str | None,
        llm_client_override: BaseLLMClient | None,
    ) -> tuple[TicketProcessor, dict[str, Any]]:
        """Assemble mock-mode evaluation components."""
        # Active similarity threshold: provisional 0.15 baseline unless overridden
        active_threshold = (
            similarity_threshold_override
            if similarity_threshold_override is not None
            else cls.MOCK_SIMILARITY_THRESHOLD
        )

        # Injected LLM Client
        if llm_client_override is not None:
            llm_client: BaseLLMClient = llm_client_override
        else:
            llm_client = MockLLMClient(
                responses=mock_llm_responses or "",
                model_name="mock-llm-v1",
            )

        # Offline deterministic embedding function and in-memory indexer
        embedding_fn = MockEmbeddingFunction(dimension=384)
        indexer = KBIndexer(embedding_function=embedding_fn, in_memory=True)

        # Index knowledge base if articles or path exist
        if articles is not None:
            indexer.index_kb(articles=articles)
        else:
            doc_path = Path(kb_path) if kb_path else cfg.kb_directory / "documentation.json"
            if doc_path.exists():
                indexer.index_kb(kb_path=doc_path)
            else:
                logger.info(
                    "Mock evaluation KB documentation not found at '%s'. "
                    "In-memory indexer initialized empty.",
                    doc_path,
                )

        # In-memory retriever with explicit similarity threshold
        retriever = KBRetriever(
            indexer=indexer,
            in_memory=True,
            similarity_threshold=active_threshold,
        )

        # Verify that the retriever actually retains the configured threshold
        if retriever.similarity_threshold != active_threshold:
            raise RuntimeError(
                f"Retriever threshold mismatch: expected {active_threshold}, "
                f"got {retriever.similarity_threshold}"
            )

        # Injected classifier and generator
        classifier = TicketClassifier(
            llm_client=llm_client,
            prompt_version=cfg.classification_prompt_version,
            settings=cfg,
        )
        generator = ResponseGenerator(
            llm_client=llm_client,
            prompt_version="v1",
            settings=cfg,
        )
        router = DeterministicRouter(settings=cfg)
        safety_scanner = InputSafetyScanner(
            enable_injection_shield=cfg.enable_prompt_injection_shield
        )

        # Repository resolution
        repo = (
            audit_repository
            if audit_repository is not None
            else DecisionAuditRepository.from_settings(cfg)
        )

        processor = TicketProcessor(
            classifier=classifier,
            retriever=retriever,
            generator=generator,
            router=router,
            safety_scanner=safety_scanner,
            audit_repository=repo,
            processing_run_id=processing_run_id,
            settings=cfg,
        )

        metadata: dict[str, Any] = {
            "execution_mode": "mock",
            "mode": "mock",
            "classifier_class": classifier.__class__.__name__,
            "classifier_model": llm_client.model_name,
            "classifier_llm_client_class": (
                f"{classifier.__class__.__name__} ({llm_client.model_name})"
            ),
            "classifier_prompt_version": classifier.prompt_version,
            "generator_class": generator.__class__.__name__,
            "generator_model": llm_client.model_name,
            "generator_llm_client_class": (
                f"{generator.__class__.__name__} ({llm_client.model_name})"
            ),
            "generator_prompt_version": generator.prompt_version,
            "embedding_class": embedding_fn.__class__.__name__,
            "embedding_model": embedding_fn.name(),
            "embedding_function_class": (
                f"{embedding_fn.__class__.__name__} ({embedding_fn.name()})"
            ),
            "prompt_version": classifier.prompt_version,
            "retriever_class": retriever.__class__.__name__,
            "retriever_similarity_threshold": retriever.similarity_threshold,
            "retrieval_similarity_threshold": retriever.similarity_threshold,
            "retrieval_top_k": cfg.retrieval_top_k,
            "router_class": router.__class__.__name__,
            "safety_scanner_class": safety_scanner.__class__.__name__,
            "confidence_threshold_auto_respond": router.confidence_threshold_auto_respond,
            "confidence_threshold_review": router.confidence_threshold_review,
            "audit_repository_class": repo.__class__.__name__ if repo is not None else None,
            "git_commit": _get_git_commit(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        return processor, metadata

    @classmethod
    def _build_live_processor(
        cls,
        cfg: Settings,
        audit_repository: DecisionAuditRepository | None,
        processing_run_id: str | None,
        kb_path: Path | str | None,
        articles: Sequence[KBArticle] | None,
        similarity_threshold_override: float | None,
        llm_client_override: BaseLLMClient | None,
    ) -> tuple[TicketProcessor, dict[str, Any]]:
        """Assemble live-mode evaluation components with strict fail-fast validation.

        Never silently falls back to mocks if credentials or model weights are missing.
        """
        # 1. Validate Groq credentials before processing
        if llm_client_override is None:
            resolved_key = (
                cfg.groq_api_key.get_secret_value().strip()
                if cfg.groq_api_key is not None
                else None
            )
            if not resolved_key:
                raise ConfigurationError(
                    "Missing Groq API key: Live evaluation mode requires a valid GROQ_API_KEY. "
                    "Never silently falling back to mock provider."
                )

        # 2. Verify and load actual sentence-transformers embedding model
        embedding_adapter = SentenceTransformerEmbeddingAdapter(model_name=cfg.embedding_model_name)
        try:
            embedding_adapter._load_model()
        except Exception as exc:
            raise RuntimeError(
                f"Live evaluation mode requires sentence-transformers weights for "
                f"'{cfg.embedding_model_name}'. Loading failed: {exc}. "
                f"Never silently falling back to mock embedding function."
            ) from exc

        # Active similarity threshold: configured production threshold unless overridden
        active_threshold = (
            similarity_threshold_override
            if similarity_threshold_override is not None
            else cfg.retrieval_similarity_threshold
        )

        # Injected live LLM Client
        llm_client: BaseLLMClient = (
            llm_client_override
            if llm_client_override is not None
            else LangChainGroqClient(settings=cfg)
        )

        # Persistent ChromaDB KB indexer
        indexer = KBIndexer(
            persist_directory=cfg.chroma_persist_directory,
            collection_name=cfg.chroma_collection_name,
            embedding_function=embedding_adapter,
            in_memory=False,
        )

        if articles is not None:
            indexer.index_kb(articles=articles)
        elif kb_path is not None:
            indexer.index_kb(kb_path=kb_path)

        # KBRetriever with live similarity threshold
        retriever = KBRetriever(
            indexer=indexer,
            similarity_threshold=active_threshold,
        )

        if retriever.similarity_threshold != active_threshold:
            raise RuntimeError(
                f"Retriever threshold mismatch: expected {active_threshold}, "
                f"got {retriever.similarity_threshold}"
            )

        classifier = TicketClassifier(
            llm_client=llm_client,
            prompt_version=cfg.classification_prompt_version,
            settings=cfg,
        )
        generator = ResponseGenerator(
            llm_client=llm_client,
            prompt_version="v1",
            settings=cfg,
        )
        router = DeterministicRouter(settings=cfg)
        safety_scanner = InputSafetyScanner(
            enable_injection_shield=cfg.enable_prompt_injection_shield
        )

        repo = (
            audit_repository
            if audit_repository is not None
            else DecisionAuditRepository.from_settings(cfg)
        )

        processor = TicketProcessor(
            classifier=classifier,
            retriever=retriever,
            generator=generator,
            router=router,
            safety_scanner=safety_scanner,
            audit_repository=repo,
            processing_run_id=processing_run_id,
            settings=cfg,
        )

        metadata: dict[str, Any] = {
            "execution_mode": "live",
            "mode": "live",
            "classifier_class": classifier.__class__.__name__,
            "classifier_model": llm_client.model_name,
            "classifier_llm_client_class": (
                f"{classifier.__class__.__name__} ({llm_client.model_name})"
            ),
            "classifier_prompt_version": classifier.prompt_version,
            "generator_class": generator.__class__.__name__,
            "generator_model": llm_client.model_name,
            "generator_llm_client_class": (
                f"{generator.__class__.__name__} ({llm_client.model_name})"
            ),
            "generator_prompt_version": generator.prompt_version,
            "embedding_class": embedding_adapter.__class__.__name__,
            "embedding_model": embedding_adapter.name(),
            "embedding_function_class": (
                f"{embedding_adapter.__class__.__name__} ({embedding_adapter.name()})"
            ),
            "prompt_version": classifier.prompt_version,
            "retriever_class": retriever.__class__.__name__,
            "retriever_similarity_threshold": retriever.similarity_threshold,
            "retrieval_similarity_threshold": retriever.similarity_threshold,
            "retrieval_top_k": cfg.retrieval_top_k,
            "router_class": router.__class__.__name__,
            "safety_scanner_class": safety_scanner.__class__.__name__,
            "confidence_threshold_auto_respond": router.confidence_threshold_auto_respond,
            "confidence_threshold_review": router.confidence_threshold_review,
            "audit_repository_class": repo.__class__.__name__ if repo is not None else None,
            "git_commit": _get_git_commit(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        return processor, metadata
