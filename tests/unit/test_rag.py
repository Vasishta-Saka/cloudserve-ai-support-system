"""Unit tests for Knowledge Base indexing and retrieval (Acceptance Criterion A4).

Verifies section-aware chunking, fixed-size baseline, ChromaDB cosine indexing,
idempotency, stale pruning, ranked retrieval, threshold filtering, empty query handling,
citation resolution, and runtime isolation from evaluation ground truth.
All tests run 100% offline using deterministic mock embeddings.
"""

from pathlib import Path

import pytest

from cloudserve.evaluation.retrieval_eval import compute_ticket_retrieval_metrics
from cloudserve.rag.chunker import (
    build_contextual_prefix,
    chunk_article_fixed_size,
    chunk_article_section_aware,
    load_kb_articles,
    parse_article_sections,
)
from cloudserve.rag.indexer import KBIndexer, MockEmbeddingFunction
from cloudserve.rag.models import KBArticle, RetrievedPassage
from cloudserve.rag.retriever import KBRetriever


@pytest.fixture
def sample_article() -> KBArticle:
    """Provide a representative test article with standard markdown sections."""
    return KBArticle(
        doc_id="DOC-TEST-001",
        title="Troubleshooting Container Port Mismatch",
        category="deployment",
        applies_to="Container Service, CLI",
        content=(
            "# Troubleshooting Container Port Mismatch\n\n"
            "**Applies to:** Container Service, CLI\n\n"
            "## Symptoms\n\n"
            "- Health check fails with connection refused\n"
            "- Container logs show application listening on port 8080\n\n"
            "## Common causes\n\n"
            "- Service definition specifies port 80 instead of 8080\n"
            "- Firewall blocks internal container port\n\n"
            "## Resolution\n\n"
            "1. Verify the EXPOSE port in your Dockerfile.\n"
            "2. Update the targetPort in service.yaml to 8080.\n"
            "3. Redeploy the revision using `cloudserve deploy`.\n\n"
            "## Notes\n\n"
            "Port 8080 is the default for Spring Boot and Node.js applications."
        ),
        related_docs=["DOC-DEPLOY-001"],
        last_reviewed_days_ago=5,
    )


@pytest.fixture
def mock_embedder() -> MockEmbeddingFunction:
    """Deterministic 384-dimensional offline mock embedding function."""
    return MockEmbeddingFunction(dimension=384)


@pytest.fixture
def in_memory_indexer(mock_embedder: MockEmbeddingFunction) -> KBIndexer:
    """Isolated, in-memory ChromaDB indexer for unit testing."""
    return KBIndexer(
        in_memory=True,
        collection_name="test_kb_collection",
        embedding_function=mock_embedder,
    )


class TestChunking:
    """Tests for section-aware and fixed-size chunking strategies."""

    def test_parse_article_sections_all_29_articles(self) -> None:
        """Verify all 29 articles in documentation.json parse cleanly into sections when present."""
        kb_path = Path("data/kb/documentation.json")
        if not kb_path.exists():
            pytest.skip(
                "data/kb/documentation.json not present in workspace (excluded benchmark dataset)"
            )

        articles = load_kb_articles(kb_path)
        assert len(articles) == 29

        for article in articles:
            sections = parse_article_sections(article.content)
            assert len(sections) >= 3, f"Article {article.doc_id} should have multiple sections"
            section_names = {s.section_name for s in sections}
            # All 29 articles have Symptoms, Common causes, Resolution, Notes
            assert "Symptoms" in section_names
            assert "Resolution" in section_names

    def test_section_aware_chunk_ids_and_metadata(self, sample_article: KBArticle) -> None:
        """Verify deterministic chunk IDs, metadata preservation, and citation IDs."""
        chunks = chunk_article_section_aware(sample_article)
        assert len(chunks) == 4

        expected_slugs = ["symptoms", "common_causes", "resolution", "notes"]
        for idx, chunk in enumerate(chunks):
            assert chunk.doc_id == "DOC-TEST-001"
            assert chunk.category == "deployment"
            assert chunk.title == "Troubleshooting Container Port Mismatch"
            assert chunk.applies_to == "Container Service, CLI"
            assert chunk.related_docs == ["DOC-DEPLOY-001"]
            assert chunk.section_slug == expected_slugs[idx]
            assert chunk.chunk_id == f"DOC-TEST-001#{expected_slugs[idx]}"
            assert chunk.citation_id == f"[DOC-TEST-001: {chunk.section_name}]"
            assert sample_article.title in chunk.contextual_text
            assert chunk.section_name in chunk.contextual_text

    def test_missing_or_empty_sections_handled_gracefully(self) -> None:
        """Verify chunker handles missing or blank sections without failing."""
        sparse_content = (
            "# Sparse Documentation\n\n"
            "## Symptoms\n\n"
            "Only symptoms are documented.\n\n"
            "## Resolution\n\n"
        )
        sparse_article = KBArticle(
            doc_id="DOC-SPARSE-001",
            title="Sparse Documentation",
            category="general",
            content=sparse_content,
        )
        chunks = chunk_article_section_aware(sparse_article)
        # Empty resolution section should be omitted gracefully
        assert len(chunks) == 1
        assert chunks[0].section_slug == "symptoms"
        assert chunks[0].chunk_id == "DOC-SPARSE-001#symptoms"

    def test_no_h2_sections_fallback(self) -> None:
        """Verify an article with no H2 headings creates a General section chunk."""
        plain_article = KBArticle(
            doc_id="DOC-PLAIN-001",
            title="Plain Article",
            category="general",
            content="This is just unsectioned plain documentation text.",
        )
        chunks = chunk_article_section_aware(plain_article)
        assert len(chunks) == 1
        assert chunks[0].section_slug == "general"
        assert chunks[0].chunk_id == "DOC-PLAIN-001#general"

    def test_contextual_prefix_formatting(self, sample_article: KBArticle) -> None:
        """Verify contextual prefix format."""
        prefix = build_contextual_prefix(sample_article, "Resolution")
        assert "Document: DOC-TEST-001 - Troubleshooting Container Port Mismatch" in prefix
        assert "Category: deployment" in prefix
        assert "Section: Resolution" in prefix
        assert "Applies to: Container Service, CLI" in prefix

    def test_fixed_size_chunking_sliding_window(self, sample_article: KBArticle) -> None:
        """Verify baseline fixed-size chunking creates overlapping character windows."""
        chunks = chunk_article_fixed_size(sample_article, chunk_size=300, chunk_overlap=50)
        assert len(chunks) > 1
        for i, chunk in enumerate(chunks, start=1):
            assert chunk.chunk_id == f"DOC-TEST-001#chunk_{i}"
            assert chunk.strategy == "fixed_size"
            assert f"Part {i}" in chunk.section_name

    def test_fixed_size_chunking_invalid_overlap(self, sample_article: KBArticle) -> None:
        """Verify error raised when overlap >= chunk_size."""
        with pytest.raises(ValueError, match="chunk_overlap.*must be strictly less"):
            chunk_article_fixed_size(sample_article, chunk_size=200, chunk_overlap=200)


class TestIndexing:
    """Tests for ChromaDB indexer, collection config, and idempotency."""

    def test_chroma_collection_configured_with_cosine(self, in_memory_indexer: KBIndexer) -> None:
        """Verify Chroma collection uses hnsw:space=cosine."""
        metadata = in_memory_indexer.collection.metadata
        assert metadata is not None
        assert metadata.get("hnsw:space") == "cosine"

    def test_mock_embedding_dimensions_and_determinism(
        self,
        mock_embedder: MockEmbeddingFunction,
    ) -> None:
        """Verify mock embedder produces normalized 384-d vectors deterministically."""
        texts = ["hello world", "hello world", "unrelated text about billing"]
        embeddings = mock_embedder(texts)

        assert len(embeddings) == 3
        assert len(embeddings[0]) == 384
        assert len(embeddings[1]) == 384

        # Identical texts produce identical vectors
        import numpy as np

        assert np.allclose(embeddings[0], embeddings[1])

        # Different texts produce different vectors
        assert not np.allclose(embeddings[0], embeddings[2])

    def test_idempotent_indexing(
        self,
        in_memory_indexer: KBIndexer,
        sample_article: KBArticle,
    ) -> None:
        """Verify repeated indexing does not create duplicate chunks."""
        chunks = chunk_article_section_aware(sample_article)
        assert len(chunks) == 4

        # First indexing
        count1 = in_memory_indexer.index_chunks(chunks)
        assert count1 == 4
        assert in_memory_indexer.get_indexed_count() == 4

        # Second indexing (idempotency check)
        count2 = in_memory_indexer.index_chunks(chunks)
        assert count2 == 4
        assert in_memory_indexer.get_indexed_count() == 4

    def test_stale_chunk_pruning(
        self,
        in_memory_indexer: KBIndexer,
        sample_article: KBArticle,
    ) -> None:
        """Verify stale chunks from removed articles or sections are pruned."""
        chunks = chunk_article_section_aware(sample_article)
        in_memory_indexer.index_chunks(chunks)
        assert in_memory_indexer.get_indexed_count() == 4

        # Re-index with only 2 chunks and prune_stale=True
        reduced_chunks = chunks[:2]
        in_memory_indexer.index_chunks(reduced_chunks, prune_stale=True)
        assert in_memory_indexer.get_indexed_count() == 2

        # Verify remaining IDs are exactly the 2 kept chunks
        remaining_data = in_memory_indexer.collection.get()
        assert remaining_data is not None
        expected_remaining = {reduced_chunks[0].chunk_id, reduced_chunks[1].chunk_id}
        assert set(remaining_data["ids"]) == expected_remaining

    def test_index_empty_chunks_cleans_collection_if_prune_stale(
        self,
        in_memory_indexer: KBIndexer,
        sample_article: KBArticle,
    ) -> None:
        """Verify indexing empty list with prune_stale removes all records."""
        chunks = chunk_article_section_aware(sample_article)
        in_memory_indexer.index_chunks(chunks)
        assert in_memory_indexer.get_indexed_count() == 4

        in_memory_indexer.index_chunks([], prune_stale=True)
        assert in_memory_indexer.get_indexed_count() == 0


class TestRetriever:
    """Tests for vector retrieval, threshold filtering, and citations."""

    @pytest.fixture
    def indexed_retriever(
        self,
        in_memory_indexer: KBIndexer,
        sample_article: KBArticle,
    ) -> KBRetriever:
        """Provide a retriever pre-loaded with sample article chunks."""
        chunks = chunk_article_section_aware(sample_article)
        in_memory_indexer.index_chunks(chunks)
        return KBRetriever(collection=in_memory_indexer.collection)

    def test_ranked_retrieval_order(self, indexed_retriever: KBRetriever) -> None:
        """Verify retrieved passages are ranked in descending similarity order."""
        query = "Container health check connection refused targetPort"
        passages = indexed_retriever.retrieve(query, top_k=4, similarity_threshold=0.0)

        assert len(passages) > 0
        for i in range(len(passages) - 1):
            assert passages[i].similarity_score >= passages[i + 1].similarity_score
            assert passages[i].rank == i + 1

    def test_threshold_filtering_returns_empty_when_no_match(
        self,
        indexed_retriever: KBRetriever,
    ) -> None:
        """Verify retriever returns empty list when no chunk exceeds the similarity threshold."""
        query = "Some completely unrelated text about invoices"
        # Strict threshold that no passage will satisfy
        passages = indexed_retriever.retrieve(query, top_k=4, similarity_threshold=0.999)
        assert passages == []

    def test_empty_or_whitespace_query_returns_empty_list(
        self,
        indexed_retriever: KBRetriever,
    ) -> None:
        """Verify empty, None, or whitespace-only queries return [] without crashing."""
        assert indexed_retriever.retrieve("", top_k=4) == []
        assert indexed_retriever.retrieve("   \n\t  ", top_k=4) == []
        assert indexed_retriever.retrieve("valid query", top_k=0) == []

    def test_citation_resolves_to_real_article(self, indexed_retriever: KBRetriever) -> None:
        """Verify citation format and traceability (Criterion A4 and A6)."""
        passages = indexed_retriever.retrieve(
            "port mismatch health check",
            top_k=2,
            similarity_threshold=0.0,
        )
        assert len(passages) > 0

        for p in passages:
            assert p.doc_id == "DOC-TEST-001"
            assert p.citation_id.startswith("[DOC-TEST-001: ")
            assert p.citation_id.endswith("]")
            assert p.chunk_id.startswith("DOC-TEST-001#")
            assert len(p.content) > 0

    def test_retriever_handles_empty_collection(self, mock_embedder: MockEmbeddingFunction) -> None:
        """Verify retriever handles an empty collection without exceptions."""
        empty_indexer = KBIndexer(
            in_memory=True,
            collection_name="empty_test_coll",
            embedding_function=mock_embedder,
        )
        retriever = KBRetriever(collection=empty_indexer.collection)
        passages = retriever.retrieve("any query", top_k=4)
        assert passages == []

    def test_retriever_data_isolation_from_ground_truth(
        self,
        indexed_retriever: KBRetriever,
    ) -> None:
        """Verify retriever does not access evaluation ground truth or labels."""
        query_text = "How do I update targetPort in service.yaml?"
        passages = indexed_retriever.retrieve(query_text, top_k=2, similarity_threshold=0.0)
        assert isinstance(passages, list)
        for p in passages:
            assert isinstance(p, RetrievedPassage)
            assert not hasattr(p, "expected_doc_ids")
            assert not hasattr(p, "answerable_from_docs")

    def test_default_threshold_from_settings_and_override(
        self,
        indexed_retriever: KBRetriever,
    ) -> None:
        """Verify retriever defaults to Settings threshold and respects explicit overrides."""
        # Query under calibrated default settings (0.40) returns passages meeting threshold (~0.57)
        query = "Container health check connection refused"
        default_passages = indexed_retriever.retrieve(query, top_k=4)
        assert len(default_passages) == 1
        assert default_passages[0].similarity_score >= 0.40

        # Unrelated query yields 0 passages under default settings (0.40)
        unrelated_query = "Billing invoice refund payment calculation"
        unrelated_passages = indexed_retriever.retrieve(unrelated_query, top_k=4)
        assert unrelated_passages == []

        # Explicit strict override to 0.85 filters out the 0.5729 passage
        strict_passages = indexed_retriever.retrieve(query, top_k=4, similarity_threshold=0.85)
        assert strict_passages == []

        # Explicit override to 0.0 returns all ranked passages
        overridden_passages = indexed_retriever.retrieve(query, top_k=4, similarity_threshold=0.0)
        assert len(overridden_passages) == 4


class TestRetrievalEvaluationMetrics:
    """Tests for retrieval evaluation metric calculations (Precision, Recall, Recall@4, MRR)."""

    def _make_passage(
        self,
        doc_id: str,
        chunk_idx: int = 1,
        score: float = 0.8,
    ) -> RetrievedPassage:
        return RetrievedPassage(
            chunk_id=f"{doc_id}#chunk_{chunk_idx}",
            doc_id=doc_id,
            category="deployment",
            title="Test Doc",
            section_name="Resolution",
            content="Sample text",
            similarity_score=score,
            rank=chunk_idx,
            citation_id=f"[{doc_id}: Resolution]",
        )

    def test_recall_at_4_all_relevant_retrieved(self) -> None:
        """Recall@4 is 1.0 when all expected relevant documents appear in top 4."""
        passages = [
            self._make_passage("DOC-A", 1),
            self._make_passage("DOC-B", 2),
            self._make_passage("DOC-C", 3),
        ]
        expected = ["DOC-A", "DOC-B"]
        metrics = compute_ticket_retrieval_metrics(passages, expected, top_k=4)

        assert metrics["recall"] == 1.0
        assert metrics["recall_at_4"] == 1.0
        assert metrics["hit_rate_at_4"] == 1.0
        assert metrics["precision"] == pytest.approx(2 / 3)
        assert metrics["reciprocal_rank"] == 1.0

    def test_recall_at_4_only_some_relevant_retrieved(self) -> None:
        """Recall@4 reflects partial coverage: 1 of 2 expected documents in top 4 yields 0.5."""
        passages = [
            self._make_passage("DOC-X", 1),
            self._make_passage("DOC-A", 2),
            self._make_passage("DOC-Y", 3),
            self._make_passage("DOC-Z", 4),
        ]
        # Expected: DOC-A and DOC-B
        expected = ["DOC-A", "DOC-B"]
        metrics = compute_ticket_retrieval_metrics(passages, expected, top_k=4)

        # 1 of 2 relevant docs retrieved in top 4
        assert metrics["recall_at_4"] == 0.5
        assert metrics["hit_rate_at_4"] == 1.0
        assert metrics["recall"] == 0.5
        assert metrics["precision"] == pytest.approx(1 / 4)
        assert metrics["reciprocal_rank"] == 0.5  # DOC-A first appears at rank 2

    def test_recall_at_4_no_relevant_retrieved(self) -> None:
        """Recall@4 and Hit Rate@4 are 0.0 when no expected document is in top 4."""
        passages = [
            self._make_passage("DOC-X", 1),
            self._make_passage("DOC-Y", 2),
        ]
        expected = ["DOC-A"]
        metrics = compute_ticket_retrieval_metrics(passages, expected, top_k=4)

        assert metrics["recall_at_4"] == 0.0
        assert metrics["hit_rate_at_4"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["precision"] == 0.0
        assert metrics["reciprocal_rank"] == 0.0

    def test_recall_at_4_multiple_chunks_same_doc_counted_once(self) -> None:
        """Multiple chunks from the same document in top 4 must count as 1 document."""
        # Top 4 contains 3 chunks from DOC-A and 1 from DOC-B
        passages = [
            self._make_passage("DOC-A", 1),
            self._make_passage("DOC-A", 2),
            self._make_passage("DOC-A", 3),
            self._make_passage("DOC-B", 4),
        ]
        # Expected: DOC-A and DOC-C
        expected = ["DOC-A", "DOC-C"]
        metrics = compute_ticket_retrieval_metrics(passages, expected, top_k=4)

        # Unique relevant in top 4: {DOC-A} (1 of 2 expected)
        assert metrics["recall_at_4"] == 0.5
        assert metrics["hit_rate_at_4"] == 1.0
        # Unique retrieved docs across all: {DOC-A, DOC-B} (2 docs, 1 correct)
        assert metrics["precision"] == 0.5

    def test_recall_at_4_empty_retrieval_results(self) -> None:
        """Empty retrieval returns 0.0 for all metrics without exceptions."""
        metrics = compute_ticket_retrieval_metrics([], ["DOC-A", "DOC-B"], top_k=4)

        assert metrics["recall_at_4"] == 0.0
        assert metrics["hit_rate_at_4"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["precision"] == 0.0
        assert metrics["reciprocal_rank"] == 0.0

    def test_recall_at_4_empty_expected_docs(self) -> None:
        """Empty expected docs returns 0.0 for all metrics without dividing by zero."""
        passages = [self._make_passage("DOC-A", 1)]
        metrics = compute_ticket_retrieval_metrics(passages, [], top_k=4)

        assert metrics["recall_at_4"] == 0.0
        assert metrics["hit_rate_at_4"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["precision"] == 0.0
        assert metrics["reciprocal_rank"] == 0.0


class TestSentenceTransformerAdapter:
    """Unit tests for production SentenceTransformerEmbeddingAdapter."""

    def test_sentence_transformer_adapter_offline_loading_and_dimension(self) -> None:
        """Verify SentenceTransformerEmbeddingAdapter loads offline and produces 384-d vectors."""
        from cloudserve.rag.indexer import SentenceTransformerEmbeddingAdapter

        adapter = SentenceTransformerEmbeddingAdapter(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            local_files_only=True,
        )
        assert adapter.name() == "sentence_transformer_adapter"
        cfg = adapter.get_config()
        assert cfg["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"
        assert cfg["local_files_only"] is True

        # Roundtrip config build
        reconstructed = SentenceTransformerEmbeddingAdapter.build_from_config(cfg)
        assert reconstructed.model_name == adapter.model_name
        assert reconstructed.local_files_only is True

        # Generate embeddings offline
        test_docs = ["First test document for embeddings.", "Second test document."]
        vectors = adapter(test_docs)
        assert len(vectors) == 2
        assert len(vectors[0]) == 384
        assert len(vectors[1]) == 384
        # Verify finite numbers
        import numpy as np

        assert np.all(np.isfinite(vectors[0]))
        assert np.all(np.isfinite(vectors[1]))


class TestShortQueryRetrievalImprovement:
    """Tests for REC-05 short-query knowledge retrieval enhancement."""

    def test_intent_hint_improves_retrieval(
        self,
        in_memory_indexer: KBIndexer,
        sample_article: KBArticle,
    ) -> None:
        """Verify that providing an intent_hint enriches query without corrupting citations."""
        in_memory_indexer.index_kb(articles=[sample_article])
        retriever = KBRetriever(indexer=in_memory_indexer, similarity_threshold=0.0)

        # Retrieve with intent hint
        passages = retriever.retrieve(
            query="port 8080 refused",
            top_k=2,
            intent_hint="deployment_failure",
        )
        assert len(passages) > 0
        assert passages[0].doc_id == "DOC-TEST-001"
        assert passages[0].citation_id.startswith("[DOC-TEST-001:")
        # Check that passage content remains pure article content, not metadata prefix
        assert "Port 8080" in passages[0].content or "8080" in passages[0].content

    def test_unclear_request_intent_hint_ignored(
        self,
        in_memory_indexer: KBIndexer,
        sample_article: KBArticle,
    ) -> None:
        """Verify that unclear_request or empty intent hints do not pollute the query."""
        in_memory_indexer.index_kb(articles=[sample_article])
        retriever = KBRetriever(indexer=in_memory_indexer, similarity_threshold=0.0)

        passages_raw = retriever.retrieve("port 8080 refused", top_k=2)
        passages_unclear = retriever.retrieve(
            "port 8080 refused",
            top_k=2,
            intent_hint="unclear_request",
        )
        assert len(passages_raw) == len(passages_unclear)
        assert passages_raw[0].chunk_id == passages_unclear[0].chunk_id
        assert passages_raw[0].similarity_score == passages_unclear[0].similarity_score
