"""Evaluation harness for Knowledge Base retrieval performance and chunking strategies.

Measures Precision, Recall, Recall@4, Hit Rate@4, and Mean Reciprocal Rank (MRR) against
the development benchmark dataset. Used strictly in evaluation and offline benchmarking,
never in runtime support routing.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cloudserve.config import get_settings
from cloudserve.ingestion.parsers import parse_evaluation_dataset
from cloudserve.rag.chunker import chunk_kb, load_kb_articles
from cloudserve.rag.indexer import KBIndexer, MockEmbeddingFunction
from cloudserve.rag.models import RetrievedPassage
from cloudserve.rag.retriever import KBRetriever


@dataclass(frozen=True)
class RetrievalMetrics:
    """Summary metrics for knowledge base retrieval evaluation."""

    total_tickets: int
    mean_precision: float
    precision_at_4: float
    mean_recall: float
    recall_at_4: float
    hit_rate_at_4: float
    mrr: float
    empty_rate: float
    threshold: float
    strategy: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_tickets": self.total_tickets,
            "mean_precision": round(self.mean_precision, 4),
            "precision_at_4": round(self.precision_at_4, 4),
            "mean_recall": round(self.mean_recall, 4),
            "recall_at_4": round(self.recall_at_4, 4),
            "hit_rate_at_4": round(self.hit_rate_at_4, 4),
            "mrr": round(self.mrr, 4),
            "empty_rate": round(self.empty_rate, 4),
            "threshold": self.threshold,
            "strategy": self.strategy,
        }


def compute_ticket_retrieval_metrics(
    passages: Sequence[RetrievedPassage],
    expected_doc_ids: Sequence[str] | set[str],
    top_k: int = 4,
) -> dict[str, float]:
    """Calculate retrieval metrics for an individual ticket against expected document IDs.

    Args:
        passages: Ranked list of retrieved passages.
        expected_doc_ids: Set or sequence of expected relevant document IDs.
        top_k: Cutoff rank for Recall@k and Hit Rate@k (default 4).

    Returns:
        Dictionary with precision, recall, recall_at_4, hit_rate_at_4, reciprocal_rank.
    """
    expected_set = set(expected_doc_ids)
    if not expected_set or not passages:
        return {
            "precision": 0.0,
            "precision_at_4": 0.0,
            "recall": 0.0,
            "recall_at_4": 0.0,
            "hit_rate_at_4": 0.0,
            "reciprocal_rank": 0.0,
        }

    # Extract unique retrieved doc IDs in order of first appearance across all returned passages
    retrieved_docs: list[str] = []
    for p in passages:
        if p.doc_id not in retrieved_docs:
            retrieved_docs.append(p.doc_id)

    retrieved_set = set(retrieved_docs)
    correct_all = retrieved_set & expected_set

    # Precision: share of unique retrieved docs that are correct
    precision = len(correct_all) / len(retrieved_docs) if retrieved_docs else 0.0

    # Recall: share of expected docs retrieved across all returned passages
    recall = len(correct_all) / len(expected_set)

    # Unique document IDs retrieved within top-k passages (ensuring multiple chunks
    # from the same document are counted once)
    top_k_passages = passages[:top_k]
    top_k_docs: list[str] = []
    for p in top_k_passages:
        if p.doc_id not in top_k_docs:
            top_k_docs.append(p.doc_id)

    top_k_doc_set = set(top_k_docs)
    correct_top_k = top_k_doc_set & expected_set

    # Precision@4: relevant unique documents in top-k divided by fixed denominator k (default 4).
    # Missing positions count as non-relevant per Task 7.3A requirements.
    precision_at_4 = len(correct_top_k) / float(top_k)

    # Recall@4: unique relevant documents retrieved in top four / total expected documents
    recall_at_4 = len(correct_top_k) / len(expected_set)

    # Hit Rate@4: binary indicator if at least one expected doc appears in top four
    hit_rate_at_4 = 1.0 if correct_top_k else 0.0

    # Reciprocal Rank (RR): 1 / rank of first correct document
    rr = 0.0
    for rank, doc_id in enumerate(retrieved_docs, start=1):
        if doc_id in expected_set:
            rr = 1.0 / rank
            break

    return {
        "precision": precision,
        "precision_at_4": precision_at_4,
        "recall": recall,
        "recall_at_4": recall_at_4,
        "hit_rate_at_4": hit_rate_at_4,
        "reciprocal_rank": rr,
    }


def evaluate_retrieval(
    retriever: KBRetriever,
    dataset_path: Path | str | None = None,
    top_k: int = 4,
    similarity_threshold: float = 0.65,
    strategy_name: str = "section_aware",
) -> RetrievalMetrics:
    """Evaluate retriever against tickets marked answerable from documentation.

    Uses expected_doc_ids solely for evaluation scoring. The retriever receives
    only clean_text and has zero knowledge of ground truth labels.
    """
    settings = get_settings()
    path = Path(dataset_path or settings.dev_data_directory / "development_tickets.json")
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at {path}")

    with path.open("r", encoding="utf-8") as f:
        records_raw = f.read()
    records = parse_evaluation_dataset(records_raw)

    # Filter strictly to documentation-answerable tickets with expected doc IDs
    eval_records = [
        r
        for r in records
        if r.labels is not None and r.labels.answerable_from_docs and r.labels.expected_doc_ids
    ]

    if not eval_records:
        return RetrievalMetrics(
            total_tickets=0,
            mean_precision=0.0,
            precision_at_4=0.0,
            mean_recall=0.0,
            recall_at_4=0.0,
            hit_rate_at_4=0.0,
            mrr=0.0,
            empty_rate=0.0,
            threshold=similarity_threshold,
            strategy=strategy_name,
        )

    precisions: list[float] = []
    precisions_at_4: list[float] = []
    recalls: list[float] = []
    recalls_at_4: list[float] = []
    hit_rates_at_4: list[float] = []
    reciprocal_ranks: list[float] = []
    empty_count = 0

    for record in eval_records:
        if record.labels is None or not record.labels.expected_doc_ids:
            continue
        query = record.ticket.clean_text
        expected_docs = set(record.labels.expected_doc_ids)

        passages = retriever.retrieve(
            query=query,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )

        if not passages:
            empty_count += 1
            precisions.append(0.0)
            precisions_at_4.append(0.0)
            recalls.append(0.0)
            recalls_at_4.append(0.0)
            hit_rates_at_4.append(0.0)
            reciprocal_ranks.append(0.0)
            continue

        metrics = compute_ticket_retrieval_metrics(
            passages=passages,
            expected_doc_ids=expected_docs,
            top_k=top_k,
        )
        precisions.append(metrics["precision"])
        precisions_at_4.append(metrics["precision_at_4"])
        recalls.append(metrics["recall"])
        recalls_at_4.append(metrics["recall_at_4"])
        hit_rates_at_4.append(metrics["hit_rate_at_4"])
        reciprocal_ranks.append(metrics["reciprocal_rank"])

    n = len(eval_records)
    return RetrievalMetrics(
        total_tickets=n,
        mean_precision=sum(precisions) / n,
        precision_at_4=sum(precisions_at_4) / n,
        mean_recall=sum(recalls) / n,
        recall_at_4=sum(recalls_at_4) / n,
        hit_rate_at_4=sum(hit_rates_at_4) / n,
        mrr=sum(reciprocal_ranks) / n,
        empty_rate=empty_count / n,
        threshold=similarity_threshold,
        strategy=strategy_name,
    )


def compare_chunking_strategies(
    kb_path: Path | str | None = None,
    dev_path: Path | str | None = None,
    top_k: int = 4,
    similarity_threshold: float = 0.15,
    embedding_function: Any = None,
) -> dict[str, RetrievalMetrics]:
    """Compare section-aware chunking against fixed-size chunking on development data.

    Note:
        similarity_threshold defaults to 0.15 as a provisional mock-embedding development
        baseline. 0.65 remains the unvalidated configuration placeholder for dense neural
        embeddings.
    """
    settings = get_settings()
    kb_file = Path(kb_path or settings.kb_directory / "documentation.json")
    articles = load_kb_articles(kb_file)

    embedder = embedding_function or MockEmbeddingFunction()
    results: dict[str, RetrievalMetrics] = {}

    for strategy in ["section_aware", "fixed_size"]:
        indexer = KBIndexer(
            in_memory=True,
            collection_name=f"eval_{strategy}",
            embedding_function=embedder,
        )
        chunks = chunk_kb(articles, strategy=strategy)  # type: ignore[arg-type]
        indexer.index_chunks(chunks)

        retriever = KBRetriever(collection=indexer.collection)
        metrics = evaluate_retrieval(
            retriever=retriever,
            dataset_path=dev_path,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            strategy_name=strategy,
        )
        results[strategy] = metrics

    return results


def evaluate_batch_retrieval(
    results: Sequence[Any],
    ground_truth: dict[str, Any],
    top_k: int = 4,
) -> dict[str, Any]:
    """Calculate retrieval metrics across processed batch results.

    Separates documentation-answerable tickets from unanswerable tickets.
    For unanswerable tickets, reports empty vs non-empty retrieval rates neutrally
    without asserting false-positive claims without passage-level relevance labels.

    Args:
        results: Sequence of ProcessedTicketResult objects from runner.
        ground_truth: Lookup mapping ticket_id to EvaluationRecord.
        top_k: Cutoff rank for top-k metrics (default 4).

    Returns:
        Structured dictionary matching summary_metrics.json schema.
    """
    answerable_precisions: list[float] = []
    answerable_precisions_at_4: list[float] = []
    answerable_recalls: list[float] = []
    answerable_recalls_at_4: list[float] = []
    answerable_hit_rates_at_4: list[float] = []
    answerable_rr: list[float] = []

    unanswerable_empty_count = 0
    unanswerable_non_empty_count = 0

    total_answerable = 0
    total_unanswerable = 0

    for res in results:
        ticket_id = getattr(res.ticket, "ticket_id", None) if hasattr(res, "ticket") else None
        if not ticket_id or ticket_id not in ground_truth:
            continue

        gt_record = ground_truth[ticket_id]
        labels = getattr(gt_record, "labels", None)
        passages = getattr(res, "retrieved_passages", []) or []

        is_answerable = bool(
            labels is not None
            and getattr(labels, "answerable_from_docs", True)
            and getattr(labels, "expected_doc_ids", None)
        )

        if is_answerable and labels is not None and getattr(labels, "expected_doc_ids", None):
            total_answerable += 1
            expected_docs = set(labels.expected_doc_ids)
            m = compute_ticket_retrieval_metrics(
                passages=passages,
                expected_doc_ids=expected_docs,
                top_k=top_k,
            )
            answerable_precisions.append(m["precision"])
            answerable_precisions_at_4.append(m["precision_at_4"])
            answerable_recalls.append(m["recall"])
            answerable_recalls_at_4.append(m["recall_at_4"])
            answerable_hit_rates_at_4.append(m["hit_rate_at_4"])
            answerable_rr.append(m["reciprocal_rank"])
        else:
            total_unanswerable += 1
            if len(passages) == 0:
                unanswerable_empty_count += 1
            else:
                unanswerable_non_empty_count += 1

    ans_n = max(total_answerable, 1)
    unans_n = max(total_unanswerable, 1)

    return {
        "answerable_tickets": {
            "total": total_answerable,
            "mean_precision": (
                round(sum(answerable_precisions) / ans_n, 4) if total_answerable > 0 else 0.0
            ),
            "precision_at_4": (
                round(sum(answerable_precisions_at_4) / ans_n, 4) if total_answerable > 0 else 0.0
            ),
            "mean_recall": (
                round(sum(answerable_recalls) / ans_n, 4) if total_answerable > 0 else 0.0
            ),
            "recall_at_4": (
                round(sum(answerable_recalls_at_4) / ans_n, 4) if total_answerable > 0 else 0.0
            ),
            "hit_rate_at_4": (
                round(sum(answerable_hit_rates_at_4) / ans_n, 4) if total_answerable > 0 else 0.0
            ),
            "mrr": (round(sum(answerable_rr) / ans_n, 4) if total_answerable > 0 else 0.0),
        },
        "unanswerable_tickets": {
            "total": total_unanswerable,
            "empty_retrieval_rate": (
                round(unanswerable_empty_count / unans_n, 4) if total_unanswerable > 0 else 0.0
            ),
            "non_empty_retrieval_rate": (
                round(unanswerable_non_empty_count / unans_n, 4) if total_unanswerable > 0 else 0.0
            ),
            "empty_count": unanswerable_empty_count,
            "non_empty_count": unanswerable_non_empty_count,
        },
    }
