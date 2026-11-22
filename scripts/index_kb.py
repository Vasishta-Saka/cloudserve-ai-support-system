"""Command-line script to index knowledge-base documentation into ChromaDB.

Usage:
    python scripts/index_kb.py [--strategy section_aware|fixed_size] [--dry-run] [--mock]
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

from cloudserve.config import get_settings
from cloudserve.rag.chunker import chunk_kb, load_kb_articles
from cloudserve.rag.indexer import (
    KBIndexer,
    MockEmbeddingFunction,
    SentenceTransformerEmbeddingAdapter,
)


def main() -> int:
    settings = get_settings()

    parser = argparse.ArgumentParser(
        description="Index CloudServe Knowledge Base articles into ChromaDB vector database."
    )
    parser.add_argument(
        "--kb-path",
        type=Path,
        default=settings.kb_directory / "documentation.json",
        help="Path to documentation.json",
    )
    parser.add_argument(
        "--strategy",
        choices=["section_aware", "fixed_size"],
        default="section_aware",
        help="Chunking strategy (default: section_aware)",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=settings.chroma_persist_directory,
        help="ChromaDB persistence directory",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=settings.chroma_collection_name,
        help="ChromaDB collection name",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use deterministic offline mock embedding function instead of sentence-transformers",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and chunk articles without writing to ChromaDB",
    )

    args = parser.parse_args()

    print("==================================================")
    print(" CloudServe AI Support - Knowledge Base Indexer   ")
    print("==================================================")
    print(f"KB Path:           {args.kb_path}")
    print(f"Strategy:          {args.strategy}")
    embed_mode = "Mock (Offline)" if args.mock else settings.embedding_model_name
    print(f"Persist Directory: {args.persist_dir}")
    print(f"Collection Name:   {args.collection}")
    print(f"Embedding Mode:    {embed_mode}")
    print("--------------------------------------------------")

    if not args.kb_path.exists():
        print(
            f"ERROR: Knowledge base file not found at '{args.kb_path}'.\n"
            "This repository does not include pre-packaged benchmark articles.\n"
            "Please supply an authorized knowledge-base JSON file via --kb-path <path>.\n"
            "Expected schema: "
            "[{\"doc_id\": str, \"title\": str, \"category\": str, \"content\": str, ...}]",
            file=sys.stderr,
        )
        return 1

    try:
        articles = load_kb_articles(args.kb_path)
        print(f"Successfully loaded {len(articles)} documentation articles.")
    except Exception as e:
        print(f"ERROR loading articles: {e}", file=sys.stderr)
        return 1

    # Chunk articles
    chunks = chunk_kb(articles, strategy=args.strategy)
    print(f"Generated {len(chunks)} chunks using strategy '{args.strategy}'.")

    # Category distribution
    cat_counts = Counter(c.category for c in chunks)
    print("\nChunks by Category:")
    for cat in sorted(cat_counts.keys()):
        print(f"  • {cat:<16}: {cat_counts[cat]} chunks")

    if args.dry_run:
        print("\nDry-run mode: skipping ChromaDB writes.")
        return 0

    # Index into ChromaDB
    print("\nConnecting to ChromaDB and generating embeddings...")
    embed_fn = (
        MockEmbeddingFunction(dimension=384)
        if args.mock
        else SentenceTransformerEmbeddingAdapter(model_name=settings.embedding_model_name)
    )

    try:
        indexer = KBIndexer(
            persist_directory=args.persist_dir,
            collection_name=args.collection,
            embedding_function=embed_fn,
        )
        indexed_count = indexer.index_chunks(chunks, prune_stale=True)
        total_in_coll = indexer.get_indexed_count()
        print(f"SUCCESS: Indexed {indexed_count} chunks into '{args.collection}'.")
        print(f"Total documents in collection: {total_in_coll}")
    except Exception as e:
        print(f"ERROR indexing into ChromaDB: {e}", file=sys.stderr)
        return 1

    print("==================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
