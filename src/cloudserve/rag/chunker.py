"""Markdown parsing and chunking strategies for Knowledge Base articles.

Implements two chunking strategies for comparative evaluation:
1. Section-aware contextual chunking (proposed): preserves logical sections
   (Symptoms, Common causes, Resolution, Notes) and adds rich contextual prefixes.
2. Fixed-size chunking (baseline): splits text into 800-character windows
   with 120-character overlap.
"""

import json
import re
from pathlib import Path
from typing import Any, Literal

from cloudserve.rag.models import KBArticle, KBChunk, KBSection


def parse_article_sections(content: str) -> list[KBSection]:
    """Parse markdown content into structured sections based on '## ' headers.

    Extracts all level-2 markdown headings ('## Heading') and their corresponding
    body text until the next heading or end-of-file. Empty or whitespace-only sections
    are discarded.
    """
    # Pattern to match '## <Heading>' at line start
    heading_pattern = re.compile(r"^##\s+(.+)$", re.MULTILINE)
    matches = list(heading_pattern.finditer(content))

    if not matches:
        # Fallback: single section if no H2 headings exist
        body = content.strip()
        if not body:
            return []
        return [
            KBSection(
                section_name="General",
                section_slug="general",
                content=body,
                word_count=len(body.split()),
            )
        ]

    sections: list[KBSection] = []
    for i, match in enumerate(matches):
        section_name = match.group(1).strip()
        start_idx = match.end()
        end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(content)

        section_body = content[start_idx:end_idx].strip()
        if not section_body:
            continue

        slug = re.sub(r"[^a-z0-9]+", "_", section_name.lower()).strip("_")
        sections.append(
            KBSection(
                section_name=section_name,
                section_slug=slug or "section",
                content=section_body,
                word_count=len(section_body.split()),
            )
        )

    return sections


def build_contextual_prefix(article: KBArticle, section_name: str) -> str:
    """Build a rich contextual prefix for dense embeddings.

    Prefix format:
    [Document: DOC-XXX - Title | Category: cat | Section: name | Applies to: target]
    """
    parts = [f"Document: {article.doc_id} - {article.title}", f"Category: {article.category}"]
    if section_name:
        parts.append(f"Section: {section_name}")
    if article.applies_to:
        parts.append(f"Applies to: {article.applies_to}")

    return f"[{' | '.join(parts)}]"


def chunk_article_section_aware(article: KBArticle) -> list[KBChunk]:
    """Split an article into section-aware contextual chunks.

    Treats the article title and 'Applies to' information as metadata/context
    prepended to each content chunk rather than standalone chunks.
    Content chunks correspond to:
    - Symptoms
    - Common causes
    - Resolution
    - Notes
    or any other sections present in the article markdown.
    """
    sections = parse_article_sections(article.content)
    chunks: list[KBChunk] = []

    for section in sections:
        prefix = build_contextual_prefix(article, section.section_name)
        contextual_text = f"{prefix}\n\n{section.content}".strip()
        chunk_id = f"{article.doc_id}#{section.section_slug}"

        chunks.append(
            KBChunk(
                chunk_id=chunk_id,
                doc_id=article.doc_id,
                category=article.category,
                title=article.title,
                section_name=section.section_name,
                section_slug=section.section_slug,
                content=section.content,
                contextual_text=contextual_text,
                applies_to=article.applies_to,
                related_docs=list(article.related_docs),
                strategy="section_aware",
            )
        )

    return chunks


def chunk_article_fixed_size(
    article: KBArticle,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[KBChunk]:
    """Split an article using fixed-size sliding character windows.

    Baseline chunking strategy using character windows with overlap.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
        )

    text = article.content.strip()
    if not text:
        return []

    chunks: list[KBChunk] = []
    step = chunk_size - chunk_overlap
    start = 0
    chunk_idx = 1

    while start < len(text):
        end = min(start + chunk_size, len(text))
        window = text[start:end].strip()

        if window:
            chunk_id = f"{article.doc_id}#chunk_{chunk_idx}"
            prefix = f"[Document: {article.doc_id} - {article.title} | Part {chunk_idx}]"
            contextual_text = f"{prefix}\n\n{window}".strip()

            chunks.append(
                KBChunk(
                    chunk_id=chunk_id,
                    doc_id=article.doc_id,
                    category=article.category,
                    title=article.title,
                    section_name=f"Part {chunk_idx}",
                    section_slug=f"part_{chunk_idx}",
                    content=window,
                    contextual_text=contextual_text,
                    applies_to=article.applies_to,
                    related_docs=list(article.related_docs),
                    strategy="fixed_size",
                )
            )
            chunk_idx += 1

        if end >= len(text):
            break
        start += step

    return chunks


def load_kb_articles(file_path: Path | str) -> list[KBArticle]:
    """Load and parse documentation articles from documentation.json.

    Validates schema and returns strongly-typed KBArticle instances.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Knowledge base file not found at {path}")

    with path.open("r", encoding="utf-8") as f:
        data: Any = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list of articles at {path}, got {type(data).__name__}")

    articles: list[KBArticle] = []
    for item in data:
        articles.append(
            KBArticle(
                doc_id=str(item["doc_id"]).strip(),
                title=str(item["title"]).strip(),
                category=str(item["category"]).strip(),
                applies_to=str(item.get("applies_to", "")).strip(),
                content=str(item["content"]).strip(),
                related_docs=list(item.get("related_docs", [])),
                last_reviewed_days_ago=int(item.get("last_reviewed_days_ago", 0)),
            )
        )

    return articles


def chunk_kb(
    articles: list[KBArticle],
    strategy: Literal["section_aware", "fixed_size"] = "section_aware",
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[KBChunk]:
    """Chunk all knowledge base articles using the requested strategy."""
    all_chunks: list[KBChunk] = []
    for article in articles:
        if strategy == "section_aware":
            chunks = chunk_article_section_aware(article)
        elif strategy == "fixed_size":
            chunks = chunk_article_fixed_size(
                article,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
        all_chunks.extend(chunks)

    return all_chunks
