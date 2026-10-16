"""Lightweight parent-child retrieval without generative summarization.

Every compact search record points back to the intact original extracted excerpt.
This keeps indexing inexpensive on CPU and preserves PDF page/coordinate metadata.
"""
import re

from .models import Chunk


def search_preview(text: str, limit: int = 460) -> str:
    """Extract a short, lexical/semantic search representation without an LLM call."""
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    # Preserve both likely heading/opening and the ending, where conditions often occur.
    head = int(limit * 0.75)
    tail = limit - head - 5
    head_break = cleaned.rfind(" ", 0, head)
    tail_start = cleaned.find(" ", len(cleaned) - tail)
    return cleaned[:head_break if head_break > 0 else head].strip() + " … " + cleaned[tail_start + 1:].strip()


def build_search_index(parents: list[Chunk]) -> tuple[list[Chunk], dict[int, Chunk]]:
    """Build compact children and an identity map to the complete parent excerpts."""
    children = []
    parent_by_child = {}
    for parent in parents:
        child = Chunk(
            text=search_preview(parent.text),
            page=parent.page,
            source=parent.source,
            kind=parent.kind,
            bbox=parent.bbox,
        )
        children.append(child)
        parent_by_child[id(child)] = parent
    return children, parent_by_child


def expand_ranked(ranked_children: list[tuple[Chunk, float]], mapping: dict[int, Chunk]):
    """Return original unabridged parents with their retrieval scores."""
    return [(mapping[id(child)], score) for child, score in ranked_children]


def expand_with_page_neighbors(
    ranked_children: list[tuple[Chunk, float]],
    mapping: dict[int, Chunk],
    parents: list[Chunk],
    max_excerpts: int = 14,
) -> list[tuple[Chunk, float]]:
    """Return bounded, page-complete evidence rather than arbitrary OCR snippets.

    First identify the best matching page(s). Include all items from each such
    page in original extraction order, plus continuation pages when supported.
    A heading alone is not enough context to answer a question about a table.
    """
    ranked = expand_ranked(ranked_children, mapping)
    if not ranked:
        return []
    ordered_pages = []
    for chunk, score in ranked:
        key = (chunk.source, chunk.page)
        if key not in ordered_pages:
            ordered_pages.append(key)
    focus_source, focus_page = ordered_pages[0]
    # Preserve both halves of a section whose heading falls at the page bottom.
    relevant_pages = {(focus_source, focus_page), (focus_source, focus_page + 1)}
    # Also include different source pages if ranked and room remains.
    result = []
    for parent in parents:
        if (parent.source, parent.page) in relevant_pages:
            result.append((parent, next(
                (score for chunk, score in ranked if chunk is parent), 0.0
            )))
    # Respect max_excerpts for isolated large documents, but do not arbitrarily
    # truncate page text without marking that evidence may be incomplete.
    return result[:max_excerpts]


def page_context_blocks(parents: list[Chunk], source: str, pages: set[int],
                        max_chars: int = 15000) -> list[Chunk]:
    """Aggregate Docling item text in reading order into complete page excerpts."""
    groups: dict[int, list[str]] = {}
    for item in parents:
        if item.source == source and item.page in pages:
            groups.setdefault(item.page, []).append(item.text)
    output = []
    for page in sorted(groups):
        content = "\n".join(groups[page])
        if content:
            output.append(Chunk(text=content[:max_chars], page=page, source=source))
    return output
