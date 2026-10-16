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
    """Recover nearby evidence, including a section's continuation on the next page.

    This is useful when OCR extracts table rows as separate text items.
    No LLM call is required for expansion.
    """
    ranked = expand_ranked(ranked_children, mapping)
    ordered = sorted(parents, key=lambda chunk: (chunk.source, chunk.page))
    selected: dict[int, tuple[Chunk, float]] = {}
    for parent, score in ranked:
        selected[id(parent)] = (parent, score)
        for neighbor in ordered:
            if len(selected) >= max_excerpts:
                break
            if neighbor.source == parent.source and abs(neighbor.page - parent.page) <= 1:
                selected.setdefault(id(neighbor), (neighbor, score * 0.85))
    return list(selected.values())[:max_excerpts]
