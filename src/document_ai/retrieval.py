from collections.abc import Sequence
from typing import Any

import numpy as np

from .models import Chunk


def chunk_text(
    text: str,
    page: int,
    source: str,
    chunk_size: int = 1400,
    overlap: int = 200,
) -> list[Chunk]:
    """Split page text into compact overlapping chunks while preserving source metadata."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[Chunk] = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        if end < len(cleaned):
            boundary = cleaned.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        chunks.append(Chunk(text=cleaned[start:end].strip(), page=page, source=source))
        if end == len(cleaned):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def embed_chunks(client: Any, chunks: Sequence[Chunk], model: str) -> None:
    if not chunks:
        return
    response = client.embeddings.create(model=model, input=[chunk.text for chunk in chunks])
    vectors = _normalise(
        np.asarray([item.embedding for item in response.data], dtype=np.float32)
    )
    for chunk, vector in zip(chunks, vectors, strict=True):
        chunk.embedding = vector


def rank_chunks(
    chunks: Sequence[Chunk],
    query_vector: np.ndarray,
    top_k: int = 5,
) -> list[tuple[Chunk, float]]:
    indexed = [chunk for chunk in chunks if chunk.embedding is not None]
    if not indexed:
        return []
    matrix = np.vstack([chunk.embedding for chunk in indexed])
    query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
    query = _normalise(query)[0]
    scores = matrix @ query
    order = np.argsort(scores)[::-1][:top_k]
    return [(indexed[index], float(scores[index])) for index in order]


def retrieve(
    client: Any,
    chunks: Sequence[Chunk],
    query: str,
    model: str,
    top_k: int = 5,
) -> list[tuple[Chunk, float]]:
    response = client.embeddings.create(model=model, input=[query])
    query_vector = np.asarray(response.data[0].embedding, dtype=np.float32)
    return rank_chunks(chunks, query_vector, top_k=top_k)
