import numpy as np
import pytest

from document_ai.models import Chunk
from document_ai.retrieval import chunk_text, rank_chunks


def test_chunk_text_preserves_page_and_source():
    chunks = chunk_text(
        "A " * 1000,
        page=3,
        source="paper.pdf",
        chunk_size=300,
        overlap=50,
    )
    assert len(chunks) > 1
    assert all(chunk.page == 3 for chunk in chunks)
    assert all(chunk.source == "paper.pdf" for chunk in chunks)
    assert all(chunk.text for chunk in chunks)


def test_chunk_text_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text("text", page=1, source="paper.pdf", chunk_size=100, overlap=100)


def test_rank_chunks_orders_by_cosine_similarity():
    a = Chunk(
        "alpha",
        page=1,
        embedding=np.array([1.0, 0.0], dtype=np.float32),
    )
    b = Chunk(
        "beta",
        page=2,
        embedding=np.array([0.0, 1.0], dtype=np.float32),
    )

    ranked = rank_chunks(
        [a, b],
        np.array([0.9, 0.1], dtype=np.float32),
        top_k=2,
    )

    assert ranked[0][0].text == "alpha"
    assert ranked[0][1] > ranked[1][1]
