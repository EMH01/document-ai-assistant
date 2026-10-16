import numpy as np

from document_ai.models import Chunk
from document_ai.providers import answer_from_chunks, index_chunks
from document_ai.retrieval import rank_chunks


class FakeEmbeddings:
    def encode(self, texts):
        return np.array([[1.0, 0.0] if "cat" in s else [0.0, 1.0] for s in texts])


class FakeAnswer:
    def answer(self, question, context):
        assert "paper.pdf · page 1" in context
        assert "[1]" in context
        return "Grounded answer [1]"


def test_open_provider_contract_without_network():
    chunks = [Chunk("cat fact", 1, source="paper.pdf"), Chunk("dog fact", 2)]
    index_chunks(FakeEmbeddings(), chunks)
    ranked = rank_chunks(chunks, FakeEmbeddings().encode(["cat"])[0], top_k=1)
    assert ranked[0][0].text == "cat fact"
    assert answer_from_chunks(FakeAnswer(), "Question?", ranked) == "Grounded answer [1]"


def test_empty_context_never_calls_model():
    assert "No relevant" in answer_from_chunks(FakeAnswer(), "Question?", [])
