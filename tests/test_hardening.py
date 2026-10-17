from unittest.mock import Mock
import pytest

from document_ai.models import Chunk
from document_ai.providers import GGUFAnswer, answer_from_chunks


def test_long_pages_get_independent_context_budget():
    pages = [
        (Chunk(source="a.pdf", page=1, text="A " * 10000), 0.9),
        (Chunk(source="a.pdf", page=2, text="B " * 10000), 0.8),
    ]
    answerer = GGUFAnswer()
    answerer.answer = Mock(return_value="OK")
    answer_from_chunks(answerer, "what?", pages)
    context = answerer.answer.call_args.args[1]
    assert "[1]" in context and "[2]" in context
    assert "A " in context and "B " in context
    assert len(context) < 7600


def test_small_pages_are_not_truncated():
    pages = [(Chunk(source="a.pdf", page=1, text="Short evidence"), 0.9)]
    answerer = GGUFAnswer()
    answerer.answer = Mock(return_value="OK")
    answer_from_chunks(answerer, "what?", pages)
    assert "Short evidence" in answerer.answer.call_args.args[1]
    assert "[Extracto truncado]" not in answerer.answer.call_args.args[1]
