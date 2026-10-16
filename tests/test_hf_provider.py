from types import SimpleNamespace
from unittest.mock import patch

from document_ai.providers import HuggingFaceAnswer


def test_hosted_answer_sends_evidence_and_returns_natural_response():
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(
                    content="En segundo año había 17 asignaturas. [1]"
                ))]
            )

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client_config"] = kwargs
            self.chat = SimpleNamespace(completions=FakeCompletions())

    with patch("huggingface_hub.InferenceClient", FakeClient):
        answerer = HuggingFaceAnswer("test-token", "test/model")
        answer = answerer.answer("¿Cuántas asignaturas?", "[1] Documento página 2")
    assert answer.startswith("En segundo año")
    assert captured["model"] == "test/model"
    assert captured["client_config"]["token"] == "test-token"
    assert "Documento página 2" in captured["messages"][1]["content"]
    assert captured["max_tokens"] <= 400
