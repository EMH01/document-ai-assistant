from unittest.mock import patch

from document_ai.providers import GGUFAnswer


def test_gguf_sends_document_context_and_returns_answer():
    captured = {}

    class DummyModel:
        def create_chat_completion(self, **kwargs):
            captured.update(kwargs)
            return {"choices": [{"message": {"content": "Hay 17 asignaturas. [1] [2]"}}]}

    with patch("document_ai.providers._gguf_model", return_value=DummyModel()):
        answer = GGUFAnswer().answer("¿Cuántas asignaturas?", "[1] Página 1\n[2] Página 2")
    assert answer == "Hay 17 asignaturas. [1] [2]"
    assert "Página 2" in captured["messages"][1]["content"]
    assert captured["max_tokens"] <= 256
    assert captured["temperature"] < 0.5
