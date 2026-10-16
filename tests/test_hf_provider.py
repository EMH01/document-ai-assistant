from types import SimpleNamespace
from unittest.mock import patch

from document_ai.providers import HuggingFaceAnswer, FallbackAnswer


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


def test_fallback_to_local_and_do_not_retry_remote():
    class FailedRemote:
        attempts = 0

        def answer(self, question, context):
            self.attempts += 1
            raise RuntimeError("401 Unauthorized")

    class GoodLocal:
        def answer(self, question, context):
            return "Respuesta local natural. [1]"

    remote = FailedRemote()
    provider = FallbackAnswer(remote, GoodLocal())
    assert provider.answer("Pregunta", "Fuente") == "Respuesta local natural. [1]"
    assert provider.answer("Otra", "Fuente") == "Respuesta local natural. [1]"
    assert remote.attempts == 1
    assert provider.fell_back


def test_no_token_uses_local_directly():
    class GoodLocal:
        def answer(self, question, context):
            return "Respuesta local"

    provider = FallbackAnswer(None, GoodLocal())
    assert provider.answer("Pregunta", "Contexto") == "Respuesta local"
