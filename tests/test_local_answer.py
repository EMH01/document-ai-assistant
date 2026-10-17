from types import SimpleNamespace

from document_ai import providers


def test_local_answer_keeps_thinking_private(monkeypatch):
    captured = {}

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            captured.update(kwargs)
            return "rendered prompt"

    class FakePipeline:
        tokenizer = FakeTokenizer()

        def __call__(self, prompt, **kwargs):
            captured["prompt"] = prompt
            return [{"generated_text": "La respuesta está en el documento [1]."}]

    monkeypatch.setattr(providers, "_local_pipeline", lambda model: FakePipeline())
    answer = providers.LocalAnswer("Qwen/Qwen3-0.6B").answer("¿Qué dice?", "[1] ejemplo")
    assert captured["enable_thinking"] is True
    assert captured["prompt"] == "rendered prompt"
    assert answer.startswith("La respuesta")


def test_incomplete_reasoning_is_not_exposed(monkeypatch):
    class FakePipeline:
        tokenizer = SimpleNamespace(
            apply_chat_template=lambda *args, **kwargs: "prompt"
        )

        def __call__(self, *args, **kwargs):
            return [{"generated_text": "<think>unfinished private reasoning"}]

    monkeypatch.setattr(providers, "_local_pipeline", lambda model: FakePipeline())
    answer = providers.LocalAnswer("Qwen/Qwen3-0.6B").answer("Pregunta", "Contexto")
    assert "<think>" not in answer
    assert "agotó su tiempo" in answer
