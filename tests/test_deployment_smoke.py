"""Fast deployment smoke tests without downloading public model weights."""
import importlib
from pathlib import Path


def test_streamlit_entrypoint_exists():
    assert Path("app.py").is_file()


def test_open_cpu_dependencies_are_declared():
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    assert "llama-cpp-python" in text
    assert "onnxruntime" in text
    assert "sentence-transformers" in text


def test_config_has_local_gguf_provider():
    from document_ai.config import Settings
    import inspect
    assert \"gguf\" in inspect.getsource(Settings)


def test_gguf_provider_is_importable_without_loading_model():
    module = importlib.import_module("document_ai.providers")
    assert hasattr(module, "GGUFAnswer")
