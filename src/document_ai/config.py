import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True, slots=True)
class Settings:
    provider: str = os.getenv("AI_PROVIDER", "gguf")
    local_embedding_model: str = os.getenv(
        "LOCAL_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    hf_chat_model: str = os.getenv("HF_CHAT_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
    local_chat_model: str = os.getenv("LOCAL_CHAT_MODEL", "Qwen/Qwen3-0.6B")
    chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
    embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    enable_image_captions: bool = os.getenv("ENABLE_IMAGE_CAPTIONS", "false").lower() == "true"
    top_k: int = int(os.getenv("TOP_K", "3"))
