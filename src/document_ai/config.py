import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True, slots=True)
class Settings:
    chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-5.6-luna")
    embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    enable_image_captions: bool = os.getenv("ENABLE_IMAGE_CAPTIONS", "true").lower() == "true"
    top_k: int = int(os.getenv("TOP_K", "5"))
