"""Application settings, read once from the environment and the project's .env file."""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_DIR / "frontend"
DATA_DIR = PROJECT_DIR / "data"
INDEX_DIR = DATA_DIR / "faiss_indices"
CONVERSATIONS_DIR = DATA_DIR / "conversations"

load_dotenv(PROJECT_DIR / ".env")


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _list(name: str, default: str = "") -> List[str]:
    return [x.strip() for x in _env(name, default).split(",") if x.strip()]


@dataclass(frozen=True)
class Settings:
    # Answers: groq (free), gemini, or openai (any OpenAI-compatible server); "auto" = first with a key
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "auto"))
    groq_api_key: Optional[str] = field(default_factory=lambda: _env("GROQ_API_KEY") or None)
    groq_models: List[str] = field(default_factory=lambda: _list(
        "GROQ_MODELS", "qwen/qwen3.8-27b,openai/gpt-oss-120b,allam-2-7b"))
    gemini_api_key: Optional[str] = field(
        default_factory=lambda: _env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY") or None)
    gemini_models: List[str] = field(default_factory=lambda: _list(
        "GEMINI_MODELS", _env("GEMINI_MODEL") or "gemini-2.5-flash,gemini-2.5-flash-lite"))
    openai_base_url: str = field(default_factory=lambda: _env("OPENAI_BASE_URL"))
    openai_api_key: Optional[str] = field(default_factory=lambda: _env("OPENAI_API_KEY") or None)
    openai_models: List[str] = field(default_factory=lambda: _list("OPENAI_MODELS", "qwen2.5:7b"))
    # Ask the model for fully vowelled Arabic (tashkeel): it makes the lip sync follow the real vowels
    llm_diacritics: bool = field(default_factory=lambda: _env("LLM_DIACRITICS", "true").lower() != "false")
    rag_top_k: int = field(default_factory=lambda: int(_env("RAG_TOP_K", "4")))

    # Voice (text-to-speech)
    tts_engine: str = field(default_factory=lambda: _env("TTS_ENGINE", "auto"))
    elevenlabs_api_key: Optional[str] = field(default_factory=lambda: _env("ELEVENLABS_API_KEY") or None)
    elevenlabs_voice_id: str = field(default_factory=lambda: _env("ELEVENLABS_VOICE_ID"))
    elevenlabs_model: str = field(default_factory=lambda: _env("ELEVENLABS_MODEL", "eleven_multilingual_v2"))
    edge_tts_rate: str = field(default_factory=lambda: _env("EDGE_TTS_RATE", "-5%"))

    # Server
    host: str = field(default_factory=lambda: _env("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(_env("PORT", "8000")))
    allowed_origins: List[str] = field(
        default_factory=lambda: [o.strip() for o in _env("ALLOWED_ORIGINS", "*").split(",") if o.strip()])
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))


settings = Settings()
