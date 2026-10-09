"""Validated configuration shared by the UI and terminal entry point."""

from dataclasses import dataclass, field
import os
from typing import Mapping


@dataclass(frozen=True)
class Settings:
    api_key: str = field(repr=False)
    chat_model: str = "gemini-3.8-flash"
    embedding_model: str = "gemini-embedding-001"
    retrieval_mode: str = "rag"
    retrieval_k: int = 4
    min_similarity: float = 0.35
    max_question_chars: int = 2000
    history_turns: int = 6
    session_requests_per_minute: int = 10
    process_requests_per_minute: int = 60
    max_direct_context_chars: int = 30000

    def __post_init__(self):
        if not self.api_key.strip():
            raise ValueError("Set GOOGLE_API_KEY in .env or Streamlit secrets.")
        if not self.chat_model.strip() or not self.embedding_model.strip():
            raise ValueError("Model IDs cannot be empty.")
        if self.retrieval_mode not in {"rag", "direct"}:
            raise ValueError("RETRIEVAL_MODE must be rag or direct.")
        if not 1 <= self.retrieval_k <= 12:
            raise ValueError("RETRIEVAL_K must be between 1 and 12.")
        if not 0 <= self.min_similarity <= 1:
            raise ValueError("MIN_SIMILARITY must be between 0 and 1.")
        for name in ("max_question_chars", "history_turns",
                     "session_requests_per_minute", "process_requests_per_minute",
                     "max_direct_context_chars"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive.")


def load_settings(secrets: Mapping | None = None) -> Settings:
    """Secrets take precedence; never overwrite process environment variables."""
    secrets = secrets or {}

    def value(name, default):
        return secrets.get(name, os.getenv(name, default))

    try:
        return Settings(
            api_key=str(value("GOOGLE_API_KEY", "")),
            chat_model=str(value("GEMINI_CHAT_MODEL", "gemini-3.8-flash")),
            embedding_model=str(value("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")),
            retrieval_mode=str(value("RETRIEVAL_MODE", "rag")),
            retrieval_k=int(value("RETRIEVAL_K", 4)),
            min_similarity=float(value("MIN_SIMILARITY", 0.35)),
            session_requests_per_minute=int(value("SESSION_REQUESTS_PER_MINUTE", 10)),
            process_requests_per_minute=int(value("PROCESS_REQUESTS_PER_MINUTE", 60)),
        )
    except (TypeError, ValueError) as exc:
        # Configuration may include a secret, so do not echo raw values.
        raise ValueError(
            "Check GOOGLE_API_KEY, model IDs, RETRIEVAL_MODE, RETRIEVAL_K, "
            "MIN_SIMILARITY, and request limits. See .env.example."
        ) from exc
