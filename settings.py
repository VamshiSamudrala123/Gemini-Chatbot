"""Validated configuration shared by the UI and terminal entry point."""

from dataclasses import dataclass, field
import os
from typing import Mapping


class ConfigurationError(ValueError):
    """Safe, field-specific diagnostics that never include configured values."""


class MissingAPIKeyError(ConfigurationError):
    pass


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
        if not isinstance(self.api_key, str) or not self.api_key.strip():
            raise MissingAPIKeyError("GOOGLE_API_KEY is missing or empty.")
        if not self.chat_model.strip() or not self.embedding_model.strip():
            raise ConfigurationError("GEMINI_CHAT_MODEL and GEMINI_EMBEDDING_MODEL cannot be empty.")
        if self.retrieval_mode not in {"rag", "direct"}:
            raise ConfigurationError("RETRIEVAL_MODE must be rag or direct.")
        if not 1 <= self.retrieval_k <= 12:
            raise ConfigurationError("RETRIEVAL_K must be between 1 and 12.")
        if not 0 <= self.min_similarity <= 1:
            raise ConfigurationError("MIN_SIMILARITY must be between 0 and 1.")
        for name in ("max_question_chars", "history_turns",
                     "session_requests_per_minute", "process_requests_per_minute",
                     "max_direct_context_chars"):
            if getattr(self, name) < 1:
                raise ConfigurationError(f"{name.upper()} must be positive.")


def load_settings(secrets: Mapping | None = None) -> Settings:
    """Secrets take precedence; never overwrite process environment variables."""
    secrets = secrets or {}

    def value(name, default):
        return secrets.get(name, os.getenv(name, default))

    api_key = value("GOOGLE_API_KEY", "")
    if not isinstance(api_key, str):
        raise ConfigurationError("GOOGLE_API_KEY must be a quoted text value.")
    if not api_key.strip():
        nested = any(
            isinstance(section, Mapping) and "GOOGLE_API_KEY" in section
            for section in secrets.values()
        )
        if nested:
            raise ConfigurationError(
                "GOOGLE_API_KEY is inside a TOML section. Move it above every "
                "[section] header in Streamlit secrets."
            )
        if "GOOGLE_API_KEY" in secrets:
            raise MissingAPIKeyError(
                "GOOGLE_API_KEY in Streamlit secrets is empty. "
                "A blank secret overrides the environment value."
            )
        raise MissingAPIKeyError(
            "GOOGLE_API_KEY was not found. Add it in this Streamlit app's Settings > Secrets "
            "or in .env beside app.py when running locally."
        )

    def text(name, default):
        result = value(name, default)
        if not isinstance(result, str) or not result.strip():
            raise ConfigurationError(f"{name} must be a non-empty text value.")
        return result.strip()

    def integer(name, default):
        result = value(name, default)
        if isinstance(result, bool) or not isinstance(result, (int, str)):
            raise ConfigurationError(f"{name} must be an integer.")
        try:
            return int(result)
        except ValueError:
            raise ConfigurationError(f"{name} must be an integer.") from None

    def number(name, default):
        result = value(name, default)
        if isinstance(result, bool) or not isinstance(result, (int, float, str)):
            raise ConfigurationError(f"{name} must be a number.")
        try:
            return float(result)
        except ValueError:
            raise ConfigurationError(f"{name} must be a number.") from None

    return Settings(
        api_key=api_key.strip(),
        chat_model=text("GEMINI_CHAT_MODEL", "gemini-3.8-flash"),
        embedding_model=text("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"),
        retrieval_mode=text("RETRIEVAL_MODE", "rag"),
        retrieval_k=integer("RETRIEVAL_K", 4),
        min_similarity=number("MIN_SIMILARITY", 0.35),
        session_requests_per_minute=integer("SESSION_REQUESTS_PER_MINUTE", 10),
        process_requests_per_minute=integer("PROCESS_REQUESTS_PER_MINUTE", 60),
    )
