"""Application settings loaded from environment / .env."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM
    google_api_key: str = Field(..., description="Google AI Studio API key.")
    gemini_model: str = Field(default="gemini-2.0-flash")
    embedding_model: str = Field(default="gemini-embedding-001")

    # Vector store
    chroma_host: str = Field(default="chroma")
    chroma_port: int = Field(default=8000)
    chroma_collection: str = Field(default="papers")

    # Persistence
    sqlite_path: str = Field(default="/data/sqlite/app.db")
    pdf_dir: str = Field(default="/data/pdfs")

    # API
    api_port: int = Field(default=8080)
    log_level: str = Field(default="INFO")

    # Agent runtime
    agent_max_steps: int = Field(default=8, ge=1, le=32)

    @property
    def sqlalchemy_url(self) -> str:
        return f"sqlite+aiosqlite:///{self.sqlite_path}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
