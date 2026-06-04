"""Application settings loaded from environment / .env."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
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
    gemini_model: str = Field(default="gemini-2.5-flash")
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
    cors_allow_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])

    # Agent runtime
    agent_max_steps: int = Field(default=8, ge=1, le=32)

    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def _parse_cors_allow_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @property
    def sqlalchemy_url(self) -> str:
        return f"sqlite+aiosqlite:///{self.sqlite_path}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
