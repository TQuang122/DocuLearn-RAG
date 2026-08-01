from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_prefix="RAG_",
        extra="ignore",
    )

    data_dir: Path = Path("data")
    storage_dir: Path = Path("storage/qdrant")
    export_dir: Path = Path("exports")
    qdrant_collection: str = "rag_chunks"
    max_upload_bytes: int = Field(default=25 * 1024 * 1024, ge=1024)

    chunk_size: int = Field(default=1500, ge=500)
    chunk_overlap: int = Field(default=200, ge=0)
    top_k: int = Field(default=5, ge=1, le=64)
    retrieval_mode: Literal["dense", "fusion"] = "dense"
    retrieval_candidate_k: int = Field(default=50, ge=1, le=256)
    retrieval_dense_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    retrieval_max_chunks_per_page: int = Field(default=1, ge=1, le=64)
    retrieval_fallback_to_dense: bool = True
    retrieval_telemetry_enabled: bool = False
    retrieval_shadow_sample_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    retrieval_telemetry_max_bytes: int = Field(default=5 * 1024 * 1024, ge=1024)
    retrieval_telemetry_retained_events: int = Field(default=5000, ge=100)

    embedding_warmup_enabled: bool = False

    llm_provider: Literal["gemini"] = "gemini"
    llm_model: str = "gemini-flash-lite-latest"
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_max_new_tokens: int = Field(default=10000, ge=1, le=20000)

    embedding_provider: Literal["local"] = "local"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    gemini_api_key: str | None = Field(default=None, validation_alias="GEMINI_API_KEY")
    api_key: str | None = None
    gradio_username: str | None = None
    gradio_password: str | None = None
    server_name: str = "127.0.0.1"
    server_port: int = Field(default=7860, ge=1, le=65535)

    summarize_batch_size: int = Field(default=10, ge=1)
    summarize_retrieval_k: int = Field(default=12, ge=1, le=128)
    generation_retrieval_k: int = Field(default=16, ge=1, le=128)
    quiz_default_count: int = Field(default=8, ge=1, le=50)
    flashcards_default_count: int = Field(default=15, ge=1, le=100)

    @field_validator(
        "gemini_api_key",
        "api_key",
        "gradio_username",
        "gradio_password",
        mode="before",
    )
    @classmethod
    def normalize_blank_secrets(cls, value: object) -> object:
        """Treat blank optional credentials as disabled configuration."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def validate_config(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")
        if bool(self.gradio_username) != bool(self.gradio_password):
            raise ValueError(
                "RAG_GRADIO_USERNAME and RAG_GRADIO_PASSWORD must be configured together."
            )
        if self.retrieval_shadow_sample_rate > 0 and not self.retrieval_telemetry_enabled:
            raise ValueError(
                "RAG_RETRIEVAL_TELEMETRY_ENABLED must be true when shadow sampling is enabled."
            )
        for field_name in ("data_dir", "storage_dir", "export_dir"):
            path = getattr(self, field_name)
            if not path.is_absolute():
                object.__setattr__(self, field_name, _PROJECT_ROOT / path)

        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
