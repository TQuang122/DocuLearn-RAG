"""Configuration normalization tests."""

from __future__ import annotations

from src.config import Settings


def test_blank_api_key_disables_api_auth(monkeypatch) -> None:
    """A blank documented API key should behave like missing configuration."""
    monkeypatch.setenv("RAG_API_KEY", "   ")

    configured = Settings(_env_file=None)

    assert configured.api_key is None
