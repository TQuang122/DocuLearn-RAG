"""Configuration normalization tests."""

from __future__ import annotations

import pytest

from src.config import Settings


def test_blank_api_key_disables_api_auth(monkeypatch) -> None:
    """A blank documented API key should behave like missing configuration."""
    monkeypatch.setenv("RAG_API_KEY", "   ")

    configured = Settings(_env_file=None)

    assert configured.api_key is None


def test_fusion_retrieval_settings_load_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("RAG_RETRIEVAL_MODE", "fusion")
    monkeypatch.setenv("RAG_RETRIEVAL_CANDIDATE_K", "50")
    monkeypatch.setenv("RAG_RETRIEVAL_DENSE_WEIGHT", "0.25")
    monkeypatch.setenv("RAG_RETRIEVAL_MAX_CHUNKS_PER_PAGE", "1")
    monkeypatch.setenv("RAG_RETRIEVAL_FALLBACK_TO_DENSE", "true")
    monkeypatch.setenv("RAG_RETRIEVAL_TELEMETRY_ENABLED", "true")
    monkeypatch.setenv("RAG_RETRIEVAL_SHADOW_SAMPLE_RATE", "0.1")
    monkeypatch.setenv("RAG_RETRIEVAL_TELEMETRY_MAX_BYTES", "1048576")
    monkeypatch.setenv("RAG_RETRIEVAL_TELEMETRY_RETAINED_EVENTS", "1000")
    monkeypatch.setenv("RAG_EMBEDDING_WARMUP_ENABLED", "true")

    configured = Settings(_env_file=None)

    assert configured.retrieval_mode == "fusion"
    assert configured.retrieval_candidate_k == 50
    assert configured.retrieval_dense_weight == 0.25
    assert configured.retrieval_max_chunks_per_page == 1
    assert configured.retrieval_fallback_to_dense is True
    assert configured.retrieval_telemetry_enabled is True
    assert configured.retrieval_shadow_sample_rate == 0.1
    assert configured.retrieval_telemetry_max_bytes == 1048576
    assert configured.retrieval_telemetry_retained_events == 1000
    assert configured.embedding_warmup_enabled is True


def test_shadow_sampling_requires_telemetry(monkeypatch) -> None:
    monkeypatch.setenv("RAG_RETRIEVAL_TELEMETRY_ENABLED", "false")
    monkeypatch.setenv("RAG_RETRIEVAL_SHADOW_SAMPLE_RATE", "0.1")

    with pytest.raises(ValueError, match="TELEMETRY_ENABLED"):
        Settings(_env_file=None)
