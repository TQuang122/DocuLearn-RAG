"""FastAPI boundary tests."""

from __future__ import annotations

import hashlib

from fastapi.testclient import TestClient

from src.config import settings
from src.interfaces.api import app


def test_health_returns_status() -> None:
    """Given a running API, health should be available without external services."""
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_upload_rejects_non_pdf_content() -> None:
    """Given a fake PDF, upload should reject it before writing or embedding."""
    response = TestClient(app).post(
        "/upload",
        files={"file": ("notes.pdf", b"not a pdf", "application/pdf")},
    )

    assert response.status_code == 400


def test_configured_api_key_protects_documents(monkeypatch) -> None:
    """Configured API auth should reject missing or incorrect keys."""
    monkeypatch.setattr(settings, "api_key", "test-secret")
    monkeypatch.setattr("src.interfaces.api.list_documents", lambda: [])
    client = TestClient(app)

    assert client.get("/documents").status_code == 401
    assert client.get("/documents", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/documents", headers={"X-API-Key": "test-secret"}).status_code == 200


def test_monitoring_endpoint_is_disabled_without_api_key(monkeypatch) -> None:
    monkeypatch.setattr(settings, "api_key", None)
    monkeypatch.setattr(settings, "api_key_sha256", None)
    monkeypatch.delenv("RAG_API_KEY", raising=False)
    monkeypatch.delenv("RAG_API_KEY_SHA256", raising=False)

    response = TestClient(app).get("/monitoring/retrieval/summary")

    assert response.status_code == 404


def test_monitoring_endpoint_requires_key_and_returns_summary(monkeypatch) -> None:
    monkeypatch.setattr(settings, "api_key", "test-secret")
    monkeypatch.setattr(settings, "api_key_sha256", None)
    monkeypatch.setattr(
        "src.monitoring.load_retrieval_summary",
        lambda: {"promotion_gate": {"status": "insufficient_data"}},
    )
    client = TestClient(app)

    assert client.get("/monitoring/retrieval/summary").status_code == 401
    response = client.get(
        "/monitoring/retrieval/summary",
        headers={"X-API-Key": "test-secret"},
    )

    assert response.status_code == 200
    assert response.json()["promotion_gate"]["status"] == "insufficient_data"


def test_monitoring_endpoint_accepts_hashed_api_key(monkeypatch) -> None:
    monkeypatch.setattr(settings, "api_key", None)
    monkeypatch.setattr(
        settings,
        "api_key_sha256",
        hashlib.sha256(b"test-secret").hexdigest(),
    )
    monkeypatch.setattr(
        "src.monitoring.load_retrieval_summary",
        lambda: {"promotion_gate": {"status": "insufficient_data"}},
    )
    client = TestClient(app)

    assert client.get("/monitoring/retrieval/summary").status_code == 401
    assert (
        client.get(
            "/monitoring/retrieval/summary",
            headers={"X-API-Key": "test-secret"},
        ).status_code
        == 200
    )
