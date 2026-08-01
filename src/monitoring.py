from __future__ import annotations

import hashlib
import os
import secrets
from typing import Annotated

from fastapi import FastAPI, Header, HTTPException, status

from src.config import settings
from src.retrieval_telemetry import load_retrieval_summary

MONITORING_PATH = "/monitoring/retrieval/summary"


def _configured_credentials() -> tuple[str | None, str | None]:
    api_key = os.getenv("RAG_API_KEY") or settings.api_key
    api_key_sha256 = os.getenv("RAG_API_KEY_SHA256") or settings.api_key_sha256
    return api_key, api_key_sha256


def _valid_api_key(candidate: str, api_key: str | None, api_key_sha256: str | None) -> bool:
    if api_key is not None and secrets.compare_digest(candidate, api_key):
        return True
    if api_key_sha256 is None:
        return False
    candidate_sha256 = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
    return secrets.compare_digest(candidate_sha256, api_key_sha256)


def retrieval_monitoring(
    x_api_key: Annotated[str | None, Header()] = None,
) -> dict[str, object]:
    api_key, api_key_sha256 = _configured_credentials()
    if api_key is None and api_key_sha256 is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found.")
    if x_api_key is None or not _valid_api_key(x_api_key, api_key, api_key_sha256):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key.",
        )
    return load_retrieval_summary()


def add_monitoring_route(app: FastAPI) -> None:
    if any(getattr(route, "path", None) == MONITORING_PATH for route in app.routes):
        return
    app.add_api_route(
        MONITORING_PATH,
        retrieval_monitoring,
        methods=["GET"],
        name="retrieval_monitoring",
        include_in_schema=False,
    )
