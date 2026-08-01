from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import FastAPI, Header, HTTPException, status

from src.config import settings
from src.retrieval_telemetry import load_retrieval_summary

MONITORING_PATH = "/api/monitoring/retrieval"


def retrieval_monitoring(
    x_api_key: Annotated[str | None, Header()] = None,
) -> dict[str, object]:
    expected = settings.api_key
    if expected is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found.")
    if x_api_key is None or not secrets.compare_digest(x_api_key, expected):
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
