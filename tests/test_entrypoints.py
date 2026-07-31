"""User-facing entrypoint regression tests."""

from __future__ import annotations

import subprocess
import sys


def test_cli_help_runs_from_project_root() -> None:
    """Given the installed project, CLI help should run without import workarounds."""
    result = subprocess.run(
        [sys.executable, "-m", "src.interfaces.cli", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "debug-retrieval" in result.stdout
    assert "retrieval-telemetry" in result.stdout


def test_api_exposes_complete_learning_surface() -> None:
    """Given the API module, all advertised learning routes should be registered."""
    from src.interfaces.api import app

    routes = {route.path for route in app.routes}

    assert {
        "/health",
        "/documents",
        "/upload",
        "/ask",
        "/summarize",
        "/quiz",
        "/flashcards",
    } <= routes


def test_gradio_app_imports_without_optional_logo() -> None:
    """Given no logo asset, importing the Gradio application should still succeed."""
    from app import demo

    assert demo is not None
