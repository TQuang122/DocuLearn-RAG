"""Evaluation module smoke tests."""

from __future__ import annotations

import importlib


def test_evaluation_modules_import() -> None:
    """Given the project environment, evaluation modules should import cleanly."""
    assert importlib.import_module("src.evaluation.chunking_strategy") is not None
    assert importlib.import_module("src.evaluation.ragas_evaluator") is not None
