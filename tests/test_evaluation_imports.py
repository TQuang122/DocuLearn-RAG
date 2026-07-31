"""Evaluation module smoke tests."""

from __future__ import annotations

import importlib


def test_evaluation_modules_import() -> None:
    """Given the project environment, evaluation modules should import cleanly."""
    assert importlib.import_module("src.evaluation.chunking_strategy") is not None
    assert importlib.import_module("src.evaluation.evaluation_dataset") is not None
    assert importlib.import_module("src.evaluation.external_benchmarks") is not None
    assert importlib.import_module("src.evaluation.pdf_corpus") is not None
    assert importlib.import_module("src.evaluation.ragas_evaluator") is not None
    assert importlib.import_module("src.evaluation.retrieval_evaluator") is not None
    assert importlib.import_module("src.evaluation.retrieval_reranker") is not None
    assert importlib.import_module("src.evaluation.run_external_reranking") is not None
    assert importlib.import_module("src.evaluation.run_external_retrieval") is not None
    assert importlib.import_module("src.evaluation.run_retrieval") is not None
