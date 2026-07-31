from __future__ import annotations

import json
from pathlib import Path

from src.evaluation import run_retrieval
from src.evaluation.chunking_strategy import recursive_strategies
from src.evaluation.evaluation_dataset import RetrievalEvaluationRecord
from src.evaluation.retrieval_evaluator import RetrievalEvaluationResult


def test_run_baselines_uses_all_registered_chunking_strategies(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    record = RetrievalEvaluationRecord(
        id="qa-001",
        question="What is RAG?",
        ground_truth="Retrieval-augmented generation.",
        source_file="rag.pdf",
        gold_pages=[1],
        gold_chunk_ids=[],
        answerable=True,
        question_type="definition",
        difficulty="easy",
    )
    dataset_path.write_text(
        json.dumps(record.model_dump()) + "\n",
        encoding="utf-8",
    )
    indexed_collections: list[str] = []
    evaluated_collections: list[str] = []

    def fake_ingest(**kwargs: object) -> int:
        indexed_collections.append(str(kwargs["collection_name"]))
        return 12

    def fake_evaluate(
        records: list[RetrievalEvaluationRecord],
        **kwargs: object,
    ) -> RetrievalEvaluationResult:
        assert records == [record]
        collection_name = str(kwargs["collection_name"])
        evaluated_collections.append(collection_name)
        return RetrievalEvaluationResult(
            cases=[],
            summary={
                "collection": collection_name,
                "k": 10,
                "filters": None,
                "counts": {},
                "metrics": {},
                "retrieval_latency_ms": {"p50": 1.0, "p95": 2.0},
            },
        )

    monkeypatch.setattr(run_retrieval, "ingest", fake_ingest)
    monkeypatch.setattr(run_retrieval, "evaluate_retrieval", fake_evaluate)

    reports = run_retrieval.run_baselines(
        dataset_path,
        tmp_path / "results",
        collection_prefix="benchmark",
        k=10,
    )

    expected_suffixes = [
        "rc_500_50",
        "rc_800_100",
        "rc_1000_150",
        "rc_1500_200",
    ]
    assert [name.rsplit("__", 1)[-1] for name in indexed_collections] == expected_suffixes
    assert indexed_collections == evaluated_collections
    assert [report["strategy_id"] for report in reports] == expected_suffixes


def test_evaluation_corpus_indexes_explicit_pdf_paths(tmp_path: Path, monkeypatch) -> None:
    pdf_paths = [tmp_path / "nested" / "one.pdf", tmp_path / "nested" / "two.pdf"]
    calls: list[tuple[str, object]] = []
    chunks = [object(), object(), object()]

    monkeypatch.setattr(
        run_retrieval,
        "ensure_collection",
        lambda **kwargs: calls.append(("ensure", kwargs)),
    )
    monkeypatch.setattr(
        run_retrieval,
        "build_chunks",
        lambda paths, chunker: calls.append(("build", paths)) or chunks,
    )
    monkeypatch.setattr(
        run_retrieval,
        "index_chunks",
        lambda indexed, collection_name: calls.append(("index", collection_name)) or len(indexed),
    )

    count = run_retrieval.index_evaluation_corpus(
        pdf_paths,
        collection_name="evaluation",
        strategy=recursive_strategies()[0],
    )

    assert count == 3
    assert calls[0][0] == "ensure"
    assert calls[1] == ("build", pdf_paths)
    assert calls[2] == ("index", "evaluation")
