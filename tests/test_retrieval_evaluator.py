from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.evaluation.evaluation_dataset import RetrievalEvaluationRecord
from src.evaluation.retrieval_evaluator import (
    evaluate_retrieval,
    ndcg_at_k,
    parse_metadata_filters,
    recall_at_k,
    reciprocal_rank,
    write_evaluation_results,
)
from src.schemas import ChunkMetadata, RetrievedChunk


def _chunk(chunk_id: str, filename: str, page: int) -> RetrievedChunk:
    return RetrievedChunk(
        text=f"Text for {chunk_id}",
        score=0.9,
        metadata=ChunkMetadata(
            document_id="doc-1",
            filename=filename,
            source=filename,
            page=page,
            chunk_id=chunk_id,
        ),
    )


def _record(**overrides: object) -> RetrievalEvaluationRecord:
    payload: dict[str, object] = {
        "id": "qa-001",
        "question": "What is RAG?",
        "ground_truth": "Retrieval-augmented generation.",
        "source_file": "rag.pdf",
        "gold_pages": [],
        "gold_chunk_ids": ["gold-1", "gold-2"],
        "answerable": True,
        "question_type": "definition",
        "difficulty": "easy",
    }
    return RetrievalEvaluationRecord.model_validate({**payload, **overrides})


def test_recall_at_k_counts_unique_gold_hits() -> None:
    retrieved = ["noise", "gold-1", "gold-1", "gold-2"]
    gold = {"gold-1", "gold-2"}

    assert recall_at_k(retrieved, gold, 1) == 0.0
    assert recall_at_k(retrieved, gold, 2) == 0.5
    assert recall_at_k(retrieved, gold, 4) == 1.0


def test_reciprocal_rank_uses_first_relevant_result() -> None:
    assert reciprocal_rank(["noise", "gold-1", "gold-2"], {"gold-1", "gold-2"}) == 0.5
    assert reciprocal_rank(["noise"], {"gold-1"}) == 0.0


def test_ndcg_at_k_uses_binary_relevance() -> None:
    actual = ndcg_at_k(["noise", "gold-1", "gold-2"], {"gold-1", "gold-2"}, 3)
    expected = (1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3))

    assert actual == pytest.approx(expected)


def test_metrics_reject_missing_gold_annotations() -> None:
    with pytest.raises(ValueError, match="gold cannot be empty"):
        recall_at_k(["chunk"], set(), 1)
    with pytest.raises(ValueError, match="gold cannot be empty"):
        reciprocal_rank(["chunk"], set())
    with pytest.raises(ValueError, match="gold cannot be empty"):
        ndcg_at_k(["chunk"], set(), 1)


def test_metadata_filter_json_is_validated_and_normalized() -> None:
    assert parse_metadata_filters('{"filenames":["rag.pdf"]}') == {"filename": "rag.pdf"}
    with pytest.raises(ValueError, match="JSON object"):
        parse_metadata_filters('["rag.pdf"]')


def test_evaluator_separates_unanswerable_and_missing_gold() -> None:
    records = [
        _record(),
        _record(
            id="qa-002",
            question="Which source?",
            source_file=None,
            gold_pages=[],
            gold_chunk_ids=[],
        ),
        _record(
            id="qa-003",
            question="Not in the corpus?",
            ground_truth="",
            source_file=None,
            gold_pages=[],
            gold_chunk_ids=[],
            answerable=False,
        ),
    ]
    calls: list[dict[str, object]] = []

    def retrieve_fn(
        query: str,
        *,
        k: int,
        filters: dict[str, object] | None,
        collection_name: str | None,
    ) -> list[RetrievedChunk]:
        calls.append(
            {
                "query": query,
                "k": k,
                "filters": filters,
                "collection_name": collection_name,
            }
        )
        if query == "Not in the corpus?":
            return []
        return [
            _chunk("noise", "other.pdf", 1),
            _chunk("gold-1", "rag.pdf", 2),
            _chunk("gold-2", "rag.pdf", 3),
        ]

    result = evaluate_retrieval(
        records,
        retrieve_fn=retrieve_fn,
        collection_name="evaluation__rc_500_50",
        k=10,
        filters={"filename": ["rag.pdf"]},
    )

    assert result.summary["counts"] == {
        "total": 3,
        "answerable": 2,
        "evaluated_answerable": 1,
        "annotation_required": 1,
        "unanswerable": 1,
    }
    assert result.summary["metrics"]["recall@1"] == 0.0
    assert result.summary["metrics"]["recall@3"] == 1.0
    assert result.summary["metrics"]["mrr"] == 0.5
    assert result.summary["metrics"]["unanswerable_accuracy"] == 1.0
    assert result.cases[1]["status"] == "annotation_required"
    assert result.cases[1]["metrics"] is None
    assert result.cases[2]["status"] == "unanswerable"
    assert result.cases[2]["unanswerable_retrieval_empty"] is True
    assert calls[0]["collection_name"] == "evaluation__rc_500_50"
    assert calls[0]["filters"] == {"filename": ["rag.pdf"]}
    assert calls[0]["k"] == 10


def test_page_annotations_match_filename_and_page() -> None:
    record = _record(gold_chunk_ids=[], gold_pages=[4], source_file="rag.pdf")

    result = evaluate_retrieval(
        [record],
        retrieve_fn=lambda *_args, **_kwargs: [
            _chunk("wrong-file", "other.pdf", 4),
            _chunk("gold-page", "rag.pdf", 4),
        ],
        collection_name="collection",
        k=10,
    )

    assert result.cases[0]["relevance_mode"] == "page"
    assert result.summary["metrics"]["mrr"] == 0.5
    assert result.summary["metrics"]["recall@3"] == 1.0


def test_write_evaluation_results_creates_case_and_summary_files(tmp_path: Path) -> None:
    result = evaluate_retrieval(
        [_record()],
        retrieve_fn=lambda *_args, **_kwargs: [_chunk("gold-1", "rag.pdf", 1)],
        collection_name="collection",
        k=10,
    )

    case_path, summary_path = write_evaluation_results(result, tmp_path)

    assert case_path == tmp_path / "cases.jsonl"
    assert summary_path == tmp_path / "summary.json"
    assert json.loads(case_path.read_text(encoding="utf-8"))["id"] == "qa-001"
    assert json.loads(summary_path.read_text(encoding="utf-8"))["collection"] == "collection"


def test_evaluator_requires_records_and_all_recall_cutoffs() -> None:
    with pytest.raises(ValueError, match="records cannot be empty"):
        evaluate_retrieval([], collection_name="collection", k=10)
    with pytest.raises(ValueError, match="k must be at least 10"):
        evaluate_retrieval([_record()], collection_name="collection", k=5)
