from __future__ import annotations

from src.evaluation.evaluation_dataset import RetrievalEvaluationRecord
from src.evaluation.retrieval_error_analysis import analyze_retrieval_errors


def _record(record_id: str, page: int) -> RetrievalEvaluationRecord:
    return RetrievalEvaluationRecord(
        id=record_id,
        question=f"Question {record_id}",
        ground_truth="Answer",
        source_file="gold.pdf",
        gold_pages=[page],
        gold_chunk_ids=[],
        answerable=True,
        question_type="factual",
        difficulty="easy",
    )


def _case(record_id: str, recall: float, hits: list[tuple[str, int]]) -> dict[str, object]:
    return {
        "id": record_id,
        "metrics": {"recall@10": recall},
        "retrieved": [
            {"filename": filename, "page": page} for filename, page in hits
        ],
    }


def test_error_analysis_classifies_and_tracks_recovered_failures() -> None:
    records = [_record("qa-001", 2), _record("qa-002", 3)]
    baseline = [
        _case("qa-001", 0.0, [("gold.pdf", 1), ("gold.pdf", 1)]),
        _case("qa-002", 0.0, [("other.pdf", 3)]),
    ]
    variant = [
        _case("qa-001", 1.0, [("gold.pdf", 2)]),
        _case("qa-002", 0.0, [("other.pdf", 3)]),
    ]

    report = analyze_retrieval_errors(records, baseline, variant)

    assert report["baseline_miss_at_10"] == 2
    assert report["recovered_by_variant_at_10"] == 1
    assert report["category_counts"] == {
        "same_source_wrong_page": 1,
        "source_absent_top10": 1,
    }
    assert report["failures"][0]["duplicate_page_slots"] == 1
