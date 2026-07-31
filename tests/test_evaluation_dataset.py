from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.evaluation.evaluation_dataset import (
    RetrievalEvaluationRecord,
    convert_csv_to_jsonl,
    load_evaluation_records,
)


def test_convert_csv_preserves_unknown_source_metadata(tmp_path: Path) -> None:
    input_path = tmp_path / "benchmark.csv"
    output_path = tmp_path / "benchmark.jsonl"
    report_path = tmp_path / "annotation-report.json"
    with input_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["no.", "question", "ground truth"])
        writer.writeheader()
        writer.writerow(
            {
                "no.": "1",
                "question": "What is retrieval?",
                "ground truth": "Retrieval finds relevant source passages.",
            }
        )

    report = convert_csv_to_jsonl(input_path, output_path, report_path)
    records = load_evaluation_records(output_path)

    assert records == [
        RetrievalEvaluationRecord(
            id="qa-001",
            question="What is retrieval?",
            ground_truth="Retrieval finds relevant source passages.",
            source_file=None,
            gold_pages=[],
            gold_chunk_ids=[],
            answerable=True,
            question_type=None,
            difficulty=None,
        )
    ]
    assert report["annotation_required_ids"] == ["qa-001"]
    assert json.loads(report_path.read_text(encoding="utf-8"))["missing_fields"] == {
        "source_file": 1,
        "gold_pages_or_chunk_ids": 1,
        "question_type": 1,
        "difficulty": 1,
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"id": " "}, "Value cannot be blank"),
        ({"answerable": True, "ground_truth": ""}, "non-empty ground_truth"),
        ({"source_file": None, "gold_pages": [2]}, "source_file is required"),
        (
            {"answerable": False, "ground_truth": "", "gold_chunk_ids": ["chunk-1"]},
            "Unanswerable records cannot contain",
        ),
    ],
)
def test_record_validation_rejects_invalid_boundaries(
    overrides: dict[str, object],
    message: str,
) -> None:
    payload: dict[str, object] = {
        "id": "qa-001",
        "question": "What is RAG?",
        "ground_truth": "Retrieval-augmented generation.",
        "source_file": "rag.pdf",
        "gold_pages": [1],
        "gold_chunk_ids": [],
        "answerable": True,
        "question_type": "definition",
        "difficulty": "easy",
    }

    with pytest.raises(ValidationError, match=message):
        RetrievalEvaluationRecord.model_validate({**payload, **overrides})


def test_load_records_rejects_duplicate_ids(tmp_path: Path) -> None:
    dataset_path = tmp_path / "benchmark.jsonl"
    record = {
        "id": "qa-001",
        "question": "What is RAG?",
        "ground_truth": "Retrieval-augmented generation.",
        "source_file": None,
        "gold_pages": [],
        "gold_chunk_ids": [],
        "answerable": True,
        "question_type": None,
        "difficulty": None,
    }
    dataset_path.write_text(
        f"{json.dumps(record)}\n{json.dumps(record)}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate evaluation record id"):
        load_evaluation_records(dataset_path)
