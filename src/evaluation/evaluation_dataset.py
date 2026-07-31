from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RetrievalEvaluationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    ground_truth: str
    source_file: str | None = None
    gold_pages: list[int] = Field(default_factory=list)
    gold_chunk_ids: list[str] = Field(default_factory=list)
    answerable: bool
    question_type: str | None = None
    difficulty: Literal["easy", "medium", "hard"] | None = None

    @field_validator("id", "question")
    @classmethod
    def _strip_required_strings(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Value cannot be blank.")
        return stripped

    @field_validator("ground_truth")
    @classmethod
    def _strip_ground_truth(cls, value: str) -> str:
        return value.strip()

    @field_validator("source_file", "question_type")
    @classmethod
    def _strip_optional_strings(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @field_validator("gold_pages")
    @classmethod
    def _normalize_pages(cls, pages: list[int]) -> list[int]:
        if any(page < 1 for page in pages):
            raise ValueError("gold_pages must contain positive page numbers.")
        return sorted(set(pages))

    @field_validator("gold_chunk_ids")
    @classmethod
    def _normalize_chunk_ids(cls, chunk_ids: list[str]) -> list[str]:
        return list(dict.fromkeys(chunk_id.strip() for chunk_id in chunk_ids if chunk_id.strip()))

    @model_validator(mode="after")
    def _validate_grounding(self) -> RetrievalEvaluationRecord:
        if self.answerable and not self.ground_truth:
            raise ValueError("Answerable records require a non-empty ground_truth.")
        if self.gold_pages and not self.source_file:
            raise ValueError("source_file is required when gold_pages are provided.")
        if not self.answerable and (self.gold_pages or self.gold_chunk_ids):
            raise ValueError("Unanswerable records cannot contain gold source annotations.")
        return self

    @property
    def has_gold_sources(self) -> bool:
        return bool(self.gold_chunk_ids or (self.source_file and self.gold_pages))

    @property
    def annotation_required(self) -> bool:
        return (
            (self.answerable and not self.has_gold_sources)
            or self.question_type is None
            or self.difficulty is None
        )


def load_evaluation_records(path: Path) -> list[RetrievalEvaluationRecord]:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")

    if path.suffix.lower() == ".jsonl":
        payloads: list[Any] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                payloads.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at line {line_number}: {error.msg}") from error
    elif path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("JSON evaluation dataset must contain a list.")
        payloads = payload
    else:
        raise ValueError("Evaluation dataset must use .json or .jsonl.")

    if not payloads:
        raise ValueError("Evaluation dataset cannot be empty.")

    records: list[RetrievalEvaluationRecord] = []
    seen_ids: set[str] = set()
    for index, payload in enumerate(payloads):
        if not isinstance(payload, dict):
            raise ValueError(f"Evaluation record {index} must be an object.")
        record = RetrievalEvaluationRecord.model_validate(payload)
        if record.id in seen_ids:
            raise ValueError(f"Duplicate evaluation record id: {record.id}")
        seen_ids.add(record.id)
        records.append(record)
    return records


def _record_id(raw_id: str, index: int) -> str:
    cleaned = raw_id.strip()
    if cleaned.isdigit():
        return f"qa-{int(cleaned):03d}"
    return cleaned or f"qa-{index:03d}"


def convert_csv_to_jsonl(
    input_path: Path,
    output_path: Path,
    report_path: Path,
) -> dict[str, object]:
    with input_path.open(encoding="utf-8-sig", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    if not rows:
        raise ValueError("Benchmark CSV cannot be empty.")

    records: list[RetrievalEvaluationRecord] = []
    for index, row in enumerate(rows, start=1):
        question = (row.get("question") or "").strip()
        ground_truth = (row.get("ground truth") or row.get("ground_truth") or "").strip()
        if not question or not ground_truth:
            raise ValueError(f"CSV row {index} requires question and ground truth values.")
        records.append(
            RetrievalEvaluationRecord(
                id=_record_id(row.get("no.", ""), index),
                question=question,
                ground_truth=ground_truth,
                source_file=None,
                gold_pages=[],
                gold_chunk_ids=[],
                answerable=True,
                question_type=None,
                difficulty=None,
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(record.model_dump(), ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )

    annotation_ids = [record.id for record in records if record.annotation_required]
    report: dict[str, object] = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "total_records": len(records),
        "annotation_required_count": len(annotation_ids),
        "annotation_required_ids": annotation_ids,
        "missing_fields": {
            "source_file": len(records),
            "gold_pages_or_chunk_ids": len(records),
            "question_type": len(records),
            "difficulty": len(records),
        },
        "note": (
            "Source metadata, question type, and difficulty were not present in the CSV "
            "and were intentionally left unannotated."
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert the legacy benchmark CSV to validated evaluation JSONL."
    )
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = convert_csv_to_jsonl(args.input_csv, args.output, args.report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
