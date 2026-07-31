from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from src.evaluation.evaluation_dataset import RetrievalEvaluationRecord


def _recall_at_10(case: dict[str, Any]) -> float:
    metrics = cast(dict[str, float] | None, case.get("metrics"))
    return metrics.get("recall@10", 0.0) if metrics else 0.0


def analyze_retrieval_errors(
    records: list[RetrievalEvaluationRecord],
    baseline_cases: list[dict[str, Any]],
    variant_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    records_by_id = {record.id: record for record in records}
    variant_by_id = {str(case["id"]): case for case in variant_cases}
    if set(records_by_id) != {str(case["id"]) for case in baseline_cases}:
        raise ValueError("Baseline case ids do not match evaluation records.")
    if set(records_by_id) != set(variant_by_id):
        raise ValueError("Variant case ids do not match evaluation records.")

    failures: list[dict[str, Any]] = []
    category_counts: dict[str, int] = {}
    recovered_count = 0
    for baseline in baseline_cases:
        if _recall_at_10(baseline) > 0:
            continue
        record = records_by_id[str(baseline["id"])]
        retrieved = cast(list[dict[str, Any]], baseline["retrieved"])
        same_source_hits = [
            hit for hit in retrieved if str(hit.get("filename")) == record.source_file
        ]
        category = "same_source_wrong_page" if same_source_hits else "source_absent_top10"
        category_counts[category] = category_counts.get(category, 0) + 1
        unique_pages = {
            (str(hit.get("filename")), int(hit.get("page", 0))) for hit in retrieved
        }
        variant = variant_by_id[record.id]
        recovered = _recall_at_10(variant) > 0
        recovered_count += int(recovered)
        failures.append(
            {
                "id": record.id,
                "question": record.question,
                "gold_source_file": record.source_file,
                "gold_pages": record.gold_pages,
                "category": category,
                "same_source_pages_in_baseline_top10": sorted(
                    {int(hit["page"]) for hit in same_source_hits}
                ),
                "duplicate_page_slots": len(retrieved) - len(unique_pages),
                "recovered_by_variant_at_10": recovered,
            }
        )
    return {
        "baseline_miss_at_10": len(failures),
        "recovered_by_variant_at_10": recovered_count,
        "remaining_miss_at_10": len(failures) - recovered_count,
        "category_counts": category_counts,
        "failures": failures,
    }


def load_cases(path: Path) -> list[dict[str, Any]]:
    return [
        cast(dict[str, Any], json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_error_analysis(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
