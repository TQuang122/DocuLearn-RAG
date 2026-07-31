from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, cast

from src.evaluation.evaluation_dataset import (
    RetrievalEvaluationRecord,
    load_evaluation_records,
)
from src.filters import MetadataFilter, filters_to_dict
from src.rag import retrieve
from src.schemas import RetrievedChunk

RECALL_CUTOFFS = (1, 3, 5, 10)
RetrieveFunction = Callable[..., list[RetrievedChunk]]


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationResult:
    cases: list[dict[str, Any]]
    summary: dict[str, Any]


def recall_at_k(
    retrieved: Sequence[Hashable],
    gold: set[Hashable],
    k: int,
) -> float:
    if not gold:
        raise ValueError("gold cannot be empty.")
    if k < 1:
        raise ValueError("k must be positive.")
    return len(set(retrieved[:k]) & gold) / len(gold)


def reciprocal_rank(
    retrieved: Sequence[Hashable],
    gold: set[Hashable],
) -> float:
    if not gold:
        raise ValueError("gold cannot be empty.")
    for rank, item in enumerate(retrieved, start=1):
        if item in gold:
            return 1 / rank
    return 0.0


def ndcg_at_k(
    retrieved: Sequence[Hashable],
    gold: set[Hashable],
    k: int,
) -> float:
    if not gold:
        raise ValueError("gold cannot be empty.")
    if k < 1:
        raise ValueError("k must be positive.")
    seen: set[Hashable] = set()
    discounted_gain = 0.0
    for rank, item in enumerate(retrieved[:k], start=1):
        if item in gold and item not in seen:
            discounted_gain += 1 / math.log2(rank + 1)
            seen = {*seen, item}
    ideal_gain = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(gold), k) + 1))
    return discounted_gain / ideal_gain


def _gold_and_retrieved_keys(
    record: RetrievalEvaluationRecord,
    chunks: list[RetrievedChunk],
) -> tuple[str, set[Hashable], list[Hashable]]:
    if record.gold_chunk_ids:
        return (
            "chunk_id",
            set(record.gold_chunk_ids),
            [chunk.metadata.chunk_id for chunk in chunks],
        )
    if record.source_file and record.gold_pages:
        return (
            "page",
            {(record.source_file, page) for page in record.gold_pages},
            [(chunk.metadata.filename, chunk.metadata.page) for chunk in chunks],
        )
    raise ValueError(f"Record {record.id} does not contain gold source metadata.")


def _retrieved_payload(chunks: list[RetrievedChunk]) -> list[dict[str, object]]:
    return [
        {
            "rank": rank,
            "score": chunk.score,
            "filename": chunk.metadata.filename,
            "page": chunk.metadata.page,
            "chunk_id": chunk.metadata.chunk_id,
        }
        for rank, chunk in enumerate(chunks, start=1)
    ]


def _ranking_metrics(
    retrieved_keys: list[Hashable],
    gold_keys: set[Hashable],
    k: int,
) -> dict[str, float]:
    metrics = {
        f"recall@{cutoff}": recall_at_k(retrieved_keys, gold_keys, cutoff)
        for cutoff in RECALL_CUTOFFS
    }
    return {
        **metrics,
        "mrr": reciprocal_rank(retrieved_keys[:k], gold_keys),
        "ndcg": ndcg_at_k(retrieved_keys, gold_keys, k),
    }


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _evaluate_case(
    record: RetrievalEvaluationRecord,
    chunks: list[RetrievedChunk],
    latency_ms: float,
    k: int,
) -> dict[str, Any]:
    common: dict[str, Any] = {
        "id": record.id,
        "question": record.question,
        "answerable": record.answerable,
        "latency_ms": latency_ms,
        "retrieved": _retrieved_payload(chunks),
    }
    if not record.answerable:
        return {
            **common,
            "status": "unanswerable",
            "relevance_mode": None,
            "metrics": None,
            "unanswerable_retrieval_empty": not chunks,
        }
    if not record.has_gold_sources:
        return {
            **common,
            "status": "annotation_required",
            "relevance_mode": None,
            "metrics": None,
            "unanswerable_retrieval_empty": None,
        }
    relevance_mode, gold_keys, retrieved_keys = _gold_and_retrieved_keys(record, chunks)
    return {
        **common,
        "status": "evaluated",
        "relevance_mode": relevance_mode,
        "metrics": _ranking_metrics(retrieved_keys, gold_keys, k),
        "unanswerable_retrieval_empty": None,
    }


def _summary(
    cases: list[dict[str, Any]],
    *,
    collection_name: str | None,
    k: int,
    filters: dict[str, object] | None,
) -> dict[str, Any]:
    evaluated = [case for case in cases if case["status"] == "evaluated"]
    unanswerable = [case for case in cases if case["status"] == "unanswerable"]
    metric_names = [*(f"recall@{cutoff}" for cutoff in RECALL_CUTOFFS), "mrr", "ndcg"]
    metrics = {
        name: mean(cast(dict[str, float], case["metrics"])[name] for case in evaluated)
        if evaluated
        else None
        for name in metric_names
    }
    metrics["unanswerable_accuracy"] = (
        mean(bool(case["unanswerable_retrieval_empty"]) for case in unanswerable)
        if unanswerable
        else None
    )
    latencies = [cast(float, case["latency_ms"]) for case in cases]
    answerable_count = sum(bool(case["answerable"]) for case in cases)
    return {
        "collection": collection_name,
        "k": k,
        "filters": filters,
        "counts": {
            "total": len(cases),
            "answerable": answerable_count,
            "evaluated_answerable": len(evaluated),
            "annotation_required": sum(case["status"] == "annotation_required" for case in cases),
            "unanswerable": len(unanswerable),
        },
        "metrics": metrics,
        "retrieval_latency_ms": {
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
        },
    }


def evaluate_retrieval(
    records: list[RetrievalEvaluationRecord],
    *,
    retrieve_fn: RetrieveFunction = retrieve,
    collection_name: str | None,
    k: int = 10,
    filters: dict[str, object] | None = None,
) -> RetrievalEvaluationResult:
    if not records:
        raise ValueError("records cannot be empty.")
    if k < max(RECALL_CUTOFFS):
        raise ValueError(f"k must be at least {max(RECALL_CUTOFFS)}.")

    cases: list[dict[str, Any]] = []
    for record in records:
        started_at = perf_counter()
        chunks = retrieve_fn(
            record.question,
            k=k,
            filters=filters,
            collection_name=collection_name,
        )
        latency_ms = (perf_counter() - started_at) * 1000
        cases.append(_evaluate_case(record, chunks, latency_ms, k))

    return RetrievalEvaluationResult(
        cases=cases,
        summary=_summary(
            cases,
            collection_name=collection_name,
            k=k,
            filters=filters,
        ),
    )


def write_evaluation_results(
    result: RetrievalEvaluationResult,
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = output_dir / "cases.jsonl"
    summary_path = output_dir / "summary.json"
    cases_path.write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in result.cases),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(result.summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return cases_path, summary_path


def parse_metadata_filters(raw_filters: str | None) -> dict[str, object] | None:
    if raw_filters is None:
        return None
    payload = json.loads(raw_filters)
    if not isinstance(payload, dict):
        raise ValueError("Metadata filters must be a JSON object.")
    return filters_to_dict(MetadataFilter.model_validate(payload))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval without invoking an LLM.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--filters-json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    records = load_evaluation_records(cast(Path, args.dataset))
    result = evaluate_retrieval(
        records,
        collection_name=cast(str, args.collection),
        k=cast(int, args.k),
        filters=parse_metadata_filters(cast(str | None, args.filters_json)),
    )
    write_evaluation_results(result, cast(Path, args.output_dir))
    print(json.dumps(result.summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
