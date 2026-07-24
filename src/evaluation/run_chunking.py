"""Run the registered chunking grid against a JSON evaluation set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from src.config import settings
from src.evaluation.chunking_strategy import recursive_strategies
from src.evaluation.ragas_evaluator import EvaluationCase, run_evaluation
from src.indexing import ingest
from src.rag import answer
from src.schemas import RagAnswer


def _answer_for_collection(collection_name: str):
    def answer_with_strategy(question: str) -> RagAnswer:
        return answer(question, collection_name=collection_name)

    return answer_with_strategy


def _load_cases(path: Path) -> list[EvaluationCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Evaluation file must contain a JSON list.")

    cases: list[EvaluationCase] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Evaluation case {index} must be an object.")
        question = item.get("question")
        ground_truth = item.get("ground_truth")
        if not isinstance(question, str) or not isinstance(ground_truth, str):
            raise ValueError(
                f"Evaluation case {index} requires string question and ground_truth fields."
            )
        cases.append({"question": question, "ground_truth": ground_truth})
    return cases


def run(test_cases_path: Path, output_dir: Path) -> list[dict[str, object]]:
    cases = _load_cases(test_cases_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, object]] = []

    for strategy in recursive_strategies():
        collection_name = f"{settings.qdrant_collection}__{strategy.strategy_id}"
        chunk_count = ingest(
            recreate=True,
            collection_name=collection_name,
            chunker=strategy.chunker,
        )

        evaluation = run_evaluation(
            cases,
            answer_fn=_answer_for_collection(collection_name),
        )
        frame = cast(Any, evaluation).to_pandas()
        numeric = frame.select_dtypes(include="number")
        metrics = {
            str(column): float(value) for column, value in numeric.mean(numeric_only=True).items()
        }
        report: dict[str, object] = {
            "strategy_id": strategy.strategy_id,
            "params": strategy.params,
            "chunk_count": chunk_count,
            "metrics": metrics,
        }
        (output_dir / f"{strategy.strategy_id}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        reports.append(report)

    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("test_cases", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation-results/chunking"))
    args = parser.parse_args()
    reports = run(
        cast(Path, args.test_cases),
        cast(Path, args.output_dir),
    )
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
