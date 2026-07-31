from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from src.evaluation.evaluation_dataset import load_evaluation_records
from src.evaluation.retrieval_error_analysis import (
    analyze_retrieval_errors,
    load_cases,
    write_error_analysis,
)
from src.evaluation.retrieval_evaluator import (
    evaluate_retrieval,
    parse_metadata_filters,
    write_evaluation_results,
)
from src.evaluation.retrieval_reranker import expanded_rerank_retriever


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate expanded dense retrieval with lexical reranking, without an LLM."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--dense-weight", type=float, default=0.25)
    parser.add_argument("--max-chunks-per-page", type=int, default=1)
    parser.add_argument("--filters-json")
    parser.add_argument("--baseline-cases", type=Path)
    parser.add_argument("--error-analysis-output", type=Path)
    args = parser.parse_args()

    records = load_evaluation_records(cast(Path, args.dataset))
    filters = parse_metadata_filters(cast(str | None, args.filters_json))
    max_chunks_per_page = cast(int, args.max_chunks_per_page)
    result = evaluate_retrieval(
        records,
        retrieve_fn=expanded_rerank_retriever(
            candidate_k=cast(int, args.candidate_k),
            dense_weight=cast(float, args.dense_weight),
            max_chunks_per_page=max_chunks_per_page or None,
        ),
        collection_name=cast(str, args.collection),
        k=cast(int, args.k),
        filters=filters,
    )
    result.summary["variant"] = {
        "name": "expanded_dense_lexical_rerank",
        "candidate_k": cast(int, args.candidate_k),
        "dense_weight": cast(float, args.dense_weight),
        "max_chunks_per_page": max_chunks_per_page or None,
    }
    write_evaluation_results(result, cast(Path, args.output_dir))
    if args.baseline_cases is not None or args.error_analysis_output is not None:
        if args.baseline_cases is None or args.error_analysis_output is None:
            parser.error("--baseline-cases and --error-analysis-output must be used together")
        analysis = analyze_retrieval_errors(
            records,
            load_cases(cast(Path, args.baseline_cases)),
            result.cases,
        )
        write_error_analysis(analysis, cast(Path, args.error_analysis_output))
    print(json.dumps(result.summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
