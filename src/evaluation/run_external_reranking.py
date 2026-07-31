from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from src.evaluation.chunking_strategy import ChunkingStrategy, recursive_strategies
from src.evaluation.external_benchmarks import load_external_benchmark
from src.evaluation.retrieval_evaluator import write_evaluation_results
from src.evaluation.retrieval_reranker import reranking_retriever
from src.evaluation.run_external_retrieval import (
    ExternalRetrieverFactory,
    build_exact_retriever,
    build_external_chunks,
    evaluate_external_queries,
)


def _strategy(strategy_id: str) -> ChunkingStrategy:
    strategies = {strategy.strategy_id: strategy for strategy in recursive_strategies()}
    try:
        return strategies[strategy_id]
    except KeyError as error:
        choices = ", ".join(strategies)
        raise ValueError(f"Unknown strategy '{strategy_id}'; choose one of: {choices}") from error


def run_external_reranking(
    benchmark_dir: Path,
    output_dir: Path,
    *,
    strategy_id: str = "rc_800_100",
    k: int = 10,
    candidate_k: int = 50,
    dense_weight: float = 0.25,
    max_chunks_per_page: int | None = 1,
    limit: int | None = None,
    retriever_factory: ExternalRetrieverFactory = build_exact_retriever,
) -> dict[str, object]:
    corpus, queries, manifest = load_external_benchmark(benchmark_dir)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive.")
        queries = queries[:limit]
    strategy = _strategy(strategy_id)
    chunks = build_external_chunks(corpus, strategy)
    dense_retriever = retriever_factory(chunks)
    retrieve_fn = reranking_retriever(
        dense_retriever,
        candidate_k=candidate_k,
        dense_weight=dense_weight,
        max_chunks_per_page=max_chunks_per_page,
    )
    result = evaluate_external_queries(
        queries,
        collection_name=f"external_rerank__{strategy_id}",
        k=k,
        retrieve_fn=retrieve_fn,
    )
    result.summary["variant"] = {
        "name": "expanded_dense_lexical_rerank",
        "candidate_k": candidate_k,
        "dense_weight": dense_weight,
        "max_chunks_per_page": max_chunks_per_page,
    }
    write_evaluation_results(result, output_dir)
    report: dict[str, object] = {
        "benchmark": manifest["benchmark"],
        "strategy_id": strategy.strategy_id,
        "params": strategy.params,
        "corpus_documents": len(corpus),
        "chunk_count": len(chunks),
        "query_count": len(queries),
        "summary": result.summary,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LLM-free lexical reranking on one external benchmark strategy."
    )
    parser.add_argument("benchmark_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--strategy", default="rc_800_100")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--dense-weight", type=float, default=0.25)
    parser.add_argument("--max-chunks-per-page", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    report = run_external_reranking(
        cast(Path, args.benchmark_dir),
        cast(Path, args.output_dir),
        strategy_id=cast(str, args.strategy),
        k=cast(int, args.k),
        candidate_k=cast(int, args.candidate_k),
        dense_weight=cast(float, args.dense_weight),
        max_chunks_per_page=cast(int, args.max_chunks_per_page) or None,
        limit=cast(int | None, args.limit),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
