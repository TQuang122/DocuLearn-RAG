from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from src.config import settings
from src.evaluation.chunking_strategy import ChunkingStrategy, recursive_strategies
from src.evaluation.evaluation_dataset import load_evaluation_records
from src.evaluation.pdf_corpus import load_pdf_corpus_paths
from src.evaluation.retrieval_evaluator import (
    evaluate_retrieval,
    parse_metadata_filters,
    write_evaluation_results,
)
from src.indexing import build_chunks, index_chunks, ingest
from src.store import ensure_collection


def index_evaluation_corpus(
    pdf_paths: list[Path],
    *,
    collection_name: str,
    strategy: ChunkingStrategy,
) -> int:
    ensure_collection(recreate=True, collection_name=collection_name)
    chunks = build_chunks(pdf_paths, chunker=strategy.chunker)
    return index_chunks(chunks, collection_name=collection_name)


def run_baselines(
    dataset_path: Path,
    output_dir: Path,
    *,
    collection_prefix: str,
    k: int = 10,
    filters: dict[str, object] | None = None,
    pdf_paths: list[Path] | None = None,
    corpus_id: str | None = None,
) -> list[dict[str, object]]:
    records = load_evaluation_records(dataset_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, object]] = []

    for strategy in recursive_strategies():
        collection_name = f"{collection_prefix}__{strategy.strategy_id}"
        if pdf_paths is None:
            chunk_count = ingest(
                recreate=True,
                collection_name=collection_name,
                chunker=strategy.chunker,
            )
        else:
            chunk_count = index_evaluation_corpus(
                pdf_paths,
                collection_name=collection_name,
                strategy=strategy,
            )
        result = evaluate_retrieval(
            records,
            collection_name=collection_name,
            k=k,
            filters=filters,
        )
        write_evaluation_results(result, output_dir / strategy.strategy_id)
        reports.append(
            {
                "strategy_id": strategy.strategy_id,
                "params": strategy.params,
                "corpus_id": corpus_id,
                "chunk_count": chunk_count,
                "summary": result.summary,
            }
        )

    (output_dir / "baseline_summary.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index and evaluate all registered recursive chunking strategies."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--collection-prefix",
        default=f"{settings.qdrant_collection}__retrieval_eval",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--filters-json")
    parser.add_argument("--corpus-manifest", type=Path)
    args = parser.parse_args()
    manifest = None
    pdf_paths = None
    if args.corpus_manifest is not None:
        manifest, pdf_paths = load_pdf_corpus_paths(cast(Path, args.corpus_manifest))
    reports = run_baselines(
        cast(Path, args.dataset),
        cast(Path, args.output_dir),
        collection_prefix=cast(str, args.collection_prefix),
        k=cast(int, args.k),
        filters=parse_metadata_filters(cast(str | None, args.filters_json)),
        pdf_paths=pdf_paths,
        corpus_id=manifest.corpus_id if manifest else None,
    )
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
