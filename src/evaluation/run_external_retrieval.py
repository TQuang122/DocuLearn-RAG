from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Callable, Hashable
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, cast

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from numpy.typing import NDArray

from src.config import settings
from src.embeddings import get_embeddings
from src.evaluation.chunking_strategy import ChunkingStrategy, recursive_strategies
from src.evaluation.external_benchmarks import (
    ExternalBenchmarkQuery,
    ExternalCorpusDocument,
    load_external_benchmark,
)
from src.evaluation.retrieval_evaluator import (
    RECALL_CUTOFFS,
    RetrievalEvaluationResult,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    write_evaluation_results,
)
from src.schemas import ChunkMetadata, RetrievedChunk

ExternalRetrieveFunction = Callable[..., list[RetrievedChunk]]
ExternalRetrieverFactory = Callable[[list[Document]], ExternalRetrieveFunction]
EXTERNAL_EMBED_BATCH_SIZE = 256


def build_external_chunks(
    corpus: list[ExternalCorpusDocument],
    strategy: ChunkingStrategy,
) -> list[Document]:
    documents = [
        Document(
            page_content=document.text,
            metadata={
                "document_id": document.id,
                "filename": document.id,
                "source": document.title or document.id,
                "page": 1,
                "section": document.title,
            },
        )
        for document in corpus
    ]
    chunks = strategy.chunker.split_documents(documents)
    counters: dict[str, int] = defaultdict(int)
    for chunk in chunks:
        document_id = str(chunk.metadata["document_id"])
        chunk_index = counters[document_id]
        counters[document_id] += 1
        chunk.metadata = ChunkMetadata(
            document_id=document_id,
            filename=str(chunk.metadata["filename"]),
            source=str(chunk.metadata["source"]),
            page=1,
            chunk_id=f"{document_id}:1:{chunk_index}",
            section=chunk.metadata.get("section"),
        ).model_dump()
    return chunks


class ExactExternalRetriever:
    def __init__(
        self,
        chunks: list[Document],
        vectors: NDArray[np.float32],
        embeddings: Embeddings,
    ) -> None:
        self._chunks = chunks
        self._vectors = vectors
        self._embeddings = embeddings

    def __call__(
        self,
        query: str,
        *,
        k: int,
        filters: dict[str, object] | None,
        collection_name: str | None,
    ) -> list[RetrievedChunk]:
        del filters, collection_name
        query_vector = np.asarray(self._embeddings.embed_query(query), dtype=np.float32)
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            raise ValueError("Query embedding cannot be a zero vector.")
        query_vector = query_vector / query_norm
        scores = self._vectors @ query_vector
        result_count = min(k, len(self._chunks))
        if result_count == 0:
            return []
        candidate_indices = np.argpartition(-scores, result_count - 1)[:result_count]
        ranked_indices = candidate_indices[np.argsort(-scores[candidate_indices])]
        return [
            RetrievedChunk(
                text=self._chunks[int(index)].page_content,
                score=float(scores[int(index)]),
                metadata=ChunkMetadata(**self._chunks[int(index)].metadata),
            )
            for index in ranked_indices
        ]


def build_exact_retriever(
    chunks: list[Document],
    *,
    batch_size: int = EXTERNAL_EMBED_BATCH_SIZE,
) -> ExactExternalRetriever:
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if not chunks:
        raise ValueError("chunks cannot be empty.")
    embeddings = get_embeddings()
    vector_batches = [
        embeddings.embed_documents(
            [chunk.page_content for chunk in chunks[offset : offset + batch_size]]
        )
        for offset in range(0, len(chunks), batch_size)
    ]
    vectors = np.asarray(
        [vector for batch in vector_batches for vector in batch],
        dtype=np.float32,
    )
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Corpus embeddings cannot contain zero vectors.")
    vectors = vectors / norms
    return ExactExternalRetriever(chunks, vectors, embeddings)


def _retrieved_payload(chunks: list[RetrievedChunk]) -> list[dict[str, object]]:
    return [
        {
            "rank": rank,
            "score": chunk.score,
            "document_id": chunk.metadata.document_id,
            "chunk_id": chunk.metadata.chunk_id,
            "text": chunk.text,
        }
        for rank, chunk in enumerate(chunks, start=1)
    ]


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def evaluate_external_queries(
    queries: list[ExternalBenchmarkQuery],
    *,
    collection_name: str,
    k: int = 10,
    retrieve_fn: ExternalRetrieveFunction,
) -> RetrievalEvaluationResult:
    if not queries:
        raise ValueError("queries cannot be empty.")
    if k < max(RECALL_CUTOFFS):
        raise ValueError(f"k must be at least {max(RECALL_CUTOFFS)}.")

    cases: list[dict[str, Any]] = []
    for query in queries:
        started_at = perf_counter()
        chunks = retrieve_fn(
            query.question,
            k=k,
            filters=None,
            collection_name=collection_name,
        )
        latency_ms = (perf_counter() - started_at) * 1000
        retrieved_ids = [chunk.metadata.document_id for chunk in chunks]
        gold_ids: set[Hashable] = set(query.gold_document_ids)
        metrics = {
            **{
                f"recall@{cutoff}": recall_at_k(retrieved_ids, gold_ids, cutoff)
                for cutoff in RECALL_CUTOFFS
            },
            "mrr": reciprocal_rank(retrieved_ids[:k], gold_ids),
            "ndcg": ndcg_at_k(retrieved_ids, gold_ids, k),
        }
        cases.append(
            {
                "id": query.id,
                "question": query.question,
                "answerable": query.answerable,
                "status": "evaluated",
                "relevance_mode": "document_id",
                "latency_ms": latency_ms,
                "gold_document_ids": query.gold_document_ids,
                "retrieved": _retrieved_payload(chunks),
                "metrics": metrics,
            }
        )

    metric_names = [*(f"recall@{cutoff}" for cutoff in RECALL_CUTOFFS), "mrr", "ndcg"]
    latencies = [float(case["latency_ms"]) for case in cases]
    summary = {
        "collection": collection_name,
        "k": k,
        "filters": None,
        "counts": {
            "total": len(cases),
            "answerable": sum(query.answerable for query in queries),
            "evaluated_answerable": len(cases),
            "annotation_required": 0,
            "unanswerable": 0,
        },
        "metrics": {
            name: mean(float(case["metrics"][name]) for case in cases) for name in metric_names
        },
        "retrieval_latency_ms": {
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
        },
    }
    return RetrievalEvaluationResult(cases=cases, summary=summary)


def run_external_baselines(
    benchmark_dir: Path,
    output_dir: Path,
    *,
    collection_prefix: str,
    k: int = 10,
    limit: int | None = None,
    retriever_factory: ExternalRetrieverFactory = build_exact_retriever,
) -> list[dict[str, object]]:
    corpus, queries, manifest = load_external_benchmark(benchmark_dir)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive.")
        queries = queries[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, object]] = []
    for strategy in recursive_strategies():
        collection_name = f"{collection_prefix}__{strategy.strategy_id}"
        chunks = build_external_chunks(corpus, strategy)
        retrieve_fn = retriever_factory(chunks)
        result = evaluate_external_queries(
            queries,
            collection_name=collection_name,
            k=k,
            retrieve_fn=retrieve_fn,
        )
        write_evaluation_results(result, output_dir / strategy.strategy_id)
        reports.append(
            {
                "benchmark": manifest["benchmark"],
                "strategy_id": strategy.strategy_id,
                "params": strategy.params,
                "corpus_documents": len(corpus),
                "chunk_count": len(chunks),
                "query_count": len(queries),
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
        description="Run an LLM-free external retrieval benchmark across all chunking strategies."
    )
    parser.add_argument("benchmark_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--collection-prefix",
        default=f"{settings.qdrant_collection}__external_eval",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    reports = run_external_baselines(
        cast(Path, args.benchmark_dir),
        cast(Path, args.output_dir),
        collection_prefix=cast(str, args.collection_prefix),
        k=cast(int, args.k),
        limit=cast(int | None, args.limit),
    )
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
