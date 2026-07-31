from __future__ import annotations

import json
from pathlib import Path

from langchain_core.embeddings import Embeddings

from src.evaluation import run_external_retrieval
from src.evaluation.chunking_strategy import recursive_strategies
from src.evaluation.external_benchmarks import (
    ExternalBenchmarkQuery,
    ExternalCorpusDocument,
    write_external_benchmark,
)
from src.evaluation.run_external_retrieval import (
    build_exact_retriever,
    build_external_chunks,
    evaluate_external_queries,
)
from src.schemas import ChunkMetadata, RetrievedChunk


def _retrieved(document_id: str, chunk_index: int = 0) -> RetrievedChunk:
    return RetrievedChunk(
        text=f"Text from {document_id}",
        score=1.0,
        metadata=ChunkMetadata(
            document_id=document_id,
            filename=document_id,
            source=document_id,
            page=1,
            chunk_id=f"{document_id}:1:{chunk_index}",
        ),
    )


def test_build_external_chunks_retains_document_identity() -> None:
    corpus = [ExternalCorpusDocument(id="doc-1", text="A " * 600, title="Title")]

    chunks = build_external_chunks(corpus, recursive_strategies()[0])

    assert len(chunks) > 1
    assert {chunk.metadata["document_id"] for chunk in chunks} == {"doc-1"}
    assert [chunk.metadata["chunk_id"] for chunk in chunks] == [
        f"doc-1:1:{index}" for index in range(len(chunks))
    ]


def test_external_query_evaluation_scores_document_level_relevance() -> None:
    query = ExternalBenchmarkQuery(
        id="query-1",
        question="Find the relevant document",
        gold_document_ids=["gold-1", "gold-2"],
    )

    def fake_retrieve(*args: object, **kwargs: object) -> list[RetrievedChunk]:
        return [_retrieved("wrong"), _retrieved("gold-2"), _retrieved("gold-2", 1)]

    result = evaluate_external_queries(
        [query],
        collection_name="external",
        k=10,
        retrieve_fn=fake_retrieve,
    )

    assert result.cases[0]["relevance_mode"] == "document_id"
    assert result.cases[0]["metrics"]["recall@1"] == 0.0
    assert result.cases[0]["metrics"]["recall@3"] == 0.5
    assert result.cases[0]["metrics"]["mrr"] == 0.5
    assert result.summary["metrics"]["recall@3"] == 0.5


def test_exact_retriever_embeds_in_bounded_batches(monkeypatch) -> None:
    chunks = build_external_chunks(
        [ExternalCorpusDocument(id="doc-1", text="A " * 600)],
        recursive_strategies()[0],
    )
    batch_sizes: list[int] = []

    class FakeEmbeddings(Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            batch_sizes.append(len(texts))
            return [[float(index + 1), 0.0] for index, _ in enumerate(texts)]

        def embed_query(self, text: str) -> list[float]:
            return [1.0, 0.0]

    monkeypatch.setattr(run_external_retrieval, "get_embeddings", FakeEmbeddings)

    retriever = build_exact_retriever(chunks, batch_size=2)
    results = retriever("query", k=1, filters=None, collection_name=None)

    assert len(results) == 1
    assert batch_sizes == [2] * (len(chunks) // 2) + ([1] if len(chunks) % 2 else [])


def test_external_baseline_uses_all_strategies(tmp_path: Path, monkeypatch) -> None:
    benchmark_dir = tmp_path / "benchmark"
    write_external_benchmark(
        benchmark_dir,
        benchmark="fixture",
        source="fixture",
        corpus=[ExternalCorpusDocument(id="doc-1", text="Relevant evidence")],
        queries=[
            ExternalBenchmarkQuery(
                id="query-1",
                question="What is relevant?",
                gold_document_ids=["doc-1"],
            )
        ],
    )
    built_chunk_counts: list[int] = []

    def fake_retriever_factory(chunks: list[object]):
        built_chunk_counts.append(len(chunks))
        return lambda *args, **kwargs: [_retrieved("doc-1")]

    reports = run_external_retrieval.run_external_baselines(
        benchmark_dir,
        tmp_path / "results",
        collection_prefix="external",
        k=10,
        retriever_factory=fake_retriever_factory,
    )

    assert len(built_chunk_counts) == 4
    assert all(report["summary"]["metrics"]["recall@1"] == 1.0 for report in reports)
    assert json.loads((tmp_path / "results" / "baseline_summary.json").read_text())
