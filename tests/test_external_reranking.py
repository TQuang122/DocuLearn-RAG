from __future__ import annotations

from pathlib import Path

from src.evaluation.external_benchmarks import (
    ExternalBenchmarkQuery,
    ExternalCorpusDocument,
    write_external_benchmark,
)
from src.evaluation.run_external_reranking import run_external_reranking
from src.schemas import ChunkMetadata, RetrievedChunk


def test_external_reranking_writes_report_and_expands_candidates(tmp_path: Path) -> None:
    benchmark_dir = tmp_path / "benchmark"
    write_external_benchmark(
        benchmark_dir,
        benchmark="fixture",
        source="fixture",
        corpus=[ExternalCorpusDocument(id="doc-1", text="Exact lexical evidence")],
        queries=[
            ExternalBenchmarkQuery(
                id="query-1",
                question="lexical evidence",
                gold_document_ids=["doc-1"],
            )
        ],
    )
    requested_k: list[int] = []

    def fake_factory(chunks: list[object]):
        del chunks

        def fake_retrieve(*args: object, **kwargs: object) -> list[RetrievedChunk]:
            del args
            requested_k.append(int(kwargs["k"]))
            return [
                RetrievedChunk(
                    text="Exact lexical evidence",
                    score=0.5,
                    metadata=ChunkMetadata(
                        document_id="doc-1",
                        filename="doc-1",
                        source="doc-1",
                        page=1,
                        chunk_id="doc-1:1:0",
                    ),
                )
            ]

        return fake_retrieve

    output_dir = tmp_path / "results"
    report = run_external_reranking(
        benchmark_dir,
        output_dir,
        candidate_k=50,
        retriever_factory=fake_factory,
    )

    assert requested_k == [50]
    assert report["summary"]["metrics"]["recall@1"] == 1.0
    assert report["summary"]["variant"]["dense_weight"] == 0.25
    assert (output_dir / "cases.jsonl").exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "run_summary.json").exists()
