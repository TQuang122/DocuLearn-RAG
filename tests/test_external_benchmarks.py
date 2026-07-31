from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation.external_benchmarks import (
    ExternalBenchmarkQuery,
    ExternalCorpusDocument,
    convert_beir_scifact_directory,
    convert_xquad_vi_payload,
    load_external_benchmark,
    write_external_benchmark,
)


def test_convert_xquad_vi_preserves_context_answer_and_gold_document(tmp_path: Path) -> None:
    payload = {
        "data": [
            {
                "title": "Truy xuất thông tin",
                "paragraphs": [
                    {
                        "context": "RAG kết hợp truy xuất và sinh câu trả lời.",
                        "qas": [
                            {
                                "id": "vi-1",
                                "question": "RAG kết hợp những gì?",
                                "answers": [{"text": "truy xuất và sinh câu trả lời"}],
                            }
                        ],
                    }
                ],
            }
        ]
    }

    manifest = convert_xquad_vi_payload(payload, tmp_path)
    corpus, queries, loaded_manifest = load_external_benchmark(tmp_path)

    assert manifest["benchmark"] == "xquad-vi"
    assert loaded_manifest["queries"] == 1
    assert corpus == [
        ExternalCorpusDocument(
            id="xquad-vi-001-001",
            text="RAG kết hợp truy xuất và sinh câu trả lời.",
            title="Truy xuất thông tin",
        )
    ]
    assert queries == [
        ExternalBenchmarkQuery(
            id="xquad-vi-vi-1",
            question="RAG kết hợp những gì?",
            ground_truth="truy xuất và sinh câu trả lời",
            gold_document_ids=["xquad-vi-001-001"],
        )
    ]


def test_convert_beir_scifact_preserves_all_positive_qrels(tmp_path: Path) -> None:
    source_dir = tmp_path / "source" / "scifact"
    (source_dir / "qrels").mkdir(parents=True)
    (source_dir / "corpus.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"_id": "doc-1", "title": "One", "text": "Evidence one."}),
                json.dumps({"_id": "doc-2", "title": "Two", "text": "Evidence two."}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (source_dir / "queries.jsonl").write_text(
        json.dumps({"_id": "query-1", "text": "Which evidence is relevant?"}) + "\n",
        encoding="utf-8",
    )
    (source_dir / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nquery-1\tdoc-1\t1\nquery-1\tdoc-2\t1\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "output"
    convert_beir_scifact_directory(tmp_path / "source", output_dir)
    corpus, queries, _ = load_external_benchmark(output_dir)

    assert [document.id for document in corpus] == ["doc-1", "doc-2"]
    assert queries[0].gold_document_ids == ["doc-1", "doc-2"]
    assert queries[0].ground_truth == ""


def test_external_benchmark_rejects_gold_document_missing_from_corpus(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing from the corpus"):
        write_external_benchmark(
            tmp_path,
            benchmark="example",
            source="fixture",
            corpus=[ExternalCorpusDocument(id="doc-1", text="Known document")],
            queries=[
                ExternalBenchmarkQuery(
                    id="query-1",
                    question="Where is the evidence?",
                    gold_document_ids=["doc-missing"],
                )
            ],
        )
