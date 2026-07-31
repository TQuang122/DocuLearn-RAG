from __future__ import annotations

import argparse
import csv
import json
import tempfile
import urllib.request
import zipfile
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

XQUAD_VI_URL = "https://raw.githubusercontent.com/google-deepmind/xquad/master/xquad.vi.json"
BEIR_SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"


class ExternalCorpusDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    title: str | None = None

    @field_validator("id", "text")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Value cannot be blank.")
        return stripped

    @field_validator("title")
    @classmethod
    def _strip_title(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None


class ExternalBenchmarkQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    ground_truth: str = ""
    gold_document_ids: list[str] = Field(min_length=1)
    answerable: bool = True

    @field_validator("id", "question")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Value cannot be blank.")
        return stripped

    @field_validator("ground_truth")
    @classmethod
    def _strip_ground_truth(cls, value: str) -> str:
        return value.strip()

    @field_validator("gold_document_ids")
    @classmethod
    def _normalize_gold_ids(cls, values: list[str]) -> list[str]:
        normalized = list(dict.fromkeys(value.strip() for value in values if value.strip()))
        if not normalized:
            raise ValueError("gold_document_ids cannot be empty.")
        return normalized


def _write_jsonl(path: Path, records: Sequence[BaseModel]) -> None:
    path.write_text(
        "".join(json.dumps(record.model_dump(), ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def write_external_benchmark(
    output_dir: Path,
    *,
    benchmark: str,
    source: str,
    corpus: list[ExternalCorpusDocument],
    queries: list[ExternalBenchmarkQuery],
) -> dict[str, object]:
    if not corpus:
        raise ValueError("External benchmark corpus cannot be empty.")
    if not queries:
        raise ValueError("External benchmark queries cannot be empty.")

    _validate_gold_documents(corpus, queries)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "corpus.jsonl", corpus)
    _write_jsonl(output_dir / "queries.jsonl", queries)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "benchmark": benchmark,
        "source": source,
        "relevance_level": "document",
        "corpus_documents": len(corpus),
        "queries": len(queries),
        "answerable_queries": sum(query.answerable for query in queries),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _validate_gold_documents(
    corpus: list[ExternalCorpusDocument],
    queries: list[ExternalBenchmarkQuery],
) -> None:
    corpus_ids = {document.id for document in corpus}
    missing_gold = sorted(
        {
            document_id
            for query in queries
            for document_id in query.gold_document_ids
            if document_id not in corpus_ids
        }
    )
    if missing_gold:
        preview = ", ".join(missing_gold[:5])
        raise ValueError(f"Gold document ids are missing from the corpus: {preview}")


def load_external_benchmark(
    benchmark_dir: Path,
) -> tuple[list[ExternalCorpusDocument], list[ExternalBenchmarkQuery], dict[str, Any]]:
    corpus_path = benchmark_dir / "corpus.jsonl"
    queries_path = benchmark_dir / "queries.jsonl"
    manifest_path = benchmark_dir / "manifest.json"
    for path in (corpus_path, queries_path, manifest_path):
        if not path.exists():
            raise FileNotFoundError(f"External benchmark artifact not found: {path}")

    corpus = [
        ExternalCorpusDocument.model_validate(json.loads(line))
        for line in corpus_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    queries = [
        ExternalBenchmarkQuery.model_validate(json.loads(line))
        for line in queries_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest_payload, dict):
        raise ValueError("External benchmark manifest must contain an object.")
    if not corpus or not queries:
        raise ValueError("External benchmark artifacts cannot be empty.")
    _validate_gold_documents(corpus, queries)
    return corpus, queries, manifest_payload


def convert_xquad_vi_payload(payload: dict[str, Any], output_dir: Path) -> dict[str, object]:
    corpus: list[ExternalCorpusDocument] = []
    queries: list[ExternalBenchmarkQuery] = []
    articles = payload.get("data")
    if not isinstance(articles, list):
        raise ValueError("XQuAD payload must contain a data list.")

    for article_index, raw_article in enumerate(articles, start=1):
        if not isinstance(raw_article, dict):
            raise ValueError("XQuAD articles must be objects.")
        title = str(raw_article.get("title") or "").strip() or None
        paragraphs = raw_article.get("paragraphs")
        if not isinstance(paragraphs, list):
            raise ValueError("XQuAD article must contain a paragraphs list.")
        for paragraph_index, raw_paragraph in enumerate(paragraphs, start=1):
            if not isinstance(raw_paragraph, dict):
                raise ValueError("XQuAD paragraphs must be objects.")
            context = str(raw_paragraph.get("context") or "").strip()
            document_id = f"xquad-vi-{article_index:03d}-{paragraph_index:03d}"
            corpus.append(ExternalCorpusDocument(id=document_id, text=context, title=title))
            qas = raw_paragraph.get("qas")
            if not isinstance(qas, list):
                raise ValueError("XQuAD paragraph must contain a qas list.")
            for raw_qa in qas:
                if not isinstance(raw_qa, dict):
                    raise ValueError("XQuAD QA entries must be objects.")
                answers = raw_qa.get("answers")
                if not isinstance(answers, list) or not answers or not isinstance(answers[0], dict):
                    raise ValueError("XQuAD QA entry must contain at least one answer.")
                queries.append(
                    ExternalBenchmarkQuery(
                        id=f"xquad-vi-{raw_qa.get('id')}",
                        question=str(raw_qa.get("question") or ""),
                        ground_truth=str(answers[0].get("text") or ""),
                        gold_document_ids=[document_id],
                    )
                )

    return write_external_benchmark(
        output_dir,
        benchmark="xquad-vi",
        source=XQUAD_VI_URL,
        corpus=corpus,
        queries=queries,
    )


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected object in {path} at line {line_number}.")
        records.append(payload)
    return records


def convert_beir_scifact_directory(source_dir: Path, output_dir: Path) -> dict[str, object]:
    corpus_paths = list(source_dir.rglob("corpus.jsonl"))
    query_paths = list(source_dir.rglob("queries.jsonl"))
    qrels_paths = list(source_dir.rglob("qrels/test.tsv"))
    if len(corpus_paths) != 1 or len(query_paths) != 1 or len(qrels_paths) != 1:
        raise ValueError(
            "SciFact source must contain corpus.jsonl, queries.jsonl, and qrels/test.tsv."
        )

    corpus = [
        ExternalCorpusDocument(
            id=str(record.get("_id") or ""),
            title=str(record.get("title") or "") or None,
            text="\n\n".join(
                part
                for part in (
                    str(record.get("title") or "").strip(),
                    str(record.get("text") or "").strip(),
                )
                if part
            ),
        )
        for record in _read_jsonl_objects(corpus_paths[0])
    ]
    query_text = {
        str(record.get("_id") or ""): str(record.get("text") or "")
        for record in _read_jsonl_objects(query_paths[0])
    }
    qrels: dict[str, list[str]] = defaultdict(list)
    with qrels_paths[0].open(encoding="utf-8", newline="") as input_file:
        for row in csv.DictReader(input_file, delimiter="\t"):
            if int(row["score"]) > 0:
                qrels[row["query-id"]].append(row["corpus-id"])
    queries = [
        ExternalBenchmarkQuery(
            id=f"beir-scifact-{query_id}",
            question=query_text[query_id],
            gold_document_ids=gold_ids,
        )
        for query_id, gold_ids in sorted(qrels.items())
        if query_id in query_text
    ]
    return write_external_benchmark(
        output_dir,
        benchmark="beir-scifact",
        source=BEIR_SCIFACT_URL,
        corpus=corpus,
        queries=queries,
    )


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "DocuLearn-RAG-evaluator/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
        destination.write_bytes(response.read())


def _safe_extract(archive: Path, destination: Path) -> None:
    destination_root = destination.resolve()
    with zipfile.ZipFile(archive) as zip_file:
        for member in zip_file.infolist():
            member_path = (destination / member.filename).resolve()
            if not member_path.is_relative_to(destination_root):
                raise ValueError(f"Unsafe path in benchmark archive: {member.filename}")
        zip_file.extractall(destination)


def prepare_xquad_vi(output_dir: Path, source: Path | None = None) -> dict[str, object]:
    if source is None:
        with tempfile.TemporaryDirectory(prefix="doculearn-xquad-") as temp_dir:
            source = Path(temp_dir) / "xquad.vi.json"
            _download(XQUAD_VI_URL, source)
            payload = json.loads(source.read_text(encoding="utf-8"))
    else:
        payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("XQuAD source must contain a JSON object.")
    return convert_xquad_vi_payload(payload, output_dir)


def prepare_beir_scifact(output_dir: Path, source: Path | None = None) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="doculearn-scifact-") as temp_dir:
        temp_path = Path(temp_dir)
        if source is None:
            archive = temp_path / "scifact.zip"
            _download(BEIR_SCIFACT_URL, archive)
            _safe_extract(archive, temp_path / "source")
            source_dir = temp_path / "source"
        elif source.is_file():
            _safe_extract(source, temp_path / "source")
            source_dir = temp_path / "source"
        else:
            source_dir = source
        return convert_beir_scifact_directory(source_dir, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a public retrieval benchmark.")
    parser.add_argument("benchmark", choices=("xquad-vi", "beir-scifact"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    if args.benchmark == "xquad-vi":
        manifest = prepare_xquad_vi(args.output_dir, args.source)
    else:
        manifest = prepare_beir_scifact(args.output_dir, args.source)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
