from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pypdf import PdfReader

from src.evaluation.evaluation_dataset import (
    RetrievalEvaluationRecord,
    load_evaluation_records,
)

TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
STOP_WORDS = {
    "ai",
    "bao",
    "bị",
    "các",
    "cái",
    "cho",
    "có",
    "của",
    "đâu",
    "để",
    "điều",
    "đó",
    "được",
    "gì",
    "giúp",
    "hai",
    "hay",
    "khi",
    "khác",
    "là",
    "làm",
    "mô",
    "một",
    "nào",
    "như",
    "những",
    "phần",
    "sao",
    "thế",
    "theo",
    "trong",
    "tại",
    "và",
    "vì",
    "với",
    "what",
    "which",
    "why",
}


class PdfCorpusDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    filename: str = Field(min_length=1)
    relative_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(gt=0)
    page_count: int = Field(gt=0)


class PdfCorpusManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    corpus_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    source_dir: str = Field(min_length=1)
    document_count: int = Field(gt=0)
    documents: list[PdfCorpusDocument] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_documents(self) -> PdfCorpusManifest:
        if self.document_count != len(self.documents):
            raise ValueError("document_count does not match documents.")
        filenames = [document.filename for document in self.documents]
        if len(filenames) != len(set(filenames)):
            raise ValueError("PDF filenames must be unique for source_file annotations.")
        if self.corpus_id != _corpus_id(self.documents):
            raise ValueError("corpus_id does not match manifest documents.")
        return self


class PdfPage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_file: str
    page: int = Field(gt=0)
    text: str


class AnnotationCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_file: str
    page: int = Field(gt=0)
    score: float = Field(gt=0)
    matched_terms: list[str]
    snippet: str


class ExistingAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_file: str | None
    gold_pages: list[int]
    gold_chunk_ids: list[str]
    answerable: bool


class AnnotationReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["pending", "approved", "unanswerable"] = "pending"
    source_file: str | None = None
    gold_pages: list[int] = Field(default_factory=list)
    question_type: str | None = None
    difficulty: Literal["easy", "medium", "hard"] | None = None
    notes: str = ""

    @field_validator("gold_pages")
    @classmethod
    def _normalize_pages(cls, pages: list[int]) -> list[int]:
        if any(page < 1 for page in pages):
            raise ValueError("gold_pages must contain positive page numbers.")
        return sorted(set(pages))

    @model_validator(mode="after")
    def _validate_status(self) -> AnnotationReview:
        if self.status == "approved" and (not self.source_file or not self.gold_pages):
            raise ValueError("Approved reviews require source_file and gold_pages.")
        if self.status == "unanswerable" and (self.source_file or self.gold_pages):
            raise ValueError("Unanswerable reviews cannot contain source annotations.")
        return self


class AnnotationQueueItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    ground_truth: str
    existing_annotation: ExistingAnnotation
    candidates: list[AnnotationCandidate]
    review: AnnotationReview = Field(default_factory=AnnotationReview)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _corpus_id(documents: list[PdfCorpusDocument]) -> str:
    digest = hashlib.sha256()
    for document in sorted(documents, key=lambda item: item.relative_path):
        digest.update(document.relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(document.sha256.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def build_pdf_corpus_manifest(pdf_dir: Path, output_path: Path) -> PdfCorpusManifest:
    pdf_paths = sorted(path for path in pdf_dir.rglob("*.pdf") if path.is_file())
    if not pdf_paths:
        raise ValueError(f"No PDF files found in {pdf_dir}")
    output_parent = output_path.parent.resolve()
    documents = [
        PdfCorpusDocument(
            filename=path.name,
            relative_path=Path(os.path.relpath(path.resolve(), output_parent)).as_posix(),
            sha256=_sha256(path),
            size_bytes=path.stat().st_size,
            page_count=len(PdfReader(str(path)).pages),
        )
        for path in pdf_paths
    ]
    manifest = PdfCorpusManifest(
        corpus_id=_corpus_id(documents),
        source_dir=Path(os.path.relpath(pdf_dir.resolve(), output_parent)).as_posix(),
        document_count=len(documents),
        documents=documents,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest.model_dump(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_pdf_corpus_paths(
    manifest_path: Path,
    *,
    verify: bool = True,
) -> tuple[PdfCorpusManifest, list[Path]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"PDF corpus manifest not found: {manifest_path}")
    manifest = PdfCorpusManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    paths = [
        (manifest_path.parent / document.relative_path).resolve() for document in manifest.documents
    ]
    if not verify:
        return manifest, paths
    for document, path in zip(manifest.documents, paths, strict=True):
        if not path.is_file():
            raise FileNotFoundError(f"Manifest PDF not found: {path}")
        if path.name != document.filename:
            raise ValueError(f"Manifest filename mismatch for {path}")
        if path.stat().st_size != document.size_bytes or _sha256(path) != document.sha256:
            raise ValueError(f"Manifest hash or size mismatch for {path}")
        if len(PdfReader(str(path)).pages) != document.page_count:
            raise ValueError(f"Manifest page count mismatch for {path}")
    return manifest, paths


def extract_pdf_pages(manifest_path: Path) -> list[PdfPage]:
    manifest, paths = load_pdf_corpus_paths(manifest_path)
    pages: list[PdfPage] = []
    for document, path in zip(manifest.documents, paths, strict=True):
        reader = PdfReader(str(path))
        pages.extend(
            PdfPage(
                source_file=document.filename,
                page=page_number,
                text=(page.extract_text() or "").strip(),
            )
            for page_number, page in enumerate(reader.pages, start=1)
        )
    return pages


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in TOKEN_PATTERN.findall(text.casefold())
        if len(token) > 1 and token not in STOP_WORDS and not token.isdigit()
    ]


def _snippet(text: str, terms: set[str], limit: int = 360) -> str:
    collapsed = " ".join(text.split())
    lowered = collapsed.casefold()
    positions = [lowered.find(term) for term in sorted(terms, key=len, reverse=True)]
    positions = [position for position in positions if position >= 0]
    center = min(positions) if positions else 0
    start = max(0, center - limit // 3)
    end = min(len(collapsed), start + limit)
    prefix = "…" if start else ""
    suffix = "…" if end < len(collapsed) else ""
    return f"{prefix}{collapsed[start:end].strip()}{suffix}"


def rank_annotation_candidates(
    record: RetrievalEvaluationRecord,
    pages: list[PdfPage],
    *,
    top_n: int = 5,
) -> list[AnnotationCandidate]:
    if top_n < 1:
        raise ValueError("top_n must be positive.")
    page_tokens = [Counter(_tokens(page.text)) for page in pages]
    document_frequency = Counter(
        token for token_counts in page_tokens for token in token_counts.keys()
    )
    question_counts = Counter(_tokens(record.question))
    answer_counts = Counter(_tokens(record.ground_truth))
    query_weights = {
        token: question_counts[token] * 2 + answer_counts[token]
        for token in question_counts.keys() | answer_counts.keys()
    }
    scored: list[AnnotationCandidate] = []
    for page, token_counts in zip(pages, page_tokens, strict=True):
        matched_terms = sorted(query_weights.keys() & token_counts.keys())
        if not matched_terms:
            continue
        raw_score = sum(
            query_weights[token]
            * (math.log((len(pages) + 1) / (document_frequency[token] + 1)) + 1)
            * (1 + math.log(token_counts[token]))
            for token in matched_terms
        )
        length_normalizer = math.sqrt(max(sum(token_counts.values()), 1) / 100)
        score = raw_score / max(length_normalizer, 1)
        scored.append(
            AnnotationCandidate(
                source_file=page.source_file,
                page=page.page,
                score=round(score, 6),
                matched_terms=matched_terms,
                snippet=_snippet(page.text, set(matched_terms)),
            )
        )
    return sorted(
        scored,
        key=lambda candidate: (-candidate.score, candidate.source_file, candidate.page),
    )[:top_n]


def build_annotation_queue(
    dataset_path: Path,
    manifest_path: Path,
    output_path: Path,
    *,
    top_n: int = 5,
) -> list[AnnotationQueueItem]:
    records = load_evaluation_records(dataset_path)
    pages = extract_pdf_pages(manifest_path)
    items = [
        AnnotationQueueItem(
            id=record.id,
            question=record.question,
            ground_truth=record.ground_truth,
            existing_annotation=ExistingAnnotation(
                source_file=record.source_file,
                gold_pages=record.gold_pages,
                gold_chunk_ids=record.gold_chunk_ids,
                answerable=record.answerable,
            ),
            candidates=rank_annotation_candidates(record, pages, top_n=top_n),
        )
        for record in records
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(item.model_dump(), ensure_ascii=False) + "\n" for item in items),
        encoding="utf-8",
    )
    return items


def load_annotation_queue(path: Path) -> list[AnnotationQueueItem]:
    if not path.exists():
        raise FileNotFoundError(f"Annotation queue not found: {path}")
    items = [
        AnnotationQueueItem.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [item.id for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("Annotation queue contains duplicate ids.")
    return items


def apply_annotation_queue(
    dataset_path: Path,
    manifest_path: Path,
    queue_path: Path,
    output_path: Path,
    report_path: Path,
) -> dict[str, object]:
    records = load_evaluation_records(dataset_path)
    manifest, _ = load_pdf_corpus_paths(manifest_path)
    queue = load_annotation_queue(queue_path)
    record_by_id = {record.id: record for record in records}
    queue_by_id = {item.id: item for item in queue}
    if record_by_id.keys() != queue_by_id.keys():
        missing = sorted(record_by_id.keys() - queue_by_id.keys())
        extra = sorted(queue_by_id.keys() - record_by_id.keys())
        raise ValueError(f"Annotation queue id mismatch; missing={missing}, extra={extra}")
    pending = sorted(item.id for item in queue if item.review.status == "pending")
    if pending:
        preview = ", ".join(pending[:10])
        suffix = ", …" if len(pending) > 10 else ""
        raise ValueError(
            f"Annotation queue still has {len(pending)} pending records: {preview}{suffix}"
        )

    pages_by_filename = {document.filename: document.page_count for document in manifest.documents}
    annotated: list[RetrievalEvaluationRecord] = []
    approved_count = 0
    unanswerable_count = 0
    for record in records:
        review = queue_by_id[record.id].review
        if review.status == "approved":
            if review.source_file not in pages_by_filename:
                raise ValueError(f"Unknown source_file for {record.id}: {review.source_file}")
            page_count = pages_by_filename[review.source_file]
            if any(page > page_count for page in review.gold_pages):
                raise ValueError(f"gold_pages exceed source page count for {record.id}")
            approved_count += 1
            annotated.append(
                record.model_copy(
                    update={
                        "source_file": review.source_file,
                        "gold_pages": review.gold_pages,
                        "gold_chunk_ids": [],
                        "answerable": True,
                        "question_type": review.question_type or record.question_type,
                        "difficulty": review.difficulty or record.difficulty,
                    }
                )
            )
        else:
            unanswerable_count += 1
            annotated.append(
                record.model_copy(
                    update={
                        "source_file": None,
                        "gold_pages": [],
                        "gold_chunk_ids": [],
                        "answerable": False,
                        "question_type": review.question_type or record.question_type,
                        "difficulty": review.difficulty or record.difficulty,
                    }
                )
            )

    validated = [
        RetrievalEvaluationRecord.model_validate(record.model_dump()) for record in annotated
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(record.model_dump(), ensure_ascii=False) + "\n" for record in validated),
        encoding="utf-8",
    )
    report: dict[str, object] = {
        "corpus_id": manifest.corpus_id,
        "input_dataset": str(dataset_path),
        "annotation_queue": str(queue_path),
        "output_dataset": str(output_path),
        "total_records": len(validated),
        "approved": approved_count,
        "unanswerable": unanswerable_count,
        "pending": 0,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage the PDF retrieval evaluation corpus.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser("build-manifest")
    manifest_parser.add_argument("--pdf-dir", type=Path, required=True)
    manifest_parser.add_argument("--output", type=Path, required=True)

    queue_parser = subparsers.add_parser("build-queue")
    queue_parser.add_argument("dataset", type=Path)
    queue_parser.add_argument("--manifest", type=Path, required=True)
    queue_parser.add_argument("--output", type=Path, required=True)
    queue_parser.add_argument("--top-n", type=int, default=5)

    apply_parser = subparsers.add_parser("apply-queue")
    apply_parser.add_argument("dataset", type=Path)
    apply_parser.add_argument("--manifest", type=Path, required=True)
    apply_parser.add_argument("--queue", type=Path, required=True)
    apply_parser.add_argument("--output", type=Path, required=True)
    apply_parser.add_argument("--report", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "build-manifest":
        result: object = build_pdf_corpus_manifest(args.pdf_dir, args.output).model_dump()
    elif args.command == "build-queue":
        result = {
            "records": len(
                build_annotation_queue(
                    args.dataset,
                    args.manifest,
                    args.output,
                    top_n=args.top_n,
                )
            ),
            "output": str(args.output),
        }
    else:
        result = apply_annotation_queue(
            args.dataset,
            args.manifest,
            args.queue,
            args.output,
            args.report,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
