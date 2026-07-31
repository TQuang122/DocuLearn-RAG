from __future__ import annotations

import json
from pathlib import Path

import pytest
from pypdf import PdfWriter

from src.evaluation.evaluation_dataset import (
    RetrievalEvaluationRecord,
    load_evaluation_records,
)
from src.evaluation.pdf_corpus import (
    AnnotationReview,
    PdfPage,
    apply_annotation_queue,
    build_annotation_queue,
    build_pdf_corpus_manifest,
    load_annotation_queue,
    load_pdf_corpus_paths,
    rank_annotation_candidates,
)


def _write_pdf(path: Path, pages: int) -> None:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=612, height=792)
    with path.open("wb") as output_file:
        writer.write(output_file)


def _write_dataset(path: Path) -> list[RetrievalEvaluationRecord]:
    records = [
        RetrievalEvaluationRecord(
            id="qa-001",
            question="LoRA hoạt động như thế nào?",
            ground_truth="LoRA sử dụng ma trận hạng thấp.",
            source_file=None,
            gold_pages=[],
            gold_chunk_ids=[],
            answerable=True,
            question_type="definition",
            difficulty="easy",
        ),
        RetrievalEvaluationRecord(
            id="qa-002",
            question="Thông tin không tồn tại?",
            ground_truth="Không có trong corpus.",
            source_file=None,
            gold_pages=[],
            gold_chunk_ids=[],
            answerable=True,
            question_type="factual",
            difficulty="medium",
        ),
    ]
    path.write_text(
        "".join(json.dumps(record.model_dump(), ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return records


def test_manifest_is_portable_and_detects_pdf_changes(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    _write_pdf(pdf_dir / "paper.pdf", 2)
    manifest_path = tmp_path / "evaluation" / "manifest.json"

    manifest = build_pdf_corpus_manifest(pdf_dir, manifest_path)
    loaded, paths = load_pdf_corpus_paths(manifest_path)

    assert loaded == manifest
    assert loaded.document_count == 1
    assert loaded.documents[0].page_count == 2
    assert paths == [(pdf_dir / "paper.pdf").resolve()]

    _write_pdf(pdf_dir / "paper.pdf", 1)
    with pytest.raises(ValueError, match="hash or size mismatch"):
        load_pdf_corpus_paths(manifest_path)


def test_candidate_ranking_returns_evidence_without_changing_gold_metadata() -> None:
    record = RetrievalEvaluationRecord(
        id="qa-001",
        question="LoRA dùng ma trận nào?",
        ground_truth="LoRA sử dụng ma trận hạng thấp.",
        source_file=None,
        gold_pages=[],
        gold_chunk_ids=[],
        answerable=True,
        question_type="definition",
        difficulty="easy",
    )
    pages = [
        PdfPage(source_file="other.pdf", page=1, text="Transformer attention architecture."),
        PdfPage(
            source_file="lora.pdf",
            page=4,
            text="LoRA biểu diễn cập nhật bằng hai ma trận hạng thấp.",
        ),
    ]

    candidates = rank_annotation_candidates(record, pages, top_n=2)

    assert candidates[0].source_file == "lora.pdf"
    assert candidates[0].page == 4
    assert "lora" in candidates[0].matched_terms
    assert record.source_file is None
    assert record.gold_pages == []


def test_apply_queue_requires_complete_human_review(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    _write_pdf(pdf_dir / "paper.pdf", 2)
    manifest_path = tmp_path / "manifest.json"
    build_pdf_corpus_manifest(pdf_dir, manifest_path)
    dataset_path = tmp_path / "dataset.jsonl"
    _write_dataset(dataset_path)
    queue_path = tmp_path / "queue.jsonl"
    build_annotation_queue(dataset_path, manifest_path, queue_path)

    with pytest.raises(ValueError, match="2 pending records"):
        apply_annotation_queue(
            dataset_path,
            manifest_path,
            queue_path,
            tmp_path / "annotated.jsonl",
            tmp_path / "report.json",
        )

    queue = load_annotation_queue(queue_path)
    queue[0].review = AnnotationReview(
        status="approved",
        source_file="paper.pdf",
        gold_pages=[2],
        notes="Verified manually.",
    )
    queue[1].review = AnnotationReview(status="unanswerable", notes="Verified absent.")
    queue_path.write_text(
        "".join(json.dumps(item.model_dump(), ensure_ascii=False) + "\n" for item in queue),
        encoding="utf-8",
    )

    report = apply_annotation_queue(
        dataset_path,
        manifest_path,
        queue_path,
        tmp_path / "annotated.jsonl",
        tmp_path / "report.json",
    )
    records = load_evaluation_records(tmp_path / "annotated.jsonl")

    assert report["approved"] == 1
    assert report["unanswerable"] == 1
    assert records[0].source_file == "paper.pdf"
    assert records[0].gold_pages == [2]
    assert records[1].answerable is False
