"""Indexing integrity regression tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.config import settings
from src.embeddings import _embedding_device
from src.indexing import _document_id, _splitter, save_and_ingest_pdf


def test_document_id_changes_when_same_sized_content_changes(tmp_path: Path) -> None:
    """Given the same filename and byte length, different content must get different IDs."""
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first = first_dir / "document.pdf"
    second = second_dir / "document.pdf"
    first.write_bytes(b"%PDF-AAAA")
    second.write_bytes(b"%PDF-BBBB")

    assert _document_id(first) != _document_id(second)


def test_document_id_distinguishes_filenames_for_same_content(tmp_path: Path) -> None:
    """Given identical content, separate filenames must not overwrite each other's vectors."""
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    first.write_bytes(b"%PDF-same")
    second.write_bytes(b"%PDF-same")

    assert _document_id(first) != _document_id(second)


def test_splitter_preserves_explicit_zero_overlap() -> None:
    """Given zero overlap, the splitter must not silently restore the configured default."""
    splitter = _splitter(chunk_size=500, chunk_overlap=0)

    assert splitter._chunk_overlap == 0


def test_embeddings_use_cpu_on_hugging_face_spaces(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPACE_ID", "Jky71/doculearn-rag")

    assert _embedding_device() == "cpu"


def test_failed_upload_does_not_overwrite_existing_pdf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a bad replacement, the existing source PDF must remain untouched."""
    existing = tmp_path / "document.pdf"
    existing.write_bytes(b"%PDF-existing")
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr("src.indexing.ensure_collection", lambda **_: None)

    def fail_to_parse(*_: object, **__: object) -> list[object]:
        raise ValueError("invalid PDF")

    monkeypatch.setattr("src.indexing.build_chunks", fail_to_parse)

    with pytest.raises(ValueError, match="invalid PDF"):
        save_and_ingest_pdf(b"%PDF-broken", "document.pdf")

    assert existing.read_bytes() == b"%PDF-existing"


def test_successful_upload_returns_api_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful ingestion should return filename, document ID, and chunk count."""
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    chunk = Document(
        page_content="content",
        metadata={
            "document_id": "doc-123",
            "filename": "document.pdf",
            "source": "document.pdf",
            "page": 1,
            "chunk_id": "doc-123:1:0",
        },
    )
    monkeypatch.setattr("src.indexing.build_chunks", lambda *_args, **_kwargs: [chunk])
    monkeypatch.setattr("src.indexing.ensure_collection", lambda **_kwargs: None)
    monkeypatch.setattr("src.indexing.index_chunks", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr("src.indexing._remove_stale_versions", lambda *_args, **_kwargs: None)

    result = save_and_ingest_pdf(b"%PDF-valid", "document.pdf")

    assert result == {
        "filename": "document.pdf",
        "document_id": "doc-123",
        "chunks_indexed": 1,
    }
    assert (tmp_path / "document.pdf").read_bytes() == b"%PDF-valid"
