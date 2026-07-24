"""Load PDFs, split into chunks with metadata, and index into Qdrant."""

from __future__ import annotations

import hashlib
import os
import tempfile
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Protocol

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from src.config import settings
from src.schemas import ChunkMetadata
from src.store import delete_old_document_versions, ensure_collection, get_vector_store

_PDF_SIGNATURE = b"%PDF-"


class Chunker(Protocol):
    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Split page-level documents into chunk-level documents."""
        ...


def _splitter(
    chunk_size: int | None = None, chunk_overlap: int | None = None
) -> RecursiveCharacterTextSplitter:
    size = settings.chunk_size if chunk_size is None else chunk_size
    overlap = settings.chunk_overlap if chunk_overlap is None else chunk_overlap

    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=False,
    )


def _document_id(path: Path, filename: str | None = None) -> str:
    """Build a stable identity from the logical filename and complete file content."""
    digest = hashlib.sha256()
    digest.update((filename or path.name).encode("utf-8"))
    digest.update(b"\0")
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()[:24]


def _chunk_id(doc_id: str, page: int, index: int) -> str:
    return f"{doc_id}:{page}:{index}"


def _load_pdf(
    path: Path,
    *,
    filename: str | None = None,
    source: str | None = None,
) -> list[Document]:
    loader = PyPDFLoader(str(path))
    pages = loader.load()
    logical_filename = filename or path.name
    doc_id = _document_id(path, logical_filename)
    for doc in pages:
        page_number = int(doc.metadata.get("page", 0)) + 1
        doc.metadata = {
            "document_id": doc_id,
            "filename": logical_filename,
            "source": source or logical_filename,
            "page": page_number,
            "section": doc.metadata.get("section"),
        }
    return pages


def discover_pdfs(data_dir: Path | None = None) -> list[Path]:
    directory = data_dir or settings.data_dir
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")


def build_chunks(
    pdf_paths: list[Path],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    chunker: Chunker | None = None,
    filename_override: str | None = None,
    source_override: str | None = None,
) -> list[Document]:
    page_docs: list[Document] = []
    for path in pdf_paths:
        logger.info("Loading PDF: {}", path.name)
        page_docs.extend(
            _load_pdf(
                path,
                filename=filename_override,
                source=source_override,
            )
        )

    if chunker is None:
        chunks = _splitter(chunk_size, chunk_overlap).split_documents(page_docs)
    else:
        chunks = chunker.split_documents(page_docs)

    per_doc_counter: dict[str, int] = defaultdict(int)
    for chunk in chunks:
        doc_id = chunk.metadata["document_id"]
        idx = per_doc_counter[doc_id]
        per_doc_counter[doc_id] += 1
        meta = ChunkMetadata(
            document_id=doc_id,
            filename=chunk.metadata["filename"],
            source=chunk.metadata["source"],
            page=chunk.metadata["page"],
            chunk_id=_chunk_id(doc_id, chunk.metadata["page"], idx),
            section=chunk.metadata.get("section"),
        )
        chunk.metadata = meta.model_dump()
    return chunks


def index_chunks(chunks: list[Document], collection_name: str | None = None) -> int:
    """Compute deterministic UUIDs and add chunks to the vector store.

    Re-ingesting the same content upserts instead of creating duplicates.
    """
    if not chunks:
        return 0
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, c.metadata["chunk_id"])) for c in chunks]
    get_vector_store(collection_name=collection_name).add_documents(chunks, ids=ids)
    return len(chunks)


def _remove_stale_versions(
    chunks: list[Document],
    collection_name: str | None = None,
) -> None:
    """Remove older vector versions after all replacement chunks are available."""
    current_versions = {
        (str(chunk.metadata["filename"]), str(chunk.metadata["document_id"])) for chunk in chunks
    }
    for filename, document_id in current_versions:
        delete_old_document_versions(
            filename,
            document_id,
            collection_name=collection_name,
        )


def ingest(
    recreate: bool = False,
    collection_name: str | None = None,
    chunker: Chunker | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> int:
    pdfs = discover_pdfs()
    if not pdfs:
        logger.warning("No PDF files found in {}", settings.data_dir)
        return 0

    ensure_collection(recreate=recreate, collection_name=collection_name)
    chunks = build_chunks(
        pdfs,
        chunker=chunker,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if not chunks:
        logger.warning("No chunks produced from {} PDF(s)", len(pdfs))
        return 0

    count = index_chunks(chunks, collection_name=collection_name)
    _remove_stale_versions(chunks, collection_name=collection_name)
    logger.info("Ingested {} chunks from {} PDF(s)", count, len(pdfs))
    return count


def save_and_ingest_pdf(file_bytes: bytes, filename: str) -> dict[str, object]:
    """Save an uploaded PDF to `data_dir` and ingest it into Qdrant.

    Args: file_bytes, filename. Returns: {"filename", "chunks_indexed"}. Raises: ValueError.
    """
    if not filename:
        raise ValueError("Filename is required.")
    if not filename.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are accepted.")
    if not file_bytes:
        raise ValueError("Uploaded file is empty.")
    if len(file_bytes) > settings.max_upload_bytes:
        raise ValueError(
            f"PDF exceeds the {settings.max_upload_bytes // (1024 * 1024)} MiB upload limit."
        )
    if not file_bytes.startswith(_PDF_SIGNATURE):
        raise ValueError("Uploaded content is not a PDF file.")

    safe_name = Path(filename).name
    if safe_name in {"", ".", ".."}:
        raise ValueError("Filename is invalid.")
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    dest = settings.data_dir / safe_name
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=settings.data_dir,
            prefix=".upload-",
            suffix=".pdf",
            delete=False,
        ) as temporary:
            temporary.write(file_bytes)
            temporary_path = Path(temporary.name)

        chunks = build_chunks(
            [temporary_path],
            filename_override=safe_name,
            source_override=safe_name,
        )
        if not chunks:
            raise ValueError("Uploaded PDF contains no indexable text.")

        ensure_collection(recreate=False)
        count = index_chunks(chunks)
        document_id = str(chunks[0].metadata["document_id"])
        os.replace(temporary_path, dest)
        temporary_path = None
        _remove_stale_versions(chunks)
        logger.info("Saved and indexed {} chunks from {}", count, safe_name)
        return {
            "filename": safe_name,
            "document_id": document_id,
            "chunks_indexed": count,
        }
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
