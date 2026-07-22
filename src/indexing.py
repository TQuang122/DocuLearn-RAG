import hashlib
import uuid
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from schemas import ChunkMetadata
from store import ensure_collection, get_vector_store


def _document_id(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(block)

    return hasher.hexdigest()[:16]


def _chunk_id(
    document_id: str,
    page: int,
    index: int,
) -> str:
    return f"{document_id}:{page}:{index}"


def discover_pdfs() -> list[Path]:
    """
    Tìm tất cả PDF trong thư mục dữ liệu.
    """
    settings.data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return sorted(
        path
        for path in settings.data_dir.rglob("*.pdf")
        if path.is_file()
    )


def _load_pdf(path: Path) -> list[Document]:
    path = path.expanduser()

    if not path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Path is not a file: {path}"
        )

    if path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Expected a PDF file, received: {path.suffix}"
        )

    pages = PyPDFLoader(str(path)).load()
    document_id = _document_id(path)

    for document in pages:
        original_page = document.metadata.get("page", 0)

        try:
            page_number = int(original_page) + 1
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid page metadata in PDF {path}: "
                f"{original_page!r}"
            ) from error

        document.metadata.update(
            {
                "document_id": document_id,
                "filename": path.name,
                "source": str(path.resolve()),
                "page": page_number,
                "section": document.metadata.get("section"),
            }
        )

    return pages


def _splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    resolved_chunk_size = (
        settings.chunk_size
        if chunk_size is None
        else chunk_size
    )

    resolved_chunk_overlap = (
        settings.chunk_overlap
        if chunk_overlap is None
        else chunk_overlap
    )

    if resolved_chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0."
        )

    if resolved_chunk_overlap < 0:
        raise ValueError(
            "chunk_overlap cannot be negative."
        )

    if resolved_chunk_overlap >= resolved_chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    return RecursiveCharacterTextSplitter(
        chunk_size=resolved_chunk_size,
        chunk_overlap=resolved_chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
        keep_separator=False,
    )


def build_chunks(
    pdf_paths: Iterable[Path | str],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    chunker: RecursiveCharacterTextSplitter | None = None,
) -> list[Document]:
    if chunker is not None and (
        chunk_size is not None
        or chunk_overlap is not None
    ):
        raise ValueError(
            "Pass either chunker or chunk_size/chunk_overlap, "
            "not both."
        )

    page_documents: list[Document] = []

    for path in pdf_paths:
        page_documents.extend(
            _load_pdf(Path(path))
        )

    if not page_documents:
        return []

    splitter = chunker or _splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = splitter.split_documents(page_documents)

    # Mỗi tài liệu và mỗi trang có bộ đếm chunk riêng.
    per_page_counter: defaultdict[
        tuple[str, int],
        int,
    ] = defaultdict(int)

    for chunk in chunks:
        document_id = str(
            chunk.metadata["document_id"]
        )
        page = int(chunk.metadata["page"])
        counter_key = (document_id, page)

        index = per_page_counter[counter_key]
        per_page_counter[counter_key] += 1

        metadata = ChunkMetadata(
            document_id=document_id,
            filename=str(chunk.metadata["filename"]),
            source=str(chunk.metadata["source"]),
            page=page,
            chunk_id=_chunk_id(
                document_id=document_id,
                page=page,
                index=index,
            ),
            section=chunk.metadata.get("section"),
        )

        chunk.metadata = metadata.model_dump(
            exclude_none=True
        )

    return chunks


def index_chunks(
    chunks: Sequence[Document],
    collection_name: str | None = None,
) -> int:
    if not chunks:
        return 0

    ids = [
        str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                str(chunk.metadata["chunk_id"]),
            )
        )
        for chunk in chunks
    ]

    vector_store = get_vector_store(
        collection_name=collection_name
    )

    vector_store.add_documents(
        documents=list(chunks),
        ids=ids,
    )

    return len(chunks)


def ingest(
    recreate: bool = False,
    collection_name: str | None = None,
    chunker: RecursiveCharacterTextSplitter | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> int:
    pdfs = discover_pdfs()

    ensure_collection(
        recreate=recreate,
        collection_name=collection_name,
    )

    chunks = build_chunks(
        pdf_paths=pdfs,
        chunker=chunker,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return index_chunks(
        chunks=chunks,
        collection_name=collection_name,
    )


def save_and_ingest_pdf(
    file_bytes: bytes,
    filename: str,
    collection_name: str | None = None,
) -> dict[str, str | int]:
    if not file_bytes:
        raise ValueError("Uploaded file is empty.")

    safe_name = Path(filename).name

    if not safe_name:
        raise ValueError("Filename cannot be empty.")

    if Path(safe_name).suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    settings.data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination = settings.data_dir / safe_name
    destination.write_bytes(file_bytes)

    ensure_collection(
        recreate=False,
        collection_name=collection_name,
    )

    chunks = build_chunks([destination])

    chunks_indexed = index_chunks(
        chunks=chunks,
        collection_name=collection_name,
    )

    return {
        "filename": safe_name,
        "document_id": _document_id(destination),
        "chunks_indexed": chunks_indexed,
    }