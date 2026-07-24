"""Reusable chunking strategies for controlled retrieval experiments."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.indexing import Chunker

DEFAULT_SEPARATORS = ("\n\n", "\n", ". ", " ", "")
RECURSIVE_CONFIGS = (
    ("rc_500_50", 500, 50),
    ("rc_800_100", 800, 100),
    ("rc_1000_150", 1000, 150),
    ("rc_1500_200", 1500, 200),
)


@dataclass(frozen=True, slots=True)
class RecursiveChunker:
    chunk_size: int = 500
    chunk_overlap: int = 50
    separators: tuple[str, ...] = DEFAULT_SEPARATORS

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size.")

    def split_documents(self, documents: list[Document]) -> list[Document]:
        if not documents:
            return []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=list(self.separators),
            is_separator_regex=False,
        )
        return splitter.split_documents(documents)


@dataclass(frozen=True, slots=True)
class ChunkingStrategy:
    strategy_id: str
    chunker: Chunker
    params: dict[str, int]


def recursive_strategies() -> list[ChunkingStrategy]:
    """Build the pre-registered recursive chunking strategy grid."""
    return [
        ChunkingStrategy(
            strategy_id=strategy_id,
            chunker=RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
            params={"chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
        )
        for strategy_id, chunk_size, chunk_overlap in RECURSIVE_CONFIGS
    ]
