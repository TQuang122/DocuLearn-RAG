"""Local embeddings via Sentence-Transformers (no inference provider)."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import cast

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from src.config import settings


def _embedding_device() -> str | None:
    return "cpu" if os.getenv("SPACE_ID") else None


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model, device=_embedding_device())


class LocalSentenceTransformerEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:  # type: ignore[override]
        vecs = _model().encode(texts, normalize_embeddings=True)
        tolist = getattr(vecs, "tolist", None)
        if callable(tolist):
            return cast(list[list[float]], tolist())
        return [list(map(float, v)) for v in vecs]

    def embed_query(self, text: str) -> list[float]:  # type: ignore[override]
        vec = _model().encode([text], normalize_embeddings=True)[0]
        tolist = getattr(vec, "tolist", None)
        if callable(tolist):
            return cast(list[float], tolist())
        return [float(x) for x in vec]


def get_embeddings() -> Embeddings:
    return LocalSentenceTransformerEmbeddings()
