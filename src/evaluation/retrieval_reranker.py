from __future__ import annotations

from collections.abc import Callable

from src.rag import retrieve
from src.reranking import bm25_scores, rerank_candidates, tokenize_for_reranking
from src.schemas import RetrievedChunk

__all__ = [
    "bm25_scores",
    "expanded_rerank_retriever",
    "rerank_candidates",
    "reranking_retriever",
    "tokenize_for_reranking",
]


def expanded_rerank_retriever(
    *,
    candidate_k: int = 50,
    dense_weight: float = 0.25,
    max_chunks_per_page: int | None = 1,
) -> Callable[..., list[RetrievedChunk]]:
    return reranking_retriever(
        retrieve,
        candidate_k=candidate_k,
        dense_weight=dense_weight,
        max_chunks_per_page=max_chunks_per_page,
    )


def reranking_retriever(
    base_retrieve_fn: Callable[..., list[RetrievedChunk]],
    *,
    candidate_k: int = 50,
    dense_weight: float = 0.25,
    max_chunks_per_page: int | None = 1,
) -> Callable[..., list[RetrievedChunk]]:
    def retrieve_and_rerank(
        query: str,
        *,
        k: int,
        filters: dict[str, object] | None,
        collection_name: str | None,
    ) -> list[RetrievedChunk]:
        if candidate_k < k:
            raise ValueError("candidate_k must be greater than or equal to k.")
        candidates = base_retrieve_fn(
            query,
            k=candidate_k,
            filters=filters,
            collection_name=collection_name,
        )
        return rerank_candidates(
            query,
            candidates,
            k=k,
            dense_weight=dense_weight,
            max_chunks_per_page=max_chunks_per_page,
        )

    return retrieve_and_rerank
