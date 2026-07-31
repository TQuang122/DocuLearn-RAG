from __future__ import annotations

import pytest

from src.evaluation import retrieval_reranker
from src.evaluation.retrieval_reranker import (
    bm25_scores,
    expanded_rerank_retriever,
    rerank_candidates,
    reranking_retriever,
)
from src.schemas import ChunkMetadata, RetrievedChunk


def _chunk(text: str, score: float, *, page: int, chunk_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        score=score,
        metadata=ChunkMetadata(
            document_id="doc",
            filename="source.pdf",
            source="source.pdf",
            page=page,
            chunk_id=chunk_id,
        ),
    )


def test_bm25_scores_prefer_exact_query_terms() -> None:
    chunks = [
        _chunk("unrelated transformer text", 0.9, page=1, chunk_id="one"),
        _chunk("QLoRA uses 4-bit quantization", 0.5, page=2, chunk_id="two"),
    ]

    scores = bm25_scores("QLoRA quantization", chunks)

    assert scores[1] > scores[0]


def test_reranker_can_prioritize_lexical_match_and_diversify_pages() -> None:
    chunks = [
        _chunk("noise", 0.99, page=1, chunk_id="one"),
        _chunk("more noise", 0.98, page=1, chunk_id="two"),
        _chunk("the answer uses stratify", 0.50, page=2, chunk_id="three"),
    ]

    ranked = rerank_candidates(
        "stratify",
        chunks,
        k=2,
        dense_weight=0.0,
        max_chunks_per_page=1,
    )

    assert [chunk.metadata.chunk_id for chunk in ranked] == ["three", "one"]
    assert len({chunk.metadata.page for chunk in ranked}) == 2


def test_expanded_retriever_requests_candidate_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []
    candidates = [_chunk("target", 0.8, page=1, chunk_id="one")]

    def fake_retrieve(query: str, **kwargs: object) -> list[RetrievedChunk]:
        calls.append({"query": query, **kwargs})
        return candidates

    monkeypatch.setattr(retrieval_reranker, "retrieve", fake_retrieve)
    retrieve_fn = expanded_rerank_retriever(candidate_k=50)

    result = retrieve_fn(
        "target",
        k=10,
        filters={"filename": "source.pdf"},
        collection_name="evaluation",
    )

    assert [chunk.metadata.chunk_id for chunk in result] == ["one"]
    assert calls == [
        {
            "query": "target",
            "k": 50,
            "filters": {"filename": "source.pdf"},
            "collection_name": "evaluation",
        }
    ]


def test_generic_reranking_retriever_wraps_supplied_retriever() -> None:
    calls: list[int] = []

    def base_retriever(*args: object, **kwargs: object) -> list[RetrievedChunk]:
        del args
        calls.append(int(kwargs["k"]))
        return [_chunk("exact target", 0.5, page=2, chunk_id="target")]

    retrieve_fn = reranking_retriever(base_retriever, candidate_k=30)

    result = retrieve_fn("target", k=10, filters=None, collection_name="external")

    assert calls == [30]
    assert result[0].metadata.chunk_id == "target"


def test_reranker_validates_configuration() -> None:
    chunk = _chunk("target", 0.8, page=1, chunk_id="one")

    with pytest.raises(ValueError, match="dense_weight"):
        rerank_candidates("target", [chunk], k=10, dense_weight=1.1)
    with pytest.raises(ValueError, match="max_chunks_per_page"):
        rerank_candidates("target", [chunk], k=10, max_chunks_per_page=0)
    with pytest.raises(ValueError, match="candidate_k"):
        expanded_rerank_retriever(candidate_k=5)(
            "target",
            k=10,
            filters=None,
            collection_name="evaluation",
        )
