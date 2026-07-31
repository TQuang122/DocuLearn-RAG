from __future__ import annotations

import pytest
from loguru import logger

from src import rag
from src.schemas import ChunkMetadata, RetrievedChunk


def _chunk(text: str, score: float, index: int, page: int | None = None) -> RetrievedChunk:
    resolved_page = page if page is not None else index
    return RetrievedChunk(
        text=text,
        score=score,
        metadata=ChunkMetadata(
            document_id="doc",
            filename="document.pdf",
            source="document.pdf",
            page=resolved_page,
            chunk_id=f"doc:{resolved_page}:{index}",
        ),
    )


def _configure(monkeypatch: pytest.MonkeyPatch, *, mode: str, fallback: bool = True) -> None:
    monkeypatch.setattr(rag.settings, "retrieval_mode", mode)
    monkeypatch.setattr(rag.settings, "retrieval_candidate_k", 50)
    monkeypatch.setattr(rag.settings, "retrieval_dense_weight", 0.25)
    monkeypatch.setattr(rag.settings, "retrieval_max_chunks_per_page", 1)
    monkeypatch.setattr(rag.settings, "retrieval_fallback_to_dense", fallback)
    monkeypatch.setattr(rag.settings, "retrieval_telemetry_enabled", False)
    monkeypatch.setattr(rag.settings, "retrieval_shadow_sample_rate", 0.0)


def test_dense_mode_preserves_existing_retrieval_contract(monkeypatch) -> None:
    _configure(monkeypatch, mode="dense")
    calls: list[int] = []
    telemetry: list[dict[str, object]] = []
    chunks = [_chunk("dense result", 0.9, 1)]

    monkeypatch.setattr(
        rag,
        "_dense_retrieve",
        lambda _query, *, k, filters, collection_name: calls.append(k) or chunks,
    )
    monkeypatch.setattr(rag, "_log_retrieval_telemetry", lambda **kwargs: telemetry.append(kwargs))

    result = rag.retrieve("question", k=5, filters={"filename": ["document.pdf"]})

    assert result == chunks
    assert calls == [5]
    assert telemetry[0]["mode"] == "dense"
    assert telemetry[0]["candidate_k"] == 5
    assert telemetry[0]["fallback"] is False


def test_fusion_mode_expands_candidates_and_reranks(monkeypatch) -> None:
    _configure(monkeypatch, mode="fusion")
    calls: list[int] = []
    telemetry: list[dict[str, object]] = []
    candidates = [
        _chunk("unrelated text", 0.95, 1),
        _chunk("target phrase appears here", 0.70, 2),
    ]

    monkeypatch.setattr(
        rag,
        "_dense_retrieve",
        lambda _query, *, k, filters, collection_name: calls.append(k) or candidates,
    )
    monkeypatch.setattr(rag, "_log_retrieval_telemetry", lambda **kwargs: telemetry.append(kwargs))

    result = rag.retrieve("target phrase", k=1)

    assert calls == [50]
    assert result[0].text == "target phrase appears here"
    assert telemetry[0]["mode"] == "fusion"
    assert telemetry[0]["candidate_k"] == 50
    assert telemetry[0]["fallback"] is False


def test_fusion_failure_falls_back_to_dense(monkeypatch) -> None:
    _configure(monkeypatch, mode="fusion")
    calls: list[int] = []
    telemetry: list[dict[str, object]] = []
    chunks = [_chunk("dense fallback", 0.8, 1)]

    monkeypatch.setattr(
        rag,
        "_dense_retrieve",
        lambda _query, *, k, filters, collection_name: calls.append(k) or chunks,
    )
    monkeypatch.setattr(
        rag,
        "rerank_candidates",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("rerank failed")),
    )
    monkeypatch.setattr(rag, "_log_retrieval_telemetry", lambda **kwargs: telemetry.append(kwargs))

    result = rag.retrieve("question", k=5)

    assert result == chunks
    assert calls == [50, 5]
    assert telemetry[0]["fallback"] is True


def test_fusion_failure_can_disable_dense_fallback(monkeypatch) -> None:
    _configure(monkeypatch, mode="fusion", fallback=False)
    chunks = [_chunk("candidate", 0.8, 1)]
    monkeypatch.setattr(
        rag,
        "_dense_retrieve",
        lambda _query, *, k, filters, collection_name: chunks,
    )
    monkeypatch.setattr(
        rag,
        "rerank_candidates",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("rerank failed")),
    )

    with pytest.raises(RuntimeError, match="rerank failed"):
        rag.retrieve("question", k=5)


def test_retrieval_telemetry_is_visible_in_default_log_message() -> None:
    messages: list[str] = []
    sink_id = logger.add(messages.append, format="{message}")
    try:
        rag._log_retrieval_telemetry(
            mode="fusion",
            requested_k=5,
            candidate_k=50,
            result_count=5,
            latency_ms=12.345,
            fallback=False,
            collection_name="rag_chunks",
        )
    finally:
        logger.remove(sink_id)

    message = messages[-1]
    assert "mode=fusion" in message
    assert "candidate_k=50" in message
    assert "latency_ms=12.345" in message
    assert "fallback=False" in message


def test_shadow_comparison_is_backgrounded_and_privacy_safe(monkeypatch) -> None:
    _configure(monkeypatch, mode="fusion")
    monkeypatch.setattr(rag.settings, "retrieval_telemetry_enabled", True)
    monkeypatch.setattr(rag.settings, "retrieval_shadow_sample_rate", 1.0)
    candidates = [
        _chunk("unrelated text", 0.95, 1),
        _chunk("secret target phrase", 0.70, 2),
    ]
    recorded: list[object] = []

    def fake_dense(
        _query: str,
        *,
        k: int,
        filters: dict[str, object] | None,
        collection_name: str | None,
    ) -> list[RetrievedChunk]:
        return candidates if k == 50 else candidates[:1]

    monkeypatch.setattr(rag, "_dense_retrieve", fake_dense)
    monkeypatch.setattr(rag, "_log_retrieval_telemetry", lambda **_kwargs: None)
    monkeypatch.setattr(rag, "write_retrieval_event", recorded.append)
    monkeypatch.setattr(
        rag,
        "_submit_telemetry",
        lambda function, *args: function(*args),
    )

    result = rag.retrieve("secret target phrase", k=1)

    assert result[0].text == "secret target phrase"
    event = recorded[0]
    assert event.primary_mode == "fusion"
    assert event.shadow_mode == "dense"
    assert event.top1_agreement is False
    assert event.overlap_at_k == 0
    assert "secret target phrase" not in event.model_dump_json()
