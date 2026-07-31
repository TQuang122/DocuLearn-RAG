"""Retrieval, prompts, citations, and grounded answers."""

from __future__ import annotations

import random
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Literal

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from loguru import logger

from src.config import settings
from src.filters import filters_to_qdrant
from src.llm import invoke_llm
from src.reranking import rerank_candidates
from src.retrieval_telemetry import RetrievalTelemetryEvent, write_retrieval_event
from src.schemas import ChunkMetadata, Citation, RagAnswer, RetrievedChunk
from src.store import get_vector_store, scroll_all

PROMPTS_DIR = Path(__file__).parent / "prompts"
ANSWER_TEMPLATE = "answer.jinja2"
_TELEMETRY_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="retrieval-shadow")


def _dense_retrieve(
    query: str,
    *,
    k: int,
    filters: dict[str, object] | None = None,
    collection_name: str | None = None,
) -> list[RetrievedChunk]:
    store = get_vector_store(collection_name=collection_name)
    hits = store.similarity_search_with_score(
        query=query,
        k=k,
        filter=filters_to_qdrant(filters),
    )
    return [
        RetrievedChunk(
            text=doc.page_content,
            score=float(score),
            metadata=ChunkMetadata(**doc.metadata),
        )
        for doc, score in hits
    ]


def _log_retrieval_telemetry(
    *,
    mode: str,
    requested_k: int,
    candidate_k: int,
    result_count: int,
    latency_ms: float,
    fallback: bool,
    collection_name: str,
) -> None:
    message = (
        "retrieval_completed mode={} requested_k={} candidate_k={} results={} "
        + "latency_ms={:.3f} fallback={} collection={}"
    )
    logger.bind(
        retrieval_mode=mode,
        requested_k=requested_k,
        candidate_k=candidate_k,
        result_count=result_count,
        retrieval_latency_ms=round(latency_ms, 3),
        fallback=fallback,
        collection=collection_name,
    ).info(
        message,
        mode,
        requested_k,
        candidate_k,
        result_count,
        latency_ms,
        fallback,
        collection_name,
    )


def _run_primary_retrieval(
    query: str,
    *,
    requested_k: int,
    filters: dict[str, object] | None,
    collection_name: str | None,
) -> tuple[list[RetrievedChunk], int, bool]:
    candidate_k = requested_k
    fallback = False
    if settings.retrieval_mode == "dense":
        chunks = _dense_retrieve(
            query,
            k=requested_k,
            filters=filters,
            collection_name=collection_name,
        )
        return chunks, candidate_k, fallback

    candidate_k = max(requested_k, settings.retrieval_candidate_k)
    try:
        candidates = _dense_retrieve(
            query,
            k=candidate_k,
            filters=filters,
            collection_name=collection_name,
        )
        chunks = rerank_candidates(
            query,
            candidates,
            k=requested_k,
            dense_weight=settings.retrieval_dense_weight,
            max_chunks_per_page=settings.retrieval_max_chunks_per_page,
        )
    except Exception as exc:
        if not settings.retrieval_fallback_to_dense:
            raise
        fallback = True
        logger.warning("Fusion retrieval failed; falling back to dense: {}", type(exc).__name__)
        chunks = _dense_retrieve(
            query,
            k=requested_k,
            filters=filters,
            collection_name=collection_name,
        )
    return chunks, candidate_k, fallback


def _run_shadow_retrieval(
    query: str,
    *,
    shadow_mode: Literal["dense", "fusion"],
    requested_k: int,
    filters: dict[str, object] | None,
    collection_name: str | None,
) -> list[RetrievedChunk]:
    if shadow_mode == "dense":
        return _dense_retrieve(
            query,
            k=requested_k,
            filters=filters,
            collection_name=collection_name,
        )
    candidate_k = max(requested_k, settings.retrieval_candidate_k)
    candidates = _dense_retrieve(
        query,
        k=candidate_k,
        filters=filters,
        collection_name=collection_name,
    )
    return rerank_candidates(
        query,
        candidates,
        k=requested_k,
        dense_weight=settings.retrieval_dense_weight,
        max_chunks_per_page=settings.retrieval_max_chunks_per_page,
    )


def _compare_and_write_shadow(
    event: RetrievalTelemetryEvent,
    primary_chunks: list[RetrievedChunk],
    query: str,
    filters: dict[str, object] | None,
    collection_name: str | None,
) -> None:
    shadow_mode: Literal["dense", "fusion"] = (
        "dense" if event.primary_mode == "fusion" else "fusion"
    )
    started = perf_counter()
    try:
        shadow_chunks = _run_shadow_retrieval(
            query,
            shadow_mode=shadow_mode,
            requested_k=event.requested_k,
            filters=filters,
            collection_name=collection_name,
        )
        primary_ids = [chunk.metadata.chunk_id for chunk in primary_chunks]
        shadow_ids = [chunk.metadata.chunk_id for chunk in shadow_chunks]
        overlap = len(set(primary_ids) & set(shadow_ids)) / event.requested_k
        top1_agreement = bool(primary_ids and shadow_ids and primary_ids[0] == shadow_ids[0])
        updated = event.model_copy(
            update={
                "shadow_mode": shadow_mode,
                "shadow_result_count": len(shadow_chunks),
                "shadow_latency_ms": (perf_counter() - started) * 1000,
                "top1_agreement": top1_agreement,
                "overlap_at_k": overlap,
            }
        )
    except Exception as exc:
        updated = event.model_copy(
            update={
                "shadow_mode": shadow_mode,
                "shadow_latency_ms": (perf_counter() - started) * 1000,
                "shadow_error": type(exc).__name__,
            }
        )
    write_retrieval_event(updated)


def _submit_telemetry(function: Callable[..., None], *args: object) -> None:
    future = _TELEMETRY_EXECUTOR.submit(function, *args)
    future.add_done_callback(_report_telemetry_failure)


def _report_telemetry_failure(future: Future[None]) -> None:
    error = future.exception()
    if error is not None:
        logger.error("retrieval_telemetry_failed error={}", type(error).__name__)


def retrieve(
    query: str,
    k: int | None = None,
    filters: dict[str, object] | None = None,
    collection_name: str | None = None,
) -> list[RetrievedChunk]:
    requested_k = k or settings.top_k
    name = collection_name or settings.qdrant_collection
    started = perf_counter()
    candidate_k = (
        max(requested_k, settings.retrieval_candidate_k)
        if settings.retrieval_mode == "fusion"
        else requested_k
    )
    try:
        chunks, candidate_k, fallback = _run_primary_retrieval(
            query,
            requested_k=requested_k,
            filters=filters,
            collection_name=collection_name,
        )
    except Exception as exc:
        latency_ms = (perf_counter() - started) * 1000
        logger.error(
            "retrieval_failed mode={} latency_ms={:.3f} error={}",
            settings.retrieval_mode,
            latency_ms,
            type(exc).__name__,
        )
        if settings.retrieval_telemetry_enabled:
            _submit_telemetry(
                write_retrieval_event,
                RetrievalTelemetryEvent(
                    collection=name,
                    primary_mode=settings.retrieval_mode,
                    requested_k=requested_k,
                    candidate_k=candidate_k,
                    primary_result_count=0,
                    primary_latency_ms=latency_ms,
                    primary_error=type(exc).__name__,
                ),
            )
        raise

    latency_ms = (perf_counter() - started) * 1000
    _log_retrieval_telemetry(
        mode=settings.retrieval_mode,
        requested_k=requested_k,
        candidate_k=candidate_k,
        result_count=len(chunks),
        latency_ms=latency_ms,
        fallback=fallback,
        collection_name=name,
    )
    if settings.retrieval_telemetry_enabled:
        event = RetrievalTelemetryEvent(
            collection=name,
            primary_mode=settings.retrieval_mode,
            requested_k=requested_k,
            candidate_k=candidate_k,
            primary_result_count=len(chunks),
            primary_latency_ms=latency_ms,
            fallback=fallback,
        )
        if random.random() < settings.retrieval_shadow_sample_rate:
            _submit_telemetry(
                _compare_and_write_shadow,
                event,
                chunks,
                query,
                filters,
                collection_name,
            )
        else:
            _submit_telemetry(write_retrieval_event, event)
    return chunks


def fetch_all_chunks(
    filters: dict[str, object] | None = None,
    collection_name: str | None = None,
) -> list[RetrievedChunk]:
    """Scroll every chunk matching the filter, ordered by filename → page → index."""
    name = collection_name or settings.qdrant_collection
    results: list[RetrievedChunk] = []
    for page in scroll_all(name, scroll_filter=filters_to_qdrant(filters)):
        for point in page:
            payload = point.payload or {}
            meta = payload.get("metadata") or {}
            text = payload.get("page_content") or ""
            if not meta or not text:
                continue
            results.append(RetrievedChunk(text=text, score=0.0, metadata=ChunkMetadata(**meta)))
    results.sort(
        key=lambda r: (
            r.metadata.filename,
            r.metadata.page,
            int(r.metadata.chunk_id.rsplit(":", 1)[-1]),
        )
    )
    return results


@lru_cache(maxsize=1)
def _jinja_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(PROMPTS_DIR)),
        autoescape=False,
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_prompt(template_name: str, **context: object) -> str:
    """Render an arbitrary Jinja template from the prompts directory."""
    return _jinja_env().get_template(template_name).render(**context)


def format_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    return [
        Citation(
            source_index=i,
            source_marker=f"S{i}",
            filename=c.metadata.filename,
            page=c.metadata.page,
            source_text=c.text.strip(),
            section=c.metadata.section,
            chunk_id=c.metadata.chunk_id,
        )
        for i, c in enumerate(chunks, start=1)
    ]


def answer(
    question: str,
    k: int | None = None,
    filters: dict[str, object] | None = None,
    collection_name: str | None = None,
) -> RagAnswer:
    chunks = retrieve(question, k=k, filters=filters, collection_name=collection_name)
    if not chunks:
        return RagAnswer(
            question=question,
            answer="I don't have enough information in the provided context to answer.",
        )

    prompt = render_prompt(ANSWER_TEMPLATE, question=question, chunks=chunks)
    text = invoke_llm(prompt)

    return RagAnswer(
        question=question,
        answer=text.strip(),
        citations=format_citations(chunks),
        chunks=chunks,
    )
