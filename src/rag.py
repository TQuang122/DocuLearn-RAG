from functools import lru_cache
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import ValidationError

from config import settings
from schemas import (
    ChunkMetadata,
    Citation,
    RagAnswer,
    RetrievedChunk,
)


def retrieve(
    query: str,
    k: int | None = None,
    filters: Any = None,
    collection_name: str | None = None,
) -> list[RetrievedChunk]:
    query = query.strip()

    if not query:
        raise ValueError("query cannot be empty.")

    resolved_k = settings.top_k if k is None else k

    if resolved_k <= 0:
        raise ValueError("k must be greater than 0.")

    hits = get_vector_store(
        collection_name=collection_name
    ).similarity_search_with_score(
        query=query,
        k=resolved_k,
        filter=filters_to_qdrant(filters),
    )

    results: list[RetrievedChunk] = []

    for document, score in hits:
        metadata = ChunkMetadata.model_validate(
            document.metadata
        )

        results.append(
            RetrievedChunk(
                text=document.page_content,
                score=float(score),
                metadata=metadata,
            )
        )

    return results


def _chunk_index(chunk_id: str) -> int:
    try:
        return int(chunk_id.rsplit(":", 1)[-1])
    except (TypeError, ValueError):
        # Đẩy chunk có ID không hợp lệ xuống cuối.
        return 2**31 - 1


def fetch_all_chunks(
    filters: Any = None,
    collection_name: str | None = None,
) -> list[RetrievedChunk]:
    name = collection_name or settings.qdrant_collection
    qdrant_filter = filters_to_qdrant(filters)

    results: list[RetrievedChunk] = []

    for points in scroll_all(
        name,
        scroll_filter=qdrant_filter,
    ):
        for point in points:
            payload = point.payload or {}

            metadata_payload = payload.get("metadata")
            text_payload = payload.get("page_content")

            if not isinstance(metadata_payload, dict):
                continue

            if not isinstance(text_payload, str):
                continue

            text = text_payload.strip()

            if not text:
                continue

            try:
                metadata = ChunkMetadata.model_validate(
                    metadata_payload
                )
            except ValidationError:
                # Nên log dữ liệu lỗi trong ứng dụng thực tế.
                continue

            results.append(
                RetrievedChunk(
                    text=text,
                    score=0.0,
                    metadata=metadata,
                )
            )

    return sorted(
        results,
        key=lambda result: (
            result.metadata.filename.casefold(),
            result.metadata.page,
            _chunk_index(result.metadata.chunk_id),
        ),
    )


@lru_cache(maxsize=1)
def _jinja_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(PROMPTS_DIR)),
        autoescape=False,
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_prompt(
    template_name: str,
    **context: Any,
) -> str:
    return (
        _jinja_env()
        .get_template(template_name)
        .render(**context)
    )


def format_citations(
    chunks: list[RetrievedChunk],
) -> list[Citation]:
    return [
        Citation(
            source_index=index,
            source_marker=f"S{index}",
            filename=chunk.metadata.filename,
            page=chunk.metadata.page,
            section=chunk.metadata.section,
            chunk_id=chunk.metadata.chunk_id,
        )
        for index, chunk in enumerate(
            chunks,
            start=1,
        )
    ]


def answer(
    question: str,
    k: int | None = None,
    filters: Any = None,
    collection_name: str | None = None,
) -> RagAnswer:
    question = question.strip()

    if not question:
        raise ValueError("question cannot be empty.")

    chunks = retrieve(
        query=question,
        k=k,
        filters=filters,
        collection_name=collection_name,
    )

    if not chunks:
        return RagAnswer(
            question=question,
            answer=(
                "Tôi không có đủ thông tin trong ngữ cảnh "
                "được cung cấp để trả lời."
            ),
            citations=[],
            chunks=[],
        )

    prompt = render_prompt(
        ANSWER_TEMPLATE,
        question=question,
        chunks=chunks,
    )

    response = invoke_llm(prompt)

    if not isinstance(response, str):
        raise TypeError(
            "invoke_llm() must return a string."
        )

    answer_text = response.strip()

    if not answer_text:
        answer_text = (
            "Tôi không có đủ thông tin trong ngữ cảnh "
            "được cung cấp để trả lời."
        )

    return RagAnswer(
        question=question,
        answer=answer_text,
        citations=format_citations(chunks),
        chunks=chunks,
    )