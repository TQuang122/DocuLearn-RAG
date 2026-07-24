"""FastAPI entrypoint for the DocuLearn-RAG learning service."""

from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import FastAPI as _FastAPI
from fastapi import File, Header, HTTPException, UploadFile, status
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from src.config import settings
from src.filters import MetadataFilter, filters_to_dict
from src.indexing import save_and_ingest_pdf
from src.learning import generate_flashcards, generate_quiz
from src.learning import summarize as summarize_learning
from src.rag import answer
from src.schemas import FlashcardSet, QuizSet, RagAnswer, Summary
from src.store import list_documents


class StrictRequest(BaseModel):
    """Reject misspelled request fields instead of silently ignoring them."""

    model_config = ConfigDict(extra="forbid")


class AskRequest(StrictRequest):
    question: str = Field(min_length=1)
    k: int | None = Field(default=None, ge=1, le=64)
    filters: MetadataFilter | None = None


class SummarizeRequest(StrictRequest):
    document: str | None = None
    query: str | None = None
    filters: MetadataFilter | None = None
    k: int | None = Field(default=None, ge=1, le=64)


class QuizRequest(SummarizeRequest):
    count: int | None = Field(default=None, ge=1, le=50)


class FlashcardsRequest(SummarizeRequest):
    count: int | None = Field(default=None, ge=1, le=100)


class HealthResponse(BaseModel):
    status: str


class DocumentInfo(BaseModel):
    filename: str
    document_id: str
    pages: list[int]
    page_count: int
    chunk_count: int


class UploadResponse(BaseModel):
    filename: str
    document_id: str
    chunks_indexed: int


app = _FastAPI(
    title="DocuLearn-RAG API",
    description="Grounded Q&A, summaries, quizzes, and flashcards over indexed PDFs.",
    version="0.1.0",
)


def _require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    """Protect data and generation routes when RAG_API_KEY is configured."""
    expected = settings.api_key
    if expected is None:
        return
    if x_api_key is None or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key.",
        )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/documents", response_model=list[DocumentInfo])
def documents(
    x_api_key: Annotated[str | None, Header()] = None,
) -> list[dict[str, object]]:
    _require_api_key(x_api_key)
    return list_documents()


@app.post("/upload", response_model=UploadResponse)
async def upload(
    file: Annotated[UploadFile, File()],
    x_api_key: Annotated[str | None, Header()] = None,
) -> dict[str, object]:
    _require_api_key(x_api_key)
    content = await file.read(settings.max_upload_bytes + 1)
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"PDF exceeds the {settings.max_upload_bytes}-byte upload limit.",
        )
    try:
        return await run_in_threadpool(save_and_ingest_pdf, content, file.filename or "")
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except (OSError, RuntimeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The document could not be indexed.",
        ) from exc


@app.post("/ask", response_model=RagAnswer)
def ask(
    request: AskRequest,
    x_api_key: Annotated[str | None, Header()] = None,
) -> RagAnswer:
    _require_api_key(x_api_key)
    return answer(
        request.question,
        k=request.k,
        filters=filters_to_dict(request.filters),
    )


@app.post("/summarize", response_model=Summary)
def summarize(
    request: SummarizeRequest,
    x_api_key: Annotated[str | None, Header()] = None,
) -> Summary:
    _require_api_key(x_api_key)
    return summarize_learning(
        document=request.document,
        query=request.query,
        filters=filters_to_dict(request.filters),
        k=request.k,
    )


@app.post("/quiz", response_model=QuizSet)
def quiz(
    request: QuizRequest,
    x_api_key: Annotated[str | None, Header()] = None,
) -> QuizSet:
    _require_api_key(x_api_key)
    return generate_quiz(
        document=request.document,
        query=request.query,
        filters=filters_to_dict(request.filters),
        count=request.count,
        k=request.k,
    )


@app.post("/flashcards", response_model=FlashcardSet)
def flashcards(
    request: FlashcardsRequest,
    x_api_key: Annotated[str | None, Header()] = None,
) -> FlashcardSet:
    _require_api_key(x_api_key)
    return generate_flashcards(
        document=request.document,
        query=request.query,
        filters=filters_to_dict(request.filters),
        count=request.count,
        k=request.k,
    )
