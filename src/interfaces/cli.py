"""Typer command-line entrypoint for DocuLearn-RAG."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import typer
from pydantic import BaseModel, ValidationError

from src.export import ExportFormat, export
from src.filters import MetadataFilter, filters_to_dict
from src.indexing import ingest as ingest_data
from src.learning import generate_flashcards, generate_quiz
from src.learning import summarize as summarize_learning
from src.rag import answer, retrieve
from src.retrieval_telemetry import load_retrieval_summary
from src.schemas import RetrievedChunk

app = typer.Typer(help="Index PDFs and use grounded DocuLearn-RAG learning features.")


def _parse_filters(value: str | None) -> MetadataFilter | None:
    if value is None or not value.strip():
        return None

    parsed: dict[str, object] = {}
    allowed = set(MetadataFilter.model_fields)
    for expression in value.split(","):
        key, separator, raw_value = expression.partition("=")
        key = key.strip()
        raw_value = raw_value.strip()
        if not separator or not key or not raw_value:
            raise typer.BadParameter("Filters must use key=value pairs separated by commas.")
        if key not in allowed:
            raise typer.BadParameter(
                f"Unknown filter '{key}'. Expected one of: {', '.join(sorted(allowed))}."
            )
        if key == "page":
            try:
                parsed[key] = int(raw_value)
            except ValueError as error:
                raise typer.BadParameter("The page filter must be an integer.") from error
        elif key == "filenames":
            parsed[key] = [item.strip() for item in raw_value.split("|") if item.strip()]
        else:
            parsed[key] = raw_value

    try:
        return MetadataFilter.model_validate(parsed)
    except ValidationError as error:
        raise typer.BadParameter(str(error)) from error


def _write_result(
    result: BaseModel,
    output: Path | None,
    fmt: ExportFormat,
) -> None:
    rendered = export(result, fmt=fmt, output=output)
    if output is None:
        typer.echo(rendered, nl=False)
    else:
        typer.echo(f"Exported to {rendered}")


def _print_sources(chunks: list[RetrievedChunk]) -> None:
    if not chunks:
        return
    typer.echo("\nSources:")
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.metadata
        typer.echo(f"S{index}: {metadata.filename}, page {metadata.page}")


@app.command()
def ingest(recreate: bool = False) -> None:
    """Index all PDFs in the configured data directory."""
    count = ingest_data(recreate=recreate)
    typer.echo(f"Done. {count} chunks indexed.")


@app.command()
def ask(
    question: str,
    k: int | None = None,
    filters: str | None = None,
) -> None:
    """Ask a grounded question and show its retrieved sources."""
    result = answer(question, k=k, filters=filters_to_dict(_parse_filters(filters)))
    typer.echo(result.answer)
    _print_sources(result.chunks)


@app.command("debug-retrieval")
def debug_retrieval(
    question: str,
    k: int | None = None,
    filters: str | None = None,
    as_json: bool = False,
) -> None:
    """Inspect the chunks retrieved for a question."""
    chunks = retrieve(question, k=k, filters=filters_to_dict(_parse_filters(filters)))
    if as_json:
        typer.echo(
            json.dumps(
                [chunk.model_dump(mode="json") for chunk in chunks],
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.metadata
        typer.echo(
            f"S{index} {metadata.filename} p.{metadata.page} "
            f"score={chunk.score:.4f}\n{chunk.text}\n"
        )


@app.command(
    "retrieval-telemetry",
    help="Summarize privacy-safe retrieval pilot telemetry and promotion gates.",
)
def retrieval_telemetry(
    path: Path | None = None,
    max_fallback_rate: float = 0.01,
    max_error_rate: float = 0.01,
    max_insufficient_rate: float = 0.01,
    max_primary_p95_ms: float | None = None,
    min_events: int = 100,
    min_shadow_events: int = 30,
    output: Path | None = None,
    fail_on_gate: bool = False,
) -> None:
    summary = load_retrieval_summary(
        path,
        max_fallback_rate=max_fallback_rate,
        max_error_rate=max_error_rate,
        max_insufficient_rate=max_insufficient_rate,
        max_primary_p95_ms=max_primary_p95_ms,
        min_events=min_events,
        min_shadow_events=min_shadow_events,
    )
    rendered = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if output is None:
        typer.echo(rendered, nl=False)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        _ = output.write_text(rendered, encoding="utf-8")
        typer.echo(f"Exported to {output}")
    gate = cast(dict[str, object], summary["promotion_gate"])
    status = str(gate["status"])
    if fail_on_gate and status != "pass":
        raise typer.Exit(code=1)


@app.command()
def summarize(
    document: str | None = None,
    query: str | None = None,
    filters: str | None = None,
    k: int | None = None,
    output: Path | None = None,
    fmt: ExportFormat = "text",
) -> None:
    """Create a grounded summary."""
    result = summarize_learning(
        document=document,
        query=query,
        filters=filters_to_dict(_parse_filters(filters)),
        k=k,
    )
    _write_result(result, output, fmt)


@app.command()
def quiz(
    document: str | None = None,
    query: str | None = None,
    filters: str | None = None,
    count: int | None = None,
    k: int | None = None,
    output: Path | None = None,
    fmt: ExportFormat = "text",
) -> None:
    """Generate a grounded multiple-choice quiz."""
    result = generate_quiz(
        document=document,
        query=query,
        filters=filters_to_dict(_parse_filters(filters)),
        count=count,
        k=k,
    )
    _write_result(result, output, fmt)


@app.command()
def flashcards(
    document: str | None = None,
    query: str | None = None,
    filters: str | None = None,
    count: int | None = None,
    k: int | None = None,
    output: Path | None = None,
    fmt: ExportFormat = "text",
) -> None:
    """Generate grounded flashcards."""
    result = generate_flashcards(
        document=document,
        query=query,
        filters=filters_to_dict(_parse_filters(filters)),
        count=count,
        k=k,
    )
    _write_result(result, output, fmt)


if __name__ == "__main__":
    app()
