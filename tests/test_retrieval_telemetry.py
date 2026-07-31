from __future__ import annotations

from pathlib import Path

import pytest

from src.retrieval_telemetry import (
    RetrievalTelemetryEvent,
    load_retrieval_summary,
    read_retrieval_events,
    summarize_retrieval_events,
    write_retrieval_event,
)


def _event(**updates: object) -> RetrievalTelemetryEvent:
    values: dict[str, object] = {
        "collection": "rag_chunks",
        "primary_mode": "fusion",
        "shadow_mode": "dense",
        "requested_k": 5,
        "candidate_k": 50,
        "primary_result_count": 5,
        "shadow_result_count": 5,
        "primary_latency_ms": 12.0,
        "shadow_latency_ms": 8.0,
        "top1_agreement": True,
        "overlap_at_k": 0.8,
    }
    values.update(updates)
    return RetrievalTelemetryEvent.model_validate(values)


def test_telemetry_round_trip_contains_no_query_text(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.jsonl"
    write_retrieval_event(_event(), path)

    events, malformed = read_retrieval_events(path)

    assert len(events) == 1
    assert malformed == 0
    assert "query" not in path.read_text(encoding="utf-8")
    assert "filters" not in path.read_text(encoding="utf-8")


def test_telemetry_summary_computes_operational_gate() -> None:
    events = [
        _event(),
        _event(
            fallback=True,
            primary_result_count=3,
            primary_latency_ms=20.0,
            shadow_error="RuntimeError",
            shadow_result_count=None,
            shadow_latency_ms=9.0,
            top1_agreement=None,
            overlap_at_k=None,
        ),
    ]

    summary = summarize_retrieval_events(events, min_events=1, min_shadow_events=1)

    assert summary["counts"]["events"] == 2
    assert summary["rates"]["fallback"] == 0.5
    assert summary["rates"]["shadow_error"] == 0.5
    assert summary["rates"]["insufficient_results"] == 0.5
    assert summary["rates"]["top1_agreement"] == 1
    assert summary["promotion_gate"]["status"] == "fail"


def test_empty_summary_is_insufficient_and_reports_malformed_lines(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.jsonl"
    path.write_text("not-json\n", encoding="utf-8")

    summary = load_retrieval_summary(path)

    assert summary["counts"]["events"] == 0
    assert summary["counts"]["malformed_lines"] == 1
    assert summary["promotion_gate"]["status"] == "insufficient_data"


def test_summary_rejects_invalid_promotion_thresholds() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        summarize_retrieval_events([], max_fallback_rate=1.1)
    with pytest.raises(ValueError, match="positive"):
        summarize_retrieval_events([], min_events=0)
