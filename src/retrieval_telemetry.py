from __future__ import annotations

from collections import Counter, deque
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from threading import Lock
from typing import ClassVar, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from src.config import settings

_WRITE_LOCK = Lock()


class RetrievalTelemetryEvent(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    collection: str
    primary_mode: Literal["dense", "fusion"]
    shadow_mode: Literal["dense", "fusion"] | None = None
    requested_k: int = Field(ge=1)
    candidate_k: int = Field(ge=1)
    primary_result_count: int = Field(ge=0)
    shadow_result_count: int | None = Field(default=None, ge=0)
    primary_latency_ms: float = Field(ge=0)
    shadow_latency_ms: float | None = Field(default=None, ge=0)
    fallback: bool = False
    primary_error: str | None = None
    shadow_error: str | None = None
    top1_agreement: bool | None = None
    overlap_at_k: float | None = Field(default=None, ge=0, le=1)


def telemetry_path() -> Path:
    return settings.export_dir / "retrieval_telemetry.jsonl"


def write_retrieval_event(
    event: RetrievalTelemetryEvent,
    path: Path | None = None,
    *,
    max_bytes: int | None = None,
    retained_events: int | None = None,
) -> None:
    destination = path or telemetry_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = event.model_dump_json() + "\n"
    byte_limit = max_bytes or settings.retrieval_telemetry_max_bytes
    event_limit = retained_events or settings.retrieval_telemetry_retained_events
    if byte_limit < 1 or event_limit < 2:
        raise ValueError("Telemetry retention limits must be positive.")
    with _WRITE_LOCK:
        if destination.exists() and destination.stat().st_size >= byte_limit:
            retained: deque[str] = deque(maxlen=event_limit - 1)
            with destination.open("r", encoding="utf-8") as source:
                retained.extend(source)
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                _ = handle.writelines(retained)
            _ = temporary.replace(destination)
        with destination.open("a", encoding="utf-8") as handle:
            _ = handle.write(payload)


def read_retrieval_events(
    path: Path | None = None,
) -> tuple[list[RetrievalTelemetryEvent], int]:
    source = path or telemetry_path()
    if not source.exists():
        return [], 0
    events: list[RetrievalTelemetryEvent] = []
    malformed = 0
    for line in source.read_text(encoding="utf-8").splitlines():
        try:
            events.append(RetrievalTelemetryEvent.model_validate_json(line))
        except ValueError:
            malformed += 1
    return events, malformed


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize_retrieval_events(
    events: list[RetrievalTelemetryEvent],
    *,
    malformed_lines: int = 0,
    max_fallback_rate: float = 0.01,
    max_error_rate: float = 0.01,
    max_insufficient_rate: float = 0.01,
    max_primary_p95_ms: float | None = None,
    min_events: int = 100,
    min_shadow_events: int = 30,
) -> dict[str, object]:
    rate_thresholds = (max_fallback_rate, max_error_rate, max_insufficient_rate)
    if any(value < 0 or value > 1 for value in rate_thresholds):
        raise ValueError("Rate thresholds must be between 0 and 1.")
    if min_events < 1 or min_shadow_events < 1:
        raise ValueError("Minimum event counts must be positive.")
    if max_primary_p95_ms is not None and max_primary_p95_ms < 0:
        raise ValueError("The primary p95 latency threshold cannot be negative.")
    total = len(events)
    shadowed = [event for event in events if event.shadow_mode is not None]
    primary_errors = [event for event in events if event.primary_error is not None]
    shadow_errors = [event for event in shadowed if event.shadow_error is not None]
    insufficient = [
        event
        for event in events
        if event.primary_error is None and event.primary_result_count < event.requested_k
    ]
    paired = [
        event
        for event in shadowed
        if event.shadow_error is None
        and event.top1_agreement is not None
        and event.overlap_at_k is not None
    ]
    primary_latencies = [event.primary_latency_ms for event in events]
    shadow_latencies = [
        event.shadow_latency_ms
        for event in shadowed
        if event.shadow_latency_ms is not None
    ]
    fallback_rate = sum(event.fallback for event in events) / total if total else None
    primary_error_rate = len(primary_errors) / total if total else None
    shadow_error_rate = len(shadow_errors) / len(shadowed) if shadowed else None
    insufficient_rate = len(insufficient) / total if total else None
    primary_p95 = _percentile(primary_latencies, 0.95)
    checks: dict[str, bool | None] = {
        "fallback_rate": fallback_rate is not None and fallback_rate <= max_fallback_rate,
        "primary_error_rate": (
            primary_error_rate is not None and primary_error_rate <= max_error_rate
        ),
        "shadow_error_rate": (
            shadow_error_rate is not None and shadow_error_rate <= max_error_rate
        ),
        "insufficient_result_rate": (
            insufficient_rate is not None and insufficient_rate <= max_insufficient_rate
        ),
        "primary_p95_latency": (
            None
            if max_primary_p95_ms is None or primary_p95 is None
            else primary_p95 <= max_primary_p95_ms
        ),
    }
    required_checks = [value for value in checks.values() if value is not None]
    enough_data = total >= min_events and len(shadowed) >= min_shadow_events
    return {
        "counts": {
            "events": total,
            "shadowed": len(shadowed),
            "paired": len(paired),
            "malformed_lines": malformed_lines,
            "primary_modes": dict(Counter(event.primary_mode for event in events)),
        },
        "rates": {
            "fallback": fallback_rate,
            "primary_error": primary_error_rate,
            "shadow_error": shadow_error_rate,
            "insufficient_results": insufficient_rate,
            "top1_agreement": (
                sum(bool(event.top1_agreement) for event in paired) / len(paired)
                if paired
                else None
            ),
            "mean_overlap_at_k": (
                mean(event.overlap_at_k for event in paired if event.overlap_at_k is not None)
                if paired
                else None
            ),
        },
        "latency_ms": {
            "primary_p50": _percentile(primary_latencies, 0.50),
            "primary_p95": primary_p95,
            "shadow_p50": _percentile(shadow_latencies, 0.50),
            "shadow_p95": _percentile(shadow_latencies, 0.95),
        },
        "promotion_gate": {
            "status": (
                "insufficient_data"
                if not enough_data
                else "pass"
                if all(required_checks)
                else "fail"
            ),
            "checks": checks,
            "thresholds": {
                "max_fallback_rate": max_fallback_rate,
                "max_error_rate": max_error_rate,
                "max_insufficient_rate": max_insufficient_rate,
                "max_primary_p95_ms": max_primary_p95_ms,
                "min_events": min_events,
                "min_shadow_events": min_shadow_events,
            },
        },
    }


def load_retrieval_summary(
    path: Path | None = None,
    *,
    max_fallback_rate: float = 0.01,
    max_error_rate: float = 0.01,
    max_insufficient_rate: float = 0.01,
    max_primary_p95_ms: float | None = None,
    min_events: int = 100,
    min_shadow_events: int = 30,
) -> dict[str, object]:
    events, malformed = read_retrieval_events(path)
    return summarize_retrieval_events(
        events,
        malformed_lines=malformed,
        max_fallback_rate=max_fallback_rate,
        max_error_rate=max_error_rate,
        max_insufficient_rate=max_insufficient_rate,
        max_primary_p95_ms=max_primary_p95_ms,
        min_events=min_events,
        min_shadow_events=min_shadow_events,
    )
