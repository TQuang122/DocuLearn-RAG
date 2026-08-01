from __future__ import annotations

from src.ui import runtime


def test_embedding_warmup_is_disabled_by_default(monkeypatch) -> None:
    monkeypatch.setattr(runtime.settings, "embedding_warmup_enabled", False)
    monkeypatch.setattr(
        runtime,
        "warm_up_embeddings",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected warmup")),
    )

    assert runtime.warm_up_if_enabled() is False


def test_embedding_warmup_runs_when_enabled(monkeypatch) -> None:
    monkeypatch.setattr(runtime.settings, "embedding_warmup_enabled", True)
    monkeypatch.setattr(runtime, "warm_up_embeddings", lambda: 12.5)

    assert runtime.warm_up_if_enabled() is True


def test_embedding_warmup_failure_does_not_block_startup(monkeypatch) -> None:
    monkeypatch.setattr(runtime.settings, "embedding_warmup_enabled", True)
    monkeypatch.setattr(
        runtime,
        "warm_up_embeddings",
        lambda: (_ for _ in ()).throw(RuntimeError("model unavailable")),
    )

    assert runtime.warm_up_if_enabled() is False
