from __future__ import annotations

from pathlib import Path
from typing import cast

import gradio as gr

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


def test_launch_preserves_the_app_with_custom_routes(monkeypatch, tmp_path: Path) -> None:
    class DemoStub:
        def __init__(self) -> None:
            self.app = object()
            self.queued_app = object()
            self.launch_kwargs: dict[str, object] = {}

        def queue(self, **_: object) -> DemoStub:
            self.app = self.queued_app
            return self

        def launch(self, **kwargs: object) -> None:
            self.launch_kwargs = kwargs

    demo = DemoStub()
    routed_apps: list[object] = []
    monkeypatch.setattr(runtime.settings, "export_dir", tmp_path)
    monkeypatch.setattr(runtime, "warm_up_if_enabled", lambda: False)
    monkeypatch.setattr(runtime, "add_monitoring_route", routed_apps.append)

    runtime.launch_demo(cast(gr.Blocks, demo))

    assert routed_apps == [demo.queued_app]
    assert demo.launch_kwargs["_app"] is demo.queued_app
