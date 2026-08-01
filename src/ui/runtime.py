from __future__ import annotations

import gradio as gr
from loguru import logger

from src.config import settings
from src.embeddings import warm_up_embeddings
from src.ui.helpers import CSS, THEME
from src.ui.interactive import INTERACTIVE_HEAD_HTML


def warm_up_if_enabled() -> bool:
    if not settings.embedding_warmup_enabled:
        return False
    try:
        latency_ms = warm_up_embeddings()
    except Exception as exc:
        logger.warning("embedding_warmup_failed error={}", type(exc).__name__)
        return False
    logger.info("embedding_warmup_completed latency_ms={:.3f}", latency_ms)
    return True


def launch_demo(demo: gr.Blocks) -> None:
    settings.export_dir.mkdir(parents=True, exist_ok=True)
    _ = warm_up_if_enabled()
    auth = None
    if settings.gradio_username and settings.gradio_password:
        auth = (settings.gradio_username, settings.gradio_password)
    _ = demo.queue(default_concurrency_limit=2).launch(
        allowed_paths=[str(settings.export_dir.resolve())],
        auth=auth,
        css=CSS,
        head=INTERACTIVE_HEAD_HTML,
        max_file_size=settings.max_upload_bytes,
        server_name=settings.server_name,
        server_port=settings.server_port,
        theme=THEME,
    )
