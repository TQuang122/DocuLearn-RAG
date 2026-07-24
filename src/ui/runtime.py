from __future__ import annotations

import gradio as gr

from src.config import settings
from src.ui.helpers import CSS, THEME


def launch_demo(demo: gr.Blocks) -> None:
    settings.export_dir.mkdir(parents=True, exist_ok=True)
    auth = None
    if settings.gradio_username and settings.gradio_password:
        auth = (settings.gradio_username, settings.gradio_password)
    demo.queue(default_concurrency_limit=2).launch(
        allowed_paths=[str(settings.export_dir.resolve())],
        auth=auth,
        css=CSS,
        max_file_size=settings.max_upload_bytes,
        server_name=settings.server_name,
        server_port=settings.server_port,
        theme=THEME,
    )
