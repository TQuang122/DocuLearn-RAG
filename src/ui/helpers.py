from __future__ import annotations

import html
from pathlib import Path
from uuid import uuid4

import gradio as gr

from src.config import settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSS = (PROJECT_ROOT / "static" / "style.css").read_text(encoding="utf-8")
PROGRESS = gr.Progress()

THEME = gr.themes.Base().set(
    background_fill_primary="#08090a",
    background_fill_primary_dark="#08090a",
    background_fill_secondary="#0f1012",
    background_fill_secondary_dark="#0f1012",
    block_background_fill="transparent",
    block_background_fill_dark="transparent",
    block_border_color="transparent",
    block_border_color_dark="transparent",
    block_border_width="0px",
    body_background_fill="#08090a",
    body_background_fill_dark="#08090a",
    body_text_color="#f7f8f8",
    body_text_color_dark="#f7f8f8",
    input_background_fill="#17181b",
    input_background_fill_dark="#17181b",
    input_border_color="rgba(255,255,255,.10)",
    input_border_color_dark="rgba(255,255,255,.10)",
    input_placeholder_color="#7c818a",
    input_placeholder_color_dark="#7c818a",
    button_primary_background_fill="#4f5bc0",
    button_primary_background_fill_dark="#4f5bc0",
    button_primary_background_fill_hover="#5a65ca",
    button_primary_background_fill_hover_dark="#5a65ca",
    button_primary_border_color="#7170ff",
    button_primary_border_color_dark="#7170ff",
    button_primary_border_color_hover="#828fff",
    button_primary_border_color_hover_dark="#828fff",
    button_primary_text_color="#f7f8f8",
    button_primary_text_color_dark="#f7f8f8",
    button_primary_text_color_hover="#ffffff",
    button_primary_text_color_hover_dark="#ffffff",
    button_secondary_background_fill="#17181b",
    button_secondary_background_fill_dark="#17181b",
    button_secondary_background_fill_hover="#202126",
    button_secondary_background_fill_hover_dark="#202126",
    button_secondary_border_color="rgba(255,255,255,.10)",
    button_secondary_border_color_dark="rgba(255,255,255,.10)",
    button_secondary_text_color="#c6c9d0",
    button_secondary_text_color_dark="#c6c9d0",
)


def status_html(message: str, *, trusted: bool = False) -> str:
    body = message if trusted else html.escape(message, quote=False)
    return f'<div class="status-bar">{body}</div>'


def result_markdown(
    *, value: str = "", elem_classes: str | list[str] = "result-markdown"
) -> gr.Markdown:
    return gr.Markdown(value=value, elem_classes=elem_classes)


def int_value(value: object) -> int:
    if isinstance(value, (int, float, str)):
        return int(value)
    return 0


def write_export(md_text: str, filename: str) -> str | None:
    if not md_text or "Lỗi:" in md_text or md_text.startswith("Error:"):
        return None
    stem = Path(filename).stem
    output_path = settings.export_dir / f"{stem}-{uuid4().hex[:12]}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md_text, encoding="utf-8")
    return str(output_path)


def export_download(md_text: str, filename: str) -> dict:
    path = write_export(md_text, filename)
    return gr.update(value=path, visible=path is not None)
