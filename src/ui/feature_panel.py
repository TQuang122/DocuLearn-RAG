from __future__ import annotations

from typing import NamedTuple

import gradio as gr

from src.ui.helpers import result_markdown


class FeaturePanel(NamedTuple):
    query: gr.Textbox
    button: gr.Button
    item_count: gr.Slider | None
    retrieval_count: gr.Slider
    raw: gr.Code
    markdown: gr.Markdown
    download: gr.File
    html: gr.HTML | None


def build_feature_panel(
    *,
    description: str,
    button_label: str,
    retrieval_count: int,
    download_label: str,
    item_count: tuple[int, int, int, str] | None = None,
    interactive: bool = False,
) -> FeaturePanel:
    gr.Markdown(description, elem_classes="feature-sub")
    with gr.Row(equal_height=False, elem_classes="feature-layout"):
        with gr.Column(scale=4, min_width=0, elem_classes="feature-controls"):
            query = gr.Textbox(label="Topic (optional)", lines=1)
            button = gr.Button(
                button_label,
                variant="primary",
                elem_classes="gen-btn",
            )
            with gr.Accordion("Advanced options", open=False):
                item_count_component = None
                if item_count is not None:
                    minimum, maximum, value, label = item_count
                    item_count_component = gr.Slider(
                        minimum,
                        maximum,
                        value=value,
                        step=1,
                        label=label,
                    )
                k = gr.Slider(
                    1,
                    64,
                    value=retrieval_count,
                    step=1,
                    label="Retrieval count (k)",
                )
            with gr.Accordion("JSON debug", open=False):
                raw = gr.Code(label="", language="json", show_label=False)
        with gr.Column(scale=8, min_width=0, elem_classes="feature-output"):
            html_component = (
                gr.HTML(value="", sanitize_html=False, elem_classes="interactive-output")
                if interactive
                else None
            )
            markdown = result_markdown()
            download = gr.File(label=download_label, interactive=False)
    return FeaturePanel(
        query,
        button,
        item_count_component,
        k,
        raw,
        markdown,
        download,
        html_component,
    )
