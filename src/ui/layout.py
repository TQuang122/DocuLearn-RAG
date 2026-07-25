from __future__ import annotations

import gradio as gr

from src.config import settings
from src.ui.callbacks import (
    ask_chat,
    clear_chat,
    delete_selected_docs,
    generate_flashcard_set_interactive,
    generate_quiz_set_interactive,
    summarize_documents,
)
from src.ui.content import (
    BRAND_HEADER_HTML,
    EMPTY_LIBRARY_HTML,
    INFO_NOTE_HTML,
    LIBRARY_HEADING_HTML,
    UPLOAD_HEADING_HTML,
    USAGE_MARKDOWN,
)
from src.ui.feature_panel import build_feature_panel
from src.ui.helpers import status_html, write_export
from src.ui.uploads import pages_for_selection, refresh_docs, upload_pdf


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="DocuLearn-RAG", fill_width=True, fill_height=True) as demo:
        gr.HTML(BRAND_HEADER_HTML, elem_classes="brand-shell")
        gr.HTML(INFO_NOTE_HTML, elem_classes="site-info-note")
        doc_map_state = gr.State({})

        with gr.Row(equal_height=False, elem_classes="main-layout", elem_id="workspace"):
            with gr.Column(scale=4, min_width=0, elem_classes="control-stack"):
                gr.HTML(UPLOAD_HEADING_HTML)
                upload = gr.File(
                    label="Drop PDFs here or browse files",
                    show_label=False,
                    file_types=[".pdf"],
                    file_count="multiple",
                    type="filepath",
                    elem_classes="upload-dropzone",
                )
                upload_btn = gr.Button(
                    "Upload & index",
                    variant="primary",
                    elem_classes="gen-btn",
                )
                upload_status = gr.HTML(status_html("Ready."))

                with gr.Accordion("Gemini API key", open=False):
                    gr.Markdown(
                        "Enter your API key to use Gemini. Get one at: "
                        "[Google AI Studio](https://aistudio.google.com/app/api-keys). ",
                        elem_classes="help-markdown",
                    )
                    gemini_key_input = gr.Textbox(
                        label="Gemini API Key",
                        type="password",
                        placeholder="AIza...",
                        lines=1,
                        max_lines=1,
                    )

                with gr.Accordion("User guide", open=False):
                    gr.Markdown(USAGE_MARKDOWN, elem_classes="help-markdown")

                with gr.Accordion("System info", open=False):
                    gr.Markdown(
                        f"""
                        - LLM model: `{settings.llm_model}`
                        - Embedding model: `{settings.embedding_model}`
                        - Collection: `{settings.qdrant_collection}`
                        - Data dir: `{settings.data_dir}`
                        - Storage dir: `{settings.storage_dir}`
                        """,
                        elem_classes="help-markdown",
                    )

            with gr.Column(scale=7, min_width=0, elem_classes="preview-col"):
                gr.HTML(LIBRARY_HEADING_HTML)
                with gr.Row(elem_classes="library-actions"):
                    refresh_btn = gr.Button("Refresh")
                    delete_btn = gr.Button("Delete selected", variant="stop")
                doc_summary = gr.HTML(EMPTY_LIBRARY_HTML, elem_classes="doc-summary")
                docs = gr.CheckboxGroup(label="Select documents", choices=[], value=[])
                page = gr.Dropdown(
                    label="Page (only applies when exactly 1 document is selected)",
                    choices=["(All pages)"],
                    value="(All pages)",
                )
                doc_list_md = gr.Markdown("")

        with gr.Tabs(selected="qa", elem_classes="learning-tabs") as _learning_tabs:
            with gr.Tab("Q&A", id="qa"):
                chatbot = gr.Chatbot(
                    elem_classes="qa-chat",
                    height=420,
                    show_label=False,
                )
                gr.Markdown(
                    "Type your question below and press **Enter** to chat about "
                    "your selected documents.",
                    elem_classes="feature-sub",
                )
                q = gr.Textbox(
                    label="",
                    show_label=False,
                    lines=1,
                    placeholder="Ask a question and press Enter…",
                    elem_classes="qa-input",
                )
                clear_chat_btn = gr.Button("Clear chat", size="sm", elem_classes="clear-chat-btn")
                with gr.Accordion("Advanced options", open=False):
                    k_ask = gr.Slider(1, 32, value=6, step=1, label="Top-k retrieval")
                with gr.Accordion("JSON debug", open=False):
                    ask_raw = gr.Code(label="", language="json", show_label=False)

            with gr.Tab("Summary", id="summary"):
                summary_panel = build_feature_panel(
                    description=(
                        "Generate a summary based on your selected documents "
                        "(and by page if only 1 document is selected)."
                    ),
                    button_label="Generate summary",
                    retrieval_count=10,
                    download_label="Download Markdown",
                )

            with gr.Tab("Quiz", id="quiz"):
                quiz_panel = build_feature_panel(
                    description="Create a quiz from your selected documents.",
                    button_label="Generate quiz",
                    retrieval_count=10,
                    download_label="Download Markdown",
                    item_count=(1, 30, 3, "Number of questions"),
                    interactive=True,
                )

            with gr.Tab("Flashcards", id="flashcards"):
                flashcard_panel = build_feature_panel(
                    description="Create flashcards from your selected documents for quick review.",
                    button_label="Generate flashcards",
                    retrieval_count=16,
                    download_label="Download Markdown",
                    item_count=(1, 40, 15, "Number of cards"),
                    interactive=True,
                )

        gr.HTML('<div class="footer-text"><span>DocuLearn-RAG</span></div>')

        refresh_btn.click(
            fn=refresh_docs,
            inputs=[],
            outputs=[docs, doc_map_state, page, doc_summary, doc_list_md],
            api_name="refresh_documents",
        )
        docs.change(
            fn=pages_for_selection,
            inputs=[doc_map_state, docs],
            outputs=[page],
            api_name="pages_for_selection",
        )
        upload_btn.click(
            fn=upload_pdf,
            inputs=[upload],
            outputs=[upload_status, docs, doc_map_state, page, doc_summary, doc_list_md],
            api_name="upload_pdf",
        )
        delete_btn.click(
            fn=delete_selected_docs,
            inputs=[docs],
            outputs=[docs, doc_summary],
            api_name="delete_selected_docs",
        ).then(
            fn=refresh_docs,
            inputs=[],
            outputs=[docs, doc_map_state, page, doc_summary, doc_list_md],
        )
        clear_chat_btn.click(
            fn=clear_chat,
            inputs=[],
            outputs=[chatbot],
            api_name="clear_chat",
        )
        q.submit(
            fn=ask_chat,
            inputs=[q, chatbot, k_ask, docs, page, gemini_key_input],
            outputs=[chatbot, ask_raw, q],
            api_name="ask_chat",
        )
        summary_panel.button.click(
            fn=summarize_documents,
            inputs=[
                summary_panel.query,
                summary_panel.retrieval_count,
                docs,
                page,
                gemini_key_input,
            ],
            outputs=[summary_panel.markdown, summary_panel.raw],
            api_name="summarize_documents",
        ).then(
            fn=lambda text: write_export(text, "summary.md"),
            inputs=[summary_panel.markdown],
            outputs=[summary_panel.download],
        )
        assert quiz_panel.item_count is not None
        assert quiz_panel.html is not None
        quiz_panel.button.click(
            fn=generate_quiz_set_interactive,
            inputs=[
                quiz_panel.query,
                quiz_panel.item_count,
                quiz_panel.retrieval_count,
                docs,
                page,
                gemini_key_input,
            ],
            outputs=[quiz_panel.html, quiz_panel.markdown, quiz_panel.raw],
            api_name="generate_quiz_set",
        ).then(
            fn=lambda text: write_export(text, "quiz.md"),
            inputs=[quiz_panel.markdown],
            outputs=[quiz_panel.download],
        )
        assert flashcard_panel.item_count is not None
        assert flashcard_panel.html is not None
        flashcard_panel.button.click(
            fn=generate_flashcard_set_interactive,
            inputs=[
                flashcard_panel.query,
                flashcard_panel.item_count,
                flashcard_panel.retrieval_count,
                docs,
                page,
                gemini_key_input,
            ],
            outputs=[flashcard_panel.html, flashcard_panel.markdown, flashcard_panel.raw],
            api_name="generate_flashcard_set",
        ).then(
            fn=lambda text: write_export(text, "flashcards.md"),
            inputs=[flashcard_panel.markdown],
            outputs=[flashcard_panel.download],
        )

    return demo
