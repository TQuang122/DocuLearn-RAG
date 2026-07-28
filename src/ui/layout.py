from __future__ import annotations

import gradio as gr

from src.config import settings
from src.ui.callbacks import (
    ask_chat,
    clear_chat,
    delete_selected_docs,
    generate_flashcard_set_interactive,
    generate_quiz_set_interactive,
    prepare_delete,
    reset_delete_confirmation,
    summarize_documents,
)
from src.ui.content import (
    BRAND_HEADER_HTML,
    EMPTY_LIBRARY_HTML,
    FLASHCARD_HEADING_HTML,
    INFO_NOTE_HTML,
    LIBRARY_HEADING_HTML,
    QA_HEADING_HTML,
    QUIZ_HEADING_HTML,
    SUMMARY_HEADING_HTML,
    UPLOAD_HEADING_HTML,
    USAGE_MARKDOWN,
)
from src.ui.feature_panel import build_feature_panel
from src.ui.helpers import export_download, status_html
from src.ui.uploads import pages_for_selection, refresh_docs, scope_summary_html, upload_pdf


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="DocuLearn-RAG",
        fill_width=True,
        fill_height=True,
    ) as demo:
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
                    elem_classes=["upload-dropzone", "source-file-picker"],
                )
                upload_btn = gr.Button(
                    "Upload & index PDFs",
                    variant="primary",
                    elem_classes=["gen-btn", "source-index-btn"],
                )
                upload_status = gr.HTML(
                    status_html("Ready to index."),
                    elem_classes="source-status",
                )

                with gr.Accordion(
                    "Gemini API key",
                    open=False,
                    elem_classes="setup-accordion",
                ):
                    gr.Markdown(
                        (
                            "**Status:** Gemini is configured via the environment. "
                            "You can leave this field empty.\n\n"
                            if settings.gemini_api_key
                            else "**Status:** No Gemini key is configured yet. "
                            "Enter one for this session or set `GEMINI_API_KEY`.\n\n"
                        )
                        + "Enter a key to use Gemini. Get one at: "
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

                with gr.Accordion(
                    "User guide",
                    open=False,
                    elem_classes="setup-accordion",
                ):
                    gr.Markdown(USAGE_MARKDOWN, elem_classes="help-markdown")

                with gr.Accordion(
                    "System info",
                    open=False,
                    elem_classes="setup-accordion",
                ):
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
                    refresh_btn = gr.Button(
                        "Refresh",
                        elem_classes="library-refresh-btn",
                    )
                    delete_btn = gr.Button(
                        "Delete selected",
                        variant="stop",
                        elem_classes="library-delete-btn",
                    )
                delete_prompt = gr.HTML(visible=False, elem_classes="delete-confirmation")
                with gr.Row(
                    visible=False,
                    elem_classes="delete-confirmation-actions",
                ) as delete_confirmation_actions:
                    confirm_delete_btn = gr.Button(
                        "Confirm delete",
                        variant="stop",
                        elem_classes="library-delete-confirm-btn",
                    )
                    cancel_delete_btn = gr.Button(
                        "Cancel",
                        elem_classes="library-delete-cancel-btn",
                    )
                doc_summary = gr.HTML(EMPTY_LIBRARY_HTML, elem_classes="doc-summary")
                docs = gr.CheckboxGroup(
                    label="Documents",
                    info="Choose one or more PDFs to define the learning scope.",
                    choices=[],
                    value=[],
                    elem_classes="library-document-picker",
                )
                page = gr.Dropdown(
                    label="Page scope",
                    info="Choose all pages or focus on one page after selecting a single PDF.",
                    choices=["All pages"],
                    value="All pages",
                    interactive=False,
                    elem_classes="library-page-picker",
                )
                doc_list_md = gr.Markdown(
                    "",
                    visible=False,
                    elem_classes="library-file-index",
                )

        scope_summary = gr.HTML(
            scope_summary_html([], "All pages"),
            elem_classes="scope-summary-shell",
        )
        with gr.Tabs(selected="qa", elem_classes="learning-tabs") as _learning_tabs:
            with gr.Tab(
                "Q&A",
                id="qa",
                elem_id="qa",
                elem_classes=["qa-tab", "qa-stage"],
            ):
                gr.HTML(QA_HEADING_HTML, elem_classes="qa-heading-shell")
                chatbot = gr.Chatbot(
                    elem_classes="qa-chat",
                    height=380,
                    min_height=260,
                    max_height=520,
                    show_label=False,
                    layout="bubble",
                    buttons=["copy_all"],
                    feedback_options=[],
                    placeholder="Ask a question to start your research thread.",
                )
                with gr.Group(elem_classes="qa-composer"):
                    gr.HTML(
                        '<div class="qa-composer-hint"><strong>Selected scope</strong>'
                        '<span aria-hidden="true"> · </span>'
                        'Indexed documents and page filters apply automatically.</div>',
                        elem_classes="qa-composer-note",
                    )
                    with gr.Row(elem_classes="qa-composer-row"):
                        q = gr.Textbox(
                            label="Ask a question",
                            show_label=False,
                            lines=1,
                            placeholder="Ask about your selected documents…",
                            submit_btn=False,
                            scale=1,
                            min_width=0,
                            elem_id="qa-query",
                            elem_classes="qa-input",
                        )
                        send_btn = gr.Button(
                            "Send",
                            variant="primary",
                            scale=0,
                            min_width=56,
                            elem_id="qa-send",
                            elem_classes="qa-send-btn",
                        )
                    gr.HTML(
                        '<div class="qa-composer-meta">Press <strong>Enter</strong>'
                        ' to send <span aria-hidden="true">·</span> '
                        'source passages included.</div>',
                        elem_classes="qa-composer-note",
                    )
                with gr.Row(elem_classes="qa-actions"):
                    gr.HTML(
                        '<div class="qa-grounding-note">Grounded in your selected PDF scope.</div>',
                        elem_classes="qa-composer-note",
                    )
                    clear_chat_btn = gr.Button(
                        "Clear conversation",
                        size="sm",
                        elem_classes="clear-chat-btn",
                    )
                with gr.Accordion(
                    "Advanced options, JSON debug",
                    open=False,
                    elem_classes="setup-accordion",
                ):
                    k_ask = gr.Slider(
                        1,
                        32,
                        value=6,
                        step=1,
                        label="Top-k retrieval",
                        elem_classes="feature-slider",
                    )
                    ask_raw = gr.Code(label="", language="json", show_label=False)

            with gr.Tab("Summary", id="summary", elem_id="summary"):
                summary_panel = build_feature_panel(
                    description=(
                        "Generate a summary based on your selected documents "
                        "(and by page if only 1 document is selected)."
                    ),
                    button_label="Generate summary",
                    retrieval_count=10,
                    download_label="Download Markdown",
                    heading_html=SUMMARY_HEADING_HTML,
                    panel_class="summary-panel",
                )

            with gr.Tab("Quiz", id="quiz", elem_id="quiz"):
                quiz_panel = build_feature_panel(
                    description="Create a quiz from your selected documents.",
                    button_label="Generate quiz",
                    retrieval_count=10,
                    download_label="Download Markdown",
                    item_count=(1, 30, 3, "Number of questions"),
                    interactive=True,
                    wrap_markdown=True,
                    heading_html=QUIZ_HEADING_HTML,
                    panel_class="quiz-panel",
                )

            with gr.Tab("Flashcards", id="flashcards", elem_id="flashcards"):
                flashcard_panel = build_feature_panel(
                    description="Create flashcards from your selected documents for quick review.",
                    button_label="Generate flashcards",
                    retrieval_count=16,
                    download_label="Download Markdown",
                    item_count=(1, 40, 15, "Number of cards"),
                    interactive=True,
                    wrap_markdown=True,
                    heading_html=FLASHCARD_HEADING_HTML,
                    panel_class="flashcard-panel",
                )

        gr.HTML('<div class="footer-text"><span>DocuLearn-RAG</span></div>')

        refresh_btn.click(
            fn=refresh_docs,
            inputs=[],
            outputs=[docs, doc_map_state, page, doc_summary, doc_list_md],
            api_name="refresh_documents",
        ).then(
            fn=scope_summary_html,
            inputs=[docs, page],
            outputs=[scope_summary],
            api_name="scope_summary_refresh",
        )
        docs.change(
            fn=pages_for_selection,
            inputs=[doc_map_state, docs],
            outputs=[page],
            api_name="pages_for_selection",
        ).then(
            fn=scope_summary_html,
            inputs=[docs, page],
            outputs=[scope_summary],
            api_name="scope_summary_selection",
        )
        upload_btn.click(
            fn=upload_pdf,
            inputs=[upload],
            outputs=[upload_status, docs, doc_map_state, page, doc_summary, doc_list_md],
            api_name="upload_pdf",
        ).then(
            fn=scope_summary_html,
            inputs=[docs, page],
            outputs=[scope_summary],
            api_name="scope_summary_upload",
        )
        page.change(
            fn=scope_summary_html,
            inputs=[docs, page],
            outputs=[scope_summary],
            api_name="scope_summary",
        )
        delete_btn.click(
            fn=prepare_delete,
            inputs=[docs],
            outputs=[
                delete_prompt,
                confirm_delete_btn,
                cancel_delete_btn,
                delete_btn,
                delete_confirmation_actions,
            ],
            api_name="prepare_delete",
        )
        confirm_delete_btn.click(
            fn=delete_selected_docs,
            inputs=[docs],
            outputs=[docs, doc_summary],
            api_name="delete_selected_docs",
        ).then(
            fn=refresh_docs,
            inputs=[],
            outputs=[docs, doc_map_state, page, doc_summary, doc_list_md],
        ).then(
            fn=scope_summary_html,
            inputs=[docs, page],
            outputs=[scope_summary],
            api_name="scope_summary_after_delete",
        ).then(
            fn=reset_delete_confirmation,
            inputs=[],
            outputs=[
                delete_prompt,
                confirm_delete_btn,
                cancel_delete_btn,
                delete_btn,
                delete_confirmation_actions,
            ],
        )
        cancel_delete_btn.click(
            fn=reset_delete_confirmation,
            inputs=[],
            outputs=[
                delete_prompt,
                confirm_delete_btn,
                cancel_delete_btn,
                delete_btn,
                delete_confirmation_actions,
            ],
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
        send_btn.click(
            fn=ask_chat,
            inputs=[q, chatbot, k_ask, docs, page, gemini_key_input],
            outputs=[chatbot, ask_raw, q],
            api_name="ask_chat_button",
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
            fn=lambda text: export_download(text, "summary.md"),
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
            fn=lambda text: export_download(text, "quiz.md"),
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
            fn=lambda text: export_download(text, "flashcards.md"),
            inputs=[flashcard_panel.markdown],
            outputs=[flashcard_panel.download],
        )

    return demo
