from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.ui.content import BRAND_HEADER_HTML
from src.ui.helpers import CSS
from src.ui.interactive import INTERACTIVE_HEAD_HTML
from src.ui.uploads import (
    pages_for_selection,
    read_uploaded_pdfs,
    refresh_docs,
    scope_summary_html,
    upload_pdf,
)


def test_uploaded_path_is_normalized_to_bytes(tmp_path: Path) -> None:
    upload = tmp_path / "notes.pdf"
    upload.write_bytes(b"%PDF-1.7")

    payloads = read_uploaded_pdfs(str(upload))

    assert payloads == [(b"%PDF-1.7", "notes.pdf")]


def test_single_document_selection_enables_its_pages() -> None:
    update = pages_for_selection({"notes.pdf": {"pages": [1, 3]}}, ["notes.pdf"])

    assert update["choices"] == ["All pages", "1", "3"]
    assert update["interactive"] is True


def test_empty_upload_keeps_six_output_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.ui.uploads.refresh_docs",
        lambda: (
            {"choices": [], "value": []},
            {},
            {"choices": ["All pages"], "value": "All pages"},
            "",
            "",
        ),
    )

    outputs = upload_pdf(None)

    assert len(outputs) == 6


def test_demo_keeps_learning_event_contract() -> None:
    from app import demo

    config = demo.get_config_file()
    contracts = {
        dependency["api_name"]: (
            len(dependency["inputs"]),
            len(dependency["outputs"]),
        )
        for dependency in config["dependencies"]
        if dependency["backend_fn"]
    }

    assert len(config["components"]) == 102
    assert len(config["dependencies"]) == 22
    assert contracts["refresh_documents"] == (0, 5)
    assert contracts["pages_for_selection"] == (2, 1)
    assert contracts["upload_pdf"] == (1, 6)
    assert contracts["ask_chat"] == (6, 3)
    assert contracts["ask_chat_button"] == (6, 3)
    assert contracts["summarize_documents"] == (5, 2)
    assert contracts["generate_quiz_set"] == (6, 3)
    assert contracts["generate_flashcard_set"] == (6, 3)
    assert contracts["clear_chat"] == (0, 1)
    assert contracts["delete_selected_docs"] == (1, 2)
    assert contracts["prepare_delete"] == (1, 5)
    assert contracts["reset_delete_confirmation"] == (0, 5)
    assert contracts["reset_delete_confirmation_1"] == (0, 5)
    assert contracts["scope_summary_refresh"] == (2, 1)
    assert contracts["scope_summary_selection"] == (2, 1)
    assert contracts["scope_summary_upload"] == (2, 1)
    assert contracts["scope_summary"] == (2, 1)
    assert contracts["scope_summary_after_delete"] == (2, 1)
    assert contracts["lambda"] == (1, 1)
    assert contracts["lambda_1"] == (1, 1)
    assert contracts["lambda_2"] == (1, 1)


def test_demo_isolated_from_host_theme_css() -> None:
    from app import demo

    assert demo.elem_id == "doculearn-app"
    assert ".gradio-container {" in CSS
    assert "isolation: isolate" in CSS
    assert "html body #root > #doculearn-app::before" in CSS
    assert "--spacing-xxl: 16px !important" in CSS
    assert "margin-inline: auto !important" in CSS
    assert ".gradio-container .panel-heading h2" in CSS
    assert "font-size: 1.25rem !important" in CSS
    assert ".gradio-container *" in CSS
    assert "#doculearn-app,\n.gradio-container" in CSS
    assert "display=swap" in CSS
    assert "--type-body: 1rem" in CSS
    assert "--type-display: clamp(2.5rem, 5.6vw, 4.5rem)" in CSS
    assert ".product-header" in CSS and "height: 68px" in CSS
    assert "height: 368px !important" in CSS
    assert "min-height: 456px" in CSS
    assert ".gradio-container .hero-copy h1" in CSS
    assert "margin: 0 !important" in CSS
    assert ".gradio-container .main-layout" in CSS
    assert "width: calc(100% - var(--space-3) - var(--space-3)) !important" in CSS
    assert "--accent-info: #76a7ff" in CSS
    assert ".gradio-container .control-stack .panel-icon" in CSS
    assert ".gradio-container .summary-status span" in CSS
    assert "_syncAccessibility" in INTERACTIVE_HEAD_HTML
    assert 'data-doculearn-tabindex' in INTERACTIVE_HEAD_HTML
    assert 'source-file-picker button[aria-label*="upload" i]' in INTERACTIVE_HEAD_HTML
    assert "attributeFilter: ['aria-hidden']" in INTERACTIVE_HEAD_HTML


def test_setup_accordions_share_one_scoped_component_style() -> None:
    from app import demo

    config = demo.get_config_file()
    labels = {
        component["props"]["label"]
        for component in config["components"]
        if component["type"] == "accordion"
        and "setup-accordion" in component["props"]["elem_classes"]
    }

    assert labels == {
        "Gemini API key",
        "User guide",
        "System info",
        "Advanced options, JSON debug",
        "Raw markdown",
    }


def test_document_workspace_uses_scoped_ui_components() -> None:
    from app import demo

    config = demo.get_config_file()
    scoped_classes = {
        css_class
        for component in config["components"]
        for css_class in component["props"].get("elem_classes", [])
    }

    assert {
        "source-file-picker",
        "source-index-btn",
        "library-document-picker",
        "library-page-picker",
        "library-refresh-btn",
        "library-delete-btn",
    } <= scoped_classes


def test_learning_flow_copy_and_credential_guidance_are_consistent() -> None:
    from src.ui.content import INFO_NOTE_HTML, USAGE_MARKDOWN

    assert "Upload &amp; index PDFs" in INFO_NOTE_HTML
    assert "Upload & index PDFs" in USAGE_MARKDOWN


def test_missing_gemini_key_error_names_recovery_action() -> None:
    from src.ui.callbacks import _generation_error

    error = _generation_error(RuntimeError("Missing Gemini API key."))

    assert "Gemini API key is missing" in str(error)
    assert "GEMINI_API_KEY" in str(error)


def test_delete_confirmation_escapes_names_and_preserves_restore_path() -> None:
    from src.ui.callbacks import prepare_delete

    prompt, confirm, cancel, delete, actions = prepare_delete(["<notes>.pdf"])

    assert "&lt;notes&gt;.pdf" in prompt
    assert confirm["visible"] is True
    assert cancel["visible"] is True
    assert delete["visible"] is False
    assert actions["visible"] is True


def test_scope_summary_shows_active_documents_page_and_ready_state() -> None:
    summary = scope_summary_html(["notes.pdf", "paper.pdf"], "3")

    assert "Indexed &amp; ready" in summary
    assert "2 documents" in summary
    assert "notes.pdf, paper.pdf" in summary
    assert "Page: 3" in summary


def test_upload_dropzone_keeps_internal_icon_buttons_scoped() -> None:
    assert ".upload-dropzone button {" not in CSS
    assert ".upload-dropzone > button:not(.icon-button)" in CSS
    assert ".source-file-picker .icon-button-wrapper .icon-button" in CSS


def test_learning_tabs_target_gradio_six_tablist_dom() -> None:
    assert ".learning-tabs .tab-nav" not in CSS
    assert '.learning-tabs .tab-container[role="tablist"] button' in CSS
    assert "min-height: 44px !important" in CSS
    assert "height: 44px !important" in CSS
    assert 'button[data-tab-id="flashcards"]::before' in CSS
    assert ".qa-tab.qa-stage" in CSS
    assert '[aria-label*="delete" i]' in CSS


def test_qa_composer_uses_flat_html_notes() -> None:
    from app import demo

    config = demo.get_config_file()
    html_classes = {
        css_class
        for component in config["components"]
        if component["type"] == "html"
        for css_class in component["props"].get("elem_classes", [])
    }

    assert "qa-composer-note" in html_classes
    assert ".qa-composer .qa-composer-note > div" in CSS
    assert "height: 56px !important" in CSS
    assert "resize: none !important" in CSS
    assert "flex: 1 1 auto !important" in CSS
    assert "width: 100% !important" in CSS
    assert "max-width: none !important" in CSS
    assert "max-width: 56px !important" in CSS
    textbox = next(
        component
        for component in config["components"]
        if component["type"] == "textbox"
        and "qa-input" in component["props"].get("elem_classes", [])
    )
    assert textbox["props"]["submit_btn"] is False
    assert textbox["props"]["scale"] == 1
    assert textbox["props"]["min_width"] == 0
    assert textbox["props"]["elem_id"] == "qa-query"
    assert "#qa-query" in CSS
    assert ".qa-composer-row" in CSS
    send_button = next(
        component
        for component in config["components"]
        if component["type"] == "button"
        and "qa-send-btn" in component["props"].get("elem_classes", [])
    )
    assert send_button["props"]["elem_id"] == "qa-send"
    assert ".qa-chat .message.user .prose" in CSS


def test_summary_panel_has_focused_brief_surface() -> None:
    from app import demo

    config = demo.get_config_file()
    html_values = [
        component["props"].get("value", "")
        for component in config["components"]
        if component["type"] == "html"
    ]
    assert any("Build a focused summary" in value for value in html_values)
    assert ".summary-heading" in CSS
    assert ".summary-presets" in CSS
    assert ".summary-result" in CSS


def test_quiz_panel_has_focused_learning_heading() -> None:
    from app import demo

    config = demo.get_config_file()
    html_values = [
        component["props"].get("value", "")
        for component in config["components"]
        if component["type"] == "html"
    ]
    assert any("Test your understanding" in value for value in html_values)
    assert any("Practice-ready" in value for value in html_values)
    assert ".quiz-heading" in CSS
    assert ".quiz-panel > .feature-sub" in CSS


def test_flashcard_panel_has_spaced_repetition_heading() -> None:
    from app import demo

    config = demo.get_config_file()
    html_values = [
        component["props"].get("value", "")
        for component in config["components"]
        if component["type"] == "html"
    ]
    assert any("Review what you know" in value for value in html_values)
    assert any("Study-ready" in value for value in html_values)
    assert ".flashcard-heading" in CSS
    assert ".flashcard-panel > .feature-sub" in CSS


def test_hero_typography_is_scoped_against_host_theme_overrides() -> None:
    assert ".gradio-container .hero-copy h1" in CSS
    assert "font-size: clamp(2.5rem, 5.6vw, 4.5rem) !important" in CSS
    assert ".gradio-container .eyebrow" in CSS
    assert "background-image: none !important" in CSS
    assert "#root > div" in CSS
    assert "body::before" in CSS


def test_qa_heading_typography_is_scoped_against_host_theme_overrides() -> None:
    assert ".gradio-container .qa-eyebrow" in CSS
    assert ".gradio-container .qa-heading h2" in CSS
    assert ".gradio-container .qa-heading p" in CSS
    assert "color: var(--accent-hover) !important" in CSS
    assert "color: var(--text-tertiary) !important" in CSS


def test_feature_heading_typography_is_scoped_against_host_theme_overrides() -> None:
    for selector in (
        ".gradio-container .summary-heading h2",
        ".gradio-container .quiz-heading h2",
        ".gradio-container .flashcard-heading h2",
    ):
        assert selector in CSS
    assert ".gradio-container .summary-eyebrow" in CSS
    assert ".gradio-container .quiz-eyebrow" in CSS
    assert ".gradio-container .flashcard-eyebrow" in CSS
    assert ".gradio-container .flashcard-mode-note" in CSS


def test_workflow_strip_typography_is_scoped_against_host_theme_overrides() -> None:
    assert ".gradio-container .workflow-strip" in CSS
    assert ".gradio-container .workflow-label" in CSS
    assert ".gradio-container .workflow-intro strong" in CSS
    assert ".gradio-container .workflow-steps li" in CSS
    assert ".gradio-container .workflow-steps span" in CSS


def test_flashcard_surface_keeps_navigation_controls_only() -> None:
    from src.schemas import Flashcard, FlashcardSet
    from src.ui.interactive import render_flashcard_html

    cards = FlashcardSet(
        scope="corpus",
        cards=[Flashcard(front="Front", back="Back")],
        citations=[],
    )
    markup = render_flashcard_html(cards)

    assert 'id="fc-prev"' in markup
    assert 'class="fc-shuffle-btn"' in markup
    assert 'id="fc-next"' in markup
    assert "How well did you remember?" not in markup
    assert "Study mode" not in markup
    assert "Again" not in markup
    assert "Got it" not in markup


def test_flashcard_surface_keeps_flip_hint_and_sources() -> None:
    from src.schemas import Citation, Flashcard, FlashcardSet
    from src.ui.interactive import render_flashcard_html

    cards = FlashcardSet(
        scope="corpus",
        cards=[Flashcard(front="Front", back="Back", source_markers=["S1"])],
        citations=[
            Citation(
                source_index=1,
                source_marker="S1",
                filename="notes.pdf",
                page=2,
                source_text="Passage",
                chunk_id="notes:2:1",
            )
        ],
    )
    markup = render_flashcard_html(cards)

    assert "Tap to reveal" in markup
    assert "notes.pdf" in markup
    assert ".fc-sources" in CSS


def test_advanced_sliders_use_scoped_component_style() -> None:
    from app import demo

    config = demo.get_config_file()
    slider_labels = {
        component["props"]["label"]
        for component in config["components"]
        if component["type"] == "slider"
        and "feature-slider" in component["props"]["elem_classes"]
    }

    assert slider_labels == {
        "Top-k retrieval",
        "Retrieval count (k)",
        "Number of questions",
        "Number of cards",
    }


def test_quiz_surface_uses_single_question_navigation() -> None:
    from src.schemas import QuizItem, QuizSet
    from src.ui.interactive import render_quiz_html

    quiz = QuizSet(
        scope="corpus",
        items=[
            QuizItem(
                question="Which answer is correct?",
                options=["A", "B", "C", "D"],
                correct_index=0,
                explanation="The first option is correct.",
            )
        ],
        citations=[],
    )
    markup = render_quiz_html(quiz)

    assert 'id="iq-prev"' in markup
    assert 'id="iq-next"' in markup
    assert "IQ_PREV" in markup
    assert "IQ_NEXT" in markup
    assert ".iq-question-index" in CSS
    assert ".iq-next-btn" in CSS


def test_quiz_surface_includes_result_retry_and_sources() -> None:
    from src.schemas import Citation, QuizItem, QuizSet
    from src.ui.interactive import render_quiz_html

    quiz = QuizSet(
        scope="corpus",
        items=[
            QuizItem(
                question="Which answer is correct?",
                options=["A", "B", "C", "D"],
                correct_index=0,
                explanation="The first option is correct.",
                source_markers=["S1"],
            )
        ],
        citations=[
            Citation(
                source_index=1,
                source_marker="S1",
                filename="notes.pdf",
                page=2,
                source_text="Passage text",
                chunk_id="notes:2:1",
            )
        ],
    )
    markup = render_quiz_html(quiz)

    assert "Retry incorrect" in markup
    assert "iq-accuracy" in markup
    assert "notes.pdf" in markup
    assert "Passage text" in markup
    assert "IQ_RETRY_INCORRECT" in markup
    assert ".iq-sources" in CSS


def test_quiz_surface_supports_exam_mode_and_keyboard_shortcuts() -> None:
    from src.schemas import QuizItem, QuizSet
    from src.ui.interactive import render_quiz_html

    quiz = QuizSet(
        scope="corpus",
        items=[
            QuizItem(
                question="Which answer is correct?",
                options=["A", "B", "C", "D"],
                correct_index=0,
                explanation="The first option is correct.",
                difficulty="Hard",
            )
        ],
        citations=[],
    )
    markup = render_quiz_html(quiz)

    assert 'data-mode="exam"' in markup
    assert "IQ_SET_MODE" in markup
    assert "1–4 answer" in markup
    assert '"difficulty": "Hard"' in markup
    assert ".iq-modebar" in CSS
    assert ".iq-difficulty" in CSS


def test_library_refresh_keeps_page_scope_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.ui.uploads.list_documents",
        lambda: [
            {
                "filename": "notes.pdf",
                "chunk_count": 3,
                "pages": [1, 2],
            }
        ],
    )

    _, _, page_update, summary, _ = refresh_docs()

    assert page_update["interactive"] is False
    assert '<div class="library-stat"><strong>1</strong><span>File</span></div>' in summary


def test_refactor_preserves_stylesheet() -> None:
    digest = hashlib.sha256(CSS.encode()).hexdigest()

    assert digest == "5e226fd37fda5f72c102d139216b454413dca7d9fbcbd3343a5987ae7ae3bc2a"


def test_mobile_heading_keeps_word_boundary_when_break_is_hidden() -> None:
    assert "from<br> <span>your documents" in BRAND_HEADER_HTML


def test_document_list_renders_hostile_filename_as_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.ui.uploads.list_documents",
        lambda: [
            {
                "filename": "notes`\n# injected [x](javascript:alert(1)).pdf",
                "chunk_count": 1,
                "pages": [1],
            }
        ],
    )

    *_, filenames_text = refresh_docs()

    assert "\n# injected" not in filenames_text
    assert "](javascript:" not in filenames_text
