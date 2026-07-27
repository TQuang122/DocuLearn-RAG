from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.ui.content import BRAND_HEADER_HTML
from src.ui.helpers import CSS
from src.ui.uploads import (
    pages_for_selection,
    read_uploaded_pdfs,
    refresh_docs,
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

    assert len(config["components"]) == 87
    assert len(config["dependencies"]) == 13
    assert contracts["refresh_documents"] == (0, 5)
    assert contracts["pages_for_selection"] == (2, 1)
    assert contracts["upload_pdf"] == (1, 6)
    assert contracts["ask_chat"] == (6, 3)
    assert contracts["summarize_documents"] == (5, 2)
    assert contracts["generate_quiz_set"] == (6, 3)
    assert contracts["generate_flashcard_set"] == (6, 3)
    assert contracts["clear_chat"] == (0, 1)
    assert contracts["delete_selected_docs"] == (1, 2)
    assert contracts["lambda"] == (1, 1)
    assert contracts["lambda_1"] == (1, 1)
    assert contracts["lambda_2"] == (1, 1)


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


def test_upload_dropzone_keeps_internal_icon_buttons_scoped() -> None:
    assert ".upload-dropzone button {" not in CSS
    assert ".upload-dropzone > button:not(.icon-button)" in CSS
    assert ".source-file-picker .icon-button-wrapper .icon-button" in CSS


def test_learning_tabs_target_gradio_six_tablist_dom() -> None:
    assert ".learning-tabs .tab-nav" not in CSS
    assert '.learning-tabs .tab-container[role="tablist"] button' in CSS
    assert 'button[data-tab-id="flashcards"]::before' in CSS


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

    assert digest == "b581535d44cda7a7e717f58d3a6ab93e87abb89fe153fa6edb86855474bacd85"


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
