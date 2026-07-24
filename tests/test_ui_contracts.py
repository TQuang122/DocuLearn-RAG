from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.ui.content import BRAND_HEADER_HTML
from src.ui.helpers import CSS, select_learning_tab
from src.ui.uploads import (
    pages_for_selection,
    read_uploaded_pdfs,
    refresh_docs,
    upload_pdf,
)


@pytest.mark.parametrize(
    ("label", "tab_id"),
    [
        ("Hỏi đáp", "qa"),
        ("Tóm tắt", "summary"),
        ("Quiz", "quiz"),
        ("Flashcards", "flashcards"),
    ],
)
def test_learning_navigation_selects_registered_tab(label: str, tab_id: str) -> None:
    selected = select_learning_tab(label)

    assert selected.selected == tab_id


def test_uploaded_path_is_normalized_to_bytes(tmp_path: Path) -> None:
    upload = tmp_path / "notes.pdf"
    upload.write_bytes(b"%PDF-1.7")

    payloads = read_uploaded_pdfs(str(upload))

    assert payloads == [(b"%PDF-1.7", "notes.pdf")]


def test_single_document_selection_enables_its_pages() -> None:
    update = pages_for_selection({"notes.pdf": {"pages": [1, 3]}}, ["notes.pdf"])

    assert update["choices"] == ["(Tất cả trang)", "1", "3"]
    assert update["interactive"] is True


def test_empty_upload_keeps_six_output_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.ui.uploads.refresh_docs",
        lambda: (
            {"choices": [], "value": []},
            {},
            {"choices": ["(Tất cả trang)"], "value": "(Tất cả trang)"},
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

    assert len(config["components"]) == 86
    assert len(config["dependencies"]) == 11
    assert contracts["select_learning_tab"] == (1, 1)
    assert contracts["refresh_documents"] == (0, 5)
    assert contracts["pages_for_selection"] == (2, 1)
    assert contracts["upload_pdf"] == (1, 6)
    assert contracts["ask_chat"] == (6, 3)
    assert contracts["summarize_documents"] == (5, 2)
    assert contracts["generate_quiz_set"] == (6, 2)
    assert contracts["generate_flashcard_set"] == (6, 2)


def test_refactor_preserves_stylesheet() -> None:
    digest = hashlib.sha256(CSS.encode()).hexdigest()

    assert digest == "6eabc35b36ceaba4fec9de486fa2496143d496bdeae47d48ba6d785daffd11d3"


def test_mobile_heading_keeps_word_boundary_when_break_is_hidden() -> None:
    assert "từ<br> <span>tài liệu" in BRAND_HEADER_HTML


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
