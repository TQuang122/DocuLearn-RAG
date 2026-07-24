"""Input and rendering boundary tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.export import export
from src.filters import MetadataFilter
from src.learning import _validate_items
from src.schemas import (
    Citation,
    Flashcard,
    FlashcardSet,
    QuizItem,
    QuizSet,
    RagAnswer,
    Summary,
)


def test_metadata_filter_rejects_negative_pages() -> None:
    """Given an impossible page number, parsing should fail at the boundary."""
    with pytest.raises(ValidationError):
        MetadataFilter.model_validate({"page": -1})


def test_metadata_filter_rejects_unknown_fields() -> None:
    """Given a mistyped filter, parsing should fail instead of silently ignoring it."""
    with pytest.raises(ValidationError):
        MetadataFilter.model_validate({"filenmae": "doc.pdf"})


def test_markdown_export_escapes_untrusted_html() -> None:
    """Given model and filename HTML, exported Markdown must not contain executable tags."""
    result = RagAnswer(
        question="q",
        answer='<img src=x onerror="alert(1)">',
        citations=[
            Citation(
                source_index=1,
                source_marker="S1",
                filename="<script>alert(2)</script>.pdf",
                page=1,
            )
        ],
    )

    rendered = export(result, fmt="md")

    assert "<img" not in rendered
    assert "<script>" not in rendered
    assert "&lt;img" in rendered
    assert "&lt;script&gt;" in rendered


@pytest.mark.parametrize(
    "result",
    [
        RagAnswer(
            question="q",
            answer="[click](javascript:alert(1)) ![pixel](https://tracker.invalid/pixel)",
        ),
        Summary(
            scope="corpus",
            summary="[click](javascript:alert(1)) ![pixel](https://tracker.invalid/pixel)",
        ),
        QuizSet(
            scope="corpus",
            items=[
                QuizItem(
                    question="[click](javascript:alert(1))",
                    options=["![pixel](https://tracker.invalid/pixel)", "B", "C", "D"],
                    correct_index=0,
                    explanation="https://tracker.invalid/explanation",
                )
            ],
        ),
        FlashcardSet(
            scope="corpus",
            cards=[
                Flashcard(
                    front="[click](javascript:alert(1))",
                    back="![pixel](https://tracker.invalid/pixel)",
                    hint="https://tracker.invalid/hint",
                )
            ],
        ),
    ],
)
def test_markdown_export_neutralizes_external_links_and_images(result) -> None:
    rendered = export(result, fmt="md")

    assert "](javascript:" not in rendered
    assert "![" not in rendered
    assert "https://tracker.invalid" not in rendered


def test_markdown_export_keeps_hostile_filename_inside_citation() -> None:
    result = RagAnswer(
        question="q",
        answer="answer",
        citations=[
            Citation(
                source_index=1,
                source_marker="S1",
                filename="notes`\n# injected [link](https://tracker.invalid).pdf",
                page=1,
            )
        ],
    )

    rendered = export(result, fmt="md")

    assert "\n# injected" not in rendered
    assert "](https://tracker.invalid)" not in rendered


def test_quiz_item_without_valid_source_marker_is_rejected() -> None:
    """Given fabricated citation markers, a supposedly grounded item must be rejected."""
    payload = {
        "items": [
            {
                "question": "Question",
                "options": ["A", "B", "C", "D"],
                "correct_index": 0,
                "explanation": "Explanation",
                "source_markers": ["S999"],
            }
        ]
    }

    with pytest.raises(RuntimeError, match="No valid quiz items"):
        _validate_items(payload, "items", QuizItem, "question", "quiz items", {"S1"})
