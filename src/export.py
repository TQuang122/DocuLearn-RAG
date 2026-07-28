"""Export learning outputs to JSON or Markdown."""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from src.schemas import Citation, FlashcardSet, QuizSet, RagAnswer, Summary

ExportFormat = Literal["text", "md", "json"]
_ACTIVE_URI_SCHEME = re.compile(r"\b(?:https?|ftp|javascript|data):", re.IGNORECASE)


def _safe(value: str) -> str:
    """Keep untrusted text readable without retaining active Markdown links."""
    escaped = html.escape(value, quote=False).replace("[", r"\[").replace("]", r"\]")
    return _ACTIVE_URI_SCHEME.sub(lambda match: match.group(0)[:-1] + r"\:", escaped)


def _safe_inline(value: str) -> str:
    """Keep untrusted metadata on one inert Markdown line."""
    return _safe(" ".join(value.splitlines())).replace("`", r"\`")


def safe_markdown_code(value: str) -> str:
    """Render untrusted single-line metadata as an inert Markdown code span."""
    normalized = html.escape(" ".join(value.splitlines()), quote=False)
    normalized = _ACTIVE_URI_SCHEME.sub(
        lambda match: match.group(0)[:-1] + r"\:",
        normalized,
    )
    longest_run = max((len(run) for run in re.findall(r"`+", normalized)), default=0)
    fence = "`" * (longest_run + 1)
    padding = " " if normalized.startswith("`") or normalized.endswith("`") else ""
    return f"{fence}{padding}{normalized}{padding}{fence}"


def _citation_line(c: Citation) -> str:
    parts = [f"**[{c.source_marker}]** {_safe_inline(c.filename)} · Page {c.page}"]
    if c.section:
        parts.append(f"Section: {_safe_inline(c.section)}")
    return " | ".join(parts)


def _details_block(summary: str, content: str) -> str:
    safe_summary = html.escape(summary, quote=False)
    safe_content = html.escape(content, quote=False)
    return (
        "<details>"
        f"<summary>{safe_summary}</summary>"
        f'<div style="margin:8px 0 0 0; white-space:pre-wrap;">{safe_content}</div>'
        "</details>"
    )


def _citation_source_text_block(c: Citation) -> str:
    if not c.source_text:
        return ""
    return "\n  " + _details_block("View source passage", c.source_text)


def _marker_details(citations: list[Citation], markers: list[str]) -> list[str]:
    by_marker = {c.source_marker: c for c in citations}
    lines: list[str] = []
    for m in markers:
        c = by_marker.get(m)
        if c is None:
            continue
        summary = f"**[{c.source_marker}]** {_safe_inline(c.filename)} · Page {c.page}"
        if c.section:
            summary += f" | Section: {_safe_inline(c.section)}"
        if c.source_text:
            lines.append(f"- {_details_block(summary, c.source_text)}")
        else:
            lines.append(f"- {summary}")
    return lines


def _citations_block(citations: list[Citation]) -> str:
    if not citations:
        return ""
    lines = ["## Sources", ""]
    for c in citations:
        lines.append(f"- {_citation_line(c)}{_citation_source_text_block(c)}")
    return "\n".join(lines) + "\n"


def _render_with_sources(body_lines: list[str], citations: list[Citation]) -> str:
    lines = [*body_lines, ""]
    c = _citations_block(citations)
    if c:
        lines.append(c)
    return "\n".join(lines).rstrip() + "\n"


def _to_markdown(model: BaseModel) -> str:
    if isinstance(model, Summary):
        title = "# Summary" + (f": {_safe_inline(model.target)}" if model.target else "")
        lines: list[str] = [title, "", f"_Scope: {model.scope}_", ""]
        if model.summary:
            lines.extend([_safe(model.summary.strip()), ""])
        if model.key_points:
            lines.extend(["## Key Points", "", *[f"- {_safe(kp)}" for kp in model.key_points], ""])
        return _render_with_sources(lines, model.citations)

    if isinstance(model, RagAnswer):
        return _render_with_sources([_safe(model.answer.strip())], model.citations)

    if isinstance(model, QuizSet):
        title = "# Quiz" + (f": {_safe_inline(model.target)}" if model.target else "")
        lines = [title, "", f"_Scope: {model.scope} | Items: {len(model.items)}_", ""]
        for idx, item in enumerate(model.items, start=1):
            meta_parts: list[str] = []
            if item.topic:
                meta_parts.append(f"topic: {_safe_inline(item.topic)}")
            if item.difficulty:
                meta_parts.append(f"difficulty: {_safe_inline(item.difficulty)}")
            meta_suffix = f" _({' | '.join(meta_parts)})_" if meta_parts else ""

            lines.extend([f"## Q{idx}.{meta_suffix}", "", _safe(item.question.strip()), ""])
            for opt_idx, option in enumerate(item.options):
                lines.append(f"- {chr(ord('A') + opt_idx)}) {_safe(option)}")
            lines.append("")
            lines.append(f"**Answer:** {chr(ord('A') + item.correct_index)}")
            if item.explanation:
                lines.append(f"**Explanation:** {_safe(item.explanation.strip())}")
            if item.source_markers:
                lines.extend(
                    ["**Sources:**", *_marker_details(model.citations, item.source_markers)]
                )
            lines.append("")

        c = _citations_block(model.citations)
        if c:
            lines.append(c)
        return "\n".join(lines).rstrip() + "\n"

    if isinstance(model, FlashcardSet):
        title = "# Flashcards" + (f": {_safe_inline(model.target)}" if model.target else "")
        lines = [title, "", f"_Scope: {model.scope} | Cards: {len(model.cards)}_", ""]
        for idx, card in enumerate(model.cards, start=1):
            topic = f" — {_safe_inline(card.topic)}" if card.topic else ""
            lines.extend([f"## Card {idx}{topic}", ""])
            lines.append(f"**Front:** {_safe(card.front.strip())}")
            lines.append(f"**Back:** {_safe(card.back.strip())}")
            if card.hint:
                lines.append(f"**Hint:** {_safe(card.hint.strip())}")
            if card.source_markers:
                lines.extend(
                    ["**Sources:**", *_marker_details(model.citations, card.source_markers)]
                )
            lines.append("")

        c = _citations_block(model.citations)
        if c:
            lines.append(c)
        return "\n".join(lines).rstrip() + "\n"

    raise TypeError(f"Unsupported model type: {type(model).__name__}")


def export(
    model: BaseModel, *, fmt: ExportFormat = "text", output: Path | None = None
) -> str | Path:
    """Render model to a string, optionally writing it to disk.
    Args: model, fmt, output (optional).
    Returns: rendered string if output is None; otherwise the written path.
    Raises: TypeError for unsupported model type; ValueError for unknown fmt.
    """
    if fmt == "json":
        text = model.model_dump_json(indent=2) + "\n"
    elif fmt in {"text", "md"}:
        text = _to_markdown(model)
    else:
        raise ValueError(f"Unknown fmt '{fmt}'. Expected 'text' | 'md' | 'json'.")

    if output is None:
        return text

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return output
