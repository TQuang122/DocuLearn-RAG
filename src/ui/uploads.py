from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import gradio as gr

from src.export import safe_markdown_code
from src.filters import MetadataFilter, filters_to_dict
from src.indexing import save_and_ingest_pdf
from src.store import list_documents
from src.ui.content import EMPTY_LIBRARY_HTML
from src.ui.helpers import int_value, status_html


def read_uploaded_pdf(file_obj: object) -> tuple[bytes, str]:
    if isinstance(file_obj, str):
        path = Path(file_obj)
        return path.read_bytes(), path.name

    raw_path = getattr(file_obj, "path", None)
    raw_name = getattr(file_obj, "orig_name", None)
    if isinstance(raw_path, str) and raw_path:
        path = Path(raw_path)
        name = raw_name.strip() if isinstance(raw_name, str) and raw_name.strip() else path.name
        return path.read_bytes(), name

    if isinstance(file_obj, dict):
        raw_path = file_obj.get("path")
        raw_name = file_obj.get("orig_name") or file_obj.get("name")
        if isinstance(raw_path, str) and raw_path:
            path = Path(raw_path)
            name = raw_name.strip() if isinstance(raw_name, str) and raw_name.strip() else path.name
            return path.read_bytes(), name

    raise TypeError(f"Unsupported uploaded file type: {type(file_obj).__name__}")


def read_uploaded_pdfs(file_obj: object) -> list[tuple[bytes, str]]:
    if file_obj is None:
        return []
    if isinstance(file_obj, (list, tuple)):
        return [read_uploaded_pdf(item) for item in file_obj]
    return [read_uploaded_pdf(file_obj)]


def build_filters(filenames: list[str] | None, page: int | None) -> dict[str, object] | None:
    payload: dict[str, object] = {}
    if filenames:
        payload["filenames"] = filenames
    if page is not None:
        payload["page"] = page
    return filters_to_dict(MetadataFilter.model_validate(payload)) if payload else None


def refresh_docs() -> tuple[
    dict[str, Any],
    dict[str, dict[str, object]],
    dict[str, Any],
    str,
    str,
]:
    docs = list_documents()
    choices = [str(doc["filename"]) for doc in docs]
    doc_map = {str(doc["filename"]): doc for doc in docs}
    if docs:
        summary = (
            '<div class="library-stats">'
            f"<strong>{len(docs)}</strong> tài liệu đã index"
            '<span aria-hidden="true">·</span>'
            f"<strong>{sum(int_value(doc['chunk_count']) for doc in docs)}</strong> đoạn văn"
            "</div>"
        )
    else:
        summary = EMPTY_LIBRARY_HTML
    filenames_text = (
        "\n".join(f"- {safe_markdown_code(name)}" for name in choices) if choices else ""
    )
    return (
        gr.update(choices=choices, value=[]),
        doc_map,
        gr.update(
            choices=["(Tất cả trang)"],
            value="(Tất cả trang)",
            interactive=bool(docs),
        ),
        summary,
        filenames_text,
    )


def pages_for_selection(doc_map: dict[str, Any], selected: list[str]) -> dict[str, Any]:
    if len(selected) != 1:
        return gr.update(
            choices=["(Tất cả trang)"],
            value="(Tất cả trang)",
            interactive=False,
        )
    doc = doc_map.get(selected[0]) or {}
    raw_pages = doc.get("pages")
    pages = raw_pages if isinstance(raw_pages, list) else []
    page_choices = ["(Tất cả trang)", *[str(page) for page in pages]]
    return gr.update(
        choices=page_choices,
        value="(Tất cả trang)",
        interactive=True,
    )


def upload_pdf(
    file: object | None,
) -> tuple[str, dict[str, Any], dict[str, dict[str, object]], dict[str, Any], str, str]:
    payloads = read_uploaded_pdfs(file)
    if not payloads:
        choices, doc_map, page_dropdown, summary, filenames_text = refresh_docs()
        return (
            status_html("Vui lòng chọn ít nhất một file PDF."),
            choices,
            doc_map,
            page_dropdown,
            summary,
            filenames_text,
        )

    successes: list[str] = []
    failures: list[str] = []
    chunks_total = 0
    for file_bytes, filename in payloads:
        try:
            info = save_and_ingest_pdf(file_bytes, filename)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            failures.append(f"{filename}: {error}")
            continue
        successes.append(str(info["filename"]))
        chunks_total += int_value(info.get("chunks_indexed"))

    parts: list[str] = []
    if successes:
        parts.append(f"Đã nạp {len(successes)} file · {chunks_total} đoạn")
    if failures:
        parts.append(f"Không thể nạp {len(failures)} file")
    details = ""
    if failures:
        items = "".join(
            f"<li><code>{html.escape(item, quote=False)}</code></li>" for item in failures
        )
        details = f"<details><summary>Xem lỗi</summary><ul>{items}</ul></details>"
    body = (" · ".join(parts) if parts else "Không có file hợp lệ.") + details

    choices, doc_map, page_dropdown, summary, filenames_text = refresh_docs()
    return (
        status_html(body, trusted=True),
        choices,
        doc_map,
        page_dropdown,
        summary,
        filenames_text,
    )
