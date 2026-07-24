from __future__ import annotations

import json

import gradio as gr

from src.export import export
from src.learning import generate_flashcards, generate_quiz, summarize
from src.llm import set_runtime_gemini_api_key
from src.rag import answer
from src.ui.helpers import PROGRESS
from src.ui.uploads import build_filters


def ask_question(
    question: str,
    k: int,
    selected_docs: list[str],
    page: str,
    gemini_key: str,
) -> tuple[str, str]:
    if not question or not question.strip():
        return "Vui lòng nhập câu hỏi.", ""
    page_num = None if page == "(Tất cả trang)" else int(page)
    set_runtime_gemini_api_key(gemini_key)
    result = answer(
        question.strip(),
        k=int(k),
        filters=build_filters(selected_docs, page_num),
    )
    return (
        str(export(result, fmt="md")),
        json.dumps(result.model_dump(), ensure_ascii=False, indent=2),
    )


def ask_chat(
    message: str,
    history: list[dict[str, str]] | None,
    k: int,
    selected_docs: list[str],
    page: str,
    gemini_key: str,
) -> tuple[list[dict[str, str]], str, str]:
    user_text = (message or "").strip()
    turns = list(history or [])
    if not user_text:
        return turns, "", ""
    answer_md, raw = ask_question(user_text, k, selected_docs, page, gemini_key)
    turns.append({"role": "user", "content": user_text})
    turns.append({"role": "assistant", "content": answer_md})
    return turns, raw, ""


def summarize_documents(
    query: str,
    k: int,
    selected_docs: list[str],
    page: str,
    gemini_key: str,
    progress: gr.Progress = PROGRESS,
) -> tuple[str, str]:
    page_num = None if page == "(Tất cả trang)" else int(page)
    set_runtime_gemini_api_key(gemini_key)
    progress(0.3, desc="Đang truy xuất ngữ cảnh…")
    try:
        result = summarize(
            query=query.strip() or None,
            k=int(k),
            filters=build_filters(selected_docs, page_num),
        )
    except RuntimeError as error:
        raise _empty_result_error() from error
    progress(0.9, desc="Đang định dạng kết quả…")
    return str(export(result, fmt="md")), result.model_dump_json(indent=2)


def generate_quiz_set(
    query: str,
    count: int,
    k: int,
    selected_docs: list[str],
    page: str,
    gemini_key: str,
    progress: gr.Progress = PROGRESS,
) -> tuple[str, str]:
    page_num = None if page == "(Tất cả trang)" else int(page)
    set_runtime_gemini_api_key(gemini_key)
    progress(0.3, desc="Đang truy xuất ngữ cảnh…")
    try:
        result = generate_quiz(
            query=query.strip() or None,
            count=int(count),
            k=int(k),
            filters=build_filters(selected_docs, page_num),
        )
    except RuntimeError as error:
        raise _empty_result_error() from error
    progress(0.9, desc="Đang định dạng kết quả…")
    return str(export(result, fmt="md")), result.model_dump_json(indent=2)


def generate_flashcard_set(
    query: str,
    count: int,
    k: int,
    selected_docs: list[str],
    page: str,
    gemini_key: str,
    progress: gr.Progress = PROGRESS,
) -> tuple[str, str]:
    page_num = None if page == "(Tất cả trang)" else int(page)
    set_runtime_gemini_api_key(gemini_key)
    progress(0.3, desc="Đang truy xuất ngữ cảnh…")
    try:
        result = generate_flashcards(
            query=query.strip() or None,
            count=int(count),
            k=int(k),
            filters=build_filters(selected_docs, page_num),
        )
    except RuntimeError as error:
        raise _empty_result_error() from error
    progress(0.9, desc="Đang định dạng kết quả…")
    return str(export(result, fmt="md")), result.model_dump_json(indent=2)


def _empty_result_error() -> gr.Error:
    return gr.Error(
        "Chưa có nội dung phù hợp để tạo kết quả. Hãy nạp tài liệu, chọn phạm vi rồi thử lại."
    )
