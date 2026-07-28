from __future__ import annotations

import json

import gradio as gr

from src.export import export
from src.learning import generate_flashcards, generate_quiz, summarize
from src.llm import set_runtime_gemini_api_key
from src.rag import answer
from src.store import delete_document_by_filename, list_documents
from src.ui.helpers import PROGRESS
from src.ui.interactive import render_flashcard_html, render_quiz_html
from src.ui.uploads import build_filters


def _chat_message_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


def _normalize_chat_history(history: list[dict[str, object]] | None) -> list[gr.ChatMessage]:
    normalized: list[gr.ChatMessage] = []
    for message in history or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        text = _chat_message_text(message.get("content"))
        if role in {"user", "assistant", "system"} and text:
            normalized.append(gr.ChatMessage(role=str(role), content=text))
    return normalized


def ask_question(
    question: str,
    k: int,
    selected_docs: list[str],
    page: str,
    gemini_key: str,
) -> tuple[str, str]:
    if not question or not question.strip():
        return "Please enter a question.", ""
    page_num = None if page == "All pages" else int(page)
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
    history: list[dict[str, object]] | None,
    k: int,
    selected_docs: list[str],
    page: str,
    gemini_key: str,
) -> tuple[list[gr.ChatMessage], str, str]:
    user_text = (message or "").strip()
    turns = _normalize_chat_history(history)
    if not user_text:
        return turns, "", ""
    answer_md, raw = ask_question(user_text, k, selected_docs, page, gemini_key)
    turns.append(gr.ChatMessage(role="user", content=user_text))
    turns.append(gr.ChatMessage(role="assistant", content=answer_md))
    return turns, raw, ""


def summarize_documents(
    query: str,
    k: int,
    selected_docs: list[str],
    page: str,
    gemini_key: str,
    progress: gr.Progress = PROGRESS,
) -> tuple[str, str]:
    page_num = None if page == "All pages" else int(page)
    set_runtime_gemini_api_key(gemini_key)
    progress(0.3, desc="Retrieving context…")
    try:
        result = summarize(
            query=query.strip() or None,
            k=int(k),
            filters=build_filters(selected_docs, page_num),
        )
    except RuntimeError as error:
        raise _generation_error(error) from error
    progress(0.9, desc="Formatting results…")
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
    page_num = None if page == "All pages" else int(page)
    set_runtime_gemini_api_key(gemini_key)
    progress(0.3, desc="Retrieving context…")
    try:
        result = generate_quiz(
            query=query.strip() or None,
            count=int(count),
            k=int(k),
            filters=build_filters(selected_docs, page_num),
        )
    except RuntimeError as error:
        raise _generation_error(error) from error
    progress(0.9, desc="Formatting results…")
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
    page_num = None if page == "All pages" else int(page)
    set_runtime_gemini_api_key(gemini_key)
    progress(0.3, desc="Retrieving context…")
    try:
        result = generate_flashcards(
            query=query.strip() or None,
            count=int(count),
            k=int(k),
            filters=build_filters(selected_docs, page_num),
        )
    except RuntimeError as error:
        raise _generation_error(error) from error
    progress(0.9, desc="Formatting results…")
    return str(export(result, fmt="md")), result.model_dump_json(indent=2)


def generate_quiz_set_interactive(
    query: str,
    count: int,
    k: int,
    selected_docs: list[str],
    page: str,
    gemini_key: str,
    progress: gr.Progress = PROGRESS,
) -> tuple[str, str, str]:
    """Generate quiz and return (interactive_html, markdown, json)."""
    page_num = None if page == "All pages" else int(page)
    set_runtime_gemini_api_key(gemini_key)
    progress(0.3, desc="Retrieving context…")
    try:
        result = generate_quiz(
            query=query.strip() or None,
            count=int(count),
            k=int(k),
            filters=build_filters(selected_docs, page_num),
        )
    except RuntimeError as error:
        raise _generation_error(error) from error
    progress(0.9, desc="Formatting results…")
    html_str = render_quiz_html(result)
    md_str = str(export(result, fmt="md"))
    return html_str, md_str, result.model_dump_json(indent=2)


def generate_flashcard_set_interactive(
    query: str,
    count: int,
    k: int,
    selected_docs: list[str],
    page: str,
    gemini_key: str,
    progress: gr.Progress = PROGRESS,
) -> tuple[str, str, str]:
    """Generate flashcards and return (interactive_html, markdown, json)."""
    page_num = None if page == "All pages" else int(page)
    set_runtime_gemini_api_key(gemini_key)
    progress(0.3, desc="Retrieving context…")
    try:
        result = generate_flashcards(
            query=query.strip() or None,
            count=int(count),
            k=int(k),
            filters=build_filters(selected_docs, page_num),
        )
    except RuntimeError as error:
        raise _generation_error(error) from error
    progress(0.9, desc="Formatting results…")
    html_str = render_flashcard_html(result)
    md_str = str(export(result, fmt="md"))
    return html_str, md_str, result.model_dump_json(indent=2)


def delete_selected_docs(
    selected: list[str],
    progress: gr.Progress = PROGRESS,
) -> tuple[list[str], str]:
    """Delete selected documents from the vector store."""
    if not selected:
        raise gr.Error("No documents selected for deletion.")
    for fn in selected:
        progress(None, desc=f"Deleting {fn}…")
        delete_document_by_filename(str(fn))
    remaining = list_documents()
    filenames = [str(d["filename"]) for d in remaining]
    return filenames, f"Deleted {len(selected)} document(s)."


def clear_chat() -> tuple[list[list[str]], str]:
    """Clear the chat history."""
    return [], ""


def _empty_result_error() -> gr.Error:
    return gr.Error(
        "No relevant content found. Please upload documents, select a scope, and try again."
    )


def _generation_error(error: RuntimeError) -> gr.Error:
    if "Missing Gemini API key" in str(error):
        return gr.Error(
            "Gemini API key is missing. Open the Gemini API key section, enter a key, "
            "or set GEMINI_API_KEY before generating."
        )
    return _empty_result_error()
