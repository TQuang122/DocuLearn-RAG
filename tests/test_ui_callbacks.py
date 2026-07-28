from __future__ import annotations

import gradio as gr

from src.ui import callbacks


def test_normalize_chat_history_flattens_gradio_text_messages() -> None:
    history = [
        {
            "role": "user",
            "content": [{"text": "previous question", "type": "text"}],
            "metadata": None,
            "options": None,
        },
        {
            "role": "assistant",
            "content": [{"text": "previous answer", "type": "text"}],
        },
    ]

    normalized = callbacks._normalize_chat_history(history)

    assert [(message.role, message.content) for message in normalized] == [
        ("user", "previous question"),
        ("assistant", "previous answer"),
    ]


def test_ask_chat_returns_renderable_assistant_message(monkeypatch) -> None:
    monkeypatch.setattr(
        callbacks,
        "ask_question",
        lambda *args: ("Rendered answer", '{"answer":"Rendered answer"}'),
    )
    history = [
        {
            "role": "user",
            "content": [{"text": "old question", "type": "text"}],
        }
    ]

    turns, raw, cleared_query = callbacks.ask_chat(
        "new question", history, 6, [], "All pages", ""
    )

    assert turns[-1].role == "assistant"
    assert turns[-1].content == "Rendered answer"
    assert raw == '{"answer":"Rendered answer"}'
    assert cleared_query == ""
    rendered = gr.Chatbot(layout="bubble").postprocess(turns)
    assert rendered.root[-1].role == "assistant"
    assert rendered.root[-1].content[0].text == "Rendered answer"
