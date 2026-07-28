---
title: DocuLearn RAG
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
python_version: "3.12"
hardware:
  accelerator: cpu
short_description: PDF learning assistant with Q&A, summaries, flashcards
---

# DocuLearn-RAG

Ứng dụng học tập cục bộ trên PDF, gồm hỏi đáp RAG, tóm tắt, quiz và
flashcards có trích dẫn nguồn. Giao diện web dùng Gradio; dự án cũng cung cấp
CLI và REST API.

## Cài đặt

Yêu cầu Python 3.12 và [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-groups
cp .env.example .env
```

Điền `GEMINI_API_KEY` trong `.env`, hoặc nhập key trực tiếp trong giao diện.
Không commit file `.env`.

## Chạy ứng dụng

Gradio:

```bash
uv run python app.py
```

Mặc định giao diện chạy tại `http://127.0.0.1:7860`.

CLI:

```bash
uv run doculearn-rag --help
uv run doculearn-rag ingest
uv run doculearn-rag ask "Nội dung chính của tài liệu là gì?"
uv run doculearn-rag summarize --document "example.pdf" --output exports/summary.md
```

FastAPI:

```bash
uv run uvicorn src.interfaces.api:app --host 127.0.0.1 --port 8000
```

OpenAPI docs ở `http://127.0.0.1:8000/docs`. Nếu cấu hình `RAG_API_KEY`, gửi
header `X-API-Key` cho mọi route trừ `/health`.

## Dữ liệu và cấu hình

- PDF đầu vào: `data/`
- Qdrant cục bộ: `storage/qdrant/`
- File Markdown tải từ UI: `exports/`
- Kích thước upload mặc định: 25 MiB

Các đường dẫn tương đối luôn được resolve theo thư mục gốc dự án, nên CLI/API
có thể chạy từ working directory khác.

## Kiểm tra

```bash
uv run pytest -q
uv run ruff check app.py src tests
uv run ruff format --check app.py src tests
uv run basedpyright --level error src
```

## Đánh giá chunking

File test JSON có dạng:

```json
[
  {
    "question": "Câu hỏi cần đánh giá",
    "ground_truth": "Câu trả lời tham chiếu"
  }
]
```

Chạy:

```bash
uv run python -m src.evaluation.run_chunking evaluation_cases.json
```

Evaluation gọi Gemini và cần `GEMINI_API_KEY`. Dự án ghim Ragas `0.3.9` cùng
`langchain-community==0.3.31` vì Ragas `0.4.3` hiện import một module VertexAI
đã bị xóa ở dòng LangChain Community mới.
