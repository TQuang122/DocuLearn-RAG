---
title: DocuLearn-RAG
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
python_version: "3.12"
short_description: PDF-grounded learning assistant with Q&A, summaries, quizzes, and flashcards
---

# DocuLearn-RAG

DocuLearn-RAG is a local, PDF-grounded learning workspace. It indexes your
documents, retrieves relevant passages, and turns them into source-backed
answers, summaries, quizzes, and flashcards.

The project provides three interfaces:

- A Gradio web app for interactive study.
- A Typer CLI for scripting and batch workflows.
- A FastAPI service for application integrations.

## Features

- Upload and index PDFs into a local Qdrant vector store.
- Ask questions against the full library or a selected document/page range.
- Generate single-document and multi-document summaries.
- Generate multiple-choice quizzes with citations.
- Generate flashcards with source passages.
- Export summaries, quizzes, and flashcards as Markdown or text.
- Inspect retrieved chunks with the CLI or the REST API.
- Configure optional API authentication for the FastAPI and Gradio surfaces.

## How it works

```text
PDF files
   │
   ▼
Page extraction → chunking → local embeddings → Qdrant
                                      │
                                      ▼
                         retrieval with metadata filters
                                      │
                                      ▼
                 Gemini-grounded answer or learning artifact
                                      │
                                      ▼
                    citations, Markdown, or interactive UI
```

Each indexed chunk retains its filename, page number, document ID, section,
and chunk ID. These metadata fields power document/page filters and source
citations in the generated output.

## Requirements

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- A Gemini API key for generation features

The default embedding model is downloaded locally through
`sentence-transformers`, so the first indexing run may take longer and require
additional disk space.

## Installation

From the project root:

```bash
uv sync --all-groups
cp .env.example .env
```

Set `GEMINI_API_KEY` in `.env`, or enter the key in the Gradio interface. Keep
`.env` private; it is intentionally excluded from Git.

## Run the web app

```bash
uv run python app.py
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860). The default port and host
can be changed with `RAG_SERVER_PORT` and `RAG_SERVER_NAME`.

If port 7860 is already in use, stop the previous process or choose another
port:

```bash
RAG_SERVER_PORT=7861 uv run python app.py
```

The app flow is:

1. Upload one or more PDFs.
2. Click **Upload & index**.
3. Select the document and optional page scope.
4. Use Q&A, Summary, Quiz, or Flashcards.

## CLI

List available commands:

```bash
uv run doculearn-rag --help
```

Index every PDF in the configured data directory:

```bash
uv run doculearn-rag ingest
```

Recreate the vector collection before indexing:

```bash
uv run doculearn-rag ingest --recreate
```

Ask a grounded question and print its sources:

```bash
uv run doculearn-rag ask "What is the main idea of the document?"
```

Apply metadata filters with comma-separated `key=value` pairs. Multiple
filenames are separated with `|`:

```bash
uv run doculearn-rag ask \
  "Explain the training procedure." \
  --filters "filename=notes.pdf,page=3"
```

Inspect retrieved chunks as JSON:

```bash
uv run doculearn-rag debug-retrieval \
  "What is reinforcement learning?" \
  --as-json
```

Generate learning artifacts and export them:

```bash
uv run doculearn-rag summarize --document notes.pdf --output exports/summary.md --fmt markdown
uv run doculearn-rag quiz --document notes.pdf --count 8 --output exports/quiz.md --fmt markdown
uv run doculearn-rag flashcards --document notes.pdf --count 15 --output exports/flashcards.md --fmt markdown
```

Use `--query`, `--filters`, and `--k` to narrow the source scope or control the
number of retrieved chunks.

## REST API

Start the API server:

```bash
uv run uvicorn src.interfaces.api:app --host 127.0.0.1 --port 8000
```

Interactive OpenAPI documentation is available at
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

Available routes:

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Health check |
| `GET` | `/documents` | List indexed documents |
| `POST` | `/upload` | Upload and index a PDF |
| `POST` | `/ask` | Ask a grounded question |
| `POST` | `/summarize` | Generate a summary |
| `POST` | `/quiz` | Generate a quiz |
| `POST` | `/flashcards` | Generate flashcards |

If `RAG_API_KEY` is configured, send it with protected requests:

```bash
curl -H "X-API-Key: $RAG_API_KEY" http://127.0.0.1:8000/documents
```

`/health` remains public. Request models reject unknown fields and validate
question length, retrieval limits, quiz counts, and flashcard counts.

## Configuration

Environment variables use the `RAG_` prefix unless noted otherwise.

| Variable | Default | Description |
| --- | --- | --- |
| `GEMINI_API_KEY` | unset | Gemini generation credential |
| `RAG_API_KEY` | unset | Optional API authentication key |
| `RAG_DATA_DIR` | `data` | Input PDF directory |
| `RAG_STORAGE_DIR` | `storage/qdrant` | Local Qdrant storage |
| `RAG_EXPORT_DIR` | `exports` | Generated Markdown/text files |
| `RAG_MAX_UPLOAD_BYTES` | `26214400` | Upload limit, 25 MiB |
| `RAG_SERVER_NAME` | `127.0.0.1` | Gradio bind host |
| `RAG_SERVER_PORT` | `7860` | Gradio port |
| `RAG_TOP_K` | `5` | Default retrieval count |
| `RAG_CHUNK_SIZE` | `1500` | Chunk size |
| `RAG_CHUNK_OVERLAP` | `200` | Chunk overlap |
| `RAG_LLM_MODEL` | `gemini-flash-lite-latest` | Gemini model |
| `RAG_EMBEDDING_MODEL` | multilingual MiniLM | Local embedding model |

See `.env.example` for the complete list of supported settings.

## Project layout

```text
app.py                 Gradio entrypoint
src/config.py          Environment-backed settings
src/indexing.py        PDF loading, chunking, and ingestion
src/store.py           Local Qdrant storage operations
src/rag.py             Retrieval, prompts, and citations
src/learning.py        Summaries, quizzes, and flashcards
src/interfaces/cli.py  Typer CLI
src/interfaces/api.py  FastAPI service
src/ui/                Gradio layout, callbacks, and interactive surfaces
src/prompts/           Jinja2 generation prompts
static/style.css       DocuLearn-RAG design system styles
tests/                 Unit, boundary, API, and UI contract tests
assets/                Architecture and pipeline diagrams
```

## Testing and quality checks

Run the test suite:

```bash
uv run pytest -q
```

Run formatting and static checks:

```bash
uv run ruff check app.py src tests
uv run ruff format --check app.py src tests
uv run basedpyright --level error src
```

The test suite covers configuration boundaries, PDF indexing, vector-store
contracts, API routes, CLI entrypoints, Markdown safety, and the Gradio UI
surface contract.

## Chunking evaluation

Evaluation cases are JSON objects with a question and reference answer:

```json
[
  {
    "question": "What is the main contribution?",
    "ground_truth": "The reference answer."
  }
]
```

Run the evaluation command with a Gemini key configured:

```bash
uv run python -m src.evaluation.run_chunking evaluation_cases.json
```

Evaluation may incur Gemini API usage. The project pins compatible Ragas and
LangChain Community versions to keep the evaluation imports stable.

## Data and generated files

- Put local PDFs in `data/`. PDF files are ignored by Git.
- The local Qdrant collection is stored under `storage/qdrant/`.
- UI and CLI exports are written to `exports/`.
- Temporary QA screenshots matching `qa-*.png` are ignored by Git.

Do not commit API keys, local vector-store contents, generated exports, or
private source PDFs.

## License and project status

DocuLearn-RAG is an educational and research-oriented project. Review the
repository history and dependency licenses before using it in a production
service.
