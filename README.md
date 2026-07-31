---
title: DocuLearn-RAG
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
python_version: "3.12"
short_description: "PDF learning assistant: Q&A, summaries, quizzes, flashcards"
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
| `RAG_RETRIEVAL_MODE` | `dense` | Retrieval mode: `dense` or guarded `fusion` pilot |
| `RAG_RETRIEVAL_CANDIDATE_K` | `50` | Dense candidate pool used by fusion retrieval |
| `RAG_RETRIEVAL_DENSE_WEIGHT` | `0.25` | Dense score weight in fusion ranking |
| `RAG_RETRIEVAL_MAX_CHUNKS_PER_PAGE` | `1` | Page diversity cap in fusion results |
| `RAG_RETRIEVAL_FALLBACK_TO_DENSE` | `true` | Fall back to dense retrieval if fusion fails |
| `RAG_RETRIEVAL_TELEMETRY_ENABLED` | `false` | Persist privacy-safe retrieval telemetry |
| `RAG_RETRIEVAL_SHADOW_SAMPLE_RATE` | `0.0` | Fraction of requests compared with the other mode |
| `RAG_CHUNK_SIZE` | `1500` | Chunk size |
| `RAG_CHUNK_OVERLAP` | `200` | Chunk overlap |
| `RAG_LLM_MODEL` | `gemini-flash-lite-latest` | Gemini model |
| `RAG_EMBEDDING_MODEL` | multilingual MiniLM | Local embedding model |

See `.env.example` for the complete list of supported settings.

Fusion retrieval is opt-in. Set `RAG_RETRIEVAL_MODE=fusion` to run the
validated candidate-50 configuration. Each retrieval emits structured Loguru
telemetry with its mode, candidate count, result count, latency, collection,
and fallback status. Keep dense fallback enabled during the pilot.

For a sampled pilot that returns fusion results while comparing dense retrieval
in a background worker:

```env
RAG_RETRIEVAL_MODE=fusion
RAG_RETRIEVAL_FALLBACK_TO_DENSE=true
RAG_RETRIEVAL_TELEMETRY_ENABLED=true
RAG_RETRIEVAL_SHADOW_SAMPLE_RATE=0.1
```

Events are appended to `exports/retrieval_telemetry.jsonl`. They contain a
random event ID, modes, result counts, latency, fallback/error types, top-1
agreement, and overlap at k. Query text, filters, filenames selected by the
user, and generated answers are never persisted. Summarize the pilot with:

```bash
uv run doculearn-rag retrieval-telemetry \
  --max-fallback-rate 0.01 \
  --max-error-rate 0.01 \
  --max-insufficient-rate 0.01 \
  --max-primary-p95-ms 250 \
  --min-events 100 \
  --min-shadow-events 30
```

The operational gate is `insufficient_data` until telemetry includes at least
100 total events and 30 shadow comparisons by default. Retrieval quality still
requires the annotated product benchmark; agreement between dense and fusion
is not a relevance label.

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
src/evaluation/        Retrieval and answer-quality evaluation tools
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

## RAG evaluation

### Evaluation data

The canonical retrieval dataset is
`data/benchmark_rag.jsonl`. Each line follows this schema:

```json
{
  "id": "qa-001",
  "question": "What is the main contribution?",
  "ground_truth": "The reference answer.",
  "source_file": null,
  "gold_pages": [],
  "gold_chunk_ids": [],
  "answerable": true,
  "question_type": null,
  "difficulty": null
}
```

Regenerate it from the legacy CSV without inventing source metadata:

```bash
uv run python -m src.evaluation.evaluation_dataset \
  'data/[Data]-Benchmark-Rag.csv' \
  --output data/benchmark_rag.jsonl \
  --report data/benchmark_rag_annotation_report.json
```

Use `data/benchmark_rag_annotation_report.json` as the annotation queue. For
each answerable record, inspect the source PDF and add either:

- `source_file` plus one or more 1-based `gold_pages`; or
- one or more exact `gold_chunk_ids`.

Also annotate `question_type` and `difficulty` (`easy`, `medium`, or `hard`).
Do not infer source metadata from the reference answer alone. Set
`answerable` to `false` only when the selected corpus genuinely cannot answer
the question; unanswerable records must not contain gold source metadata.

### Retrieval baseline

Retrieval evaluation uses the existing local embedding model and Qdrant
collection, but it never calls Gemini or another LLM. Evaluate one existing
collection:

```bash
uv run python -m src.evaluation.retrieval_evaluator \
  data/benchmark_rag.jsonl \
  --collection doculearn_rag \
  --k 10 \
  --output-dir evaluation-results/retrieval/current
```

Pass the same metadata filters accepted by production retrieval when needed:

```bash
uv run python -m src.evaluation.retrieval_evaluator \
  data/benchmark_rag.jsonl \
  --collection doculearn_rag \
  --k 10 \
  --filters-json '{"filename":"paper.pdf","page":2}' \
  --output-dir evaluation-results/retrieval/filtered
```

To rebuild isolated collections and compare all registered recursive
chunking strategies (`rc_500_50`, `rc_800_100`, `rc_1000_150`, and
`rc_1500_200`):

```bash
uv run python -m src.evaluation.run_retrieval \
  data/benchmark_rag.jsonl \
  --k 10 \
  --output-dir evaluation-results/retrieval/chunking
```

Each strategy directory contains:

- `cases.jsonl`: ranked chunks, latency, annotation status, and metrics for
  every question;
- `summary.json`: Recall@1/3/5/10, MRR, nDCG at the configured `k`, p50/p95
  retrieval latency, and coverage counts.

`baseline_summary.json` compares all four strategies. Ranking metrics include
only answerable records with gold source metadata. Missing annotations are
reported as `annotation_required`, not silently scored as failures.
Unanswerable questions are reported separately through
`unanswerable_accuracy`, which measures whether retrieval returned no chunks.

`k` must be at least 10 so every requested recall cutoff is observable.

### Frozen product PDF corpus

The seven product evaluation PDFs live below `data/[Data]-Documents-PDFs/`,
while production ingestion intentionally discovers only top-level PDFs. Create
an evaluation-only manifest so experiments use the exact intended corpus:

```bash
uv run python -m src.evaluation.pdf_corpus build-manifest \
  --pdf-dir 'data/[Data]-Documents-PDFs' \
  --output data/evaluation/pdf_corpus_manifest.json

uv run python -m src.evaluation.pdf_corpus build-queue \
  data/benchmark_rag.jsonl \
  --manifest data/evaluation/pdf_corpus_manifest.json \
  --output evaluation-results/annotation/benchmark_rag_queue.jsonl
```

The queue suggests likely pages but leaves every review `pending`. Candidates
must be checked against the PDF before setting `status` to `approved` or
`unanswerable`. Export is rejected while any record remains pending:

```bash
uv run python -m src.evaluation.pdf_corpus apply-queue \
  data/benchmark_rag.jsonl \
  --manifest data/evaluation/pdf_corpus_manifest.json \
  --queue evaluation-results/annotation/benchmark_rag_queue.jsonl \
  --output data/benchmark_rag_annotated.jsonl \
  --report data/benchmark_rag_annotation_apply_report.json
```

Run all four chunking strategies on the frozen product corpus with
`--corpus-manifest data/evaluation/pdf_corpus_manifest.json`.

### Public retrieval benchmarks

DocuLearn-RAG can run two reproducible external baselines without Gemini:

- **XQuAD-VI** checks Vietnamese retrieval on translated QA passages with gold
  source paragraphs.
- **BEIR SciFact** checks scientific retrieval against standard document-level
  qrels, including queries with more than one relevant document.

Download and normalize either benchmark:

```bash
uv run python -m src.evaluation.external_benchmarks xquad-vi \
  --output-dir data/evaluation/external/xquad-vi

uv run python -m src.evaluation.external_benchmarks beir-scifact \
  --output-dir data/evaluation/external/beir-scifact
```

Then run the same four chunking strategies used by the PDF baseline:

```bash
uv run python -m src.evaluation.run_external_retrieval \
  data/evaluation/external/xquad-vi \
  --output-dir evaluation-results/retrieval/xquad-vi

uv run python -m src.evaluation.run_external_retrieval \
  data/evaluation/external/beir-scifact \
  --output-dir evaluation-results/retrieval/beir-scifact
```

Add `--limit 10` for a quick smoke test. Generated benchmark corpora are
ignored by Git and can be rebuilt from their official sources. External
results measure document-level retrieval and must not be averaged with the
PDF-specific evaluation set: only manually verified PDF pages or chunks
measure performance on the actual DocuLearn-RAG study corpus. Temporary Qdrant
collections are not created: public baselines use exact in-memory cosine search
with the same configured embedding model.

### Retrieval evaluation versus Ragas

Retrieval evaluation asks whether the correct source appears in the ranked
chunks. It is deterministic with respect to the indexed collection and does
not generate answers.

Ragas answer evaluation runs the full answer pipeline and uses Gemini as a
judge for faithfulness, answer relevancy, context precision, and context
recall. Run it separately when a `GEMINI_API_KEY` is configured:

```bash
uv run python -m src.evaluation.run_chunking \
  evaluation_cases.json \
  --output-dir evaluation-results/ragas
```

Ragas may incur API usage. Do not compare its answer-quality scores directly
with deterministic retrieval metrics.

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
