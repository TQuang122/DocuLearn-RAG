# RAG evaluation

This package has two deliberately separate evaluation paths.

## Deterministic retrieval evaluation

- `evaluation_dataset.py` validates JSONL records and converts the legacy CSV.
- `retrieval_evaluator.py` measures Recall@1/3/5/10, MRR, nDCG, and p50/p95
  retrieval latency without calling an LLM.
- `run_retrieval.py` rebuilds and evaluates isolated collections for
  `rc_500_50`, `rc_800_100`, `rc_1000_150`, and `rc_1500_200`.
- `external_benchmarks.py` downloads and normalizes XQuAD-VI or BEIR SciFact.
- `run_external_retrieval.py` indexes each public corpus independently and
  evaluates document-level qrels across the same four chunking strategies.
- `pdf_corpus.py` freezes an explicit PDF corpus, prepares a human annotation
  queue, and applies only completed reviews.

Convert the benchmark:

```bash
uv run python -m src.evaluation.evaluation_dataset \
  'data/[Data]-Benchmark-Rag.csv' \
  --output data/benchmark_rag.jsonl \
  --report data/benchmark_rag_annotation_report.json
```

Annotate the records listed in the report with verified `source_file` and
`gold_pages`, or exact `gold_chunk_ids`. Records lacking verified gold source
metadata remain visible in per-case output but are excluded from aggregate
ranking metrics.

Evaluate an existing collection:

```bash
uv run python -m src.evaluation.retrieval_evaluator \
  data/benchmark_rag.jsonl \
  --collection doculearn_rag \
  --k 10 \
  --output-dir evaluation-results/retrieval/current
```

Evaluate all chunking strategies:

```bash
uv run python -m src.evaluation.run_retrieval \
  data/benchmark_rag.jsonl \
  --k 10 \
  --output-dir evaluation-results/retrieval/chunking
```

Use `--filters-json` with either command to apply production-compatible
metadata filters.

## Product PDF corpus and annotation

The production PDF discovery function only reads top-level files in `data/`.
Evaluation can instead use an explicit immutable manifest without changing
production ingestion:

```bash
uv run python -m src.evaluation.pdf_corpus build-manifest \
  --pdf-dir 'data/[Data]-Documents-PDFs' \
  --output data/evaluation/pdf_corpus_manifest.json
```

The manifest stores relative paths, SHA-256 hashes, byte sizes, and page counts.
Loading it fails if a PDF has moved or changed.

Generate the annotation queue:

```bash
uv run python -m src.evaluation.pdf_corpus build-queue \
  data/benchmark_rag.jsonl \
  --manifest data/evaluation/pdf_corpus_manifest.json \
  --output evaluation-results/annotation/benchmark_rag_queue.jsonl \
  --top-n 5
```

Candidate pages use deterministic lexical overlap from the question and
reference answer. They are navigation aids, not gold labels. Review every
record and edit its `review` object:

- `status: "approved"` requires a verified `source_file` and `gold_pages`;
- `status: "unanswerable"` requires both source fields to remain empty;
- `status: "pending"` blocks dataset export.

Apply a completed queue:

```bash
uv run python -m src.evaluation.pdf_corpus apply-queue \
  data/benchmark_rag.jsonl \
  --manifest data/evaluation/pdf_corpus_manifest.json \
  --queue evaluation-results/annotation/benchmark_rag_queue.jsonl \
  --output data/benchmark_rag_annotated.jsonl \
  --report data/benchmark_rag_annotation_apply_report.json
```

Run the annotated dataset against exactly the frozen PDFs:

```bash
uv run python -m src.evaluation.run_retrieval \
  data/benchmark_rag_annotated.jsonl \
  --corpus-manifest data/evaluation/pdf_corpus_manifest.json \
  --k 10 \
  --output-dir evaluation-results/retrieval/product-pdf
```

After choosing a chunking strategy, test expanded dense retrieval with
BM25-style lexical reranking independently of production:

```bash
uv run python -m src.evaluation.run_reranking \
  data/benchmark_rag_annotated.jsonl \
  --collection doculearn_product_eval__rc_800_100 \
  --candidate-k 50 \
  --k 10 \
  --dense-weight 0.25 \
  --max-chunks-per-page 1 \
  --output-dir evaluation-results/retrieval/product-pdf-rerank \
  --baseline-cases \
    evaluation-results/retrieval/product-pdf-baseline/rc_800_100/cases.jsonl \
  --error-analysis-output \
    evaluation-results/retrieval/product-pdf-rerank/error_analysis.json
```

This retrieves a larger dense candidate pool, combines normalized dense and
lexical scores, and limits duplicate page occupancy. It does not call an LLM
or change `src.rag.retrieve`; promotion to production requires a separate
decision after comparing quality, latency, and per-question regressions.

## Public benchmark baselines

Prepare the Vietnamese XQuAD benchmark:

```bash
uv run python -m src.evaluation.external_benchmarks xquad-vi \
  --output-dir data/evaluation/external/xquad-vi
```

Prepare BEIR SciFact:

```bash
uv run python -m src.evaluation.external_benchmarks beir-scifact \
  --output-dir data/evaluation/external/beir-scifact
```

Each directory contains `corpus.jsonl`, `queries.jsonl`, and `manifest.json`.
External query records use `gold_document_ids` because BEIR qrels can identify
multiple relevant documents. This is deliberately separate from the PDF
dataset schema, whose gold evidence is expressed as pages or exact chunks.

Run all four chunking strategies without an LLM:

```bash
uv run python -m src.evaluation.run_external_retrieval \
  data/evaluation/external/xquad-vi \
  --output-dir evaluation-results/retrieval/xquad-vi

uv run python -m src.evaluation.run_external_retrieval \
  data/evaluation/external/beir-scifact \
  --output-dir evaluation-results/retrieval/beir-scifact
```

Validate the same expanded dense plus lexical reranker on one selected
chunking strategy before considering production promotion:

```bash
uv run python -m src.evaluation.run_external_reranking \
  data/evaluation/external/xquad-vi \
  --output-dir evaluation-results/retrieval/xquad-vi-rerank \
  --strategy rc_800_100 \
  --candidate-k 50 \
  --k 10 \
  --dense-weight 0.25 \
  --max-chunks-per-page 1

uv run python -m src.evaluation.run_external_reranking \
  data/evaluation/external/beir-scifact \
  --output-dir evaluation-results/retrieval/beir-scifact-rerank \
  --strategy rc_800_100 \
  --candidate-k 50 \
  --k 10 \
  --dense-weight 0.25 \
  --max-chunks-per-page 1
```

Compare each `summary.json` and `cases.jsonl` with the matching dense baseline.
Promotion decisions should consider aggregate quality, p50/p95 latency, and
per-query regressions rather than relying on a single metric.

Use `--limit 10` for a quick integration smoke test. Omit it for a reportable
baseline. Ranking is measured at document level while the indexed documents
are still split by each registered chunking strategy. Repeated chunks from the
same document do not receive duplicate relevance credit. External baselines
use exact in-memory cosine search with the configured production embedding
model, avoiding large temporary local Qdrant collections.

Public benchmarks test language and retrieval-engine behavior. They do not
replace the document-grounded benchmark for the PDFs used by DocuLearn-RAG.

## Ragas answer evaluation

`ragas_evaluator.py` converts generated DocuLearn-RAG answers into a Ragas
dataset. `run_chunking.py` runs the full retrieval-plus-generation pipeline
and writes mean answer-quality metrics.

This path uses the configured local embedding model and Gemini as judge. It
requires `GEMINI_API_KEY`, may incur API usage, and must not be confused with
the LLM-free retrieval baseline.
