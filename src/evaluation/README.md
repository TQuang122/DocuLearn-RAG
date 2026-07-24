# RAG evaluation

`chunking_strategy.py` defines the pre-registered recursive chunking grid.
`ragas_evaluator.py` converts DocuLearn-RAG answers into a Ragas dataset.
`run_chunking.py` rebuilds one isolated Qdrant collection per strategy and
writes mean metric values to `evaluation-results/chunking/`.

Evaluation uses the configured local embedding model and Gemini as judge, so
it requires `GEMINI_API_KEY` and may incur API usage.
