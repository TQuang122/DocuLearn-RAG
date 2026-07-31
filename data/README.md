# Data directory

The top level contains the local PDF corpus and the manually annotated
DocuLearn-RAG evaluation dataset. PDF files are processed by extracting their
content, splitting it into chunks, and indexing those chunks in the vector
store.

Downloaded public benchmark artifacts are generated under
`data/evaluation/external/`. They are ignored by Git and can be recreated with
`python -m src.evaluation.external_benchmarks`.

`data/evaluation/pdf_corpus_manifest.json` freezes the product evaluation
corpus using relative paths, hashes, file sizes, and page counts. Annotation
queues are generated under the ignored `evaluation-results/` directory and do
not become gold datasets until every record has been manually reviewed.
