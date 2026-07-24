# Interfaces

- `api.py`: FastAPI routes for health, documents, upload, Q&A, summaries, quiz,
  and flashcards.
- `cli.py`: Typer CLI exposed as the `doculearn-rag` console command
  (`notebooklm` remains as a compatibility alias).
- `app.py` at the project root: Gradio web interface.

All interfaces call the same modules in `src` and share validation, filtering,
indexing, retrieval, and generation behavior.
