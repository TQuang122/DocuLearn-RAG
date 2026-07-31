# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Primary user is inferred from the product workflow and existing copy: a student
or researcher working from one or more PDF documents who wants to study and
verify material without leaving the document context.

## Product Purpose

DocuLearn-RAG turns uploaded PDFs into a grounded learning workspace. It indexes
documents, retrieves relevant passages, and produces cited answers, summaries,
quizzes, and flashcards. Success means a user can move from source documents to
reliable study artifacts in one workflow while retaining source and page
context.

## Positioning

The product combines document-scoped retrieval with learning-oriented outputs:
the same indexed PDF scope powers Q&A, summaries, quizzes, and flashcards, with
citations and source passages retained for verification.

## Operating Context

Users upload one or more PDFs, index them into a local Qdrant-backed store,
select documents and an optional page scope, then work through Q&A, Summary,
Quiz, and Flashcards tabs. The workspace is available as a Gradio app locally
and as a Hugging Face Space. CLI and FastAPI surfaces support scripted and
integrated workflows.

## Capabilities and Constraints

- PDF upload, extraction, chunking, embeddings, and metadata-aware retrieval.
- Document and page filters define the active study scope.
- Gemini is used for generation features and requires `GEMINI_API_KEY` or an
  entered key in the UI.
- Outputs include citations and can be exported as Markdown or text.
- The local vector store and uploaded PDFs are runtime data; they are not
  assumed to be committed or available in a fresh hosted Space.
- The app must remain compatible with Python 3.12, Gradio 6.20, local
  development, and Hugging Face Spaces deployment.
- Product terminology is DocuLearn-RAG; prior AIVietnam/AIVL naming is not a
  product commitment.
- Open decision: authentication, persistence, and collaboration requirements
  beyond the current optional API key are not established.

## Brand Commitments

- Product name: DocuLearn-RAG.
- Voice: clear, practical, source-aware, and study-oriented.
- Existing document-learning workflow and source-backed output behavior must be
  preserved during UI work.

## Evidence on Hand

- Interactive Gradio entrypoint: `app.py` and `src/ui/layout.py`.
- Product and deployment instructions: `README.md`.
- Source-grounded generation and retrieval code under `src/`.
- Automated UI and behavior contracts under `tests/`.
- Existing design authority: `DESIGN.md`.
- No testimonials, customer studies, or external product claims are present;
  future work must not fabricate them.

## Product Principles

- Keep every generated learning artifact traceable to the selected source
  scope.
- Make the path from PDF upload to study action obvious and low-friction.
- Preserve one consistent scope across Q&A, summaries, quizzes, and flashcards.
- Prefer useful, verifiable output over unsupported confidence.

## Accessibility & Inclusion

The existing product requirements include WCAG 2.2 AA contrast targets, visible
keyboard focus, keyboard access to upload, accordions, tabs, inputs, and actions,
44px minimum touch targets, responsive behavior at narrow widths, and reduced
motion support. Vietnamese content should remain readable without narrow forced
columns.
