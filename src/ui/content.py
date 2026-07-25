BRAND_HEADER_HTML = """
<header class="product-header">
  <a class="brand-lockup" href="#workspace" aria-label="DocuLearn-RAG">
    <span class="brand-mark" aria-hidden="true">
      <svg viewBox="0 0 32 32" role="img">
        <path d="M8 4.5h12.5a3 3 0 0 1 3 3V22H11a3 3 0 0 1-3-3V4.5Z"/>
        <path d="M11.5 8h12a3 3 0 0 1 3 3v16.5H14.5a3 3 0 0 1-3-3V8Z"/>
        <path d="M16 14h6.5M16 18h6.5M16 22h4"/>
      </svg>
    </span>
    <span>DocuLearn-RAG</span>
  </a>
  <div class="header-status">
    <span class="status-dot" aria-hidden="true"></span>
    Local-first workspace
  </div>
</header>
<section class="hero-shell">
  <div class="hero-copy">
    <div class="eyebrow">Document intelligence workspace</div>
    <h1>Learn deeper from<br> <span>your documents.</span></h1>
    <p>Turn your PDFs into a sourced learning space — Q&A, summaries,
    quizzes, and flashcards in one workflow.</p>
  </div>
  <div class="hero-visual" aria-hidden="true">
    <div class="document-orbit orbit-one"></div>
    <div class="document-orbit orbit-two"></div>
    <div class="document-stack">
      <div class="document-sheet sheet-back"></div>
      <div class="document-sheet sheet-mid"></div>
      <div class="document-sheet sheet-front">
        <span></span><span></span><span></span><span></span>
      </div>
      <div class="rag-node node-a"></div>
      <div class="rag-node node-b"></div>
      <div class="rag-node node-c"></div>
    </div>
  </div>
</section>
"""

INFO_NOTE_HTML = """
<div class="workflow-strip">
  <div class="workflow-intro">
    <span class="workflow-label">Get started</span>
    <strong>One learning flow, four tools.</strong>
  </div>
  <ol class="workflow-steps">
    <li><span>01</span>Upload PDF</li>
    <li><span>02</span>Select scope</li>
    <li><span>03</span>Enter Gemini key</li>
    <li><span>04</span>Start learning</li>
  </ol>
</div>
"""

UPLOAD_HEADING_HTML = """
<div class="panel-heading">
  <span class="panel-icon" aria-hidden="true">
    <svg viewBox="0 0 24 24"><path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5"/>
    <path d="M5 14v4a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-4"/></svg>
  </span>
  <div><h2>Document sources</h2><p>Upload and index your PDFs.</p></div>
</div>
"""

LIBRARY_HEADING_HTML = """
<div class="panel-heading">
  <span class="panel-icon" aria-hidden="true">
    <svg viewBox="0 0 24 24">
    <path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H11v17H6.5A2.5 2.5 0 0 0 4 22V5.5Z"/>
    <path d="M20 5.5A2.5 2.5 0 0 0 17.5 3H13v17h4.5A2.5 2.5 0 0 1 20 22V5.5Z"/></svg>
  </span>
  <div><h2>Indexed library</h2><p>Select documents and scope before querying.</p></div>
</div>
"""

EMPTY_LIBRARY_HTML = """
<div class="library-empty">
  <span class="library-empty-icon" aria-hidden="true">
    <svg viewBox="0 0 24 24">
      <path d="M7 3.5h7l3 3V20H7a2 2 0 0 1-2-2V5.5a2 2 0 0 1 2-2Z"/>
      <path d="M14 3.5V7h3M8.5 11h5M8.5 14.5h6"/>
    </svg>
  </span>
  <span><strong>Library is empty</strong>
  <small>Upload PDFs in the left panel to start your learning space.</small></span>
</div>
"""

USAGE_MARKDOWN = """
1. **Upload a PDF** in the left panel, then click **Upload & index**.  
2. **Select documents** to study from the indexed list.  
3. Use the tabs for **Q&A**, **summary**, **quiz**, or **flashcards**.  
4. If only 1 document is selected, you can filter by **page**.

**Tip:** When you ask clear topic-focused questions, the RAG results are more relevant.
"""
