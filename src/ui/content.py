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
    <h1>Học sâu hơn từ<br> <span>tài liệu của bạn.</span></h1>
    <p>Biến PDF thành một không gian học có dẫn nguồn — hỏi đáp, tóm tắt,
    quiz và flashcards trên cùng một luồng làm việc.</p>
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
    <span class="workflow-label">Bắt đầu</span>
    <strong>Một luồng học, bốn công cụ.</strong>
  </div>
  <ol class="workflow-steps">
    <li><span>01</span>Tải PDF</li>
    <li><span>02</span>Chọn phạm vi</li>
    <li><span>03</span>Nhập Gemini key</li>
    <li><span>04</span>Bắt đầu học</li>
  </ol>
</div>
"""

UPLOAD_HEADING_HTML = """
<div class="panel-heading">
  <span class="panel-icon" aria-hidden="true">
    <svg viewBox="0 0 24 24"><path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5"/>
    <path d="M5 14v4a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-4"/></svg>
  </span>
  <div><h2>Nguồn tài liệu</h2><p>Tải và lập chỉ mục PDF của bạn.</p></div>
</div>
"""

LIBRARY_HEADING_HTML = """
<div class="panel-heading">
  <span class="panel-icon" aria-hidden="true">
    <svg viewBox="0 0 24 24">
    <path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H11v17H6.5A2.5 2.5 0 0 0 4 22V5.5Z"/>
    <path d="M20 5.5A2.5 2.5 0 0 0 17.5 3H13v17h4.5A2.5 2.5 0 0 1 20 22V5.5Z"/></svg>
  </span>
  <div><h2>Thư viện đã index</h2><p>Chọn tài liệu và phạm vi trước khi truy vấn.</p></div>
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
  <span><strong>Thư viện đang trống</strong>
  <small>Tải PDF ở khung bên trái để bắt đầu không gian học của bạn.</small></span>
</div>
"""

USAGE_MARKDOWN = """
1. **Tải PDF** ở khối bên trái rồi bấm **Nạp và index**.  
2. **Chọn tài liệu** muốn học trong danh sách đã index.  
3. Dùng các tab để **hỏi đáp**, **tóm tắt**, **tạo quiz** hoặc **flashcards**.  
4. Nếu chỉ chọn đúng 1 tài liệu, bạn có thể lọc thêm theo **trang**.

**Mẹo:** Khi đặt câu hỏi rõ ràng theo chủ đề, kết quả RAG thường sát và dễ học hơn.
"""
