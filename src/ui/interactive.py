"""Render QuizSet/FlashcardSet as self-contained interactive HTML.

All interactivity runs in the browser.  ``<script>`` tags are *not* embedded in
the dynamic HTML (Gradio uses ``innerHTML`` which ignores them).  Instead, card
data is stored in ``data-cards`` / ``data-quiz`` attributes and picked up by a
MutationObserver (injected via ``gr.Blocks(head=...)``).
"""

from __future__ import annotations

import html as html_mod
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.schemas import FlashcardSet, QuizSet


def _attr_escape(s: str) -> str:
    """Escape *s* for safe embedding in a single-quoted HTML attribute."""
    return s.replace("&", "&amp;").replace("'", "&#39;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------------------
# Head script injected via gr.Blocks(head=…)  —  runs once on page load.
# Uses a MutationObserver to detect dynamic HTML updates and initialise the
# interactive components (flashcards / quiz) from data-* attributes.
# ---------------------------------------------------------------------------

INTERACTIVE_HEAD_HTML = """<script>
(function() {
'use strict';

// ----- helpers -----
function _esc(s) { var d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

function _syncAccessibility() {
  document.querySelectorAll(
    '[aria-hidden="true"] button, [aria-hidden="true"] a, ' +
    '[aria-hidden="true"] input, [aria-hidden="true"] textarea, ' +
    '[aria-hidden="true"] select, [aria-hidden="true"] [role="button"]'
  ).forEach(function(el) {
    if (!el.hasAttribute('data-doculearn-tabindex')) {
      el.setAttribute('data-doculearn-tabindex', el.getAttribute('tabindex') || '');
    }
    el.setAttribute('tabindex', '-1');
  });
  document.querySelectorAll('[data-doculearn-tabindex]').forEach(function(el) {
    if (el.closest('[aria-hidden="true"]')) return;
    var original = el.getAttribute('data-doculearn-tabindex');
    if (original) el.setAttribute('tabindex', original);
    else el.removeAttribute('tabindex');
    el.removeAttribute('data-doculearn-tabindex');
  });
  document.querySelectorAll(
    '.source-file-picker button[aria-label*="upload" i]'
  ).forEach(function(btn) {
    var visibleLabel = (btn.textContent || '').replace(/\\s+/g, ' ').trim();
    if (visibleLabel) btn.setAttribute('aria-label', visibleLabel);
  });
}

// ==================== Flashcards ====================
var _fcD = null, _fcI = 0, _fcF = false, _fcO = null, _fcK = null, _fcR = null, _fcFilt = 0, _fcMode = 'learn';

function _fcW() { return document.querySelector('.fc-card-wrapper'); }

function _fcShow(dir) {
  var c = _fcD[_fcO[_fcI]]; if (!c) return;
  var ft = document.getElementById('fc-front-text');
  var bt = document.getElementById('fc-back-text');
  var hi = document.getElementById('fc-hint');
  var pr = document.getElementById('fc-progress');
  if (ft) { ft.textContent = c.front; ft.style.animation = ''; void ft.offsetWidth; ft.style.animation = 'fc-slide-in ' + (dir || 'left') + ' 0.35s ease-out'; }
  if (bt) bt.textContent = c.back;
  var sources = document.getElementById('fc-sources');
  if (sources) {
    sources.innerHTML = (c.sources || []).map(function(s) {
      return '<details><summary>' + _esc(s.marker + ' · ' + s.filename + ' · p.' + s.page) + '</summary>' +
        (s.text ? '<div>' + _esc(s.text) + '</div>' : '') + '</details>';
    }).join('');
  }
  if (hi) { hi.textContent = c.hint || ''; hi.style.opacity = '0'; setTimeout(function() { hi.style.opacity = '1'; }, 100); }
  var mastered = _fcR.filter(function(r) { return r === 'got-it'; }).length;
  if (pr) pr.textContent = 'Card ' + (_fcI + 1) + ' of ' + _fcD.length + ' · ' + mastered + ' mastered';
  var pn = document.getElementById('fc-prev'), nn = document.getElementById('fc-next');
  if (pn) pn.style.visibility = _fcI === 0 ? 'hidden' : 'visible';
  if (nn) nn.style.visibility = _fcI === _fcD.length - 1 ? 'hidden' : 'visible';
  var dots = document.getElementById('fc-dots');
  if (dots) {
    dots.innerHTML = _fcD.map(function(_, i) {
      return '<span class="fc-dot' + (i === _fcI ? ' fc-dot-active' : '') + '"></span>';
    }).join('');
  }
  document.querySelectorAll('.fc-mode-btn').forEach(function(btn) {
    btn.classList.toggle('fc-mode-active', btn.dataset.mode === _fcMode);
    btn.setAttribute('aria-pressed', btn.dataset.mode === _fcMode ? 'true' : 'false');
  });

  var w = _fcW();
  if (w && _fcF) { w.classList.remove('fc-flipped'); _fcF = false; }

}

function _fcInit(root) {
  if (!root) root = document.querySelector('.fc-root');
  if (!root) return;
  var raw = root.getAttribute('data-cards');
  if (!raw) return;
  try { _fcD = JSON.parse(raw); } catch(e) { return; }
  _fcI = 0; _fcF = false;
  _fcO = _fcD.map(function(_, i) { return i; });
  _fcK = new Array(_fcD.length).fill(null);
  _fcR = new Array(_fcD.length).fill(null);
  _fcFilt = 0;
  _fcShow('left');
}

window.FC_FLIP = function() {
  var w = _fcW(); if (!w) return;
  _fcF = !_fcF; w.classList.toggle('fc-flipped', _fcF);
};

window.FC_NEXT = function() {
  if (_fcI >= _fcD.length - 1) return;
  if (_fcFilt === 1) { var ni; for (ni = _fcI + 1; ni < _fcD.length; ni++) { if (_fcK[_fcO[ni]] === false) break; } if (ni >= _fcD.length) return; _fcI = ni; }
  else { _fcI++; }
  _fcF = false; var w = _fcW(); if (w) w.classList.remove('fc-flipped'); _fcShow('right');
};

window.FC_PREV = function() {
  if (_fcI <= 0) return;
  if (_fcFilt === 1) { for (var pi = _fcI - 1; pi >= 0; pi--) { if (_fcK[_fcO[pi]] === false) break; } if (pi < 0) return; _fcI = pi; }
  else { _fcI--; }
  _fcF = false; var w = _fcW(); if (w) w.classList.remove('fc-flipped'); _fcShow('left');
};

window.FC_SHUFFLE = function() {
  for (var i = _fcO.length - 1; i > 0; i--) {
    var j = Math.floor(Math.random() * (i + 1));
    var t = _fcO[i]; _fcO[i] = _fcO[j]; _fcO[j] = t;
  }
  _fcK = new Array(_fcD.length).fill(null);
  _fcI = 0; _fcF = false; var w = _fcW(); if (w) w.classList.remove('fc-flipped'); _fcShow('left');
};

window.FC_RATE = function(value) {
  if (!_fcF) return;
  var originalIndex = _fcO[_fcI];
  _fcR[originalIndex] = value;
  _fcShow('left');
  if (_fcI < _fcD.length - 1) { FC_NEXT(); }
  else { FC_COMPLETE(); }
};

window.FC_COMPLETE = function() {
  var complete = document.getElementById('fc-complete');
  var mastered = _fcR.filter(function(r) { return r === 'got-it'; }).length;
  var difficult = _fcR.filter(function(r) { return r === 'again' || r === 'hard'; }).length;
  if (complete) {
    complete.style.display = 'block';
    document.getElementById('fc-mastered-num').textContent = mastered;
    document.getElementById('fc-difficult-num').textContent = difficult;
    complete.scrollIntoView({behavior:'smooth',block:'center'});
  }
};

window.FC_REVIEW = function() {
  var review = _fcD.map(function(_, idx) { return idx; }).filter(function(idx) { return _fcR[idx] !== 'got-it'; });
  if (!review.length) return;
  _fcO = review; _fcI = 0; _fcF = false; _fcFilt = 0; _fcMode = 'review';
  var complete = document.getElementById('fc-complete');
  if (complete) complete.style.display = 'none';
  _fcShow('left');
};

window.FC_RESTART = function() {
  _fcO = _fcD.map(function(_, idx) { return idx; }); _fcI = 0; _fcF = false; _fcFilt = 0;
  _fcR = new Array(_fcD.length).fill(null);
  var complete = document.getElementById('fc-complete');
  if (complete) complete.style.display = 'none';
  var w = _fcW(); if (w) w.classList.remove('fc-flipped'); _fcShow('left');
};

window.FC_SET_MODE = function(mode) {
  if (mode !== 'learn' && mode !== 'review' && mode !== 'shuffle' && mode !== 'exam') return;
  _fcMode = mode;
  if (mode === 'review') {
    _fcO = _fcD.map(function(_, idx) { return idx; }).filter(function(idx) { return _fcR[idx] !== 'got-it'; });
    if (!_fcO.length) _fcO = _fcD.map(function(_, idx) { return idx; });
  } else {
    _fcO = _fcD.map(function(_, idx) { return idx; });
    if (mode === 'shuffle') FC_SHUFFLE();
  }
  _fcI = 0; _fcF = false; _fcFilt = 0;
  var complete = document.getElementById('fc-complete');
  if (complete) complete.style.display = 'none';
  var w = _fcW(); if (w) w.classList.remove('fc-flipped'); _fcShow('left');
};



// ==================== Quiz ====================
var _qD = null, _qOrder = null, _qDone = null, _qSel = null, _qI = 0, _qTC = 0, _qTW = 0, _qV = null, _qMode = 'practice';
var _qL = ['A','B','C','D'];
var _qSVG_OK = '<svg class="iq-verdict-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/></svg>';
var _qSVG_BAD = '<svg class="iq-verdict-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>';

function _qRender() {
  var c = document.getElementById('iq-list'); if (!c) return;
  var i = _qOrder[_qI], q = _qD[i], answered = _qDone[i], h = '';
  var complete = _qOrder.every(function(idx) { return _qDone[idx]; });
  var showExplanation = answered && (_qMode === 'practice' || complete);
  h += '<div class="iq-card" style="animation-delay:0ms">';
  h += '<div class="iq-question"><span class="iq-question-index">' + (i + 1) + '</span>';
  if (q.difficulty) h += '<span class="iq-difficulty">' + _esc(q.difficulty) + '</span>';
  h += _esc(q.q) + '</div>';
  h += '<div class="iq-options">';
  for (var j = 0; j < q.opts.length; j++) {
    var state = answered ? (j === q.correct ? ' iq-correct' : j === _qSel[i] ? ' iq-wrong' : '') : '';
    h += '<div class="iq-option' + state + (answered ? ' iq-disabled' : '') + '" data-i="' + i + '" data-j="' + j + '" onclick="IQ_ANSWER(this)">';
    h += '<span class="iq-option-label">' + _qL[j] + '</span>';
    h += '<span class="iq-option-text">' + _esc(q.opts[j]) + '</span></div>';
  }
  h += '</div>';
  h += '<div class="iq-explain" id="iq-ex-' + i + '" style="display:' + (showExplanation ? 'block' : 'none') + '">';
  if (showExplanation) {
    h += '<strong>' + (q.correct === _qSel[i] ? 'Correct' : 'Review this answer') + '</strong><br>' + _esc(q.explanation);
    if (q.sources && q.sources.length) {
      h += '<div class="iq-sources"><span class="iq-sources-label">Sources</span>';
      q.sources.forEach(function(s) {
        h += '<details><summary>' + _esc(s.marker + ' · ' + s.filename + ' · p.' + s.page) + '</summary>';
        if (s.text) h += '<div>' + _esc(s.text) + '</div>';
        h += '</details>';
      });
      h += '</div>';
    }
  }
  h += '</div></div>';
  c.innerHTML = h;
  var prev = document.getElementById('iq-prev'), next = document.getElementById('iq-next');
  if (prev) prev.disabled = i === 0;
  if (next) { next.disabled = !answered; next.textContent = _qI === _qOrder.length - 1 ? 'See score' : 'Next question'; }
  var ct = document.getElementById('iq-counter');
  var answeredCount = _qOrder.filter(function(idx) { return _qDone[idx]; }).length;
  if (ct) ct.textContent = 'Question ' + (_qI + 1) + ' of ' + _qOrder.length + ' · ' + answeredCount + ' answered';
  var fill = document.getElementById('iq-fill');
  if (fill) fill.style.width = Math.round(answeredCount / _qOrder.length * 100) + '%';
  document.querySelectorAll('.iq-mode-btn').forEach(function(btn) {
    btn.classList.toggle('iq-mode-active', btn.dataset.mode === _qMode);
    btn.setAttribute('aria-pressed', btn.dataset.mode === _qMode ? 'true' : 'false');
  });
}

function _qInit(root) {
  if (!root) root = document.querySelector('.iq-root');
  if (!root) return;
  var raw = root.getAttribute('data-quiz');
  if (!raw) return;
  try { _qD = JSON.parse(raw); } catch(e) { return; }
  _qOrder = _qD.map(function(_, idx) { return idx; });
  _qDone = new Array(_qD.length).fill(false); _qSel = new Array(_qD.length).fill(null); _qI = 0;
  _qTC = 0; _qTW = 0;
  _qV = new Array(_qD.length).fill(false);
  _qRender();
}

window.IQ_ANSWER = function(el) {
  var i = parseInt(el.dataset.i), j = parseInt(el.dataset.j); var q = _qD[i];
  var ops = document.querySelectorAll('.iq-option[data-i="' + i + '"]');
  if (_qDone[i] || (ops[0] && ops[0].classList.contains('iq-disabled'))) return;
  _qSel[i] = j;
  if (j === q.correct) { _qTC++; }
  else { _qTW++; }
  _qV[i] = j === q.correct;
  _qDone[i] = true;
  _qRender();
  var ans = _qOrder.filter(function(idx) { return _qDone[idx]; }).length;
  var total = _qOrder.length;
  if (ans === total) {
    setTimeout(function() {
      var sc = document.getElementById('iq-score');
      if (sc) {
        var target = _qTC;
        var current = 0;
        var step = Math.max(1, Math.floor(target * 16 / 600));
        var countInt = setInterval(function() {
          current += step;
          if (current >= target) { current = target; clearInterval(countInt); }
          document.getElementById('iq-score-num').textContent = current;
        }, 16);
        var ring = document.getElementById('iq-ring-fill');
        if (ring) {
          var circ = 326.73;
          ring.style.strokeDashoffset = circ * (1 - target / total);
        }
        document.getElementById('iq-correct-num').textContent = _qTC;
        document.getElementById('iq-wrong-num').textContent = _qTW;
        document.getElementById('iq-score-total').textContent = '/ ' + total;
        document.getElementById('iq-accuracy').textContent = Math.round(_qTC / total * 100) + '% accuracy';
        var strip = document.getElementById('iq-verdict-strip');
        if (strip) {
          var vh = '';
          for (var vi = 0; vi < _qOrder.length; vi++) {
            var originalIndex = _qOrder[vi];
            vh += '<span class="iq-vdot ' + (_qV[originalIndex] ? 'iq-vdot-correct' : 'iq-vdot-wrong') + '" data-i="' + originalIndex + '" onclick="IQ_SCROLL_TO(' + originalIndex + ')"></span>';
          }
          strip.innerHTML = vh;
        }
        sc.style.display = 'block';
        sc.classList.add('iq-score-reveal');
        sc.scrollIntoView({behavior:'smooth',block:'center'});
      }
    }, 600);
  }
};

window.IQ_PREV = function() {
  if (_qI <= 0) return;
  _qI--; _qRender();
};

window.IQ_NEXT = function() {
  if (!_qDone[_qOrder[_qI]]) return;
  if (_qI === _qOrder.length - 1) {
    var sc = document.getElementById('iq-score');
    if (sc) sc.scrollIntoView({behavior:'smooth',block:'center'});
    return;
  }
  _qI++; _qRender();
};

window.IQ_RETRY_INCORRECT = function() {
  var wrong = _qOrder.filter(function(idx) { return !_qV[idx]; });
  if (!wrong.length) return;
  _qOrder = wrong;
  _qOrder.forEach(function(idx) { _qDone[idx] = false; _qSel[idx] = null; });
  _qI = 0; _qTC = 0; _qTW = 0;
  var sc = document.getElementById('iq-score');
  if (sc) sc.style.display = 'none';
  _qRender();
};

window.IQ_SET_MODE = function(mode) {
  if (mode !== 'practice' && mode !== 'exam') return;
  _qMode = mode; _qRender();
};

window.IQ_SCROLL_TO = function(i) {
  var target = _qOrder.indexOf(i);
  if (target < 0) return;
  _qI = target; _qRender();
};

window.IQ_RESTART = function() {
  _qOrder = _qD.map(function(_, idx) { return idx; });
  _qDone = new Array(_qD.length).fill(false); _qSel = new Array(_qD.length).fill(null); _qI = 0; _qTC = 0; _qTW = 0;
  _qV = new Array(_qD.length).fill(false);
  var sc = document.getElementById('iq-score');
  if (sc) sc.style.display = 'none';
  var ct = document.getElementById('iq-counter');
  if (ct) ct.textContent = 'Question 1 of ' + _qD.length + ' · 0 answered';
  var fill = document.getElementById('iq-fill');
  if (fill) fill.style.width = '0%';
  _qRender();
};

// ==================== Keyboard ====================
document.addEventListener('keydown', function(e) {
  var hasFc = document.querySelector('.fc-root');
  var hasIq = document.querySelector('.iq-root');
  if (!hasFc && !hasIq) return;
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  if (hasFc) {
    if (e.key === 'ArrowLeft') { FC_PREV(); }
    else if (e.key === 'ArrowRight') { FC_NEXT(); }
    else if (e.key === ' ') { e.preventDefault(); FC_FLIP(); }
    else if (e.key === '1') { FC_RATE('again'); }
    else if (e.key === '2') { FC_RATE('hard'); }
    else if (e.key === '3') { FC_RATE('got-it'); }
  }
  if (hasIq) {
    if (/^[1-4]$/.test(e.key)) {
      var option = document.querySelector('.iq-option[data-i="' + _qOrder[_qI] + '"][data-j="' + (Number(e.key) - 1) + '"]');
      if (option) option.click();
    } else if (e.key === 'Enter') { IQ_NEXT(); }
  }
});

// ==================== MutationObserver ====================
var _obs = new MutationObserver(function(muts) {
  _syncAccessibility();
  for (var mi = 0; mi < muts.length; mi++) {
    for (var ni = 0; ni < muts[mi].addedNodes.length; ni++) {
      var n = muts[mi].addedNodes[ni];
      if (n.nodeType === 1) {
        if (n.classList.contains('fc-root')) _fcInit(n);
        if (n.classList.contains('iq-root')) _qInit(n);
      }
    }
  }
});
_obs.observe(document.body, {
  childList: true,
  subtree: true,
  attributes: true,
  attributeFilter: ['aria-hidden']
});

// ==================== Check existing ====================
document.addEventListener('DOMContentLoaded', function() {
  _syncAccessibility();
  var fcEl = document.querySelector('.fc-root'); if (fcEl) _fcInit(fcEl);
  var iqEl = document.querySelector('.iq-root'); if (iqEl) _qInit(iqEl);
});

})();
</script>"""


# ---------------------------------------------------------------------------
# Quiz  (click-to-answer with instant feedback)
# ---------------------------------------------------------------------------


def render_quiz_html(quiz_set: QuizSet) -> str:
    """Return HTML with quiz data in ``data-quiz`` attribute for the MutationObserver."""
    if not quiz_set.items:
        return '<div class="iq-root iq-empty">No questions available.</div>'

    citation_map = {
        citation.source_marker: {
            "marker": citation.source_marker,
            "filename": citation.filename,
            "page": citation.page,
            "text": citation.source_text or "",
        }
        for citation in quiz_set.citations
    }
    data = [
        {
            "q": item.question,
            "opts": item.options,
            "correct": item.correct_index,
            "explanation": item.explanation or "No explanation provided.",
            "difficulty": item.difficulty or "",
            "sources": [
                citation_map[marker]
                for marker in item.source_markers
                if marker in citation_map
            ],
        }
        for item in quiz_set.items
    ]
    data_json = json.dumps(data, ensure_ascii=False)
    scope_escaped = html_mod.escape(
        f"{quiz_set.scope}: {quiz_set.target}" if quiz_set.target else quiz_set.scope
    )

    SVG_OK = (
        '<svg class="iq-verdict-icon" viewBox="0 0 24 24" fill="none" '
        'stroke="currentColor" stroke-width="2.5" stroke-linecap="round" '
        'stroke-linejoin="round">'
        '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/></svg>'
    )
    SVG_BAD = (
        '<svg class="iq-verdict-icon" viewBox="0 0 24 24" fill="none" '
        'stroke="currentColor" stroke-width="2.5" stroke-linecap="round" '
        'stroke-linejoin="round">'
        '<circle cx="12" cy="12" r="10"/>'
        '<line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>'
    )

    return f"""<div class="iq-root" data-quiz='{_attr_escape(data_json)}'>
<div class="iq-header">
  <span class="iq-scope">{scope_escaped}</span>
  <span class="iq-counter" id="iq-counter">0 / {len(data)} answered</span>
</div>
<div class="iq-modebar">
  <span class="iq-mode-label">Mode</span>
  <button class="iq-mode-btn iq-mode-active" data-mode="practice" aria-pressed="true" onclick="IQ_SET_MODE('practice')">Practice</button>
  <button class="iq-mode-btn" data-mode="exam" aria-pressed="false" onclick="IQ_SET_MODE('exam')">Exam</button>
  <span class="iq-shortcuts">1–4 answer · Enter next</span>
</div>
<div class="iq-progress-track">
  <div class="iq-progress-fill" id="iq-fill" style="width:0%"></div>
</div>
<div class="iq-list" id="iq-list"></div>
<div class="iq-nav">
  <button id="iq-prev" onclick="IQ_PREV()" disabled>Previous</button>
  <button id="iq-next" class="iq-next-btn" onclick="IQ_NEXT()" disabled>Next question</button>
</div>
<div class="iq-score" id="iq-score" style="display:none">
  <div class="iq-score-kicker">Quiz complete</div>
  <div class="iq-score-ring">
    <svg width="120" height="120" viewBox="0 0 120 120">
      <circle class="iq-ring-bg" cx="60" cy="60" r="52" fill="none" stroke-width="8"/>
      <circle class="iq-ring-fill" id="iq-ring-fill" cx="60" cy="60" r="52" fill="none" stroke-width="8" stroke-dasharray="326.73" stroke-dashoffset="326.73" transform="rotate(-90 60 60)" stroke-linecap="round"/>
    </svg>
    <div class="iq-score-num" id="iq-score-num">0</div>
    <div class="iq-score-total" id="iq-score-total">/ {len(data)}</div>
  </div>
  <div class="iq-score-label">Your Score</div>
  <div class="iq-accuracy" id="iq-accuracy">0% accuracy</div>
  <div class="iq-score-detail">
    <span class="iq-verdict-correct">{SVG_OK} <span id="iq-correct-num">0</span> correct</span>
    <span class="iq-verdict-wrong">{SVG_BAD} <span id="iq-wrong-num">0</span> wrong</span>
  </div>
  <div class="iq-verdict-strip" id="iq-verdict-strip"></div>
  <div class="iq-score-actions">
    <button class="iq-score-btn iq-retry-btn" onclick="IQ_RETRY_INCORRECT()">Retry incorrect</button>
    <button class="iq-score-btn" onclick="IQ_RESTART()">Restart quiz</button>
  </div>
</div>
</div>"""


# ---------------------------------------------------------------------------
# Flashcards  (3D card flip with prev / next / shuffle)
# ---------------------------------------------------------------------------


def render_flashcard_html(flashcard_set: FlashcardSet) -> str:
    """Return HTML with flashcard data in ``data-cards`` attribute for the MutationObserver."""
    if not flashcard_set.cards:
        return '<div class="fc-root fc-empty">No cards available.</div>'

    citation_map = {
        citation.source_marker: {
            "marker": citation.source_marker,
            "filename": citation.filename,
            "page": citation.page,
            "text": citation.source_text or "",
        }
        for citation in flashcard_set.citations
    }
    data = [
        {
            "front": card.front,
            "back": card.back,
            "hint": card.hint or "",
            "sources": [
                citation_map[marker]
                for marker in card.source_markers
                if marker in citation_map
            ],
        }
        for card in flashcard_set.cards
    ]
    data_json = json.dumps(data, ensure_ascii=False)
    scope_escaped = html_mod.escape(
        f"{flashcard_set.scope}: {flashcard_set.target}"
        if flashcard_set.target
        else flashcard_set.scope
    )

    SVG_PREV = (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
        'width="16" height="16"><path d="m15 18-6-6 6-6"/></svg>'
    )
    SVG_NEXT = (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
        'width="16" height="16"><path d="m9 18 6-6-6-6"/></svg>'
    )
    SVG_SHUFFLE = (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
        'width="16" height="16">'
        '<polyline points="16 3 21 3 21 8"/>'
        '<line x1="4" y1="20" x2="21" y2="3"/>'
        '<polyline points="21 16 21 21 16 21"/>'
        '<line x1="15" y1="15" x2="21" y2="21"/>'
        '<line x1="4" y1="4" x2="9" y2="9"/></svg>'
    )

    return f"""<div class="fc-root" data-cards='{_attr_escape(data_json)}'>
<div class="fc-header">
  <span class="fc-scope">{scope_escaped}</span>
  <span class="fc-progress" id="fc-progress">1 / {len(data)}</span>
</div>
<div class="fc-card-wrapper" onclick="FC_FLIP()">
  <div class="fc-card-inner" id="fc-inner">
    <div class="fc-face" id="fc-front"><h3>Question</h3><p id="fc-front-text"></p><span class="fc-flip-hint">Tap to reveal</span></div>
    <div class="fc-face fc-back" id="fc-back"><h3>Answer</h3><p id="fc-back-text"></p><div class="fc-sources" id="fc-sources"></div></div>
  </div>
</div>
<div class="fc-dots" id="fc-dots"></div>
<div class="fc-hint" id="fc-hint"></div>
<div class="fc-nav">
  <button id="fc-prev" onclick="FC_PREV()">{SVG_PREV} Prev</button>
  <button class="fc-shuffle-btn" onclick="FC_SHUFFLE()">{SVG_SHUFFLE} Shuffle</button>
  <button id="fc-next" onclick="FC_NEXT()">Next {SVG_NEXT}</button>
</div>
<div class="fc-complete" id="fc-complete" style="display:none">
  <div class="fc-complete-kicker">Deck complete</div>
  <h3>Nice work. Keep the difficult cards warm.</h3>
  <div class="fc-complete-stats">
    <span><strong id="fc-mastered-num">0</strong><small>mastered</small></span>
    <span><strong id="fc-difficult-num">0</strong><small>to review</small></span>
  </div>
  <div class="fc-complete-actions">
    <button onclick="FC_REVIEW()">Review difficult cards</button>
    <button onclick="FC_RESTART()">Restart deck</button>
  </div>
</div>
</div>"""
