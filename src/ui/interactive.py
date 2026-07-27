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

// ==================== Flashcards ====================
var _fcD = null, _fcI = 0, _fcF = false, _fcO = null, _fcK = null, _fcFilt = 0;

function _fcW() { return document.querySelector('.fc-card-wrapper'); }

function _fcShow(dir) {
  var c = _fcD[_fcO[_fcI]]; if (!c) return;
  var ft = document.getElementById('fc-front-text');
  var bt = document.getElementById('fc-back-text');
  var hi = document.getElementById('fc-hint');
  var pr = document.getElementById('fc-progress');
  if (ft) { ft.textContent = c.front; ft.style.animation = ''; void ft.offsetWidth; ft.style.animation = 'fc-slide-in ' + (dir || 'left') + ' 0.35s ease-out'; }
  if (bt) bt.textContent = c.back;
  if (hi) { hi.textContent = c.hint || ''; hi.style.opacity = '0'; setTimeout(function() { hi.style.opacity = '1'; }, 100); }
  if (pr) pr.textContent = (_fcI + 1) + ' / ' + _fcD.length;
  var pn = document.getElementById('fc-prev'), nn = document.getElementById('fc-next');
  if (pn) pn.style.visibility = _fcI === 0 ? 'hidden' : 'visible';
  if (nn) nn.style.visibility = _fcI === _fcD.length - 1 ? 'hidden' : 'visible';
  var dots = document.getElementById('fc-dots');
  if (dots) {
    dots.innerHTML = _fcD.map(function(_, i) {
      return '<span class="fc-dot' + (i === _fcI ? ' fc-dot-active' : '') + '"></span>';
    }).join('');
  }
  var mk = document.getElementById('fc-mastery');
  if (mk) {
    var kb = mk.querySelector('.fc-known-btn'), ub = mk.querySelector('.fc-unknown-btn');
    if (_fcK[_fcO[_fcI]] === true) { if (kb) kb.classList.add('fc-active'); if (ub) ub.classList.remove('fc-active'); }
    else if (_fcK[_fcO[_fcI]] === false) { if (ub) ub.classList.add('fc-active'); if (kb) kb.classList.remove('fc-active'); }
    else { if (kb) kb.classList.remove('fc-active'); if (ub) ub.classList.remove('fc-active'); }
  }
  var w = _fcW();
  if (w && _fcF) { w.classList.remove('fc-flipped'); _fcF = false; }
  var fb = document.getElementById('fc-filter-btn');
  if (fb) { fb.textContent = _fcFilt === 1 ? 'Unknown' : 'All'; }
}

function _fcInit(root) {
  if (!root) root = document.querySelector('.fc-root');
  if (!root) return;
  var raw = root.getAttribute('data-cards');
  if (!raw) return;
  _fcD = JSON.parse(raw);
  _fcI = 0; _fcF = false;
  _fcO = _fcD.map(function(_, i) { return i; });
  _fcK = new Array(_fcD.length).fill(null);
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

window.FC_KNOWN = function() {
  var ci = _fcO[_fcI];
  if (_fcK[ci] === true) _fcK[ci] = null; else _fcK[ci] = true;
  _fcShow();
};

window.FC_UNKNOWN = function() {
  var ci = _fcO[_fcI];
  if (_fcK[ci] === false) _fcK[ci] = null; else _fcK[ci] = false;
  _fcShow();
};

window.FC_FILTER = function() {
  _fcFilt = _fcFilt === 1 ? 0 : 1;
  if (_fcFilt === 1 && _fcK[_fcO[_fcI]] !== false) {
    var ni; for (ni = 0; ni < _fcD.length; ni++) { if (_fcK[_fcO[ni]] === false) break; }
    if (ni < _fcD.length) { _fcI = ni; _fcF = false; var w = _fcW(); if (w) w.classList.remove('fc-flipped'); _fcShow('right'); }
    else { _fcFilt = 0; _fcShow(); }
    return;
  }
  _fcShow();
};

// ==================== Quiz ====================
var _qD = null, _qDone = null, _qTC = 0, _qTW = 0, _qV = null;
var _qL = ['A','B','C','D'];
var _qSVG_OK = '<svg class="iq-verdict-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/></svg>';
var _qSVG_BAD = '<svg class="iq-verdict-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>';

function _qRender() {
  var c = document.getElementById('iq-list'); if (!c) return;
  var h = '';
  for (var i = 0; i < _qD.length; i++) { var q = _qD[i];
    h += '<div class="iq-card" style="animation-delay:' + (i * 100) + 'ms">';
    h += '<div class="iq-question">' + _esc(q.q) + '</div>';
    h += '<div class="iq-options">';
    for (var j = 0; j < q.opts.length; j++) {
      h += '<div class="iq-option" data-i="' + i + '" data-j="' + j + '" onclick="IQ_ANSWER(this)">';
      h += '<span class="iq-option-label">' + _qL[j] + '</span>';
      h += '<span class="iq-option-text">' + _esc(q.opts[j]) + '</span></div>';
    }
    h += '</div>';
    h += '<div class="iq-explain" id="iq-ex-' + i + '" style="display:none"></div>';
    h += '</div>';
  }
  c.innerHTML = h;
}

function _qInit(root) {
  if (!root) root = document.querySelector('.iq-root');
  if (!root) return;
  var raw = root.getAttribute('data-quiz');
  if (!raw) return;
  _qD = JSON.parse(raw);
  _qDone = new Array(_qD.length).fill(false);
  _qTC = 0; _qTW = 0;
  _qV = new Array(_qD.length).fill(false);
  _qRender();
}

window.IQ_ANSWER = function(el) {
  var i = parseInt(el.dataset.i), j = parseInt(el.dataset.j); var q = _qD[i];
  var ops = document.querySelectorAll('.iq-option[data-i="' + i + '"]');
  if (ops[0] && ops[0].classList.contains('iq-disabled')) return;
  ops.forEach(function(o) { o.classList.add('iq-disabled'); });
  if (j === q.correct) { el.classList.add('iq-correct'); _qTC++; }
  else { el.classList.add('iq-wrong'); ops[q.correct].classList.add('iq-correct'); _qTW++; }
  _qV[i] = j === q.correct;
  _qDone[i] = true;
  var ex = document.getElementById('iq-ex-' + i);
  if (ex) {
    var ok = j === q.correct;
    var verdictClass = ok ? 'iq-verdict-correct' : 'iq-verdict-wrong';
    ex.innerHTML = '<div class="' + verdictClass + '" style="margin-bottom:6px">' +
      (ok ? _qSVG_OK : _qSVG_BAD) + ' ' + (ok ? 'Correct' : 'Wrong') + '</div>' +
      '<strong>' + _qL[q.correct] + '. ' + _esc(q.opts[q.correct]) + '</strong><br>' +
      _esc(q.explanation);
    ex.style.display = 'block';
  }
  var ans = _qDone.filter(Boolean).length;
  var ct = document.getElementById('iq-counter');
  if (ct) ct.textContent = ans + ' / ' + _qD.length + ' answered';
  var fill = document.getElementById('iq-fill');
  if (fill) fill.style.width = Math.round(ans / _qD.length * 100) + '%';
  if (ans === _qD.length) {
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
          ring.style.strokeDashoffset = circ * (1 - target / _qD.length);
        }
        document.getElementById('iq-correct-num').textContent = _qTC;
        document.getElementById('iq-wrong-num').textContent = _qTW;
        var strip = document.getElementById('iq-verdict-strip');
        if (strip) {
          var vh = '';
          for (var vi = 0; vi < _qV.length; vi++) {
            vh += '<span class="iq-vdot ' + (_qV[vi] ? 'iq-vdot-correct' : 'iq-vdot-wrong') + '" data-i="' + vi + '" onclick="IQ_SCROLL_TO(' + vi + ')"></span>';
          }
          strip.innerHTML = vh;
        }
        sc.style.display = 'block';
        sc.classList.add('iq-score-reveal');
        sc.scrollIntoView({behavior:'smooth',block:'center'});
      }
    }, 600);
  } else {
    var cards = document.querySelectorAll('.iq-card');
    for (var k = i + 1; k < _qD.length; k++) {
      if (!_qDone[k] && cards[k]) {
        cards[k].scrollIntoView({behavior:'smooth',block:'center'}); break;
      }
    }
  }
};

window.IQ_SCROLL_TO = function(i) {
  var cards = document.querySelectorAll('.iq-card');
  if (cards[i]) cards[i].scrollIntoView({behavior:'smooth',block:'center'});
};

window.IQ_RESTART = function() {
  _qDone = new Array(_qD.length).fill(false); _qTC = 0; _qTW = 0;
  _qV = new Array(_qD.length).fill(false);
  var sc = document.getElementById('iq-score');
  if (sc) sc.style.display = 'none';
  var ct = document.getElementById('iq-counter');
  if (ct) ct.textContent = '0 / ' + _qD.length + ' answered';
  var fill = document.getElementById('iq-fill');
  if (fill) fill.style.width = '0%';
  _qRender();
};

// ==================== Keyboard ====================
document.addEventListener('keydown', function(e) {
  if (!document.querySelector('.fc-root')) return;
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  if (e.key === 'ArrowLeft') { FC_PREV(); }
  else if (e.key === 'ArrowRight') { FC_NEXT(); }
  else if (e.key === ' ') { e.preventDefault(); FC_FLIP(); }
  else if (e.key === 'f' || e.key === 'F') { FC_FILTER(); }
});

// ==================== MutationObserver ====================
var _obs = new MutationObserver(function(muts) {
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
_obs.observe(document.body, { childList: true, subtree: true });

// ==================== Check existing ====================
document.addEventListener('DOMContentLoaded', function() {
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

    data = [
        {
            "q": item.question,
            "opts": item.options,
            "correct": item.correct_index,
            "explanation": item.explanation or "No explanation provided.",
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

    return f"""<div class="iq-root" data-quiz='{data_json}'>
<div class="iq-header">
  <span class="iq-scope">{scope_escaped}</span>
  <span class="iq-counter" id="iq-counter">0 / {len(data)} answered</span>
</div>
<div class="iq-progress-track">
  <div class="iq-progress-fill" id="iq-fill" style="width:0%"></div>
</div>
<div class="iq-list" id="iq-list"></div>
<div class="iq-score" id="iq-score" style="display:none">
  <div class="iq-score-ring">
    <svg width="120" height="120" viewBox="0 0 120 120">
      <circle class="iq-ring-bg" cx="60" cy="60" r="52" fill="none" stroke-width="8"/>
      <circle class="iq-ring-fill" id="iq-ring-fill" cx="60" cy="60" r="52" fill="none" stroke-width="8" stroke-dasharray="326.73" stroke-dashoffset="326.73" transform="rotate(-90 60 60)" stroke-linecap="round"/>
    </svg>
    <div class="iq-score-num" id="iq-score-num">0</div>
    <div class="iq-score-total">/ {len(data)}</div>
  </div>
  <div class="iq-score-label">Your Score</div>
  <div class="iq-score-detail">
    <span class="iq-verdict-correct">{SVG_OK} <span id="iq-correct-num">0</span> correct</span>
    <span class="iq-verdict-wrong">{SVG_BAD} <span id="iq-wrong-num">0</span> wrong</span>
  </div>
  <div class="iq-verdict-strip" id="iq-verdict-strip"></div>
  <button class="iq-score-btn" onclick="IQ_RESTART()">Try Again</button>
</div>
</div>"""


# ---------------------------------------------------------------------------
# Flashcards  (3D card flip with prev / next / shuffle)
# ---------------------------------------------------------------------------


def render_flashcard_html(flashcard_set: FlashcardSet) -> str:
    """Return HTML with flashcard data in ``data-cards`` attribute for the MutationObserver."""
    if not flashcard_set.cards:
        return '<div class="fc-root fc-empty">No cards available.</div>'

    data = [{"front": c.front, "back": c.back, "hint": c.hint or ""} for c in flashcard_set.cards]
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

    return f"""<div class="fc-root" data-cards='{data_json}'>
<div class="fc-header">
  <span class="fc-scope">{scope_escaped}</span>
  <span class="fc-progress" id="fc-progress">1 / {len(data)}</span>
</div>
<div class="fc-card-wrapper" onclick="FC_FLIP()">
  <div class="fc-card-inner" id="fc-inner">
    <div class="fc-face" id="fc-front"><h3>Question</h3><p id="fc-front-text"></p><span class="fc-flip-hint">Tap to reveal</span></div>
    <div class="fc-face fc-back" id="fc-back"><h3>Answer</h3><p id="fc-back-text"></p></div>
  </div>
</div>
<div class="fc-dots" id="fc-dots"></div>
<div class="fc-hint" id="fc-hint"></div>
<div class="fc-mastery" id="fc-mastery">
  <button class="fc-known-btn" onclick="FC_KNOWN()">Got it</button>
  <button class="fc-unknown-btn" onclick="FC_UNKNOWN()">Still learning</button>
</div>
<div class="fc-nav">
  <button id="fc-prev" onclick="FC_PREV()">{SVG_PREV} Prev</button>
  <button class="fc-shuffle-btn" onclick="FC_SHUFFLE()">{SVG_SHUFFLE} Shuffle</button>
  <button class="fc-filter-btn" id="fc-filter-btn" onclick="FC_FILTER()">All</button>
  <button id="fc-next" onclick="FC_NEXT()">Next {SVG_NEXT}</button>
</div>
</div>"""
