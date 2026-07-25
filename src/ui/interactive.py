"""Render QuizSet/FlashcardSet as self-contained interactive HTML with client-side JS.

All interactivity runs in the browser.  No server round-trips after the initial
HTML generation.  Gradio needs ``sanitize_html=False`` on the ``gr.HTML``
component to allow ``<script>`` tags.
"""

from __future__ import annotations

import html as html_mod
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.schemas import FlashcardSet, QuizSet


# ---------------------------------------------------------------------------
# Quiz  (click-to-answer with instant feedback)
# ---------------------------------------------------------------------------


def render_quiz_html(quiz_set: QuizSet) -> str:
    """Return a self-contained HTML block with clickable quiz cards."""
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

    return f"""<div class="iq-root">
<div class="iq-header">
  <span class="iq-scope">{scope_escaped}</span>
  <span class="iq-counter" id="iq-counter">0/{len(data)} answered</span>
</div>
<div class="iq-list" id="iq-list"></div>
</div>
<script>
(function(){{
var Q={data_json};var L=['A','B','C','D','E','F'];
var done=new Array(Q.length).fill(false);

function e(s){{var d=document.createElement('div');d.textContent=s;return d.innerHTML;}}

function R(){{
  var c=document.getElementById('iq-list');if(!c)return;
  var h='';
  for(var i=0;i<Q.length;i++){{var q=Q[i];
    h+='<div class="iq-card" id="iqc-'+i+'"><div class="iq-no">Q '+(i+1)+'/'+Q.length+'</div>';
    h+='<div class="iq-q">'+e(q.q)+'</div><div class="iq-ops">';
    for(var j=0;j<q.opts.length;j++){{
      h+='<div class="iq-opt" data-i="'+i+'" data-j="'+j+'" onclick="QK(this)">';
      h+='<span class="iq-l">'+L[j]+'</span><span class="iq-t">'+e(q.opts[j])+'</span></div>';
    }}
    h+='</div><div class="iq-fb" id="iqfb-'+i+'"></div></div>';
  }}
  c.innerHTML=h;
}}

window.QK=function(el){{
  var i=parseInt(el.dataset.i),j=parseInt(el.dataset.j);var q=Q[i];
  var ops=document.querySelectorAll('.iq-opt[data-i="'+i+'"]');
  if(ops[0]&&ops[0].classList.contains('iq-d'))return;
  ops.forEach(function(o){{o.classList.add('iq-d');}});
  el.classList.add(j===q.correct?'iq-ok':'iq-ko');
  ops[q.correct].classList.add('iq-ok');
  done[i]=true;
  var fb=document.getElementById('iqfb-'+i);
  if(fb){{
    var ok=j===q.correct;
    fb.innerHTML=(
      '<div class="iq-v iq-v-'+(ok?'t':'f')+'">'+
        (ok?'&#10003;':'&#10007;')+' '+        (ok?'Correct':'Wrong')+'</div>'+
      '<div class="iq-a"><b>'+L[q.correct]+'. '+e(q.opts[q.correct])+'</b></div>'+
      '<div class="iq-x">'+e(q.explanation)+'</div>'
    );fb.classList.add('iq-s');
  }}
  var n=done.filter(Boolean).length;
  var ct=document.getElementById('iq-counter');
  if(ct)ct.textContent=n+'/'+Q.length+' answered';
  for(var k=i+1;k<Q.length;k++){{if(!done[k]){{
    var nx=document.getElementById('iqc-'+k);
    if(nx)nx.scrollIntoView({{behavior:'smooth',block:'center'}});break;
  }}}}
}};

R();
}})();
</script>"""


# ---------------------------------------------------------------------------
# Flashcards  (3D card flip with prev / next / shuffle)
# ---------------------------------------------------------------------------


def render_flashcard_html(flashcard_set: FlashcardSet) -> str:
    """Return a self-contained HTML block with interactive flashcard deck."""
    if not flashcard_set.cards:
        return '<div class="fc-root fc-empty">No cards available.</div>'

    data = [{"front": c.front, "back": c.back, "hint": c.hint or ""} for c in flashcard_set.cards]
    data_json = json.dumps(data, ensure_ascii=False)
    scope_escaped = html_mod.escape(
        f"{flashcard_set.scope}: {flashcard_set.target}"
        if flashcard_set.target
        else flashcard_set.scope
    )

    return f"""<div class="fc-root">
<div class="fc-scope">{scope_escaped}</div>
<div class="fc-stage" id="fc-stage">
  <div class="fc-card" id="fc-card" onclick="FK()">
    <div class="fc-f" id="fc-f"></div>
    <div class="fc-b" id="fc-b"></div>
  </div>
  <div class="fc-ht" id="fc-ht"></div>
  <div class="fc-pg" id="fc-pg"></div>
</div>
<div class="fc-ctrls">
  <button class="fc-btn" id="fc-pr" onclick="FP()">&#9664; Prev</button>
  <button class="fc-btn fc-btn-sh" onclick="FS()">&#8635; Shuffle</button>
  <button class="fc-btn" id="fc-nx" onclick="FN()">Next &#9654;</button>
</div>
</div>
<script>
(function(){{
var D={data_json};var I=0;var F=false;
var O=D.map(function(_,i){{return i;}});

function e(s){{var d=document.createElement('div');d.textContent=s;return d.innerHTML;}}

function C(){{
  var c=D[O[I]];if(!c)return;
  var f=document.getElementById('fc-f'),b=document.getElementById('fc-b');
  var h=document.getElementById('fc-ht'),p=document.getElementById('fc-pg');
  var cd=document.getElementById('fc-card');
  if(f)f.innerHTML=e(c.front);if(b)b.innerHTML=e(c.back);
  if(h)h.innerHTML=c.hint?e(c.hint):'&nbsp;';
  if(p)p.textContent=(I+1)+'/'+D.length;
  if(cd&&F){{cd.classList.remove('f');F=false;}}
  var pr=document.getElementById('fc-pr'),nx=document.getElementById('fc-nx');
  if(pr)pr.style.visibility=I===0?'hidden':'visible';
  if(nx)nx.style.visibility=I===D.length-1?'hidden':'visible';
}}

window.FK=function(){{
  var cd=document.getElementById('fc-card');if(!cd)return;
  F=!F;cd.classList.toggle('f',F);
}};
window.FN=function(){{if(I<D.length-1){{I++;F=false;C();}}}};
window.FP=function(){{if(I>0){{I--;F=false;C();}}}};
window.FS=function(){{
  for(var i=O.length-1;i>0;i--){{
    var j=Math.floor(Math.random()*(i+1));
    var t=O[i];O[i]=O[j];O[j]=t;
  }}I=0;F=false;C();
}};

C();
}})();
</script>"""
