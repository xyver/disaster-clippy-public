"""
Minimal FastAPI app: a search/ask page plus a JSON API.

    GET  /                 web page (search + ask, with cited passages)
    GET  /api/info         index summary (documents, editions, embedder)
    GET  /api/search?q=...&k=8&mode=hybrid&filter=edition:2025-26
    POST /api/ask          {"question": "...", "filters": {"edition": "2025-26"}}
    GET  /files/{doc_id}   the original PDF, when the manifest gives no public url

This is a starting point, not a product UI. Replace the page freely; the API
is what other front ends should call.
"""


import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .chat import ChatService
from .config import ClippyConfig
from .schemas import SearchResult


def _parse_filters(values: Optional[List[str]]) -> Optional[Dict[str, Any]]:
    out: Dict[str, Any] = {}
    for v in values or []:
        if ":" in v:
            key, val = v.split(":", 1)
            out[key] = val
    return out or None


def create_app(config: ClippyConfig, store):
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import FileResponse, HTMLResponse
    from pydantic import BaseModel

    app = FastAPI(title="clippy_core", version="0.2.0")
    chat = ChatService(store, config=config)

    def doc_paths() -> Dict[str, str]:
        rows = store.conn.execute(
            "SELECT DISTINCT json_extract(metadata,'$.doc_id') AS d, json_extract(metadata,'$.path') AS p "
            "FROM chunks").fetchall()
        return {r["d"]: r["p"] for r in rows if r["d"] and r["p"]}

    def serialise(results: List[SearchResult]) -> List[Dict[str, Any]]:
        out = []
        for n, r in enumerate(results, 1):
            d = r.to_dict()
            d["n"] = n
            if not d["url"] and r.metadata.get("doc_id"):
                page = r.metadata.get("page_start")
                d["url"] = f"/files/{r.metadata['doc_id']}" + (f"#page={page}" if page else "")
            out.append(d)
        return out

    class AskRequest(BaseModel):
        question: str
        filters: Optional[Dict[str, Any]] = None

    @app.get("/api/info")
    def info():
        return store.info()

    @app.get("/api/search")
    def search(q: str, k: int = 8, mode: str = "hybrid", filter: Optional[List[str]] = Query(None)):
        results = store.search(q, n_results=k, filters=_parse_filters(filter), mode=mode)
        return {"query": q, "mode": mode, "results": serialise(results)}

    @app.post("/api/ask")
    async def ask(req: AskRequest):
        resp = await chat.chat(req.question, filters=req.filters)
        return {"answer": resp.text, "method": resp.method.value, "error": resp.error,
                "results": serialise(resp.search_results)}

    @app.get("/files/{doc_id}")
    def files(doc_id: str):
        path = doc_paths().get(doc_id)
        if not path or not Path(path).exists():
            raise HTTPException(404, "Document not available")
        return FileResponse(path, media_type="application/pdf")

    @app.get("/", response_class=HTMLResponse)
    def page():
        editions = sorted({d["edition"] for d in store.info()["documents"] if d.get("edition")})
        return PAGE.replace("__EDITIONS__", json.dumps(editions))

    return app


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rules search</title>
<style>
:root{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#6b6a65;--line:#e4e1d8;--card:#fff;--accent:#2f5d8a;--quote:#f3f0e8}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--muted:#a09e96;--line:#2e2d2a;--card:#1f1e1c;--accent:#8fb7df;--quote:#262521}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:780px;margin:0 auto;padding:32px 16px 64px}h1{font-size:1.4rem;margin:0 0 4px}p.sub{color:var(--muted);margin:0 0 20px}
form{display:flex;gap:8px;flex-wrap:wrap}input,select,button{font:inherit;padding:10px 12px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg)}
input{flex:1 1 320px}button{background:var(--accent);color:var(--bg);border:0;cursor:pointer}button.alt{background:transparent;color:var(--accent);border:1px solid var(--accent)}
#answer{white-space:pre-wrap;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:20px 0;display:none}
.r{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:12px 0}
.cite{font-weight:600}.cite a{color:var(--accent);text-decoration:none}.meta{color:var(--muted);font-size:.85rem}
blockquote{margin:8px 0 0;padding:10px 12px;background:var(--quote);border-radius:6px;white-space:pre-wrap;font-size:.92rem;max-height:14em;overflow:auto}
</style></head><body><main>
<h1>Rules search</h1><p class="sub">Every answer comes with the rule text and a link to the page it came from.</p>
<form id="f"><input id="q" placeholder="Ask a question or search for a term" autofocus>
<select id="ed"><option value="">All editions</option></select>
<button type="submit" data-act="ask">Ask</button><button type="button" class="alt" data-act="search">Search only</button></form>
<div id="answer"></div><div id="results"></div>
<script>
const eds=__EDITIONS__, sel=document.getElementById('ed');
eds.forEach(e=>{const o=document.createElement('option');o.value=o.textContent=e;sel.appendChild(o)});
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function show(results){document.getElementById('results').innerHTML=results.map(r=>`<div class="r">
<div class="cite">[${r.n}] ${r.url?`<a href="${esc(r.url)}" target="_blank" rel="noopener">${esc(r.citation)}</a>`:esc(r.citation)}</div>
<div class="meta">${esc(r.metadata.section_title||'')}</div><blockquote>${esc(r.content)}</blockquote></div>`).join('')||'<p class="sub">No results.</p>'}
async function run(act){const q=document.getElementById('q').value.trim();if(!q)return;const ed=sel.value,ans=document.getElementById('answer');
document.getElementById('results').innerHTML='<p class="sub">Searching…</p>';ans.style.display='none';
if(act==='ask'){const r=await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q,filters:ed?{edition:ed}:null})}).then(r=>r.json());
ans.textContent=r.method==='simple'?'No LLM is configured, so these are the matching passages. Set ANTHROPIC_API_KEY or OPENAI_API_KEY for written answers.':r.answer;ans.style.display='block';show(r.results)}
else{const p=new URLSearchParams({q});if(ed)p.append('filter','edition:'+ed);const r=await fetch('/api/search?'+p).then(r=>r.json());show(r.results)}}
document.getElementById('f').addEventListener('submit',e=>{e.preventDefault();run('ask')});
document.querySelector('[data-act=search]').addEventListener('click',()=>run('search'));
</script></main></body></html>"""
