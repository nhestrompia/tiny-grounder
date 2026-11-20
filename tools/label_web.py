"""Local web UI for writing the human evaluation set: one page screenshot at a time, you write what you would say
and click the element it refers to.

    uv run python tools/label_web.py --pages data/heldout/pages/test --assign data/heldout/confirm_pages.txt \\
        --out data/heldout/confirm.jsonl --target 150 --annotator user

Opens http://127.0.0.1:8765 in the browser. Pages come, in file order, from the `--assign` list (one page id per
line, `#` comments allowed). Each saved entry is appended to the output JSONL at once as a `records.Example`
(kind "hand", mode "do", `equiv_ids` from `records.equiv_ids`) plus the fields `split`, `annotator` and `tags`.
"Not on this page" saves a null example (`target_id` null, no equivalents). Stop with Ctrl-C; running the same
command again resumes where you left off. Pages you finished or skipped are kept in `{out}.pages.jsonl`.

Keys (the instruction box keeps focus): Enter = save, Alt+Right = next page, Alt+S = skip page, Alt+Z = undo,
Alt+0 = not on this page, Alt+1/2/3 = tags icon / position / vague, Esc = clear the clicked element,
Alt+N = show or hide element numbers. Clicking the same spot again picks the next larger box under it.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grounder.data.pages import load_page, page_paths  # noqa: E402
from grounder.records import Example, Page, equiv_ids  # noqa: E402
from tools.label import KIND, _fractions, png_size, screenshot_path  # noqa: E402

TAGS: tuple[str, ...] = ("icon", "position", "vague")
MODE = "do"
Response = tuple[int, str, bytes]


def read_assignment(path: Path) -> list[str]:
    """Page ids, one per line, in order; blank lines and `#` comments are ignored; duplicates dropped."""
    seen: dict[str, None] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line:
            seen.setdefault(line)
    return list(seen)


def default_split(assign: Path) -> str:
    """`confirm_pages.txt` -> "confirm"."""
    return assign.stem.removesuffix("_pages")


def make_entry(page: Page, instruction: str, target_id: int | None, tags: list[str], split: str, annotator: str) -> dict:
    """One output line: an Example (kind "hand", mode "do") with `split`, `annotator` and `tags` added."""
    if target_id is not None and target_id not in {e.id for e in page.elements}:
        raise ValueError(f"element {target_id} is not on page {page.page_id}")
    ids = [] if target_id is None else equiv_ids(page.elements, target_id)
    entry = Example(page.page_id, KIND, MODE, instruction, [], target_id, ids).to_dict()
    entry.update(split=split, annotator=annotator, tags=[t for t in TAGS if t in tags])
    return entry


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)


class LabelSession:
    """Everything the UI does, without HTTP: which page is next, saving, undo, skip, progress.

    The output JSONL is the source of truth for entries; `{out}.pages.jsonl` records pages left with "next" or
    "skip" (`{"page_id": ..., "action": ...}`), so a re-run does not show them again. A page is also finished
    once it has `per_page` entries.
    """

    def __init__(
        self, pages_dir: Path, assign: Path, out: Path, target: int, annotator: str,
        split: str | None = None, per_page: int = 3,
    ) -> None:
        self.out, self.target, self.annotator, self.per_page = out, target, annotator, per_page
        self.split = split or default_split(assign)
        self.marks_path = out.with_name(out.name + ".pages.jsonl")
        self.lock = threading.Lock()
        files = {p.name.removesuffix(".json.gz"): p for p in page_paths(pages_dir)}
        self.order = [pid for pid in read_assignment(assign) if pid in files]
        self.missing = [pid for pid in read_assignment(assign) if pid not in files]
        self.files = files
        self._pages: dict[str, Page] = {}
        out.parent.mkdir(parents=True, exist_ok=True)

    # -- state ---------------------------------------------------------------------------------

    def page(self, page_id: str) -> Page:
        if page_id not in self._pages:
            self._pages[page_id] = load_page(self.files[page_id])
        return self._pages[page_id]

    def entries(self) -> list[dict]:
        return read_jsonl(self.out)

    def finished(self, entries: list[dict] | None = None) -> set[str]:
        entries = self.entries() if entries is None else entries
        counts: dict[str, int] = {}
        for e in entries:
            counts[e["page_id"]] = counts.get(e["page_id"], 0) + 1
        done = {pid for pid, n in counts.items() if n >= self.per_page}
        return done | {m["page_id"] for m in read_jsonl(self.marks_path)}

    def state(self) -> dict:
        """The JSON the UI renders: the current page, its entries so far, and progress."""
        with self.lock:
            entries = self.entries()
            done = self.finished(entries)
            pid = next((p for p in self.order if p not in done and screenshot_path(self.files[p]).exists()), None)
            progress = {"n": len(entries), "target": self.target, "pages_left": sum(p not in done for p in self.order)}
            if pid is None:
                return {"page": None, "progress": progress, "tags": list(TAGS)}
            page = self.page(pid)
            with screenshot_path(self.files[pid]).open("rb") as f:
                width, height = png_size(f.read(24))
            return {
                "page": {
                    "id": pid, "url": page.url, "title": page.title, "image": f"/shot/{pid}.png",
                    "width": width, "height": height,
                    "elements": [{"id": e.id, "role": e.role, "box": _fractions(page, e, width, height)}
                                 for e in page.elements],
                },
                "entries": [{"query": e["query"], "target_id": e["target_id"], "tags": e.get("tags", [])}
                            for e in entries if e["page_id"] == pid],
                "per_page": self.per_page, "progress": progress, "tags": list(TAGS),
            }

    # -- actions -------------------------------------------------------------------------------

    def save(self, page_id: str, instruction: str, target_id: int | None, tags: list[str]) -> dict:
        instruction = " ".join(instruction.split())
        if not instruction:
            raise ValueError("write what you would tell someone to do")
        if page_id not in self.files:
            raise ValueError(f"unknown page {page_id!r}")
        with self.lock:
            entry = make_entry(self.page(page_id), instruction, target_id, tags, self.split, self.annotator)
            with self.out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        return entry

    def mark(self, page_id: str, action: str) -> None:
        """Leave a page ("next" or "skip"); it is not shown again."""
        if action not in ("next", "skip") or page_id not in self.files:
            raise ValueError("bad page action")
        with self.lock:
            with self.marks_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"page_id": page_id, "action": action}) + "\n")

    def undo(self) -> dict | None:
        """Remove the last saved entry and reopen its page; returns the removed entry (None if there is none)."""
        with self.lock:
            entries = self.entries()
            if not entries:
                return None
            removed = entries.pop()
            write_jsonl(self.out, entries)
            if self.marks_path.exists():  # reopen its page: it is the first unfinished page in order
                write_jsonl(self.marks_path, [m for m in read_jsonl(self.marks_path) if m["page_id"] != removed["page_id"]])
            return removed

    def screenshot(self, page_id: str) -> bytes | None:
        path = screenshot_path(self.files[page_id]) if page_id in self.files else None
        return path.read_bytes() if path is not None and path.exists() else None

    # -- HTTP-free request handling ------------------------------------------------------------

    def handle(self, method: str, path: str, body: bytes = b"") -> Response:
        """Route one request; returns (status, content type, body). Never raises for bad input."""
        try:
            if method == "GET":
                if path == "/":
                    return 200, "text/html; charset=utf-8", PAGE_HTML.encode()
                if path == "/api/state":
                    return _json(self.state())
                if path.startswith("/shot/") and path.endswith(".png"):
                    data = self.screenshot(path[len("/shot/"):-len(".png")])
                    return (200, "image/png", data) if data is not None else (404, "text/plain", b"no screenshot")
            elif method == "POST":
                data = json.loads(body or b"{}")
                if path == "/api/save":
                    target = data.get("target_id")
                    if target is not None and not isinstance(target, int):
                        raise ValueError("target_id must be an element number or null")
                    self.save(str(data.get("page_id", "")), str(data.get("instruction", "")), target,
                              [str(t) for t in data.get("tags", [])])
                    return _json(self.state())
                if path in ("/api/next", "/api/skip"):
                    self.mark(str(data.get("page_id", "")), path.removeprefix("/api/"))
                    return _json(self.state())
                if path == "/api/undo":
                    self.undo()
                    return _json(self.state())
            return 404, "text/plain", b"not found"
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            return 400, "application/json", json.dumps({"error": str(e)}).encode()


def _json(obj: dict) -> Response:
    return 200, "application/json", json.dumps(obj).encode()


def make_handler(session: LabelSession) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, method: str) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            status, content_type, data = session.handle(method, self.path.split("?")[0], self.rfile.read(length))
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            self._reply("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._reply("POST")

        def log_message(self, *args) -> None:
            pass

    return Handler


PAGE_HTML = """<!doctype html><meta charset="utf-8"><title>Write instructions</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{color-scheme:light dark;--bg:#f6f6f4;--fg:#1c1c1c;--card:#fff;--line:#d9d9d4;--accent:#1a5fd0;--sel:#e0472b}
@media (prefers-color-scheme:dark){:root{--bg:#18181a;--fg:#ececec;--card:#232326;--line:#3a3a3f;--accent:#6aa0ff;--sel:#ff7a5c}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif}
.layout{display:grid;grid-template-columns:minmax(0,1fr) 360px;gap:16px;padding:16px;align-items:start}
@media (max-width:900px){.layout{grid-template-columns:1fr}}
.shot{background:#fff;border:1px solid var(--line);border-radius:8px;overflow:hidden}
.stage{position:relative;cursor:crosshair;user-select:none}.stage img{display:block;width:100%}
.box{position:absolute;pointer-events:none;border:1px solid rgba(26,95,208,.0)}
.show-all .box{border-color:rgba(26,95,208,.35)}
.box.hover{border:2px solid var(--accent);background:rgba(26,95,208,.12)}
.box.sel{border:3px solid var(--sel);background:rgba(224,71,43,.18)}
.box b{display:none;position:absolute;left:-1px;top:-1px;transform:translateY(-100%);padding:0 4px;font:600 11px monospace;background:var(--accent);color:#fff}
.nums .box b,.box.sel b{display:block}.box.sel b{background:var(--sel)}
.panel{position:sticky;top:16px;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px;
 max-height:calc(100vh - 32px);overflow:auto}
h1{font-size:15px;margin:0 0 4px}.meta{font-size:12px;opacity:.7;word-break:break-all;margin-bottom:8px}
.prog{height:6px;background:var(--line);border-radius:3px;margin:6px 0 2px}.prog i{display:block;height:100%;background:var(--accent);border-radius:3px}
details{font-size:13px;margin:8px 0}details p{margin:4px 0}
label.q{display:block;font-weight:600;margin:10px 0 4px}
input[type=text]{width:100%;padding:8px 10px;font:inherit;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg)}
.target{margin:8px 0;font-size:13px;min-height:20px}.target b{color:var(--sel)}
.tags,.row{display:flex;flex-wrap:wrap;gap:10px;margin:8px 0;font-size:13px}
button{font:inherit;padding:6px 12px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg);cursor:pointer}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
kbd{font:11px monospace;border:1px solid var(--line);border-radius:3px;padding:0 3px;opacity:.8}
.err{color:var(--sel);min-height:18px;font-size:13px}.saved{font-size:13px;margin-top:8px}.saved li{margin:2px 0}
</style>
<div class="layout">
 <div class="shot"><div class="stage" id="stage"><img id="img" alt="page screenshot"></div></div>
 <div class="panel">
  <h1 id="progress">Loading…</h1><div class="prog"><i id="bar" style="width:0"></i></div>
  <div class="meta" id="meta"></div>
  <details><summary>How to write</summary>
   <p>Write it the way you would tell a friend sitting next to you: <i>"open the pricing page"</i>, <i>"get to my
   saved items"</i>, <i>"the little cart icon"</i>.</p>
   <p>Vary it: some literal (using the words on the button), some vague or goal-like (<i>"I want to change how
   dark the screen is"</i>), some about icons or position (<i>"the magnifier at the top right"</i>).</p>
   <p>Don't read the element's text to copy it, unless that is what you would naturally say. Then click the
   element on the screenshot. Nothing fits? Tick <i>Not on this page</i> and write something sensible that this page
   cannot do.</p>
   <p>Aim for 1&ndash;3 instructions per page; skip pages that are broken or empty.</p></details>
  <label class="q" for="text">What would you tell someone to do on this page?</label>
  <input type="text" id="text" autocomplete="off" autofocus>
  <div class="target" id="target">Click the element on the screenshot.</div>
  <div class="tags" id="tags"></div>
  <div class="row"><label><input type="checkbox" id="absent"> Not on this page <kbd>Alt 0</kbd></label>
   <label><input type="checkbox" id="numbers"> Show numbers <kbd>Alt N</kbd></label>
   <label><input type="checkbox" id="outlines"> Outline all</label></div>
  <div class="err" id="err"></div>
  <div class="row"><button class="primary" id="save">Save <kbd>Enter</kbd></button>
   <button id="next">Next page <kbd>Alt →</kbd></button><button id="skip">Skip <kbd>Alt S</kbd></button>
   <button id="undo">Undo <kbd>Alt Z</kbd></button></div>
  <ul class="saved" id="saved"></ul>
 </div>
</div>
<script>
let S=null, selected=null, hover=null, lastClick=null, cycle=0;
const $=id=>document.getElementById(id), stage=$("stage"), img=$("img");
async function api(path, body){
  const r=await fetch(path,{method:body===undefined?"GET":"POST",body:body===undefined?undefined:JSON.stringify(body)});
  const j=await r.json(); if(!r.ok){$("err").textContent=j.error||"error"; return null;} return j;
}
function boxEl(id){return stage.querySelector('[data-id="'+id+'"]');}
function render(state){
  S=state; selected=null; hover=null; lastClick=null; $("err").textContent=""; $("text").value="";
  $("absent").checked=false; document.querySelectorAll(".tag").forEach(c=>c.checked=false);
  const p=S.progress, pct=Math.min(100,100*p.n/Math.max(1,p.target));
  $("progress").textContent=p.n+" / "+p.target+" written"+(p.n>=p.target?" — target reached":"");
  $("bar").style.width=pct+"%";
  stage.querySelectorAll(".box").forEach(b=>b.remove());
  const TAGTEXT={icon:"refers to an icon",position:"refers to a position",vague:"is vague / goal-like"};
  $("tags").innerHTML=S.tags.map((t,i)=>'<label><input type="checkbox" class="tag" value="'+t+'"> '+TAGTEXT[t]+
    ' <kbd>Alt '+(i+1)+'</kbd></label>').join("");
  if(!S.page){img.removeAttribute("src"); $("meta").textContent="No pages left in this list. Thank you!"; $("saved").innerHTML=""; return;}
  const pg=S.page;
  $("meta").textContent=(pg.title||"")+" — "+pg.url+" ("+pg.elements.length+" elements)";
  img.src=pg.image;
  for(const e of pg.elements){
    const d=document.createElement("div"); d.className="box"; d.dataset.id=e.id;
    const [x,y,w,h]=e.box; d.style.cssText="left:"+100*x+"%;top:"+100*y+"%;width:"+100*w+"%;height:"+100*h+"%";
    d.innerHTML="<b>"+e.id+"</b>"; stage.appendChild(d);
  }
  $("saved").innerHTML=(S.entries||[]).map(e=>"<li>“"+e.query.replace(/</g,"&lt;")+"” → "+(e.target_id===null?"not on page":"#"+e.target_id)+"</li>").join("");
  $("text").focus();
}
function under(ev){
  const pg=S.page, r=stage.getBoundingClientRect(), fx=(ev.clientX-r.left)/r.width, fy=(ev.clientY-r.top)/r.height;
  return pg.elements.filter(e=>fx>=e.box[0]&&fx<=e.box[0]+e.box[2]&&fy>=e.box[1]&&fy<=e.box[1]+e.box[3])
    .sort((a,b)=>a.box[2]*a.box[3]-b.box[2]*b.box[3]);
}
function mark(id,cls){stage.querySelectorAll("."+cls).forEach(b=>b.classList.remove(cls)); if(id!==null){const b=boxEl(id); b&&b.classList.add(cls);}}
stage.addEventListener("mousemove",ev=>{if(!S||!S.page)return; const u=under(ev); hover=u.length?u[0].id:null; mark(hover,"hover");});
stage.addEventListener("mouseleave",()=>mark(null,"hover"));
stage.addEventListener("click",ev=>{
  if(!S||!S.page)return; const u=under(ev); if(!u.length)return;
  const key=Math.round(ev.clientX/6)+","+Math.round(ev.clientY/6);
  cycle=(key===lastClick)?(cycle+1)%u.length:0; lastClick=key;
  selected=u[cycle].id; $("absent").checked=false; mark(selected,"sel");
  $("target").innerHTML="Target: <b>#"+selected+"</b> ("+u[cycle].role+")"+(u.length>1?" — click again for a larger box":"");
  $("text").focus();
});
$("absent").addEventListener("change",()=>{
  if($("absent").checked){selected=null; mark(null,"sel"); $("target").innerHTML="<b>Not on this page</b>";}
});
$("numbers").addEventListener("change",()=>stage.classList.toggle("nums",$("numbers").checked));
$("outlines").addEventListener("change",()=>stage.classList.toggle("show-all",$("outlines").checked));
async function save(){
  if(!S||!S.page)return; const text=$("text").value.trim();
  if(!text){$("err").textContent="Write what you would tell someone first."; return;}
  if(selected===null&&!$("absent").checked){$("err").textContent="Click the element on the screenshot, or tick “Not on this page”."; return;}
  const tags=[...document.querySelectorAll(".tag:checked")].map(c=>c.value);
  const st=await api("/api/save",{page_id:S.page.id,instruction:text,target_id:$("absent").checked?null:selected,tags});
  if(st){ if(st.page&&st.page.id===S.page.id) {keepPage(st);} else render(st); }
}
function keepPage(st){ // same page again: refresh the saved list and progress, keep the screenshot and boxes
  S=Object.assign(S,{entries:st.entries,progress:st.progress});
  selected=null; $("text").value=""; $("absent").checked=false; document.querySelectorAll(".tag").forEach(c=>c.checked=false);
  mark(null,"sel"); $("target").textContent="Saved. Write another, or go to the next page.";
  const p=st.progress; $("progress").textContent=p.n+" / "+p.target+" written"+(p.n>=p.target?" — target reached":"");
  $("bar").style.width=Math.min(100,100*p.n/Math.max(1,p.target))+"%";
  $("saved").innerHTML=st.entries.map(e=>"<li>“"+e.query.replace(/</g,"&lt;")+"” → "+(e.target_id===null?"not on page":"#"+e.target_id)+"</li>").join("");
  $("text").focus();
}
async function act(path){ if(!S||!S.page&&path!=="/api/undo")return; const st=await api(path,{page_id:S.page?S.page.id:""}); if(st)render(st); }
$("save").onclick=save; $("next").onclick=()=>act("/api/next"); $("skip").onclick=()=>act("/api/skip"); $("undo").onclick=()=>act("/api/undo");
document.addEventListener("keydown",ev=>{
  if(ev.key==="Enter"&&!ev.altKey&&!ev.metaKey&&!ev.ctrlKey){ev.preventDefault(); save();}
  else if(ev.key==="Escape"){selected=null; mark(null,"sel"); $("target").textContent="Click the element on the screenshot.";}
  else if(ev.altKey){
    const c=ev.code; let hit=true;
    if(c==="ArrowRight")act("/api/next"); else if(c==="KeyS")act("/api/skip"); else if(c==="KeyZ")act("/api/undo");
    else if(c==="Digit0"){$("absent").checked=!$("absent").checked; $("absent").dispatchEvent(new Event("change"));}
    else if(c==="KeyN"){$("numbers").checked=!$("numbers").checked; $("numbers").dispatchEvent(new Event("change"));}
    else if(c==="Digit1"||c==="Digit2"||c==="Digit3"){const t=document.querySelectorAll(".tag")[+c.slice(-1)-1]; if(t)t.checked=!t.checked;}
    else hit=false;
    if(hit)ev.preventDefault();
  }
});
api("/api/state").then(s=>s&&render(s));
</script>
"""


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pages", required=True, type=Path, help="pages dir, e.g. data/heldout/pages/test")
    parser.add_argument("--assign", required=True, type=Path, help="text file of page ids to label, in order")
    parser.add_argument("--out", required=True, type=Path, help="output JSONL")
    parser.add_argument("--target", type=int, default=150, help="number of entries to write (progress display)")
    parser.add_argument("--annotator", required=True, help="who is writing (stored on every entry)")
    parser.add_argument("--split", help="stored on every entry (default: from the assign file name)")
    parser.add_argument("--per-page", type=int, default=3, help="move on after this many entries on a page")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    session = LabelSession(args.pages, args.assign, args.out, args.target, args.annotator, args.split, args.per_page)
    if session.missing:
        print(f"warning: {len(session.missing)} assigned pages not found under {args.pages}, e.g. {session.missing[:3]}")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(session))
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"{len(session.entries())}/{args.target} entries so far. Open {url}  (Ctrl-C to stop; run again to resume)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\n{len(session.entries())} entries in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
