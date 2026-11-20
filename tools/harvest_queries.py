"""Real-query set from the agent logs (improvement plan v2.1, step 2).

Pipeline (each stage reads the previous stage's file under data/realq/):

    collect     runs/cli*/logs/*.server.jsonl  -> queries.jsonl     every unique (URL, query) of a `find` call
    extract     queries.jsonl                  -> pages/{site}/*.json.gz, extract_log.jsonl   (headless Chromium)
    candidates  queries.jsonl + pages          -> candidates.jsonl  per-query candidate elements for labelling
    build       candidates.jsonl + labels.jsonl -> examples.jsonl   standard Examples (kind "real" / "null")

Labelling happens in between, with tools/label_realq.py. The clicked element is only ONE candidate: labels are
chosen against the task goal and the page, never copied from the click.

    uv run python tools/harvest_queries.py all          # collect + extract + candidates
    uv run python tools/harvest_queries.py build
"""

from __future__ import annotations

import argparse
import ast
import glob
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grounder.baselines.bm25 import rank_elements  # noqa: E402
from grounder.data.crawl import registrable_domain  # noqa: E402
from grounder.data.pages import load_pages, write_examples, write_page  # noqa: E402
from grounder.records import Element, Example, Page, equiv_ids  # noqa: E402
from tools.leakage_audit import normalize_url  # noqa: E402

RUN_DIRS = ("cli-grounder", "cli-bm25", "cli-weak-grounder", "cli-weak-bm25")
REALQ = ROOT / "data" / "realq"
TOP_BM25 = 8
_URL_LINE = re.compile(r"^url:\s*(\S+)")
_ACTION = re.compile(r"^(clicked|typed into) (\w+) (['\"].*['\"])(?: and pressed Enter)?\s*$")


# ---- parsing the logs ---------------------------------------------------------------------------------------


def norm_query(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().casefold())


def key_of(url: str, query: str) -> str:
    return hashlib.sha1(f"{normalize_url(url)}\0{norm_query(query)}".encode()).hexdigest()[:12]


def parse_rows(text: str) -> list[dict[str, Any]]:
    """The element lines of a find / snapshot result: `id|role|text|context[|score=x]`."""
    rows = []
    for line in text.splitlines():
        parts = line.split("|")
        if len(parts) < 4 or not parts[0].isdigit():
            continue
        score = None
        if parts[-1].startswith("score="):
            score, parts = parts[-1][6:], parts[:-1]
        rows.append({"id": int(parts[0]), "role": parts[1], "text": "|".join(parts[2:-1]), "context": parts[-1], "score": score})
    return rows


def parse_action(result: str) -> tuple[str, str, str] | None:
    """(verb, role, text) of a click/type result line, or None for refusals and errors."""
    m = _ACTION.match((result or "").split("\n")[0])
    if not m:
        return None
    try:
        text = ast.literal_eval(m.group(3))
    except (ValueError, SyntaxError):
        return None
    return ("click" if m.group(1) == "clicked" else "type"), m.group(2), str(text)


def read_run(path: str) -> tuple[list[dict], dict]:
    events, summary = [], {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            ev = json.loads(line)
            if ev.get("type") == "call":
                events.append(ev)
            elif ev.get("type") == "summary":
                summary = ev
    return events, summary


def collect(root: Path = ROOT, run_dirs: Iterable[str] = RUN_DIRS) -> list[dict[str, Any]]:
    """One record per unique (normalised URL, normalised query), with every run it occurred in."""
    uniq: dict[str, dict[str, Any]] = {}
    total = 0
    for run in run_dirs:
        for path in sorted(glob.glob(str(root / "runs" / run / "logs" / "*.server.jsonl"))):
            events, summary = read_run(path)
            listing: list[dict] = []  # rows of the latest find / snapshot: click ids refer to it
            for i, ev in enumerate(events):
                result = ev.get("result") or ""
                if ev["name"] in ("find", "snapshot"):
                    listing = parse_rows(result)
                if ev["name"] != "find":
                    continue
                total += 1
                m = _URL_LINE.match(result)
                if not m or not ev.get("args", {}).get("query"):
                    continue
                url, query = m.group(1), ev["args"]["query"]
                shown = parse_rows(result)
                nxt = next((e for e in events[i + 1:] if e["name"] in ("click", "type", "find", "snapshot", "go_back")), None)
                action = None
                if nxt and nxt["name"] in ("click", "type"):
                    parsed = parse_action(nxt.get("result") or "")
                    if parsed:
                        action = {"verb": parsed[0], "role": parsed[1], "text": parsed[2]}
                    else:  # refused: take the element from the listing the id refers to
                        row = next((r for r in listing if r["id"] == nxt["args"].get("id")), None)
                        if row:
                            action = {"verb": nxt["name"], "role": row["role"], "text": row["text"], "refused": True}
                rec = uniq.setdefault(key_of(url, query), {
                    "key": key_of(url, query), "url": url, "query": query.strip(), "runs": [],
                })
                rec["runs"].append({
                    "run": run, "file": Path(path).name, "task_id": summary.get("task_id", ""),
                    "goal": summary.get("goal", ""), "success": bool(summary.get("success")),
                    "agent": Path(path).name.split("__")[0], "config": summary.get("config", ""),
                    "call": ev.get("n"), "agent_saw": shown, "next_action": action,
                })
    out = sorted(uniq.values(), key=lambda r: (r["runs"][0]["task_id"], r["url"], r["query"]))
    print(f"find calls: {total}; unique (url, query): {len(out)}", file=sys.stderr)
    return out


# ---- extraction ---------------------------------------------------------------------------------------------


def page_id_for(url: str) -> str:
    return "realq-" + hashlib.sha1(normalize_url(url).encode()).hexdigest()[:12]


def extract_pages(queries: list[dict], pages_dir: Path, log_path: Path) -> None:
    """Re-open each distinct URL once in headless Chromium and store the extracted Page; failures are logged."""
    from demo.browser import BrowserSession, ToolError

    done = {json.loads(l)["page_id"]: json.loads(l) for l in log_path.read_text().splitlines()} if log_path.exists() else {}
    urls: dict[str, str] = {}
    for q in queries:
        urls.setdefault(page_id_for(q["url"]), q["url"])
    log = open(log_path, "a", encoding="utf-8")
    for n, (pid, url) in enumerate(urls.items(), 1):
        if pid in done and done[pid]["ok"]:
            continue
        rec: dict[str, Any] = {"page_id": pid, "url": url, "ok": False, "reason": ""}
        try:
            with BrowserSession(url) as session:
                page = session.extract()
                if not page.elements:
                    rec["reason"] = "no elements extracted"
                else:
                    page.page_id, page.site = pid, registrable_domain(url) or "unknown"
                    write_page(page, pages_dir)
                    rec.update(ok=True, elements=len(page.elements), final_url=session.url)
        except ToolError as exc:
            rec["reason"] = f"load failed: {exc}"
        except Exception as exc:  # noqa: BLE001 - record and keep going; one bad page must not stop the run
            rec["reason"] = f"{type(exc).__name__}: {exc}"[:300]
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"[{n}/{len(urls)}] {'ok ' if rec['ok'] else 'FAIL'} {url[:100]} {rec['reason']}", file=sys.stderr)
    log.close()


# ---- candidates ---------------------------------------------------------------------------------------------


def map_rows(elements: list[Element], rows: Iterable[dict], by_context: bool = True) -> list[int]:
    """Ids of the re-extracted elements that match the (role, text[, context]) of rows seen in a log.
    Ids differ between extractions, so the match is by content; ties are all returned."""
    ids: list[int] = []
    for r in rows:
        text = r["text"].casefold()
        exact = [e for e in elements if e.text.casefold() == text and e.role == r["role"]]
        if by_context and len(exact) > 1:
            narrowed = [e for e in exact if e.context.casefold() == r.get("context", "").casefold()]
            exact = narrowed or exact
        if not exact:
            exact = [e for e in elements if e.text.casefold() == text]
        ids += [e.id for e in exact]
    return list(dict.fromkeys(ids))


def build_candidates(rec: dict, page: Page) -> dict[str, Any]:
    els = page.elements
    by_id = {e.id: e for e in els}
    sources: dict[int, list[str]] = defaultdict(list)
    clicked = [r["next_action"] for r in rec["runs"] if r["next_action"]]
    uniq_clicked = {(a["verb"], a["role"], a["text"]) for a in clicked}
    for verb, role, text in sorted(uniq_clicked):
        for i in map_rows(els, [{"role": role, "text": text}]):
            sources[i].append(f"agent-{verb}")
    seen_rows = {(r["role"], r["text"]): r for run in rec["runs"] for r in run["agent_saw"]}
    for i in map_rows(els, seen_rows.values()):
        sources[i].append("agent-saw")
    ranked = rank_elements(rec["query"], els)
    for rank, (i, s) in enumerate(zip(ranked.ids, ranked.scores)):
        if rank < TOP_BM25 and s > 0:
            sources[i].append(f"bm25#{rank + 1}")
    q = norm_query(rec["query"])
    for e in els:
        t = e.text.casefold()
        if t and (t == q or (len(t) >= 4 and t in q) or (len(q) >= 4 and q in t)):
            sources[e.id].append("text-match")
    order = sorted(sources, key=lambda i: (min((int(s[5:]) for s in sources[i] if s.startswith("bm25#")), default=99), i))
    cands = [{"id": i, "role": by_id[i].role, "text": by_id[i].text, "context": by_id[i].context, "sources": sources[i]}
             for i in order]
    return {
        "key": rec["key"], "query": rec["query"], "url": rec["url"], "page_id": page.page_id,
        "n_elements": len(els),
        "goals": sorted({r["goal"] for r in rec["runs"]}),
        "task_ids": sorted({r["task_id"] for r in rec["runs"]}),
        "outcomes": [{"task_id": r["task_id"], "agent": r["agent"], "config": r["config"], "run": r["run"],
                      "success": r["success"]} for r in rec["runs"]],
        "clicked": [{"verb": v, "role": r, "text": t} for v, r, t in sorted(uniq_clicked)],
        "candidates": cands,
    }


def write_candidates(queries: list[dict], pages_dir: Path, out: Path) -> int:
    pages = load_pages(pages_dir) if pages_dir.exists() else {}
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for rec in queries:
            page = pages.get(page_id_for(rec["url"]))
            if page is None:
                continue
            f.write(json.dumps(build_candidates(rec, page), ensure_ascii=False) + "\n")
            n += 1
    return n


# ---- examples -----------------------------------------------------------------------------------------------


def build_examples(cands_path: Path, labels_path: Path, pages_dir: Path, out: Path) -> dict[str, int]:
    pages = load_pages(pages_dir)
    cands = {c["key"]: c for c in map(json.loads, open(cands_path, encoding="utf-8"))}
    labels: dict[str, dict] = {}
    for line in open(labels_path, encoding="utf-8"):
        lab = json.loads(line)
        labels[lab["key"]] = lab  # last label for a key wins
    examples: list[Example] = []
    stats = {"real": 0, "null": 0, "ambiguous": 0, "skipped": 0}
    for key, lab in labels.items():
        c = cands.get(key)
        if c is None or lab.get("skip"):
            stats["skipped"] += 1
            continue
        page = pages[c["page_id"]]
        source = c["task_ids"][0]
        stats["ambiguous"] += bool(lab.get("ambiguous"))
        if lab.get("null") or not lab.get("target_ids"):
            examples.append(Example(page_id=page.page_id, kind="null", mode="do", query=c["query"], source=source))
            stats["null"] += 1
            continue
        ids = [int(i) for i in lab["target_ids"]]
        equiv = sorted({j for i in ids for j in equiv_ids(page.elements, i)} | set(ids))
        examples.append(Example(page_id=page.page_id, kind="real", mode="do", query=c["query"],
                                target_id=ids[0], equiv_ids=equiv, source=source))
        stats["real"] += 1
    examples.sort(key=lambda e: (e.source, e.page_id, e.query))
    write_examples(examples, out)
    return stats


# ---- CLI ----------------------------------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("stage", choices=["collect", "extract", "candidates", "all", "build"])
    p.add_argument("--out", type=Path, default=REALQ)
    a = p.parse_args()
    out: Path = a.out
    out.mkdir(parents=True, exist_ok=True)
    queries_path, pages_dir = out / "queries.jsonl", out / "pages"
    if a.stage in ("collect", "all"):
        qs = collect()
        queries_path.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in qs))
    if a.stage in ("extract", "all"):
        extract_pages(read_jsonl(queries_path), pages_dir, out / "extract_log.jsonl")
    if a.stage in ("candidates", "all"):
        n = write_candidates(read_jsonl(queries_path), pages_dir, out / "candidates.jsonl")
        print(f"candidates: {n} queries with a page", file=sys.stderr)
    if a.stage == "build":
        stats = build_examples(out / "candidates.jsonl", out / "labels.jsonl", pages_dir, out / "examples.jsonl")
        print(json.dumps(stats), file=sys.stderr)


if __name__ == "__main__":
    main()
