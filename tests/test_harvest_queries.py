import json
from pathlib import Path

from grounder.data.pages import load_examples, load_pages_from, write_page
from grounder.records import Element, Page
from tools import harvest_queries as hq
from tools import label_realq as lr


def el(i: int, role: str, text: str, ctx: str = "", href: str = "") -> Element:
    return Element(id=i, role=role, text=text, context=ctx, href=href)


def make_page(pid: str = "realq-aaa") -> Page:
    els = [el(0, "link", "Home"), el(1, "searchbox", "Search the docs"), el(2, "link", "Pods"), el(3, "link", "Pods"),
           el(4, "button", "Submit")]
    return Page(page_id=pid, url="https://k8s.io/docs/", site="k8s.io", title="t", viewport=[1280, 800], doc_size=[1280, 800], elements=els)


def test_parse_rows_and_actions() -> None:
    rows = hq.parse_rows("url: u\ntitle: t\n5 elements shown (id|role|text|context):\n21|link|Visit the Wayback Machine||score=3.098\n"
                         "4|button|A | B|ctx here|score=1.0\n7|link|Plain|ctx")
    assert [r["id"] for r in rows] == [21, 4, 7]
    assert rows[1]["text"] == "A | B" and rows[1]["context"] == "ctx here" and rows[1]["score"] == "1.0"
    assert hq.parse_action("clicked link 'Visit'\nurl: x") == ("click", "link", "Visit")
    assert hq.parse_action("clicked link \"What's new\"") == ("click", "link", "What's new")
    assert hq.parse_action("typed into searchbox 'Quick search' and pressed Enter") == ("type", "searchbox", "Quick search")
    assert hq.parse_action("refused: the element at that point is not a <a>") is None


def test_collect_dedups_by_normalised_url_and_query(tmp_path: Path) -> None:
    logs = tmp_path / "runs" / "r1" / "logs"
    logs.mkdir(parents=True)

    def run(name: str, query: str, url: str, success: bool) -> None:
        ev = [
            {"type": "call", "n": 1, "name": "find", "args": {"query": query},
             "result": f"url: {url}\ntitle: T\n1|link|Pods||score=2.0\n2|link|Home||score=0.0"},
            {"type": "call", "n": 2, "name": "click", "args": {"id": 1}, "result": "clicked link 'Pods'\nurl: x"},
            {"type": "call", "n": 3, "name": "find", "args": {"query": "no url"}, "result": "error"},
            {"type": "summary", "task_id": "t1", "goal": "Open pods", "success": success, "config": "find"},
        ]
        (logs / f"{name}.server.jsonl").write_text("\n".join(json.dumps(e) for e in ev))

    run("a", "Pods link", "https://www.k8s.io/docs/", True)
    run("b", "pods  LINK", "http://k8s.io/docs", False)
    out = hq.collect(tmp_path, ["r1"])
    assert len(out) == 1
    rec = out[0]
    assert [r["success"] for r in rec["runs"]] == [True, False]
    assert rec["runs"][0]["next_action"] == {"verb": "click", "role": "link", "text": "Pods"}
    assert rec["runs"][0]["goal"] == "Open pods"


def test_candidates_map_by_content_not_id() -> None:
    page = make_page()
    rec = {"key": "k", "url": page.url, "query": "pods link", "runs": [{
        "run": "r", "file": "f", "task_id": "t1", "goal": "g", "success": True, "agent": "a", "config": "find", "call": 1,
        "agent_saw": [{"id": 99, "role": "link", "text": "Pods", "context": "", "score": "1"}],
        "next_action": {"verb": "click", "role": "link", "text": "Pods"}}]}
    c = hq.build_candidates(rec, page)
    ids = {x["id"]: x["sources"] for x in c["candidates"]}
    assert {2, 3} <= set(ids) and "agent-click" in ids[2] and "agent-saw" in ids[3]
    assert any(s.startswith("bm25#") for s in ids[2])


def test_parse_spec() -> None:
    assert lr.parse_spec("3, 5 ? # two ok") == ([3, 5], False, True, False, "two ok")
    assert lr.parse_spec("none") == ([], True, False, False, "")
    assert lr.parse_spec("skip")[3] is True


def test_apply_and_build_examples(tmp_path: Path) -> None:
    page = make_page()
    pages_dir = tmp_path / "pages"
    write_page(page, pages_dir)
    cands = [
        {"key": "k1", "query": "pods link", "url": page.url, "page_id": page.page_id, "n_elements": 5, "goals": ["g"],
         "task_ids": ["t1"], "outcomes": [], "clicked": [],
         "candidates": [{"id": i, "role": "link", "text": "x", "context": "", "sources": []} for i in range(5)]},
        {"key": "k2", "query": "missing thing", "url": page.url, "page_id": page.page_id, "n_elements": 5, "goals": ["g"],
         "task_ids": ["t2"], "outcomes": [], "clicked": [],
         "candidates": [{"id": 0, "role": "link", "text": "x", "context": "", "sources": []}]},
        {"key": "k3", "query": "skipped", "url": page.url, "page_id": page.page_id, "n_elements": 5, "goals": ["g"],
         "task_ids": ["t3"], "outcomes": [], "clicked": [],
         "candidates": [{"id": 0, "role": "link", "text": "x", "context": "", "sources": []}]},
    ]
    cpath = tmp_path / "candidates.jsonl"
    cpath.write_text("".join(json.dumps(c) + "\n" for c in cands))
    spec = tmp_path / "spec.txt"
    spec.write_text("0: 2 ?\n1: none # not here\n2: skip\n")
    assert lr.apply(spec, lr.load_candidates(cpath), tmp_path / "labels.jsonl", "test") == 3
    stats = hq.build_examples(cpath, tmp_path / "labels.jsonl", pages_dir, tmp_path / "examples.jsonl")
    assert stats == {"real": 1, "null": 1, "ambiguous": 1, "skipped": 1}
    exs = load_examples_sorted(tmp_path / "examples.jsonl")
    real = next(e for e in exs if e.kind == "real")
    assert real.target_id == 2 and real.equiv_ids == [2, 3] and real.source == "t1"  # "Pods" twice -> equivalents
    assert next(e for e in exs if e.kind == "null").target_id is None
    assert set(load_pages_from([pages_dir])) == {page.page_id}


def load_examples_sorted(path: Path):
    return load_examples(path)
