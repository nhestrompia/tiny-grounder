from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from grounder.data import mind2web as m
from grounder.data.pages import load_examples, load_pages

HTML = """<html backend_node_id="1"><body backend_node_id="2">
<section backend_node_id="3"><h2 backend_node_id="4"><text backend_node_id="5">Your trip</text></h2>
<ul backend_node_id="6">
<li backend_node_id="7"><text backend_node_id="8">Blue shirt $24</text>
<button backend_node_id="9"><text backend_node_id="10">Add to cart</text></button></li>
<li backend_node_id="11"><text backend_node_id="12">Red shirt $30</text>
<button backend_node_id="13"><text backend_node_id="14">Add to cart</text></button></li></ul>
<input backend_node_id="15" type="search" placeholder="Find a shirt" value=""/>
<a backend_node_id="16" aria_label="Home page"/>
<select backend_node_id="17"><option backend_node_id="18" option_selected="true"><text backend_node_id="19">Large</text></option></select>
</section></body></html>"""


def cand(nid: int, tag: str, box: str, **attrs):
    a = {"backend_node_id": str(nid), "bounding_box_rect": box, "class": "x", **attrs}
    return {"tag": tag, "attributes": json.dumps(a), "backend_node_id": str(nid)}


def row(task="Buy a blue shirt", n_actions=2):
    actions = []
    for i in range(n_actions):
        pos = [cand(13, "button", "100,900,80,30"), cand(9, "button", "100,100,80,30")]
        neg = [cand(15, "input", "10,10,200,30", type="search", placeholder="Find a shirt"),
               cand(16, "a", "300,10,40,30", aria_label="Home page"),
               cand(17, "select", "400,10,40,30"), cand(3, "section", "0,0,1280,2000")]
        actions.append({"action_uid": f"{task}-u{i}", "cleaned_html": HTML, "operation": {"op": "CLICK"},
                        "pos_candidates": pos, "neg_candidates": neg})
    return {"website": "shop", "domain": "Shopping", "subdomain": "x", "annotation_id": "a1",
            "confirmed_task": task, "action_reprs": [f"[button]  step {i} -> CLICK" for i in range(n_actions)],
            "actions": actions}


def test_roles_follow_tag_type_and_role_attribute():
    assert m.role_of("a", {}) == "link" and m.role_of("select", {}) == "combobox"
    assert m.role_of("input", {"type": "search"}) == "searchbox" and m.role_of("input", {"type": "submit"}) == "button"
    assert m.role_of("div", {"role": "tab"}) == "tab" and m.role_of("div", {"role": "weird"}) == "other"
    assert m.role_of("h3", {}) == "text" and m.role_of("span", {"is_clickable": "true"}) == "other"
    assert m.role_of("div", {}) == "other" and m.role_of("svg", {}) == "image"


def test_step_converts_text_context_box_target_and_history():
    page, ex = m.convert_action(row(), 1)
    by_text = {(e.text, e.tag): e for e in page.elements}
    ids = [e.id for e in page.elements]
    assert ids == list(range(len(ids)))
    red = next(e for e in page.elements if e.text == "Add to cart" and e.abs_box[1] == 900)
    assert red.context == "Red shirt $30 · Your trip" and red.role == "button"
    assert red.box == [0.0781, 1.25, 0.0625, 0.0417] and not red.in_viewport
    assert by_text[("Find a shirt", "input")].role == "searchbox"
    assert by_text[("Home page", "a")].role == "link"
    assert by_text[("Large", "select")].role == "combobox"
    assert ex.kind == ex.mode == "goal" and ex.query == "Buy a blue shirt" and ex.history == ["[button]  step 0 -> CLICK"]
    assert ex.target_id == red.id and sorted(ex.equiv_ids) == sorted(
        e.id for e in page.elements if e.text == "Add to cart")
    assert page.site == "shop" and page.viewport == [1280, 720] and page.doc_size == [1280, 2000]


def test_history_keeps_the_last_three_and_steps_without_positives_are_skipped():
    r = row(n_actions=5)
    assert m.convert_action(r, 4)[1].history == ["[button]  step 1 -> CLICK", "[button]  step 2 -> CLICK", "[button]  step 3 -> CLICK"]
    r["actions"][2]["pos_candidates"] = []
    assert m.convert_action(r, 2) is None and len(list(m.convert_row(r))) == 4


def test_convert_writes_pages_and_examples_and_read_rows_accepts_all_layouts(tmp_path):
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "a.json").write_text(json.dumps([row("t1")]))
    (tmp_path / "in" / "b.jsonl").write_text(json.dumps(row("t2")) + "\n")
    assert [r["confirmed_task"] for r in m.read_rows(tmp_path / "in")] == ["t1", "t2"]
    stats = m.convert(m.read_rows(tmp_path / "in"), tmp_path / "out", "test_task")
    assert stats == {"tasks": 2, "steps": 4, "elements": 24, "skipped_steps": 0}
    pages = load_pages(tmp_path / "out" / "pages" / "test_task")
    examples = load_examples(tmp_path / "out" / "test_task.jsonl")
    assert len(pages) == len(examples) == 4 and all(e.page_id in pages for e in examples)


def test_cli_requires_input_for_test_splits(tmp_path):
    with pytest.raises(SystemExit):
        m._main(["--split", "test_task", "--out", str(tmp_path)])


class FakeHub:
    """A local stand-in for the hub: file listing plus resolve URLs that honour Range."""

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files, self.requests, self.ignore_range = files, [], False
        hub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                hub.requests.append((self.path, self.headers.get("Range")))
                if self.path.startswith("/api/datasets/osunlp/Mind2Web/tree/main/data/train"):
                    body = json.dumps([{"type": "file", "path": p, "size": len(b)} for p, b in hub.files.items()]
                                      + [{"type": "file", "path": "data/train/readme.md", "size": 1}]).encode()
                    return self.reply(200, body)
                data = hub.files["data/train/" + self.path.rsplit("/", 1)[1]]
                if (rng := self.headers.get("Range")) and not hub.ignore_range:
                    start = int(rng.removeprefix("bytes=").split("-")[0])
                    return self.reply(206, data[start:])
                return self.reply(200, data)

            def reply(self, code, body):
                self.send_response(code)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


@pytest.fixture
def hub():
    files = {
        "data/train/train_10.json": json.dumps([row("t10a"), row("t10b")]).encode(),
        "data/train/train_2.json": json.dumps([row("t2a")]).encode(),
    }
    fake = FakeHub(files)
    yield fake
    fake.close()


def test_list_train_files_sorts_numerically_and_ignores_other_files(hub):
    assert [p for p, _ in m.list_train_files(hub.url)] == ["data/train/train_2.json", "data/train/train_10.json"]


def test_download_caches_and_resumes_a_partial_file_with_range(hub, tmp_path):
    path, size = "data/train/train_10.json", len(hub.files["data/train/train_10.json"])
    tmp_path.joinpath("train_10.json.part").write_bytes(hub.files[path][:100])
    dest = m.download(path, size, tmp_path, hub.url)
    assert dest.read_bytes() == hub.files[path] and not (tmp_path / "train_10.json.part").exists()
    assert hub.requests[-1][1] == "bytes=100-"
    before = len(hub.requests)
    assert m.download(path, size, tmp_path, hub.url) == dest and len(hub.requests) == before  # complete: no request


def test_download_restarts_when_the_server_ignores_range(hub, tmp_path):
    path = "data/train/train_2.json"
    tmp_path.joinpath("train_2.json.part").write_bytes(b"garbage")
    hub.ignore_range = True
    dest = m.download(path, len(hub.files[path]), tmp_path, hub.url)
    assert dest.read_bytes() == hub.files[path]


def test_fetch_train_rows_downloads_all_files_or_streams_a_limit(hub, tmp_path):
    rows = list(m.fetch_train_rows(tmp_path / "cache", hub=hub.url))
    assert [r["confirmed_task"] for r in rows] == ["t2a", "t10a", "t10b"]
    assert sorted(p.name for p in (tmp_path / "cache").iterdir()) == ["train_10.json", "train_2.json"]
    limited = list(m.fetch_train_rows(tmp_path / "other", limit=2, hub=hub.url))
    assert [r["confirmed_task"] for r in limited] == ["t2a", "t10a"] and not (tmp_path / "other").exists()
    only = list(m.fetch_train_rows(tmp_path / "c3", only=["train_10.json"], hub=hub.url))
    assert len(only) == 2
