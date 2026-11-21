"""Held-out human-eval set: site filtering, split by site, writer packets and the labelling UI's logic."""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import pytest

from grounder.data.pages import write_page
from grounder.records import Element, Example, Page
from tools import heldout_prepare as prep
from tools import heldout_sites as sites
from tools import label_web
from tools.label import PNG_SIGNATURE


def make_png(width: int, height: int) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    rows = b"".join(b"\x00" + b"\x80" * width for _ in range(height))
    return PNG_SIGNATURE + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


def make_page(page_id: str, site: str = "a.test", n_interactive: int = 12) -> Page:
    els = [Element(id=i, role="link", text=f"Link {i}", context="nav", abs_box=[10 * i, 100, 64, 20], href=f"/{i}")
           for i in range(n_interactive)]
    els.append(Element(id=n_interactive, role="text", text="Just text", abs_box=[0, 0, 10, 10]))
    els.append(Element(id=n_interactive + 1, role="link", text="Link 0", context="footer", abs_box=[0, 900, 64, 20], href="/0"))
    return Page(page_id, f"https://{site}/{page_id}", site, "T", [1280, 800], [1280, 1000], els)


@pytest.fixture
def pages_dir(tmp_path) -> Path:
    root = tmp_path / "test"
    for pid, site in (("p1", "a.test"), ("p2", "a.test"), ("p3", "b.test")):
        path = write_page(make_page(pid, site), root)
        path.with_name(f"{pid}.png").write_bytes(make_png(640, 500))  # half-scale
    return root


# --- site list -----------------------------------------------------------------------------------


def test_filter_candidates_drops_training_demo_blocked_and_duplicate_domains():
    kept, dropped = sites.filter_candidates(
        ["new.org", "www.new.org", "seen.com", "docs.seen.com", "demo.io", "bad.net", "fine.co.uk", "sexshop.com"],
        train={"seen.com"}, demo={"demo.io"}, blocked={"bad.net"})
    assert kept == ["new.org", "fine.co.uk"]
    assert dropped == {"www.new.org": "duplicate domain", "seen.com": "training domain", "docs.seen.com": "training domain",
                       "demo.io": "demo task domain", "bad.net": "blocklist", "sexshop.com": "blocklist"}


def test_task_domains_reads_urls_from_yaml_text():
    text = "- id: a\n  start_url: https://en.wikipedia.org/wiki/X\n- id: b\n  start_url: \"http://numpy.org/doc/\"\n"
    assert sites.task_domains(text) == {"wikipedia.org", "numpy.org"}


def test_shipped_site_list_avoids_training_and_demo_domains():
    root = Path(__file__).resolve().parent.parent
    listed = root / "data/heldout/sites.txt"
    if not listed.exists():
        pytest.skip("no site list")
    from grounder.data.crawl import read_sites, registrable_domain, start_url

    train_file = root / "data/leakage/heldout_domains.txt"
    if not train_file.exists():
        pytest.skip("no training-domain list")
    train = sites.read_domain_file(train_file)
    demo = sites.task_domains((root / "demo/tasks.yaml").read_text())
    domains = [registrable_domain(start_url(e)) for e in read_sites(listed)]
    assert len(domains) == len(set(domains))
    assert not set(domains) & (train | demo)


# --- split and packets ---------------------------------------------------------------------------


def test_split_by_site_never_shares_a_site_and_is_deterministic():
    by_site = {f"s{i}.test": [f"s{i}p{j}" for j in range(1 + i % 3)] for i in range(40)}
    confirm, select = prep.split_by_site(by_site, 30, seed=1)
    assert not set(confirm) & set(select)
    assert set(confirm) | set(select) == set(by_site)
    n = sum(len(by_site[s]) for s in confirm)
    assert abs(n - 30) <= 2
    assert prep.split_by_site(by_site, 30, seed=1) == (confirm, select)
    assert prep.split_by_site(by_site, 30, seed=2)[0] != confirm


def test_interleave_covers_exactly_the_pages_of_the_sites():
    by_site = {"a": ["a1", "a2"], "b": ["b1"], "c": ["c1"]}
    order = prep.interleave(by_site, ["a", "c"], seed=3)
    assert sorted(order) == ["a1", "a2", "c1"]


def test_unusable_reason_needs_screenshot_ten_interactive_elements_and_sane_height():
    assert prep.unusable_reason(make_page("p"), 4000) == ""
    assert prep.unusable_reason(make_page("p"), None) == "no screenshot"
    assert "interactive" in prep.unusable_reason(make_page("p", n_interactive=8), 4000)
    assert "taller" in prep.unusable_reason(make_page("p"), prep.MAX_SHOT_HEIGHT + 1)


def test_leakage_reason_flags_training_domain_url_and_near_duplicates():
    from tools.leakage_audit import TrainIndex, page_info

    def texts_page(pid: str, site: str, words: list[str]) -> Page:
        els = [Element(id=i, role="link", text=w, href="/") for i, w in enumerate(words)]
        return Page(pid, f"https://{site}/{pid}", site, "T", [1280, 800], [1280, 800], els)

    words = [f"word{i}" for i in range(12)]
    index = TrainIndex()
    index.add(page_info("web", texts_page("t1", "train.test", words).to_dict()))
    assert prep.leakage_reason(texts_page("x", "other.test", [f"zz{i}" for i in range(12)]), index) == ""
    assert prep.leakage_reason(texts_page("x", "train.test", ["a"] * 3), index) == "training domain"
    assert prep.leakage_reason(texts_page("t1", "other.test", ["a"] * 3), index) == ""  # same path, other domain
    assert "near-duplicate" in prep.leakage_reason(texts_page("x", "other.test", words + ["extra"]), index)


def test_packet_lists_elements_with_pixel_boxes_of_the_screenshot(pages_dir):
    page = make_page("p1")
    page.elements[0].text = "a|b  c"
    data = prep.packet(page, pages_dir / "a.test" / "p1.png", root=pages_dir.parent)
    assert data["screenshot"] == "test/a.test/p1.png"
    assert data["screenshot_size"] == [640, 500]
    assert data["n_elements"] == len(page.elements) == len(data["elements"])
    first = data["elements"][0].split("|")
    assert first == ["0", "link", "a/b c", "nav", "0,50,32,10"]  # abs_box [0,100,64,20] at half scale
    assert all(len(line.split("|")) == 5 for line in data["elements"])
    json.dumps(data)


# --- labelling UI --------------------------------------------------------------------------------


def session(pages_dir: Path, tmp_path: Path, ids=("p1", "p2", "p3"), per_page=2, name="confirm_pages.txt"):
    assign = tmp_path / name
    assign.write_text("# pages\n" + "\n".join(ids) + "\nnot_there\n")
    return label_web.LabelSession(pages_dir, assign, tmp_path / "out.jsonl", target=5, annotator="tester", per_page=per_page)


def lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_assignment_split_default_and_missing_pages(pages_dir, tmp_path):
    s = session(pages_dir, tmp_path)
    assert s.order == ["p1", "p2", "p3"] and s.missing == ["not_there"]
    assert s.split == "confirm"
    assert session(pages_dir, tmp_path, name="select_pages.txt").split == "select"


def test_save_writes_example_with_equivalents_split_annotator_and_tags(pages_dir, tmp_path):
    s = session(pages_dir, tmp_path)
    s.save("p1", "  go   home ", 0, ["vague", "bogus", "icon"])
    (entry,) = lines(tmp_path / "out.jsonl")
    ex = Example.from_dict(entry)
    assert (ex.page_id, ex.kind, ex.mode, ex.query, ex.target_id) == ("p1", "hand", "do", "go home", 0)
    assert ex.equiv_ids == [0, 13]  # element 13 has the same role, text and href
    assert entry["split"] == "confirm" and entry["annotator"] == "tester" and entry["tags"] == ["icon", "vague"]


def test_save_null_target_and_validation(pages_dir, tmp_path):
    s = session(pages_dir, tmp_path)
    s.save("p1", "log me out", None, [])
    entry = lines(tmp_path / "out.jsonl")[0]
    assert entry["target_id"] is None and entry["equiv_ids"] == []
    with pytest.raises(ValueError):
        s.save("p1", "   ", 0, [])
    with pytest.raises(ValueError):
        s.save("p1", "x", 999, [])
    with pytest.raises(ValueError):
        s.save("nope", "x", 0, [])
    assert len(lines(tmp_path / "out.jsonl")) == 1


def test_pages_advance_after_per_page_entries_next_and_skip(pages_dir, tmp_path):
    s = session(pages_dir, tmp_path)
    assert s.state()["page"]["id"] == "p1"
    s.save("p1", "one", 0, [])
    assert s.state()["page"]["id"] == "p1" and len(s.state()["entries"]) == 1
    s.save("p1", "two", 1, [])  # per_page reached
    assert s.state()["page"]["id"] == "p2"
    s.mark("p2", "skip")
    st = s.state()
    assert st["page"]["id"] == "p3" and st["progress"] == {"n": 2, "target": 5, "pages_left": 1}
    s.mark("p3", "next")
    assert s.state()["page"] is None


def test_resume_does_not_repeat_finished_pages(pages_dir, tmp_path):
    s = session(pages_dir, tmp_path)
    s.save("p1", "one", 0, [])
    s.mark("p1", "next")
    s.mark("p2", "skip")
    again = session(pages_dir, tmp_path)
    assert again.state()["page"]["id"] == "p3"
    assert again.state()["progress"]["n"] == 1
    again.save("p3", "three", 2, [])
    assert len(lines(tmp_path / "out.jsonl")) == 2  # appended, nothing rewritten


def test_undo_removes_last_entry_and_reopens_its_page(pages_dir, tmp_path):
    s = session(pages_dir, tmp_path)
    assert s.undo() is None
    s.save("p1", "one", 0, [])
    s.mark("p1", "next")
    assert s.state()["page"]["id"] == "p2"
    removed = s.undo()
    assert removed["query"] == "one"
    assert lines(tmp_path / "out.jsonl") == []
    assert s.state()["page"]["id"] == "p1"


def test_state_boxes_are_image_fractions(pages_dir, tmp_path):
    page = session(pages_dir, tmp_path).state()["page"]
    assert (page["width"], page["height"]) == (640, 500)
    assert page["elements"][0]["box"] == pytest.approx([0.0, 0.1, 0.05, 0.02])  # [0,100,64,20] on a 1280x1000 doc
    assert page["image"] == "/shot/p1.png"


def test_page_without_screenshot_is_not_served(pages_dir, tmp_path):
    (pages_dir / "a.test" / "p1.png").unlink()
    assert session(pages_dir, tmp_path).state()["page"]["id"] == "p2"


def test_http_handlers_without_a_browser(pages_dir, tmp_path):
    s = session(pages_dir, tmp_path)
    status, ctype, body = s.handle("GET", "/")
    assert status == 200 and ctype.startswith("text/html") and b"What would you tell someone" in body
    status, _, body = s.handle("GET", "/api/state")
    assert status == 200 and json.loads(body)["page"]["id"] == "p1"
    assert s.handle("GET", "/shot/p1.png")[:2] == (200, "image/png")
    assert s.handle("GET", "/shot/zzz.png")[0] == 404
    post = lambda path, obj: s.handle("POST", path, json.dumps(obj).encode())  # noqa: E731
    status, _, body = post("/api/save", {"page_id": "p1", "instruction": "hello", "target_id": 3, "tags": ["position"]})
    assert status == 200 and json.loads(body)["progress"]["n"] == 1
    assert post("/api/save", {"page_id": "p1", "instruction": "", "target_id": 3})[0] == 400
    assert post("/api/save", {"page_id": "p1", "instruction": "x", "target_id": "3"})[0] == 400
    assert s.handle("POST", "/api/save", b"not json")[0] == 400
    assert post("/api/skip", {"page_id": "p1"})[0] == 200
    assert json.loads(s.handle("GET", "/api/state")[2])["page"]["id"] == "p2"
    assert post("/api/undo", {})[0] == 200
    assert lines(tmp_path / "out.jsonl") == []
    assert s.handle("GET", "/nope")[0] == 404
