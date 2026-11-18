from __future__ import annotations

import json
import re
import struct
import zlib
from pathlib import Path

import pytest

from grounder.data.pages import write_page
from grounder.records import Element, Example, Page
from tools import label


def make_png(width: int, height: int) -> bytes:
    """A valid grey PNG of the given size."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    rows = b"".join(b"\x00" + b"\x80" * width for _ in range(height))
    return label.PNG_SIGNATURE + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


def make_page(page_id: str = "p1") -> Page:
    els = [
        Element(id=0, role="link", text="Home", abs_box=[128, 100, 64, 20], href="/"),
        Element(id=1, role="button", text='Buy <now> "x"', context="Shop", abs_box=[640, 1000, 128, 40]),
        Element(id=2, role="link", text="home", abs_box=[0, 0, 10, 10], href="/"),
    ]
    return Page(page_id, "https://a.test/", "a.test", "A <title>", [1280, 800], [1280, 2000], els)


@pytest.fixture
def split(tmp_path) -> Path:
    for page_id in ("p1", "p2"):
        path = write_page(make_page(page_id), tmp_path / "test")
        path.with_name(f"{page_id}.png").write_bytes(make_png(640, 1000))  # half-scale screenshot
    return tmp_path / "test"


def scripted(*answers: str):
    queue = list(answers)
    shown: list[str] = []

    def ask(prompt: str) -> str:
        shown.append(prompt)
        return queue.pop(0)

    return ask, shown


def lines(out: Path) -> list[dict]:
    return [json.loads(line) for line in out.read_text().splitlines()]


def test_png_size_reads_ihdr_and_rejects_other_data():
    assert label.png_size(make_png(7, 5)) == (7, 5)
    with pytest.raises(ValueError):
        label.png_size(b"GIF89a" + bytes(30))


def test_render_scales_boxes_to_the_image_and_escapes_text():
    html = label.render_page_html(make_page(), make_png(640, 1000))
    # scale = 640 / 1280 = 0.5: element 1 is at x 320/640 = 50%, y 500/1000 = 50%, 64/640 = 10% wide, 20/1000 = 2% high
    assert 'left:50.000%;top:50.000%;width:10.000%;height:2.000%' in html
    assert 'left:10.000%;top:5.000%' in html
    assert len(re.findall(r'class="box"', html)) == 3
    assert "<b>0</b>" in html and "<b>1</b>" in html and "<b>2</b>" in html
    assert "Buy &lt;now&gt; &quot;x&quot;" in html and "Buy <now>" not in html
    assert "A &lt;title&gt;" in html and "data:image/png;base64," in html


def test_make_examples_sets_equivalents_and_optional_goal():
    page = make_page()
    only, = label.make_examples(page, "open home", 0)
    assert (only.kind, only.mode, only.target_id, only.equiv_ids) == ("hand", "do", 0, [0, 2])
    do, goal = label.make_examples(page, "buy it", 1, "get a gift")
    assert (goal.mode, goal.query, goal.equiv_ids) == ("goal", "get a gift", [1]) and do.mode == "do"


def test_resume_skips_labelled_pages_and_next_moves_on(split, tmp_path):
    out, opened = tmp_path / "hand.jsonl", []
    ask, shown = scripted("open home", "0", "", "", "click buy", "1", "", "", ":quit")
    labeller = label.Labeller(out, tmp_path / "html", ask, shown.append, opened.append)
    assert labeller.run(split) == 2
    assert [p.name for p in opened] == ["p1.html", "p2.html"] and (tmp_path / "html" / "p1.html").exists()
    assert [(e["page_id"], e["query"], e["target_id"], e["kind"]) for e in lines(out)] == [
        ("p1", "open home", 0, "hand"), ("p2", "click buy", 1, "hand")]
    again, _ = scripted()
    reopened: list[Path] = []
    assert label.Labeller(out, tmp_path / "html", again, lambda s: None, reopened.append).run(split) == 0
    assert reopened == []


def test_invalid_element_numbers_are_asked_again_and_empty_cancels(split, tmp_path):
    out = tmp_path / "hand.jsonl"
    ask, shown = scripted("open home", "9", "abc", "0", "", "", ":quit")
    messages: list[str] = []
    labeller = label.Labeller(out, tmp_path / "html", ask, messages.append, lambda p: None)
    assert labeller.run(split) == 1
    assert sum("is not an element number" in m for m in messages) == 2
    cancel, _ = scripted("open home", "", ":quit")
    out2 = tmp_path / "hand2.jsonl"
    out2.write_text("")
    assert label.Labeller(out2, tmp_path / "html", cancel, lambda s: None, lambda p: None).run(split) == 0
    assert out2.read_text() == ""


def test_undo_removes_the_whole_last_entry(split, tmp_path):
    out = tmp_path / "hand.jsonl"
    ask, _ = scripted("open home", "0", "", "buy it", "1", "gift", ":undo", ":undo", ":undo", ":quit")
    messages: list[str] = []
    labeller = label.Labeller(out, tmp_path / "html", ask, messages.append, lambda p: None)
    assert labeller.run(split) == 0
    assert out.read_text() == "" and "nothing to undo in this session" in messages


def test_pages_without_a_screenshot_are_skipped(tmp_path):
    write_page(make_page("p3"), tmp_path / "test")
    messages: list[str] = []
    labeller = label.Labeller(tmp_path / "o.jsonl", tmp_path / "h", scripted()[0], messages.append, lambda p: None)
    assert labeller.run(tmp_path / "test") == 0
    assert any("no screenshot" in m for m in messages)


def test_output_is_valid_example_jsonl(split, tmp_path):
    out = tmp_path / "hand.jsonl"
    ask, _ = scripted("open home", "0", "", ":quit")
    label.Labeller(out, tmp_path / "h", ask, lambda s: None, lambda p: None).run(split)
    assert Example.from_dict(lines(out)[0]).equiv_ids == [0, 2]
