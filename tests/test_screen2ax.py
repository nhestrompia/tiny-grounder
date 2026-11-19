from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from grounder.data.pages import load_pages
from tools import screen2ax


def node(role: str, pos: str, size: str, name: str | None = None, children: list | None = None, **kw: Any) -> dict:
    return {"name": name, "role": role, "description": kw.get("description"), "role_description": "", "value": kw.get("value"),
            "position": pos, "size": size, "children": children or []}


def window(title: str, labels: list[str], pos: str = "152.00;38.00") -> dict:
    buttons = [node("AXButton", f"{10 + 40 * i};20", "30;30", label) for i, label in enumerate(labels)]
    return node("AXWindow", pos, "800;600", title, [node("AXGroup", "0;0", "800;600", None, buttons)])


def row(tree: dict) -> dict:
    return {"accessibility": json.dumps(tree)}


LABELS = ["A", "B", "C", "D", "E"]


def test_positions_are_window_relative() -> None:
    page = screen2ax.convert_tree(window("Doc", LABELS), 3)
    assert (page.page_id, page.title, page.viewport) == ("s2ax-3", "Doc", [800, 600])
    first = page.elements[0]
    assert first.abs_box == [162, 58, 30, 30]  # window origin (152, 38) + (10, 20)
    assert first.box == [0.0125, 0.0333, 0.0375, 0.05]  # relative to the window, not to the screen
    assert page.site.startswith("s2ax-") and page.url == f"ax://{page.site}" and page.lang == ""


def test_node_without_a_frame_and_outside_nodes() -> None:
    tree = window("Doc", ["A"])
    tree["children"][0]["children"] += [node("AXButton", "", "", "No frame"), node("AXButton", "5000;5000", "30;30", "Far away")]
    assert [e.text for e in screen2ax.convert_tree(tree, 0).elements] == ["A"]


def test_site_is_a_hash_of_the_window_title() -> None:
    a, b, c = (screen2ax.convert_tree(window(t, LABELS), i) for i, t in enumerate(["Prefs", "Prefs", "Other"]))
    assert a.site == b.site != c.site
    assert screen2ax.convert_tree(window("", LABELS), 0).site.startswith("s2ax-")


def test_dedupe_and_min_elements() -> None:
    rows = [row(window("A", LABELS)), row(window("A again", LABELS, pos="0.00;0.00")), row(window("Small", LABELS[:2])),
            row(window("B", LABELS[:4] + ["Z"]))]
    result = screen2ax.convert_rows(iter(rows))
    assert [p.page_id for p in result.pages] == ["s2ax-0", "s2ax-3"]  # row indexes survive the drops
    assert dict(result.dropped) == {"duplicate": 1, "too_few_elements": 1} and result.rows == 4
    assert len(screen2ax.convert_rows(iter(rows), min_elements=2).pages) == 3


def test_outputs_have_the_pages_ax_layout(tmp_path: Path) -> None:
    result = screen2ax.convert_rows(iter([row(window("Prefs", LABELS))]))
    screen2ax.write_outputs(result.pages, tmp_path)
    page = result.pages[0]
    assert (tmp_path / "train" / page.site / "s2ax-0.json.gz").is_file()
    assert list(load_pages(tmp_path / "train")) == ["s2ax-0"]
    source = (tmp_path / "SOURCE.txt").read_text()
    assert "Apache-2.0" in source and "2507.16704" in source
