from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools import ax_tree, guirilla


def node(id: str, role: str, pos: tuple[int, int], size: tuple[int, int], name: str = "", children: list | None = None, **kw: Any) -> dict:
    return {"id": id, "name": name, "role": role, "description": kw.pop("description", None), "role_description": kw.pop("role_description", ""),
            "value": kw.pop("value", None), "absolute_position": f"{pos[0]:.2f};{pos[1]:.2f}", "size": f"{size[0]};{size[1]}",
            "enabled": kw.pop("enabled", True), "children": children or []}


def tree() -> dict:
    return node("win", "AXWindow", (100, 200), (800, 600), "Notes", [
        node("toolbar", "AXToolbar", (100, 200), (800, 40), "Main bar", [
            node("new", "AXButton", (110, 205), (30, 30), "New"),
            node("lbl", "AXStaticText", (150, 205), (40, 20), "Folders"),
            node("search", "AXSearchField", (200, 205), (200, 30), "", description="Search"),
        ]),
        node("group", "AXGroup", (100, 240), (800, 300), "", [
            node("tabs", "AXTabGroup", (100, 240), (800, 30), "", [node("tab1", "AXRadioButton", (110, 240), (60, 30), "Inbox")]),
            node("pw", "AXTextField", (110, 300), (100, 20), "Password", value="secret", role_description="secure text field"),
            node("txt", "AXTextField", (110, 330), (100, 20), "", value="typed text"),
            node("off", "AXButton", (2000, 2000), (30, 30), "Offscreen"),
            node("flat", "AXButton", (110, 360), (0, 0), "Flat"),
            node("pic", "AXImage", (110, 400), (50, 50), "Logo"),
        ]),
    ])


def test_convert_tree_follows_axdump_rules() -> None:
    conv = guirilla.convert_tree(tree(), 7, "com.apple.Notes")
    page = conv.page
    assert (page.page_id, page.site, page.url, page.viewport, page.title) == ("guirilla-7", "com-apple-notes", "ax://com.apple.Notes", [800, 600], "Notes")
    roles = [(e.text, e.role, e.tag) for e in page.elements]
    assert roles == [("New", "button", "AXButton"), ("Search", "searchbox", "AXSearchField"), ("Inbox", "tab", "AXRadioButton"),
                     ("Password", "textbox", "AXTextField"), ("typed text", "textbox", "AXTextField")]
    assert conv.node_to_element == {"new": 0, "search": 1, "tab1": 2, "pw": 3, "txt": 4}
    new = page.elements[0]
    assert new.abs_box == [110, 205, 30, 30] and new.box == [0.0125, 0.0083, 0.0375, 0.05] and new.source == "ax"


def test_context_has_window_group_and_sibling_texts() -> None:
    page = guirilla.convert_tree(tree(), 1, "com.apple.Notes").page
    assert page.elements[0].context == "Notes · Main bar · Folders"
    assert page.elements[3].context == "Notes"


def test_secure_value_is_not_used_as_text() -> None:
    secure = node("s", "AXTextField", (110, 300), (100, 20), "", value="secret", role_description="secure text field")
    assert ax_tree.pick_text(secure) == ""
    assert ax_tree.pick_text(node("n", "AXTextField", (0, 0), (1, 1), "", value=3)) == ""


def test_resolve_target_by_id_then_by_point() -> None:
    conv = guirilla.convert_tree(tree(), 1, "com.apple.Notes")

    def target(id: str, pos: tuple[int, int], size: tuple[int, int] = (10, 10), key: str = "id") -> dict:
        n = node(id, "AXStaticText", pos, size)
        return {**n, key: n.pop("id")} if key != "id" else n

    assert guirilla.resolve_target(conv, target("new", (999, 999))) == (0, "id")
    assert guirilla.resolve_target(conv, target("new", (999, 999), key="original_id")) == (0, "id")
    # the image is not kept; the centre of its frame is inside the toolbar button, the smallest element there
    assert guirilla.resolve_target(conv, target("pic", (115, 210))) == (0, "point")
    assert guirilla.resolve_target(conv, None) == (None, "none")
    assert guirilla.resolve_target(conv, target("pic", (5, 5))) == (None, "none")
    assert guirilla.resolve_target(conv, {"id": "pic", "absolute_position": "", "size": ""}) == (None, "none")


def test_smallest_containing_element_wins() -> None:
    outer = node("w", "AXWindow", (0, 0), (500, 500), "W", [
        node("big", "AXTextField", (0, 0), (400, 400), "big"), node("small", "AXButton", (10, 10), (20, 20), "small")])
    page = guirilla.convert_tree(outer, 1, "x.App").page
    assert ax_tree.element_at(page.elements, (15.0, 15.0)) == 1


def row(screen_id: int, target: str | None, point: tuple[int, int] = (0, 0), app: str = "com.apple.Notes", task: str = "Create a note") -> dict:
    """A row whose target node has id `target` and a 10x10 frame at `point`."""
    element = {"id": target, "absolute_position": f"{point[0]};{point[1]}", "size": "10;10"}
    return {"screen_id": screen_id, "app_name": app, "task": task, "action": "left click", "accessibility": json.dumps(tree()),
            "element_data": json.dumps(element) if target else ""}


def test_convert_rows_counts_drops_and_builds_examples() -> None:
    rows = [row(1, "new"), row(2, "missing", (5, 5)), row(3, None), row(4, "pic", (115, 210))]
    data, conv = guirilla.convert_rows(iter(rows))
    assert (conv.rows, len(data.pages), len(data.examples)) == (4, 4, 2)
    assert dict(conv.dropped) == {"no_target": 2}
    assert dict(conv.resolved_by) == {"id": 1, "none": 2, "point": 1}
    ex = data.examples[0]
    assert (ex.page_id, ex.kind, ex.mode, ex.query, ex.target_id, ex.equiv_ids) == ("guirilla-1", "plain", "do", "Create a note", 0, [0])
    assert dict(data.target_roles()) == {"button": 2}


def test_unseen_apps_filter(tmp_path: Path) -> None:
    (tmp_path / "train" / "safari").mkdir(parents=True)
    rows = [row(1, "new", app="com.apple.Safari"), row(2, "new", app="com.apple.Notes")]
    data, _ = guirilla.convert_rows(iter(rows))
    text, unseen = guirilla.unseen_app_report(data, guirilla.train_app_slugs(tmp_path / "train"))
    assert [ex.page_id for ex in unseen] == ["guirilla-2"]
    assert text.startswith("rows from apps with training pages: 1 / 2")
    assert guirilla.app_slug("com.apple.Safari") == "safari"


def other_tree(label: str) -> dict:
    return node("win", "AXWindow", (0, 0), (400, 300), "W", [node("b", "AXButton", (10, 10), (30, 30), label)])


def tree_row(screen_id: int, app: str, label: str) -> dict:
    return {**row(screen_id, "b", (15, 15), app=app), "accessibility": json.dumps(other_tree(label))}


def train_and_test() -> tuple[guirilla.Dataset, guirilla.HeldOut]:
    test, _ = guirilla.convert_rows(iter([tree_row(100, "com.test.Shared", "Test only"), tree_row(101, "com.test.Other", "Twin")]))
    train, _ = guirilla.convert_rows(iter([
        tree_row(1, "com.TEST.shared", "Own"),    # app of the test split (case-insensitive)
        tree_row(2, "com.train.A", "Twin"),       # tree identical to a test screen, own app
        tree_row(3, "com.train.A", "Fine"), tree_row(4, "com.train.B", "Fine too")]))
    return train, guirilla.HeldOut.of(test)


def test_drop_leaks_by_app_and_by_tree() -> None:
    train, held_out = train_and_test()
    kept, dropped = guirilla.drop_leaks(train, held_out, keep_test_apps=False)
    assert sorted(kept.pages) == ["guirilla-3", "guirilla-4"] and [ex.page_id for ex in kept.examples] == ["guirilla-3", "guirilla-4"]
    assert dict(dropped) == {"test_app_screens": 1, "test_app_rows": 1, "test_tree_screens": 1, "test_tree_rows": 1}
    text = guirilla.overlap_text(train, held_out)
    assert text.startswith("overlap with test apps: 1 / 3 bundle ids, 1 / 4 screens, 1 / 4 rows (com.TEST.shared)")
    assert "identical to a test screen: 1, 1 rows" in text


def test_keep_test_apps_still_drops_identical_trees() -> None:
    train, held_out = train_and_test()
    kept, dropped = guirilla.drop_leaks(train, held_out, keep_test_apps=True)
    assert sorted(kept.pages) == ["guirilla-1", "guirilla-3", "guirilla-4"]
    assert dropped["test_app_screens"] == 0 and dropped["test_tree_screens"] == 1


def test_dev_holdout_is_deterministic_and_by_app() -> None:
    rows = [tree_row(i, f"com.app.n{i % 40}", f"L{i}") for i in range(400)]
    data, _ = guirilla.convert_rows(iter(rows))
    train, dev = guirilla.split_dev(data, 0.25)
    again_train, again_dev = guirilla.split_dev(data, 0.25)
    assert sorted(dev.pages) == sorted(again_dev.pages) and sorted(train.pages) == sorted(again_train.pages)
    assert dev.pages and train.pages and len(dev.pages) + len(train.pages) == 400
    assert not dev.apps() & train.apps()
    assert {ex.page_id for ex in dev.examples} == set(dev.pages)
    assert not guirilla.is_dev_app("com.app.x", 0.0) and guirilla.is_dev_app("com.app.x", 1.0)
    assert guirilla.is_dev_app("COM.app.x", 0.5) == guirilla.is_dev_app("com.app.x", 0.5)


def test_element_data_is_json_or_a_python_repr() -> None:
    assert guirilla.parse_element_data('{"id": "a", "enabled": true}')["id"] == "a"
    assert guirilla.parse_element_data("{'id': 'a', 'value': None, 'enabled': True}") == {"id": "a", "value": None, "enabled": True}


def test_split_urls() -> None:
    assert guirilla.URL.format(split="train", k=7, n=96).endswith("/data/train-00007-of-00096.parquet")
    assert guirilla.URL.format(split="test", k=0, n=5).endswith("/data/test-00000-of-00005.parquet")


def test_outputs_load_with_the_eval_readers(tmp_path: Path) -> None:
    from grounder.data.pages import load_examples, load_pages

    data, _ = guirilla.convert_rows(iter([row(1, "new")]))
    guirilla.write_outputs(data, tmp_path, "dev")
    assert list(load_pages(tmp_path / "pages" / "dev")) == ["guirilla-1"]
    assert load_examples(tmp_path / "examples" / "dev.jsonl")[0].target_id == 0
    assert "CC-BY-NC-4.0" in (tmp_path / "LICENSE-NOTE.txt").read_text()


def test_node_without_frame_is_skipped() -> None:
    framed = node("w", "AXWindow", (0, 0), (100, 100), "W", [node("a", "AXButton", (0, 0), (10, 10), "A")])
    framed["children"][0]["size"] = ""
    assert guirilla.convert_tree(framed, 1, "x.App").page.elements == []


def test_placeholder_names_are_empty() -> None:
    assert ax_tree.pick_text(node("n", "AXButton", (0, 0), (1, 1), "empty-name-AXButton")) == ""
    assert ax_tree.pick_text(node("n", "AXButton", (0, 0), (1, 1), "empty-name-AXButton", description="Close")) == "Close"
