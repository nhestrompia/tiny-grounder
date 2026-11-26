from __future__ import annotations

import json
from pathlib import Path

from tools import convert_dpublic as dpublic


HEADER = "tag\tname\ttext\tclass\tdescription\tposition (top-left x&y)\tsize (w&h)\n"


def row(tag: str, name: str, text: str, x: int, y: int, w: int, h: int) -> str:
    return f"{tag}\t{name}\t{text}\t\t\t({x}, {y})\t({w}, {h})"


def tree() -> str:
    return HEADER + "\n".join([
        row("push-button", "Save", '""', 10, 10, 60, 30),
        row("entry", "Search", '""', 100, 10, 100, 30),
        row("label", "To", "To", 80, 15, 20, 20),
        row("document-text", "Untitled note", '""', 0, 100, 400, 300),
        row("document-web", "A web page title", '""', 0, 410, 400, 30),
        row("document-web", "Message body", '""', 0, 450, 400, 200),
        row("section", "Static text", "Static text", 10, 700, 100, 20),
    ])


def test_parse_a11y_tsv_reconstructs_multiline_and_tabbed_text() -> None:
    text = HEADER + (
        "section\t\tDear friends,\n"
        "\n"
        "Alice\n"
        "\t\t\t(0, 0)\t(100, 20)\n"
    ) + row("paragraph", "", "Please find t\the file", 0, 30, 100, 20)
    parsed = dpublic.parse_a11y_tsv(text)
    assert parsed.malformed_rows == 0 and len(parsed.rows) == 2
    assert parsed.rows[0].text == "Dear friends,\n\nAlice"
    assert parsed.rows[1].text == "Please find t\the file"
    assert parsed.rows[0].rect == [0.0, 0.0, 100.0, 20.0]


def test_flatten_atspi_maps_roles_and_uses_editable_message_body_heuristic() -> None:
    screen, malformed = dpublic.flatten_atspi(tree(), "p", "thunderbird")
    assert malformed == 0
    assert [(e.tag, e.role, e.text) for e in screen.page.elements] == [
        ("push-button", "button", "Save"),
        ("entry", "textbox", "Search"),
        ("document-text", "textbox", "Untitled note"),
        ("document-web", "textbox", "Message body"),
    ]
    assert screen.page.viewport == [400, 720] and screen.page.site == "thunderbird"
    assert all(e.context.startswith("Thunderbird") for e in screen.page.elements)
    target_id, target_row = dpublic.convert_step(screen, dpublic.TargetAction("click", 20, 20, None))
    assert target_id == 0 and target_row and target_row.tag == "push-button"


def test_action_parser_resolves_arithmetic_and_string_variables_without_execution() -> None:
    parsed = dpublic.parse_action_code(
        "import pyautogui\n"
        "body = 'hello'\n"
        "print('not executed')\n"
        "pyautogui.click(10 + 5, 40 / 2)\n"
        "pyautogui.typewrite(body, interval=0.05)\n"
    )
    target, reason = dpublic.first_target_action(parsed.calls)
    assert reason is None and target is not None
    assert (target.x, target.y, target.typed_text) == (15.0, 20.0, "hello")


def test_action_parser_accepts_moveto_click_idiom() -> None:
    parsed = dpublic.parse_action_code("import pyautogui\npyautogui.moveTo(35, 133, duration=0.2)\npyautogui.click()")
    target, reason = dpublic.first_target_action(parsed.calls)
    assert reason is None and target is not None and (target.x, target.y) == (35.0, 133.0)
    assert dpublic.first_target_action(dpublic.parse_action_code("DONE").calls)[1] == "no_pyautogui_action"


def write_fixture(raw: Path) -> str:
    path = "final_env_b0_honest/pyautogui/screenshot_a11y_tree/o4-mini/thunderbird/task-one/better_log.json"
    steps = [
        {"a11y_tree": tree(), "actions": ["import pyautogui\npyautogui.click(20, 20)"]},
        {"a11y_tree": tree(), "actions": ["import pyautogui\npyautogui.click(120, 20)\npyautogui.write('hello')"]},
        {"a11y_tree": tree(), "actions": ["import pyautogui\npyautogui.click(20, 150)"]},
        {"a11y_tree": tree(), "actions": ["DONE"]},
    ]
    payload = {"task": {"instruction": "Write a note"}, "steps": steps}
    dest = raw / path
    dest.parent.mkdir(parents=True)
    dest.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_convert_logs_builds_goal_examples_history_targets_and_tags(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    path = write_fixture(raw)
    result = dpublic.convert_logs([path], raw, train_apps=set(), max_steps=10)
    assert result.source_steps == 4 and result.trajectories == 1
    assert len(result.examples) == 3 and len(result.pages) == 1
    assert result.dropped == {"no_pyautogui_action": 1}
    assert [ex.target_id for ex in result.examples] == [0, 1, 2]
    assert all(ex.kind == ex.mode == "goal" and ex.query == "Write a note" for ex in result.examples)
    assert result.examples[1].history == ["[push-button]  Save -> CLICK"]
    assert result.examples[2].history == ["[push-button]  Save -> CLICK", "[entry]  Search -> TYPE: hello"]
    assert [ex.tags for ex in result.examples[:2]] == [["goal-only"], ["goal-only"]]
    assert result.apps == {"thunderbird"}


def test_train_app_aliases_cover_chrome_and_vscode() -> None:
    train = {"google-chrome", "chromium-browser", "code", "vscode"}
    assert dpublic.is_train_app("chrome", train)
    assert dpublic.is_train_app("vs_code", train)
    assert not dpublic.is_train_app("thunderbird", train)
    assert not dpublic.is_train_app("libreoffice_writer", train)
