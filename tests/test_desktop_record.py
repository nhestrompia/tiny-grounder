from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from grounder.data.pages import load_examples, load_page, page_paths
from grounder.records import Element
from tools import desktop_record as dr

FIXTURES = Path(__file__).parent / "fixtures" / "desktop"


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def script_task() -> dr.DesktopTask:
    target = dr.TargetSpec("textbox", description="Script", nth=1)
    compile_button = dr.TargetSpec("button", description="Compile", nth=1)
    return dr.DesktopTask(
        "script-test", "Script Editor", "Check the syntax of a tiny draft.", ("throwaway-document",),
        (
            dr.Step("click", target),
            dr.Step("type", target, text="set total to 2 + 2"),
            dr.Step("click", compile_button),
        ),
    )


class FakeDriver(dr.SystemEventsDriver):
    def __init__(self, running: bool = False, sheets: tuple[bool, ...] = (), frontmost: str = "Script Editor") -> None:
        self.running, self.sheets, self.front = running, list(sheets), frontmost
        self.log: list[tuple] = []

    def is_running(self, app): return self.running
    def launch(self, app): self.log.append(("launch", app))
    def frontmost(self): return self.front
    def window_count(self, app): return 1
    def has_sheet(self, app): return self.sheets.pop(0) if self.sheets else False
    def press_key(self, key): self.log.append(("key", key))
    def type_text(self, text): self.log.append(("type", text))
    def click_at(self, app, x, y): self.log.append(("click", app, round(x), round(y)))
    def quit(self, app): self.log.append(("quit", app))


def recorder_for(fixture_name: str, driver: FakeDriver) -> dr.DesktopRecorder:
    raw = fixture(fixture_name)
    return dr.DesktopRecorder(driver, lambda app: copy.deepcopy(raw), sleep=lambda seconds: None)


def test_frozen_tasks_validate_and_meet_preregistered_quotas():
    tasks = dr.load_tasks(Path("demo/tasks_desktop.yaml"))
    assert 30 <= len(tasks) <= 40
    assert len({t.app for t in tasks}) >= 6
    goal_only = sum("goal-only" in t.tags for t in tasks) / len(tasks)
    icon = sum("icon" in t.tags for t in tasks) / len(tasks)
    assert goal_only >= 0.30 and icon >= 0.20
    assert all(3 <= len(t.path) <= 10 for t in tasks)


def test_exact_resolution_includes_equivalent_ids_and_nth_selects_one():
    elements = [
        Element(0, "button", "Open"), Element(1, "button", "Open"), Element(2, "button", "Open now"),
    ]
    resolved = dr.resolve_target(elements, dr.TargetSpec("button", label="Open"))
    assert isinstance(resolved, dr.Resolution)
    assert resolved.primary.id == 0 and resolved.target_ids == (0, 1) and resolved.match_kind == "exact"
    second = dr.resolve_target(elements, dr.TargetSpec("button", label="Open", nth=2))
    assert isinstance(second, dr.Resolution) and second.primary.id == 1 and second.target_ids == (0, 1)


def test_substring_resolution_is_conservative_and_never_guesses():
    elements = [Element(0, "button", "Open folder"), Element(1, "button", "Open document")]
    assert dr.resolve_target(elements, dr.TargetSpec("button", label="Open")) == dr.DROP_AMBIGUOUS
    one = dr.resolve_target(elements, dr.TargetSpec("button", label="folder"))
    assert isinstance(one, dr.Resolution) and one.target_ids == (0,)
    assert dr.resolve_target(elements, dr.TargetSpec("link", label="folder")) == dr.DROP_NO_MATCH
    assert dr.resolve_target(elements, dr.TargetSpec("button", label="Open", nth=3)) == dr.DROP_NTH_MISSING


def test_key_chords_are_allowlisted_and_compile_to_system_events():
    assert 'keystroke "a" using {command down}' in dr.key_script("cmd+a")
    assert 'key code 36' in dr.key_script("return")
    assert 'key code 53' in dr.key_script("escape")
    for bad in ("", "cmd+q", "cmd+delete", "delete", "a", "cmd+shift+q"):
        with pytest.raises(ValueError):
            dr.key_parts(bad)


def test_click_script_balances_nested_system_events_tells():
    driver = dr.SystemEventsDriver()
    seen = []
    driver._osascript = lambda *lines: seen.append(lines) or ""
    driver.click_at("Font Book", 10.4, 20.6)
    lines = seen[0]
    assert lines.count("end tell") == 2
    assert 'click at {10, 21}' in lines


def test_type_text_scripts_escape_quotes_and_split_lines_at_explicit_returns():
    scripts = dr.type_text_scripts('set greeting to "hello"\nreturn message & "!"')
    assert scripts == [
        'tell application "System Events" to keystroke "set greeting to \\"hello\\""',
        'tell application "System Events" to key code 36',
        'tell application "System Events" to keystroke "return message & \\"!\\""',
    ]
    assert dr.type_text_scripts("back\\slash") == [
        'tell application "System Events" to keystroke "back\\\\slash"',
    ]
    assert dr.type_text_scripts("one\r\ntwo") == [
        'tell application "System Events" to keystroke "one"',
        'tell application "System Events" to key code 36',
        'tell application "System Events" to keystroke "two"',
    ]


def test_role_only_target_requires_nth_and_selects_that_control():
    with pytest.raises(ValueError):
        dr.parse_target({"role": "textbox"}, "task")
    target = dr.parse_target({"role": "textbox", "nth": 2}, "task")
    elements = [Element(0, "textbox", "first"), Element(1, "textbox", "second")]
    resolved = dr.resolve_target(elements, target)
    assert isinstance(resolved, dr.Resolution)
    assert resolved.primary.id == 1 and resolved.match_kind == "role"


def test_recording_writes_compatible_examples_pages_report_and_manifest(tmp_path):
    driver = FakeDriver(sheets=(True, False))
    recorder = recorder_for("script_editor_window.json", driver)
    task = script_task()
    result, manifest_path = dr.record_and_write([task], recorder, tmp_path, "d_test", Path("demo/tasks_desktop.yaml"))

    assert result.counts["examples"] == 3 and result.counts["pages"] == 1
    assert not result.drops and result.tag_counts["goal-only"] == 2 and result.tag_counts["icon"] == 1 and result.tag_counts["literal"] == 1
    examples_path = tmp_path / "d_test.jsonl"
    rows = [json.loads(line) for line in examples_path.read_text().splitlines()]
    assert rows[0]["tags"] == ["goal-only"] and rows[2]["tags"] == ["goal-only", "icon"]
    assert rows[0]["history"] == [] and rows[1]["history"] == ["[AXTextArea]  Script -> CLICK"]
    assert rows[1]["history"] == ["[AXTextArea]  Script -> CLICK"]
    assert rows[2]["history"][-1] == "[AXTextArea]  Script -> TYPE: set total to 2 + 2"
    # The exact loader tools/next_step_recall.py uses round-trips the automatic tags.
    loaded = load_examples(examples_path)
    assert [e.target_id for e in loaded] == [2, 2, 1]
    assert loaded[2].tags == ["goal-only", "icon"]

    [page_path] = page_paths(tmp_path / "pages_d_test")
    page = load_page(page_path)
    assert page.site == "script-editor" and len(page.elements) == 3
    assert ("type", "set total to 2 + 2") in driver.log
    assert ("click", "Script Editor", 552, 402) in driver.log  # centre of the text area fixture
    assert ("key", "cmd+w") in driver.log and ("key", "cmd+d") in driver.log and ("quit", "Script Editor") in driver.log

    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "recorded" and manifest["tasks_file"]["tasks"] == 1
    paths = {entry["path"] for entry in manifest["outputs"]}
    assert any(p.endswith("d_test.jsonl") for p in paths) and any(p.endswith(".json.gz") for p in paths)
    assert all(len(entry["sha256"]) == 64 for entry in manifest["outputs"])


def test_unresolvable_step_is_dropped_and_later_steps_are_not_attempted(tmp_path):
    driver = FakeDriver(frontmost="Font Book")
    recorder = recorder_for("font_book_window.json", driver)
    missing = dr.TargetSpec("button", label="No such control")
    task = dr.DesktopTask(
        "font-test", "Font Book", "Try a missing control.", (),
        (dr.Step("click", missing), dr.Step("click", dr.TargetSpec("searchbox", label="Search")),
         dr.Step("click", dr.TargetSpec("other", label="Courier New"))),
    )
    result = recorder.record_task(task)
    assert not result.examples
    assert [(d.reason, d.step_index) for d in result.drops] == [(dr.DROP_NO_MATCH, 0)]
    assert result.not_attempted == 2
    assert not [entry for entry in driver.log if entry[0] == "click"]


def test_wrong_app_dump_is_a_safety_drop_before_any_action():
    driver = FakeDriver()
    raw = fixture("font_book_window.json") | {"app": "Finder"}
    recorder = dr.DesktopRecorder(driver, lambda app: copy.deepcopy(raw), sleep=lambda seconds: None)
    result = recorder.record_task(script_task())
    assert result.drops[0].reason == dr.DROP_WRONG_APP
    assert result.not_attempted == 2 and not result.examples
    assert not [entry for entry in driver.log if entry[0] in {"click", "type"}]
    assert [entry for entry in driver.log if entry[0] == "key"] == [("key", "cmd+w")]  # cleanup only


def test_frontmost_mismatch_prevents_prepare_keys_and_input():
    driver = FakeDriver(frontmost="Finder")
    recorder = recorder_for("script_editor_window.json", driver)
    result = recorder.record_task(script_task())
    assert result.drops[0].reason == dr.DROP_WRONG_APP
    assert not [entry for entry in driver.log if entry[0] in {"click", "type", "key"}]


def test_old_axdump_without_label_provenance_drops_instead_of_guessing_tags():
    driver = FakeDriver()
    raw = fixture("script_editor_window.json")
    for element in raw["elements"]:
        element.pop("visible_text")
        element.pop("nonvisual_text")
    recorder = dr.DesktopRecorder(driver, lambda app: copy.deepcopy(raw), sleep=lambda seconds: None)
    result = recorder.record_task(script_task())
    assert result.drops[0].reason == dr.DROP_LABEL_PROVENANCE
    assert not result.examples and not [entry for entry in driver.log if entry[0] in {"click", "type"}]
