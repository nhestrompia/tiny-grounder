from __future__ import annotations

from pathlib import Path

import pytest

from grounder.data.pages import page_paths
from grounder.extract import ax
from tests.test_ax_collect import raw_dump
from tools import ax_auto, ax_collect

OTHER_WORDS = "one two three four five six seven eight"


class FakeMac(ax_auto.Mac):
    """Records every action; no subprocess is ever started."""

    def __init__(self, installed=None, running=(), frontmost="Notes", windows=(1, 2), sheets=()) -> None:
        self.installed, self.running, self.front = installed, set(running), frontmost
        self.windows, self.sheets = list(windows), list(sheets)
        self.log: list[tuple[str, str]] = []

    def is_installed(self, app): return self.installed is None or app in self.installed
    def is_running(self, app): return app in self.running
    def launch(self, app): self.log.append(("launch", app))
    def open_url(self, url): self.log.append(("url", url))
    def frontmost(self): return self.front
    def window_count(self, app): return self.windows.pop(0) if len(self.windows) > 1 else self.windows[0]
    def has_sheet(self, app): return self.sheets.pop(0) if self.sheets else False

    def press(self, key):
        ax_auto.check_key(key)
        self.log.append(("press", key))

    def quit(self, app): self.log.append(("quit", app))

    def pressed(self) -> list[str]:
        return [v for k, v in self.log if k == "press"]


def make(tmp_path: Path, dumps, split="train", mac=None, **kw):
    queue = list(dumps)
    lines: list[str] = []

    def dump():
        if not queue:
            raise ax.AxdumpError("no window", 3)
        return queue.pop(0)

    collector = ax_collect.Collector(tmp_path / "pages", split, dump,
                                     screenshot=lambda o, s, p: lines.append("shot"), show=lambda s: None)
    auto = ax_auto.AutoCollector(collector, mac or FakeMac(), sleep=lambda s: None, show=lines.append, **kw)
    return auto, collector, lines


def test_plan_has_launch_then_generic_scenes_and_panes_for_system_settings():
    assert [s.name for s in ax_auto.build_plan("TextEdit")] == [
        "launch", "settings window", "new window", "open panel", "print panel", "find bar"]
    assert "open panel" not in [s.name for s in ax_auto.build_plan("Finder")]
    panes = ax_auto.build_plan("System Settings")
    assert len(panes) == len(ax_auto.PANE_IDS) + 1 >= 30
    assert panes[1].url.startswith("x-apple.systempreferences:com.apple.") and not any(s.keys for s in panes)


def test_new_window_only_where_cmd_n_creates_no_data():
    for app in ("Calendar", "Photos", "Music", "Contacts", "Freeform", "Shortcuts", "Mail", "Notes", "Reminders"):
        assert "new window" not in [s.name for s in ax_auto.build_plan(app)], app
    assert "new window" in [s.name for s in ax_auto.build_plan("Safari")]


def test_default_apps_keep_test_apps_out_of_train():
    assert not set(ax_auto.TEST_APPS) & set(ax_auto.TRAIN_APPS)
    assert ax_auto.select_apps("", "test") == list(ax_auto.TEST_APPS)
    assert ax_auto.select_apps("Notes, Mail", "train") == ["Notes", "Mail"]


def test_only_allowed_keystrokes_are_ever_built():
    assert 'keystroke "," using command down' in ax_auto.key_script("cmd+,")
    assert ax_auto.key_script("escape").endswith("key code 53")
    for bad in ("cmd+q", "cmd+delete", "cmd+backspace", "return", "enter"):
        with pytest.raises(ValueError):
            ax_auto.key_script(bad)
    with pytest.raises(ValueError):
        ax_auto.Scene("bad", keys=("cmd+q",))
    for scene in ax_auto.GENERIC_SCENES:
        assert set(scene.keys + scene.undo) <= ax_auto.ALLOWED_KEYS


def test_scenes_are_dumped_and_undone_and_duplicates_dropped(tmp_path):
    mac = FakeMac(frontmost="TextEdit", windows=(1, 2, 2, 3))
    auto, collector, lines = make(tmp_path, [
        *[raw_dump("TextEdit", **({"words": OTHER_WORDS} if k == 1 else {})) for k in range(6)]], mac=mac)
    auto.run_app("TextEdit")
    assert auto.results["TextEdit"] == {"saved": 2, "duplicate": 4}
    assert mac.pressed()[:4] == ["cmd+,", "cmd+w", "cmd+n", "cmd+w"] and mac.pressed()[-2:] == ["cmd+f", "escape"]
    assert any(l.startswith("[TextEdit] launch -> saved textedit-") and "8 elements" in l for l in lines)
    assert "[TextEdit] settings window -> saved" in "".join(lines) and "[TextEdit] new window -> duplicate" in lines
    assert len(page_paths(tmp_path / "pages" / "train")) == 2


def test_window_is_closed_only_when_one_opened_and_save_sheet_gets_dont_save(tmp_path):
    mac = FakeMac(windows=(1,), sheets=(True, False))  # window count never grows: no cmd+w
    auto, _, _ = make(tmp_path, [raw_dump()], mac=mac)
    auto._run_scene("Notes", ax_auto.GENERIC_SCENES[0])
    assert mac.pressed() == ["cmd+,"]
    mac = FakeMac(windows=(1, 2), sheets=(True, False))
    auto, _, _ = make(tmp_path / "b", [raw_dump()], mac=mac)
    auto._run_scene("Notes", ax_auto.GENERIC_SCENES[1])
    assert mac.pressed() == ["cmd+n", "cmd+w", "cmd+d"]


def test_stubborn_save_sheet_is_cancelled_with_escape(tmp_path):
    mac = FakeMac(windows=(1, 2), sheets=(True, True))
    auto, _, lines = make(tmp_path, [raw_dump()], mac=mac)
    auto._run_scene("Notes", ax_auto.GENERIC_SCENES[1])
    assert mac.pressed() == ["cmd+n", "cmd+w", "cmd+d", "escape"] and "cancelled" in lines[0]


def test_not_installed_and_host_apps_are_skipped_without_launching(tmp_path):
    mac = FakeMac(installed={"Notes", "Terminal"})
    auto, _, lines = make(tmp_path, [], mac=mac)
    auto.run_app("Keynote")
    auto.run_app("Terminal", host="terminal")
    assert mac.log == [] and auto.notes == {"Keynote": "not installed", "Terminal": "hosts this script"}
    assert lines[0] == "[Keynote] skipped: not installed"


def test_wrong_frontmost_app_is_not_saved(tmp_path):
    mac = FakeMac(frontmost="Notes")
    auto, _, lines = make(tmp_path, [raw_dump("Terminal")], mac=mac)
    auto.run_app("Notes")  # the first dump is Terminal; the rest run out of dumps -> errors
    assert auto.results["Notes"]["wrong app"] == 1 and "[Notes] launch -> wrong app (frontmost is Terminal)" in lines
    assert page_paths(tmp_path / "pages") == []


def test_keys_are_not_sent_while_another_app_is_frontmost(tmp_path):
    mac = FakeMac(frontmost="Terminal")
    auto, _, lines = make(tmp_path, [], mac=mac)
    assert auto._run_scene("Notes", ax_auto.GENERIC_SCENES[1]) == "wrong app"
    assert mac.pressed() == []


def test_quit_only_when_we_launched_it_and_never_finder_or_terminal(tmp_path):
    for app, running, quits in (("Notes", False, True), ("Notes", True, False), ("Finder", False, False),
                                ("Terminal", False, False)):
        mac = FakeMac(running={app} if running else (), frontmost=app)
        auto, _, _ = make(tmp_path / f"{app}{running}", [], mac=mac, max_per_app=0)
        auto.collector.dump = lambda: (_ for _ in ()).throw(ax.AxdumpError("no window", 3))
        auto.run_app(app)
        assert (("quit", app) in mac.log) is quits
        assert auto.results[app]["error"] > 0


def test_test_split_refuses_apps_used_in_train_and_takes_screenshots(tmp_path):
    train, _, _ = make(tmp_path, [raw_dump("Notes")])
    train.run_app("Notes", host="x")
    test, _, lines = make(tmp_path, [raw_dump("Pages")], split="test", max_per_app=1)
    test.run_app("Notes")
    assert test.notes["Notes"] == "already in the train split" and "shot" not in lines
    test.run_app("Pages")
    assert "shot" in lines and len(page_paths(tmp_path / "pages" / "test")) == 1


def test_max_per_app_stops_the_walk(tmp_path):
    auto, _, _ = make(tmp_path, [raw_dump(), raw_dump(words=OTHER_WORDS)], max_per_app=1)
    auto.run_app("Notes")
    assert auto.results["Notes"] == {"saved": 1}


def test_permission_error_aborts_and_unknown_errors_are_logged(tmp_path):
    auto, _, _ = make(tmp_path, [])
    auto.collector.dump = lambda: (_ for _ in ()).throw(ax.AxdumpError("needs Accessibility", 2))
    with pytest.raises(ax.AxdumpError):
        auto.run_app("Notes")


def test_dry_run_prints_the_plan_and_touches_nothing(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(ax_auto, "Mac", lambda: pytest.fail("no mac access in a dry run"))
    assert ax_auto._main(["--out", str(tmp_path), "--apps", "Notes", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "Notes" in out and "settings window: cmd+, (undo: cmd+w)" in out and "launch: activate only" in out
    assert not list(tmp_path.iterdir())


def test_summary_table_lists_each_app(tmp_path):
    auto, _, _ = make(tmp_path, [raw_dump()])
    auto.run_app("Notes")
    auto.run_app("Keynote", host="Keynote")
    rows = auto.summary().splitlines()
    assert rows[0].startswith("app") and "saved" in rows[0]
    assert rows[1].startswith("Notes") and rows[2].rstrip().endswith("hosts this script")
