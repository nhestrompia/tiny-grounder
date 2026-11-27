from __future__ import annotations

from pathlib import Path

import pytest

from grounder.data.pages import load_page, page_paths
from grounder.data.teacher import pick_pages
from grounder.extract import ax
from tools import ax_collect, label
from tests.test_label_tool import make_png


def raw_dump(app: str = "Notes", words: str = "alpha beta gamma delta epsilon zeta eta theta", **extra) -> dict:
    elements = [
        {"id": i, "role": "button", "text": f"{w} button", "context": "Notes window", "tag": "AXButton",
         "box": [0.1 * (i % 9), 0.1, 0.05, 0.05], "abs_box": [110 + 80 * i, 220, 40, 30], "in_viewport": True,
         "enabled": True, "conf": 1.0, "source": "ax", "href": ""}
        for i, w in enumerate(words.split())
    ]
    return {"viewport": [800, 600], "doc_size": [800, 600], "title": "Window", "app": app,
            "origin": [100.0, 200.0], "elements": elements, **extra}


class Session:
    """A collector whose axdump returns queued dumps and whose screenshots are recorded."""

    def __init__(self, tmp_path: Path, split: str = "train", **kw) -> None:
        self.dumps: list = []
        self.shots: list = []
        self.lines: list[str] = []
        self.out = tmp_path / "pages_ax"
        self.collector = ax_collect.Collector(
            self.out, split, self._dump, screenshot=lambda o, s, p: self.shots.append((o, s, p)),
            show=self.lines.append, **kw)

    def _dump(self) -> dict:
        item = self.dumps.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def run(self, *dumps) -> list[bool]:
        self.dumps.extend(dumps)
        return [self.collector.collect_once() for _ in dumps]


def test_page_is_written_in_the_crawl_layout_with_a_stable_id(tmp_path):
    s = Session(tmp_path)
    s.run(raw_dump("Visual Studio Code"))
    [path] = page_paths(s.out / "train")
    page = load_page(path)
    assert path.parent.name == "visual-studio-code" and page.site == "visual-studio-code"
    assert page.page_id.startswith("visual-studio-code-") and path.name == f"{page.page_id}.json.gz"
    assert page.url == "ax://visual-studio-code" and len(page.elements) == 8
    again = Session(tmp_path / "other")
    again.run(raw_dump("Visual Studio Code"))
    assert load_page(page_paths(again.out / "train")[0]).page_id == page.page_id
    assert "saved train/visual-studio-code/" in s.lines[0] and "button 8" in s.lines[0]


def test_near_duplicates_are_skipped_but_other_windows_and_apps_are_kept(tmp_path):
    s = Session(tmp_path)
    near = raw_dump(words="alpha beta gamma delta epsilon zeta eta theta")  # identical text
    other_window = raw_dump(words="one two three four five six seven eight")
    other_app = raw_dump("Mail")
    s.run(raw_dump(), near, other_window, other_app)
    assert len(page_paths(s.out / "train")) == 3
    assert "near-duplicate" in s.lines[1]
    assert s.collector.saved == {"Notes": 2, "Mail": 1} and s.collector.skipped == {"Notes": 1}


def test_duplicates_are_detected_against_pages_saved_in_earlier_sessions(tmp_path):
    first = Session(tmp_path)
    first.run(raw_dump())
    second = Session(tmp_path)
    second.run(raw_dump())
    assert "near-duplicate" in second.lines[0] and len(page_paths(first.out / "train")) == 1


def test_too_few_elements_is_skipped(tmp_path):
    s = Session(tmp_path)
    s.run(raw_dump(words="a b c d"))
    assert "fewer than 5 elements" in s.lines[0] and page_paths(s.out) == []
    strict = Session(tmp_path / "x", min_elements=9)
    strict.run(raw_dump())
    assert page_paths(strict.out) == []


def test_an_app_never_lands_in_both_splits(tmp_path):
    train = Session(tmp_path, "train")
    train.run(raw_dump())
    test = Session(tmp_path, "test")
    test.run(raw_dump(words="one two three four five six seven eight"), raw_dump("Mail"))
    assert "already has pages in train" in test.lines[0]
    assert [p.parent.name for p in page_paths(test.out / "test")] == ["mail"]
    assert test.collector.skipped == {"Notes": 1}


def test_test_split_takes_a_screenshot_beside_the_page_and_train_does_not(tmp_path):
    test = Session(tmp_path, "test")
    test.run(raw_dump())
    [(origin, size, png)] = test.shots
    [page_path] = page_paths(test.out / "test")
    assert origin == [100.0, 200.0] and size == [800, 600]
    assert png == label.screenshot_path(page_path)
    train = Session(tmp_path / "t", "train")
    train.run(raw_dump())
    assert train.shots == []


def test_screenshot_command_uses_window_points():
    cmd = ax_collect.screenshot_command([100.4, -20.0], [800, 600], Path("/x/p.png"))
    assert cmd == ["screencapture", "-x", "-R", "100,-20,800,600", "/x/p.png"]


def test_screenshot_failure_warns_and_keeps_the_page(tmp_path):
    def broken(origin, size, path):
        raise RuntimeError("not permitted")

    s = Session(tmp_path, "test")
    s.collector.screenshot = broken
    s.run(raw_dump())
    assert len(page_paths(s.out / "test")) == 1 and "WARNING: no screenshot" in s.lines[0]


def test_label_tool_draws_ax_boxes_from_the_window_relative_box(tmp_path):
    page = ax.page_from_dump(raw_dump(), "p", "notes")
    html = label.render_page_html(page, make_png(1600, 1200))  # a 2x capture of the 800x600 window
    first = page.elements[0].box
    assert f"left:{first[0] * 100:.3f}%;top:{first[1] * 100:.3f}%" in html


def test_permission_error_stops_the_session_and_other_errors_continue(tmp_path):
    s = Session(tmp_path)
    assert s.run(ax.AxdumpError("no window", 3)) == [True]
    assert s.run(ax.AxdumpError("needs the Accessibility permission", 2)) == [False]
    assert "axdump failed: needs the Accessibility permission" in s.lines[-1]


def test_run_session_quits_on_q_and_counts_down_before_dumping(tmp_path):
    s = Session(tmp_path)
    s.dumps.append(raw_dump())
    answers = iter(["", "q"])
    waits: list[int] = []
    ax_collect.run_session(s.collector, 3, ask=lambda _: next(answers), wait=waits.append)
    assert waits == [3] and s.collector.saved == {"Notes": 1}
    assert s.collector.tally() == "Notes: 1 saved, 0 skipped"


def test_stats_and_pick_pages_read_the_collected_directory(tmp_path):
    s = Session(tmp_path)
    s.run(raw_dump(), raw_dump("Mail"))
    test = Session(tmp_path, "test")
    test.run(raw_dump("Finder"))
    stats = ax_collect.split_stats(s.out)
    assert stats == {("train", "notes"): 1, ("train", "mail"): 1, ("test", "finder"): 1}
    text = ax_collect.format_stats(stats)
    assert "finder" in text and "test" in text and "3 apps, 3 pages" in text
    pages = pick_pages([s.out / "train"], None, 0, 0.0)
    assert sorted(p.site for p in pages) == ["mail", "notes"]


def test_main_without_the_binary_names_the_build_command(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ax, "AXDUMP", tmp_path / "a" / "b" / "c" / "axdump")
    assert ax_collect._main(["--out", str(tmp_path), "--split", "train"]) == 2
    assert "swift build -c release" in capsys.readouterr().err


def test_main_stats_does_not_need_a_split(tmp_path, capsys):
    assert ax_collect._main(["--out", str(tmp_path), "--stats"]) == 0
    assert "0 apps, 0 pages" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        ax_collect._main(["--out", str(tmp_path)])


def test_find_window_id_matches_bounds_and_skips_overlays():
    windows = [
        {"id": 1, "layer": 25, "owner": "UserNotificationCenter", "bounds": {"X": 100, "Y": 50, "Width": 800, "Height": 600}},
        {"id": 2, "layer": 0, "owner": "T3 Code", "bounds": {"X": 0, "Y": 0, "Width": 1400, "Height": 900}},
        {"id": 3, "layer": 0, "owner": "AirPort Utility", "bounds": {"X": 100, "Y": 51, "Width": 800, "Height": 600}},
    ]
    assert ax_collect.find_window_id([100.4, 50.0], [800, 600], windows) == 3
    assert ax_collect.find_window_id([300.0, 50.0], [800, 600], windows) is None


def test_window_capture_command_captures_one_window_without_shadow():
    assert ax_collect.window_capture_command(42, Path("/x/p.png")) == ["screencapture", "-x", "-o", "-l42", "/x/p.png"]


def test_capture_screenshot_refuses_when_no_window_matches(tmp_path):
    import pytest
    with pytest.raises(RuntimeError, match="no on-screen window"):
        ax_collect.capture_screenshot([0.0, 0.0], [10, 10], tmp_path / "p.png", lister=lambda: [])
