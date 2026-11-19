"""Unattended collector of macOS accessibility pages: opens apps and safe views itself, dumps each one.

    uv run python tools/ax_auto.py --out data/pages_ax --split train
    uv run python tools/ax_auto.py --out data/pages_ax --split test          # Numbers, Pages, Keynote, Font Book, Disk Utility
    uv run python tools/ax_auto.py --out data/pages_ax --apps "Notes,Mail" --dry-run

Per app: launch it (`open -a`), dump the frontmost window, then walk a fixed list of view-changing scenes (settings
window, new window, open/print panel, find bar; System Settings panes by URL), dumping after each and undoing it
(cmd+w or Escape). Only the keystrokes in ALLOWED_KEYS are ever sent, and never a click. An app that was not running
before is quit politely afterwards (never Finder, Terminal or the app hosting this script, which is not dumped at
all). Saving, near-duplicate skipping, the train/test leak guard and test-split screenshots are those of
tools/ax_collect.py. Do not touch the keyboard or mouse while it runs. The terminal needs Accessibility (keystrokes,
axdump) and, for `test`, Screen Recording. See docs/macos-collection.md.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from grounder.extract import ax
from tools.ax_collect import DEFAULT_MIN_ELEMENTS, DUPLICATE, SAVED, SPLITS, TOO_FEW, Collector, slugify

TEST_APPS = ("Numbers", "Pages", "Keynote", "Font Book", "Disk Utility")
TRAIN_APPS = (
    "Finder", "Safari", "Mail", "Notes", "Reminders", "Calendar", "Contacts", "Maps", "Photos", "Music", "Podcasts",
    "TV", "Books", "News", "Stocks", "Weather", "Clock", "Calculator", "Preview", "TextEdit", "Terminal",
    "Activity Monitor", "Console", "Script Editor", "Shortcuts", "Freeform", "App Store", "System Settings",
    "Dictionary", "Digital Color Meter", "Automator",
)
NEVER_QUIT = frozenset({"Finder", "Terminal"})

# The only keys ever sent. No cmd+q, cmd+delete, return/enter or clicks.
ALLOWED_KEYS = frozenset({"cmd+,", "cmd+n", "cmd+w", "cmd+d", "cmd+o", "cmd+p", "cmd+f", "escape"})
DONT_SAVE = "cmd+d"  # the "Don't Save" button of a save-changes sheet
ESCAPE_KEY_CODE = 53

SYSTEM_SETTINGS = "System Settings"
PANE_IDS = (
    "com.apple.wifi-settings-extension", "com.apple.BluetoothSettings", "com.apple.Network-Settings.extension",
    "com.apple.Displays-Settings.extension", "com.apple.Sound-Settings.extension",
    "com.apple.Keyboard-Settings.extension", "com.apple.Trackpad-Settings.extension",
    "com.apple.Mouse-Settings.extension", "com.apple.Battery-Settings.extension",
    "com.apple.Appearance-Settings.extension", "com.apple.Accessibility-Settings.extension",
    "com.apple.Notifications-Settings.extension", "com.apple.Focus-Settings.extension",
    "com.apple.Screen-Time-Settings.extension", "com.apple.SystemProfiler.AboutExtension",
    "com.apple.Software-Update-Settings.extension", "com.apple.settings.Storage",
    "com.apple.Date-Time-Settings.extension", "com.apple.Sharing-Settings.extension",
    "com.apple.Desktop-Settings.extension", "com.apple.Wallpaper-Settings.extension",
    "com.apple.Lock-Screen-Settings.extension", "com.apple.settings.PrivacySecurity.extension",
    "com.apple.Users-Groups-Settings.extension", "com.apple.Passwords-Settings.extension",
    "com.apple.Internet-Accounts-Settings.extension", "com.apple.ControlCenter-Settings.extension",
    "com.apple.Siri-Settings.extension", "com.apple.Spotlight-Settings.extension",
    "com.apple.Print-Scan-Settings.extension", "com.apple.Game-Center-Settings.extension",
    "com.apple.LoginItems-Settings.extension",
)

DEFAULT_SETTLE = 2.5
UNDO_PAUSE = 0.5
COMMAND_TIMEOUT = 15.0
WRONG_APP, ERROR = "wrong app", "error"
COLUMNS = (SAVED, DUPLICATE, TOO_FEW, WRONG_APP, ERROR)

Sleep = Callable[[float], None]


def check_key(key: str) -> None:
    if key not in ALLOWED_KEYS:
        raise ValueError(f"refusing to send {key!r}; allowed: {sorted(ALLOWED_KEYS)}")


def key_script(key: str) -> str:
    """The System Events AppleScript line for one allowed key."""
    check_key(key)
    if key == "escape":
        return f'tell application "System Events" to key code {ESCAPE_KEY_CODE}'
    return f'tell application "System Events" to keystroke "{key.removeprefix("cmd+")}" using command down'


@dataclass(frozen=True)
class Scene:
    """One view of an app to dump: keys (or a settings URL) to reach it, and how to undo it."""

    name: str
    keys: tuple[str, ...] = ()
    url: str = ""
    undo: tuple[str, ...] = ()
    closes_window: bool = False  # the keys open a window that cmd+w must close (only if one really opened)

    def __post_init__(self) -> None:
        for key in (*self.keys, *self.undo):
            check_key(key)


GENERIC_SCENES = (
    Scene("settings window", keys=("cmd+,",), undo=("cmd+w",), closes_window=True),
    Scene("new window", keys=("cmd+n",), undo=("cmd+w",), closes_window=True),
    Scene("open panel", keys=("cmd+o",), undo=("escape",)),
    Scene("print panel", keys=("cmd+p",), undo=("escape",)),
    Scene("find bar", keys=("cmd+f",), undo=("escape",)),
)
# cmd+o / cmd+p act on the selection in Finder, so only the harmless scenes run there.
SKIPPED_SCENES = {"Finder": {"open panel", "print panel"}}
# cmd+n creates real data in many apps (a Calendar event, a Photos album, a Music playlist, a Contacts card, a
# Freeform board, a Shortcut, a Mail draft), so the new-window scene runs only where it just opens a window or an
# unsaved document.
NEW_WINDOW_APPS = frozenset({
    "Finder", "Safari", "TextEdit", "Script Editor", "Terminal", "Maps", "Dictionary", "Numbers", "Pages", "Keynote",
})


def build_plan(app: str) -> list[Scene]:
    """Scenes for `app`: the plain launch first, then its panes (System Settings) or the generic safe views."""
    plan = [Scene("launch")]
    if app == SYSTEM_SETTINGS:
        return plan + [Scene(f"pane {pane}", url=f"x-apple.systempreferences:{pane}") for pane in PANE_IDS]
    skipped = set(SKIPPED_SCENES.get(app, set()))
    if app not in NEW_WINDOW_APPS:
        skipped.add("new window")
    return plan + [s for s in GENERIC_SCENES if s.name not in skipped]


def applescript_string(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


class Mac:
    """The macOS side effects (open, osascript, keystrokes); tests replace the whole object."""

    def _run(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(command, capture_output=True, text=True, timeout=COMMAND_TIMEOUT)

    def _check(self, command: list[str]) -> str:
        proc = self._run(command)
        if proc.returncode != 0:
            raise RuntimeError(f"{command[0]} failed: {proc.stderr.strip() or proc.returncode}")
        return proc.stdout.strip()

    def _osascript(self, *lines: str) -> str:
        return self._check(["osascript", *(arg for line in lines for arg in ("-e", line))])

    def is_installed(self, app: str) -> bool:
        return self._run(["open", "-Ra", app]).returncode == 0

    def is_running(self, app: str) -> bool:
        return self._osascript(f"application {applescript_string(app)} is running") == "true"

    def launch(self, app: str) -> None:
        self._check(["open", "-a", app])

    def open_url(self, url: str) -> None:
        self._check(["open", url])

    def frontmost(self) -> str:
        return self._osascript(
            'tell application "System Events" to get name of first application process whose frontmost is true')

    def window_count(self, app: str) -> int:
        return int(self._osascript(
            f'tell application "System Events" to count windows of process {applescript_string(app)}'))

    def has_sheet(self, app: str) -> bool:
        return self._osascript(
            f'tell application "System Events" to tell process {applescript_string(app)}',
            "if (count of windows) is 0 then return false", "return exists sheet 1 of window 1", "end tell",
        ) == "true"

    def press(self, key: str) -> None:
        self._osascript(key_script(key))

    def quit(self, app: str) -> None:
        self._osascript(f"quit app {applescript_string(app)}")


class AutoCollector:
    """Walks the plan of each app, saves each scene through a `Collector` and keeps per-app tallies."""

    def __init__(
        self, collector: Collector, mac: Mac, settle: float = DEFAULT_SETTLE, max_per_app: int = 0,
        sleep: Sleep = time.sleep, show: Callable[[str], None] = print,
    ) -> None:
        self.collector, self.mac, self.settle = collector, mac, settle
        self.max_per_app, self.sleep, self.show = max_per_app, sleep, show
        self.results: dict[str, Counter[str]] = {}
        self.notes: dict[str, str] = {}

    def run_app(self, app: str, host: str = "") -> None:
        """Launch, dump and clean up one app, unless it hosts this script, is missing or is held out."""
        self.results[app] = Counter()
        refusal = self._refusal(app, host)
        if refusal:
            self.notes[app] = refusal
            self.show(f"[{app}] skipped: {refusal}")
            return
        was_running = self.mac.is_running(app)
        try:
            self.mac.launch(app)
            self.sleep(self.settle)
            for scene in build_plan(app):
                if self.max_per_app and self.results[app][SAVED] >= self.max_per_app:
                    break
                self.results[app][self._run_scene(app, scene)] += 1
        finally:
            if not was_running and app not in NEVER_QUIT:
                self.mac.quit(app)

    def _refusal(self, app: str, host: str) -> str:
        if app.casefold() == host.casefold():
            return "hosts this script"
        if not self.mac.is_installed(app):
            return "not installed"
        if self.collector.other_split_has(slugify(app)):
            return f"already in the {next(s for s in SPLITS if s != self.collector.split)} split"
        return ""

    def _run_scene(self, app: str, scene: Scene) -> str:
        """Reach the scene, dump it and undo it; returns the tally column and logs one line."""
        try:
            status, detail = self._perform(app, scene)
        except ax.AxdumpError as e:
            if e.returncode == 2:
                raise  # no Accessibility permission: nothing else can work either
            status, detail = ERROR, str(e)
        except (RuntimeError, OSError, subprocess.SubprocessError, ValueError, KeyError) as e:
            status, detail = ERROR, f"{type(e).__name__}: {e}"
        self.show(f"[{app}] {scene.name} -> {detail}")
        return status

    def _perform(self, app: str, scene: Scene) -> tuple[str, str]:
        if scene.url:
            self.mac.open_url(scene.url)
        elif scene.keys and not self._ensure_frontmost(app):
            return WRONG_APP, "wrong app (could not bring it to the front)"
        windows = self.mac.window_count(app) if scene.closes_window else 0
        for key in scene.keys:
            self.mac.press(key)
        try:
            self.sleep(self.settle)
            return self._dump_and_save(app)
        finally:
            self._undo(app, scene, windows)

    def _ensure_frontmost(self, app: str) -> bool:
        if self.mac.frontmost().casefold() != app.casefold():
            self.mac.launch(app)
            self.sleep(UNDO_PAUSE)
        return self.mac.frontmost().casefold() == app.casefold()

    def _dump_and_save(self, app: str) -> tuple[str, str]:
        raw = self.collector.dump()
        reported = raw.get("app") or "unknown"
        if reported.casefold() != app.casefold():
            return WRONG_APP, f"wrong app (frontmost is {reported})"
        outcome = self.collector.save_dump(raw)
        if outcome.kind == SAVED:
            return SAVED, f"saved {outcome.page.page_id} {len(outcome.page.elements)} elements"
        return (outcome.kind if outcome.kind in COLUMNS else ERROR), outcome.kind

    def _undo(self, app: str, scene: Scene, windows_before: int) -> None:
        if not scene.closes_window:
            for key in scene.undo:
                self.mac.press(key)
        elif self.mac.window_count(app) > windows_before:
            self._close_window(app)

    def _close_window(self, app: str) -> None:
        """cmd+w the window we opened; answer a save-changes sheet with Don't Save, or cancel it with Escape."""
        self.mac.press("cmd+w")
        self.sleep(UNDO_PAUSE)
        if not self.mac.has_sheet(app):
            return
        self.mac.press(DONT_SAVE)
        self.sleep(UNDO_PAUSE)
        if self.mac.has_sheet(app):
            self.mac.press("escape")
            self.show(f"[{app}] save sheet did not accept Don't Save; cancelled it, window left open")

    def summary(self) -> str:
        header = f"{'app':<24}" + "".join(f"{c:>11}" for c in COLUMNS) + "  note"
        rows = [
            f"{app:<24}" + "".join(f"{tally[c]:>11}" for c in COLUMNS) + f"  {self.notes.get(app, '')}"
            for app, tally in self.results.items()
        ]
        return "\n".join([header, *rows])


def select_apps(arg: str, split: str) -> list[str]:
    """The `--apps` comma list, or the split's default apps."""
    return [a.strip() for a in arg.split(",") if a.strip()] or list(TEST_APPS if split == "test" else TRAIN_APPS)


def format_plan(apps: list[str]) -> str:
    lines = []
    for app in apps:
        lines.append(app)
        for s in build_plan(app):
            action = " ".join(s.keys) or s.url or "activate only"
            lines.append(f"  {s.name}: {action}" + (f" (undo: {' '.join(s.undo)})" if s.undo else ""))
    return "\n".join(lines)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="pages dir, e.g. data/pages_ax")
    parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument("--apps", default="", help="comma list overriding the split's default apps")
    parser.add_argument("--settle", type=float, default=DEFAULT_SETTLE, help="seconds to wait after each action")
    parser.add_argument("--min-elements", type=int, default=DEFAULT_MIN_ELEMENTS)
    parser.add_argument("--max-per-app", type=int, default=0, help="stop an app after this many saved pages (0 = no cap)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = parser.parse_args(argv)
    apps = select_apps(args.apps, args.split)
    if args.dry_run:
        print(format_plan(apps))
        return 0
    if not ax.AXDUMP.exists():
        print(f"{ax.AXDUMP} not found: run `swift build -c release` in {ax.AXDUMP.parents[2]}", file=sys.stderr)
        return 2
    mac = Mac()
    host = mac.frontmost()
    print(f"{host} is left alone. Do not touch the keyboard or mouse until the summary appears.")
    collector = Collector(Path(args.out), args.split, lambda: ax.run_axdump(["--frontmost"]), args.min_elements)
    auto = AutoCollector(collector, mac, args.settle, args.max_per_app)
    try:
        for app in apps:
            auto.run_app(app, host)
    except KeyboardInterrupt:
        print()
    except ax.AxdumpError as e:
        print(f"axdump failed: {e}", file=sys.stderr)
    print(auto.summary())
    return 0


if __name__ == "__main__":
    sys.exit(_main())
