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
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from grounder.data.dedupe import DEDUPE_JACCARD, jaccard
from grounder.data.pages import page_id_for, text_signature, write_page
from grounder.extract import ax
from grounder.records import Element, Page, action_repr, equiv_ids
from tools.ax_collect import (DEFAULT_MIN_ELEMENTS, DUPLICATE, SAVED, SCREENSHOT_TIMEOUT, SPLITS, TOO_FEW, Collector,
                              capture_screenshot, screenshot_command, slugify)
from tools.mac_policy import AppsPolicy, load_policy
from tools.mac_safety import (
    Scrubber, denied_click_term, fixed_type_text, private_view_reason, signin_or_payment_reason, typing_allowed,
)

TEST_APPS = ("Numbers", "Pages", "Keynote", "Font Book", "Disk Utility")
TRAIN_APPS = (
    "Finder", "Safari", "Mail", "Notes", "Reminders", "Calendar", "Contacts", "Maps", "Photos", "Music", "Podcasts",
    "TV", "Books", "News", "Stocks", "Weather", "Clock", "Calculator", "Preview", "TextEdit", "Terminal",
    "Activity Monitor", "Console", "Script Editor", "Shortcuts", "Freeform", "App Store", "System Settings",
    "Dictionary", "Digital Color Meter", "Automator",
)
NEVER_QUIT = frozenset({"Finder", "Terminal"})

# The only keys ever sent. No cmd+q, cmd+delete, return/enter or clicks.
ALLOWED_KEYS = frozenset({"cmd+,", "cmd+n", "cmd+w", "cmd+d", "cmd+o", "cmd+p", "cmd+f", "cmd+t", "escape"})
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
        if self._run(["open", "-Ra", app]).returncode == 0:
            return True
        bundle = f"{app}.app"
        return any((Path(base) / bundle).exists() for base in (
            "/Applications", "/System/Applications", "/System/Applications/Utilities", str(Path.home() / "Applications"),
        ))

    def is_running(self, app: str) -> bool:
        return self._osascript(f"application {applescript_string(app)} is running") == "true"

    def launch(self, app: str, args: Iterable[str] = ()) -> None:
        command = ["open", "-a", app]
        extra = list(args)
        if extra:
            command += ["--args", *extra]
        self._check(command)

    def open_path(self, app: str, path: Path) -> None:
        self._check(["open", "-a", app, str(path)])

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

    def click_at(self, app: str, x: float, y: float) -> None:
        self._osascript(
            'tell application "System Events"',
            f'tell process {applescript_string(app)}',
            f'click at {{{round(x)}, {round(y)}}}',
            'end tell',
            'end tell',
        )

    def type_text(self, text: str) -> None:
        self._osascript(f'tell application "System Events" to keystroke {applescript_string(text)}')

    def process_id(self, app: str) -> int | None:
        proc = self._run(["pgrep", "-x", app])
        if proc.returncode != 0:
            return None
        try:
            return int(proc.stdout.splitlines()[0])
        except (IndexError, ValueError):
            return None

    def bundle_id(self, app: str) -> str:
        return self._osascript(f"id of app {applescript_string(app)}")

    def export_prefs(self, bundle_id: str, path: str) -> bool:
        """Snapshot an app's preferences domain; False when it has none yet."""
        return self._run(["defaults", "export", bundle_id, path]).returncode == 0

    def import_prefs(self, bundle_id: str, path: str) -> None:
        self._run(["defaults", "import", bundle_id, path])

    def delete_prefs(self, bundle_id: str) -> None:
        self._run(["defaults", "delete", bundle_id])

    def quit(self, app: str) -> None:
        """Politely quit an app this collector launched, with SIGTERM as a last-resort cleanup for that same app."""
        try:
            self._osascript(f"quit app {applescript_string(app)}")
        except RuntimeError:
            pass
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if self.process_id(app) is None:
                return
            time.sleep(0.1)
        pid = self.process_id(app)
        if pid is not None:
            os.kill(pid, signal.SIGTERM)


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



# --- bounded random-but-safe trajectory walks -----------------------------------------------------

WALK_CLICK, WALK_TYPE = "click", "type"
WALK_SCHEMA_VERSION = 1


class UnsafeStateError(RuntimeError):
    """The current window is a sign-in/payment or never-save view; the app must be abandoned."""

    def __init__(self, reason: str, kind: str) -> None:
        super().__init__(reason)
        self.reason, self.kind = reason, kind


@dataclass(frozen=True)
class StateResult:
    page_id: str
    page: Page
    is_new: bool
    duplicate_of: str = ""


class WalkStateStore:
    """Per-app novelty index. Near-duplicates (Jaccard >= 0.9) resolve to the representative saved page."""

    def __init__(self, pages_dir: Path, threshold: float = DEDUPE_JACCARD) -> None:
        self.pages_dir, self.threshold = pages_dir, threshold
        self.entries: list[tuple[frozenset, str, Page]] = []
        self.by_id: dict[str, Page] = {}

    def add(self, raw: dict[str, Any], app: str) -> StateResult:
        site = slugify(app)
        page = ax.page_from_dump(raw, f"{site}-pending", site)
        page.page_id = page_id_for(site, page)
        if page.page_id in self.by_id:
            return StateResult(page.page_id, self.by_id[page.page_id], False, page.page_id)
        signature = text_signature(page)
        for other_signature, other_id, _other_page in self.entries:
            if jaccard(signature, other_signature) >= self.threshold:
                return StateResult(other_id, self.by_id[other_id], False, other_id)
        write_page(page, self.pages_dir)
        self.entries.append((signature, page.page_id, page))
        self.by_id[page.page_id] = page
        return StateResult(page.page_id, page, True)

    @property
    def distinct_states(self) -> int:
        return len(self.entries)


def state_png_path(page_path: Path) -> Path:
    return page_path.with_name(page_path.name.removesuffix(".json.gz") + ".png")


def save_icon_crops(png_path: Path, raw: dict[str, Any], page: Page, crops_dir: Path) -> int:
    """Crop icon-only elements (no visible text, but nonvisual accessibility text) to 64 px PNGs."""
    try:
        from PIL import Image
    except ImportError:
        return 0
    if not png_path.exists():
        return 0
    rows = {row.get("id"): row for row in raw.get("elements") or [] if isinstance(row, dict)}
    origin = raw.get("origin") or [0.0, 0.0]
    viewport = raw.get("viewport") or [1, 1]
    count = 0
    with Image.open(png_path) as image:
        sx, sy = image.width / max(1, viewport[0]), image.height / max(1, viewport[1])
        for element in page.elements:
            row = rows.get(element.id) or {}
            if str(row.get("visible_text") or "").strip() or not str(row.get("nonvisual_text") or "").strip():
                continue
            x, y, w, h = element.abs_box
            cx, cy = (x - origin[0]) * sx + w * sx / 2, (y - origin[1]) * sy + h * sy / 2
            side = max(8.0, max(w * sx, h * sy))
            left, top = round(cx - side / 2), round(cy - side / 2)
            right, bottom = round(cx + side / 2), round(cy + side / 2)
            # Clamp inside the capture; very small edge-clipped crops are better than no record.
            left, top = max(0, left), max(0, top)
            right, bottom = min(image.width, max(left + 1, right)), min(image.height, max(top + 1, bottom))
            crop = image.crop((left, top, right, bottom))
            crop.thumbnail((64, 64))
            out = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
            out.paste(crop, ((64 - crop.width) // 2, (64 - crop.height) // 2))
            crops_dir.mkdir(parents=True, exist_ok=True)
            out.save(crops_dir / f"{page.page_id}-{element.id}.png")
            count += 1
    return count


def screenshot_mismatch(png_path: Path, viewport: list[int], tolerance: float = 0.03) -> str:
    """Why a window capture doesn't look like the dumped window (wrong aspect ratio or not a whole-number scale)."""
    try:
        from PIL import Image
    except ImportError:
        return ""
    with Image.open(png_path) as image:
        width, height = image.size
    vw, vh = max(1, viewport[0]), max(1, viewport[1])
    sx, sy = width / vw, height / vh
    if abs(sx - sy) > tolerance * max(sx, sy) or abs(sx - round(sx)) > tolerance * max(1.0, sx):
        return f"capture {width}x{height} does not match window {vw}x{vh}"
    return ""


def probe_screen_recording() -> tuple[bool, str]:
    """Whether Screen Recording is available, using the same capture path as ax_collect."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "probe.png"
        try:
            proc = subprocess.run(screenshot_command([0.0, 0.0], [4, 4], path), capture_output=True, text=True,
                                  timeout=SCREENSHOT_TIMEOUT)
        except Exception as exc:
            return False, str(exc)
        if proc.returncode != 0:
            return False, proc.stderr.strip() or f"screencapture exited with {proc.returncode}"
        return path.exists(), "" if path.exists() else "screencapture produced no file"


@dataclass(frozen=True)
class WalkCandidate:
    element: Element
    action: str
    weight: float

LOGIC_QUICKSTART_TEMPLATE = Path(
    "/Applications/Logic Pro.app/Contents/Resources/Library.bundle/Projects/Templates/Quickstart.logicx")


def prepare_throwaway_project(canonical: str, temp_dir: str) -> Path | None:
    """Copy a factory blank project into the throwaway folder, when a safe factory template exists.

    Logic Pro ships a small Quickstart project template. The copy is a new temporary document instance: opening it
    never opens or writes a user project, and the collector deletes the whole temporary folder after quitting Logic.
    Final Cut Pro ships no comparable blank library package, so it still requires a verifiable untitled library.
    """
    if canonical != "Logic Pro" or not temp_dir or not LOGIC_QUICKSTART_TEMPLATE.is_dir():
        return None
    target = Path(temp_dir) / "TinyGrounderThrowaway.logicx"
    shutil.copytree(LOGIC_QUICKSTART_TEMPLATE, target)
    for directory in [target, *(p for p in target.rglob("*") if p.is_dir())]:
        directory.chmod(directory.stat().st_mode | 0o200)
    return target


class RandomWalkCollector:
    """Bounded random-but-safe walks through one macOS app at a time.

    Every click label passes the hard denylist in tools/mac_safety plus the app policy. Only search fields and
    throwaway text documents receive keyboard input, and then only fixed harmless strings. Each recorded step is
    pre-action page id -> action -> post-action page id; pages themselves are scrubbed before they are written.
    """

    def __init__(
        self, mac: Mac, policy: AppsPolicy, split: str, out: Path, settle: float = 1.2,
        sleep: Sleep = time.sleep, show: Callable[[str], None] = print, seed: int = 0,
        max_steps: int = 0, walk_budget: int = 0, state_budget: int = 0,
        screenshots: bool = False, binary: Path = ax.AXDUMP,
        dump_app: Callable[[str], dict[str, Any]] | None = None,
    ) -> None:
        self.mac, self.policy, self.split, self.out = mac, policy, split, Path(out)
        self.settle, self.sleep, self.show, self.seed = settle, sleep, show, seed
        self.max_steps, self.walk_budget, self.state_budget = max_steps, walk_budget, state_budget
        self.screenshots, self.binary = screenshots, binary
        self.scrubber = Scrubber.from_system()
        self.root = self.out / split
        self.pages_dir = self.root / "pages"
        self.walks_dir = self.root / "walks"
        self.denials: dict[str, Counter[str]] = {}
        self.denial_examples: list[dict[str, str]] = []
        self.app_results: dict[str, dict[str, Any]] = {}
        self.screenshot_failures: Counter[str] = Counter()
        self.icon_crops = 0
        self.dump_app = dump_app

    def _rng(self, app: str) -> random.Random:
        return random.Random((self.seed, self.split, app).__repr__())

    def _dump(self, launch_name: str) -> dict[str, Any]:
        raw = (self.dump_app(launch_name) if self.dump_app is not None else
               ax.run_axdump(["--app", launch_name, "--timeout", "8"], binary=self.binary, timeout=13))
        return self.scrubber.scrub_raw_dump(raw)

    def _state(self, raw: dict[str, Any], app: str, store: WalkStateStore) -> StateResult:
        signin = signin_or_payment_reason(raw)
        if signin:
            raise UnsafeStateError(signin, "signin_or_payment")
        # private_view_reason gets only global history/account/profile terms here; policy terms are checked below.
        policy = self.policy.policy_for(app)
        private = private_view_reason(raw, policy.deny_views)
        if private:
            raise UnsafeStateError(private, "forbidden_view")
        result = store.add(raw, app)
        if result.is_new and self.screenshots:
            self._screenshot(raw, result)
        return result

    def _screenshot(self, raw: dict[str, Any], result: StateResult) -> None:
        page_path = self.pages_dir / result.page.site / f"{result.page.page_id}.json.gz"
        png = state_png_path(page_path)
        try:
            capture_screenshot(raw["origin"], raw["viewport"], png)
        except Exception as exc:
            self.screenshot_failures[result.page.site] += 1
            self.show(f"[{result.page.site}] screenshot failed: {exc}")
            return
        mismatch = screenshot_mismatch(png, raw["viewport"])
        if mismatch:
            png.unlink(missing_ok=True)
            self.screenshot_failures[result.page.site] += 1
            self.show(f"[{result.page.site}] screenshot discarded: {mismatch}")
            return
        self.icon_crops += save_icon_crops(png, raw, result.page, png.parent / "crops")

    def _deny(self, app: str, element: Element, term: str) -> None:
        self.denials.setdefault(app, Counter())[term] += 1
        if len(self.denial_examples) < 300:
            self.denial_examples.append({"app": app, "label": element.text, "term": term})

    def candidates(self, page: Page, app: str, throwaway_document: bool, frequency: Counter[str]) -> list[WalkCandidate]:
        policy = self.policy.policy_for(app)
        out: list[WalkCandidate] = []
        for element in page.elements:
            if not element.enabled or not element.in_viewport or not element.text.strip():
                continue
            label = element.text
            if element.role in {"textbox", "searchbox"}:
                if typing_allowed(policy.typing, element.role, element.tag, label, throwaway_document):
                    out.append(WalkCandidate(element, WALK_TYPE, 0.7))
                continue
            if element.role not in policy.click_roles:
                continue
            folded = label.casefold()
            context = f"{label} {element.context}".casefold()
            if policy.allow_labels and not any(term.casefold() in context for term in policy.allow_labels):
                continue
            term = denied_click_term(label, policy.deny_labels)
            if term:
                self._deny(app, element, term)
                continue
            key = f"{element.role}\t{folded}"
            base = {"button": 2.5, "link": 2.0, "tab": 2.5, "menuitem": 1.2, "radio": 1.2,
                    "combobox": 0.8, "other": 1.0}.get(element.role, 0.5)
            out.append(WalkCandidate(element, WALK_CLICK, base / (1 + frequency[key])))
        return out

    def _choose(self, candidates: list[WalkCandidate], rng: random.Random) -> WalkCandidate:
        return rng.choices(candidates, weights=[candidate.weight for candidate in candidates], k=1)[0]

    def _ensure_frontmost(self, launch_name: str) -> bool:
        if self.mac.frontmost().casefold() != launch_name.casefold():
            self.mac.launch(launch_name)
            self.sleep(UNDO_PAUSE)
        return self.mac.frontmost().casefold() == launch_name.casefold()

    def _perform(self, launch_name: str, candidate: WalkCandidate, typed_text: str) -> None:
        if not self._ensure_frontmost(launch_name):
            raise RuntimeError(f"wrong frontmost app: {self.mac.frontmost()}")
        x, y, w, h = candidate.element.abs_box
        self.mac.click_at(launch_name, x + w / 2, y + h / 2)
        if candidate.action == WALK_TYPE:
            self.sleep(0.15)
            self.mac.type_text(typed_text)

    def run_walk(self, app: str, walk_index: int, store: WalkStateStore, rng: random.Random,
                 throwaway_document: bool, frequency: Counter[str]) -> dict[str, Any]:
        policy = self.policy.policy_for(app)
        launch_name = policy.launch_name
        max_steps = self.max_steps or policy.max_steps
        walk_id = f"{slugify(app)}-{self.split}-{walk_index:03d}"
        steps: list[dict[str, Any]] = []
        end_reason = "step budget reached"
        unchanged = 0
        for step_index in range(max_steps):
            pre_raw = self._dump(launch_name)
            pre = self._state(pre_raw, app, store)
            candidates = self.candidates(pre.page, app, throwaway_document, frequency)
            if not candidates:
                end_reason = "no safe actionable elements"
                break
            candidate = self._choose(candidates, rng)
            typed = fixed_type_text(self.seed + walk_index + step_index) if candidate.action == WALK_TYPE else ""
            frequency[f"{candidate.element.role}\t{candidate.element.text.casefold()}"] += 1
            self._perform(launch_name, candidate, typed)
            self.sleep(self.settle)
            try:
                frontmost = self.mac.frontmost()
            except Exception:
                frontmost = ""
            if frontmost and frontmost.casefold() != launch_name.casefold():
                end_reason = f"wrong app after action (frontmost is {frontmost})"
                break
            post_raw = self._dump(launch_name)
            post = self._state(post_raw, app, store)
            target = candidate.element
            steps.append({
                "index": step_index,
                "pre_page_id": pre.page_id,
                "post_page_id": post.page_id,
                "action": {"type": candidate.action, "target_id": target.id, "role": target.role,
                           "label": target.text, "tag": target.tag, "typed_text": typed},
                "target": {"id": target.id, "equiv_ids": equiv_ids(pre.page.elements, target.id),
                           "visible_text": self._provenance(pre_raw, target.id, "visible_text"),
                           "nonvisual_text": self._provenance(pre_raw, target.id, "nonvisual_text")},
            })
            unchanged = unchanged + 1 if post.page_id == pre.page_id else 0
            if unchanged >= 2:
                end_reason = "state stopped changing"
                break
        return {"schema_version": WALK_SCHEMA_VERSION, "walk_id": walk_id, "app": app, "split": self.split,
                "seed": self.seed, "steps": steps, "end_reason": end_reason}

    @staticmethod
    def _provenance(raw: dict[str, Any], element_id: int, key: str) -> str:
        for row in raw.get("elements") or []:
            if isinstance(row, dict) and row.get("id") == element_id:
                return str(row.get(key) or "")
        return ""

    def _write_walks(self, canonical: str, walks: list[dict[str, Any]]) -> str:
        if not walks:
            return ""
        path = self.walks_dir / f"{slugify(canonical)}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for walk in walks:
                f.write(json.dumps(walk, sort_keys=True) + "\n")
        return str(path)

    def _wait_for_window(self, launch_name: str, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                if self.mac.window_count(launch_name) > 0:
                    return
            except Exception:
                pass
            self.sleep(0.5)
        raise RuntimeError(f"{launch_name} opened no accessible window within {timeout:.0f}s")

    def run_app(self, app: str, host: str = "") -> dict[str, Any]:
        canonical = self.policy.canonical(app)
        policy = self.policy.policy_for(canonical)
        result: dict[str, Any] = {"status": "skipped", "reason": "", "walks": 0, "states": 0, "steps": 0,
                                  "denials": {}, "walk_file": ""}
        self.app_results[canonical] = result
        refusal = self.policy.refusal(canonical, self.split)
        if canonical.casefold() == host.casefold() or policy.launch_name.casefold() == host.casefold():
            refusal = "hosts this script"
        if refusal:
            result["reason"] = refusal
            self.show(f"[{canonical}] skipped: {refusal}")
            return result
        if not self.mac.is_installed(policy.launch_name):
            result["reason"] = "not installed"
            self.show(f"[{canonical}] skipped: not installed")
            return result

        rng = self._rng(canonical)
        store = WalkStateStore(self.pages_dir)
        walks: list[dict[str, Any]] = []
        frequency: Counter[str] = Counter()
        was_running = self.mac.is_running(policy.launch_name)
        if was_running:
            # A running app is the user's live session: its preferences can't be restored safely while it runs.
            result["reason"] = "already running (close it to collect)"
            self.show(f"[{canonical}] skipped: {result['reason']}")
            return result
        temp_dir = ""
        initial_windows = 0
        prefs_dir = tempfile.mkdtemp(prefix=f"tiny-grounder-prefs-{slugify(canonical)}-")
        prefs_file = os.path.join(prefs_dir, "prefs.plist")
        try:
            bundle = self.mac.bundle_id(policy.launch_name)
        except Exception:
            bundle = ""
        had_prefs = bool(bundle) and self.mac.export_prefs(bundle, prefs_file)
        try:
            throwaway_path = None
            if policy.throwaway_project:
                temp_dir = tempfile.mkdtemp(prefix=f"tiny-grounder-{slugify(canonical)}-")
                throwaway_path = prepare_throwaway_project(canonical, temp_dir)
            try:
                initial_windows = self.mac.window_count(policy.launch_name)
            except Exception:
                initial_windows = 0
            if throwaway_path is not None:
                self.mac.open_path(policy.launch_name, throwaway_path)
            else:
                self.mac.launch(policy.launch_name, policy.launch_args)
            self.sleep(max(self.settle, 2.0))
            for key in policy.start_keys:
                check_key(key)
                if self._ensure_frontmost(policy.launch_name):
                    self.mac.press(key)
                    self.sleep(self.settle)
            wait_timeout = 25.0 if canonical in {"Logic Pro", "Final Cut Pro", "Xcode", "Docker"} else 8.0
            try:
                self._wait_for_window(policy.launch_name, wait_timeout / 2)
            except RuntimeError:
                # Menu-bar utilities often open no window on activation. cmd+, is the safe settings key and is
                # already allowlisted; if the app has no preferences window either, the second wait fails cleanly.
                if self._ensure_frontmost(policy.launch_name):
                    self.mac.press("cmd+,")
                    self.sleep(self.settle)
                self._wait_for_window(policy.launch_name, wait_timeout / 2)
            # A throwaway app is collected only when its initial window is demonstrably new/untitled. This refuses
            # existing libraries/projects; media-app project creation is deliberately conservative (see report).
            first_raw = self._dump(policy.launch_name)
            first_state = self._state(first_raw, canonical, store)
            title = (first_raw.get("title") or "").casefold()
            verifiable_untitled = "untitled" in title or "unnamed" in title
            throwaway_document = (policy.throwaway_project or canonical in {"Freeform", "Stickies", "TextEdit"}) and verifiable_untitled
            if policy.throwaway_project and throwaway_path is None:
                raise UnsafeStateError("no factory blank project/library template for a temp-folder copy", "throwaway_missing")
            budget = self.walk_budget or policy.walk_budget
            state_budget = self.state_budget or policy.state_budget
            walk_index = 0
            failures = 0
            while len(walks) < budget and store.distinct_states < state_budget and failures < max(3, budget // 3):
                try:
                    walk = self.run_walk(canonical, walk_index, store, rng, throwaway_document, frequency)
                except UnsafeStateError:
                    raise
                except Exception as exc:
                    failures += 1
                    self.show(f"[{canonical}] walk {walk_index} failed: {type(exc).__name__}: {exc}")
                    self._close_open_dialogs(policy.launch_name)
                    continue
                if walk["steps"]:
                    walks.append(walk)
                    walk_index += 1
                else:
                    failures += 1
                self._close_open_dialogs(policy.launch_name)
            if walks:
                path = self.walks_dir / f"{slugify(canonical)}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w", encoding="utf-8") as f:
                    for walk in walks:
                        f.write(json.dumps(walk, sort_keys=True) + "\n")
                result["walk_file"] = str(path)
            result.update(status="collected", walks=len(walks), states=store.distinct_states,
                          steps=sum(len(w["steps"]) for w in walks), denials=dict(self.denials.get(canonical, {})))
            self.show(f"[{canonical}] collected {len(walks)} walks, {store.distinct_states} states")
        except UnsafeStateError as exc:
            result["reason"] = f"{exc.kind}: {exc.reason}"
            result["walk_file"] = self._write_walks(canonical, walks)
            if walks:
                result.update(status="partial", walks=len(walks), states=store.distinct_states,
                              steps=sum(len(w["steps"]) for w in walks), denials=dict(self.denials.get(canonical, {})))
                self.show(f"[{canonical}] stopped early: {result['reason']}; kept {len(walks)} safe walks")
            else:
                self.show(f"[{canonical}] skipped: {result['reason']}")
        except (ax.AxdumpError, RuntimeError, OSError, subprocess.SubprocessError, ValueError) as exc:
            result["reason"] = f"{type(exc).__name__}: {exc}"
            self.show(f"[{canonical}] error: {result['reason']}")
        finally:
            try:
                self._cleanup_windows(policy.launch_name, initial_windows)
            except Exception:
                pass
            if not was_running:
                try:
                    self.mac.quit(policy.launch_name)
                except Exception:
                    pass
            # Put back any preference a walk changed (radio buttons and pop-ups in Settings windows persist).
            if bundle:
                try:
                    if had_prefs:
                        self.mac.import_prefs(bundle, prefs_file)
                    else:
                        self.mac.delete_prefs(bundle)
                    result["prefs_restored"] = True
                except Exception as exc:
                    result["prefs_restored"] = False
                    self.show(f"[{canonical}] WARNING: preferences not restored: {exc}")
            shutil.rmtree(prefs_dir, ignore_errors=True)
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)
        return result

    def _close_open_dialogs(self, launch_name: str) -> None:
        """Best-effort close of dialogs/sheets opened by the walk: Escape first, then a visible Cancel button."""
        for _ in range(3):
            try:
                if not self._ensure_frontmost(launch_name):
                    return
                if not self.mac.has_sheet(launch_name):
                    return
                self.mac.press("escape")
                self.sleep(UNDO_PAUSE)
                if not self.mac.has_sheet(launch_name):
                    return
                raw = self._dump(launch_name)
                page = ax.page_from_dump(raw, "dialog-pending", slugify(launch_name))
                cancel = next((e for e in page.elements if e.role == "button" and e.text.strip().casefold() == "cancel"), None)
                if cancel is None:
                    return
                x, y, w, h = cancel.abs_box
                self.mac.click_at(launch_name, x + w / 2, y + h / 2)
                self.sleep(UNDO_PAUSE)
            except Exception:
                return

    def _cleanup_windows(self, launch_name: str, initial_windows: int) -> None:
        """Close only windows added during collection; answer save sheets with Don't Save, never Save."""
        self._close_open_dialogs(launch_name)
        for _ in range(8):
            try:
                if self.mac.window_count(launch_name) <= initial_windows:
                    break
                if not self._ensure_frontmost(launch_name):
                    break
                self.mac.press("cmd+w")
                self.sleep(UNDO_PAUSE)
                if self.mac.has_sheet(launch_name):
                    self.mac.press(DONT_SAVE)
                    self.sleep(UNDO_PAUSE)
                    if self.mac.has_sheet(launch_name):
                        self.mac.press("escape")
                        self.sleep(UNDO_PAUSE)
            except Exception:
                break

    def write_report(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        report = {
            "schema_version": WALK_SCHEMA_VERSION,
            "split": self.split,
            "policy": str(self.policy.path),
            "seed": self.seed,
            "apps": self.app_results,
            "safety_denials": {app: dict(counts) for app, counts in self.denials.items()},
            "safety_denial_examples": self.denial_examples,
            "privacy_scrub_counts": dict(self.scrubber.counts),
            "screenshots": {"enabled": self.screenshots, "failures": dict(self.screenshot_failures),
                            "icon_crops": self.icon_crops},
        }
        path = self.root / "collection-report.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        return path




def frontmost_is_loginwindow(binary: Path = ax.AXDUMP) -> bool:
    """True when the WindowServer frontmost application is loginwindow (the session is locked or curtained)."""
    try:
        ax.run_axdump(["--frontmost", "--timeout", "3"], binary=binary, timeout=6)
        return False
    except ax.AxdumpError as exc:
        match = re.search(r"pid (\d+)", str(exc))
        if not match:
            return False
        proc = subprocess.run(["ps", "-p", match.group(1), "-o", "comm="], capture_output=True, text=True, timeout=3)
        return "loginwindow" in proc.stdout

def run_walk_mode(args: argparse.Namespace) -> int:
    policy = load_policy(args.policy)
    apps = [a.strip() for a in args.apps.split(",") if a.strip()] or policy.apps_for_split(args.split)
    refused = [(app, policy.refusal(app, args.split)) for app in apps]
    refused = [(app, reason) for app, reason in refused if reason]
    if refused and args.apps:
        for app, reason in refused:
            print(f"[{app}] skipped: {reason}")
        apps = [app for app in apps if not policy.refusal(app, args.split)]
    if args.dry_run:
        for app in apps:
            policy_app = policy.policy_for(app)
            print(f"{app}: up to {policy_app.walk_budget} walks x {policy_app.max_steps} steps, "
                  f"{policy_app.state_budget} states; roles={sorted(policy_app.click_roles)} typing={policy_app.typing}")
        return 0
    if not args.binary.exists():
        print(f"{args.binary} not found: run `swift build -c release` in {ax.AXDUMP.parents[2]}", file=sys.stderr)
        return 2
    if frontmost_is_loginwindow(args.binary):
        print("the macOS session is at loginwindow (locked or curtained); GUI collection cannot run", file=sys.stderr)
        return 4
    screenshots, screenshot_reason = probe_screen_recording()
    if not screenshots:
        print(f"Screen Recording is not granted; continuing without screenshots ({screenshot_reason})")
    mac = Mac()
    host = mac.frontmost()
    print(f"{host} is left alone. Do not touch the keyboard or mouse until collection finishes.")
    walker = RandomWalkCollector(
        mac, policy, args.split, Path(args.out), settle=args.settle, seed=args.seed,
        max_steps=args.max_steps, walk_budget=args.walk_budget, state_budget=args.state_budget,
        screenshots=screenshots, binary=args.binary,
    )
    try:
        for app in apps:
            walker.run_app(app, host)
    except KeyboardInterrupt:
        print()
    report = walker.write_report()
    print(f"wrote walk collection report {report}")
    return 0

def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="pages dir for scene mode, walk output root in walk mode")
    parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument("--mode", choices=("scenes", "walks"), default="scenes",
                        help="legacy fixed safe scenes, or bounded random-but-safe goal-walk collection")
    parser.add_argument("--apps", default="", help="comma list overriding the split's default apps")
    parser.add_argument("--settle", type=float, default=DEFAULT_SETTLE, help="seconds to wait after each action")
    parser.add_argument("--min-elements", type=int, default=DEFAULT_MIN_ELEMENTS)
    parser.add_argument("--max-per-app", type=int, default=0, help="stop an app after this many saved pages (0 = no cap)")
    parser.add_argument("--policy", type=Path, default=Path("configs/apps_policy.yaml"), help="app policy for walk mode")
    parser.add_argument("--binary", type=Path, default=ax.AXDUMP, help="axdump binary")
    parser.add_argument("--seed", type=int, default=0, help="random seed for walk mode")
    parser.add_argument("--max-steps", type=int, default=0, help="walk-mode override for steps per walk")
    parser.add_argument("--walk-budget", type=int, default=0, help="walk-mode override for walks per app")
    parser.add_argument("--state-budget", type=int, default=0, help="walk-mode override for distinct states per app")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = parser.parse_args(argv)
    if args.mode == "walks":
        return run_walk_mode(args)
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
