"""Record the frozen held-out macOS oracle tasks as goal-mode next-step examples.

    uv run --extra demo python tools/desktop_record.py --dry-run
    uv run --extra demo python tools/desktop_record.py

Input is ``demo/tasks_desktop.yaml`` (frozen before any app was dumped). For every oracle step the recorder
activates the app, dumps the front window with ``axdump``, resolves the oracle target in that pre-action page,
performs the action, and writes a ``records.Example`` plus the ``records.Page`` in the custom-set layout read by
``tools/next_step_recall.py``::

    d_macos|data/nextstep/d_macos.jsonl|data/nextstep/pages_d_macos

Actions use System Events rather than AXPress/AXSetValue. Justification: the recorder already has the exact
window-relative and screen-point geometry of the resolved element in axdump, and System Events' process-scoped
``click at {x, y}``/keyboard commands exercise the same input path a user does without extending the extractor
with a parallel mutation API. Only a small allowlist of non-destructive key chords is supported. Unresolvable or
ambiguous targets are never guessed; the step is dropped, the rest of that task is not attempted, and the reason
is counted in the build report. No ranker is imported or run here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from grounder.data.pages import write_page
from grounder.extract import ax
from grounder.records import ROLE_TO_ID, Element, Example, Page, action_repr, equiv_ids
from tools.ax_collect import page_id_for, slugify
from tools.nextstep_tags import tag_target

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = ROOT / "demo/tasks_desktop.yaml"
DEFAULT_OUT = ROOT / "data/nextstep"
DEFAULT_NAME = "d_macos"
DEFAULT_SETTLE = 1.0
UNDO_PAUSE = 0.4
COMMAND_TIMEOUT = 15.0

CLICK, TYPE, SELECT, KEY = "click", "type", "select", "key"
ACTIONS = frozenset({CLICK, TYPE, SELECT, KEY})

# Recording needs a few editing/navigation chords beyond ax_auto's read-only scene keys. Destructive and
# app-level chords (delete, cmd+q, cmd+delete, cmd+backspace, enter-as-open in Finder, ...) are deliberately absent.
NAMED_KEY_CODES = {"return": 36, "tab": 48, "escape": 53, "space": 49,
                   "left": 123, "right": 124, "down": 125, "up": 126}
ALLOWED_COMMAND_LETTERS = frozenset({"a", "d", "f", "n", "w", "z", ","})
MODIFIERS = {
    "cmd": "command down", "command": "command down",
    "shift": "shift down", "option": "option down", "alt": "option down",
    "ctrl": "control down", "control": "control down",
}

DROP_NO_MATCH = "no_matching_element"
DROP_AMBIGUOUS = "ambiguous_substring_match"
DROP_NTH_MISSING = "nth_match_missing"
DROP_WRONG_APP = "wrong_app"
DROP_ACTION_ERROR = "action_error"
DROP_EMPTY_PAGE = "empty_page"


def norm(text: str) -> str:
    """Case/whitespace-normalised oracle or element text."""
    return re.sub(r"\s+", " ", text).strip().casefold()


def applescript_string(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


@dataclass(frozen=True)
class TargetSpec:
    role: str
    label: str = ""
    description: str = ""
    nth: int | None = None

    @property
    def needle(self) -> str:
        return self.label if self.label else self.description

    @property
    def text_source(self) -> str:
        # This is provenance for automatic item tags: `description` means the oracle intentionally used an
        # accessibility-only name, while `label` means visible text/placeholder/value.
        return "description" if self.description and not self.label else "label"


@dataclass(frozen=True)
class Step:
    action: str
    target: TargetSpec
    text: str = ""
    key: str = ""
    option: str = ""


@dataclass(frozen=True)
class DesktopTask:
    id: str
    app: str
    goal: str
    tags: tuple[str, ...]
    path: tuple[Step, ...]
    prepare: tuple[str, ...] = ()


def parse_target(raw: dict[str, Any], task_id: str) -> TargetSpec:
    if not isinstance(raw, dict) or not raw.get("role"):
        raise ValueError(f"{task_id}: every step needs target.role")
    if raw["role"] not in ROLE_TO_ID:
        raise ValueError(f"{task_id}: unknown target role {raw["role"]!r}")
    label, description = raw.get("label", ""), raw.get("description", "")
    if bool(label) == bool(description):
        raise ValueError(f"{task_id}: target needs exactly one of label or description")
    nth = raw.get("nth")
    if nth is not None and (not isinstance(nth, int) or nth < 1):
        raise ValueError(f"{task_id}: target.nth must be a positive integer")
    return TargetSpec(str(raw["role"]), str(label), str(description), nth)


def parse_task(raw: dict[str, Any]) -> DesktopTask:
    task_id = raw.get("id", "<missing id>")
    path = []
    for raw_step in raw.get("path") or []:
        action = raw_step.get("action")
        if action not in ACTIONS:
            raise ValueError(f"{task_id}: unsupported action {action!r}")
        step = Step(action, parse_target(raw_step.get("target") or {}, task_id),
                    str(raw_step.get("text", "")), str(raw_step.get("key", "")), str(raw_step.get("option", "")))
        if action == TYPE and not step.text:
            raise ValueError(f"{task_id}: type step needs text")
        if action == KEY:
            key_parts(step.key)  # validate the chord while loading, before anything is dumped or clicked
        path.append(step)
    if not 3 <= len(path) <= 10:
        raise ValueError(f"{task_id}: path must have 3-10 actions")
    prepare = tuple(str(p.get("key", "")) for p in raw.get("prepare") or [])
    for key in prepare:
        key_parts(key)
    return DesktopTask(str(raw["id"]), str(raw["app"]), str(raw["goal"]), tuple(raw.get("tags") or []),
                       tuple(path), prepare)


def load_tasks(path: str | Path) -> list[DesktopTask]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a list of tasks")
    return [parse_task(item) for item in raw]


def key_parts(key: str) -> tuple[list[str], str]:
    """Validate and split a safe key chord; returns (AppleScript modifiers, base key)."""
    parts = [p.strip().casefold() for p in key.split("+") if p.strip()]
    if not parts:
        raise ValueError("empty key chord")
    base, modifiers = parts[-1], parts[:-1]
    unknown = [m for m in modifiers if m not in MODIFIERS]
    if unknown or len(set(modifiers)) != len(modifiers):
        raise ValueError(f"unsupported or repeated modifier in {key!r}")
    if base in NAMED_KEY_CODES:
        if modifiers and base not in {"tab", "space", "left", "right", "down", "up"}:
            raise ValueError(f"modifiers are not supported for {base!r} in the recorder allowlist")
    elif len(base) == 1 or base == ",":
        if not modifiers:
            raise ValueError(f"plain printable key {key!r} is not an oracle action; use a type step")
        if any(m in {"cmd", "command"} for m in modifiers) and base not in ALLOWED_COMMAND_LETTERS:
            raise ValueError(f"refusing unsafe or unnecessary command chord {key!r}")
    else:
        raise ValueError(f"unsupported key {key!r}")
    return [MODIFIERS[m] for m in modifiers], base


def key_script(key: str) -> str:
    """One System Events line for an allowlisted key chord."""
    modifiers, base = key_parts(key)
    suffix = ""
    if modifiers:
        suffix = " using {" + ", ".join(modifiers) + "}"
    if base in NAMED_KEY_CODES:
        return f'tell application "System Events" to key code {NAMED_KEY_CODES[base]}{suffix}'
    return f'tell application "System Events" to keystroke {applescript_string(base)}{suffix}'


@dataclass(frozen=True)
class Resolution:
    primary: Element
    target_ids: tuple[int, ...]
    text_source: str
    match_kind: str
    match_count: int


def candidates_for(elements: Iterable[Element], target: TargetSpec) -> tuple[list[Element], str]:
    """Exact role+text matches, or substring matches only when no exact match exists."""
    needle = norm(target.needle)
    same_role = [e for e in elements if e.role == target.role]
    exact = [e for e in same_role if norm(e.text) == needle]
    if exact:
        return exact, "exact"
    return [e for e in same_role if needle in norm(e.text)], "substring"


def resolve_target(elements: list[Element], target: TargetSpec) -> Resolution | str:
    """Resolve an oracle target conservatively; a string return value is the drop reason."""
    candidates, match_kind = candidates_for(elements, target)
    if not candidates:
        return DROP_NO_MATCH
    if target.nth is not None:
        if target.nth > len(candidates):
            return DROP_NTH_MISSING
        selected = [candidates[target.nth - 1]]
    elif match_kind == "substring" and len(candidates) > 1:
        # Exact duplicate labels are interchangeable and therefore safe; multiple different substring
        # containers are not. Never pick the visually first one merely because it is first.
        return DROP_AMBIGUOUS
    else:
        selected = candidates
    ids: set[int] = set()
    for element in selected:
        ids.add(element.id)
        ids.update(equiv_ids(elements, element.id))
    return Resolution(selected[0], tuple(sorted(ids)), target.text_source, match_kind, len(candidates))


class SystemEventsDriver:
    """Small injectable macOS effect layer; tests replace every method with a fake."""

    def _run(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(command, capture_output=True, text=True, timeout=COMMAND_TIMEOUT)

    def _check(self, command: list[str]) -> str:
        proc = self._run(command)
        if proc.returncode != 0:
            raise RuntimeError(f"{command[0]} failed: {proc.stderr.strip() or proc.returncode}")
        return proc.stdout.strip()

    def _osascript(self, *lines: str) -> str:
        return self._check(["osascript", *(arg for line in lines for arg in ("-e", line))])

    def is_running(self, app: str) -> bool:
        return self._osascript(f"application {applescript_string(app)} is running") == "true"

    def launch(self, app: str) -> None:
        self._check(["open", "-a", app])

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

    def press_key(self, key: str) -> None:
        self._osascript(key_script(key))

    def type_text(self, text: str) -> None:
        lines = text.split("\n")
        for index, chunk in enumerate(lines):
            if chunk:
                self._osascript(f'tell application "System Events" to keystroke {applescript_string(chunk)}')
            if index + 1 < len(lines):
                self.press_key("return")

    def click_at(self, app: str, x: float, y: float) -> None:
        self._osascript(
            'tell application "System Events"',
            f'tell process {applescript_string(app)}',
            f'click at {{{round(x)}, {round(y)}}}',
            'end tell',
        )

    def quit(self, app: str) -> None:
        self._osascript(f"quit app {applescript_string(app)}")


def ensure_frontmost(driver: SystemEventsDriver, app: str) -> None:
    """Refuse keyboard/coordinate input unless the intended app is still frontmost."""
    actual = driver.frontmost()
    if norm(actual) != norm(app):
        raise RuntimeError(f"{app} is not frontmost (frontmost is {actual})")


def frontmost_matches(driver: SystemEventsDriver, app: str) -> bool:
    try:
        return norm(driver.frontmost()) == norm(app)
    except Exception:
        return False


def click_point(element: Element) -> tuple[float, float]:
    x, y, w, h = element.abs_box
    return x + w / 2, y + h / 2


def perform_action(driver: SystemEventsDriver, app: str, step: Step, resolution: Resolution) -> None:
    """Perform one resolved oracle step through System Events after a frontmost-app check."""
    ensure_frontmost(driver, app)
    x, y = click_point(resolution.primary)
    if step.action in {CLICK, SELECT}:
        driver.click_at(app, x, y)
        return
    driver.click_at(app, x, y)  # type/key act in the resolved control, so focus it first
    if step.action == TYPE:
        driver.type_text(step.text)
    elif step.action == KEY:
        driver.press_key(step.key)
    else:
        raise ValueError(f"unsupported action {step.action!r}")


def history_entry(step: Step, element: Element) -> str:
    if step.action == TYPE:
        return action_repr(element, step.text)
    if step.action == KEY:
        return f"[{element.tag or element.role}]  {element.text} -> KEY: {step.key}"
    return action_repr(element)


@dataclass
class Drop:
    task: str
    step_index: int
    action: str
    reason: str
    target: str


@dataclass
class TaggedExample:
    example: Example
    tags: set[str]


@dataclass
class TaskResult:
    examples: list[TaggedExample] = field(default_factory=list)
    drops: list[Drop] = field(default_factory=list)
    not_attempted: int = 0
    pages: dict[str, Page] = field(default_factory=dict)


@dataclass
class RecordResult:
    examples: list[TaggedExample]
    drops: list[Drop]
    counts: Counter[str]
    tag_counts: Counter[str]
    pages: dict[str, Page]
    per_task: dict[str, dict[str, int]]


DumpApp = Callable[[str], dict[str, Any]]
Sleep = Callable[[float], None]


class DesktopRecorder:
    """Records one frozen oracle task at a time with injectable dump/effect layers."""

    def __init__(
        self, driver: SystemEventsDriver, dump_app: DumpApp, settle: float = DEFAULT_SETTLE,
        sleep: Sleep = time.sleep,
    ) -> None:
        self.driver, self.dump_app, self.settle, self.sleep = driver, dump_app, settle, sleep

    def page_from_raw(self, raw: dict[str, Any], app: str) -> Page:
        site = slugify(raw.get("app") or app)
        page = ax.page_from_dump(raw, f"{site}-pending", site)
        page.page_id = page_id_for(site, page)
        return page

    def record_task(self, task: DesktopTask) -> TaskResult:
        """Run one task. If a step cannot be resolved/performed, later steps are not attempted."""
        result = TaskResult()
        was_running = self.driver.is_running(task.app)
        self.driver.launch(task.app)
        self.sleep(self.settle)
        initial_windows = self.driver.window_count(task.app)
        try:
            try:
                ensure_frontmost(self.driver, task.app)
            except RuntimeError as exc:
                first = task.path[0]
                result.drops.append(Drop(task.id, 0, first.action, DROP_WRONG_APP, str(exc)))
                result.not_attempted = len(task.path) - 1
                return result
            for key in task.prepare:
                ensure_frontmost(self.driver, task.app)
                self.driver.press_key(key)
                self.sleep(UNDO_PAUSE)
            history: list[str] = []
            for index, step in enumerate(task.path):
                raw = self.dump_app(task.app)
                raw_app = raw.get("app") or ""
                if norm(raw_app) != norm(task.app):
                    result.drops.append(Drop(task.id, index, step.action, DROP_WRONG_APP,
                                             f"front window belongs to {raw_app}"))
                    result.not_attempted = len(task.path) - index - 1
                    break
                page = self.page_from_raw(raw, task.app)
                if not page.elements:
                    result.drops.append(Drop(task.id, index, step.action, DROP_EMPTY_PAGE, step.target.needle))
                    result.not_attempted = len(task.path) - index - 1
                    break
                resolved = resolve_target(page.elements, step.target)
                if isinstance(resolved, str):
                    result.drops.append(Drop(task.id, index, step.action, resolved, step.target.needle))
                    result.not_attempted = len(task.path) - index - 1
                    break
                try:
                    perform_action(self.driver, task.app, step, resolved)
                except Exception as exc:  # the effect layer boundary reports one concise reason
                    result.drops.append(Drop(task.id, index, step.action, DROP_ACTION_ERROR, str(exc)))
                    result.not_attempted = len(task.path) - index - 1
                    break
                tags = tag_target(resolved.primary, task.goal, history, resolved.text_source)
                example = Example(page_id=page.page_id, kind="goal", mode="goal", query=task.goal,
                                  history=list(history), target_id=resolved.primary.id,
                                  equiv_ids=list(resolved.target_ids), source=task.id)
                result.examples.append(TaggedExample(example, tags))
                result.pages[page.page_id] = page
                history.append(history_entry(step, resolved.primary))
                self.sleep(self.settle)
        finally:
            self._cleanup(task, initial_windows, was_running)
        return result

    def _cleanup(self, task: DesktopTask, initial_windows: int, was_running: bool) -> None:
        """Close the throwaway document or a window this task added; never save it."""
        app = task.app
        close_window = "throwaway-document" in task.tags
        try:
            close_window = close_window or self.driver.window_count(app) > initial_windows
            if close_window and frontmost_matches(self.driver, app):
                self.driver.press_key("cmd+w")
                self.sleep(UNDO_PAUSE)
                if frontmost_matches(self.driver, app) and self.driver.has_sheet(app):
                    self.driver.press_key("cmd+d")  # Don't Save in a close-document sheet
                    self.sleep(UNDO_PAUSE)
                    if frontmost_matches(self.driver, app) and self.driver.has_sheet(app):
                        self.driver.press_key("escape")
                        self.sleep(UNDO_PAUSE)
        finally:
            if not was_running:
                self.driver.quit(app)


def dedupe_examples(rows: Iterable[TaggedExample]) -> tuple[list[TaggedExample], int]:
    """Remove exact duplicate next-step examples while keeping repeated states with different histories."""
    out, seen = [], set()
    duplicates = 0
    for row in rows:
        e = row.example
        ident = (e.source, e.page_id, tuple(e.equiv_ids), tuple(e.history), e.target_id)
        if ident in seen:
            duplicates += 1
            continue
        seen.add(ident)
        out.append(row)
    return out, duplicates


def collect_results(task_results: dict[str, TaskResult]) -> RecordResult:
    examples = [row for result in task_results.values() for row in result.examples]
    examples, duplicates = dedupe_examples(examples)
    drops = [drop for result in task_results.values() for drop in result.drops]
    counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()
    pages: dict[str, Page] = {}
    per_task: dict[str, dict[str, int]] = {}
    for task_id, result in task_results.items():
        for row in result.examples:
            counts["examples_raw"] += 1
        counts["dropped_steps"] += len(result.drops)
        counts["not_attempted_after_drop"] += result.not_attempted
        for drop in result.drops:
            counts[f"drop:{drop.reason}"] += 1
        pages.update(result.pages)
        kept = sum(1 for row in examples if row.example.source == task_id)
        per_task[task_id] = {"examples": kept, "dropped": len(result.drops), "not_attempted": result.not_attempted}
    counts["duplicates"] = duplicates
    needed = {row.example.page_id for row in examples}
    pages = {pid: page for pid, page in pages.items() if pid in needed}
    counts["examples"] = len(examples)
    counts["pages"] = len(pages)
    for row in examples:
        tag_counts.update(row.tags)
    return RecordResult(examples, drops, counts, tag_counts, pages, per_task)


def tagged_example_dict(row: TaggedExample) -> dict[str, Any]:
    """Example serialization plus a compatible extra `tags` array (ignored by Example.from_dict)."""
    data = row.example.to_dict()
    data["tags"] = sorted(row.tags)
    return data


def write_tagged_examples(rows: Iterable[TaggedExample], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(tagged_example_dict(row), sort_keys=True) + "\n")


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def file_entry(path: str | Path, root: Path = ROOT) -> dict[str, Any]:
    path = Path(path)
    try:
        display = str(path.relative_to(root))
    except ValueError:
        display = str(path)
    return {"path": display, "sha256": sha256_file(path), "bytes": path.stat().st_size}


def write_build_report(result: RecordResult | None, path: str | Path, status: str, message: str = "") -> Path:
    report = {
        "status": status,
        "message": message,
        "action_method": "System Events process-scoped coordinate clicks and allowlisted keyboard input",
        "counts": dict(result.counts) if result else {},
        "tag_counts": dict(result.tag_counts) if result else {},
        "per_task": result.per_task if result else {},
        "drops": [drop.__dict__ for drop in result.drops] if result else [],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_manifest(
    out_dir: str | Path, name: str, tasks_file: str | Path, tasks_count: int, status: str,
    output_files: Iterable[str | Path], counts: dict[str, int] | None = None, message: str = "",
) -> Path:
    """Freeze output bytes with SHA-256; also used for an explicit zero-output permission failure."""
    out_dir = Path(out_dir)
    outputs = [file_entry(path) for path in sorted((Path(p) for p in output_files), key=str)]
    manifest = {
        "set": name,
        "status": status,
        "message": message,
        "tasks_file": file_entry(tasks_file) | {"tasks": tasks_count},
        "counts": counts or {},
        "outputs": outputs,
    }
    path = out_dir / f"{name}.manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def record_and_write(
    tasks: list[DesktopTask], recorder: DesktopRecorder, out_dir: str | Path, name: str,
    tasks_file: str | Path,
) -> tuple[RecordResult, Path]:
    """Record all tasks, write pages/examples/report, and freeze every output in the manifest."""
    out_dir = Path(out_dir)
    pages_dir = out_dir / f"pages_{name}"
    task_results: dict[str, TaskResult] = {}
    for task in tasks:
        print(f"[{task.id}] {task.app}: {len(task.path)} steps", flush=True)
        task_results[task.id] = recorder.record_task(task)
    result = collect_results(task_results)
    written_pages = [write_page(page, pages_dir) for page in result.pages.values()]
    examples_path = out_dir / f"{name}.jsonl"
    write_tagged_examples(result.examples, examples_path)
    report_path = write_build_report(result, out_dir / f"{name}-build.json", "recorded")
    manifest = write_manifest(out_dir, name, tasks_file, len(tasks), "recorded",
                              [examples_path, report_path, *written_pages], dict(result.counts))
    return result, manifest


def dry_run(tasks: list[DesktopTask]) -> str:
    lines = [f"{len(tasks)} tasks on {len({t.app for t in tasks})} apps"]
    for task in tasks:
        actions = ", ".join(step.action for step in task.path)
        lines.append(f"{task.id}: {task.app}, {len(task.path)} steps ({actions})")
    return "\n".join(lines)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--binary", type=Path, default=ax.AXDUMP)
    parser.add_argument("--settle", type=float, default=DEFAULT_SETTLE)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true", help="validate and print the frozen plan only")
    args = parser.parse_args(argv)
    tasks = load_tasks(args.tasks)
    if args.dry_run:
        print(dry_run(tasks))
        return 0

    def dump(app: str) -> dict[str, Any]:
        return ax.run_axdump(["--app", app, "--timeout", str(args.timeout)], binary=args.binary,
                             timeout=args.timeout + 5)

    recorder = DesktopRecorder(SystemEventsDriver(), dump, settle=args.settle)
    try:
        result, manifest = record_and_write(tasks, recorder, args.out, args.name, args.tasks)
    except ax.AxdumpError as exc:
        # A missing Accessibility permission cannot be fixed by retrying. Persist an explicit zero-output
        # manifest so nobody mistakes the absent files for a frozen eval set.
        status = "not_recorded_permission_missing" if exc.returncode == 2 else "not_recorded_axdump_error"
        report = write_build_report(None, args.out / f"{args.name}-build.json", status, str(exc))
        manifest = write_manifest(args.out, args.name, args.tasks, len(tasks), status, [report],
                                  message=str(exc))
        print(f"axdump failed: {exc}\nwrote {manifest}", file=sys.stderr)
        return 2 if exc.returncode == 2 else 3
    print(f"wrote {len(result.examples)} examples, {len(result.pages)} pages; manifest {manifest}")
    print(f"tags: {dict(result.tag_counts)}; drops: {dict(result.counts)}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
