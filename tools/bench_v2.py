"""Freeze and verify the agent benchmark v2 snapshots (plan v2.1, step 7).

    uv run python -m tools.bench_v2 record [--tasks-file demo/tasks_v2.yaml] [--ids a b] [--branches 6]
    uv run python -m tools.bench_v2 verify [--tasks-file demo/tasks_v2.yaml] [--ids a b]
    uv run python -m tools.bench_v2 manifest

Each task in the tasks file carries, besides what `demo.browser.Task` reads, the scripted solution used here:

    path:   [step, ...]      the correct route from start_url (record: drives the live site; verify: replays it)
    answer: "..."            what a correct done(answer) says (verify matches it against answer_regex, and checks that
                             the regex also matches the text of the page the path ends on; `no_evidence: true` skips that)
    wrong:  [[step, ...]]    optional extra routes (plausible wrong branches) to record

A step is one of (locators: role+name, text, css, label; `nth` picks the n-th match, default first):

    {goto: URL}
    {click: {role: link, name: "Docs"}}        {click: {text: "..."}}  {click: {css: "..."}}  {click: {label: "Search"}}
    {type: {role: searchbox, name: "Search"}, text: "turing", enter: true}
    {press: "Enter"}        {wait: 1000}        {scroll: bottom}
    {click: {...}, external: true}             a click that leaves the site: skipped when recording (the target is not
                                               archived), executed on replay, where the stub page still has its URL

`record` drives every path with Playwright's HAR recording (`BrowserSession(record_har=...)`), scrolls each page so lazy
content is archived, then visits the first few same-site links of the start page (`--branches`) so a plausible wrong
click replays too. `manifest` writes sha256 of the task file and every snapshot (data/bench_v2/manifest.json). `sanitize` removes failed-request entries from existing HARs (record does it already). `verify` replays offline (dead proxy, no network): the do-nothing run must FAIL the task's check and
the scripted path must pass it (and its answer match `answer_regex`). Results are written (merged) to
data/bench_v2/verify.json, or to `--results FILE`.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urldefrag, urlsplit

import yaml

from demo.browser import BrowserSession, Task, ToolError, answer_matches, load_tasks, parse_tasks
from grounder.data.crawl import registrable_domain

ROOT = Path(__file__).resolve().parent.parent
TASKS_V2 = ROOT / "demo" / "tasks_v2.yaml"
HAR_DIR = ROOT / "data" / "bench_v2" / "har"
VERIFY_FILE = ROOT / "data" / "bench_v2" / "verify.json"
STEP_SETTLE_MS = 900
SCROLL_STEPS = 12


def locate(page: Any, spec: dict[str, Any]) -> Any:
    nth = int(spec.get("nth", 0))
    if "role" in spec:
        kw = {"name": spec["name"], "exact": bool(spec.get("exact", False))} if "name" in spec else {}
        loc = page.get_by_role(spec["role"], **kw)
    elif "text" in spec:
        loc = page.get_by_text(spec["text"], exact=bool(spec.get("exact", False)))
    elif "label" in spec:
        loc = page.get_by_label(spec["label"], exact=bool(spec.get("exact", False)))
    elif "css" in spec:
        loc = page.locator(spec["css"])
    else:
        raise ValueError(f"locator needs role/text/label/css: {spec}")
    return loc.nth(nth)


def scroll_through(page: Any) -> None:
    """Scroll to the bottom in steps (lazy images / sections get requested, hence archived), then back to the top."""
    try:
        height = page.evaluate("document.documentElement.scrollHeight")
        for i in range(1, SCROLL_STEPS + 1):
            page.mouse.wheel(0, height / SCROLL_STEPS)
            page.wait_for_timeout(120)
        page.wait_for_timeout(400)
        page.evaluate("window.scrollTo(0, 0)")
    except Exception:
        pass


def settle(page: Any) -> None:
    try:
        page.wait_for_load_state("load", timeout=10_000)
    except Exception:
        pass
    page.wait_for_timeout(STEP_SETTLE_MS)


def run_steps(page: Any, steps: list[dict[str, Any]], recording: bool, scroll: bool) -> None:
    for step in steps:
        if recording and step.get("external"):
            continue
        if "goto" in step:
            page.goto(step["goto"], wait_until="load")
        elif "click" in step:
            locate(page, step["click"]).click(timeout=10_000)
        elif "type" in step:
            loc = locate(page, step["type"])
            loc.click(timeout=10_000)
            page.keyboard.press("ControlOrMeta+A")
            page.keyboard.type(step["text"], delay=20)
            if step.get("enter"):
                page.keyboard.press("Enter")
        elif "press" in step:
            page.keyboard.press(step["press"])
        elif "wait" in step:
            page.wait_for_timeout(int(step["wait"]))
        elif "scroll" in step:
            scroll_through(page)
        else:
            raise ValueError(f"unknown step {step}")
        settle(page)
        if scroll:
            scroll_through(page)


def same_site_links(page: Any, limit: int) -> list[str]:
    """First `limit` distinct same-site page URLs linked from the current page (no fragments, no files)."""
    host = registrable_domain(page.url)
    hrefs = page.evaluate("[...document.querySelectorAll('a[href]')].map(a => a.href)")
    seen: list[str] = []
    for h in hrefs:
        h = urldefrag(h)[0]
        if not h.startswith("http") or registrable_domain(h) != host or h == urldefrag(page.url)[0] or h in seen:
            continue
        if re.search(r"\.(pdf|zip|png|jpe?g|svg|gif|mp4|xml|json|tar\.gz|dmg|exe)(\?|$)", urlsplit(h).path, re.I):
            continue
        seen.append(h)
        if len(seen) >= limit:
            break
    return seen


def load_raw(path: Path) -> list[dict[str, Any]]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def sanitize_har(path: Path) -> int:
    """Drop entries of a recorded HAR zip whose request failed (status <= 0, e.g. CORS-blocked fonts): Playwright's replay
    serves them with status -1 and the page's load event then never fires. Returns how many entries were dropped; they
    fall through to the replay's miss handler instead (aborted at once)."""
    import shutil
    import zipfile

    tmp = path.with_suffix(".tmp")
    dropped = 0
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename.endswith(".har"):
                har = json.loads(data)
                entries = har["log"]["entries"]
                keep = [e for e in entries if (e.get("response") or {}).get("status", 0) > 0]
                dropped = len(entries) - len(keep)
                har["log"]["entries"] = keep
                data = json.dumps(har).encode("utf-8")
            zout.writestr(item, data)
    shutil.move(tmp, path)
    return dropped


def write_manifest(tasks: list[Task], tasks_file: Path) -> Path:
    """data/bench_v2/manifest.json: sha256 of the task file and of every snapshot, so 'frozen' is checkable."""
    import hashlib

    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    out = HAR_DIR.parent / "manifest.json"
    manifest = {"tasks_file": {"path": str(tasks_file.relative_to(ROOT)), "sha256": sha(tasks_file), "tasks": len(tasks)},
                "snapshots": {t.id: {"sha256": sha(har_path(t)), "bytes": har_path(t).stat().st_size}
                              for t in tasks if har_path(t).exists()}}
    out.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return out


def har_path(task: Task) -> Path:
    return ROOT / task.har if task.har else HAR_DIR / f"{task.id}.zip"


def record_task(task: Task, raw: dict[str, Any], branches: int) -> Path:
    out = har_path(task)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    with BrowserSession(task.start_url, record_har=out) as s:
        page = s.page
        scroll_through(page)
        extra = same_site_links(page, branches)
        run_steps(page, raw.get("path", []), recording=True, scroll=True)
        for wrong in raw.get("wrong", []):
            page.goto(task.start_url, wait_until="load")
            run_steps(page, wrong, recording=True, scroll=True)
        for url in extra:  # plausible wrong clicks from the start page
            try:
                page.goto(url, wait_until="load", timeout=20_000)
                settle(page)
                scroll_through(page)
            except Exception:
                pass
    sanitize_har(out)
    return out


def verify_task(task: Task, raw: dict[str, Any]) -> dict[str, Any]:
    har = har_path(task)
    res: dict[str, Any] = {"id": task.id, "har_mb": round(har.stat().st_size / 1e6, 2) if har.exists() else None}
    if not har.exists():
        return {**res, "ok": False, "why": "no snapshot"}
    try:
        with BrowserSession(task.start_url, har=har) as s:
            res["do_nothing_fails"] = not (s.succeeded(task.check) and answer_matches(task.answer_regex, ""))
        with BrowserSession(task.start_url, har=har) as s:
            run_steps(s.page, raw.get("path", []), recording=False, scroll=False)
            res["check_ok"] = s.succeeded(task.check)
            res["answer_ok"] = answer_matches(task.answer_regex, str(raw.get("answer", "")))
            # the regex must be TRUE of the frozen page the agent ends on, not just of our scripted answer
            res["evidence_ok"] = (not task.answer_regex or bool(raw.get("no_evidence"))
                                  or re.search(task.answer_regex, s.text(), re.I) is not None)
            res["final_url"] = s.url
            res["misses"] = len(s.replay.misses) if s.replay else 0
            res["network_failures"] = len(s.replay.failed) if s.replay else 0
            res["miss_urls"] = (s.replay.misses[:3] if s.replay else [])
    except (ToolError, Exception) as exc:  # noqa: BLE001 - a broken path is a verdict, not a crash
        return {**res, "ok": False, "why": f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"}
    res["ok"] = bool(res.get("do_nothing_fails") and res.get("check_ok") and res.get("answer_ok") and res.get("evidence_ok"))
    return res


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["record", "verify", "sanitize", "manifest"])
    p.add_argument("--tasks-file", type=Path, default=TASKS_V2)
    p.add_argument("--ids", nargs="+")
    p.add_argument("--results", type=Path, default=VERIFY_FILE, help="verify: results file (merged with what it holds)")
    p.add_argument("--branches", type=int, default=6, help="record: same-site links of the start page to visit as wrong branches")
    args = p.parse_args(argv)
    raws = {r["id"]: r for r in load_raw(args.tasks_file)}
    tasks = [t for t in parse_tasks(list(raws.values())) if not args.ids or t.id in args.ids]
    if args.command == "manifest":
        print(f"wrote {write_manifest(tasks, args.tasks_file)}")
        return 0
    results: dict[str, Any] = {}
    if args.results.exists() and args.command == "verify":
        results = json.loads(args.results.read_text(encoding="utf-8"))
    bad = 0
    for t in tasks:
        if args.command == "sanitize":
            n = sanitize_har(har_path(t)) if har_path(t).exists() else -1
            print(f"{t.id}: dropped {n} failed entries")
        elif args.command == "record":
            try:
                out = record_task(t, raws[t.id], args.branches)
                print(f"recorded {t.id}: {out.stat().st_size / 1e6:.1f} MB", flush=True)
            except Exception as exc:  # noqa: BLE001
                bad += 1
                print(f"FAILED to record {t.id}: {type(exc).__name__}: {str(exc).splitlines()[0][:200]}", flush=True)
        else:
            r = verify_task(t, raws[t.id])
            results[t.id] = r
            bad += not r["ok"]
            print(("ok   " if r["ok"] else "FAIL ") + json.dumps(r, ensure_ascii=False), flush=True)
    if args.command == "verify":
        args.results.parent.mkdir(parents=True, exist_ok=True)
        args.results.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(tasks) - bad}/{len(tasks)} ok")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
