"""Extract System-1 decision labels from benchmark agent logs.

The input runs live outside this repository because they are large and gitignored. The extractor reads them,
replays the frozen HAR archives into `Page` files, and writes one JSONL split per decision:

    uv run --extra demo --with tldextract python tools/decision_labels.py extract \
      --source-root /path/to/tiny-grounder --out data/decisions --workers 4

Only bench v2/v3 run directories are used: v4/v5 archives are absent on the data machine, cli runs used live pages,
and desktop tasks are not web pages. SHOW and NEXT reuse tools.next_step_recall.build_bench. ENOUGH, WORKED and DONE
come from the state machine in this file, which keeps the complete tool-call sequence rather than only click/type rows.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import zipfile
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grounder.data.pages import page_paths, write_page
from grounder.records import Element, action_repr
from tools.harvest_queries import parse_action, read_run
from tools.next_step_recall import build_bench

ACTION_TOOLS = {"click", "type", "fill", "go_back", "click_xy", "type_text", "scroll"}
OBSERVATION_TOOLS = {"snapshot", "observe", "find", "inspect", "read", "screenshot"}
TEXT_RECHECK_TOOLS = {"snapshot", "observe", "find", "inspect", "read"}
COMMITMENT_TOOLS = ACTION_TOOLS | {"done"}

SOURCE_TASKS = {
    "bench-v2": "demo/tasks_v2.yaml",
    "bench-v2-smoke": "demo/tasks_v2.yaml",
    "bench-v3": "demo/tasks_milestone.yaml",
    "bench-v3-pilot": "demo/tasks_milestone.yaml",
    "bench-v3-pilot2": "demo/tasks_milestone.yaml",
}
FROZEN_TEST_SITES = {
    "saucedemo.com", "demoblaze.com", "tutorialsninja.com", "the-internet.herokuapp.com", "blazedemo.com",
}
SPLIT_RATIOS = {"train": 0.70, "dev": 0.15, "test": 0.15}

RULES = {
    "show": "show.next_action_success_v1",
    "next": "next.click_type_success_v1",
    "enough": "enough.screenshot_or_failure_v1",
    "worked": "worked.error_recheck_retry_v1",
    "done": "done.successful_done_next_v1",
}
DECISIONS = ("show", "next", "enough", "worked", "done")

_TAGS = {"a": "link", "input": "textbox", "textarea": "textbox", "select": "combobox"}


def run_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        row["agent"].replace(":", "-"),
        re.sub(r"[^a-zA-Z0-9_-]", "-", row.get("arm", row["config"])),
        row["task_id"],
        str(row["repeat"]),
    )


def scan_states(events: list[dict[str, Any]], start_url: str) -> list[dict[str, Any]]:
    """Group a server log into page states separated by committed browser actions.

    A state starts at the initial page or after an action attempt. Observation calls inside the state do not start a
    new state. The first action or `done` call is the state's commitment. Later calls belong to the next state.
    """
    states: list[dict[str, Any]] = [{
        "step": 0, "url": start_url, "history": [], "events": [], "previous_action": None,
    }]
    url, history, action_ordinal = start_url, [], 0
    for order, ev in enumerate(events):
        before = url
        name = ev.get("name", "")
        result = ev.get("result") or ""
        urls = re.findall(r"^url:\s*(\S+)", result, re.M)
        after = urls[-1] if urls else before
        info = {
            "event": ev, "order": order, "name": name, "args": ev.get("args", {}),
            "is_error": bool(ev.get("is_error")), "url_before": before, "url_after": after,
        }
        states[-1]["events"].append(info)
        url = after
        if name not in ACTION_TOOLS:
            states[-1]["url"] = url
            continue
        action_ordinal += 1
        previous = {
            "name": name, "args": ev.get("args", {}), "is_error": bool(ev.get("is_error")),
            "url_before": before, "url_after": url, "call_n": ev.get("n"),
            "image_bytes": int(ev.get("image_bytes") or 0), "result_first_line": result.splitlines()[0] if result else "",
        }
        parsed = None if ev.get("is_error") else parse_action(result)
        if name in ("click", "type") and parsed:
            verb, role, text = parsed
            element = Element(
                0, _TAGS.get(role, role), text,
                tag=role if role in _TAGS else {"link": "a", "textbox": "input", "searchbox": "input"}.get(role, ""),
            )
            history.append(action_repr(element, ev.get("args", {}).get("text", "") if verb == "type" else None))
        elif not ev.get("is_error") and name in ("click_xy", "type_text"):
            history.append("[unknown]  -> " + ("TYPE: " + ev.get("args", {}).get("text", "")
                                               if name == "type_text" else "CLICK"))
        states.append({
            "step": action_ordinal, "url": url, "history": list(history), "events": [], "previous_action": previous,
        })
    return states


def commitment(state: dict[str, Any]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """The first action/done call in a state, plus observation calls made before it."""
    observations = []
    for info in state["events"]:
        if info["name"] in COMMITMENT_TOOLS:
            return info, observations
        if info["name"] in OBSERVATION_TOOLS:
            observations.append(info)
    return None, observations


def same_target_retry(previous: dict[str, Any], next_call: dict[str, Any]) -> bool:
    """Whether the next committed call repeats the same action on the same target."""
    if previous["name"] != next_call["name"]:
        return False
    a, b = previous["args"], next_call["args"]
    if "id" in a or "id" in b:
        return a.get("id") == b.get("id")
    if previous["name"] == "click_xy":
        return (abs(float(a.get("x", -1.0)) - float(b.get("x", -2.0))) <= 2.0
                and abs(float(a.get("y", -1.0)) - float(b.get("y", -2.0))) <= 2.0)
    if previous["name"] == "fill":
        ids = lambda fields: sorted(str(f.get("id")) for f in fields if isinstance(f, dict))
        return ids(a.get("fields", [])) == ids(b.get("fields", [])) and bool(ids(a.get("fields", [])))
    return False


def done_label(next_call: dict[str, Any], run_success: bool) -> tuple[bool, str]:
    positive = next_call["name"] == "done" and run_success
    if positive:
        return True, "done_called_and_run_successful"
    if next_call["name"] == "done":
        return False, "done_called_but_run_failed"
    return False, "agent_committed_to_more_work"


def enough_label(observations: list[dict[str, Any]], previous_action: dict[str, Any] | None,
                 run_success: bool) -> tuple[bool, str]:
    explicit = any(o["name"] == "screenshot" for o in observations)
    automatic = bool(previous_action and previous_action.get("image_bytes"))
    if explicit:
        return True, "screenshot_before_commitment"
    if automatic:
        return True, "vision_observation_returned_by_previous_action"
    if run_success:
        return False, "succeeded_without_screenshot"
    return True, "failed_without_screenshot"


def worked_label(previous_action: dict[str, Any] | None, next_call: dict[str, Any] | None,
                 observations: list[dict[str, Any]]) -> tuple[bool, str] | None:
    if previous_action is None or next_call is None:
        return None
    if previous_action["is_error"]:
        return False, "tool_error"
    if any(o["name"] in TEXT_RECHECK_TOOLS or o["name"] == "screenshot" for o in observations):
        return False, "immediate_recheck"
    if same_target_retry(previous_action, next_call):
        return False, "immediate_retry_same_target"
    return True, "proceeded"


def page_id_for(har: str, task_id: str, url: str) -> str:
    return "next-" + hashlib.sha256(f"{har}\0{task_id}\0{url}".encode()).hexdigest()[:20]


def state_records(run: dict[str, Any], run_dir: str, task: Any, events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter]:
    """Decision states of one run before page availability is known."""
    drops: Counter = Counter()
    records = []
    arm = run.get("arm", run.get("config", ""))
    site = run.get("site", "")
    if site in FROZEN_TEST_SITES:
        drops["frozen_test_site"] += 1
        return [], drops
    har = getattr(task, "har", "") or f"data/bench_v2/har/{task.id}.zip"
    for state in scan_states(events, task.start_url):
        next_call, observations = commitment(state)
        if next_call is None:
            drops["no_committed_call"] += 1
            continue
        url = next_call["url_before"] or state["url"]
        if not url.startswith(("http://", "https://")):
            drops["bad_url"] += 1
            continue
        previous = state["previous_action"]
        enough, enough_reason = enough_label(observations, previous, bool(run.get("success")))
        done, done_reason = done_label(next_call, bool(run.get("success")))
        worked = worked_label(previous, next_call, observations)
        record = {
            "goal": run.get("goal") or task.goal,
            "history": state["history"],
            "url": url,
            "page_id": page_id_for(har, task.id, url),
            "har": har,
            "task_id": task.id,
            "site": site,
            "run_dir": run_dir,
            "run_id": "__".join(run_key(run)),
            "step": state["step"],
            "call_n": next_call["event"].get("n"),
            "agent": run.get("agent", ""),
            "arm": arm,
            "run_success": bool(run.get("success")),
            "next_call": next_call["name"],
            "observations": observations,
            "previous_action": previous,
            "labels": {"enough": (enough, enough_reason), "done": (done, done_reason), "worked": worked},
        }
        records.append(record)
    return records, drops


def load_source_tasks() -> dict[str, dict[str, Any]]:
    """Tasks by run directory. v2 task ids recur in the milestone file with different HAR snapshots."""
    from demo.browser import load_tasks

    by_file = {}
    for tasks_file in sorted(set(SOURCE_TASKS.values())):
        by_file[tasks_file] = {task.id: task for task in load_tasks(ROOT / tasks_file)}
    return {run_dir: by_file[tasks_file] for run_dir, tasks_file in SOURCE_TASKS.items()}


def collect_states(source_root: Path) -> tuple[list[dict[str, Any]], Counter, Counter]:
    tasks_by_source = load_source_tasks()
    records: list[dict[str, Any]] = []
    drops: Counter = Counter()
    run_counts: Counter = Counter()
    for run_dir in SOURCE_TASKS:
        rows_path = source_root / "runs" / run_dir / "runs.jsonl"
        logs_dir = source_root / "runs" / run_dir / "logs"
        if not rows_path.exists() or not logs_dir.exists():
            drops[f"missing_source:{run_dir}"] += 1
            continue
        rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
        runs = {run_key(row): row for row in rows}
        run_counts[f"runs:{run_dir}"] = len(rows)
        for path in sorted(logs_dir.glob("*.server.jsonl")):
            key = tuple(path.name.removesuffix(".server.jsonl").split("__"))
            run = runs.get(key)
            if run is None:
                drops["log_without_run_row"] += 1
                continue
            task = tasks_by_source[run_dir].get(run["task_id"])
            if task is None:
                drops["unknown_task"] += 1
                continue
            events, _summary = read_run(str(path))
            run_counts[f"logs:{run_dir}"] += 1
            recs, run_drops = state_records(run, run_dir, task, events)
            records.extend(recs)
            drops.update(run_drops)
    return records, drops, run_counts


def cached_page_ids(pages_dir: Path) -> set[str]:
    if not pages_dir.exists():
        return set()
    return {p.name.removesuffix(".json.gz") for p in page_paths(pages_dir)}


def archived_request_urls(har_path: Path) -> set[str]:
    with zipfile.ZipFile(har_path) as archive:
        data = json.loads(archive.read("har.har"))
    return {entry["request"]["url"].split("#", 1)[0] for entry in data["log"]["entries"]}


def _extract_task_pages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Replay one task archive and write its missing pages. Runs in a worker process."""
    from demo.browser import BrowserSession

    source_root = Path(payload["source_root"])
    pages_dir = Path(payload["pages_dir"])
    har_path = source_root / payload["har"]
    items = payload["items"]
    results = []
    if not items:
        return results
    try:
        with BrowserSession(items[0]["url"], har=har_path) as browser:
            for item in items:
                rec = {"page_id": item["page_id"], "url": item["url"], "task_id": item["task_id"], "ok": False, "error": ""}
                try:
                    if browser.url != item["url"]:
                        browser._guard(lambda: browser.page.goto(item["url"], wait_until="load"))
                        browser._settle()
                    page = browser.extract()
                    if not page.elements or page.title.lower() == "page not archived":
                        raise ValueError("empty/unarchived page")
                    page.page_id, page.site = item["page_id"], item["site"]
                    write_page(page, pages_dir)
                    rec["ok"] = True
                except Exception as exc:  # noqa: BLE001 - one bad page must not stop the batch
                    rec["error"] = f"{type(exc).__name__}: {exc}"[:300]
                results.append(rec)
    except Exception as exc:  # noqa: BLE001 - browser startup failure applies to the whole task batch
        for item in items:
            results.append({"page_id": item["page_id"], "url": item["url"], "task_id": item["task_id"],
                            "ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]})
    return results


def extract_pages(records: list[dict[str, Any]], source_root: Path, pages_dir: Path,
                  workers: int = 4) -> tuple[set[str], list[dict[str, Any]], Counter]:
    """Populate `pages_dir` with every archived state page; return cached ids, failures and drop counts."""
    pages_dir.mkdir(parents=True, exist_ok=True)
    cached = cached_page_ids(pages_dir)
    refs: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    unique: dict[str, dict[str, Any]] = {}
    drops: Counter = Counter()
    for rec in records:
        unique.setdefault(rec["page_id"], {
            "page_id": rec["page_id"], "url": rec["url"], "task_id": rec["task_id"],
            "site": rec["site"], "har": rec["har"],
        })
    archive_urls: dict[str, set[str]] = {}
    for item in unique.values():
        if item["page_id"] in cached:
            continue
        har_path = source_root / item["har"]
        if not har_path.exists():
            drops["missing_har"] += 1
            continue
        if item["har"] not in archive_urls:
            try:
                archive_urls[item["har"]] = archived_request_urls(har_path)
            except Exception:
                archive_urls[item["har"]] = set()
        if archive_urls[item["har"]] and item["url"].split("#", 1)[0] not in archive_urls[item["har"]]:
            drops["url_not_in_archive"] += 1
            continue
        refs[(item["task_id"], item["har"])][item["page_id"]] = item
    payloads = [
        {"source_root": str(source_root), "pages_dir": str(pages_dir), "har": har,
         "items": sorted(items.values(), key=lambda x: (x["url"], x["page_id"]))}
        for (_task, har), items in sorted(refs.items())
    ]
    failures: list[dict[str, Any]] = []
    start = time.time()
    if workers > 1 and len(payloads) > 1:
        with cf.ProcessPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(_extract_task_pages, payloads, chunksize=1):
                failures.extend(r for r in result if not r["ok"])
    else:
        for payload in payloads:
            failures.extend(r for r in _extract_task_pages(payload) if not r["ok"])
    cached = cached_page_ids(pages_dir)
    missing = len(unique) - len(cached & set(unique))
    print(f"pages: {len(cached & set(unique))}/{len(unique)} available, {len(failures)} extraction failures, "
          f"{time.time() - start:.1f}s", file=sys.stderr)
    drops["page_extraction_failure"] += len(failures)
    drops["page_unavailable_unique"] += missing
    return cached & set(unique), failures, drops


def provenance(rec: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": rec["run_id"], "run_dir": rec["run_dir"], "task_id": rec["task_id"], "step": rec["step"],
        "call_n": rec["call_n"], "agent": rec["agent"], "arm": rec["arm"], "run_success": rec["run_success"],
        "site": rec["site"], "url": rec["url"],
    }


def base_row(decision: str, rec: dict[str, Any], label: Any, reason: str) -> dict[str, Any]:
    previous = rec.get("previous_action")
    return {
        "decision": decision,
        "goal": rec["goal"],
        "history": rec["history"],
        "page_id": rec["page_id"],
        "label": label,
        "provenance": provenance(rec),
        "rule": RULES[decision],
        "label_reason": reason,
        "features": {
            "arm": rec["arm"],
            "agent": rec["agent"],
            "step": rec["step"],
            "previous_action": previous["name"] if previous else "",
            "previous_action_error": bool(previous and previous["is_error"]),
            "url_changed_after_previous_action": bool(previous and previous["url_before"] != previous["url_after"]),
        },
    }


def behavioural_rows(records: list[dict[str, Any]], available_pages: set[str]) -> tuple[dict[str, list[dict[str, Any]]], Counter]:
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in DECISIONS}
    drops: Counter = Counter()
    for rec in records:
        for decision in ("enough", "done"):
            if rec["page_id"] not in available_pages:
                drops[f"{decision}:page_unrecoverable"] += 1
                continue
            label, reason = rec["labels"][decision]
            rows[decision].append(base_row(decision, rec, label, reason))
        worked = rec["labels"]["worked"]
        if worked is None:
            drops["worked:no_previous_action" if rec["previous_action"] is None else "worked:no_next_commitment"] += 1
        elif rec["page_id"] not in available_pages:
            drops["worked:page_unrecoverable"] += 1
        else:
            label, reason = worked
            rows["worked"].append(base_row("worked", rec, label, reason))
    return rows, drops


def rows_from_next_examples(examples: list[Any], provenance_rows: list[dict[str, Any]], run_dir: str,
                            seen: set[tuple[str, tuple[int, ...], str]] | None = None
                            ) -> tuple[dict[str, list[dict[str, Any]]], int]:
    """Convert PR #5 next-step examples and provenance into SHOW/NEXT decision rows."""
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in DECISIONS}
    seen = seen if seen is not None else set()
    duplicates = 0
    for example, prov in zip(examples, provenance_rows):
        key = (example.page_id, tuple(example.equiv_ids), prov["action"])
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        rec = {
            "goal": example.query, "history": example.history, "url": prov["url"], "page_id": example.page_id,
            "har": "", "task_id": prov["task"], "site": prov["site"], "run_dir": run_dir, "run_id": prov["run_id"],
            "step": prov["step"], "call_n": prov.get("call_n"), "agent": prov["agent"], "arm": prov["arm"],
            "run_success": prov["run_success"], "previous_action": None,
        }
        show_label = {"target_id": example.target_id, "target_ids": example.equiv_ids}
        rows["show"].append(base_row("show", rec, show_label, "next_action_target_in_successful_run"))
        next_label = {
            "action_type": prov["action"], "target_id": example.target_id,
            "target_ids": example.equiv_ids, "text": prov.get("type_text", ""),
        }
        rows["next"].append(base_row("next", rec, next_label, "next_click_type_in_successful_run"))
    return rows, duplicates


def show_next_rows(source_root: Path, out_root: Path, pages_dir: Path) -> tuple[dict[str, list[dict[str, Any]]], Counter]:
    """Run the PR #5 replay path for every bench v2/v3 directory and convert it to SHOW/NEXT rows."""
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in DECISIONS}
    drops: Counter = Counter()
    seen: set[tuple[str, tuple[int, ...], str]] = set()
    for run_dir, tasks_file in SOURCE_TASKS.items():
        if not (source_root / "runs" / run_dir / "runs.jsonl").exists():
            continue
        eval_set, stats = build_bench(
            runs_dir=f"runs/{run_dir}", tasks_file=tasks_file, name=run_dir,
            root=source_root, out_root=out_root, cache_dir=pages_dir,
        )
        for key, value in stats.items():
            if key in SHOW_NEXT_DROP_KEYS:
                drops[f"show_next:{run_dir}:{key}"] += value
        build_path = out_root / "data/nextstep" / f"{run_dir}-build.json"
        build = json.loads(build_path.read_text())
        provenance_rows = build["provenance"]
        if len(provenance_rows) != len(eval_set.examples):
            raise ValueError(f"{run_dir}: provenance/examples length mismatch")
        converted, duplicates = rows_from_next_examples(eval_set.examples, provenance_rows, run_dir, seen)
        for name in DECISIONS:
            rows[name].extend(converted[name])
        drops["show_next:cross_directory_duplicate"] += duplicates
    return rows, drops


def merge_rows(parts: Iterable[dict[str, list[dict[str, Any]]]]) -> dict[str, list[dict[str, Any]]]:
    merged: dict[str, list[dict[str, Any]]] = {name: [] for name in DECISIONS}
    for part in parts:
        for name in DECISIONS:
            merged[name].extend(part.get(name, []))
    return merged


def assign_sites(site_counts: dict[str, int]) -> dict[str, str]:
    """Deterministic greedy split by site, approximately 70/15/15 by row count."""
    total = sum(site_counts.values())
    if not total:
        return {}
    order = sorted(site_counts, key=lambda s: (-site_counts[s], hashlib.sha256(s.encode()).hexdigest(), s))
    counts: Counter = Counter()
    assignment = {}
    for site in order:
        n = site_counts[site]

        def cost(split: str) -> float:
            return (counts[split] + n) / (SPLIT_RATIOS[split] * total)

        split = min(SPLIT_RATIOS, key=lambda s: (cost(s), {"train": 0, "dev": 1, "test": 2}[s]))
        assignment[site] = split
        counts[split] += n
    return assignment


def write_splits(rows: dict[str, list[dict[str, Any]]], out_dir: Path) -> tuple[dict[str, Any], dict[str, str]]:
    site_counts: Counter = Counter()
    for decision, items in rows.items():
        for row in items:
            site_counts[row["provenance"]["site"]] += 1
    assignment = assign_sites(dict(site_counts))
    summary: dict[str, Any] = {"decisions": {}, "splits": {s: {"rows": 0, "sites": []} for s in SPLIT_RATIOS}}
    for decision in DECISIONS:
        out = out_dir / decision
        out.mkdir(parents=True, exist_ok=True)
        counts = {split: Counter() for split in SPLIT_RATIOS}
        base_rates = {}
        for split in SPLIT_RATIOS:
            split_rows = []
            for row in rows[decision]:
                row_split = assignment[row["provenance"]["site"]]
                if row_split != split:
                    continue
                row = dict(row)
                row["split"] = split
                split_rows.append(row)
            with open(out / f"{split}.jsonl", "w", encoding="utf-8") as fh:
                for row in split_rows:
                    fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            counts[split]["rows"] = len(split_rows)
            counts[split]["sites"] = len({r["provenance"]["site"] for r in split_rows})
            if split_rows and isinstance(split_rows[0]["label"], bool):
                base_rates[split] = sum(bool(r["label"]) for r in split_rows) / len(split_rows)
        summary["decisions"][decision] = {
            "rows": {split: counts[split]["rows"] for split in SPLIT_RATIOS},
            "sites": {split: counts[split]["sites"] for split in SPLIT_RATIOS},
            "base_rates": base_rates,
        }
    for site, split in sorted(assignment.items()):
        summary["splits"][split]["sites"].append(site)
    for split in SPLIT_RATIOS:
        summary["splits"][split]["rows"] = sum(
            summary["decisions"][d]["rows"][split] for d in DECISIONS
        )
        summary["splits"][split]["site_count"] = len(summary["splits"][split]["sites"])
    return summary, assignment


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_test_manifest(out_dir: Path) -> dict[str, Any]:
    """Freeze the test rows and every page they reference by SHA-256."""
    referenced_pages = set()
    entries = []
    for decision in DECISIONS:
        path = out_dir / decision / "test.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        referenced_pages.update(row["page_id"] for row in rows)
        entries.append({"path": str(path.relative_to(out_dir)), "sha256": sha256_file(path),
                        "bytes": path.stat().st_size, "rows": len(rows)})
    for path in sorted((out_dir / "pages").glob("*/*.json.gz")):
        if path.name.removesuffix(".json.gz") in referenced_pages:
            entries.append({"path": str(path.relative_to(out_dir)), "sha256": sha256_file(path),
                            "bytes": path.stat().st_size})
    manifest = {
        "version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "algorithm": "sha256",
        "files": entries,
    }
    (out_dir / "test-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def screenshot_audit(source_root: Path, scan_agent_images: bool = True) -> dict[str, Any]:
    """Count screenshot-producing calls in server logs and image payloads in agent transcripts."""
    audit: dict[str, Any] = {"run_dirs": {}, "notes": [
        "server.jsonl stores image dimensions and byte counts, not PNG payloads",
        "agent.jsonl stores image/png payloads returned to Claude as base64; Codex transcripts may differ",
    ]}
    for run_dir in SOURCE_TASKS:
        root = source_root / "runs" / run_dir
        if not (root / "runs.jsonl").exists():
            continue
        rows = [json.loads(line) for line in (root / "runs.jsonl").read_text().splitlines() if line.strip()]
        by_arm: dict[str, Counter] = defaultdict(Counter)
        for row in rows:
            arm = row.get("arm", row.get("config", ""))
            by_arm[arm]["runs"] += 1
            by_arm[arm]["run_images"] += int(row.get("images") or 0)
            by_arm[arm]["run_image_bytes"] += int(row.get("image_bytes") or 0)
            by_arm[arm]["run_steps"] += int(row.get("steps") or 0)
        for path in sorted((root / "logs").glob("*.server.jsonl")):
            arm = path.name.split("__")[1]
            for line in path.read_text().splitlines():
                ev = json.loads(line)
                if ev.get("type") != "call":
                    continue
                by_arm[arm]["calls"] += 1
                if ev.get("image_bytes"):
                    by_arm[arm]["image_events"] += 1
                    by_arm[arm]["image_event_bytes"] += int(ev.get("image_bytes") or 0)
                    if ev.get("name") == "screenshot":
                        by_arm[arm]["explicit_screenshot_calls"] += 1
                    if ev.get("name") in ACTION_TOOLS:
                        by_arm[arm]["action_steps_with_image"] += 1
        if scan_agent_images:
            for path in sorted((root / "logs").glob("*.agent.jsonl")):
                arm = path.name.split("__")[1]
                count = 0
                with open(path, encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        count += line.count('"type":"image"')
                by_arm[arm]["agent_image_payloads"] += count
        audit["run_dirs"][run_dir] = {
            arm: dict(counts) for arm, counts in sorted(by_arm.items())
        }
    return audit


def extract(args: argparse.Namespace) -> dict[str, Any]:
    source_root = Path(args.source_root).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pages_dir = out_dir / "pages"
    records, scan_drops, run_counts = collect_states(source_root)
    print(f"states: {len(records)} labelled decision states", file=sys.stderr)
    available, page_failures, page_drops = extract_pages(records, source_root, pages_dir, workers=args.workers)
    rows_show_next, show_next_drops = show_next_rows(source_root, ROOT, pages_dir)
    available = cached_page_ids(pages_dir) & {r["page_id"] for r in records}
    rows_behavioural, behavioural_drops = behavioural_rows(records, available)
    rows = merge_rows([rows_behavioural, rows_show_next])
    split_summary, assignment = write_splits(rows, out_dir)
    manifest = write_test_manifest(out_dir)
    audit = screenshot_audit(source_root, scan_agent_images=not args.skip_agent_image_scan)
    drops: Counter = Counter()
    for part in (scan_drops, page_drops, behavioural_drops, show_next_drops):
        drops.update(part)
    show_next_counts = {}
    for run_dir in SOURCE_TASKS:
        build_path = ROOT / "data/nextstep" / f"{run_dir}-build.json"
        if build_path.exists():
            show_next_counts[run_dir] = json.loads(build_path.read_text())["counts"]
    summary = {
        **split_summary,
        "show_next_counts": show_next_counts,
        "source_root": str(source_root),
        "source_run_counts": dict(run_counts),
        "states": len(records),
        "unique_pages": len({r["page_id"] for r in records}),
        "available_pages": len(available),
        "page_failures": page_failures[:200],
        "drops": dict(sorted(drops.items())),
        "screenshot_audit": audit,
        "split_assignment": assignment,
        "test_manifest_files": len(manifest["files"]),
        "artifact_bytes": sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (out_dir / "screenshot-audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("extract", help="extract labels, replay pages, split rows and freeze the test manifest")
    ex.add_argument("--source-root", type=Path, required=True, help="checkout that contains runs/ and data/bench_v*/har")
    ex.add_argument("--out", type=Path, default=ROOT / "data/decisions")
    ex.add_argument("--workers", type=int, default=4)
    ex.add_argument("--skip-agent-image-scan", action="store_true", help="skip the large agent transcript scan")
    args = parser.parse_args(argv)
    if args.command == "extract":
        summary = extract(args)
        print(json.dumps({"decisions": summary["decisions"], "splits": summary["splits"],
                          "states": summary["states"], "drops": summary["drops"]}, indent=2))


if __name__ == "__main__":
    main()
