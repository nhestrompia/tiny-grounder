"""Build the frozen pre-registered W web next-step evaluation set live.

    uv run python -m tools.build_w_shops \
      --training-domains /path/to/data/leakage/heldout_domains.txt --workers 5

The output is a standard custom next-step set:
`w_shops|data/nextstep/w_shops.jsonl|data/nextstep/pages/w_shops`.

No ranker is imported or run here. Target resolution is deterministic DOM evidence only: the scripted locator's
promoted clickable control is matched to the extraction by tag, document-absolute geometry and link target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo.browser import BrowserSession, Task, check_success, load_tasks, parse_tasks
from grounder.data.crawl import registrable_domain
from grounder.data.pages import write_examples, write_page
from grounder.extract.web import extract_page_sync
from grounder.item_tags import TargetLabel, item_tags
from grounder.records import Element, Example, Page, action_repr, equiv_ids, equiv_key
from tools.bench_v2 import RECORD_UA, locate, settle

ACTION_STEPS = {"click", "type", "select", "press"}
SKIP_STEPS = {"goto", "wait", "scroll"}
BENCHMARK_HOSTS = {
    "saucedemo.com", "www.saucedemo.com", "demoblaze.com", "www.demoblaze.com", "tutorialsninja.com",
    "the-internet.herokuapp.com", "blazedemo.com",
}

# The extraction candidates plus labels (a click inside a label activates its control). Promotion mirrors what the
# browser actually activates when a text locator names a child inside a link/button/control.
CONTROL_SELECTOR = (
    "a[href],button,input:not([type=hidden]),select,textarea,summary,label,[onclick],"
    "[contenteditable]:not([contenteditable=false]),[tabindex],"
    "[role=button],[role=link],[role=checkbox],[role=radio],[role=switch],[role=tab],[role=menuitem],"
    "[role=menuitemcheckbox],[role=menuitemradio],[role=option],[role=combobox],[role=textbox],[role=searchbox],"
    "[role=slider],[role=spinbutton],[role=treeitem]"
)

# Label logic follows grounder/extract/dom.js. It additionally separates visible label text from accessibility,
# tooltip and alt text so the pre-registered item tags do not depend on hand annotation.
FINGERPRINT_JS = r"""
(el, CONTROL_SELECTOR) => {
  const flat = (s) => (s || "").replace(/\s+/g, " ").trim();
  const clip = (s, n) => flat(s).slice(0, n).trim();
  const isFormField = (x) => ["INPUT", "SELECT", "TEXTAREA"].includes(x.tagName);
  const BUTTON_INPUT_TYPES = new Set(["button", "submit", "reset", "image", "file", "color"]);
  const SKIP_LABEL_TAGS = new Set(["SELECT", "TEXTAREA", "SCRIPT", "STYLE"]);

  function labelOwnText(label, control) {
    let out = "";
    const walk = (node) => {
      for (const child of node.childNodes) {
        if (child === control) continue;
        if (child.nodeType === 3) out += child.data;
        else if (child.nodeType === 1 && !SKIP_LABEL_TAGS.has(child.tagName)) {
          out += " ";
          walk(child);
        }
      }
    };
    walk(label);
    return flat(out);
  }
  function labelTextOf(x) {
    if (!isFormField(x) || BUTTON_INPUT_TYPES.has(x.type) || !x.labels) return "";
    return flat([...x.labels].map((label) => labelOwnText(label, x)).join(" "));
  }
  function labelledByText(x) {
    const ids = flat(x.getAttribute("aria-labelledby")).split(" ").filter(Boolean);
    const root = x.getRootNode();
    return flat(ids.map((id) => {
      const ref = root.getElementById ? root.getElementById(id) : null;
      return ref ? (ref.getAttribute("aria-label") || ref.innerText || ref.textContent) : "";
    }).join(" "));
  }
  function labelledByVisibleText(x) {
    const ids = flat(x.getAttribute("aria-labelledby")).split(" ").filter(Boolean);
    const root = x.getRootNode();
    return flat(ids.map((id) => {
      const ref = root.getElementById ? root.getElementById(id) : null;
      return ref ? ref.innerText : "";
    }).join(" "));
  }
  function visibleText(x) {
    if (isFormField(x)) return "";
    return flat(x.innerText !== undefined ? x.innerText : x.textContent);
  }
  function iconText(x) {
    const icon = x.querySelector("img[alt]:not([alt='']),[title]:not([title='']),svg > title");
    if (!icon) return "";
    return flat(icon.getAttribute("alt") || icon.getAttribute("title") || icon.textContent);
  }
  function buttonValue(x) {
    if (x.tagName !== "INPUT" || !["button", "submit", "reset"].includes(x.type)) return "";
    return flat(x.getAttribute("value")) || { submit: "Submit", reset: "Reset" }[x.type] || "";
  }
  function selectedOptionText(x) {
    return x.tagName === "SELECT" && x.selectedOptions[0] ? flat(x.selectedOptions[0].text) : "";
  }
  function hrefOf(x) {
    if (x.tagName !== "A" || !x.hasAttribute("href")) return "";
    let url;
    try { url = new URL(x.href); } catch (e) { return clip(x.getAttribute("href"), 200); }
    const sameOrigin = url.origin === location.origin || (url.protocol === "file:" && location.protocol === "file:");
    return (sameOrigin ? url.pathname + url.search + url.hash : url.href).slice(0, 200);
  }

  let control = el.closest(CONTROL_SELECTOR) || el;
  if (control.tagName === "LABEL" && control.control) control = control.control;
  const rect = control.getBoundingClientRect();
  const aria = flat(control.getAttribute("aria-label"));
  const labelledby = labelledByText(control);
  const visibleCandidates = [
    labelledByVisibleText(control), labelTextOf(control), flat(control.getAttribute("placeholder")),
    buttonValue(control), visibleText(control), selectedOptionText(control),
  ];
  const nonvisualCandidates = [aria, labelledby, flat(control.getAttribute("alt")),
                               flat(control.getAttribute("title")), iconText(control)];
  const visible = clip(visibleCandidates.find(Boolean) || "", 80);
  const nonvisual = clip(nonvisualCandidates.find(Boolean) || "", 80);
  const labelCandidates = [aria, labelledby, labelTextOf(control), flat(control.getAttribute("alt")),
                           flat(control.getAttribute("placeholder")), buttonValue(control), visibleText(control),
                           flat(control.getAttribute("title")), iconText(control), selectedOptionText(control)];
  return {
    tag: control.tagName.toLowerCase(),
    abs_box: [rect.left + scrollX, rect.top + scrollY, rect.width, rect.height],
    href: hrefOf(control),
    label: clip(labelCandidates.find(Boolean) || "", 80),
    visible_text: visible,
    nonvisual_text: nonvisual,
    original_tag: el.tagName.toLowerCase(),
    promoted: control !== el,
    in_frame: !!window.frameElement,
  };
}
"""


@dataclass
class TargetMatch:
    ids: list[int]
    equivalent_ids: list[int]
    method: str = ""
    reason: str = ""


def _box_distance(a: list[float], b: list[float]) -> float:
    return max(abs(float(x) - float(y)) for x, y in zip(a, b, strict=True))


def _candidates(elements: list[Element], fingerprint: dict[str, Any], tolerance: float) -> list[Element]:
    return [
        e for e in elements
        if e.tag == fingerprint["tag"]
        and (e.href or "") == (fingerprint.get("href") or "")
        and len(e.abs_box) == 4
        and _box_distance(e.abs_box, fingerprint["abs_box"]) <= tolerance
    ]


def _match_at(elements: list[Element], fingerprint: dict[str, Any], tolerance: float, method: str) -> TargetMatch | None:
    candidates = _candidates(elements, fingerprint, tolerance)
    if not candidates:
        return None
    keys = {equiv_key(e) for e in candidates}
    if len(candidates) > 1 and len(keys) != 1:
        return TargetMatch([], [], method=method, reason="ambiguous_extraction_match")
    ids = [e.id for e in candidates]
    equivalent = sorted(set(ids) | {j for i in ids for j in equiv_ids(elements, i)})
    return TargetMatch(ids, equivalent, method=method)


def match_target_fingerprint(elements: list[Element], fingerprint: dict[str, Any]) -> TargetMatch:
    """Match a live DOM target to extracted element id(s), or give a precise failure reason."""
    if fingerprint.get("in_frame"):
        return TargetMatch([], [], reason="framed_target")
    if len(fingerprint.get("abs_box") or []) != 4:
        return TargetMatch([], [], reason="target_not_in_extraction")
    exact = _match_at(elements, fingerprint, 0.51, "exact_geometry")
    if exact:
        return exact
    loose = _match_at(elements, fingerprint, 1.51, "loose_geometry")
    if loose:
        return loose
    return TargetMatch([], [], reason="target_not_in_extraction")


def action_value(step: dict[str, Any], action: str) -> str | None:
    if action == "type":
        return str(step.get("text", ""))
    if action == "press":
        return str(step.get("press", ""))
    if action == "select":
        option = (step.get("select") or {}).get("option") or {}
        for key in ("label", "value", "index"):
            if key in option:
                return str(option[key])
    return None


def action_history(element: Element, action: str, value: str | None = None) -> str:
    """Goal-mode history in bench-v2's action_repr style, with SELECT as the obvious extension."""
    if action == "select":
        return f"[{element.tag or element.role}]  {element.text} -> SELECT: {value or ''}".rstrip()
    if action == "press":
        return f"[{element.tag or element.role}]  {element.text} -> PRESS: {value or ''}".rstrip()
    return action_repr(element, value if action == "type" else None)


def canonical_page_id(page: Page) -> str:
    """Content-address an extraction, so identical states share one frozen page without inventing identity."""
    material = page.to_dict()
    material.pop("page_id", None)
    material.pop("site", None)
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "wshops-" + hashlib.sha256(encoded).hexdigest()[:20]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(out_dir: Path, files: list[Path], summary: dict[str, Any]) -> Path:
    """Write SHA-256 and byte counts for every frozen payload file, sorted by repository-relative output path."""
    rows = {}
    for path in sorted(files, key=lambda p: str(p)):
        rows[path.relative_to(out_dir).as_posix()] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    out = out_dir / "w_shops.manifest.json"
    out.write_text(json.dumps({"format": 1, "summary": summary, "files": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def path_urls(raw: dict[str, Any]) -> list[str]:
    return [raw["start_url"]] + [step["goto"] for step in raw.get("path", []) if "goto" in step]


def host_of(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


def canonical_site(url: str) -> str:
    """The five benchmark sites as users name them (www is decoration; Heroku subdomains stay distinct)."""
    host = host_of(url)
    if host == "the-internet.herokuapp.com":
        return host
    return registrable_domain(url)


def leakage_drop(raw: dict[str, Any], training_domains: set[str]) -> str:
    for url in path_urls(raw):
        domain = registrable_domain(url)
        if domain in training_domains:
            return f"training_domain:{domain}"
        if host_of(url) not in BENCHMARK_HOSTS:
            return f"outside_benchmark_sites:{host_of(url)}"
    return ""


def perform_targeted_action(page: Any, action: str, locator: Any, step: dict[str, Any]) -> None:
    """Execute exactly the bench_v2 oracle action using an already resolved locator."""
    if action == "click":
        locator.click(timeout=10_000)
    elif action == "type":
        locator.click(timeout=10_000)
        page.keyboard.press("ControlOrMeta+A")
        page.keyboard.type(step["text"], delay=20)
        if step.get("enter"):
            page.keyboard.press("Enter")
    elif action == "select":
        option = dict((step.get("select") or {}).get("option") or {})
        locator.select_option(timeout=10_000, **option)
    elif action == "press":
        page.keyboard.press(step["press"])
    else:
        raise ValueError(f"not a targeted action: {action}")


def perform_untargeted_step(page: Any, step: dict[str, Any]) -> None:
    if "goto" in step:
        page.goto(step["goto"], wait_until="load")
    elif "press" in step:
        page.keyboard.press(step["press"])
    elif "wait" in step:
        page.wait_for_timeout(int(step["wait"]))
    elif "scroll" in step:
        height = page.evaluate("document.documentElement.scrollHeight")
        for i in range(1, 13):
            page.mouse.wheel(0, height / 12)
            page.wait_for_timeout(120)
        page.wait_for_timeout(400)
        page.evaluate("window.scrollTo(0, 0)")
    else:
        raise ValueError(f"unknown step {step}")


def first_line(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {str(exc).splitlines()[0][:240]}"


def empty_task_result(task: Task, tasks_file: Path) -> dict[str, Any]:
    return {
        "task_id": task.id, "site": task.site or canonical_site(task.start_url), "tasks_file": str(tasks_file),
        "examples": [], "pages": {}, "provenance": [], "drop_records": [], "drops": Counter(),
        "skipped": Counter(), "candidate_actions": 0, "dropped_task": "", "failure": "", "final_url": "",
    }


def build_task(job: dict[str, Any]) -> dict[str, Any]:
    """Run one scripted oracle path live and return examples/pages/provenance (all discarded if the path breaks)."""
    tasks_file = Path(job["tasks_file"])
    raw = job["raw"]
    task = parse_tasks([raw])[0]
    result = empty_task_result(task, tasks_file)
    reason = leakage_drop(raw, job["training_domains"])
    if reason:
        result["dropped_task"] = reason.split(":", 1)[0]
        result["failure"] = reason
        return result

    history: list[str] = []
    current_step = -1
    try:
        with BrowserSession(task.start_url, user_agent=RECORD_UA, headless=not job.get("headed", False)) as session:
            page = session.page
            for step_index, step in enumerate(raw.get("path", [])):
                current_step = step_index
                action = next((kind for kind in ACTION_STEPS if kind in step), None)
                if action is None:
                    kind = next(iter(step), "unknown")
                    result["skipped"][kind] += 1
                    perform_untargeted_step(page, step)
                    settle(page)
                    continue

                result["candidate_actions"] += 1
                if action == "press":
                    # A scripted key press has no locator: its target is the browser's active element. Resolve that
                    # element rather than guessing; if focus is on a non-control the step is dropped.
                    spec = {"active_element": True}
                    locator = None
                else:
                    spec = dict(step[action])
                    locator = locate(page, spec)
                    # The oracle runner auto-waits for an actionable element. Waiting here makes the pre-action extraction
                    # the page the action really uses rather than a transient loading state.
                    locator.wait_for(state="visible", timeout=10_000)
                extracted: Page | None = None
                extraction_error = ""
                try:
                    extracted = extract_page_sync(page)
                except Exception as exc:  # noqa: BLE001 - count the step, then still execute the oracle action
                    extraction_error = first_line(exc)

                fingerprint: dict[str, Any] = {}
                match = TargetMatch([], [], reason="target_not_in_extraction")
                element: Element | None = None
                if extracted is None:
                    match = TargetMatch([], [], reason="extraction_error")
                else:
                    try:
                        if action == "press":
                            fingerprint = page.evaluate(
                                "sel => ((" + FINGERPRINT_JS + ")(document.activeElement, sel))", CONTROL_SELECTOR)
                        else:
                            assert locator is not None
                            fingerprint = locator.evaluate(FINGERPRINT_JS, CONTROL_SELECTOR)
                        match = match_target_fingerprint(extracted.elements, fingerprint)
                    except Exception as exc:  # noqa: BLE001
                        fingerprint = {"error": first_line(exc)}
                        match = TargetMatch([], [], reason="fingerprint_error")
                    if match.ids:
                        element = next(e for e in extracted.elements if e.id == match.ids[0])

                value = action_value(step, action)
                if element is not None and extracted is not None and extracted.elements:
                    site = task.site or canonical_site(extracted.url) or canonical_site(task.start_url)
                    extracted.site = site
                    extracted.page_id = canonical_page_id(extracted)
                    result["pages"][extracted.page_id] = extracted
                    label = TargetLabel(element.text, fingerprint.get("visible_text", ""), fingerprint.get("nonvisual_text", ""))
                    tags = item_tags(task.goal, history, label)
                    example = Example(extracted.page_id, "goal", "goal", task.goal, list(history),
                                      match.ids[0], match.equivalent_ids, task.id, tags)
                    result["examples"].append(example)
                    result["provenance"].append({
                        "task": task.id, "step": step_index, "action": action, "locator": spec,
                        "url": extracted.url, "page_id": extracted.page_id, "target_id": match.ids[0],
                        "target_ids": match.equivalent_ids, "resolution": match.method,
                        "fingerprint": fingerprint, "element": element.to_dict(), "value": value,
                        "tags": tags,
                    })
                else:
                    if extracted is not None and not extracted.elements and match.reason == "target_not_in_extraction":
                        match.reason = "empty_extraction"
                    result["drops"][match.reason] += 1
                    record = {"task": task.id, "step": step_index, "action": action, "locator": spec,
                              "url": page.url, "reason": match.reason, "fingerprint": fingerprint}
                    if extraction_error:
                        record["error"] = extraction_error
                    result["drop_records"].append(record)

                perform_targeted_action(page, action, locator, step)
                settle(page)

                # History records the action even when extraction missed its target (the step remains dropped). The
                # fallback is the scripted control's own DOM label, never an invented or guessed extracted id.
                if element is not None:
                    history.append(action_history(element, action, value))
                else:
                    fallback = Element(0, "unknown", str(fingerprint.get("label") or ""), tag=str(fingerprint.get("tag") or ""))
                    history.append(action_history(fallback, action, value))

            result["final_url"] = page.url
            if not check_success(task.check, page.url, session.text()):
                result["dropped_task"] = "final_check_failed"
                result["failure"] = f"final check failed at {page.url}"
    except Exception as exc:  # noqa: BLE001 - a broken live oracle path drops the whole task by pre-registration policy
        result["dropped_task"] = "step_failed"
        result["failure"] = f"step {current_step}: {first_line(exc)}"
        result["traceback"] = traceback.format_exc(limit=3)

    if result["dropped_task"]:
        result["examples"] = []
        result["pages"] = {}
        result["provenance"] = []
    return result


def build_task_star(job: dict[str, Any]) -> dict[str, Any]:
    return build_task(job)


def input_hash(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def write_outputs(results: list[dict[str, Any]], out_dir: Path, inputs: dict[str, Any]) -> dict[str, Any]:
    examples_path = out_dir / "w_shops.jsonl"
    pages_dir = out_dir / "pages" / "w_shops"
    build_path = out_dir / "w_shops-build.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)

    examples: list[Example] = []
    pages: dict[str, Page] = {}
    provenance: list[dict[str, Any]] = []
    drop_records: list[dict[str, Any]] = []
    drops = Counter()
    skipped = Counter()
    dropped_tasks: dict[str, str] = {}
    per_task: dict[str, dict[str, Any]] = {}
    per_site = Counter()
    candidate_actions = 0

    for result in results:
        task_id = result["task_id"]
        candidate_actions += result["candidate_actions"]
        skipped.update(result["skipped"])
        drops.update(result["drops"])
        drop_records.extend(result["drop_records"])
        if result["dropped_task"]:
            dropped_tasks[task_id] = result["failure"] or result["dropped_task"]
            per_task[task_id] = {"examples": 0, "site": result["site"], "dropped": result["dropped_task"], "failure": result["failure"]}
            continue
        n = len(result["examples"])
        examples.extend(result["examples"])
        provenance.extend(result["provenance"])
        per_site[result["site"]] += n
        per_task[task_id] = {"examples": n, "site": result["site"], "candidate_actions": result["candidate_actions"],
                             "dropped_steps": dict(result["drops"]), "final_url": result["final_url"]}
        for pid, page in result["pages"].items():
            if pid in pages and pages[pid] != page:
                raise ValueError(f"page id collision with different content: {pid}")
            pages[pid] = page

    write_examples(examples, examples_path)
    page_files = [write_page(page, pages_dir) for page in pages.values()]
    tag_counts = Counter(tag for example in examples for tag in example.tags)
    untagged = sum(not example.tags for example in examples)
    counts = {
        "tasks": len(results), "kept_tasks": len(results) - len(dropped_tasks), "dropped_tasks": len(dropped_tasks),
        "sites": len(per_site), "examples": len(examples), "pages": len(pages),
        "candidate_actions": candidate_actions, "dropped_steps": sum(drops.values()),
    }
    build = {
        "format": 1, "inputs": inputs, "counts": counts,
        "per_task": per_task, "per_site": dict(sorted(per_site.items())),
        "dropped_steps_by_reason": dict(sorted(drops.items())), "drop_records": drop_records,
        "dropped_task_reasons": dropped_tasks, "skipped_untargeted_steps": dict(sorted(skipped.items())),
        "tag_counts": dict(sorted(tag_counts.items())) | {"untagged": untagged},
        "provenance": provenance,
    }
    build_path.write_text(json.dumps(build, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = write_manifest(out_dir, [examples_path, build_path, *page_files], counts)
    build["manifest"] = str(manifest)
    return build


def load_jobs(tasks_files: list[Path], training_domains: set[str], ids: set[str], headed: bool) -> tuple[list[dict[str, Any]], list[Task]]:
    jobs: list[dict[str, Any]] = []
    tasks: list[Task] = []
    for tasks_file in tasks_files:
        raw_tasks = yaml.safe_load(tasks_file.read_text(encoding="utf-8"))
        parsed = parse_tasks(raw_tasks)
        by_id = {row["id"]: row for row in raw_tasks}
        for task in parsed:
            if ids and task.id not in ids:
                continue
            tasks.append(task)
            jobs.append({"tasks_file": str(tasks_file), "raw": by_id[task.id],
                         "training_domains": training_domains, "headed": headed})
    return jobs, tasks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks-file", type=Path, action="append")
    parser.add_argument("--out", type=Path, default=ROOT / "data/nextstep")
    parser.add_argument("--training-domains", type=Path, default=ROOT / "data/leakage/heldout_domains.txt")
    parser.add_argument("--ids", nargs="+", default=[])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--headed", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    if not args.training_domains.exists():
        parser.error(f"training-domain list not found: {args.training_domains}")
    training_domains = {line.strip() for line in args.training_domains.read_text(encoding="utf-8").splitlines() if line.strip()}
    tasks_files = [path.resolve() for path in (args.tasks_file or [ROOT / "demo/tasks_v4.yaml", ROOT / "demo/tasks_v5.yaml"])]
    jobs, tasks = load_jobs(tasks_files, training_domains, set(args.ids), args.headed)
    if not jobs:
        parser.error("no tasks selected")

    print(f"building W shops: {len(jobs)} tasks, {args.workers} worker(s); no ranker is run", flush=True)
    if args.workers == 1:
        results = [build_task(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(build_task_star, jobs, chunksize=1))

    inputs = {
        "tasks_files": [input_hash(path) for path in tasks_files],
        "training_domains": input_hash(args.training_domains),
        "user_agent": RECORD_UA,
        "task_order": [task.id for task in tasks],
    }
    build = write_outputs(results, args.out, inputs)
    print(json.dumps(build["counts"], sort_keys=True), flush=True)
    print(f"dropped steps: {json.dumps(build['dropped_steps_by_reason'], sort_keys=True)}", flush=True)
    print(f"tags: {json.dumps(build['tag_counts'], sort_keys=True)}", flush=True)
    if build["dropped_task_reasons"]:
        print(f"dropped tasks: {json.dumps(build['dropped_task_reasons'], sort_keys=True)}", flush=True)
    print(f"wrote {args.out / 'w_shops.jsonl'} and {args.out / 'w_shops.manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
