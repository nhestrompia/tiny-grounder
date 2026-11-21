"""Prove scripted tasks are solvable through the capped accessibility baseline, without a model.

    uv run python -m tools.verify_tasks_v3

The YAML locators identify the oracle's next link. Only refs actually returned by snapshot()
are clicked through BenchServer. All pages replay through the dead-proxy HAR session.
This validates the baseline interface, not an agent's ability to choose the correct link.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import tempfile
import threading
from pathlib import Path

from demo.browser import BrowserSession, answer_matches, parse_tasks
from demo.mcp_server import BenchServer
from tools.bench_v2 import har_path, load_raw, locate


def verify(raw: dict) -> dict:
    task = parse_tasks([raw])[0]
    result = {"id": task.id, "refs": []}
    try:
        with tempfile.TemporaryDirectory() as tmp, BrowserSession(task.start_url, har=har_path(task)) as session:
            server = BenchServer("a11y-cap", session, None, task.goal, task.check,
                                 Path(tmp) / "calls.jsonl", task.id, answer_regex=task.answer_regex, max_calls=10000)
            try:
                for step in raw["path"]:
                    target = locate(session.page, step["click"])
                    label = " ".join(target.inner_text().split()).casefold()
                    target.evaluate("el => el.setAttribute('data-v3-oracle-target', '1')")
                    first = server.invoke("snapshot")
                    pages = int(re.search(r"snapshot page 1 of (\d+)", first).group(1))
                    snapshots = [first] + [server.invoke("snapshot", page=i) for i in range(2, pages + 1)]
                    links = [line for snapshot in snapshots for line in snapshot.splitlines()
                             if re.search(r'\blink\b.*\[ref=[a-z0-9]+\]', line)]
                    links.sort(key=lambda line: label not in " ".join(line.split()).casefold())
                    ref = None
                    for line in links:
                        candidate = re.search(r"\[ref=([a-z0-9]+)\]", line).group(1)
                        if session.page.locator(f"aria-ref={candidate}").get_attribute("data-v3-oracle-target"):
                            ref = candidate
                            break
                    target.evaluate("el => el.removeAttribute('data-v3-oracle-target')")
                    if ref is None:
                        raise ValueError(f"scripted link absent from accessibility snapshot: {label}")
                    before = session.url
                    server.invoke("click", id=ref)
                    if session.url == before:
                        raise ValueError(f"accessibility click did not navigate: {label}")
                    result["refs"].append({"label": label, "ref": ref, "snapshot_pages": pages, "url": session.url})
                final_first = server.invoke("snapshot")
                final_pages = int(re.search(r"snapshot page 1 of (\d+)", final_first).group(1))
                final_snapshot = "\n".join([final_first] + [server.invoke("snapshot", page=i)
                                                             for i in range(2, final_pages + 1)])
                result["aria_evidence_ok"] = not task.answer_regex or bool(re.search(task.answer_regex, final_snapshot, re.I))
                server.invoke("done", answer=raw.get("answer", ""))
                result.update(check_ok=session.succeeded(task.check),
                              answer_ok=answer_matches(task.answer_regex, raw.get("answer", "")),
                              evidence_ok=not task.answer_regex or bool(re.search(task.answer_regex, session.text(), re.I)),
                              network_failures=len(session.replay.failed), final_url=session.url)
                result["ok"] = all(result[k] for k in ("check_ok", "answer_ok", "evidence_ok", "aria_evidence_ok")) and not result["network_failures"]
            finally:
                server.finish("scripted verification")
    except Exception as exc:
        result.update(ok=False, why=f"{type(exc).__name__}: {str(exc).splitlines()[0]}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-file", type=Path, default=Path("demo/tasks_v3.yaml"))
    parser.add_argument("--out", type=Path, default=Path("data/bench_v3/a11y_verify.json"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    results = {}
    lock = threading.Lock()

    def run(raw):
        result = verify(raw)
        with lock:
            results[raw["id"]] = result
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(results, indent=1))
            print(("OK " if result["ok"] else "FAIL ") + raw["id"] + " " + result.get("why", ""), flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run, load_raw(args.tasks_file)))
    return int(any(not r["ok"] for r in results.values()))


if __name__ == "__main__":
    raise SystemExit(main())
