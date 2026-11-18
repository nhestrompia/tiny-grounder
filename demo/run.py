"""Run the agent demo: every selected task, in every selected configuration, several times.

    uv run python -m demo.run --out runs/demo --ranker bm25
    uv run python -m demo.run --out runs/demo --model-dir dist/m --configs top20 find --tasks wiki-search-turing

Logs go to `OUT/logs/<task>__<config>__<repeat>.jsonl` (one JSON event per model response and tool call, then a
`summary` line); the results table to `OUT/results.md`. A run whose log already has a summary is skipped, so an
interrupted sweep resumes; delete the log to repeat a run. Needs ANTHROPIC_API_KEY (or another Anthropic credential).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from demo.agent import CONFIGS, BM25Ranker, ClaudePlanner, Planner, Ranker, RunResult, run_task
from demo.browser import Task, load_tasks
from demo.report import LOGS_DIR, write_results


def log_path(out: Path, task: Task, config: str, repeat: int) -> Path:
    return out / LOGS_DIR / f"{task.id}__{config}__{repeat}.jsonl"


def run_one(task: Task, config: str, repeat: int, planner: Planner, ranker: Ranker, out: Path) -> RunResult | None:
    """Run and log one task; returns None when it was already logged. Browser and planner errors are recorded."""
    path = log_path(out, task, config, repeat)
    if path.exists() and '"type": "summary"' in path.read_text(encoding="utf-8"):
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:

        def log(event: dict[str, Any]) -> None:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
            fh.flush()

        try:
            result = run_task(task, config, planner, ranker, log, repeat)
        except Exception as exc:  # one failed run must not abort the sweep; it counts as a failure
            result = RunResult(task.id, config, repeat, stop="error", error=f"{type(exc).__name__}: {exc}")
        log({"type": "summary", **result.to_dict()})
    return result


def make_ranker(name: str, model_dir: str | None) -> Ranker:
    if name == "bm25":
        return BM25Ranker()
    if not model_dir:
        raise SystemExit("--ranker model needs --model-dir (or use --ranker bm25)")
    from grounder.runtime import Grounder

    return Grounder(model_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="output directory (logs/ and results.md)")
    parser.add_argument("--configs", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    parser.add_argument("--tasks", nargs="+", help="task ids (default: all in tasks.yaml)")
    parser.add_argument("--tasks-file", help="alternative tasks YAML")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--ranker", choices=["model", "bm25"], default="model")
    parser.add_argument("--model-dir", help="exported grounder directory (for --ranker model)")
    parser.add_argument("--model", default="claude-opus-5-5", help="planner model id")
    parser.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    args = parser.parse_args(argv)

    tasks = load_tasks(args.tasks_file) if args.tasks_file else load_tasks()
    if args.tasks:
        unknown = set(args.tasks) - {t.id for t in tasks}
        if unknown:
            raise SystemExit(f"unknown task ids: {', '.join(sorted(unknown))}")
        tasks = [t for t in tasks if t.id in args.tasks]
    ranker = make_ranker(args.ranker, args.model_dir)
    planner = ClaudePlanner(args.model, args.effort)
    out = Path(args.out)

    for task in tasks:
        for config in args.configs:
            for repeat in range(args.repeats):
                result = run_one(task, config, repeat, planner, ranker, out)
                if result is not None:
                    print(f"{task.id} {config} #{repeat}: {'ok' if result.success else 'FAIL'} "
                          f"calls={result.calls} steps={result.steps} in={result.tokens_in} out={result.output_tokens} "
                          f"{result.seconds:.1f}s {result.error}".rstrip(), file=sys.stderr)
    print(write_results(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
