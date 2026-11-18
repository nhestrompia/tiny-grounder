"""Results table from the per-run logs written by `demo.run`.

    uv run python -m demo.report --out runs/demo
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from demo.agent import CONFIGS

LOGS_DIR = "logs"
RESULTS_FILE = "results.md"


def load_runs(out: str | Path) -> list[dict]:
    """The summary record of every finished run log under `out/logs`; unfinished logs are ignored."""
    runs = []
    for path in sorted((Path(out) / LOGS_DIR).glob("*.jsonl")):
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        summaries = [r for r in records if r.get("type") == "summary"]
        if summaries:
            runs.append(summaries[-1])
    return runs


def results_table(runs: list[dict]) -> str:
    """Markdown: one row per configuration, then success per task."""
    by_config: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        by_config[run["config"]].append(run)
    configs = [c for c in CONFIGS if c in by_config]

    lines = [
        "| config | runs | success | mean tokens in (uncached / cache read / cache write) | mean tokens out "
        "| mean time (s) | mean calls | mean steps |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for config in configs:
        rs = by_config[config]
        lines.append(
            f"| {config} | {len(rs)} | {mean(r['success'] for r in rs):.1%} | {mean(r['tokens_in'] for r in rs):,.0f} "
            f"({mean(r['input_tokens'] for r in rs):,.0f} / {mean(r['cache_read_tokens'] for r in rs):,.0f} / "
            f"{mean(r['cache_creation_tokens'] for r in rs):,.0f}) "
            f"| {mean(r['output_tokens'] for r in rs):,.0f} | {mean(r['seconds'] for r in rs):.1f} "
            f"| {mean(r['calls'] for r in rs):.1f} | {mean(r['steps'] for r in rs):.1f} |"
        )

    per_task: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        per_task[run["task_id"]][run["config"]].append(run["success"])
    lines += ["", "| task | " + " | ".join(configs) + " |", "|---|" + "---|" * len(configs)]
    for task_id in sorted(per_task):
        cells = [f"{sum(v)}/{len(v)}" if (v := per_task[task_id].get(c)) else "-" for c in configs]
        lines.append(f"| {task_id} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_results(out: str | Path) -> str:
    table = results_table(load_runs(out))
    (Path(out) / RESULTS_FILE).write_text(table + "\n", encoding="utf-8")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="directory holding logs/")
    print(write_results(parser.parse_args().out))


if __name__ == "__main__":
    main()
