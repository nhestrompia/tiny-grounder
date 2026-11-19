"""Results tables for `demo.cli_bench`.

    uv run python -m demo.cli_report --out runs/cli      # writes OUT/results.md and OUT/runs.jsonl

One row per agent x config (success rate with n, medians of wall time, tool calls, tool-output characters, image
bytes, agent-reported input tokens and cost), then a per-task success matrix (successes/runs).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from demo.cli_bench import FOREIGN_NOTE, load_runs
from demo.mcp_server import BENCH_CONFIGS

RESULTS_FILE = "results.md"
RUNS_FILE = "runs.jsonl"


def med(values: list[Any], fmt: str = "{:,.0f}") -> str:
    vals = [v for v in values if v is not None]
    return fmt.format(median(vals)) if vals else "-"


def config_order(config: str) -> int:
    return BENCH_CONFIGS.index(config) if config in BENCH_CONFIGS else len(BENCH_CONFIGS)


def results_table(runs: list[dict[str, Any]]) -> str:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in runs:
        groups[(r["agent"], r["config"])].append(r)
    keys = sorted(groups, key=lambda k: (k[0], config_order(k[1]), k[1]))

    lines = [
        "| agent | config | success (n) | done called | median wall (s) | median tool calls | median tool-output chars "
        "| median image KB | median input tokens | median output tokens | median cost ($) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key in keys:
        rs = groups[key]
        ok = sum(bool(r["success"]) for r in rs)
        done = sum(bool(r["done_called"]) for r in rs)
        lines.append(
            f"| {key[0]} | {key[1]} | {ok / len(rs):.0%} ({ok}/{len(rs)}) | {done}/{len(rs)} "
            f"| {med([r['wall_seconds'] for r in rs], '{:.1f}')} | {med([r['tool_calls'] for r in rs], '{:.0f}')} "
            f"| {med([r['tool_chars'] for r in rs])} | {med([r['image_bytes'] / 1024 for r in rs])} "
            f"| {med([r.get('input_tokens') for r in rs])} | {med([r.get('output_tokens') for r in rs])} "
            f"| {med([r.get('cost_usd') for r in rs], '{:.4f}')} |"
        )

    cols = [f"{a} / {c}" for a, c in keys]
    per_task: dict[str, dict[tuple[str, str], list[bool]]] = defaultdict(lambda: defaultdict(list))
    for r in runs:
        per_task[r["task_id"]][(r["agent"], r["config"])].append(bool(r["success"]))
    lines += ["", "| task | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for task_id in sorted(per_task):
        cells = [f"{sum(v)}/{len(v)}" if (v := per_task[task_id].get(k)) else "-" for k in keys]
        lines.append(f"| {task_id} | " + " | ".join(cells) + " |")

    flagged = [r for r in runs if r.get(FOREIGN_NOTE)]
    if flagged:
        lines += ["", f"**Warning: {len(flagged)} run(s) used tools outside the `ui` MCP server:** "
                  + "; ".join(f"{r['agent']}/{r['config']}/{r['task_id']}#{r['repeat']}: {', '.join(r[FOREIGN_NOTE])}" for r in flagged)]
    if over := sum(r.get("oversized_results") or 0 for r in runs):
        lines += ["", f"Claude Code replaced {over} oversized tool result(s) (over its MCP output cap) by a stub pointing at a "
                  "file it cannot read here; `tool-output chars` counts what the server returned, not what the model saw "
                  "(see `model_visible_chars` in runs.jsonl)."]
    timed_out = sum(bool(r.get("timed_out")) for r in runs)
    errors = sum(1 for r in runs if r.get("agent_error"))
    lines += ["", f"{len(runs)} runs; {timed_out} timed out; {errors} with an agent-side error message."]
    return "\n".join(lines)


def write_results(out: str | Path) -> str:
    out = Path(out)
    runs = load_runs(out)
    (out / RUNS_FILE).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in runs), encoding="utf-8")
    table = results_table(runs)
    (out / RESULTS_FILE).write_text(table + "\n", encoding="utf-8")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="directory holding logs/")
    print(write_results(parser.parse_args().out))


if __name__ == "__main__":
    main()
