"""Results tables for `demo.cli_bench`.

    uv run python -m demo.cli_report --out runs/cli      # writes OUT/results.md and OUT/runs.jsonl

One row per agent x arm (success rate with n, medians of wall time, tool calls, tool-output characters, image
bytes, agent-reported input tokens and cost, tool-output tokens per success), then a per-task success matrix
(successes/runs). An arm is `record["arm"]` (`find[exact]`, `a11y-cap`...), the config name for v1 runs.

Benchmark v2 adds `paired_analysis`: per agent, the success difference between arms (find[blend] vs find[exact], and
find[blend] vs the best baseline) with a bootstrap 95% CI that resamples SITES (tasks of one site are correlated;
`cluster="task"` resamples tasks instead) over per-task success rates, the tool-output tokens per success of each arm
and their ratio, and the plan's gates (non-inferiority margins, >=25% fewer tokens than the best baseline).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Callable

import numpy as np

from demo.cli_bench import FOREIGN_NOTE, load_runs
from demo.mcp_server import BENCH_CONFIGS

ARM_EXACT = "find[exact]"
ARM_BLEND = "find[blend]"
BASELINE_CONFIGS = ("a11y-cap", "a11y", "vision", "full", "top20")  # arms that show the agent a page view, not find
MARGIN_VS_EXACT = 2.0  # points: the ranker ships over exact only if the paired CI rules out a larger loss
MARGIN_VS_BASELINE = 5.0
TOKEN_GATE = 0.75  # find must use <= 75% of the best baseline's tool-output tokens per success
BOOTSTRAP = 4000
RESULTS_FILE = "results.md"
RUNS_FILE = "runs.jsonl"


def med(values: list[Any], fmt: str = "{:,.0f}") -> str:
    vals = [v for v in values if v is not None]
    return fmt.format(median(vals)) if vals else "-"


def arm_of(run: dict[str, Any]) -> str:
    return run.get("arm") or run["config"]


def config_order(arm: str) -> int:
    base = arm.split("[")[0]
    return BENCH_CONFIGS.index(base) if base in BENCH_CONFIGS else len(BENCH_CONFIGS)


def site_of(run: dict[str, Any]) -> str:
    """Cluster of a run: its site, else (v1 records) the first word of the task id."""
    return run.get("site") or run["task_id"].split("-")[0]


def tokens_of(run: dict[str, Any]) -> float:
    """Tool-output tokens of a run: the recorded estimate (chars/4 + screenshots), else chars/4 of the tool text."""
    if run.get("tool_tokens_est") is not None:
        return float(run["tool_tokens_est"])
    return run.get("tool_chars", 0) / 4 + run.get("image_bytes", 0) / 1000  # v1 fallback; images ~ bytes/1000 tokens


def tokens_per_success(runs: list[dict[str, Any]]) -> float | None:
    ok = sum(bool(r["success"]) for r in runs)
    return sum(tokens_of(r) for r in runs) / ok if ok else None


def results_table(runs: list[dict[str, Any]]) -> str:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in runs:
        groups[(r["agent"], arm_of(r))].append(r)
    keys = sorted(groups, key=lambda k: (k[0], config_order(k[1]), k[1]))

    lines = [
        "| agent | arm | success (n) | done called | median wall (s) | median tool calls | median tool-output chars "
        "| median image KB | median input tokens | median output tokens | median cost ($) | tool tokens / success |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
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
            f"| {med([r.get('cost_usd') for r in rs], '{:.4f}')} "
            f"| {'-' if (tps := tokens_per_success(rs)) is None else f'{tps:,.0f}'} |"
        )

    cols = [f"{a} / {c}" for a, c in keys]
    per_task: dict[str, dict[tuple[str, str], list[bool]]] = defaultdict(lambda: defaultdict(list))
    for r in runs:
        per_task[r["task_id"]][(r["agent"], arm_of(r))].append(bool(r["success"]))
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
    if replay := [r for r in runs if r.get("replay")]:
        lines += ["", f"{len(replay)} run(s) replayed frozen snapshots offline; {sum(r.get('replay_misses', 0) for r in replay)} "
                  f"request(s) missed the archive, {sum(r.get('replay_network_failures', 0) for r in replay)} reached the dead-end network, "
                  f"{sum(r.get('replay_live_requests', 0) for r in replay)} were answered by a live server."]
    timed_out = sum(bool(r.get("timed_out")) for r in runs)
    errors = sum(1 for r in runs if r.get("agent_error"))
    lines += ["", f"{len(runs)} runs; {timed_out} timed out; {errors} with an agent-side error message."]
    return "\n".join(lines)


# ---- paired analysis (benchmark v2) ---------------------------------------------------------------------------------


def task_table(runs: list[dict[str, Any]], agent: str, arm: str) -> dict[str, dict[str, Any]]:
    """task_id -> {"site", "rate" (mean success over repeats), "ok", "n", "tokens"} for one agent x arm."""
    out: dict[str, dict[str, Any]] = {}
    for r in runs:
        if r["agent"] != agent or arm_of(r) != arm:
            continue
        t = out.setdefault(r["task_id"], {"site": site_of(r), "ok": 0, "n": 0, "tokens": 0.0})
        t["ok"] += bool(r["success"])
        t["n"] += 1
        t["tokens"] += tokens_of(r)
    for t in out.values():
        t["rate"] = t["ok"] / t["n"]
    return out


def cluster_bootstrap(units: list[tuple[str, Any]], stat: Callable[[list[Any]], float], cluster: str = "site",
                      reps: int = BOOTSTRAP, seed: int = 0) -> tuple[float, float, float]:
    """(point estimate, 2.5%, 97.5%) of `stat` over `units` = [(site, value)], resampling whole sites (cluster="site")
    or single units (cluster="task") with replacement. NaN replicates (a statistic undefined on a sample) are dropped."""
    point = stat([v for _, v in units])
    if cluster == "task":
        groups = [[v] for _, v in units]
    else:
        by_site: dict[str, list[Any]] = defaultdict(list)
        for site, v in units:
            by_site[site].append(v)
        groups = list(by_site.values())
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(reps):
        picked = rng.integers(0, len(groups), len(groups))
        val = stat([v for i in picked for v in groups[i]])
        if not np.isnan(val):
            draws.append(val)
    if not draws:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi)


def paired_success(a: dict[str, dict], b: dict[str, dict], cluster: str = "site") -> tuple[int, float, float, float]:
    """(tasks, diff, lo, hi) in points: mean over the tasks both arms ran of rate_a - rate_b."""
    common = sorted(set(a) & set(b))
    units = [(a[t]["site"], a[t]["rate"] - b[t]["rate"]) for t in common]
    if not units:
        return 0, float("nan"), float("nan"), float("nan")
    point, lo, hi = cluster_bootstrap(units, lambda xs: float(np.mean(xs)), cluster)
    return len(units), 100 * point, 100 * lo, 100 * hi


def paired_tokens(a: dict[str, dict], b: dict[str, dict], cluster: str = "site") -> tuple[float, float, float, float, float]:
    """(tokens/success of a, of b, ratio a/b, lo, hi of the ratio) over the common tasks."""
    common = sorted(set(a) & set(b))
    units = [(a[t]["site"], (a[t]["tokens"], a[t]["ok"], b[t]["tokens"], b[t]["ok"])) for t in common]

    def ratio(xs: list[tuple[float, int, float, int]]) -> float:
        ta, oa = sum(x[0] for x in xs), sum(x[1] for x in xs)
        tb, ob = sum(x[2] for x in xs), sum(x[3] for x in xs)
        return (ta / oa) / (tb / ob) if oa and ob and tb else float("nan")

    if not units:
        return (float("nan"),) * 5
    point, lo, hi = cluster_bootstrap(units, ratio, cluster)
    ta, oa = sum(u[1][0] for u in units), sum(u[1][1] for u in units)
    tb, ob = sum(u[1][2] for u in units), sum(u[1][3] for u in units)
    return (ta / oa if oa else float("nan")), (tb / ob if ob else float("nan")), point, lo, hi


def best_baseline(runs: list[dict[str, Any]], agent: str) -> str | None:
    """The page-view arm (a11y-cap, vision, ...) with the highest success for `agent` (ties: fewer tokens per success)."""
    arms = sorted({arm_of(r) for r in runs if r["agent"] == agent and arm_of(r).split("[")[0] in BASELINE_CONFIGS})
    if not arms:
        return None

    def key(arm: str) -> tuple[float, float]:
        rs = [r for r in runs if r["agent"] == agent and arm_of(r) == arm]
        return (-sum(bool(r["success"]) for r in rs) / len(rs), tokens_per_success(rs) or float("inf"))

    return min(arms, key=key)


def fmt_ci(point: float, lo: float, hi: float, digits: int = 1) -> str:
    return f"{point:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]"


def paired_analysis(runs: list[dict[str, Any]], cluster: str = "site") -> str:
    """Markdown: per agent, success of every arm, then the two paired comparisons with their gates."""
    agents = sorted({r["agent"] for r in runs})
    lines = ["## Paired analysis", "",
             f"Success difference in percentage points over tasks (mean over repeats per task), 95% CI from a {BOOTSTRAP}-draw "
             f"bootstrap that resamples **{cluster}s**. Tokens = tool-output tokens per success (chars/4 + 1 per 750 screenshot "
             "pixels); ratio < 1 means fewer tokens.", ""]
    ran = False
    for agent in agents:
        arms = sorted({arm_of(r) for r in runs if r["agent"] == agent}, key=lambda a: (config_order(a), a))
        tables = {arm: task_table(runs, agent, arm) for arm in arms}
        if ARM_BLEND not in tables:
            continue
        ran = True
        lines += [f"### {agent}", "", "| arm | tasks | runs | success | tool tokens / success |", "|---|---|---|---|---|"]
        for arm in arms:
            rs = [r for r in runs if r["agent"] == agent and arm_of(r) == arm]
            ok = sum(bool(r["success"]) for r in rs)
            tps = tokens_per_success(rs)
            lines.append(f"| {arm} | {len(tables[arm])} | {len(rs)} | {ok / len(rs):.1%} | {'-' if tps is None else f'{tps:,.0f}'} |")
        lines += ["", "| comparison | tasks | success diff (pts) [95% CI] | tokens/success a | b | ratio a/b [95% CI] | gate |",
                  "|---|---|---|---|---|---|---|"]
        comparisons: list[tuple[str, str, float]] = []
        if ARM_EXACT in tables:
            comparisons.append((ARM_BLEND, ARM_EXACT, MARGIN_VS_EXACT))
        if base := best_baseline(runs, agent):
            comparisons.append((ARM_BLEND, base, MARGIN_VS_BASELINE))
            if ARM_EXACT in tables:
                comparisons.append((ARM_EXACT, base, MARGIN_VS_BASELINE))
        for a, b, margin in comparisons:
            n, d, lo, hi = paired_success(tables[a], tables[b], cluster)
            ta, tb, ratio, rlo, rhi = paired_tokens(tables[a], tables[b], cluster)
            gate = [f"non-inferior (-{margin:g} pts): {'YES' if lo > -margin else 'no'}"]
            if b != ARM_EXACT:
                gate.append(f">=25% fewer tokens: {'YES' if ratio <= TOKEN_GATE else 'no'} "
                            f"(CI upper {'<' if rhi < TOKEN_GATE else '>='} {TOKEN_GATE})")
            else:
                gate.append(f"tokens no higher: {'YES' if ratio <= 1.0 else 'no'}")
            lines.append(f"| {a} vs {b} | {n} | {fmt_ci(d, lo, hi)} | {ta:,.0f} | {tb:,.0f} | {ratio:.2f} [{rlo:.2f}, {rhi:.2f}] "
                         f"| {'; '.join(gate)} |")
        lines += [""]
        tags = sorted({t for r in runs for t in r.get("tags") or []})
        if tags:
            lines += ["| tag | tasks | " + " | ".join(arms) + " |", "|---|---|" + "---|" * len(arms)]
            for tag in tags:
                ids = {r["task_id"] for r in runs if tag in (r.get("tags") or [])}
                cells = []
                for arm in arms:
                    rates = [t["rate"] for k, t in tables[arm].items() if k in ids]
                    cells.append(f"{100 * float(np.mean(rates)):.0f}%" if rates else "-")
                lines.append(f"| {tag} | {len(ids)} | " + " | ".join(cells) + " |")
            lines += [""]
    if not ran:
        return ""
    return "\n".join(lines)


def write_results(out: str | Path) -> str:
    out = Path(out)
    runs = load_runs(out)
    (out / RUNS_FILE).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in runs), encoding="utf-8")
    table = results_table(runs)
    if paired := paired_analysis(runs):
        table += "\n\n" + paired
    (out / RESULTS_FILE).write_text(table + "\n", encoding="utf-8")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="directory holding logs/")
    print(write_results(parser.parse_args().out))


if __name__ == "__main__":
    main()
