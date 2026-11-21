"""Pre-registered competitor comparison. Reads completed run records only."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import yaml

from demo.cli_report import BOOTSTRAP, cluster_bootstrap, site_of, tokens_per_success
from tools.milestone_report import ci, divide, json_safe, number, paired_median, per_success

ARMS = ("B", "M", "pw-mcp-compact", "agent-browser")
AGENTS = ("claude:claude-haiku-5-5", "claude:claude-sonnet-5-5", "codex:gpt-6-luna", "codex:gpt-6.1-terra")
COMPARISONS = (("B", "pw-mcp-compact"), ("B", "agent-browser"), ("M", "pw-mcp-compact"))
NAN = float("nan")
ADJUSTED = "wall_seconds_without_harness"


def arm_kind(run: dict) -> str:
    """Use config and ranker, independent of display labels and order."""
    config = run["config"]
    if config == "find+observe" and run.get("ranker") in ("exact", "exact+exact"):
        return "B"
    if config == "pw-mcp+observe":
        return "M"
    if config in ("pw-mcp-compact", "agent-browser", "pw-mcp"):
        return config
    raise ValueError(f"Unknown arm config/ranker: {config!r}/{run.get('ranker')!r}")


def harness_failure(run: dict) -> bool:
    return ("no server summary" in (run.get("agent_error") or "").lower()
            or not run.get("has_server_summary", run.get("summary") is not None))


def load_runs(runs_dir: Path) -> list[dict]:
    """Read JSON records and check their corresponding server log, without rewriting logs."""
    runs = []
    for path in sorted((runs_dir / "logs").glob("*.json")):
        try:
            run = json.loads(path.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            # A record can be incomplete or removed during a benchmark retry.
            # It remains a missing registered observation in this report.
            continue
        server = path.with_suffix(".server.jsonl")
        summary = None
        if server.exists():
            for line in server.read_text().splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue  # A running server can have an unfinished final line.
                if event.get("type") == "summary":
                    summary = event
        run["has_server_summary"] = summary is not None
        runs.append(run)
    return runs


def index_runs(runs: list[dict]) -> dict:
    indexed = defaultdict(dict)
    for run in runs:
        arm = arm_kind(run)
        key = (run["task_id"], run["agent"])
        if key in indexed[arm]:
            raise ValueError(f"Duplicate run for {key!r}, {arm}")
        indexed[arm][key] = run
    return indexed


def bootstrap(units, stat):
    return list(cluster_bootstrap(units, stat, reps=BOOTSTRAP, seed=0)) if units else [NAN] * 3


def gates(success, tokens, latency, provisional=False):
    result = {
        "success": all(map(math.isfinite, success)) and success[1] > -10,
        "tokens": all(map(math.isfinite, tokens)) and tokens[2] < 1,
        "latency": all(map(math.isfinite, latency)) and latency[2] <= 1.10,
    }
    result["overall"] = all(result.values())
    return {key: value and not provisional for key, value in result.items()}


def compare(indexed, arm, reference, expected_keys):
    a, b = indexed.get(arm, {}), indexed.get(reference, {})
    expected_keys = set(expected_keys)
    missing = sorted(expected_keys - (a.keys() & b.keys()))
    common = sorted(a.keys() & b.keys())
    units, excluded = [], []
    for key in common:
        if harness_failure(a[key]) or harness_failure(b[key]):
            excluded.append(key)
            continue
        if site_of(a[key]) != site_of(b[key]):
            raise ValueError(f"Site mismatch for {key!r}")
        units.append((site_of(a[key]), (a[key], b[key])))

    def ratio(ps, measure):
        return divide(measure([x for x, _ in ps]), measure([y for _, y in ps]))

    success = bootstrap(units, lambda ps: 100 * sum(int(x["success"]) - int(y["success"]) for x, y in ps) / len(ps))
    tokens = bootstrap(units, lambda ps: ratio(ps, lambda rs: per_success(rs, "input_tokens")))
    latency = bootstrap(units, paired_median)
    resources = {}
    for field in ("tool_tokens_est", "input_tokens", "uncached_input_tokens", "cost_usd", "wall_seconds", ADJUSTED):
        measure = ((lambda rs: tokens_per_success(rs) if any(r["success"] for r in rs) else NAN)
                   if field == "tool_tokens_est" else lambda rs, field=field: per_success(rs, field))
        selected = [(s, p) for s, p in units if field != "cost_usd" or p[0]["agent"].startswith("claude:")]
        ps = [p for _, p in selected]
        resources[field] = {"arm": measure([x for x, _ in ps]), "reference": measure([y for _, y in ps]),
                            "ratio": bootstrap(selected, lambda ps: ratio(ps, measure))}
    # Missing adjusted telemetry must not silently change the paired population.
    adjusted = (bootstrap(units, lambda ps: paired_median(ps, ADJUSTED))
                if units and all(x.get(ADJUSTED) is not None and y.get(ADJUSTED) is not None for _, (x, y) in units)
                else [NAN] * 3)
    return {"comparison": f"{arm} vs {reference}", "paired_units": len(units),
            "sites": len({s for s, _ in units}), "missing_pairs": len(missing), "missing_pair_keys": missing,
            "arm_only": len(a.keys() - b.keys()), "reference_only": len(b.keys() - a.keys()),
            "harness_excluded_pairs": len(excluded), "harness_excluded_keys": excluded,
            "success_delta_pp": success, "token_ratio": tokens, "latency_ratio": latency,
            "adjusted_latency_ratio": adjusted, "resources_per_success": resources,
            "gates": gates(success, tokens, latency, bool(missing))}


def secondary(runs):
    groups = defaultdict(list)
    for r in runs:
        for scope in ("pooled", r["agent"]):
            groups[scope, arm_kind(r)].append(r)
    rows = []
    for (scope, arm), all_runs in sorted(groups.items()):
        rs = [r for r in all_runs if not harness_failure(r)]
        def med(field):
            vals = [r[field] for r in rs if r.get(field) is not None]
            return float(median(vals)) if vals else NAN
        def total(field):
            return sum(r[field] for r in all_runs) if all(r.get(field) is not None for r in all_runs) else NAN
        units = [(site_of(r), r) for r in rs]
        claude = [r for r in rs if r["agent"].startswith("claude:")]
        rows.append({"agent": scope, "arm": arm, "n": len(rs), "recorded_runs": len(all_runs),
                     "harness_failures": len(all_runs) - len(rs),
                     "success_rate": bootstrap(units, lambda xs: sum(bool(r["success"]) for r in xs) / len(xs)),
                     "input_tokens_per_success": per_success(rs, "input_tokens"),
                     "claude_cost_per_success": per_success(claude, "cost_usd"), "claude_n": len(claude),
                     "tool_tokens_per_success": tokens_per_success(rs) if rs else NAN,
                     "uncached_input_tokens_per_success": per_success(rs, "uncached_input_tokens"),
                     "wall_seconds_per_success": per_success(rs, "wall_seconds"),
                     "median_wall_seconds": med("wall_seconds"),
                     "p90_wall_seconds": float(np.percentile([r["wall_seconds"] for r in rs], 90)) if rs else NAN,
                     "median_adjusted_wall_seconds": med(ADJUSTED),
                     "median_tool_calls": med("tool_calls"), "median_turns": med("turns"),
                     **{f"total_{f}": total(f) for f in ("safety_ms", "settle_ms", "proxy_ms", "harness_overhead_ms", "replay_live_requests")},
                     "foreign_tool_runs": sum(bool(r.get("foreign_tools")) for r in all_runs),
                     "foreign_tools": sorted({t for r in all_runs for t in r.get("foreign_tools", [])}),
                     "timeouts": sum(bool(r.get("timed_out")) for r in all_runs),
                     "agent_errors": sum(bool(r.get("agent_error")) for r in all_runs)})
    return rows


def analyze(runs, tasks=None, agents=AGENTS):
    indexed = index_runs(runs)
    task_ids = [t["id"] for t in tasks] if tasks is not None else sorted({r["task_id"] for r in runs})
    expected = {(t, a) for t in task_ids for a in agents}
    unexpected = {(r["task_id"], r["agent"]) for r in runs} - expected
    if unexpected:
        raise ValueError(f"Runs outside registered task/agent population: {sorted(unexpected)!r}")
    pooled = [compare(indexed, a, b, expected) for a, b in COMPARISONS]
    provisional = any(r["missing_pairs"] for r in pooled)
    by_agent = {agent: [compare(index_runs([r for r in runs if r["agent"] == agent]), a, b,
                               {k for k in expected if k[1] == agent}) for a, b in COMPARISONS] for agent in agents}
    if provisional:
        for row in pooled + [r for rows in by_agent.values() for r in rows]:
            row["gates"] = {k: False for k in row["gates"]}
    return {"bootstrap_draws": BOOTSTRAP, "seed": 0, "confidence": .95, "provisional": provisional,
            "expected_pairs": len(expected), "recorded_runs": len(runs),
            "arm_labels": {a: sorted({r.get("arm", r["config"]) for r in rs.values()}) for a, rs in indexed.items()},
            "pooled": pooled, "per_agent": by_agent,
            "beats_existing_compact_tools": all(r["gates"]["overall"] for r in pooled[:2]),
            "middleware_passes": pooled[2]["gates"]["overall"], "chart_data": secondary(runs)}


def gate_table(rows, provisional=False):
    lines = ["| Comparison | Pairs / sites | Missing | Harness excluded | Success Δ pp [95% CI] | Input/success ratio [95% CI] | Paired median wall ratio [95% CI] | Success / tokens / latency | Overall |",
             "|---|---|---|---|---|---|---|---|---|"]
    def status(value):
        return "PENDING" if provisional else "PASS" if value else "FAIL"
    for r in rows:
        lines.append(f"| {r['comparison']} | {r['paired_units']} / {r['sites']} | {r['missing_pairs']} | {r['harness_excluded_pairs']} | "
                     + " | ".join(ci(r[f]) for f in ("success_delta_pp", "token_ratio", "latency_ratio"))
                     + " | " + " / ".join(status(r["gates"][k]) for k in ("success", "tokens", "latency"))
                     + f" | {status(r['gates']['overall'])} |")
    return lines


def markdown(report):
    provisional = report["provisional"]
    lines = ["# Competitor report" + (" (PROVISIONAL)" if provisional else ""), "",
             f"{report['recorded_runs']} records. Expected {report['expected_pairs']} task/agent pairs in each of four registered arms.",
             "B = find+observe exact. M = pw-mcp+observe. Stock pw-mcp is optional and has no gates.",
             "Primary estimates pool agents and pair by task and agent. Each bootstrap resamples whole sites, with 4,000 draws, seed 0, and 95% percentile bounds.",
             "Gates require success delta lower CI > -10 pp, total input tokens/success ratio upper CI < 1.0, and median paired wall_seconds ratio upper CI <= 1.10.",
             "Per-success resources sum all paired run usage, including unsuccessful runs, and divide by successes. Latency includes unsuccessful pairs.",
             "Harness failures have no server summary and are excluded pairwise. Missing records differ from excluded harness failures. Undefined estimates fail gates. Undefined bootstrap draws are dropped by the shared helper.",
             "Any missing registered pair makes every gate PENDING. Pilot records are never pooled into full-run records.", "",
             *gate_table(report["pooled"], provisional), "",
             "Beats existing compact tools (B must pass BOTH): " + ("PENDING" if provisional else "PASS" if report["beats_existing_compact_tools"] else "FAIL"), ""]
    for agent, rows in report["per_agent"].items():
        lines += [f"## {agent} (secondary)", "", *gate_table(rows, provisional), ""]
    lines += ["## Paired secondary resources (no gates)", "",
              "Claude cost uses Claude pairs only. Missing telemetry is N/A. Adjusted latency uses wall_seconds_without_harness, retaining child runtime.", "",
              "| Scope | Comparison | Resource / success | Arm | Reference | Ratio [95% CI] |", "|---|---|---|---|---|---|"]
    for scope, rows in {"pooled": report["pooled"], **report["per_agent"]}.items():
        for row in rows:
            for field, v in row["resources_per_success"].items():
                lines.append(f"| {scope} | {row['comparison']} | {field} | {number(v['arm'])} | {number(v['reference'])} | {ci(v['ratio'])} |")
            lines.append(f"| {scope} | {row['comparison']} | adjusted paired median latency | N/A | N/A | {ci(row['adjusted_latency_ratio'])} |")
    lines += ["", "## Arm summaries and chart data (no gates)", "",
              "Performance excludes harness failures. Diagnostic counts and overhead totals include all recorded runs. Pooled cost covers Claude only.", ""]
    fields = ["agent", "arm", "n", "recorded_runs", "harness_failures", "success_rate", "input_tokens_per_success", "claude_cost_per_success",
              "claude_n", "tool_tokens_per_success", "uncached_input_tokens_per_success", "wall_seconds_per_success", "median_wall_seconds",
              "p90_wall_seconds", "median_adjusted_wall_seconds", "median_tool_calls", "median_turns", "total_safety_ms", "total_settle_ms",
              "total_proxy_ms", "total_harness_overhead_ms", "total_replay_live_requests", "foreign_tool_runs", "foreign_tools", "timeouts", "agent_errors"]
    # Separate tables keep the text readable while preserving every chart/diagnostic field.
    for group in (fields[:9], [*fields[:2], *fields[9:17]], [*fields[:2], *fields[17:]]):
        lines += ["| " + " | ".join(group) + " |", "|" + "---|" * len(group)]
        for row in report["chart_data"]:
            def cell(field):
                v = row[field]
                if field == "success_rate":
                    return ci(v)
                if isinstance(v, float) or v is None:
                    return number(v)
                return str(v)
            lines.append("| " + " | ".join(cell(f) for f in group) + " |")
        lines += [""]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, default=REPO / "demo/tasks_milestone.yaml")
    args = parser.parse_args(argv)
    report = analyze(load_runs(args.runs_dir), yaml.safe_load(args.tasks.read_text()))
    for path, contents in ((args.out, markdown(report)), (args.json, json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
    if report["provisional"]:
        print("PROVISIONAL: registered pairs missing; all gates PENDING.")
    print("\n".join(gate_table(report["pooled"], report["provisional"])))


if __name__ == "__main__":
    main()
