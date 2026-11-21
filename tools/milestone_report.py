"""Pre-registered Plan v3 milestone analysis; reads runs without launching agents."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

# Support the documented direct-file invocation as well as python -m tools.milestone_report.
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import yaml

from demo.cli_report import BOOTSTRAP, arm_of, cluster_bootstrap, site_of, tokens_per_success

COMPARISONS = (("B", "baseline"), ("C", "baseline"), ("C", "B"))
NAN = float("nan")


def arm_kind(run: dict) -> str:
    """Map recorded labels by config and ranker kind; never infer from display order."""
    label = arm_of(run)
    config, bracket, ranker = label.partition("[")
    if bracket:
        if not ranker.endswith("]"):
            raise ValueError(f"Unknown arm: {label!r}")
        ranker = ranker[:-1]
    elif "@" in label:
        config, ranker = label.split("@", 1)
    else:
        ranker = run.get("ranker", "")
    if run.get("config", config) != config:
        raise ValueError(f"Arm/config mismatch: {label!r}")
    if config == "a11y-cap" and not bracket and "@" not in label:
        return "baseline"
    if config == "find+observe":
        kinds = [part.split(":", 1)[0] for part in ranker.split("+")]
        if kinds in (["exact"], ["exact", "exact"]):
            return "B"
        if kinds == ["exact", "blend"]:
            return "C"
    raise ValueError(f"Unknown arm: {label!r}, ranker={ranker!r}")


def index_runs(runs: list[dict]) -> dict:
    indexed = defaultdict(dict)
    labels = defaultdict(set)
    for run in runs:
        kind = arm_kind(run)
        labels[kind].add(arm_of(run))
        key = (run["task_id"], run["agent"])
        if key in indexed[kind]:
            raise ValueError(f"Duplicate run for {key!r}, {kind}")
        indexed[kind][key] = run
    if any(len(v) > 1 for v in labels.values()):
        raise ValueError(f"Multiple arm labels map to the same treatment: {dict(labels)!r}")
    return indexed


def per_success(runs: list[dict], field: str) -> float:
    if not runs or any(r.get(field) is None for r in runs):
        return NAN
    successes = sum(bool(r["success"]) for r in runs)
    return sum(r[field] for r in runs) / successes if successes else NAN


def divide(a: float, b: float) -> float:
    return a / b if b > 0 else NAN


def paired_median(pairs: list[tuple[dict, dict]], field: str = "wall_seconds", both: bool = False) -> float:
    values = [a[field] / b[field] for a, b in pairs
              if (not both or (a["success"] and b["success"]))
              and a.get(field) is not None and b.get(field) is not None and b[field] > 0]
    return float(median(values)) if values else NAN


def gates(success: list, tokens: list, latency: list, reference: str) -> dict:
    """All bounds must exist; the strict and inclusive thresholds are pre-registered."""
    def finite(metric):
        return all(math.isfinite(v) for v in metric)

    result = {
        "success": finite(success) and success[1] > (-2 if reference == "B" else -10),
        "tokens": finite(tokens) and (tokens[2] <= 1 if reference == "B" else tokens[2] < .5),
        "latency": finite(latency) and latency[2] <= (1 if reference == "B" else 1.10),
        "success_minus5_secondary": math.isfinite(success[1]) and success[1] > -5,
    }
    result["overall"] = all(result[k] for k in ("success", "tokens", "latency"))
    return result


def compare(indexed: dict, arm: str, reference: str) -> dict:
    a, b = indexed.get(arm, {}), indexed.get(reference, {})
    common = sorted(a.keys() & b.keys())
    units = []
    for key in common:
        if site_of(a[key]) != site_of(b[key]):
            raise ValueError(f"Site mismatch for {key!r}")
        units.append((site_of(a[key]), (a[key], b[key])))

    def bootstrap(stat):
        return list(cluster_bootstrap(units, stat, reps=BOOTSTRAP, seed=0)) if units else [NAN] * 3

    def rate_delta(ps):
        return 100 * sum(int(x["success"]) - int(y["success"]) for x, y in ps) / len(ps)

    def tps(rs):
        value = tokens_per_success(rs)
        return value if value is not None else NAN

    def ratio(ps, measure):
        return divide(measure([x for x, _ in ps]), measure([y for _, y in ps]))

    pairs = [p for _, p in units]
    success = bootstrap(rate_delta)
    tokens = bootstrap(lambda ps: ratio(ps, tps))
    latency = bootstrap(paired_median)
    measures = {"tool_tokens_est": tps}
    for field in ("wall_seconds", "input_tokens", "uncached_input_tokens", "cost_usd"):
        measures[field] = lambda rs, field=field: per_success(rs, field)
    resources = {}
    for field, measure in measures.items():
        selected = units
        if field == "cost_usd":
            selected = [(site, p) for site, p in units if p[0]["agent"].startswith("claude:")]
        ps = [p for _, p in selected]
        ci = (list(cluster_bootstrap(selected, lambda ps: ratio(ps, measure), reps=BOOTSTRAP, seed=0))
              if selected else [NAN] * 3)
        resources[field] = {"arm": measure([x for x, _ in ps]), "reference": measure([y for _, y in ps]), "ratio": ci}
    return {
        "comparison": f"{arm} vs {reference}", "paired_units": len(common),
        "sites": len({s for s, _ in units}),
        "missing_pairs": len(a.keys() ^ b.keys()),
        "arm_only": len(a.keys() - b.keys()), "reference_only": len(b.keys() - a.keys()),
        "success_delta_pp": success, "token_ratio": tokens, "latency_ratio": latency,
        "gates": gates(success, tokens, latency, reference), "resources_per_success": resources,
        "both_success_units": sum(x["success"] and y["success"] for x, y in pairs),
        "both_success_latency_ratio": bootstrap(lambda ps: paired_median(ps, both=True)),
        "agent_duration_ms_ratio": bootstrap(lambda ps: paired_median(ps, "agent_duration_ms")),
    }


def secondary(runs: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in runs:
        for agent in ("pooled", r["agent"]):
            groups[agent, arm_kind(r)].append(r)
    rows = []
    for (agent, arm), rs in sorted(groups.items()):
        def med(field):
            vals = [r[field] for r in rs if r.get(field) is not None]
            return float(median(vals)) if vals else NAN
        walls = [r["wall_seconds"] for r in rs]
        shares = [r["grounding_ms"] / (1000 * r["wall_seconds"]) for r in rs
                  if r.get("grounding_ms") is not None and r["wall_seconds"] > 0]
        rows.append({"agent": agent, "arm": arm, "n": len(rs),
                     "successes": sum(bool(r["success"]) for r in rs),
                     "p50_wall_seconds": float(np.percentile(walls, 50)),
                     "p90_wall_seconds": float(np.percentile(walls, 90)),
                     "median_tool_calls": med("tool_calls"), "median_turns": med("turns"),
                     "median_grounding_ms": med("grounding_ms"),
                     "median_grounding_share": float(median(shares)) if shares else NAN,
                     "median_agent_duration_ms": med("agent_duration_ms"),
                     "agent_duration_n": sum(r.get("agent_duration_ms") is not None for r in rs),
                     "timeouts": sum(bool(r.get("timed_out")) for r in rs),
                     "agent_errors": sum(bool(r.get("agent_error")) for r in rs)})
    return rows


def breakdown(runs: list[dict], tasks: list[dict]) -> list[dict]:
    metadata = {t["id"]: t for t in tasks}
    groups = defaultdict(list)
    for r in runs:
        if r["task_id"] not in metadata:
            raise ValueError(f"Task missing from tasks file: {r['task_id']}")
        task = metadata[r["task_id"]]
        tags = [tag for tag in ("multistep", "vague", "long") if tag in task.get("tags", [])]
        source = next((v for v in ("v2", "v3") if f"bench_{v}" in task.get("har", "")), "unknown source")
        for tag in [*tags, source]:
            groups[tag, arm_kind(r)].append(r)
    return [{"tag": tag, "arm": arm, "n": len(rs), "success_rate": sum(bool(r["success"]) for r in rs) / len(rs),
             "tokens_per_success": tokens_per_success(rs)} for (tag, arm), rs in sorted(groups.items())]


def analyze(runs: list[dict], tasks: list[dict]) -> dict:
    indexed = index_runs(runs)
    pooled = [compare(indexed, a, b) for a, b in COMPARISONS]
    by_agent = {agent: [compare(index_runs([r for r in runs if r["agent"] == agent]), a, b)
                        for a, b in COMPARISONS] for agent in sorted({r["agent"] for r in runs})}
    return {"bootstrap_draws": BOOTSTRAP, "seed": 0, "arm_labels": {k: sorted({arm_of(r) for r in v.values()}) for k, v in indexed.items()},
            "pooled": pooled, "per_agent": by_agent,
            "model_ships": pooled[1]["gates"]["overall"] and pooled[2]["gates"]["overall"],
            "secondary": secondary(runs), "breakdown": breakdown(runs, tasks)}


def number(value: float | None) -> str:
    return f"{value:.3f}" if value is not None and math.isfinite(value) else "N/A"


def ci(values: list) -> str:
    return f"{number(values[0])} [{number(values[1])}, {number(values[2])}]"


def status(value: bool) -> str:
    return "PASS" if value else "FAIL"


def gate_table(rows: list[dict]) -> list[str]:
    lines = ["| Comparison | Pairs / sites | Missing pairs | Success Δ pp [95% CI]; gate | Token ratio [95% CI]; gate | Latency ratio [95% CI]; gate | Overall |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        cells = [f"{ci(r[field])}; {status(r['gates'][gate])}" for field, gate in
                 (("success_delta_pp", "success"), ("token_ratio", "tokens"), ("latency_ratio", "latency"))]
        lines.append(f"| {r['comparison']} | {r['paired_units']} / {r['sites']} | {r['missing_pairs']} | " + " | ".join(cells) + f" | {status(r['gates']['overall'])} |")
    return lines


def markdown(report: dict) -> str:
    lines = ["# Milestone report", "",
             "Pre-registered comparisons: B vs baseline, C vs baseline, C vs B. Baseline = a11y-cap; B = keyword observe; C = model observe.",
             "Primary: pooled agents, paired by task and agent. Secondary: each agent separately. All CIs use 4000 site-bootstrap draws, seed 0, 95% percentile bounds. Every task and agent of a site moves together; within-agent analyses resample sites within that agent.",
             "Success gate vs baseline: lower CI > -10 percentage points; > -5 is secondary. Tool-output tokens/success = sum(tool_tokens_est) / sum(success); ratio upper CI < 0.5.",
             "Latency gate: median of paired wall_seconds ratios over all paired units, upper CI <= 1.10. Failed runs remain included. Successful-run-only latency is never a gate.",
             "C vs B gates: success lower CI > -2 points, token ratio upper CI <= 1.0, latency ratio upper CI <= 1.0. Model ships only if C also passes every baseline gate.",
             "Input tokens, uncached input tokens, Claude cost, wall time/success, both-success latency, agent duration, and all secondary tables have no gates.",
             "Missing pairs are excluded separately for each comparison. Undefined bootstrap draws are dropped by the shared v2 helper; undefined estimates or bounds fail gates. N/A means unavailable or undefined. Partial-run results are provisional.", "",
             "## Gate summary", "", *gate_table(report["pooled"]), "",
             f"Model ships in observe: {status(report['model_ships'])}.", "",
             "Recorded arms: " + "; ".join(f"{k} = {', '.join(v)}" for k, v in report["arm_labels"].items()), ""]
    for agent, rows in report["per_agent"].items():
        lines += [f"## {agent} (secondary)", "", *gate_table(rows), ""]
    lines += ["## Paired secondary measures", "", "Values use the same paired units as each comparison. Cost uses Claude pairs only. Missing telemetry gives N/A, not zero.", "",
              "| Scope | Comparison | Metric / success | Arm | Reference | Ratio [95% CI] |", "|---|---|---|---|---|---|"]
    scopes = {"pooled": report["pooled"], **report["per_agent"]}
    for scope, rows in scopes.items():
        for r in rows:
            for field, v in r["resources_per_success"].items():
                lines.append(f"| {scope} | {r['comparison']} | {field} | {number(v['arm'])} | {number(v['reference'])} | {ci(v['ratio'])} |")
    lines += ["", "| Scope | Comparison | Both-success pairs | Both-success wall ratio [95% CI] | Agent duration ratio [95% CI] | Success -5 secondary |", "|---|---|---|---|---|---|"]
    for scope, rows in scopes.items():
        for r in rows:
            lines.append(f"| {scope} | {r['comparison']} | {r['both_success_units']} | {ci(r['both_success_latency_ratio'])} | {ci(r['agent_duration_ms_ratio'])} | {status(r['gates']['success_minus5_secondary'])} |")
    lines += ["", "## All-run secondary measures", "", "Grounding share is the median per-run grounding_ms / (1000 × wall_seconds). Agent duration is the agent-reported model time.", "",
              "| Agent | Arm | Success / n | Wall p50 / p90 s | Tool calls median | Turns median | Grounding median ms / share | Agent duration median ms (n) | Timeouts | Agent errors |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in report["secondary"]:
        lines.append(f"| {r['agent']} | {r['arm']} | {r['successes']} / {r['n']} | {number(r['p50_wall_seconds'])} / {number(r['p90_wall_seconds'])} | {number(r['median_tool_calls'])} | {number(r['median_turns'])} | {number(r['median_grounding_ms'])} / {number(r['median_grounding_share'])} | {number(r['median_agent_duration_ms'])} ({r['agent_duration_n']}) | {r['timeouts']} | {r['agent_errors']} |")
    lines += ["", "## Tag and source breakdown", "", "Unpaired descriptive results; tags and HAR source come from the tasks file. No CIs.", "",
              "| Tag / source | Arm | n | Success rate | Tool tokens / success |", "|---|---|---|---|---|"]
    for r in report["breakdown"]:
        lines.append(f"| {r['tag']} | {r['arm']} | {r['n']} | {r['success_rate']:.1%} | {number(r['tokens_per_success'])} |")
    return "\n".join(lines) + "\n"


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, default=REPO / "demo/tasks_milestone.yaml")
    args = parser.parse_args(argv)
    runs = [json.loads(line) for line in args.runs.read_text().splitlines() if line.strip()]
    report = analyze(runs, yaml.safe_load(args.tasks.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(markdown(report), encoding="utf-8")
    args.out.with_suffix(".json").write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("\n".join(gate_table(report["pooled"])))


if __name__ == "__main__":
    main()
