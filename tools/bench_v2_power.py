"""Power calculation for agent benchmark v2 (writes docs/bench-v2-power.md).

    uv run python -m tools.bench_v2_power [--out docs/bench-v2-power.md] [--tasks 60]

Model. For one agent and two arms A, B, task t has true success rates p_At, p_Bt. A run is Bernoulli(p). With R repeats
per task and arm, the paired per-task difference is d_t = mean_r(A) - mean_r(B), and

    Var(d_t) = sigma_b^2 + sigma_w^2 / R

    sigma_b^2  between-task variance of the TRUE difference p_At - p_Bt (does not shrink with repeats),
    sigma_w^2  within-task run noise of one A-run minus one B-run: E_t[p_At(1-p_At) + p_Bt(1-p_Bt)] (shrinks as 1/R).

Tasks of one site are correlated (intra-class correlation rho of d_t within a site), so N tasks on S sites, m = N/S per
site, carry the design effect DE = 1 + (m - 1) rho. The CI half-width of the mean paired difference is
z * sqrt(DE * Var(d_t) / N); the sample size for a one-sided non-inferiority test with margin M and power 1-beta (true
difference 0) solves (z_a + z_b) * SE <= M. sigma_w^2 and sigma_b^2 are estimated from the v1 logs (runs/cli*/logs, 38 tasks x
2 repeats, 8 arms x agents); v1 tasks were easy (success 79-100%), so a harder planning scenario is tabulated next to them.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from grounder.data.crawl import registrable_domain

ROOT = Path(__file__).resolve().parent.parent
Z_TWO = 1.96  # two-sided 95% CI
Z_ONE = 1.645  # one-sided alpha = 0.05
Z_POWER = 0.8416  # power 0.80
PLAN_RHO = 0.25


def load_v1() -> dict[tuple[str, str, str], dict[str, list[int]]]:
    """(agent, arm, ranker) -> task_id -> [success, ...] over every v1 run directory."""
    out: dict[tuple[str, str, str], dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted((ROOT / "runs").glob("cli*/logs/*.json")):
        if path.name.endswith((".server.jsonl", ".agent.jsonl")):
            continue
        r = json.loads(path.read_text(encoding="utf-8"))
        if "success" not in r or "task_id" not in r:
            continue
        out[(r["agent"], r["config"], r.get("ranker", ""))][r["task_id"]].append(int(bool(r["success"])))
    return out


def task_sites() -> dict[str, str]:
    import yaml

    return {t["id"]: registrable_domain(t["start_url"])
            for t in yaml.safe_load((ROOT / "demo" / "tasks.yaml").read_text(encoding="utf-8"))}


def components(a: dict[str, list[int]], b: dict[str, list[int]], sites: dict[str, str]) -> dict[str, float]:
    """Variance components of d_t = mean(A) - mean(B) from two runs-per-task tables (repeats may differ)."""
    tasks = sorted(set(a) & set(b))
    d, within, wnoise = [], [], []
    for t in tasks:
        xa, xb = np.array(a[t], float), np.array(b[t], float)
        ra, rb = len(xa), len(xb)
        if ra < 2 or rb < 2:
            continue
        va, vb = xa.var(ddof=1), xb.var(ddof=1)  # unbiased per-run variance
        d.append(xa.mean() - xb.mean())
        within.append(va / ra + vb / rb)  # sampling variance of this task's d_t
        wnoise.append(va + vb)  # per-run noise of (A-run minus B-run)
    d_arr = np.array(d)
    var_d = float(d_arr.var(ddof=1))
    sigma_w2 = float(np.mean(wnoise))
    mean_within = float(np.mean(within))
    r_eff = sigma_w2 / mean_within if mean_within > 0 else float(np.mean([len(a[t]) for t in tasks]))  # repeats used (2 in v1)
    sigma_b2 = max(0.0, var_d - mean_within)
    # one-way ANOVA ICC of d_t over sites (tasks sharing a start domain)
    groups: dict[str, list[float]] = defaultdict(list)
    for t, x in zip([t for t in tasks if len(a[t]) >= 2 and len(b[t]) >= 2], d):
        groups[sites.get(t, t)].append(x)
    k, n = len(groups), len(d)
    grand = float(np.mean(d))
    ssb = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups.values())
    ssw = sum(float(np.sum((np.array(g) - np.mean(g)) ** 2)) for g in groups.values())
    m0 = (n - sum(len(g) ** 2 for g in groups.values()) / n) / (k - 1) if k > 1 else 1.0
    msb, msw = ssb / (k - 1) if k > 1 else 0.0, ssw / (n - k) if n > k else 1.0
    icc = max(0.0, (msb - msw) / (msb + (m0 - 1) * msw)) if msb + (m0 - 1) * msw > 0 else 0.0
    return {"tasks": float(len(d)), "mean_d": grand, "var_d": var_d, "sigma_w2": sigma_w2, "sigma_b2": sigma_b2,
            "repeats": r_eff, "icc": icc, "sites": float(k)}


def half_width(n: int, repeats: int, sigma_b2: float, sigma_w2: float, rho: float, per_site: float) -> float:
    de = 1 + (per_site - 1) * rho
    return Z_TWO * math.sqrt(de * (sigma_b2 + sigma_w2 / repeats) / n)


def tasks_for_ci(target: float, repeats: int, sigma_b2: float, sigma_w2: float, rho: float, per_site: float) -> int:
    de = 1 + (per_site - 1) * rho
    return math.ceil(de * (sigma_b2 + sigma_w2 / repeats) * (Z_TWO / target) ** 2)


def tasks_for_noninferiority(margin: float, repeats: int, sigma_b2: float, sigma_w2: float, rho: float, per_site: float,
                             true_diff: float = 0.0) -> int | None:
    """Tasks so that a one-sided 95% test rejects 'loss >= margin' with 80% power when the true difference is `true_diff`."""
    gap = margin + true_diff
    if gap <= 0:
        return None
    de = 1 + (per_site - 1) * rho
    return math.ceil(de * (sigma_b2 + sigma_w2 / repeats) * ((Z_ONE + Z_POWER) / gap) ** 2)


def required_true_difference(n: int, margin: float, repeats: int, sigma_b2: float, sigma_w2: float, rho: float,
                             per_site: float) -> float:
    """True difference (A - B; positive = A better) at which the one-sided 95% non-inferiority test with `margin` has 80%
    power on n tasks: (z_a + z_b) SE - margin."""
    de = 1 + (per_site - 1) * rho
    return (Z_ONE + Z_POWER) * math.sqrt(de * (sigma_b2 + sigma_w2 / repeats) / n) - margin


def table(rows: list[list[str]], header: list[str]) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header),
                      *("| " + " | ".join(r) + " |" for r in rows)])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=ROOT / "docs" / "bench-v2-power.md")
    p.add_argument("--tasks", type=int, default=60, help="tasks in the frozen benchmark")
    p.add_argument("--sites", type=int, default=30, help="distinct sites among them (for the design effect)")
    args = p.parse_args(argv)

    runs, sites = load_v1(), task_sites()
    pairs = [
        ("find[grounder] vs find[bm25] (the v2 ranker comparison)", "find", "grounder", "find", "bm25"),
        ("find[grounder] vs vision (the v2 baseline comparison)", "find", "grounder", "vision", "grounder"),
        ("find[bm25] vs vision", "find", "bm25", "vision", "grounder"),
        ("find[grounder] vs full (list)", "find", "grounder", "full", "grounder"),
    ]
    est: list[tuple[str, str, dict[str, float]]] = []
    for agent in sorted({k[0] for k in runs}):
        for label, ca, ra, cb, rb in pairs:
            a, b = runs.get((agent, ca, ra)), runs.get((agent, cb, rb))
            if a and b:
                c = components(a, b, sites)
                if c["tasks"] >= 20:
                    est.append((agent, label, c))
    per_site = args.tasks / args.sites
    rows = [[agent, label, f"{int(c['tasks'])}", f"{c['mean_d'] * 100:+.1f}", f"{math.sqrt(c['sigma_b2']):.3f}",
             f"{c['sigma_w2']:.3f}", f"{c['icc']:.2f}"] for agent, label, c in est]
    v1_table = table(rows, ["agent", "comparison", "tasks", "mean diff (pts)", "sigma_b (true per-task difference sd)",
                            "sigma_w^2 (per-run noise)", "site ICC"])

    ranker = [c for _, l, c in est if "(the v2 ranker" in l]
    # v1 ICC estimates are noisy (38 tasks on ~17 sites, about 2 tasks per site) and meaningless where sigma_b ~ 0, so plan on
    # a judgemental rho = 0.25 (v1 estimates where the difference varies by task: see the table) and show the sensitivity.
    informative = [c["icc"] for _, _, c in est if c["sigma_b2"] > 0.01]
    rho = PLAN_RHO
    v1_b2 = float(np.max([c["sigma_b2"] for c in ranker])) if ranker else 0.0
    v1_w2 = float(np.mean([c["sigma_w2"] for c in ranker])) if ranker else 0.2
    # planning scenarios: v1 as measured, and tasks that are hard enough to separate arms
    hard_w2 = 0.5  # mean success ~0.75, task difficulty spread: E[p(1-p)] ~ 0.125 per arm, runs independent -> 0.25... x2 headroom
    scenarios = {
        "v1-like (easy tasks, ceiling effects; sigma from the v1 ranker comparison, floored)":
            (max(v1_b2, 0.0025), max(v1_w2, 0.05)),
        "ranker comparison, harder tasks (sigma_b = 0.07, sigma_w^2 = 0.25)": (0.07 ** 2, 0.25),
        "ranker vs page-view baseline, harder tasks (sigma_b = 0.20, sigma_w^2 = 0.25)": (0.20 ** 2, 0.25),
        "conservative (sigma_b = 0.30, sigma_w^2 = 0.35)": (0.30 ** 2, 0.35),
    }
    lines = ["# Agent benchmark v2: power calculation", "",
             "Plan v2.1 step 7: *run a quick power calculation before freezing the task count*. Generated by "
             "`tools/bench_v2_power.py` from the v1 logs (`runs/cli*/logs`); re-run it after the first full sweep to "
             "replace the planning scenarios with the measured v2 variance.", "",
             "## Model", "",
             "For two arms A, B and one agent, with R repeats per task and arm: `d_t = mean_r(A) - mean_r(B)`,",
             "`Var(d_t) = sigma_b^2 + sigma_w^2 / R`. sigma_b^2 is the between-task variance of the *true* difference "
             "(repeats cannot reduce it); sigma_w^2 is run noise, `E[p_A(1-p_A) + p_B(1-p_B)]` (shrinks with R). "
             f"Tasks of one site are correlated, so N tasks on S sites carry the design effect `DE = 1 + (N/S - 1) rho`; "
             f"planning values here: N = {args.tasks}, S = {args.sites} sites (m = {per_site:.1f} tasks per site), "
             f"rho = {rho:.2f} (judgement; v1 estimates where the difference varies by task: "
             f"{', '.join(f'{x:.2f}' for x in informative) or 'n/a'}).",
             "CI half-width = `1.96 sqrt(DE Var(d_t) / N)`. Non-inferiority at margin M: one-sided 95% test, 80% power, "
             "true difference 0: `N = DE Var(d_t) ((1.645 + 0.842) / M)^2`.", "",
             "## What v1 says", "",
             "v1: 38 tasks x 2 repeats; its tasks were easy (79-100% success) and the rankers tied, so v1 sigma_b is "
             "tiny for the ranker comparison and the numbers below understate v2. Per-task difference variance is "
             "estimated as `Var(d_hat_t)` minus the average within-task sampling variance (floored at 0).", "", v1_table, ""]
    for name, (b2, w2) in scenarios.items():
        lines += [f"## Scenario: {name}", "", f"sigma_b^2 = {b2:.4f}, sigma_w^2 = {w2:.3f}, rho = {rho:.2f}, m = {per_site:.1f}", ""]
        rows = []
        for n in (40, 60, 80, 120):
            rows.append([str(n)] + [f"{half_width(n, r, b2, w2, rho, per_site) * 100:.1f}" for r in (1, 2, 3, 5, 8)])
        lines += ["CI half-width (percentage points) by tasks (rows) and repeats (columns 1, 2, 3, 5, 8):", "",
                  table(rows, ["tasks", "R=1", "R=2", "R=3", "R=5", "R=8"]), ""]
        rows = []
        for r in (1, 2, 3, 5, 8):
            t5 = tasks_for_ci(0.05, r, b2, w2, rho, per_site)
            n2 = tasks_for_noninferiority(0.02, r, b2, w2, rho, per_site)
            n5 = tasks_for_noninferiority(0.05, r, b2, w2, rho, per_site)
            rows.append([str(r), str(t5), str(n2), str(n5), str(tasks_for_noninferiority(0.02, r, b2, w2, rho, per_site, -0.01))])
        lines += ["Tasks needed:", "", table(rows, ["repeats R", "CI half-width <= 5 pts", "non-inferiority, margin 2 pts",
                                                   "non-inferiority, margin 5 pts", "margin 2 pts if the true diff is -1 pt"]), ""]
    b2, w2 = scenarios["ranker comparison, harder tasks (sigma_b = 0.07, sigma_w^2 = 0.25)"]
    bb2, bw2 = scenarios["ranker vs page-view baseline, harder tasks (sigma_b = 0.20, sigma_w^2 = 0.25)"]
    e_b2, e_w2 = scenarios["v1-like (easy tasks, ceiling effects; sigma from the v1 ranker comparison, floored)"]
    n = args.tasks
    sens = [[f"{r:.2f}"] + [f"{half_width(n, R, b2, w2, r, per_site) * 100:.1f}" for R in (3, 5, 8)] for r in (0.0, 0.25, 0.5)]
    lines += ["## Recommendation", "",
              f"**Freeze {n} tasks (about {args.sites} sites) and run 3 repeats per arm; extend only the two `find` arms when "
              "the first sweep says so.**", "",
              f"* CI half-width <= 5 points for a paired difference needs {tasks_for_ci(0.05, 3, e_b2, e_w2, rho, per_site)} "
              f"tasks at R = 3 if the v2 tasks behave like v1 (ceiling effects), but {tasks_for_ci(0.05, 3, b2, w2, rho, per_site)} "
              f"(blend vs exact) and {tasks_for_ci(0.05, 3, bb2, bw2, rho, per_site)} (find vs page view) at R = 3 once success "
              f"is around 75%: with {n} tasks the half-width is "
              f"{half_width(n, 3, e_b2, e_w2, rho, per_site) * 100:.1f} / {half_width(n, 3, b2, w2, rho, per_site) * 100:.1f} / "
              f"{half_width(n, 3, bb2, bw2, rho, per_site) * 100:.1f} points in the three scenarios. Repeats only attack "
              "sigma_w^2; sigma_b^2 (tasks on which the arms truly differ) needs more tasks, not more runs.",
              f"* The 2-point non-inferiority test for blend vs exact needs "
              f"{tasks_for_noninferiority(0.02, 3, e_b2, e_w2, rho, per_site)} tasks at R = 3 even in the v1-like scenario "
              f"and {tasks_for_noninferiority(0.02, 8, b2, w2, rho, per_site)} at R = 8 with harder tasks. **{n} tasks cannot "
              "demonstrate non-inferiority at 2 points** unless the two rankers produce near-identical runs (sigma_w^2 and "
              "sigma_b^2 close to 0, as in v1 for the Sonnet and GPT-6.2 find arms). Pre-register the reading: the "
              "gate passes only if the paired CI lower bound is above -2 points; otherwise report 'not shown "
              "non-inferior' and the CI, and do not read a wide CI as equivalence.",
              f"* The 5-point margin against the best baseline needs {tasks_for_noninferiority(0.05, 3, bb2, bw2, rho, per_site)} "
              f"tasks at R = 3 in the harder scenario if the true difference is 0. With {n} tasks the test has 80% power only "
              f"if `find` is truly better than the baseline by about "
              f"{required_true_difference(n, 0.05, 3, bb2, bw2, rho, per_site) * 100:+.0f} points (R = 3) / "
              f"{required_true_difference(n, 0.05, 8, bb2, bw2, rho, per_site) * 100:+.0f} points (R = 8); in the v1-like "
              f"scenario {required_true_difference(n, 0.05, 3, e_b2, e_w2, rho, per_site) * 100:+.0f} points. The same holds for "
              f"the 2-point blend-vs-exact test: it would need the blend to be truly better by "
              f"{required_true_difference(n, 0.02, 3, b2, w2, rho, per_site) * 100:+.0f} points (R = 3, harder tasks) / "
              f"{required_true_difference(n, 0.02, 3, e_b2, e_w2, rho, per_site) * 100:+.0f} points (v1-like). "
              "Read a failed gate as 'not shown', never as 'worse'.",
              "* Because the harness resumes (a run whose record exists is skipped), the cheap extension is: run all arms at "
              "R = 3, compute the paired report, and if the blend-vs-exact half-width exceeds 5 points re-run only "
              f"`find@exact` and `find@blend` with `--repeats 8` (about 2 x {n} x 5 = {2 * n * 5} extra runs per agent). Sensitivity to the site "
              "correlation (blend vs exact, harder-task scenario, half-width in points):", "",
              table(sens, ["rho", "R=3", "R=5", "R=8"]), ""]
    open(args.out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
