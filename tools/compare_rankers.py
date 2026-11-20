"""Compare rankers (BM25, exact+BM25, XS, M, router, blend) with paired, group-resampled bootstrap CIs.

    uv run python tools/compare_rankers.py \
      --set "dev|data/mix/dev.jsonl|data/pages/train data/pages/dev" --set "web_test|data/examples/test.jsonl|data/pages/test" \
      --sample dev=3000 --model xs=runs/train/xs-s0/model.pt,runs/train/xs-s1/model.pt --model m=runs/train/m-s0/model.pt \
      --tune dev --report web_test --out runs/compare

Rankers: bm25, exact (exact label + BM25), one neural ranker per --model family (per seed), and per family
`router[F]` (exact when query coverage >= tau, else the model) and `blend[F]` (log p + alpha * bm25/max). tau and
alpha are chosen on the --tune set ONLY (grid search, maximise recall@5 then top-1, seed-mean) and never on a set
named guirilla*. Metrics (top-1, recall@5, MRR) are per example, null examples excluded.

Bootstrap: 2000 draws, fixed seed. Each draw resamples groups (page site, or example source) with replacement,
identically for every ranker (paired); neural rankers also draw one seed per draw, shared by rankers of the same
family (router[xs] and blend[xs] use the same checkpoints as xs), independent across families. Differences are
reported against bm25 and against the first model family.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from grounder.baselines import bm25, exact
from grounder.baselines.blend import blend
from grounder.baselines.router import example_coverage, route
from grounder.data.pages import load_examples, load_page, page_paths
from grounder.eval import shares_words
from grounder.ranking import Ranked, positive_ids
from grounder.records import Example, Page

MISSING = 10**9
TAU_GRID = (0.0, 0.2, 0.34, 0.5, 0.67, 0.75, 0.9, 1.0, 1.01)  # 0 = always exact, 1.01 = always model
ALPHA_GRID = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0)
METRICS = ("top1", "recall5", "mrr")


# ---------------------------------------------------------------- metrics

def first_rank(ranked: Ranked, positives: set[int]) -> int:
    """1-based rank of the best-ranked positive, MISSING if none is ranked."""
    return next((i + 1 for i, eid in enumerate(ranked.ids) if eid in positives), MISSING)


def metric_values(ranks: np.ndarray) -> dict[str, np.ndarray]:
    """Per-example metric values from ranks of any shape."""
    return {"top1": (ranks <= 1).astype(float), "recall5": (ranks <= 5).astype(float),
            "mrr": np.where(ranks >= MISSING, 0.0, 1.0 / np.minimum(ranks, MISSING))}


def mean_metrics(ranks: np.ndarray) -> dict[str, float]:
    """Seed-mean metrics; `ranks` is (seeds, n)."""
    if ranks.shape[-1] == 0:
        return {k: float("nan") for k in METRICS}
    return {k: float(v.mean()) for k, v in metric_values(ranks).items()}


# ---------------------------------------------------------------- bootstrap

def paired_bootstrap(
    ranks: dict[str, np.ndarray], families: dict[str, str | None], groups: np.ndarray,
    pairs: list[tuple[str, str]], draws: int = 2000, seed: int = 0, chunk: int = 250,
) -> dict[tuple[str, str], dict[str, dict[str, float]]]:
    """Paired bootstrap CIs of metric(a) - metric(b) for each pair.

    ranks[name]: (seeds, n) first-positive ranks. groups: (n,) group label per example. Each draw resamples the
    groups with replacement (same draw for every ranker) and, per family, picks one seed (rankers sharing a family
    share the seed; family None = a single fixed ranker, seed row 0). Returns {(a, b): {metric: {diff, lo, hi}}}:
    diff is the point estimate (seed-mean over all examples), lo / hi the 2.5 / 97.5 percentiles of the draws.
    """
    _, gid = np.unique(groups, return_inverse=True)
    n_groups = int(gid.max()) + 1
    counts = np.bincount(gid, minlength=n_groups).astype(float)
    sums = {
        name: {m: np.stack([np.bincount(gid, weights=v[s], minlength=n_groups) for s in range(r.shape[0])])
               for m, v in metric_values(r).items()}
        for name, r in ranks.items()
    }
    rng = np.random.default_rng(seed)
    collected: dict[tuple[str, str], dict[str, list[np.ndarray]]] = {p: {m: [] for m in METRICS} for p in pairs}
    done = 0
    while done < draws:
        d = min(chunk, draws - done)
        c = rng.multinomial(n_groups, np.full(n_groups, 1.0 / n_groups), size=d).astype(float)
        denom = c @ counts
        seed_idx: dict[str, np.ndarray] = {}
        cur: dict[str, dict[str, np.ndarray]] = {}
        for name, r in ranks.items():
            fam = families.get(name)
            if fam is None or r.shape[0] == 1:
                idx = np.zeros(d, dtype=int)
            else:
                if fam not in seed_idx:
                    seed_idx[fam] = rng.integers(0, 1 << 30, d)
                idx = seed_idx[fam] % r.shape[0]
            cur[name] = {m: (c @ s.T)[np.arange(d), idx] / denom for m, s in sums[name].items()}
        for a, b in pairs:
            for m in METRICS:
                collected[(a, b)][m].append(cur[a][m] - cur[b][m])
        done += d
    result: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
    for (a, b), per in collected.items():
        ma, mb = mean_metrics(ranks[a]), mean_metrics(ranks[b])
        result[(a, b)] = {}
        for m in METRICS:
            arr = np.concatenate(per[m])
            result[(a, b)][m] = {"diff": ma[m] - mb[m], "lo": float(np.percentile(arr, 2.5)), "hi": float(np.percentile(arr, 97.5))}
    return result


# ---------------------------------------------------------------- data

def check_tune_name(name: str) -> None:
    if name.lower().startswith("guirilla"):
        raise SystemExit(f"refusing to tune on {name!r}: GUIrilla is report-only (licence), never used for choices")


@dataclass
class EvalSet:
    name: str
    pages: dict[str, Page]
    examples: list[Example]
    bm25: list[Ranked] = field(default_factory=list)
    exact: list[Ranked] = field(default_factory=list)
    cov: np.ndarray = field(default_factory=lambda: np.zeros(0))
    shares: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    model: dict[str, list[list[Ranked]]] = field(default_factory=dict)  # family -> seed -> rankings

    def positives(self) -> list[set[int]]:
        return [positive_ids(e) for e in self.examples]


def load_pages_for(dirs: list[str], needed: set[str]) -> dict[str, Page]:
    """Only the pages whose id is needed (the file name is the page id): big train dirs are not loaded in full."""
    pages: dict[str, Page] = {}
    for d in dirs:
        for path in page_paths(d):
            pid = path.name[: -len(".json.gz")]
            if pid in needed and pid not in pages:
                pages[pid] = load_page(path)
    return pages


def load_set(spec: str, sample: int | None, seed: int = 0) -> EvalSet:
    name, examples_path, pages_dirs = spec.split("|")
    examples = [e for e in load_examples(examples_path) if e.kind != "null"]
    if sample and len(examples) > sample:
        examples = random.Random(seed).sample(examples, sample)
    pages = load_pages_for(pages_dirs.split(), {e.page_id for e in examples})
    return EvalSet(name, pages, [e for e in examples if e.page_id in pages])


def prepare_lexical(s: EvalSet) -> None:
    s.bm25 = [bm25.rank(s.pages[e.page_id], e) for e in s.examples]
    s.exact = [exact.rank(s.pages[e.page_id], e) for e in s.examples]
    s.cov = np.array([example_coverage(s.pages[e.page_id], e) for e in s.examples])
    s.shares = np.array([shares_words(s.pages[e.page_id], e) for e in s.examples], dtype=bool)


def model_rankings(s: EvalSet, ckpt: str, cache_dir: Path, device: str) -> list[Ranked]:
    ident = f"{ckpt}|{Path(ckpt).stat().st_mtime_ns}|{s.name}|{len(s.examples)}|" + "|".join(e.query for e in s.examples[:50])
    path = cache_dir / f"{hashlib.sha1(ident.encode()).hexdigest()[:16]}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    from grounder.baselines.model import ModelRanker

    out = ModelRanker(ckpt, device).rank_all(s.pages, s.examples)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(out))
    return out


# ---------------------------------------------------------------- ranker ranks

def ranks_of(s: EvalSet, rankings: list[Ranked]) -> np.ndarray:
    return np.array([first_rank(r, p) for r, p in zip(rankings, s.positives())], dtype=np.int64)


def all_ranks(s: EvalSet, tau: dict[str, float], alpha: dict[str, float]) -> tuple[dict[str, np.ndarray], dict[str, str | None]]:
    """name -> (seeds, n) first-positive ranks for every ranker, and name -> family."""
    ranks = {"bm25": ranks_of(s, s.bm25)[None], "exact": ranks_of(s, s.exact)[None]}
    fam: dict[str, str | None] = {"bm25": None, "exact": None}
    for f, seeds in s.model.items():
        ranks[f] = np.stack([ranks_of(s, r) for r in seeds])
        ranks[f"router[{f}]"] = np.stack(
            [ranks_of(s, [route(c, tau[f], lex, m) for c, lex, m in zip(s.cov, s.exact, r)]) for r in seeds])
        ranks[f"blend[{f}]"] = np.stack(
            [ranks_of(s, [blend(m, b, alpha[f]) for m, b in zip(r, s.bm25)]) for r in seeds])
        fam[f] = fam[f"router[{f}]"] = fam[f"blend[{f}]"] = f
    return ranks, fam


def tune(s: EvalSet) -> tuple[dict[str, float], dict[str, float], dict]:
    """tau and alpha per family on `s`: max seed-mean recall@5, then top-1 (ties: larger tau / smaller alpha)."""
    tau, alpha, log = {}, {}, {}
    exact_ranks = ranks_of(s, s.exact)
    for f, seeds in s.model.items():
        model_ranks = np.stack([ranks_of(s, r) for r in seeds])
        tgrid = [(t, mean_metrics(np.where((s.cov >= t)[None], exact_ranks[None], model_ranks))) for t in TAU_GRID]
        tau[f] = max(tgrid, key=lambda g: (round(g[1]["recall5"], 12), round(g[1]["top1"], 12), g[0]))[0]
        agrid = [(a, mean_metrics(np.stack([ranks_of(s, [blend(x, b, a) for x, b in zip(r, s.bm25)]) for r in seeds])))
                 for a in ALPHA_GRID]
        alpha[f] = max(agrid, key=lambda g: (round(g[1]["recall5"], 12), round(g[1]["top1"], 12), -g[0]))[0]
        log[f] = {"tau": {str(t): m for t, m in tgrid}, "alpha": {str(a): m for a, m in agrid}}
    return tau, alpha, log


# ---------------------------------------------------------------- report

def group_labels(s: EvalSet, by: str) -> np.ndarray:
    if by == "source":
        return np.array([e.source or s.name for e in s.examples])
    return np.array([s.pages[e.page_id].site for e in s.examples])


def slices(s: EvalSet) -> dict[str, np.ndarray]:
    kinds = np.array([e.kind for e in s.examples])
    out = {"all": np.ones(len(s.examples), dtype=bool), "shared words": s.shares, "no shared words": ~s.shares}
    out.update({f"kind: {k}": kinds == k for k in sorted(set(kinds))})
    return {k: v for k, v in out.items() if v.any()}


def report_set(s: EvalSet, tau: dict, alpha: dict, group_by: str, draws: int) -> dict:
    ranks, fam = all_ranks(s, tau, alpha)
    groups = group_labels(s, group_by)
    bases = ["bm25"] + list(s.model)[:1]
    out: dict = {"n": len(s.examples), "n_groups": int(len(set(groups))), "bases": bases, "slices": {}}
    for name, mask in slices(s).items():
        sub = {k: v[:, mask] for k, v in ranks.items()}
        boot = paired_bootstrap(sub, fam, groups[mask], [(k, b) for k in sub for b in bases if k != b], draws=draws)
        rows = {}
        for k, r in sub.items():
            m: dict = mean_metrics(r)
            if r.shape[0] > 1:
                per_seed = {mm: v.mean(axis=1) for mm, v in metric_values(r).items()}
                m["seed_range"] = {mm: [float(v.min()), float(v.max())] for mm, v in per_seed.items()}
            m["vs"] = {b: boot[(k, b)] for b in bases if k != b}
            rows[k] = m
        out["slices"][name] = {"n": int(mask.sum()), "rankers": rows}
    return out


def to_markdown(results: dict, tuned: dict, group_by: str, draws: int) -> str:
    L = ["# Ranker comparison", "",
         f"tau / alpha tuned on `{tuned['tune_set']}` only; bootstrap: {draws} draws, resampling `{group_by}` groups"
         " (and seeds for neural rankers), seed 0. Numbers in %; Δ = difference [95% CI].", "",
         "Tuned: " + "; ".join(f"{f}: tau={tuned['tau'][f]}, alpha={tuned['alpha'][f]}" for f in tuned["tau"]), ""]

    def cell(r: dict, b: str, m: str) -> str:
        v = r["vs"].get(b, {}).get(m)
        return "-" if v is None else f"{100 * v['diff']:+.1f} [{100 * v['lo']:+.1f}, {100 * v['hi']:+.1f}]"

    for set_name, res in results.items():
        L += [f"## {set_name} (n={res['n']}, {res['n_groups']} groups)", ""]
        bases = res["bases"]
        for sl, d in res["slices"].items():
            head = ["ranker", "top-1", "recall@5", "MRR"] + [f"Δ{m} vs {b}" for b in bases for m in ("top-1", "recall@5")]
            L += [f"### {set_name} / {sl} (n={d['n']})", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
            for k, r in d["rankers"].items():
                diffs = [cell(r, b, m) for b in bases for m in ("top1", "recall5")]
                L.append(f"| {k} | {100 * r['top1']:.1f} | {100 * r['recall5']:.1f} | {100 * r['mrr']:.1f} | " + " | ".join(diffs) + " |")
            L.append("")
    return "\n".join(L)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", action="append", required=True, help='"name|examples.jsonl|pagesdir [pagesdir ...]" (repeatable)')
    p.add_argument("--sample", action="append", default=[], help="NAME=N: random sample of N non-null examples (seed 0)")
    p.add_argument("--model", action="append", default=[], help="FAMILY=ckpt1,ckpt2,... (seeds; missing files are skipped)")
    p.add_argument("--tune", required=True, help="set name used ONLY to choose tau and alpha")
    p.add_argument("--report", nargs="*", help="set names to report on (default: all but --tune)")
    p.add_argument("--group-by", choices=["page_site", "source"], default="page_site")
    p.add_argument("--draws", type=int, default=2000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    check_tune_name(args.tune)
    import torch

    torch.set_num_threads(args.threads)
    samples = {k: int(v) for k, v in (x.split("=") for x in args.sample)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sets: dict[str, EvalSet] = {}
    for spec in args.set:
        name = spec.split("|")[0]
        sets[name] = load_set(spec, samples.get(name))
        print(f"{name}: {len(sets[name].examples)} examples", flush=True)
    if args.tune not in sets:
        p.error(f"--tune {args.tune!r} is not one of the --set names")
    families: dict[str, list[str]] = {}
    for spec in args.model:
        fam, paths = spec.split("=")
        families[fam] = [x for x in paths.split(",") if Path(x).exists()]
        print(f"{fam}: seeds {families[fam]}", flush=True)
    families = {f: c for f, c in families.items() if c}
    report = args.report if args.report is not None else [n for n in sets if n != args.tune]
    for s in sets.values():
        if s.name not in report and s.name != args.tune:
            continue
        prepare_lexical(s)
        for fam, ckpts in families.items():
            s.model[fam] = [model_rankings(s, c, out / "cache", args.device) for c in ckpts]
            print(f"{s.name}: ranked with {fam} ({len(ckpts)} seeds)", flush=True)
    tau, alpha, tune_log = tune(sets[args.tune])
    tuned = {"tune_set": args.tune, "tau": tau, "alpha": alpha, "grid": tune_log, "checkpoints": families}
    results = {n: report_set(sets[n], tau, alpha, args.group_by, args.draws) for n in report}
    (out / "compare.json").write_text(json.dumps({"tuned": tuned, "results": results}, indent=2))
    md = to_markdown(results, tuned, args.group_by, args.draws)
    (out / "compare.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
