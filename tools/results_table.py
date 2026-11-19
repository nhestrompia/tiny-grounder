"""Markdown results tables from the metrics JSON files written by grounder.eval.

    uv run python -m tools.results_table --runs runs/train --models xs-s0 s-s0 bm25 --splits dev web_test guirilla_test

One summary table has a column group (top-1, top-5) per eval set; then one table per set with top-1 per example kind.
A run directory is named `<config>-s<seed>` (e.g. `m-s1`) or is `bm25`; a set a run has no file for shows `-`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def top1_row(name: str, params: int | None, metrics: dict[str, dict], kinds: list[str]) -> str:
    cells = [f"{metrics['by_kind'][k]['top1']:.3f}" if k in metrics["by_kind"] else "-" for k in kinds]
    cells.append(f"{metrics['overall']['top1']:.3f}")
    return f"| {name} | {'-' if params is None else f'{params:,}'} | " + " | ".join(cells) + " |"


def split_table(split: str, rows: dict[str, tuple[int | None, dict[str, dict]]]) -> str:
    """One table of top-1 for `split`; `rows` maps a model name to (params, metrics)."""
    kinds = sorted({k for _, m in rows.values() for k in m["by_kind"]})
    header = f"| {split} top-1 | params | " + " | ".join(kinds) + " | overall |"
    divider = "|---|---|" + "---|" * (len(kinds) + 1)
    return "\n".join([header, divider, *(top1_row(name, p, m, kinds) for name, (p, m) in rows.items())])


def summary_table(splits: list[str], rows: dict[str, tuple[int | None, dict[str, dict]]]) -> str:
    """One row per run; `rows` maps a run name to (params, {split: metrics}). Columns: top-1 and top-5 per split."""
    header = "| model | params | " + " | ".join(f"{s} top-1 | {s} top-5" for s in splits) + " |"
    divider = "|---|---|" + "---|" * (2 * len(splits))
    lines = [header, divider]
    for name, (params, by_split) in rows.items():
        cells = []
        for split in splits:
            overall = by_split.get(split, {}).get("overall")
            cells += [f"{overall['top1']:.3f}", f"{overall['top5']:.3f}"] if overall else ["-", "-"]
        lines.append(f"| {name} | {'-' if params is None else f'{params:,}'} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def config_name(run: str) -> str:
    """`m-s1` -> `m`: the config a run directory was trained from (`bm25` has none)."""
    return run.partition("-s")[0]


def param_count(config: Path) -> int:
    from grounder.model import Grounder, GrounderConfig

    return Grounder(GrounderConfig.from_yaml(config)).count_params()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", required=True, help="directory with <model>/<split>.json per model")
    parser.add_argument("--models", nargs="+", required=True, help="names of run subdirectories; bm25 has no config")
    parser.add_argument("--splits", nargs="+", default=["dev", "test"], help="eval-set names: files <model>/<name>.json")
    parser.add_argument("--configs", default="configs")
    args = parser.parse_args()

    runs = Path(args.runs)
    params = {m: None if m == "bm25" else param_count(Path(args.configs) / f"{config_name(m)}.yaml") for m in args.models}
    metrics = {m: {s: json.loads(f.read_text(encoding="utf-8")) for s in args.splits if (f := runs / m / f"{s}.json").exists()}
               for m in args.models}
    print(summary_table(args.splits, {m: (params[m], metrics[m]) for m in args.models}), end="\n\n")
    for split in args.splits:
        rows = {m: (params[m], metrics[m][split]) for m in args.models if split in metrics[m]}
        if rows:
            print(split_table(split, rows), end="\n\n")


if __name__ == "__main__":
    main()
