"""Markdown results table (top-1 per example kind) from the metrics JSON files written by grounder.eval.

    uv run python -m tools.results_table --runs runs/train --models xs s m l bm25 --splits dev test
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


def param_count(config: Path) -> int:
    from grounder.model import Grounder, GrounderConfig

    return Grounder(GrounderConfig.from_yaml(config)).count_params()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", required=True, help="directory with <model>/<split>.json per model")
    parser.add_argument("--models", nargs="+", required=True, help="names of run subdirectories; bm25 has no config")
    parser.add_argument("--splits", nargs="+", default=["dev", "test"])
    parser.add_argument("--configs", default="configs")
    args = parser.parse_args()

    runs = Path(args.runs)
    params = {m: None if m == "bm25" else param_count(Path(args.configs) / f"{m}.yaml") for m in args.models}
    for split in args.splits:
        rows = {m: (params[m], json.loads((runs / m / f"{split}.json").read_text(encoding="utf-8"))) for m in args.models}
        print(split_table(split, rows), end="\n\n")


if __name__ == "__main__":
    main()
