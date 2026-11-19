"""Mix the training sources into data/mix/{train,dev}.jsonl with per-source weights.

    uv run python -m tools.mix --out data/mix [--weights web=0.7 desktop=0.2 mind2web=0.1] [--total N]

Sources (each example gets its `source`; GUIrilla is evaluation only and is not mixed):
    web       {--web-examples}/{train,dev}.jsonl             pages + teacher + template examples (grounder.data.build)
    macos     {--macos-examples}/train.jsonl                 data/pages_ax + data/pages_s2ax, built the same way
    deskforge {data}/deskforge/examples/{train,dev}.jsonl
    mind2web  {data}/mind2web/train.jsonl                    goal mode
Weights are per group of the epoch: `web`, `desktop` (macos and deskforge together, split by their natural sizes) and
`mind2web`. Each source is up- or down-sampled deterministically to its share of `--total` (default: the natural
total). macos and mind2web have no dev split, so a fraction of their sites is held out of train as dev.
Writes `train.jsonl`, `dev.jsonl` (all sources) and `dev_<source>.jsonl` for each source.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

from grounder.data.pages import load_examples, load_pages_from, write_examples
from grounder.records import Example, Page

DEFAULT_WEIGHTS = {"web": 0.70, "desktop": 0.20, "mind2web": 0.10}
GROUP_OF = {"web": "web", "macos": "desktop", "deskforge": "desktop", "mind2web": "mind2web"}
DEV_SITE_FRACTION = 0.1


@dataclass(frozen=True)
class Source:
    """Where one source's examples live. `dev` is None when sites must be held out of `train` (needs `pages`)."""

    name: str
    train: Path
    dev: Path | None = None
    pages: tuple[Path, ...] = ()


def default_sources(data: Path, web_examples: Path, macos_examples: Path) -> list[Source]:
    return [
        Source("web", web_examples / "train.jsonl", web_examples / "dev.jsonl"),
        Source("macos", macos_examples / "train.jsonl", pages=(data / "pages_ax" / "train", data / "pages_s2ax" / "train")),
        Source("deskforge", data / "deskforge/examples/train.jsonl", data / "deskforge/examples/dev.jsonl"),
        Source("mind2web", data / "mind2web/train.jsonl", pages=(data / "mind2web/pages/train",)),
    ]


def parse_weights(items: Sequence[str]) -> dict[str, float]:
    """`web=0.7 desktop=0.2 mind2web=0.1` -> weights per group; they must cover every group and sum to 1."""
    weights = {name: float(value) for name, _, value in (item.partition("=") for item in items)}
    if set(weights) != set(DEFAULT_WEIGHTS):
        raise ValueError(f"weights must name exactly {sorted(DEFAULT_WEIGHTS)}, got {sorted(weights)}")
    if abs(sum(weights.values()) - 1) > 1e-6:
        raise ValueError(f"weights must sum to 1, got {sum(weights.values())}")
    return weights


def holdout_sites(examples: Sequence[Example], pages: Mapping[str, Page], fraction: float, seed: int) -> tuple[list[Example], list[Example]]:
    """(train, dev): about `fraction` of the sites (at least one when there are two or more) move wholly to dev."""
    sites = sorted({pages[ex.page_id].site for ex in examples})
    n_dev = min(max(round(fraction * len(sites)), 1), len(sites) - 1) if len(sites) > 1 else 0
    dev_sites = set(random.Random(seed).sample(sites, n_dev))
    train = [ex for ex in examples if pages[ex.page_id].site not in dev_sites]
    return train, [ex for ex in examples if pages[ex.page_id].site in dev_sites]


def resample(examples: Sequence[Example], target: int, seed: int) -> list[Example]:
    """`target` examples: whole copies of `examples` plus a seeded sample of the remainder (down-sampling when target < n)."""
    if target and not examples:
        raise ValueError("cannot sample from an empty source")
    copies, rest = divmod(target, max(len(examples), 1))
    picked = sorted(random.Random(seed).sample(range(len(examples)), rest))
    return list(examples) * copies + [examples[i] for i in picked]


def source_targets(natural: Mapping[str, int], weights: Mapping[str, float], total: int) -> dict[str, int]:
    """Per-source sample sizes summing to `total`: each group gets `weight * total`, split among its sources by natural size."""
    group_natural = {g: sum(n for s, n in natural.items() if GROUP_OF[s] == g) for g in weights}
    empty = [g for g, n in group_natural.items() if n == 0 and weights[g] > 0]
    if empty:
        raise ValueError(f"no examples for weighted group(s) {empty}")
    exact = {s: weights[GROUP_OF[s]] * total * n / group_natural[GROUP_OF[s]] for s, n in natural.items()}
    targets = {s: int(x) for s, x in exact.items()}
    by_remainder = sorted(exact, key=lambda s: exact[s] - targets[s], reverse=True)
    for s in by_remainder[: total - sum(targets.values())]:  # largest remainders take the leftover
        targets[s] += 1
    return targets


def load_source(source: Source, fraction: float, seed: int) -> tuple[list[Example], list[Example]]:
    """(train, dev) of one source with `source` set on every example."""
    train = load_examples(source.train)
    if source.dev is not None:
        dev = load_examples(source.dev)
    else:
        train, dev = holdout_sites(train, load_pages_from(source.pages), fraction, seed)
    return [replace(ex, source=source.name) for ex in train], [replace(ex, source=source.name) for ex in dev]


def mix(train: Mapping[str, list[Example]], weights: Mapping[str, float], total: int, seed: int) -> list[Example]:
    """The weighted, shuffled epoch built from each source's natural train examples."""
    targets = source_targets({s: len(v) for s, v in train.items()}, weights, total)
    mixed = [ex for i, (s, v) in enumerate(train.items()) for ex in resample(v, targets[s], seed + i)]
    random.Random(seed).shuffle(mixed)
    return mixed


def count_table(train: Mapping[str, list[Example]], mixed: Sequence[Example], dev: Mapping[str, list[Example]]) -> str:
    after = {s: sum(ex.source == s for ex in mixed) for s in train}
    rows = [f"{'source':<10}{'group':<10}{'natural':>10}{'mixed':>10}{'share':>8}{'dev':>8}"]
    for s, v in train.items():
        rows.append(f"{s:<10}{GROUP_OF[s]:<10}{len(v):>10}{after[s]:>10}{after[s] / max(len(mixed), 1):>8.1%}{len(dev[s]):>8}")
    rows.append(f"{'total':<20}{sum(map(len, train.values())):>10}{len(mixed):>10}{'':>8}{sum(map(len, dev.values())):>8}")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data")
    parser.add_argument("--web-examples", help="default: {data}/examples")
    parser.add_argument("--macos-examples", help="default: {data}/examples-macos")
    parser.add_argument("--out", default="data/mix")
    parser.add_argument("--weights", nargs="+", default=[f"{g}={w}" for g, w in DEFAULT_WEIGHTS.items()], metavar="GROUP=WEIGHT")
    parser.add_argument("--total", type=int, help="examples in the mixed train file (default: the natural total)")
    parser.add_argument("--dev-site-fraction", type=float, default=DEV_SITE_FRACTION)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    data = Path(args.data)
    sources = default_sources(data, Path(args.web_examples or data / "examples"), Path(args.macos_examples or data / "examples-macos"))
    loaded = {s.name: load_source(s, args.dev_site_fraction, args.seed) for s in sources}
    train = {name: tr for name, (tr, _) in loaded.items()}
    dev = {name: dv for name, (_, dv) in loaded.items()}
    mixed = mix(train, parse_weights(args.weights), args.total or sum(map(len, train.values())), args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_examples(mixed, out / "train.jsonl")
    write_examples([ex for v in dev.values() for ex in v], out / "dev.jsonl")
    for name, v in dev.items():
        write_examples(v, out / f"dev_{name}.jsonl")
    print(count_table(train, mixed, dev))
    return 0


if __name__ == "__main__":
    sys.exit(main())
