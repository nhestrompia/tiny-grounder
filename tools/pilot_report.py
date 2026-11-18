"""Print random kept teacher examples per kind, next to the element they target, for human review.

    uv run python -m tools.pilot_report --pages data/pages/dev --examples runs/pilot-gemma/teacher.jsonl
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict

from grounder.data.pages import load_examples, load_pages
from grounder.records import Example, Page


def sample_per_kind(examples: list[Example], per_kind: int, seed: int) -> dict[str, list[Example]]:
    by_kind: dict[str, list[Example]] = defaultdict(list)
    for ex in examples:
        by_kind[ex.kind].append(ex)
    rng = random.Random(seed)
    return {kind: rng.sample(group, min(per_kind, len(group))) for kind, group in sorted(by_kind.items())}


def format_row(example: Example, page: Page) -> str:
    target = next(e for e in page.elements if e.id == example.target_id)
    return (f"  {example.query!r}\n"
            f"      -> [{target.role}] {target.text!r}  context: {target.context!r}  ({page.site})")


def report(pages: dict[str, Page], examples: list[Example], per_kind: int, seed: int) -> str:
    lines = []
    for kind, group in sample_per_kind(examples, per_kind, seed).items():
        total = sum(ex.kind == kind for ex in examples)
        lines += [f"=== {kind}: {len(group)} of {total} kept ===", *(format_row(ex, pages[ex.page_id]) for ex in group), ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pages", required=True)
    parser.add_argument("--examples", required=True)
    parser.add_argument("--per-kind", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(report(load_pages(args.pages), load_examples(args.examples), args.per_kind, args.seed))


if __name__ == "__main__":
    main()
