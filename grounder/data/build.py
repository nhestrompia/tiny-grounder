"""Assemble train / dev / test example files from crawled pages and teacher output.

    uv run python -m grounder.data.build --pages data/pages --out data/examples \
        --llm-examples data/teacher_train.jsonl data/teacher_dev.jsonl

Pages come from `{pages}/{split}/...` (see grounder.data.crawl). Template examples are generated
here; LLM examples are read from the JSONL files written by grounder.data.teacher and used for the
split their page belongs to. Everything passes through grounder.data.filters, then ~10% of the
examples are copied as `null` examples (target and equivalents deleted from the page).
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Mapping

from grounder.data.crawl import SPLITS
from grounder.data.filters import filter_all, format_rejections
from grounder.data.pages import iter_pages, load_examples, write_examples
from grounder.data.template_teacher import SAMPLE_SIZE, template_examples
from grounder.records import Example, Page

NULL_FRACTION = 0.1
NULL_SOURCE_KINDS = ("plain", "paraphrase", "goal")


def can_become_null(example: Example, page: Page) -> bool:
    """True when no element left on the page still carries the target's label.

    Otherwise a query like "click Add to cart" would still have a correct answer after deletion.
    """
    gone = set(example.equiv_ids) | {example.target_id}
    target = next((e for e in page.elements if e.id == example.target_id), None)
    if target is None or example.kind not in NULL_SOURCE_KINDS:
        return False
    label = target.text.casefold()
    return not label or all(e.id in gone for e in page.elements if e.text.casefold() == label)


def add_null_examples(examples: list[Example], pages: Mapping[str, Page], fraction: float, seed: int) -> list[Example]:
    """`examples` plus null copies of about `fraction` of them.

    A null copy keeps `target_id` and `equiv_ids`; they name the elements removed from the page
    (see grounder.ranking.removed_ids).
    """
    rng = random.Random(seed)
    eligible = [ex for ex in examples if can_become_null(ex, pages[ex.page_id])]
    chosen = rng.sample(eligible, min(len(eligible), round(fraction * len(examples))))
    return examples + [replace(ex, kind="null") for ex in chosen]


def count_table(counts: Mapping[str, Counter[str]]) -> str:
    kinds = sorted({k for c in counts.values() for k in c})
    rows = [f"{'split':<6}" + "".join(f"{k:>12}" for k in kinds) + f"{'total':>12}"]
    for split, c in counts.items():
        rows.append(f"{split:<6}" + "".join(f"{c[k]:>12}" for k in kinds) + f"{sum(c.values()):>12}")
    return "\n".join(rows)


def build_split(pages: Mapping[str, Page], llm: list[Example], args: argparse.Namespace) -> tuple[list[Example], Counter[str]]:
    raw = [ex for ex in llm if ex.page_id in pages]
    if args.template:
        for page in pages.values():
            raw += template_examples(page, args.seed, args.per_page)
    result = filter_all(pages, raw)
    return add_null_examples(result.kept, pages, args.null_fraction, args.seed), result.rejected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build train/dev/test example files.")
    parser.add_argument("--pages", default="data/pages")
    parser.add_argument("--out", default="data/examples")
    parser.add_argument("--template", action=argparse.BooleanOptionalAction, default=True,
                        help="generate template-teacher examples")
    parser.add_argument("--llm-examples", nargs="*", default=[], help="JSONL files written by grounder.data.teacher")
    parser.add_argument("--per-page", type=int, default=SAMPLE_SIZE, help="elements sampled per page by the template teacher")
    parser.add_argument("--null-fraction", type=float, default=NULL_FRACTION)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    llm = [ex for path in args.llm_examples for ex in load_examples(path)]
    counts: dict[str, Counter[str]] = {}
    rejected: Counter[str] = Counter()
    for split in SPLITS:
        split_dir = Path(args.pages) / split
        pages = {p.page_id: p for p in iter_pages(split_dir)} if split_dir.is_dir() else {}
        examples, rejects = build_split(pages, llm, args)
        write_examples(examples, out / f"{split}.jsonl")
        counts[split] = Counter(ex.kind for ex in examples)
        rejected.update(rejects)
    print(count_table(counts))
    print(f"rejected: {format_rejections(rejected)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
