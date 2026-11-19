"""Print random kept teacher examples per kind, next to the element they target, for human review.

    uv run python -m tools.pilot_report --pages data/pages/dev --examples runs/pilot-gemma/teacher.jsonl

With `--summary` (the files written by `grounder.data.teacher --summary-out`) it also reports item coverage and
projects the cost and time of the full run from the pilot's.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from grounder.data.pages import load_examples, load_pages_from
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


FULL_RUN_CALLS = 21_149  # web train + dev + test and the native sources, at 10 elements per call
SUMMED_FIELDS = ("live_calls", "failed", "prompt_tokens", "completion_tokens", "cost_usd", "seconds",
                 "items_expected", "items_received")


def projection(summaries: Sequence[dict], full_calls: int = FULL_RUN_CALLS) -> str:
    """Pilot coverage, usage and the full run's cost and time (at the pilot's concurrency), from teacher summaries."""
    total = {name: sum(s[name] for s in summaries) for name in SUMMED_FIELDS}
    calls = total["live_calls"]
    coverage = total["items_received"] / max(total["items_expected"], 1)
    lines = [f"=== pilot usage and projection to {full_calls} calls ===",
             f"{calls} live calls, {total['prompt_tokens']} prompt and {total['completion_tokens']} completion tokens, "
             f"cost ${total['cost_usd']:.4f}, {total['seconds']:.0f} s",
             f"item coverage {coverage:.2%} ({total['items_received']}/{total['items_expected']}), "
             f"failed batches {total['failed']}"]
    if not calls:
        return "\n".join(lines + ["no live calls (everything came from the cache): nothing to project"])
    scale = full_calls / calls
    lines.append(f"full run: ~{total['prompt_tokens'] * scale / 1e6:.1f}M prompt and {total['completion_tokens'] * scale / 1e6:.1f}M "
                 f"completion tokens, ~${total['cost_usd'] * scale:.2f}, ~{total['seconds'] * scale / 3600:.1f} h")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pages", nargs="+", required=True, help="page directories, merged by page_id")
    parser.add_argument("--examples", required=True)
    parser.add_argument("--per-kind", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--summary", nargs="*", default=[], help="usage summaries written by the teacher (--summary-out)")
    parser.add_argument("--full-calls", type=int, default=FULL_RUN_CALLS)
    args = parser.parse_args()
    print(report(load_pages_from(args.pages), load_examples(args.examples), args.per_kind, args.seed))
    if args.summary:
        print(projection([json.loads(Path(p).read_text(encoding="utf-8")) for p in args.summary], args.full_calls))


if __name__ == "__main__":
    main()
