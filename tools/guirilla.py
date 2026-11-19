"""GUIrilla-Task -> our macOS pages and examples: the TEST split as a held-out benchmark (its only use in the release).

    uv run --extra datasets-convert python tools/guirilla.py --split test --out data/guirilla --overlap-apps
    uv run --extra datasets-convert python tools/guirilla.py --split train --out data/guirilla
    uv run python -m grounder.eval --pages data/guirilla/pages/test --examples data/guirilla/examples/test.jsonl --checkpoint ...

Source: `macpaw-research/GUIrilla-Task` (CC-BY-NC-4.0, Garkot et al. 2025, arXiv:2510.16051). The released weights are
MIT, so they must not be trained on this data: the test split is an evaluation set, and the train split exists only for
a separately licensed CC-BY-NC experiment (see docs/data-licenses.md). Only the non-image columns are read, through HTTP range requests on the parquet files (no screenshots).

`--split test` writes `{out}/pages/test/{site}/guirilla-{screen_id}.json.gz` (one page per accessibility tree) and
`{out}/examples/test.jsonl` (one "plain" / "do" example per row, query = the clean `task`).

`--split train` writes the same under `pages/train` + `examples/train.jsonl`, after two leak filters against the
test split (which is converted in memory for the comparison, a few MB):
  * rows from apps (bundle ids) that also occur in the test split are dropped, so the test stays a held-out-app
    benchmark; `--keep-test-apps` keeps them;
  * screens whose element (role, text) list equals that of a test screen are dropped (always).
About 3% of the remaining apps (a hash of the bundle id, so deterministic) go to `pages/dev` + `examples/dev.jsonl`.

Elements follow the axdump rules, see `tools/ax_tree.py`. A target node that is not kept falls back to the smallest
kept element containing the click point.

`--overlap-apps` (test split) compares app slugs with the training apps in `--train-pages`: a bundle id such as
`com.apple.Safari` is slugged by its last component ("safari") with `tools.ax_collect.slugify` (a heuristic:
`com.apple.systempreferences` does not match "system-settings"), and `examples/test_unseen_apps.jsonl` keeps
only rows from apps without training pages.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from grounder.data.pages import write_examples, write_page
from grounder.records import Example, Page, equiv_ids
from tools import convert_common
from tools.ax_collect import slugify
from tools.ax_tree import ConvertedPage, Node, element_at, node_rect, flatten_tree, tree_key

REPO = "macpaw-research/GUIrilla-Task"
SPLIT_FILES = {"test": 5, "train": 96}  # parquet files per split
URL = f"https://huggingface.co/datasets/{REPO}/resolve/main/data/{{split}}-{{k:05d}}-of-{{n:05d}}.parquet"
COLUMNS = ["screen_id", "app_name", "task", "action", "accessibility", "element_data"]
BLOCK_SIZE = 1 << 16  # small read-ahead: a larger one drags in neighbouring image bytes (7 MB instead of 1.7 MB per file)
PAGE_ID = "guirilla-{}"
APP_URL = "ax://{}"
DEV_APP_FRACTION = 0.03
DEV_BUCKETS = 10_000



@dataclass
class Dataset:
    pages: dict[str, ConvertedPage] = field(default_factory=dict)  # only pages with elements
    examples: list[Example] = field(default_factory=list)

    def target_roles(self) -> Counter[str]:
        return Counter(self.pages[ex.page_id].page.elements[ex.target_id].role for ex in self.examples)

    def select(self, page_ids: set[str]) -> Dataset:
        """The pages in `page_ids` and their examples."""
        return Dataset({i: c for i, c in self.pages.items() if i in page_ids},
                       [ex for ex in self.examples if ex.page_id in page_ids])

    def apps(self) -> set[str]:
        return {bundle_id(c.page) for c in self.pages.values()}


@dataclass
class Conversion:
    rows: int = 0
    dropped: Counter[str] = field(default_factory=Counter)
    resolved_by: Counter[str] = field(default_factory=Counter)


def convert_tree(tree: Node, screen_id: int, bundle: str) -> ConvertedPage:
    """Flatten one GUIrilla accessibility tree into a Page (site = slug of the bundle id)."""
    return flatten_tree(tree, PAGE_ID.format(screen_id), APP_URL.format(bundle), slugify(bundle))


def bundle_id(page: Page) -> str:
    return page.url.removeprefix(APP_URL.format(""))


def parse_element_data(text: str) -> Node:
    """The target node of a row: JSON in the test split, a Python dict repr (single quotes, None, True) in the train split."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


def parse_target(element_data: str) -> Node | None:
    """The clicked node of a row (None when the row has none)."""
    return parse_element_data(element_data) if element_data else None


def node_id(node: Node) -> str | None:
    """The train split calls the id `original_id` in most rows."""
    return node.get("id", node.get("original_id"))


def target_point(node: Node) -> tuple[float, float] | None:
    """Screen-point centre of the clicked node; the `left click (x, y)` action is window-relative in most train rows."""
    rect = node_rect(node)
    return (rect[0] + rect[2] / 2, rect[1] + rect[3] / 2) if rect else None


def resolve_target(converted: ConvertedPage, target: Node | None) -> tuple[int | None, str]:
    """(element id or None, how): "id" when the target node was kept, else "point" (smallest kept element at its centre), else "none"."""
    if target is None:
        return None, "none"
    if node_id(target) in converted.node_to_element:
        return converted.node_to_element[node_id(target)], "id"
    point = target_point(target)
    found = element_at(converted.page.elements, point) if point else None
    return (found, "point") if found is not None else (None, "none")


def example_for(page: Page, target_id: int, task: str) -> Example:
    return Example(page_id=page.page_id, kind="plain", mode="do", query=task, target_id=target_id,
                   equiv_ids=equiv_ids(page.elements, target_id))


def app_slug(bundle: str) -> str:
    """"com.apple.Safari" -> "safari" (last component, same slug function as tools.ax_collect)."""
    return slugify(bundle.rsplit(".", 1)[-1])


def train_app_slugs(train_pages: Path) -> set[str]:
    return {p.name for p in train_pages.iterdir() if p.is_dir()} if train_pages.is_dir() else set()


def iter_rows(split: str) -> Iterator[dict[str, Any]]:
    """The needed columns of every row of a split, read row group by row group (the image columns are never fetched)."""
    import fsspec
    import pyarrow.parquet as pq

    n = SPLIT_FILES[split]
    for k in range(n):
        with fsspec.open(URL.format(split=split, k=k, n=n), block_size=BLOCK_SIZE) as f:
            parquet = pq.ParquetFile(f)
            for i in range(parquet.num_row_groups):
                yield from parquet.read_row_group(i, columns=COLUMNS).to_pylist()


def convert_rows(rows: Iterator[dict[str, Any]]) -> tuple[Dataset, Conversion]:
    data, conv = Dataset(), Conversion()
    seen: dict[str, ConvertedPage] = {}  # every converted screen, including those without elements
    for row in rows:
        conv.rows += 1
        page_id = PAGE_ID.format(row["screen_id"])
        if page_id not in seen:
            seen[page_id] = convert_tree(json.loads(row["accessibility"]), row["screen_id"], row["app_name"])
            if seen[page_id].page.elements:
                data.pages[page_id] = seen[page_id]
        converted = seen[page_id]
        if not converted.page.elements:
            conv.dropped["no_elements"] += 1
            continue
        target_id, how = resolve_target(converted, parse_target(row["element_data"]))
        conv.resolved_by[how] += 1
        if target_id is None:
            conv.dropped["no_target"] += 1
            continue
        data.examples.append(example_for(converted.page, target_id, row["task"]))
    return data, conv


# --- test-split overlap (train leak filters) --------------------------------------------------


@dataclass
class HeldOut:
    """What the test split contains: bundle ids (case-folded) and tree keys of its screens."""

    apps: set[str]
    trees: set[str]

    @classmethod
    def of(cls, test: Dataset) -> HeldOut:
        return cls({a.casefold() for a in test.apps()}, {tree_key(c.page) for c in test.pages.values()})


def leaking_pages(data: Dataset, held_out: HeldOut) -> tuple[set[str], set[str]]:
    """(screens of test apps, other screens identical to a test screen)."""
    by_app = {i for i, c in data.pages.items() if bundle_id(c.page).casefold() in held_out.apps}
    by_tree = {i for i, c in data.pages.items() if i not in by_app and tree_key(c.page) in held_out.trees}
    return by_app, by_tree


def count_rows(data: Dataset, page_ids: set[str]) -> int:
    return sum(ex.page_id in page_ids for ex in data.examples)


def overlap_text(data: Dataset, held_out: HeldOut) -> str:
    by_app, by_tree = leaking_pages(data, held_out)
    shared = sorted({bundle_id(data.pages[i].page) for i in by_app})
    return (f"overlap with test apps: {len(shared)} / {len(data.apps())} bundle ids, {len(by_app)} / {len(data.pages)} screens, "
            f"{count_rows(data, by_app)} / {len(data.examples)} rows ({', '.join(shared)})\n"
            f"further screens identical to a test screen: {len(by_tree)}, {count_rows(data, by_tree)} rows")


def drop_leaks(data: Dataset, held_out: HeldOut, keep_test_apps: bool) -> tuple[Dataset, Counter[str]]:
    """`data` without the screens of test apps (unless `keep_test_apps`) and without screens identical to a test screen."""
    by_app, by_tree = leaking_pages(data, held_out)
    if keep_test_apps:
        by_app = set()
    dropped = Counter({"test_app_screens": len(by_app), "test_app_rows": count_rows(data, by_app),
                       "test_tree_screens": len(by_tree), "test_tree_rows": count_rows(data, by_tree)})
    return data.select(set(data.pages) - by_app - by_tree), dropped


def is_dev_app(bundle: str, fraction: float = DEV_APP_FRACTION) -> bool:
    """Deterministic ~`fraction` of the apps, by a hash of the bundle id."""
    bucket = int(hashlib.sha1(bundle.casefold().encode("utf-8")).hexdigest()[:8], 16) % DEV_BUCKETS
    return bucket < fraction * DEV_BUCKETS


def split_dev(data: Dataset, fraction: float = DEV_APP_FRACTION) -> tuple[Dataset, Dataset]:
    """(train, dev): the screens of ~`fraction` of the apps go to dev."""
    dev_ids = {i for i, c in data.pages.items() if is_dev_app(bundle_id(c.page), fraction)}
    return data.select(set(data.pages) - dev_ids), data.select(dev_ids)


# --- reports and output -------------------------------------------------------------------------


def conversion_text(conv: Conversion) -> str:
    return f"rows read: {conv.rows}  dropped rows: {dict(conv.dropped) or 0}  target resolved by: {dict(conv.resolved_by)}"


def stats_text(name: str, data: Dataset) -> str:
    return convert_common.stats_text(name, len(data.examples), len(data.apps()),
                                     [len(c.page.elements) for c in data.pages.values()], data.target_roles())


def unseen_app_report(data: Dataset, train_slugs: set[str]) -> tuple[str, list[Example]]:
    """Text report and the examples from apps without training pages."""
    seen = [app_slug(bundle_id(data.pages[ex.page_id].page)) in train_slugs for ex in data.examples]
    seen_apps = sorted({bundle_id(data.pages[ex.page_id].page) for ex, s in zip(data.examples, seen) if s})
    text = (f"rows from apps with training pages: {sum(seen)} / {len(seen)} ({len(seen_apps)} bundle ids: {', '.join(seen_apps)})\n"
            f"rows from unseen apps: {len(seen) - sum(seen)}")
    return text, [ex for ex, s in zip(data.examples, seen) if not s]


LICENSE_NOTE = """GUIrilla-Task, converted by tools/guirilla.py.

Source:   https://huggingface.co/datasets/macpaw-research/GUIrilla-Task
License:  CC-BY-NC-4.0 (attribution, non-commercial).
Use:      test split = held-out EVALUATION only. Never train the MIT-licensed release weights on any split: train / dev
          are only for a separate, CC-BY-NC-licensed experiment (see docs/data-licenses.md).
Citation: Garkot et al. 2025, "GUIrilla: A Scalable Framework for Automated Desktop UI Exploration", arXiv:2510.16051.
"""


def write_outputs(data: Dataset, out: Path, split: str) -> None:
    for converted in data.pages.values():
        write_page(converted.page, out / "pages" / split)
    (out / "examples").mkdir(parents=True, exist_ok=True)
    write_examples(data.examples, out / "examples" / f"{split}.jsonl")
    (out / "LICENSE-NOTE.txt").write_text(LICENSE_NOTE, encoding="utf-8")


def run_test(data: Dataset, out: Path, overlap_apps: bool, train_pages: Path) -> None:
    write_outputs(data, out, "test")
    print(stats_text("test", data))
    if overlap_apps:
        text, unseen = unseen_app_report(data, train_app_slugs(train_pages))
        write_examples(unseen, out / "examples" / "test_unseen_apps.jsonl")
        print(text)


def run_train(data: Dataset, out: Path, keep_test_apps: bool, dev_fraction: float) -> None:
    held_out = HeldOut.of(convert_rows(iter_rows("test"))[0])
    print(overlap_text(data, held_out))
    data, dropped = drop_leaks(data, held_out, keep_test_apps)
    print(f"dropped (keep_test_apps={keep_test_apps}): {dict(dropped)}")
    train, dev = split_dev(data, dev_fraction)
    for name, part in (("train", train), ("dev", dev)):
        write_outputs(part, out, name)
        print(stats_text(name, part))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=sorted(SPLIT_FILES), default="test")
    parser.add_argument("--out", type=Path, default=Path("data/guirilla"))
    parser.add_argument("--overlap-apps", action="store_true", help="test: report overlap with training apps, write test_unseen_apps.jsonl")
    parser.add_argument("--train-pages", type=Path, default=Path("data/pages_ax/train"))
    parser.add_argument("--keep-test-apps", action="store_true", help="train: do not drop rows from apps that occur in the test split")
    parser.add_argument("--dev-fraction", type=float, default=DEV_APP_FRACTION, help="train: share of apps held out as dev")
    args = parser.parse_args(argv)

    data, conv = convert_rows(iter_rows(args.split))
    print(conversion_text(conv))
    if args.split == "test":
        run_test(data, args.out, args.overlap_apps, args.train_pages)
    else:
        run_train(data, args.out, args.keep_test_apps, args.dev_fraction)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
