"""Screen2AX-Tree -> our macOS pages (no instructions), to go through the LLM teacher like `data/pages_ax`.

    uv run --extra datasets-convert python tools/screen2ax.py --out data/pages_s2ax

Source: `macpaw-research/Screen2AX-Tree` (Apache-2.0, Muryn et al. 2025, arXiv:2507.16704, MacPaw). 1,127 macOS
windows as accessibility trees; only the `accessibility` column is read (HTTP range requests, no screenshots).

Node positions are relative to the window (verified: ~90% of the nodes lie inside the window size, only 18% inside
the window rectangle read as absolute); the window node's own position is its screen position. `absolutise`
turns the tree into the absolute-position form that `tools/ax_tree.py` flattens with the axdump rules.

Nothing in the dataset names the app (no app column, the README says nothing, window titles are the document or
dialog name), so the site is `s2ax-<hash of the window title>` and the page id `s2ax-<row index>`. Pages with fewer
than `--min-elements` elements are dropped, and so are pages whose element (role, text) list was already seen.
Everything goes to split `train`: `{out}/train/{site}/{page_id}.json.gz`, the layout of `data/pages_ax`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from grounder.data.pages import write_page
from grounder.records import Page
from tools.ax_collect import DEFAULT_MIN_ELEMENTS
from tools.ax_tree import Node, flatten_tree, parse_pair, tree_key

REPO = "macpaw-research/Screen2AX-Tree"
N_FILES = 2
URL = f"https://huggingface.co/datasets/{REPO}/resolve/main/data/train-{{k:05d}}-of-{N_FILES:05d}.parquet"
BLOCK_SIZE = 1 << 16  # small read-ahead: a larger one drags in neighbouring image bytes
PAGE_ID = "s2ax-{}"
SPLIT = "train"

SOURCE_NOTE = """Screen2AX-Tree, converted by tools/screen2ax.py (accessibility trees only, no screenshots, no instructions).

Source:   https://huggingface.co/datasets/macpaw-research/Screen2AX-Tree
License:  Apache-2.0
Citation: Muryn et al. 2025, "Screen2AX: Vision-Based Approach for Automatic macOS Accessibility Generation" (MacPaw),
          arXiv:2507.16704.
"""


def shifted(position: str, origin: list[float]) -> str:
    """A node's window-relative "x;y" as an absolute "x;y"; empty stays empty (a node without a frame)."""
    if not position:
        return ""
    x, y = parse_pair(position)
    return f"{origin[0] + x:.2f};{origin[1] + y:.2f}"


def absolutise(node: Node, origin: list[float]) -> Node:
    """A copy of the tree with `absolute_position` (window origin + relative position) on every node."""
    return {**node, "absolute_position": shifted(node.get("position") or "", origin),
            "children": [absolutise(child, origin) for child in node.get("children") or []]}


def convert_tree(tree: Node, row: int) -> Page:
    """One window tree -> Page; the root's `position` is the window's screen position, the origin of its children."""
    origin = parse_pair(tree["position"])
    site = "s2ax-" + hashlib.sha1((tree.get("name") or "").encode("utf-8")).hexdigest()[:8]
    root = {**absolutise(tree, origin), "absolute_position": tree["position"]}
    return flatten_tree(root, PAGE_ID.format(row), f"ax://{site}", site).page


@dataclass
class Converted:
    pages: list[Page] = field(default_factory=list)
    dropped: Counter[str] = field(default_factory=Counter)
    rows: int = 0


def convert_rows(rows: Iterator[dict[str, Any]], min_elements: int = DEFAULT_MIN_ELEMENTS) -> Converted:
    out, seen = Converted(), set()
    for index, row in enumerate(rows):
        out.rows += 1
        page = convert_tree(json.loads(row["accessibility"]), index)
        key = tree_key(page)
        if len(page.elements) < min_elements:
            out.dropped["too_few_elements"] += 1
        elif key in seen:
            out.dropped["duplicate"] += 1
        else:
            seen.add(key)
            out.pages.append(page)
    return out


def iter_rows() -> Iterator[dict[str, Any]]:
    """The `accessibility` column of every row (the image column is never fetched)."""
    import fsspec
    import pyarrow.parquet as pq

    for k in range(N_FILES):
        with fsspec.open(URL.format(k=k), block_size=BLOCK_SIZE) as f:
            parquet = pq.ParquetFile(f)
            for i in range(parquet.num_row_groups):
                yield from parquet.read_row_group(i, columns=["accessibility"]).to_pylist()


def write_outputs(pages: list[Page], out: Path) -> None:
    for page in pages:
        write_page(page, out / SPLIT)
    (out / "SOURCE.txt").write_text(SOURCE_NOTE, encoding="utf-8")


def stats_text(result: Converted) -> str:
    counts = [len(p.elements) for p in result.pages]
    p10, median, p90 = np.percentile(counts, [10, 50, 90]) if counts else (0, 0, 0)
    return (f"rows: {result.rows}  pages kept: {len(result.pages)}  sites: {len({p.site for p in result.pages})}  "
            f"dropped: {dict(result.dropped) or 0}\nelements per page: median {median:.0f}  p10 {p10:.0f}  p90 {p90:.0f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/pages_s2ax"))
    parser.add_argument("--min-elements", type=int, default=DEFAULT_MIN_ELEMENTS)
    args = parser.parse_args(argv)

    result = convert_rows(iter_rows(), args.min_elements)
    write_outputs(result.pages, args.out)
    print(stats_text(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
