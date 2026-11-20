"""Turn the writers' raw select items into evaluation examples, with checks.

    uv run python tools/heldout_build.py --raw data/heldout/select_raw_*.jsonl --out data/heldout/select.jsonl

Raw lines: {"page_id", "query", "target_ids": [...] or [], "mode": "do"|"goal", "style", "target_type", "tags",
"annotator"}. Each becomes a `records.Example`: kind "hand" with the first target as `target_id` and `equiv_ids`
= the listed targets plus their equivalents (same role, text and link), or kind "null" when no target is listed.
`source` is the page's site, so the evaluation can group by site; style / target_type / tags / annotator are kept
as extra fields. Items whose page or target ids do not exist, or duplicate queries on a page, are dropped and
reported.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from grounder.data.pages import load_pages
from grounder.records import Example, equiv_ids

HARD_STYLES = {"paraphrase", "goal"}


def is_hard(item: dict) -> bool:
    """The plan's hard slice: paraphrase, goal or an icon/unlabelled target."""
    return item.get("style") in HARD_STYLES or item.get("target_type") == "icon"


def build(raw_items: list[dict], pages: dict) -> tuple[list[dict], Counter]:
    out, problems, seen = [], Counter(), set()
    for item in raw_items:
        page = pages.get(item.get("page_id"))
        if page is None:
            problems["unknown page"] += 1
            continue
        query = (item.get("query") or "").strip()
        key = (page.page_id, query.lower())
        if not query or key in seen:
            problems["empty or duplicate query"] += 1
            continue
        ids = {e.id for e in page.elements}
        targets = [int(t) for t in item.get("target_ids") or []]
        if any(t not in ids for t in targets):
            problems["unknown target id"] += 1
            continue
        seen.add(key)
        mode = item.get("mode") if item.get("mode") in ("do", "goal") else "do"
        if targets:
            equiv = sorted({e for t in targets for e in equiv_ids(page.elements, t)})
            ex = Example(page.page_id, "hand", mode, query, [], targets[0], equiv, source=page.site)
        else:
            ex = Example(page.page_id, "null", mode, query, [], None, [], source=page.site)
        row = ex.to_dict() | {
            "style": item.get("style", ""), "target_type": item.get("target_type", ""), "tags": item.get("tags", []),
            "annotator": item.get("annotator", ""), "hard": is_hard(item), "split": "select",
        }
        out.append(row)
    return out, problems


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", nargs="+", required=True)
    p.add_argument("--pages", default="data/heldout/pages/test")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    raw = [json.loads(line) for f in args.raw for line in Path(f).read_text().splitlines() if line.strip()]
    rows, problems = build(raw, load_pages(args.pages))
    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    styles = Counter((r["annotator"], r["style"]) for r in rows)
    print(f"{len(raw)} raw -> {len(rows)} examples; dropped: {dict(problems) or 'none'}")
    print(f"hard {sum(r['hard'] for r in rows)}, null {sum(r['kind'] == 'null' for r in rows)}, "
          f"icon {sum(r['target_type'] == 'icon' for r in rows)}, sites {len({r['source'] for r in rows})}")
    for (who, style), n in sorted(styles.items()):
        print(f"  {who:20s} {style:10s} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
