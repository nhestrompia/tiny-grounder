"""Convert GroundCUA (ServiceNow/GroundCUA, MIT) into pages and instruction-mode ("do") examples.

Each screenshot's human-annotated elements become one page (`groundcua-<app>`). The instruction files in
`instruction_tuning.tar.gz` (~1.1M instructions: functional, text, icon, spatial, general) become examples; an
instruction's target is the annotated element with the same box, or for point-only instructions the smallest element
containing the point.

All 83 UI-Vision apps are GroundCUA apps, so UI-Vision is a seen-apps score for models trained on this
(docs/model-uniqueness-results.md). VLC Media Player, a held-out test app, is never converted.

    uv run python tools/groundcua.py --raw data/groundcua/raw --out data/groundcua
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grounder.data.pages import slugify, write_page  # noqa: E402
from grounder.records import Element, Example, Page  # noqa: E402

EXCLUDED_APPS = frozenset({"VLC Media Player"})
DEV_FRACTION = 0.1
ROLE_BY_CATEGORY = {
    "Button": "button", "Menu": "menuitem", "Input Elements": "textbox", "Information Display": "text",
    "Navigation": "link", "Sidebar": "group", "Visual Elements": "image", "Others": "other",
}
# instruction key in all_instructions -> (example kind, tag)
INSTRUCTION_KINDS = {
    "functional_instruction": ("paraphrase", "functional"),
    "text_instruction": ("plain", "text"),
    "logo_and_icon_instruction": ("plain", "icon"),
    "general_instruction": ("plain", "general"),
}
POINT_FILES = {"functional_instructions_extra.json": ("paraphrase", "functional"),
               "spatial_data.json": ("context", "spatial")}
MAX_QUERY = 300


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        head = f.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path}: not a PNG")
    return struct.unpack(">II", head[16:24])


def is_dev(app: str) -> bool:
    return int(hashlib.sha1(app.casefold().encode()).hexdigest()[:8], 16) % 1000 < DEV_FRACTION * 1000


def box_key(box: list[float]) -> tuple[int, ...]:
    return tuple(round(v) for v in box)


def build_page(app: str, image_hash: str, width: int, height: int, annotations: list[dict[str, Any]]) -> Page:
    elements = []
    for ann in annotations:
        x1, y1, x2, y2 = (float(v) for v in ann["bbox"])
        w, h = x2 - x1, y2 - y1
        text = (ann.get("text") or "").strip()
        if w <= 0 or h <= 0 or text.casefold() == "cursor":  # the mouse pointer is annotated too
            continue
        elements.append(Element(id=len(elements), role=ROLE_BY_CATEGORY.get(ann.get("category") or "", "other"),
                                text=text, tag=ann.get("category") or "",
                                box=[x1 / width, y1 / height, w / width, h / height], abs_box=[x1, y1, w, h],
                                source="groundcua"))
    site = f"groundcua-{slugify(app)}"
    return Page(page_id=f"{site}-{image_hash[:12]}", url=f"groundcua://{app}/{image_hash}.png", site=site,
                title=app, viewport=[width, height], doc_size=[width, height], elements=elements)


def element_at(page: Page, x: float, y: float) -> int | None:
    hits = [(el.abs_box[2] * el.abs_box[3], el.id) for el in page.elements
            if el.abs_box[0] <= x <= el.abs_box[0] + el.abs_box[2] and el.abs_box[1] <= y <= el.abs_box[1] + el.abs_box[3]]
    return min(hits)[1] if hits else None


def iter_instruction_files(tar_path: Path) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    with tarfile.open(tar_path) as tar:
        for member in tar:
            if member.isfile() and member.name.endswith(".json"):
                f = tar.extractfile(member)
                if f is not None:
                    yield Path(member.name).name, json.load(f)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=Path("data/groundcua/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/groundcua"))
    args = ap.parse_args(argv)
    counts: Counter[str] = Counter()

    pages: dict[str, Page] = {}  # image hash -> page
    by_box: dict[str, dict[tuple[int, ...], int]] = {}
    for ann_path in sorted((args.raw / "data").glob("*/*.json")):
        app = ann_path.parent.name
        if app in EXCLUDED_APPS:
            counts["skip_excluded_app_page"] += 1
            continue
        image = args.raw / "images" / app / f"{ann_path.stem}.png"
        if not image.exists():
            counts["skip_missing_image"] += 1
            continue
        width, height = png_size(image)
        page = build_page(app, ann_path.stem, width, height, json.loads(ann_path.read_text()))
        if not page.elements:
            counts["skip_empty_page"] += 1
            continue
        write_page(page, args.out / "pages")
        pages[ann_path.stem] = page
        by_box[ann_path.stem] = {box_key([e.abs_box[0], e.abs_box[1], e.abs_box[0] + e.abs_box[2],
                                          e.abs_box[1] + e.abs_box[3]]): e.id for e in page.elements}
        counts["pages"] += 1
        if counts["pages"] % 5000 == 0:
            print(f"[groundcua] {counts['pages']} pages", flush=True)

    outs = {s: (args.out / f"{s}.jsonl").open("w", encoding="utf-8") for s in ("train", "dev")}
    apps: dict[str, Counter[str]] = {"train": Counter(), "dev": Counter()}
    seen: set[tuple[str, int, str]] = set()
    for name, records in iter_instruction_files(args.raw / "instruction_tuning.tar.gz"):
        for rec in records:
            page = pages.get(rec.get("id") or "")
            if page is None:
                counts["drop_no_page" if rec.get("platform") not in EXCLUDED_APPS else "drop_excluded_app"] += 1
                continue
            queries: list[tuple[str, str, str]] = []
            if name in POINT_FILES:
                x, y = rec["cordinate"]
                target = element_at(page, float(x), float(y))
                kind, tag = POINT_FILES[name]
                queries.append((rec.get("instruction") or "", kind, tag))
            else:
                target = by_box[rec["id"]].get(box_key(rec["bbox"]))
                for key, text in (rec.get("all_instructions") or {}).items():
                    if key in INSTRUCTION_KINDS and isinstance(text, str):
                        queries.append((text, *INSTRUCTION_KINDS[key]))
            if target is None:
                counts["drop_no_target"] += 1
                continue
            split = "dev" if is_dev(page.title) else "train"
            for query, kind, tag in queries:
                query = " ".join(query.split())[:MAX_QUERY]
                key = (page.page_id, target, query.casefold())
                if not query or key in seen:
                    counts["drop_empty_or_duplicate"] += 1
                    continue
                seen.add(key)
                ex = Example(page_id=page.page_id, kind=kind, mode="do", query=query, target_id=target,
                             equiv_ids=[target], source="groundcua", tags=["groundcua", tag])
                outs[split].write(json.dumps(ex.to_dict()) + "\n")
                counts[f"examples_{split}"] += 1
                counts[f"tag_{tag}"] += 1
                apps[split][page.title] += 1
        print(f"[groundcua] {name}: {dict(counts)}", flush=True)
    for f in outs.values():
        f.close()
    report = {"counts": dict(counts), "apps": {s: dict(a.most_common()) for s, a in apps.items()},
              "excluded_apps": sorted(EXCLUDED_APPS)}
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"[groundcua] done: {dict(counts)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
