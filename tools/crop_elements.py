"""Replay frozen pages and crop element screenshots.

Prototype audit:

    uv run --extra demo python tools/crop_elements.py audit \
      --source-root /path/to/tiny-grounder --decisions data/decisions --rows 20
"""

from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grounder.data.pages import load_page
from tools.decision_labels import SOURCE_TASKS

VIEWPORT_W = 1280
VIEWPORT_H = 800


def crop_elements(page_url: str, har: str | Path, element_boxes: Iterable[Iterable[float]]) -> list[bytes]:
    """Return one PNG crop per `[x, y, width, height]` document box, replaying `page_url` from `har`.

    The page is scrolled before every crop so boxes below the fold are captured in viewport coordinates. Crops are
    clipped at viewport edges; boxes with no visible area return no entry.
    """
    from demo.browser import BrowserSession
    from PIL import Image

    crops: list[bytes] = []
    with BrowserSession(page_url, har=har) as browser:
        page = browser.page
        for raw in element_boxes:
            x, y, w, h = [float(v) for v in raw]
            if w <= 0 or h <= 0:
                continue
            browser._guard(lambda: page.evaluate(
                "([x, y]) => window.scrollTo(Math.max(0, x - 80), Math.max(0, y - 80))", [x, y],
            ))
            sx, sy = browser._guard(lambda: page.evaluate("[window.scrollX, window.scrollY]"))
            vx, vy = x - float(sx), y - float(sy)
            left, top = max(0.0, vx), max(0.0, vy)
            right, bottom = min(float(VIEWPORT_W), vx + w), min(float(VIEWPORT_H), vy + h)
            if right - left < 1 or bottom - top < 1:
                continue
            png = browser._guard(lambda: page.screenshot(
                type="png", clip={"x": left, "y": top, "width": right - left, "height": bottom - top},
            ))
            # Playwright can return fractional-device-pixel edge cases on some platforms. Re-encoding through PIL
            # keeps the prototype output strictly decodable and makes failures explicit.
            image = Image.open(BytesIO(png))
            buf = BytesIO()
            image.save(buf, format="PNG")
            crops.append(buf.getvalue())
    return crops


def load_rows(decisions: Path) -> list[dict[str, Any]]:
    rows = []
    for split in ("train", "dev"):
        path = decisions / "next" / f"{split}.jsonl"
        if path.exists():
            rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return rows


def tasks_by_run_dir() -> dict[str, dict[str, Any]]:
    from demo.browser import load_tasks

    by_file = {}
    for tasks_file in sorted(set(SOURCE_TASKS.values())):
        by_file[tasks_file] = {task.id: task for task in load_tasks(ROOT / tasks_file)}
    return {run_dir: by_file[tasks_file] for run_dir, tasks_file in SOURCE_TASKS.items()}


def page_file(decisions: Path, row: dict[str, Any]) -> Path | None:
    pid = row["page_id"]
    matches = list((decisions / "pages").glob(f"*/{pid}.json.gz"))
    return matches[0] if matches else None


def select_rows(rows: list[dict[str, Any]], decisions: Path, count: int) -> list[dict[str, Any]]:
    """Pick up to half the audit rows with an empty-text primary target and half with visible text."""
    icon, text = [], []
    for row in rows:
        path = page_file(decisions, row)
        if path is None:
            continue
        page = load_page(path)
        target = next((e for e in page.elements if e.id == row["label"]["target_id"]), None)
        if target is None:
            continue
        enriched = dict(row)
        enriched["_page_path"] = str(path)
        enriched["_icon_only"] = not bool(target.text.strip())
        (icon if enriched["_icon_only"] else text).append(enriched)
    half = count // 2
    chosen = icon[:half] + text[:count - half]
    remaining = [r for r in icon[half:] + text[count - half:]]
    for row in remaining:
        if len(chosen) >= count:
            break
        chosen.append(row)
    return chosen[:count]


def audit(args: argparse.Namespace) -> dict[str, Any]:
    decisions = Path(args.decisions).resolve()
    source_root = Path(args.source_root).resolve()
    rows = select_rows(load_rows(decisions), decisions, args.rows)
    tasks = tasks_by_run_dir()
    crop_dir = decisions / "crop-audit"
    crop_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index, row in enumerate(rows):
        page = load_page(row["_page_path"])
        ids = row["label"].get("target_ids") or [row["label"]["target_id"]]
        elements = [e for e in page.elements if e.id in ids]
        boxes = [e.abs_box for e in elements]
        task = tasks[row["provenance"]["run_dir"]][row["provenance"]["task_id"]]
        har = source_root / (task.har or f"data/bench_v2/har/{task.id}.zip")
        start = time.perf_counter()
        error = ""
        crops: list[bytes] = []
        try:
            crops = crop_elements(row["provenance"]["url"], har, boxes)
        except Exception as exc:  # noqa: BLE001 - audit every row and report failures
            error = f"{type(exc).__name__}: {exc}"[:300]
        seconds = time.perf_counter() - start
        paths = []
        for crop_index, png in enumerate(crops):
            path = crop_dir / f"{index:02d}-{crop_index}-{row['page_id']}.png"
            path.write_bytes(png)
            paths.append(str(path))
        results.append({
            "row": index, "page_id": row["page_id"], "task_id": row["provenance"]["task_id"],
            "url": row["provenance"]["url"], "icon_only": row["_icon_only"], "crop_count": len(crops),
            "requested_crops": len(boxes), "seconds": seconds, "error": error, "paths": paths,
        })
    report: dict[str, Any] = {"rows": len(results), "results": results, "groups": {}}
    for name, subset in {
        "icon_only": [r for r in results if r["icon_only"]],
        "visible_text": [r for r in results if not r["icon_only"]],
    }.items():
        times = [r["seconds"] for r in subset]
        counts = [r["crop_count"] for r in subset]
        report["groups"][name] = {
            "rows": len(subset),
            "crops": sum(counts),
            "errors": sum(bool(r["error"]) for r in subset),
            "mean_seconds_per_row": statistics.mean(times) if times else 0.0,
            "median_seconds_per_row": statistics.median(times) if times else 0.0,
            "mean_crops_per_row": statistics.mean(counts) if counts else 0.0,
        }
    out = decisions / "crop-audit.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    au = sub.add_parser("audit", help="crop target elements for a small NEXT-row sample")
    au.add_argument("--source-root", type=Path, required=True)
    au.add_argument("--decisions", type=Path, default=ROOT / "data/decisions")
    au.add_argument("--rows", type=int, default=20)
    args = parser.parse_args(argv)
    if args.command == "audit":
        report = audit(args)
        print(json.dumps(report["groups"], indent=2))


if __name__ == "__main__":
    main()
