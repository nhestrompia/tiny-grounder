"""From crawled held-out pages to the human-eval assignment lists and writer packets (plan v2.1, step 3).

    uv run python -m tools.heldout_prepare --pages data/heldout/pages/test --out data/heldout

1. Keep a page only if it has a screenshot, at least 10 interactive elements, and does not collide with training:
   same registrable domain, same normalised URL, or a near-duplicate element-text set (Jaccard >= 0.9, the logic of
   tools/leakage_audit.py). Dropped pages and the reason go to `{out}/dropped_pages.jsonl`.
2. Split the kept pages BY SITE: about `--confirm-pages` pages (whole sites) into `{out}/confirm_pages.txt`, the
   rest into `{out}/select_pages.txt`. Each file lists page ids, interleaved across sites in a seeded random order.
3. Write one writer packet per select page to `{out}/packets/{page_id}.json` (see `packet`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grounder.data.pages import load_page, page_paths  # noqa: E402
from grounder.records import Page  # noqa: E402
from tools.label import _fractions, png_size, screenshot_path  # noqa: E402

INTERACTIVE_ROLES = frozenset(
    {"button", "link", "textbox", "searchbox", "checkbox", "radio", "switch", "combobox", "option", "tab",
     "menuitem", "slider"})
MIN_INTERACTIVE = 10
MAX_SHOT_HEIGHT = 10_000  # taller full-page screenshots are unreadable in the UI and for agents


def interactive_count(page: Page) -> int:
    return sum(e.role in INTERACTIVE_ROLES for e in page.elements)


def unusable_reason(page: Page, shot_height: int | None) -> str:
    """"" for a usable page, else why it is dropped. `shot_height` is the screenshot's height in pixels (None: no file)."""
    if shot_height is None:
        return "no screenshot"
    if shot_height > MAX_SHOT_HEIGHT:
        return f"screenshot taller than {MAX_SHOT_HEIGHT}px"
    if interactive_count(page) < MIN_INTERACTIVE:
        return f"fewer than {MIN_INTERACTIVE} interactive elements"
    return ""


def leakage_reason(page: Page, index) -> str:
    """Why a page collides with training data ("" when it does not). `index` is a leakage_audit.TrainIndex."""
    from tools.leakage_audit import page_info

    info = page_info("heldout", page.to_dict())
    if index.domain_hit(info):
        return "training domain"
    if index.url_hit(info):
        return "training URL"
    jaccard, other = index.near_duplicate(info)
    return f"near-duplicate of a training page ({other.url}, J={jaccard:.2f})" if other is not None else ""


def split_by_site(pages_by_site: dict[str, list[str]], confirm_pages: int, seed: int = 0) -> tuple[list[str], list[str]]:
    """(confirm sites, select sites): sites in a seeded order go to confirm until its page count is as close to
    `confirm_pages` as whole sites allow. A site is never in both."""
    sites = sorted(pages_by_site, key=lambda s: hashlib.sha1(f"{seed}:{s}".encode()).hexdigest())
    confirm: list[str] = []
    total = 0
    for site in sites:
        n = len(pages_by_site[site])
        if total >= confirm_pages or (total and abs(total + n - confirm_pages) > abs(total - confirm_pages)):
            break
        confirm.append(site)
        total += n
    return confirm, [s for s in sites if s not in confirm]


def interleave(pages_by_site: dict[str, list[str]], sites: list[str], seed: int = 0) -> list[str]:
    """The pages of `sites` in a seeded random order."""
    ids = sorted(pid for s in sites for pid in pages_by_site[s])
    random.Random(f"{seed}:order").shuffle(ids)
    return ids


def packet(page: Page, png_file: Path, root: Path = ROOT) -> dict:
    """What a writer agent reads: page identity, the screenshot path, and one line per element,
    `id|role|text|context|x,y,w,h` with the box in pixels of the screenshot (the 1280-px-wide full-page image)."""
    width, height = png_size(png_file.read_bytes()[:24])
    lines = []
    for e in page.elements:
        x, y, w, h = _fractions(page, e, width, height)
        box = ",".join(str(round(v)) for v in (x * width, y * height, w * width, h * height))
        lines.append("|".join([str(e.id), e.role, _clean(e.text), _clean(e.context), box]))
    try:
        shot = str(png_file.resolve().relative_to(root))
    except ValueError:
        shot = str(png_file)
    return {
        "page_id": page.page_id, "site": page.site, "url": page.url, "title": page.title,
        "screenshot": shot, "screenshot_size": [width, height], "n_elements": len(lines),
        "element_format": "id|role|text|context|x,y,w,h (box in screenshot pixels)", "elements": lines,
    }


def _clean(text: str) -> str:
    return " ".join(text.replace("|", "/").split())


def run(pages_dir: Path, out: Path, confirm_pages: int, seed: int) -> dict:
    from tools.leakage_audit import TRAIN_SOURCES, TrainIndex, iter_infos

    index = TrainIndex()
    for name, dirs in TRAIN_SOURCES.items():
        for info in iter_infos(name, dirs):
            index.add(info)
    kept: dict[str, list[str]] = {}
    dropped: list[dict] = []
    files: dict[str, Path] = {}
    for path in page_paths(pages_dir):
        page = load_page(path)
        png = screenshot_path(path)
        height = png_size(png.read_bytes()[:24])[1] if png.exists() else None
        reason = unusable_reason(page, height) or leakage_reason(page, index)
        if reason:
            dropped.append({"page_id": page.page_id, "url": page.url, "reason": reason})
        else:
            kept.setdefault(page.site, []).append(page.page_id)
            files[page.page_id] = path
    confirm_sites, select_sites = split_by_site(kept, confirm_pages, seed)
    out.mkdir(parents=True, exist_ok=True)
    for name, sites in (("confirm", confirm_sites), ("select", select_sites)):
        ids = interleave(kept, sites, seed)
        (out / f"{name}_pages.txt").write_text("".join(f"{i}\n" for i in ids), encoding="utf-8")
    (out / "dropped_pages.jsonl").write_text("".join(json.dumps(d) + "\n" for d in dropped), encoding="utf-8")
    packets = out / "packets"
    packets.mkdir(exist_ok=True)
    for stale in packets.glob("*.json"):
        stale.unlink()
    for site in select_sites:
        for pid in kept[site]:
            data = packet(load_page(files[pid]), screenshot_path(files[pid]))
            (packets / f"{pid}.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    return {"kept_pages": sum(map(len, kept.values())), "kept_sites": len(kept), "dropped": dropped,
            "confirm_sites": len(confirm_sites), "confirm_pages": sum(len(kept[s]) for s in confirm_sites),
            "select_sites": len(select_sites), "select_pages": sum(len(kept[s]) for s in select_sites)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pages", type=Path, default=Path("data/heldout/pages/test"))
    parser.add_argument("--out", type=Path, default=Path("data/heldout"))
    parser.add_argument("--confirm-pages", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    summary = run(args.pages, args.out, args.confirm_pages, args.seed)
    reasons: dict[str, int] = {}
    for d in summary.pop("dropped"):
        key = d["reason"].split(" (")[0]
        reasons[key] = reasons.get(key, 0) + 1
    print(json.dumps(summary), "dropped:", json.dumps(reasons))
    return 0


if __name__ == "__main__":
    sys.exit(main())
