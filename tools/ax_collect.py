"""Interactive collector of macOS accessibility pages (step 13): dump whatever window you have in front.

    uv run python tools/ax_collect.py --out data/pages_ax --split train
    uv run python tools/ax_collect.py --out data/pages_ax --split test     # held-out apps, with screenshots
    uv run python tools/ax_collect.py --out data/pages_ax --stats

Per round: press Enter, switch to the window you want within the countdown, and the frontmost app is dumped with
`axdump --frontmost` and saved as `{out}/{split}/{app slug}/{page_id}.json.gz` (the layout the teacher and the
labelling tool read). Dumps with too few elements or a text signature >= 0.9 Jaccard with a page already saved for
that app are skipped. An app that already has pages in the other split is refused, so held-out apps never leak.
For `test`, a window screenshot `{page_id}.png` is saved beside the page (needs Screen Recording permission for
the terminal; without it the page is saved and a warning is shown). See docs/macos-collection.md.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from grounder.data.crawl import DEDUPE_JACCARD, TemplateIndex, text_signature
from grounder.data.pages import load_page, page_paths, write_page
from grounder.extract import ax
from grounder.records import Page

SPLITS = ("train", "test")
DEFAULT_MIN_ELEMENTS = 5
SCREENSHOT_TIMEOUT = 15.0

Dump = Callable[[], dict[str, Any]]
Screenshot = Callable[[list[float], list[int], Path], None]


def slugify(name: str) -> str:
    """A stable lowercase file-system name for an app ("Visual Studio Code" -> "visual-studio-code")."""
    return re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-") or "app"


def page_id_for(site: str, page: Page) -> str:
    """Deterministic and unique per distinct window content: site plus a hash of the text signature."""
    digest = hashlib.sha1(repr(sorted(text_signature(page))).encode("utf-8")).hexdigest()[:10]
    return f"{site}-{digest}"


def screenshot_command(origin: list[float], size: list[int], path: Path) -> list[str]:
    """`screencapture` arguments for a silent capture of the window rectangle (screen points) into `path`."""
    x, y = (round(v) for v in origin)
    w, h = size
    return ["screencapture", "-x", "-R", f"{x},{y},{w},{h}", str(path)]


def capture_screenshot(origin: list[float], size: list[int], path: Path) -> None:
    """Take the screenshot; raises RuntimeError when it fails (typically no Screen Recording permission)."""
    proc = subprocess.run(screenshot_command(origin, size, path), capture_output=True, text=True,
                          timeout=SCREENSHOT_TIMEOUT)
    if proc.returncode != 0 or not path.exists():
        raise RuntimeError(proc.stderr.strip() or f"screencapture exited with {proc.returncode}")


def role_summary(page: Page, top: int = 3) -> str:
    return ", ".join(f"{role} {n}" for role, n in Counter(e.role for e in page.elements).most_common(top))


def split_stats(out: Path) -> dict[tuple[str, str], int]:
    """Pages per (split, app) under `out`."""
    counts: Counter[tuple[str, str]] = Counter()
    for split in SPLITS:
        for path in page_paths(out / split):
            counts[(split, path.parent.name)] += 1
    return dict(counts)


def format_stats(stats: dict[tuple[str, str], int]) -> str:
    sites = sorted({site for _, site in stats})
    lines = [f"{'app':<28}{'pages':>6}  split"]
    for site in sites:
        for split in SPLITS:
            if (split, site) in stats:
                lines.append(f"{site:<28}{stats[(split, site)]:>6}  {split}")
    lines.append(f"{len(sites)} apps, {sum(stats.values())} pages")
    return "\n".join(lines)


class Collector:
    """Dumps the frontmost window and saves it under `out/split`; `dump`, `screenshot` are injectable for tests."""

    def __init__(
        self, out: Path, split: str, dump: Dump, min_elements: int = DEFAULT_MIN_ELEMENTS,
        screenshot: Screenshot = capture_screenshot, show: Callable[[str], None] = print,
    ) -> None:
        self.out, self.split, self.dump, self.min_elements = out, split, dump, min_elements
        self.screenshot, self.show = screenshot, show
        self.saved: Counter[str] = Counter()
        self.skipped: Counter[str] = Counter()
        self._seen: dict[str, TemplateIndex] = {}

    def _index(self, site: str) -> TemplateIndex:
        """Signatures of the pages already saved for `site` in this split (loaded from disk on first use)."""
        if site not in self._seen:
            index = self._seen[site] = TemplateIndex(DEDUPE_JACCARD)
            for path in page_paths(self.out / self.split / site):
                index.add(text_signature(load_page(path)))
        return self._seen[site]

    def _other_split_has(self, site: str) -> bool:
        other = next(s for s in SPLITS if s != self.split)
        return any((self.out / other / site).glob("*.json.gz"))

    def collect_once(self) -> bool:
        """Dump, validate and save the frontmost window; returns False when the session must stop."""
        try:
            raw = self.dump()
        except ax.AxdumpError as e:
            self.show(f"axdump failed: {e}")
            return e.returncode != 2  # no Accessibility permission: retrying cannot help
        app = raw.get("app") or "unknown"
        site = slugify(app)
        page = ax.page_from_dump(raw, f"{site}-pending", site)
        page.page_id = page_id_for(site, page)
        reason = self._skip_reason(page, site)
        summary = f"{app}: {len(page.elements)} elements ({role_summary(page)})"
        if reason:
            self.skipped[app] += 1
            self.show(f"{summary} -> skipped, {reason}")
            return True
        path = write_page(page, self.out / self.split)
        self._index(site).add(text_signature(page))
        self.saved[app] += 1
        warning = self._save_screenshot(raw, path) if self.split == "test" else ""
        self.show(f"{summary} -> saved {path.relative_to(self.out)}{warning}")
        return True

    def _skip_reason(self, page: Page, site: str) -> str:
        if len(page.elements) < self.min_elements:
            return f"fewer than {self.min_elements} elements"
        if self._other_split_has(site):
            other = next(s for s in SPLITS if s != self.split)
            return f"{site} already has pages in {other}; held-out apps must stay in one split"
        if self._index(site).is_duplicate(text_signature(page)):
            return "near-duplicate of a saved page"
        return ""

    def _save_screenshot(self, raw: dict[str, Any], page_path: Path) -> str:
        """Screenshot beside the page file; returns a warning suffix when it could not be taken."""
        png = page_path.with_name(page_path.name.removesuffix(".json.gz") + ".png")
        try:
            self.screenshot(raw["origin"], raw["viewport"], png)
        except (RuntimeError, OSError, subprocess.SubprocessError) as e:
            return f" (WARNING: no screenshot, grant Screen Recording to the terminal: {e})"
        return ""

    def tally(self) -> str:
        apps = sorted(set(self.saved) | set(self.skipped))
        lines = [f"{app}: {self.saved[app]} saved, {self.skipped[app]} skipped" for app in apps]
        return "\n".join(lines) or "nothing dumped"


def countdown(seconds: int, show: Callable[[str], None] = print, sleep: Callable[[float], None] = time.sleep) -> None:
    for left in range(seconds, 0, -1):
        show(f"  dumping the frontmost window in {left}...")
        sleep(1)


def run_session(
    collector: Collector, delay: int, ask: Callable[[str], str] = input,
    wait: Callable[[int], None] = countdown,
) -> None:
    while True:
        try:
            answer = ask("bring a window to front, press Enter (q to quit) ")
        except EOFError:
            return
        if answer.strip().lower() == "q":
            return
        wait(delay)
        if not collector.collect_once():
            return


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="pages dir, e.g. data/pages_ax")
    parser.add_argument("--split", choices=SPLITS, help="split of this session; held-out apps go to test")
    parser.add_argument("--delay", type=int, default=3, help="seconds to switch windows after Enter")
    parser.add_argument("--min-elements", type=int, default=DEFAULT_MIN_ELEMENTS)
    parser.add_argument("--stats", action="store_true", help="print apps x pages x split and exit")
    args = parser.parse_args(argv)
    out = Path(args.out)
    if args.stats:
        print(format_stats(split_stats(out)))
        return 0
    if args.split is None:
        parser.error("--split is required unless --stats is given")
    if not ax.AXDUMP.exists():
        print(f"{ax.AXDUMP} not found: run `swift build -c release` in {ax.AXDUMP.parents[2]}", file=sys.stderr)
        return 2
    collector = Collector(out, args.split, lambda: ax.run_axdump(["--frontmost"]), args.min_elements)
    try:
        run_session(collector, args.delay)
    except KeyboardInterrupt:
        print()
    print(collector.tally())
    return 0


if __name__ == "__main__":
    sys.exit(_main())
