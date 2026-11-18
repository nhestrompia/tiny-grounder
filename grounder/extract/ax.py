"""macOS accessibility extractor: runs the `axdump` Swift binary and returns a validated records.Page.

    cd grounder/extract/macos && swift build -c release
    uv run python -m grounder.extract.ax --app Finder

`box` is normalised by the window frame ([x/w, y/h, w/w, h/h]); `abs_box` is in screen points; `viewport`
and `doc_size` are the window size in points. The terminal needs the Accessibility permission.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from grounder.extract.web import format_table
from grounder.records import Page, validate_element

AXDUMP = Path(__file__).parent / "macos" / ".build" / "release" / "axdump"


class AxdumpError(RuntimeError):
    """axdump failed; the message is its own explanation (missing permission, unknown app, ...)."""


def page_from_dump(raw: dict[str, Any], page_id: str, site: str) -> Page:
    """Validate the JSON printed by axdump into a records.Page."""
    return Page(
        page_id=page_id, url=f"ax://{site}", site=site, title=raw.get("title", ""),
        viewport=[int(v) for v in raw["viewport"]], doc_size=[int(v) for v in raw["doc_size"]],
        elements=[validate_element(e) for e in raw["elements"]],
    )


def dump_page(
    app: str | None = None, pid: int | None = None, binary: Path = AXDUMP, timeout: float = 30.0
) -> Page:
    """Dump the front window of an app (by name or pid)."""
    if (app is None) == (pid is None):
        raise ValueError("give exactly one of app or pid")
    if not binary.exists():
        raise AxdumpError(f"{binary} not found: run `swift build -c release` in {binary.parents[2]}")
    target = ["--app", app] if app is not None else ["--pid", str(pid)]
    proc = subprocess.run([str(binary), *target], capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise AxdumpError(proc.stderr.strip() or f"axdump exited with {proc.returncode}")
    site = app or f"pid-{pid}"
    return page_from_dump(json.loads(proc.stdout), f"ax-{site}", site.lower())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--app")
    group.add_argument("--pid", type=int)
    parser.add_argument("--out", help="write the Page JSON here")
    args = parser.parse_args(argv)
    try:
        page = dump_page(args.app, args.pid)
    except AxdumpError as e:
        print(f"axdump failed: {e}", file=sys.stderr)
        return 2
    if args.out:
        Path(args.out).write_text(json.dumps(page.to_dict(), indent=1), encoding="utf-8")
    print(format_table(page))
    return 0


if __name__ == "__main__":
    sys.exit(main())
