"""Hand-labelling tool: write instructions for elements of held-out pages, see them on the screenshot.

    uv run python tools/label.py --pages data/pages/test --out hand_test.jsonl

Pages need the screenshot the crawler saves beside them (`{page_id}.png` next to `{page_id}.json.gz`; use
`crawl --screenshots-for test`). For each page the tool writes a self-contained HTML file (screenshot plus a
numbered box per element) to `{out stem}_html/`, opens it, and asks in the terminal:

    instruction>   what a user would say; empty line = next page, :undo = drop the last entry, :quit = stop
    element number> the number printed on the box (the element id)
    goal>          optional: a goal this step serves; adds a second example with mode "goal"

Every entry is appended to the output JSONL at once as a `records.Example` with kind "hand". Pages that
already have entries in the output file are skipped, so a re-run resumes; a page left with no entry comes
back next time. `equiv_ids` lists the elements with the same role, text and link target.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import struct
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterator

from grounder.data.pages import load_page, page_paths
from grounder.records import Example, Page, equiv_ids

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
KIND = "hand"

_STYLE = """
body{margin:0;font:14px system-ui,sans-serif;background:#222;color:#eee}
header{position:sticky;top:0;z-index:10;padding:8px 12px;background:#111;display:flex;gap:12px;align-items:center}
header input{flex:1;max-width:320px;padding:4px 8px}
.stage{position:relative;margin:0 auto;width:100%;max-width:1280px;background:#fff}
.stage img{display:block;width:100%}
.box{position:absolute;box-sizing:border-box;border:1px solid rgba(220,30,30,.8);background:rgba(220,30,30,.06)}
.box b{position:absolute;left:0;top:-1px;transform:translateY(-100%);padding:0 3px;font:600 11px monospace;
 background:#d11;color:#fff;white-space:nowrap}
.box.dim{opacity:.06}
"""
_SCRIPT = """
const filter = document.getElementById("filter");
filter.addEventListener("input", () => {
  const q = filter.value.trim().toLowerCase();
  document.querySelectorAll(".box").forEach((b) =>
    b.classList.toggle("dim", q !== "" && !b.title.toLowerCase().includes(q)));
});
"""


def png_size(data: bytes) -> tuple[int, int]:
    """(width, height) from the IHDR chunk of PNG bytes."""
    if data[:8] != PNG_SIGNATURE or data[12:16] != b"IHDR":
        raise ValueError("not a PNG file")
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def _box_html(page: Page, scale: float, img_w: int, img_h: int) -> Iterator[str]:
    for el in page.elements:
        x, y, w, h = (v * scale for v in el.abs_box)
        title = html.escape(f"{el.id} {el.role}: {el.text} | {el.context}", quote=True)
        yield (f'<div class="box" title="{title}" style="left:{x / img_w * 100:.3f}%;top:{y / img_h * 100:.3f}%;'
               f'width:{w / img_w * 100:.3f}%;height:{h / img_h * 100:.3f}%"><b>{el.id}</b></div>')


def render_page_html(page: Page, png: bytes) -> str:
    """Self-contained HTML: the screenshot with one numbered box per element, scaled to the image.

    Box positions are percentages of the image, derived from `abs_box` (document CSS pixels) times
    image width / `doc_size` width, so they stay aligned at any display width.
    """
    img_w, img_h = png_size(png)
    scale = img_w / page.doc_size[0]
    boxes = "\n".join(_box_html(page, scale, img_w, img_h))
    data = base64.b64encode(png).decode("ascii")
    return (
        f'<!doctype html><meta charset="utf-8"><title>{html.escape(page.title or page.page_id)}</title>'
        f"<style>{_STYLE}</style>"
        f'<header><span>{html.escape(page.url)} ({len(page.elements)} elements)</span>'
        f'<input id="filter" placeholder="filter by text"></header>'
        f'<div class="stage"><img src="data:image/png;base64,{data}" alt="screenshot">\n{boxes}\n</div>'
        f"<script>{_SCRIPT}</script>"
    )


def make_examples(page: Page, instruction: str, element_id: int, goal: str = "") -> list[Example]:
    """The entry for one labelled instruction: a "do" example, plus a "goal" example when a goal is given."""
    ids = equiv_ids(page.elements, element_id)
    entries = [Example(page.page_id, KIND, "do", instruction, [], element_id, ids)]
    if goal:
        entries.append(Example(page.page_id, KIND, "goal", goal, [], element_id, ids))
    return entries


def labelled_page_ids(out: Path) -> set[str]:
    if not out.exists():
        return set()
    return {json.loads(line)["page_id"] for line in out.read_text(encoding="utf-8").splitlines() if line.strip()}


def append_examples(out: Path, examples: list[Example]) -> None:
    with out.open("a", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict()) + "\n")


def drop_last_lines(out: Path, count: int) -> None:
    lines = out.read_text(encoding="utf-8").splitlines()
    out.write_text("".join(line + "\n" for line in lines[: len(lines) - count]), encoding="utf-8")


def screenshot_path(page_file: Path) -> Path:
    return page_file.with_name(page_file.name.removesuffix(".json.gz") + ".png")


def open_in_browser(path: Path) -> None:
    subprocess.run(["open", str(path)], check=True)


class Labeller:
    """The terminal loop. `ask`, `show` and `open_html` are injectable so tests can script a session."""

    def __init__(
        self, out: Path, html_dir: Path, ask: Callable[[str], str] = input,
        show: Callable[[str], None] = print, open_html: Callable[[Path], None] = open_in_browser,
    ) -> None:
        self.out, self.html_dir, self.ask, self.show, self.open_html = out, html_dir, ask, show, open_html
        self.written: list[int] = []  # examples written per entry this session, for :undo

    def run(self, pages_dir: Path) -> int:
        """Label every unlabelled page in turn; returns the number of examples written this session."""
        done = labelled_page_ids(self.out)
        todo = []
        for path in page_paths(pages_dir):
            png = screenshot_path(path)
            if png.exists():
                todo.append((path, png))
            else:
                self.show(f"skipping {path.name}: no screenshot beside it")
        for i, (path, png) in enumerate(todo, 1):
            page = load_page(path)
            if page.page_id in done:
                continue
            self.html_dir.mkdir(parents=True, exist_ok=True)
            html_path = self.html_dir / f"{page.page_id}.html"
            html_path.write_text(render_page_html(page, png.read_bytes()), encoding="utf-8")
            self.open_html(html_path)
            self.show(f"[{i}/{len(todo)}] {page.url}  ({len(page.elements)} elements) -> {html_path}")
            if self.label_page(page) == "quit":
                break
        return sum(self.written)

    def label_page(self, page: Page) -> str:
        """Prompt for entries on one page until an empty line ("next") or :quit."""
        valid = {e.id for e in page.elements}
        while True:
            instruction = self.ask("instruction> ").strip()
            if instruction == ":quit":
                return "quit"
            if not instruction:
                return "next"
            if instruction == ":undo":
                self.undo()
                continue
            element_id = self.ask_element(valid)
            if element_id is None:
                continue
            goal = self.ask("goal (optional)> ").strip()
            examples = make_examples(page, instruction, element_id, goal)
            append_examples(self.out, examples)
            self.written.append(len(examples))
            self.show(f"saved {len(examples)} example(s); {sum(self.written)} this session")

    def ask_element(self, valid: set[int]) -> int | None:
        """An element id from the user, asked again until valid; an empty line cancels."""
        while True:
            raw = self.ask("element number> ").strip()
            if not raw:
                return None
            if raw.isdigit() and int(raw) in valid:
                return int(raw)
            self.show(f"{raw!r} is not an element number on this page")

    def undo(self) -> None:
        if not self.written:
            self.show("nothing to undo in this session")
            return
        drop_last_lines(self.out, self.written.pop())
        self.show("removed the last entry")


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pages", required=True, help="pages dir of one split, e.g. data/pages/test")
    parser.add_argument("--out", required=True, help="output JSONL, e.g. hand_test.jsonl")
    args = parser.parse_args(argv)
    out = Path(args.out)
    labeller = Labeller(out, out.with_name(out.stem + "_html"))
    total = labeller.run(Path(args.pages))
    print(f"{total} entries written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
