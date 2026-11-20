"""Stdio MCP server: find / snapshot / act on one Playwright browser.

    grounder-tool --ranker bm25                       # headless
    grounder-tool --ranker model:dist/xs-s0 --headed

Tools: open(url), find(query, k), snapshot(goal, k), click(id), type(id, text, press_enter), go_back(),
read_text(max_chars). Element ids refer to the most recent find / snapshot. Click safety (blocked words, form
submits, password fields, stale pages) is grounder_tool.browser's, i.e. grounder.playwright_helper's.

All browser work runs on one dedicated thread (Playwright's sync API is thread-bound; MCP calls arrive on an event loop).
"""

from __future__ import annotations

import argparse
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from grounder.records import action_repr
from grounder_tool.api import UIGrounder, format_lines
from grounder_tool.browser import BrowserSession, ToolError
from grounder_tool.rankers import make_ranker

INSTRUCTIONS = (
    "Browser tools that keep page content out of your context. Do NOT read the page to look for an element: call "
    "find(query) with a short description of what you want (e.g. 'search box', 'link to the pricing page') and act on "
    "the best match. If you only know the goal, call snapshot(goal) for the 20 most relevant elements. Both return "
    "`id|role|text|context|score` lines, best first; ids refer to the most recent find or snapshot. Use read_text only "
    "when you need to read content, not to locate controls."
)
HISTORY_LEN = 3


class ToolServer:
    """Tool handlers over one browser. Call from a single thread."""

    def __init__(self, grounder: UIGrounder, headless: bool = True) -> None:
        self.grounder, self.headless = grounder, headless
        self.session = None
        self.history: list[str] = []

    def close(self) -> None:
        if self.session is not None:
            self.session.close()
            self.session = None

    def _need(self):
        if self.session is None:
            raise ToolError("no page is open: call open(url) first")
        return self.session

    def _header(self, n: int) -> str:
        return f"{self._need().status()}\n{n} elements (id|role|text|context|score):"

    def open(self, url: str) -> str:
        if not re.match(r"https?://", url):
            raise ToolError("only http(s) URLs can be opened")
        if self.session is None:
            self.session = BrowserSession(url, headless=self.headless)
        else:
            self.session.goto(url)
        self.history.clear()
        return self._need().status()

    def find(self, query: str, k: int = 5) -> str:
        page = self._extract()
        hits = self.grounder.find(page, query, k)
        return "\n".join([self._header(len(hits)), format_lines(hits)])

    def snapshot(self, goal: str, k: int = 20) -> str:
        page = self._extract()
        hits = self.grounder.snapshot(page, goal, k, self.history[-HISTORY_LEN:])
        return "\n".join([self._header(len(hits)), format_lines(hits)])

    def _extract(self):
        return self._need().extract()

    @staticmethod
    def _act(fn) -> Any:
        return fn()

    def click(self, id: int) -> str:  # noqa: A002
        session = self._need()
        element = self._act(lambda: session.click(id))
        self.history.append(action_repr(element))
        return f"clicked {id} ({element.role} {element.text!r})\n{session.status()}"

    def type(self, id: int, text: str, press_enter: bool = False) -> str:  # noqa: A002
        session = self._need()
        element = self._act(lambda: session.type_text(id, text, press_enter))
        self.history.append(action_repr(element, text))
        return f"typed into {id} ({element.text!r})\n{session.status()}"

    def go_back(self) -> str:
        session = self._need()
        self._act(session.go_back)
        return session.status()

    def read_text(self, max_chars: int = 4000) -> str:
        session = self._need()
        text = re.sub(r"\n{3,}", "\n\n", self._act(session.text)).strip()
        return text[:max_chars] + (f"\n[truncated, {len(text)} chars]" if len(text) > max_chars else "")


def build_mcp(server: ToolServer):
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("grounder", instructions=INSTRUCTIONS)
    pool = ThreadPoolExecutor(max_workers=1)

    def run(fn, *args):
        try:
            return pool.submit(fn, *args).result()
        except ToolError as exc:
            raise ValueError(str(exc)) from exc

    def open_(url: str) -> str:
        return run(server.open, url)

    def find(query: str, k: int = 5) -> str:
        return run(server.find, query, k)

    def snapshot(goal: str, k: int = 20) -> str:
        return run(server.snapshot, goal, k)

    def click(id: int) -> str:  # noqa: A002
        return run(server.click, id)

    def type_(id: int, text: str, press_enter: bool = False) -> str:  # noqa: A002
        return run(server.type, id, text, press_enter)

    def go_back() -> str:
        return run(server.go_back)

    def read_text(max_chars: int = 4000) -> str:
        return run(server.read_text, max_chars)

    for fn, name, desc in [
        (open_, "open", "Open an http(s) URL in the browser."),
        (find, "find", "Find the k (default 5) elements of the current page that best match a short description. "
                       "Use this instead of reading the page to locate links, buttons and fields."),
        (snapshot, "snapshot", "The k (default 20) elements most relevant to a goal on the current page."),
        (click, "click", "Click an element by id from the latest find/snapshot. Risky clicks (pay, delete, form submit) are refused."),
        (type_, "type", "Click a text field by id and type into it (replacing its content), optionally pressing Enter."),
        (go_back, "go_back", "Go back to the previous page."),
        (read_text, "read_text", "Visible text of the page, truncated to max_chars (default 4000). For reading content only."),
    ]:
        mcp.add_tool(fn, name=name, description=desc)
    return mcp, pool


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compact UI-grounding MCP server (stdio).")
    p.add_argument("--ranker", default="bm25", help="bm25 | model:<dir> | router:<dir>:<tau> (default bm25)")
    p.add_argument("--fp32", action="store_true", help="use model.onnx instead of model.int8.onnx")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--headed", action="store_true", help="show the browser window")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    grounder = UIGrounder(make_ranker(args.ranker, False if args.fp32 else None, args.threads))
    server = ToolServer(grounder, headless=not args.headed)
    mcp, pool = build_mcp(server)
    try:
        mcp.run("stdio")
    finally:
        pool.submit(server.close).result()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
