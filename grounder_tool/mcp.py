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
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from grounder.playwright_helper import Passage, extract_passages
from grounder_tool.digest import read as read_passages
from grounder.records import Element, Page, action_repr
from grounder_tool.api import UIGrounder, format_lines
from grounder_tool.browser import BrowserSession, ToolError
from grounder_tool.rankers import make_ranker
from grounder_tool.digest import MAX_BUDGET, MIN_BUDGET, LineSuffix

INSTRUCTIONS = (
    "Call find(query) with a short description to locate a control. "
    "Call observe(goal) after navigation for a compact page digest, or inspect(id) for details. "
    "Element lines use `id|role|text|context` with scores for find/snapshot. "
    "Ids refer to the latest extraction. Use read(query) to find facts in page text."
)
CDP_INSTRUCTIONS = (
    "This server is perception-only. Call observe(goal) for a compact page digest, "
    "find(query) to locate a control, or inspect(id) for details. "
    "Element lines use `id|role|text|context`; ids refer to the latest extraction. "
    "The @x,y suffix gives viewport coordinates for the executor's click. "
    "Use read(query) to find facts in page text."
)
HISTORY_LEN = 3


class ToolServer:
    """Tool handlers over one browser. Call from a single thread."""

    def __init__(self, grounder: UIGrounder, headless: bool = True, observe_grounder: UIGrounder | None = None,
                 cdp_endpoint: str = "", cdp_page_url_substring: str = "") -> None:
        self.grounder, self.headless = grounder, headless
        self.observe_grounder = observe_grounder or grounder
        self.cdp_endpoint, self.cdp_page_url_substring = cdp_endpoint, cdp_page_url_substring
        self.extract_ms = self.rank_ms = 0.0
        self.session = None
        self.history: list[str] = []

    def close(self) -> None:
        if self.session is not None:
            self.session.close()
            self.session = None

    def _need(self) -> BrowserSession:
        if self.session is None and self.cdp_endpoint:
            self.session = BrowserSession.attach(self.cdp_endpoint, self.cdp_page_url_substring)
        if self.session is None:
            raise ToolError("no page is open: call open(url) first")
        return self.session

    def _header(self, n: int) -> str:
        return f"{self._need().status()}\n{n} elements (id|role|text|context|score):"

    def open(self, url: str) -> str:
        if self.cdp_endpoint:
            raise ToolError("CDP attach is perception-only")
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
        start = time.perf_counter()
        hits = self.grounder.find(page, query, k)
        self.rank_ms = (time.perf_counter() - start) * 1000
        return self._coordinates("\n".join([self._header(len(hits)), format_lines(hits)]))

    def snapshot(self, goal: str, k: int = 20) -> str:
        page = self._extract()
        start = time.perf_counter()
        hits = self.grounder.snapshot(page, goal, k, self.history[-HISTORY_LEN:])
        self.rank_ms = (time.perf_counter() - start) * 1000
        return self._coordinates("\n".join([self._header(len(hits)), format_lines(hits)]))

    def observe(self, goal: str = "", budget: int = 600) -> str:
        """Return a goal-aware digest within the estimated token budget, including coordinates."""
        page = self._extract()
        passages = self._passages()
        start = time.perf_counter()
        out = self.observe_grounder.observe(
            page, goal, max(MIN_BUDGET, min(MAX_BUDGET, budget)), self.history[-HISTORY_LEN:],
            landmarks=self._need().landmarks, line_suffix=self._coordinate_suffix(), passages=passages,
        )
        self.rank_ms = (time.perf_counter() - start) * 1000
        return out

    def _passages(self) -> list[Passage]:
        start = time.perf_counter()
        passages = self._need()._guard(lambda: extract_passages(self._need().page))
        self.extract_ms += (time.perf_counter() - start) * 1000
        return passages

    def read(self, query: str, budget: int = 400) -> str:
        passages = self._passages()
        start = time.perf_counter()
        out = read_passages(query, passages, budget)
        self.rank_ms = (time.perf_counter() - start) * 1000
        return out

    def inspect(self, id: int | None = None, region: str = "") -> str:
        """Return element or section details within the default 400-token budget."""
        page = self._extract()
        start = time.perf_counter()
        out = self.observe_grounder.inspect(
            page, id, region, landmarks=self._need().landmarks,
            heading_positions=self._need().heading_positions, line_suffix=self._coordinate_suffix(),
        )
        self.rank_ms = (time.perf_counter() - start) * 1000
        return out

    def _extract(self) -> Page:
        start = time.perf_counter()
        page = self._need().extract()
        self.extract_ms = (time.perf_counter() - start) * 1000
        return page

    def _coordinate_suffix(self) -> LineSuffix | None:
        if not self.cdp_endpoint:
            return None
        sx, sy = self._need().page.evaluate("() => [scrollX, scrollY]")

        def suffix(element: Element) -> str:
            x, y, w, h = element.abs_box
            return f" @{x + w / 2 - sx:g},{y + h / 2 - sy:g}"

        return suffix

    def _coordinates(self, text: str) -> str:
        suffix = self._coordinate_suffix()
        if suffix is None:
            return text
        elements = {e.id: e for e in self._need().elements}
        lines = []
        for line in text.splitlines():
            identifier = line.partition("|")[0]
            if identifier.isdigit() and int(identifier) in elements:
                line += suffix(elements[int(identifier)])
            lines.append(line)
        return "\n".join(lines)

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
        if self.cdp_endpoint:
            self._extract()
        session = self._need()
        text = re.sub(r"\n{3,}", "\n\n", self._act(session.text)).strip()
        return text[:max_chars] + (f"\n[truncated, {len(text)} chars]" if len(text) > max_chars else "")


def build_mcp(server: ToolServer):
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("grounder", instructions=CDP_INSTRUCTIONS if server.cdp_endpoint else INSTRUCTIONS)
    pool = ThreadPoolExecutor(max_workers=1)

    def run(fn, *args):
        def timed():
            server.extract_ms = server.rank_ms = 0.0
            try:
                return fn(*args)
            finally:
                logging.getLogger(__name__).info(
                    "%s extract_ms=%.3f rank_ms=%.3f", fn.__name__, server.extract_ms, server.rank_ms
                )

        try:
            return pool.submit(timed).result()
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

    def observe(goal: str = "", budget: int = 600) -> str:
        return run(server.observe, goal, budget)

    def inspect(id: int | None = None, region: str = "") -> str:
        return run(server.inspect, id, region)

    def read(query: str, budget: int = 400) -> str:
        return run(server.read, query, budget)

    for fn, name, desc in [
        (read, "read", "Read the page text most relevant to a question; "
         "use it to find facts on the current page instead of guessing."),
        (observe, "observe", "Compact page header, headings, relevant text and goal-ranked controls. "
         "Budget is clamped to 100-2000 tokens. "
         "Call after navigation to choose the next action; an empty goal uses the title."),
        (inspect, "inspect", "Expand an element id, heading or context prefix from observe for details and nearby controls. "
         "Call when the digest lacks the detail needed to act."),
        (open_, "open", "Open an http(s) URL in the browser."),
        (find, "find", "Find the k (default 5) elements of the current page that best match a short description. "
                       "Use this instead of reading the page to locate links, buttons and fields."),
        (snapshot, "snapshot", "The k (default 20) elements most relevant to a goal on the current page."),
        (click, "click", "Click an element by id from the latest find/snapshot. Risky clicks (pay, delete, form submit) are refused."),
        (type_, "type", "Click a text field by id and type into it (replacing its content), optionally pressing Enter."),
        (go_back, "go_back", "Go back to the previous page."),
        (read_text, "read_text", "Visible text of the page, truncated to max_chars (default 4000). For reading content only."),
    ]:
        if not server.cdp_endpoint or name not in {"open", "click", "type", "go_back"}:
            mcp.add_tool(fn, name=name, description=desc)
    return mcp, pool


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compact UI-grounding MCP server (stdio).")
    p.add_argument("--ranker", default="exact", help="exact | bm25 | blend:<dir>:<alpha> | model:<dir> | router:<dir>:<tau> (default exact)")
    p.add_argument("--observe-ranker", default=None, help="observation ranker spec (default: --ranker)")
    p.add_argument("--cdp-endpoint", default="", help="attach perception-only to this Chromium CDP endpoint")
    p.add_argument("--cdp-page-url-substring", default="", help="select the last matching page in the first context")
    p.add_argument("--fp32", action="store_true", help="use model.onnx instead of model.int8.onnx")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--headed", action="store_true", help="show the browser window")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    grounder = UIGrounder(make_ranker(args.ranker, False if args.fp32 else None, args.threads))
    observe_grounder = (
        UIGrounder(make_ranker(args.observe_ranker, False if args.fp32 else None, args.threads))
        if args.observe_ranker else grounder
    )
    server = ToolServer(grounder, headless=not args.headed, observe_grounder=observe_grounder,
                        cdp_endpoint=args.cdp_endpoint, cdp_page_url_substring=args.cdp_page_url_substring)
    mcp, pool = build_mcp(server)
    try:
        mcp.run("stdio")
    finally:
        pool.submit(server.close).result()
        pool.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
