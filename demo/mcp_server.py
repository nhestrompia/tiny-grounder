"""Stdio MCP server that owns ONE Playwright browser session for ONE benchmark task.

    uv run python -m demo.mcp_server --config find --goal "..." --start-url https://... --log run.jsonl \
        --check '{"url_contains": "/wiki/Alan_Turing"}' --ranker grounder --model-dir dist/s-s0

Configurations (which tools the model gets; every one also has `click`, `type`, `go_back`, `done`, or the
coordinate equivalents for `vision`):

    full        snapshot() -> every extracted element, `id|role|text|context` lines (same as demo/agent.py)
    a11y        snapshot() -> Playwright's AI-mode ARIA snapshot (`page.aria_snapshot(mode="ai")`), the text the
                standard Playwright MCP server hands its agents, with `[ref=eN]` markers. click(ref) / type(ref, ...)
                resolve a ref through Playwright's own `aria-ref=` locator (again exactly what Playwright MCP does).
                Like Playwright MCP, every action returns a fresh snapshot (`--no-observe` turns that off).
    vision      screenshot() -> PNG of the 1280x800 viewport; click_xy, type_text, scroll, go_back, done. Computer-use
                style. Every action returns the new screenshot (`--no-observe` turns that off).
    top20       snapshot() -> the grounder's 20 best elements for the goal (goal mode, last actions as history)
    find        find(query) -> the grounder's 5 best elements with scores (do mode); no snapshot
    find+top20  both

`--ranker bm25` swaps the grounder for the BM25 baseline in the top20 / find configs.

The log (JSONL, `--log`) is the source of truth: one `call` event per tool call (chars returned to the model; for
images bytes and pixel size too) and one `summary` event, written once at `done`, on stdin EOF, or on SIGTERM / SIGINT:
final URL, the task check evaluated on the final page, call and step counts, context volume.

Safety refusals are those of demo.browser / grounder.playwright_helper (blocked words, form submits, password
fields). In `a11y` and `vision` the blocked words are matched against the clicked element's own label only (those
modes have no element-context records).

All browser work runs on one dedicated thread (Playwright's sync API is thread-bound; MCP calls arrive on an event loop).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from demo.agent import BM25Ranker, Ranker, ToolBox
from demo.browser import CHECK_KEYS, BrowserSession, ToolError
from grounder import playwright_helper as ph

SERVER_NAME = "ui"
BENCH_CONFIGS: tuple[str, ...] = ("full", "a11y", "vision", "top20", "find", "find+top20")
ACTION_TOOLS = ("click", "type", "go_back", "click_xy", "type_text", "scroll")
DEFAULT_MODEL_DIR = "dist/s-s0"
DEFAULT_MAX_CALLS = 60
RESULT_LOG_CHARS = 4000
SCROLL_SETTLE_MS = 300

_REF_RE = re.compile(r"^[A-Za-z0-9_]{1,16}$")

CONFIG_HINTS: dict[str, str] = {
    "full": "Call snapshot() to list every element on the current page as `id|role|text|context` lines. "
    "Element ids refer to the most recent snapshot.",
    "a11y": "Call snapshot() to get the page's accessibility tree; interactive nodes carry a [ref=eN] marker. "
    "click(ref) and type(ref, text, press_enter) take such a ref (for example 'e12'). Refs refer to the most recent snapshot.",
    "vision": "Call screenshot() to see the page (1280x800 viewport). Act with click_xy(x, y) in pixel coordinates of that "
    "screenshot, type_text(text, press_enter) into the focused field, scroll(dy) and go_back().",
    "top20": "Call snapshot() to list the 20 elements most relevant to your goal on the current page, most relevant "
    "first, as `id|role|text|context` lines. If the element you need is not listed, navigate elsewhere and snapshot again. "
    "Element ids refer to the most recent snapshot.",
    "find": "You cannot see the whole page. Call find(query) with a short description of the element you want (for "
    "example 'search box' or 'link to the download page'); it returns the 5 best matches as `id|role|text|context|score` "
    "lines, best first, with a score between 0 and 1. Element ids refer to the most recent find.",
    "find+top20": "Call snapshot() to list the 20 elements most relevant to your goal, or find(query) with a short "
    "description of one element you want (it returns the 5 best matches with a score between 0 and 1). Both return "
    "`id|role|text|context` lines; element ids refer to the most recent snapshot or find.",
}

# In-page helpers for the a11y / vision modes: label, submit and password facts about an element.
_FACTS_JS = """(el) => {
  const up = (n) => n.parentElement || (n.getRootNode() && n.getRootNode().host) || null;
  const closest = (n, sel) => { for (; n; n = up(n)) if (n.matches && n.matches(sel)) return n; return null; };
  const pick = closest(el, "a,button,input,textarea,select,summary,label,[role=button],[role=link]") || el;
  const control = closest(el, "button,input");
  const submit = !!control && (control.matches("input[type=submit],input[type=image]") ||
    (control.tagName === "BUTTON" && !!control.form && control.type === "submit"));
  const field = closest(el, "input,textarea");
  const editable = !!closest(el, "input,textarea,[contenteditable=''],[contenteditable=true]");
  const label = pick.getAttribute("aria-label") || pick.innerText || pick.value || pick.title || pick.getAttribute("alt") || "";
  return { tag: pick.tagName.toLowerCase(), label: label.trim().slice(0, 200), submit, editable,
           password: !!field && field.type === "password" };
}"""
_POINT_FACTS_JS = "([x, y]) => { const el = document.elementFromPoint(x, y); return el ? (" + _FACTS_JS + ")(el) : null; }"
_FOCUS_FACTS_JS = "() => { const el = document.activeElement; return el && el !== document.body ? (" + _FACTS_JS + ")(el) : null; }"


def png_size(data: bytes) -> tuple[int, int]:
    """(width, height) from a PNG header."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    return struct.unpack(">II", data[16:24])


def tools_for(config: str) -> tuple[str, ...]:
    """Names of the MCP tools a configuration exposes."""
    if config not in BENCH_CONFIGS:
        raise ValueError(f"unknown config {config!r}; choose from {', '.join(BENCH_CONFIGS)}")
    return {
        "full": ("snapshot", "click", "type", "go_back", "done"),
        "a11y": ("snapshot", "click", "type", "go_back", "done"),
        "vision": ("screenshot", "click_xy", "type_text", "scroll", "go_back", "done"),
        "top20": ("snapshot", "click", "type", "go_back", "done"),
        "find": ("find", "click", "type", "go_back", "done"),
        "find+top20": ("snapshot", "find", "click", "type", "go_back", "done"),
    }[config]


class BenchToolBox(ToolBox):
    """demo.agent.ToolBox with the `find+top20` combination (snapshot = top20, plus find)."""

    def __init__(self, config: str, session: BrowserSession, ranker: Ranker, goal: str) -> None:
        super().__init__("top20" if config == "find+top20" else config, session, ranker, goal)
        self.allowed = set(tools_for(config))

    def call(self, name: str, args: dict[str, Any]) -> str:
        if name not in self.allowed:
            raise ToolError(f"unknown tool {name!r}")
        try:
            return getattr(self, f"_{name}")(**args)
        except TypeError as exc:
            raise ToolError(f"invalid arguments: {exc}") from exc


class BenchServer:
    """Tool handlers + logging for one task. Not thread-safe: call `invoke` from the thread that built it.

    `invoke(name, **args)` returns the text the model sees, or `(text, png_bytes)` for screenshots, and raises
    ToolError for a refused / failed call. Every call (also failed ones) is logged.
    """

    def __init__(
        self, config: str, session: BrowserSession, ranker: Ranker | None, goal: str, check: dict[str, str],
        log_path: str | Path, task_id: str = "", observe: bool = True, max_calls: int = DEFAULT_MAX_CALLS,
    ) -> None:
        tools_for(config)  # validates the name
        self.config, self.session, self.goal, self.check, self.task_id = config, session, goal, check, task_id
        self.observe, self.max_calls = observe, max_calls
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.log_path.open("a", encoding="utf-8")
        self.started = time.perf_counter()
        self.calls = 0
        self.steps = 0
        self.errors = 0
        self.chars = 0
        self.image_bytes = 0
        self.images = 0
        self.done_called = False
        self.answer = ""
        self.finished = False
        self.box: BenchToolBox | None = None
        if config not in ("a11y", "vision"):
            if ranker is None:
                raise ValueError(f"config {config} needs a ranker")
            self.box = BenchToolBox(config, session, ranker, goal)

    # ---- logging ----------------------------------------------------------------------------------------

    def _log(self, event: dict[str, Any]) -> None:
        self._fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._fh.flush()

    def summary(self, reason: str) -> dict[str, Any]:
        try:
            url = self.session.url
        except Exception:  # browser already gone
            url = ""
        try:
            success = self.session.succeeded(self.check)
        except Exception:
            success = False
        return {
            "type": "summary", "reason": reason, "task_id": self.task_id, "config": self.config, "goal": self.goal,
            "success": success, "done_called": self.done_called, "answer": self.answer, "final_url": url,
            "calls": self.calls, "steps": self.steps, "errors": self.errors, "tool_chars": self.chars,
            "image_bytes": self.image_bytes, "images": self.images,
            "seconds": round(time.perf_counter() - self.started, 2),
        }

    def finish(self, reason: str) -> None:
        """Write the summary once (first of done / exit / signal wins) and close the log."""
        if self.finished:
            return
        self.finished = True
        self._log(self.summary(reason))
        self._fh.close()

    # ---- dispatch ---------------------------------------------------------------------------------------

    def invoke(self, name: str, **args: Any) -> str | tuple[str, bytes]:
        if self.finished:
            raise ToolError("the task is over")
        if name not in tools_for(self.config):
            raise ToolError(f"unknown tool {name!r}")
        t0 = time.perf_counter()
        event: dict[str, Any] = {"type": "call", "n": self.calls + 1, "name": name, "args": args}
        try:
            if self.calls >= self.max_calls and name != "done":
                raise ToolError(f"call budget of {self.max_calls} used up; call done(answer) now")
            self.calls += 1
            self.steps += name in ACTION_TOOLS
            out = getattr(self, f"_t_{name}")(**args)
        except ToolError as exc:
            self.errors += 1
            text = str(exc)
            self.chars += len(text)
            self._log({**event, "n": self.calls, "is_error": True, "chars": len(text), "result": text[:RESULT_LOG_CHARS],
                       "ms": round((time.perf_counter() - t0) * 1000)})
            raise
        except TypeError as exc:
            self.errors += 1
            raise ToolError(f"invalid arguments: {exc}") from exc
        text, png = (out, None) if isinstance(out, str) else out
        self.chars += len(text)
        event.update(is_error=False, chars=len(text), result=text[:RESULT_LOG_CHARS], ms=round((time.perf_counter() - t0) * 1000))
        if png is not None:
            width, height = png_size(png)
            self.image_bytes += len(png)
            self.images += 1
            event.update(image_bytes=len(png), width=width, height=height)
        self._log(event)
        if name == "done":
            self.finish("done")
        return out

    # ---- tools ------------------------------------------------------------------------------------------

    def _t_done(self, answer: str) -> str:
        self.done_called, self.answer = True, str(answer)
        return "ok"

    def _grounder(self, name: str, **args: Any) -> str:
        assert self.box is not None
        return self.box.call(name, args)

    def _t_snapshot(self) -> str:
        if self.config == "a11y":
            return self._aria()
        return self._grounder("snapshot")

    def _t_find(self, query: str) -> str:
        return self._grounder("find", query=query)

    def _t_go_back(self) -> str:
        if self.box is not None:
            return self._grounder("go_back")
        self.session.go_back()
        return self._after("went back")

    def _t_click(self, id: int | str) -> str:
        if self.box is not None:
            return self._grounder("click", id=id)
        return self._after(self._click_ref(str(id)))

    def _t_type(self, id: int | str, text: str, press_enter: bool = False) -> str:
        if self.box is not None:
            return self._grounder("type", id=id, text=text, press_enter=press_enter)
        return self._after(self._type_ref(str(id), text, press_enter))

    # -- a11y --

    def _aria(self) -> str:
        page = self.session.page
        snap = self.session._guard(lambda: page.aria_snapshot(mode="ai"))
        return f"{self.session.status()}\n{snap}"

    def _ref(self, ref: str):
        if not _REF_RE.match(ref):
            raise ToolError(f"{ref!r} is not a ref from the latest snapshot (like 'e12')")
        loc = self.session.page.locator(f"aria-ref={ref}")
        if self.session._guard(loc.count) == 0:
            raise ToolError(f"no element with ref {ref} in the latest snapshot; take a new snapshot")
        return loc.first

    def _click_ref(self, ref: str) -> str:
        loc = self._ref(ref)
        facts = self.session._guard(lambda: loc.evaluate(_FACTS_JS))
        if word := ph.blocked_text(facts["label"]):
            raise ToolError(f"refused: risky: the element mentions the blocked word {word!r}")
        if facts["submit"]:
            raise ToolError("refused: risky: the element submits a form")
        self.session._guard(lambda: loc.click(timeout=10_000))
        self.session._settle()
        return f"clicked {facts['tag']} {facts['label']!r}"

    def _type_ref(self, ref: str, text: str, press_enter: bool) -> str:
        loc = self._ref(ref)
        facts = self.session._guard(lambda: loc.evaluate(_FACTS_JS))
        if not facts["editable"]:
            raise ToolError(f"ref {ref} is a {facts['tag']}, not a text field")
        if facts["password"]:
            raise ToolError("refused: the element is a password field")
        page = self.session.page

        def act() -> None:
            loc.click(timeout=10_000)
            page.keyboard.press("ControlOrMeta+A")
            page.keyboard.type(text)
            if press_enter:
                page.keyboard.press("Enter")

        self.session._guard(act)
        self.session._settle()
        enter = " and pressed Enter" if press_enter else ""
        return f"typed into {facts['tag']} {facts['label']!r}{enter}"

    # -- vision --

    def _shot(self) -> bytes:
        return self.session._guard(lambda: self.session.page.screenshot(type="png"))

    def _t_screenshot(self) -> tuple[str, bytes]:
        return self.session.status(), self._shot()

    def _after(self, message: str) -> str | tuple[str, bytes]:
        """Result of an action in a11y / vision: the message, plus the new observation unless `--no-observe`."""
        text = f"{message}\n{self.session.status()}"
        if not self.observe:
            return text
        if self.config == "a11y":
            return f"{text}\n{self._aria_body()}"
        return text, self._shot()

    def _aria_body(self) -> str:
        return self.session._guard(lambda: self.session.page.aria_snapshot(mode="ai"))

    def _t_click_xy(self, x: float, y: float) -> str | tuple[str, bytes]:
        page = self.session.page
        vw, vh = page.viewport_size["width"], page.viewport_size["height"]
        if not (0 <= x < vw and 0 <= y < vh):
            raise ToolError(f"({x}, {y}) is outside the {vw}x{vh} screenshot")
        facts = self.session._guard(lambda: page.evaluate(_POINT_FACTS_JS, [x, y]))
        if facts is None:
            raise ToolError("nothing at that point")
        if word := ph.blocked_text(facts["label"]):
            raise ToolError(f"refused: risky: the element at that point mentions the blocked word {word!r}")
        if facts["submit"]:
            raise ToolError("refused: risky: the element at that point submits a form")
        self.session._guard(lambda: page.mouse.click(x, y))
        self.session._settle()
        return self._after(f"clicked at ({x:g}, {y:g})")

    def _t_type_text(self, text: str, press_enter: bool = False) -> str | tuple[str, bytes]:
        page = self.session.page
        facts = self.session._guard(lambda: page.evaluate(_FOCUS_FACTS_JS))
        if facts is None or not facts["editable"]:
            raise ToolError("no text field is focused; click on one first")
        if facts["password"]:
            raise ToolError("refused: the element is a password field")

        def act() -> None:
            page.keyboard.press("ControlOrMeta+A")
            page.keyboard.type(text)
            if press_enter:
                page.keyboard.press("Enter")

        self.session._guard(act)
        self.session._settle()
        enter = " and pressed Enter" if press_enter else ""
        return self._after(f"typed {text!r}{enter}")

    def _t_scroll(self, dy: float) -> str | tuple[str, bytes]:
        page = self.session.page
        self.session._guard(lambda: page.mouse.wheel(0, dy))
        page.wait_for_timeout(SCROLL_SETTLE_MS)
        return self._after(f"scrolled by {dy:g}px")


# ---- MCP wiring --------------------------------------------------------------------------------------------


def make_ranker(name: str, model_dir: str) -> Ranker:
    if name == "bm25":
        return BM25Ranker()
    from grounder.runtime import Grounder

    return Grounder(model_dir)


def build_mcp(config: str, call: Any) -> Any:
    """A FastMCP app exposing `tools_for(config)`; `call(name, **args)` runs a handler (on the browser thread)."""
    from mcp.server.fastmcp import FastMCP, Image

    mcp = FastMCP(SERVER_NAME, instructions=CONFIG_HINTS[config])

    def out(result: str | tuple[str, bytes]) -> Any:
        return result if isinstance(result, str) else [result[0], Image(data=result[1], format="png")]

    ident = "Element id from the most recent snapshot / find." if config != "a11y" else "Ref from the latest snapshot, e.g. 'e12'."

    if "snapshot" in tools_for(config):
        desc = {
            "full": "List every element of the current page (id|role|text|context). Replaces earlier ids.",
            "a11y": "Accessibility snapshot of the current page, with [ref=eN] markers on interactive nodes.",
            "top20": "List the 20 elements most relevant to your goal on the current page. Replaces earlier ids.",
            "find+top20": "List the 20 elements most relevant to your goal on the current page. Replaces earlier ids.",
        }[config]
        mcp.add_tool(lambda: out(call("snapshot")), name="snapshot", description=desc)
    if "find" in tools_for(config):
        def find(query: str) -> Any:
            return out(call("find", query=query))
        mcp.add_tool(find, name="find", description="Find the elements of the current page that best match a "
                     "description (top 5 with scores). Replaces earlier ids.")
    if config == "vision":
        mcp.add_tool(lambda: out(call("screenshot")), name="screenshot", description="Screenshot of the browser viewport (1280x800).")

        def click_xy(x: float, y: float) -> Any:
            return out(call("click_xy", x=x, y=y))

        def type_text(text: str, press_enter: bool = False) -> Any:
            return out(call("type_text", text=text, press_enter=press_enter))

        def scroll(dy: float) -> Any:
            return out(call("scroll", dy=dy))

        mcp.add_tool(click_xy, name="click_xy", description="Click at pixel coordinates (x, y) of the screenshot.")
        mcp.add_tool(type_text, name="type_text", description="Type into the focused text field (replacing its "
                     "content), optionally pressing Enter (for example to search).")
        mcp.add_tool(scroll, name="scroll", description="Scroll the page vertically by dy pixels (positive = down).")
    else:
        if config == "a11y":
            def click(ref: str) -> Any:
                return out(call("click", id=ref))

            def type_(ref: str, text: str, press_enter: bool = False) -> Any:
                return out(call("type", id=ref, text=text, press_enter=press_enter))
        else:
            def click(id: int) -> Any:  # noqa: A002
                return out(call("click", id=id))

            def type_(id: int, text: str, press_enter: bool = False) -> Any:  # noqa: A002
                return out(call("type", id=id, text=text, press_enter=press_enter))

        mcp.add_tool(click, name="click", description=f"Click an element. {ident}")
        mcp.add_tool(type_, name="type", description=f"Click a text field and type into it, optionally pressing Enter. {ident}")
    mcp.add_tool(lambda: out(call("go_back")), name="go_back", description="Go back to the previous page.")

    def done(answer: str) -> Any:
        return out(call("done", answer=answer))

    mcp.add_tool(done, name="done", description="Finish the task with a short answer (or the reason it cannot be done).")
    return mcp


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, choices=BENCH_CONFIGS)
    p.add_argument("--goal", required=True)
    p.add_argument("--start-url", required=True)
    p.add_argument("--log", required=True, help="JSONL log path (appended)")
    p.add_argument("--check", default="{}", help='task check as JSON, e.g. \'{"url_contains": "/wiki/X"}\'')
    p.add_argument("--task-id", default="")
    p.add_argument("--ranker", choices=["grounder", "bm25"], default="grounder")
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--max-calls", type=int, default=DEFAULT_MAX_CALLS)
    p.add_argument("--no-observe", action="store_true", help="a11y / vision: actions do not return a new observation")
    args = p.parse_args(argv)
    args.check = json.loads(args.check)
    if bad := set(args.check) - set(CHECK_KEYS):
        p.error(f"unknown check keys: {sorted(bad)}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="browser")

    def build() -> BenchServer:
        ranker = None if args.config in ("a11y", "vision") else make_ranker(args.ranker, args.model_dir)
        session = BrowserSession(args.start_url)
        return BenchServer(args.config, session, ranker, args.goal, args.check, args.log, args.task_id,
                           observe=not args.no_observe, max_calls=args.max_calls)

    bench = pool.submit(build).result()

    def call(name: str, **kwargs: Any) -> Any:
        return pool.submit(bench.invoke, name, **kwargs).result()

    def shutdown(reason: str) -> None:
        try:
            pool.submit(lambda: (bench.finish(reason), bench.session.close())).result(timeout=15)
        except Exception:
            pass

    def on_signal(signum: int, _frame: object) -> None:
        shutdown("signal")
        os._exit(0)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    try:
        build_mcp(args.config, call).run()
    finally:
        shutdown("exit")
        pool.shutdown(wait=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
