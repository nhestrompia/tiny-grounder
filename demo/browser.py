"""Task definitions, success checks and a Playwright session whose actions are addressed by element id.

Click safety is the one in `grounder.playwright_helper` (blocked words, form-submit detection, in-page probe);
this module only adds id addressing on top of it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

from grounder import playwright_helper as ph
from grounder.extract.web import extract_page_sync
from grounder.records import Element, Page

TASKS_FILE = Path(__file__).with_name("tasks.yaml")
CHECK_KEYS = ("url_contains", "url_regex", "text_contains")
VIEWPORT = {"width": 1280, "height": 800}
NAV_TIMEOUT_MS = 30_000
SETTLE_MS = 500


class ToolError(Exception):
    """An action the planner asked for failed or was refused; the message goes back as an error tool_result."""


@dataclass
class Task:
    id: str
    start_url: str
    goal: str
    check: dict[str, str]
    max_steps: int


def parse_tasks(raw: Any) -> list[Task]:
    """Validate the parsed YAML list; raises ValueError naming the offending task."""
    if not isinstance(raw, list) or not raw:
        raise ValueError("tasks file must be a non-empty list")
    tasks: list[Task] = []
    for i, item in enumerate(raw):
        name = item.get("id", f"#{i}") if isinstance(item, dict) else f"#{i}"
        if not isinstance(item, dict):
            raise ValueError(f"task {name}: not a mapping")
        for key in ("id", "start_url", "goal", "check", "max_steps"):
            if not item.get(key):
                raise ValueError(f"task {name}: missing {key}")
        if not re.match(r"https?://", item["start_url"]):
            raise ValueError(f"task {name}: start_url must be http(s)")
        check = item["check"]
        if not isinstance(check, dict) or not check or set(check) - set(CHECK_KEYS):
            raise ValueError(f"task {name}: check needs at least one of {', '.join(CHECK_KEYS)} and nothing else")
        try:
            re.compile(check.get("url_regex", ""))
        except re.error as exc:
            raise ValueError(f"task {name}: bad url_regex: {exc}") from exc
        if not isinstance(item["max_steps"], int) or item["max_steps"] < 1:
            raise ValueError(f"task {name}: max_steps must be a positive integer")
        tasks.append(Task(item["id"], item["start_url"], item["goal"], {k: str(v) for k, v in check.items()}, item["max_steps"]))
    if len({t.id for t in tasks}) != len(tasks):
        raise ValueError("duplicate task ids")
    return tasks


def load_tasks(path: str | Path = TASKS_FILE) -> list[Task]:
    return parse_tasks(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def check_success(check: dict[str, str], url: str, text: str) -> bool:
    """True when every condition holds on the final page (`text_contains` is case-insensitive)."""
    return all(
        {
            "url_contains": lambda v: v in url,
            "url_regex": lambda v: re.search(v, url) is not None,
            "text_contains": lambda v: v.casefold() in text.casefold(),
        }[key](value)
        for key, value in check.items()
    )


class BrowserSession:
    """A headless Chromium page. Element ids refer to the most recent `extract`."""

    def __init__(self, start_url: str, headless: bool = True) -> None:
        self._playwright = sync_playwright().start()
        try:
            self._browser = self._playwright.chromium.launch(headless=headless)
            self.page = self._browser.new_page(viewport=VIEWPORT)
            self.page.set_default_timeout(NAV_TIMEOUT_MS)
            self._guard(lambda: self.page.goto(start_url, wait_until="load"))
        except BaseException:
            self.close()
            raise
        self._settle()
        self.elements: list[Element] = []
        self._extracted_url = ""

    def __enter__(self) -> "BrowserSession":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if hasattr(self, "_browser"):
            self._browser.close()
        self._playwright.stop()

    @staticmethod
    def _guard(action):
        try:
            return action()
        except PlaywrightError as exc:
            raise ToolError(str(exc).splitlines()[0]) from exc

    def _settle(self) -> None:
        """Let a navigation triggered by the last action finish; a page that never settles is not an error."""
        try:
            self.page.wait_for_load_state("load", timeout=10_000)
        except PlaywrightError:
            pass
        self.page.wait_for_timeout(SETTLE_MS)

    @property
    def url(self) -> str:
        return self.page.url

    def text(self) -> str:
        return self._guard(lambda: self.page.inner_text("body"))

    def status(self) -> str:
        return f"url: {self.url}\ntitle: {self._guard(self.page.title)}"

    def extract(self) -> Page:
        page = self._guard(lambda: extract_page_sync(self.page))
        self.elements, self._extracted_url = page.elements, page.url
        return page

    def _element(self, element_id: int) -> Element:
        if self._extracted_url != self.url:
            raise ToolError("the page changed since the last extraction; get a fresh element list first")
        element = next((e for e in self.elements if e.id == element_id), None)
        if element is None:
            raise ToolError(f"no element with id {element_id} in the most recent element list")
        return element

    def _probe(self, element: Element, click_action: bool) -> dict[str, Any]:
        """Scroll the element into view and run the helper's in-page safety probe; raise ToolError if refused."""
        cx, cy = ph.centre(element.abs_box)
        probe = self._guard(lambda: self.page.evaluate(ph.PROBE_JS, [cx, cy, element.tag]))
        if reason := ph.verdict(probe, click_action):
            raise ToolError(f"refused: {reason}")
        return probe

    def click(self, element_id: int) -> Element:
        """Click element `element_id`, refusing risky ones (blocked words, form submits). Returns the element."""
        element = self._element(element_id)
        if word := ph.blocked_word(element):
            raise ToolError(f"refused: risky: text or context contains the blocked word {word!r}")
        if not element.enabled:
            raise ToolError("refused: the element is disabled")
        probe = self._probe(element, click_action=True)
        self._guard(lambda: self.page.mouse.click(probe["x"], probe["y"]))
        self._settle()
        return element

    def type_text(self, element_id: int, text: str, press_enter: bool) -> Element:
        """Focus a text field by clicking it, replace its content with `text`, optionally press Enter. Returns the
        element."""
        element = self._element(element_id)
        if element.role not in ph.TYPEABLE_ROLES:
            raise ToolError(f"element {element_id} is a {element.role}, not a text field")
        if not element.enabled:
            raise ToolError("refused: the element is disabled")
        probe = self._probe(element, click_action=False)

        def act() -> None:
            self.page.mouse.click(probe["x"], probe["y"])
            self.page.keyboard.press("ControlOrMeta+A")  # typing over the selection replaces any existing text
            self.page.keyboard.type(text)
            if press_enter:
                self.page.keyboard.press("Enter")

        self._guard(act)
        self._settle()
        return element

    def go_back(self) -> None:
        if self._guard(lambda: self.page.go_back(wait_until="load")) is None:
            raise ToolError("there is no previous page")
        self._settle()

    def succeeded(self, check: dict[str, str]) -> bool:
        try:
            text = self.text() if "text_contains" in check else ""
        except ToolError:
            return False
        return check_success(check, self.url, text)
