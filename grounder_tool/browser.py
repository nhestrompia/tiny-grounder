"""One Playwright page addressed by element id, with the click safety of grounder.playwright_helper
(blocked words, form submits, password fields, stale-page and overlay probe)."""

from __future__ import annotations

from typing import Any

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

from grounder import playwright_helper as ph
from grounder.records import Element, Page
from grounder_tool.extract import from_playwright

VIEWPORT = {"width": 1280, "height": 800}
NAV_TIMEOUT_MS = 30_000
SETTLE_MS = 500


class ToolError(Exception):
    """An action failed or was refused; the message goes back to the agent."""


class BrowserSession:
    def __init__(self, start_url: str, headless: bool = True) -> None:
        self._playwright = sync_playwright().start()
        try:
            self._browser = self._playwright.chromium.launch(headless=headless)
            self.page = self._browser.new_page(viewport=VIEWPORT)
            self.page.set_default_timeout(NAV_TIMEOUT_MS)
            self.goto(start_url)
        except BaseException:
            self.close()
            raise
        self.elements: list[Element] = []
        self._extracted_url = ""

    @classmethod
    def attach(cls, endpoint: str, url_substring: str = "") -> BrowserSession:
        """Read the last matching page in the first CDP context without owning its browser."""
        session = cls.__new__(cls)
        session._playwright = sync_playwright().start()
        session._attached = True
        try:
            session._browser = session._playwright.chromium.connect_over_cdp(endpoint)
            contexts = session._browser.contexts
            pages = contexts[0].pages if contexts else []
            matches = [p for p in pages if url_substring in p.url]
            if not matches:
                raise ToolError("no matching page in the first CDP context")
            session.page = matches[-1]
            session.elements = []
            session._extracted_url = ""
            return session
        except BaseException:
            session._playwright.stop()
            raise

    def close(self) -> None:
        if hasattr(self, "_browser") and not getattr(self, "_attached", False):
            self._browser.close()
        self._playwright.stop()

    @staticmethod
    def _guard(action):
        try:
            return action()
        except PlaywrightError as exc:
            raise ToolError(str(exc).splitlines()[0]) from exc

    def _require_actions(self) -> None:
        if getattr(self, "_attached", False):
            raise ToolError("CDP attach is perception-only")

    def _settle(self) -> None:
        try:
            self.page.wait_for_load_state("load", timeout=10_000)
        except PlaywrightError:
            pass
        self.page.wait_for_timeout(SETTLE_MS)

    def goto(self, url: str) -> None:
        self._require_actions()
        self._guard(lambda: self.page.goto(url, wait_until="load"))
        self._settle()

    @property
    def url(self) -> str:
        return self.page.url

    def text(self) -> str:
        return self._guard(lambda: self.page.inner_text("body"))

    def status(self) -> str:
        return f"url: {self.url}\ntitle: {self._guard(self.page.title)}"

    def extract(self) -> Page:
        page = self._guard(lambda: from_playwright(self.page))
        self.landmarks, self.heading_positions = self._guard(lambda: ph.extract_landmarks(self.page))
        self.elements, self._extracted_url = page.elements, page.url
        return page

    def _element(self, element_id: int) -> Element:
        if self._extracted_url != self.url:
            if self._extracted_url:
                raise ToolError("the page changed since the last find/snapshot; call find or snapshot again")
            raise ToolError("no element list yet; call find or snapshot first")
        element = next((e for e in self.elements if e.id == element_id), None)
        if element is None:
            raise ToolError(f"no element with id {element_id} in the most recent find/snapshot")
        return element

    def _probe(self, element: Element, click_action: bool) -> dict[str, Any]:
        cx, cy = ph.centre(element.abs_box)
        probe = self._guard(lambda: self.page.evaluate(ph.PROBE_JS, [cx, cy, element.tag]))
        if reason := ph.verdict(probe, click_action):
            raise ToolError(f"refused: {reason}")
        return probe

    def click(self, element_id: int) -> Element:
        self._require_actions()
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
        self._require_actions()
        element = self._element(element_id)
        if element.role not in ph.TYPEABLE_ROLES:
            raise ToolError(f"element {element_id} is a {element.role}, not a text field")
        if not element.enabled:
            raise ToolError("refused: the element is disabled")
        probe = self._probe(element, click_action=False)

        def act() -> None:
            self.page.mouse.click(probe["x"], probe["y"])
            self.page.keyboard.press("ControlOrMeta+A")
            self.page.keyboard.type(text)
            if press_enter:
                self.page.keyboard.press("Enter")

        self._guard(act)
        self._settle()
        return element

    def go_back(self) -> None:
        self._require_actions()
        if self._guard(lambda: self.page.go_back(wait_until="load")) is None:
            raise ToolError("there is no previous page")
        self._settle()
