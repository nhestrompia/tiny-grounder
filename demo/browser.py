"""Task definitions, success checks and a Playwright session whose actions are addressed by element id.

Click safety is the one in `grounder.playwright_helper` (blocked words, form-submit detection, in-page probe);
this module only adds id addressing on top of it.
"""

from __future__ import annotations

import re
import time
from contextlib import contextmanager
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
CHECK_KEYS = ("url_contains", "url_regex", "text_contains", "text_not_contains", "text_regex")
VIEWPORT = {"width": 1280, "height": 800}
# Recording skips video / audio (huge, never read by the agents) and well-known trackers. Playwright's recording filter
# KEEPS what matches, hence the negative lookahead.
RECORD_URL_FILTER = re.compile(
    r"^(?!.*(?:\.(?:mp4|webm|m4s|mov|mp3|ogg|m3u8)(?:\?|$)|google-analytics|googletagmanager|doubleclick|facebook\.net"
    r"|hotjar|sentry\.io|clarity\.ms|segment\.(?:com|io))).*$", re.I)
NAV_TIMEOUT_MS = 30_000
SETTLE_MS = 500


# In-page helpers for click / type safety on pages whose layout moves between extraction and action (sticky, inner-scrolling
# sidebars such as a docs table of contents; lazy content). `_HIT_LABEL_JS` names what is under a viewport point;
# `_RELOCATE_JS` finds the extracted element again by label (and href), scrolls it into view (inner scrollers included) and
# returns its fresh viewport centre.
_CONTROLS = ("a,button,input,textarea,select,summary,label,[role=button],[role=link],[role=tab],[role=menuitem],"
             "[role=option],[role=checkbox],[role=radio],[role=switch],[role=combobox],[role=searchbox],[role=textbox]")
_LABEL_FN = """const norm = (s) => (s || "").replace(/\\s+/g, " ").trim().normalize("NFKC");
  const label = (el) => norm(el.getAttribute("aria-label") || el.innerText || el.value || el.title ||
                             el.getAttribute("placeholder") || el.getAttribute("alt") || "").slice(0, 80);"""
_HIT_LABEL_JS = """([x, y, sel]) => {
  %s
  let el = document.elementFromPoint(x, y);
  if (!el) return null;
  const pick = el.closest(sel) || el;
  return { label: label(pick), href: pick.getAttribute("href") || "" };
}""" % _LABEL_FN
_RELOCATE_JS = """([text, href, sel]) => {
  %s
  const want = norm(text).slice(0, 80);
  const all = [...document.querySelectorAll(sel)].filter((el) => label(el) === want);
  const same = href ? all.filter((el) => (el.href || "").endsWith(href)) : [];
  for (const el of [...same, ...all.filter((e) => !same.includes(e))]) {
    if (!el.getClientRects().length) continue;  // hidden copies (mobile drawers...) cannot be clicked
    el.scrollIntoView({ block: "center", inline: "center", behavior: "instant" });
    const r = el.getBoundingClientRect();
    const x = r.left + r.width / 2, y = r.top + r.height / 2;
    const hit = document.elementFromPoint(x, y);
    if (hit && (el === hit || el.contains(hit) || hit.contains(el))) return { ok: true, x, y };
  }
  return { ok: false };
}""" % _LABEL_FN


def _same_label(a: str, b: str) -> bool:
    a, b = " ".join(a.split()).casefold(), " ".join(b.split()).casefold()
    return not a or not b or a[:30] in b or b[:30] in a


class ToolError(Exception):
    """An action the planner asked for failed or was refused; the message goes back as an error tool_result."""


class StaleElementsError(ToolError):
    """No element list exists yet, or the last one belongs to another page; the caller needs a fresh element list."""


@dataclass
class Task:
    id: str
    start_url: str
    goal: str
    check: dict[str, str]
    max_steps: int
    answer_regex: str = ""  # when set, done(answer) must match it (case-insensitive) for the run to count
    tags: tuple[str, ...] = ()  # benchmark v2 categories: icon, vague, long, multistep, normal
    har: str = ""  # frozen snapshot (Playwright HAR), path relative to the repo root; empty = live web
    site: str = ""  # cluster for the paired analysis (default: registrable domain of start_url)
    allow_actions: bool = False  # opt-in for demo/throwaway sites: relax click-safety and the agent's read-only instruction


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
        if any(isinstance(v, (dict, list, set, tuple)) for v in check.values()):
            raise ValueError(f"task {name}: check values must be scalars")
        if any(v is None or str(v) == "" for v in check.values()):
            raise ValueError(f"task {name}: check values must be non-empty")
        check = {k: str(v) for k, v in check.items()}
        try:
            re.compile(check.get("url_regex", ""))
            re.compile(check.get("text_regex", ""))
        except re.error as exc:
            raise ValueError(f"task {name}: bad check regex: {exc}") from exc
        if not isinstance(item["max_steps"], int) or item["max_steps"] < 1:
            raise ValueError(f"task {name}: max_steps must be a positive integer")
        answer_regex = str(item.get("answer_regex") or "")
        try:
            re.compile(answer_regex)
        except re.error as exc:
            raise ValueError(f"task {name}: bad answer_regex: {exc}") from exc
        tags = item.get("tags") or []
        if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
            raise ValueError(f"task {name}: tags must be a list of strings")
        allow_actions = item.get("allow_actions", False)
        if not isinstance(allow_actions, bool):
            raise ValueError(f"task {name}: allow_actions must be a boolean")
        tasks.append(Task(item["id"], item["start_url"], item["goal"], {k: str(v) for k, v in check.items()}, item["max_steps"],
                          answer_regex, tuple(tags), str(item.get("har") or ""), str(item.get("site") or ""),
                          allow_actions))
    if len({t.id for t in tasks}) != len(tasks):
        raise ValueError("duplicate task ids")
    return tasks


def load_tasks(path: str | Path = TASKS_FILE) -> list[Task]:
    return parse_tasks(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def answer_matches(answer_regex: str, answer: str) -> bool:
    """True when the task has no answer requirement or done(answer) matches it (case-insensitive)."""
    return not answer_regex or re.search(answer_regex, answer or "", re.IGNORECASE) is not None


def check_success(check: dict[str, str], url: str, text: str) -> bool:
    """True when every condition holds on the final page (`text_contains` is case-insensitive)."""
    return all(
        {
            "url_contains": lambda v: v in url,
            "url_regex": lambda v: re.search(v, url) is not None,
            "text_contains": lambda v: v.casefold() in text.casefold(),
            "text_not_contains": lambda v: v.casefold() not in text.casefold(),
            "text_regex": lambda v: re.search(v, text, re.IGNORECASE | re.DOTALL) is not None,
        }[key](value)
        for key, value in check.items()
    )


STUB_HTML = ("<!doctype html><title>page not archived</title><h1>Page not archived</h1>"
             "<p>This benchmark replays a frozen snapshot; {url} was not recorded.</p>")
# Replay seatbelt: a request that somehow bypasses the HAR routes dies at this closed port instead of the live web.
DEAD_PROXY = "http://127.0.0.1:9"


class Replay:
    """Counters of a HAR replay: every request the page made and the ones the archive did not contain."""

    def __init__(self) -> None:
        self.requests = 0
        self.misses: list[str] = []
        self.failed: list[str] = []  # requests that reached the network layer and died at the dead proxy (should be none)
        self.live: list[str] = []  # responses that came from a real server (server_addr set); fulfilled ones have none

    def on_response(self, response: Any) -> None:
        try:
            if response.server_addr():
                self.live.append(response.url)
        except PlaywrightError:
            pass


class BrowserSession:
    """A headless Chromium page. Element ids refer to the most recent `extract`.

    `har` replays a frozen snapshot (Playwright HAR / zip) with no live network: whatever the archive lacks is answered
    with a "page not archived" stub for a top-level navigation and aborted for everything else (`replay_miss="abort"`
    aborts navigations too). `record_har` records the live session into a HAR zip instead (used by tools.bench_v2).
    """

    def __init__(self, start_url: str, headless: bool = True, har: str | Path | None = None, replay_miss: str = "stub",
                 record_har: str | Path | None = None, cdp: bool = False, user_agent: str | None = None) -> None:
        self.replay: Replay | None = Replay() if har else None
        self._playwright = sync_playwright().start()
        try:
            launch: dict[str, Any] = {"headless": headless}
            ctx: dict[str, Any] = {"viewport": VIEWPORT}
            if user_agent:
                ctx["user_agent"] = user_agent
            if har:
                launch["proxy"] = {"server": DEAD_PROXY}
                ctx["service_workers"] = "block"
            if record_har:
                ctx.update(record_har_path=str(record_har), record_har_content="attach", record_har_mode="full",
                           service_workers="block", record_har_url_filter=RECORD_URL_FILTER)
            if cdp:
                import socket
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", 0))
                    port = sock.getsockname()[1]
                launch["args"] = [f"--remote-debugging-port={port}", "--remote-debugging-address=127.0.0.1"]
                self.cdp_endpoint = f"http://127.0.0.1:{port}"
            self._browser = self._playwright.chromium.launch(**launch)
            self.context = self._browser.new_context(**ctx)
            if har:
                self._install_replay(Path(har), replay_miss)
            if cdp:
                # External new tabs can miss Playwright's context routes. Keep all navigation on the routed page.
                self.context.add_init_script("""(() => {
                  window.open = (url) => { if (url) window.location.assign(url); return window; };
                  document.addEventListener('click', event => {
                    const a = event.target.closest && event.target.closest('a[target]');
                    if (a) a.removeAttribute('target');
                  }, true);
                })();""")
            self.page = self.context.new_page()
            self.page.set_default_timeout(NAV_TIMEOUT_MS)
            self._guard(lambda: self.page.goto(start_url, wait_until="load"))
        except BaseException:
            self.close()
            raise
        self._settle()
        self.elements: list[Element] = []
        self._extracted_url = ""

    def _install_replay(self, har: Path, replay_miss: str) -> None:
        if not har.exists():
            raise FileNotFoundError(f"HAR snapshot not found: {har}")
        replay = self.replay
        assert replay is not None

        def stub(route, request) -> None:  # registered first, so it runs last: only what the archive lacks gets here
            replay.misses.append(request.url)
            if replay_miss == "stub" and request.is_navigation_request() and request.frame.parent_frame is None:
                route.fulfill(status=404, content_type="text/html; charset=utf-8", body=STUB_HTML.format(url=request.url))
            else:
                route.abort()

        self.context.route("**/*", stub)
        self.context.route_from_har(har, not_found="fallback")
        self.context.on("request", lambda r: setattr(replay, "requests", replay.requests + 1))
        self.context.on("response", replay.on_response)
        self.context.on("requestfailed", lambda r: replay.failed.append(r.url) if "PROXY" in (r.failure or "").upper() else None)

    def __enter__(self) -> "BrowserSession":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if hasattr(self, "context"):
            try:
                self.context.close()  # writes a recorded HAR
            except PlaywrightError:
                pass
        if hasattr(self, "_browser"):
            self._browser.close()
        self._playwright.stop()

    @staticmethod
    def _guard(action):
        try:
            return action()
        except PlaywrightError as exc:
            raise ToolError(str(exc).splitlines()[0]) from exc

    @contextmanager
    def overhead(self, field):
        start = time.perf_counter()
        try:
            yield
        finally:
            setattr(self, field, getattr(self, field, 0.0) + (time.perf_counter() - start) * 1000)

    def _settle(self) -> None:
        """Let a navigation triggered by the last action finish; a page that never settles is not an error."""
        with self.overhead("settle_ms"):
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
        self.landmarks, self.heading_positions = self._guard(lambda: ph.extract_landmarks(self.page))
        self.elements, self._extracted_url = page.elements, page.url
        return page

    def _element(self, element_id: int) -> Element:
        if self._extracted_url != self.url:
            if self._extracted_url:
                raise StaleElementsError("the page changed since the last extraction; get a fresh element list first")
            raise StaleElementsError("no element list yet; get an element list first")
        element = next((e for e in self.elements if e.id == element_id), None)
        if element is None:
            raise ToolError(f"no element with id {element_id} in the most recent element list")
        return element

    def _probe_point(self, cx: float, cy: float, tag: str, click_action: bool) -> dict[str, Any]:
        probe = self._guard(lambda: self.page.evaluate(ph.PROBE_JS, [cx, cy, tag]))
        # Correctness failures (unreachable point, stale tag) always refuse — only safety refusals are opted out of.
        if not probe["ok"]:
            raise ToolError(f"refused: {probe['reason']}")
        if not getattr(self, "allow_actions", False) and (reason := ph.verdict(probe, click_action)):
            raise ToolError(f"refused: {reason}")
        # Promote the accepted hit only now: a refused probe (or one that lands on a non-control the caller
        # discards) must never leave `__tg_hit` pointing at the rejected target, or _click_at would click it.
        self._guard(lambda: self.page.evaluate("window.__tg_hit = window.__tg_hit_pending"))
        return probe

    def _probe(self, element: Element, click_action: bool) -> dict[str, Any]:
        """Scroll the element into view and run the helper's in-page safety probe; raise ToolError if refused."""
        cx, cy = ph.centre(element.abs_box)
        return self._probe_point(cx, cy, element.tag, click_action)

    def _locate(self, element: Element, click_action: bool) -> dict[str, Any]:
        """The (safety-probed) click point of `element`. The extracted position can be stale once the page has scrolled
        there: a sticky, inner-scrolling sidebar (docs tables of contents) or lazy content moves, and the point then lies on
        a different control or none. In that case the element is found again by label and href, scrolled into view (inner
        scrollers included) and probed at its fresh position. Where it cannot be found again the original outcome stands."""
        stale: ToolError | None = None
        try:
            probe = self._probe(element, click_action)
        except ToolError as exc:
            if "page changed" not in str(exc) and "not reachable" not in str(exc):
                raise
            stale, probe = exc, None
        if probe is not None:
            hit = self._guard(lambda: self.page.evaluate(_HIT_LABEL_JS, [probe["x"], probe["y"], _CONTROLS]))
            if hit is None or not element.text or _same_label(element.text, hit["label"]):
                return probe
        found = self._guard(lambda: self.page.evaluate(_RELOCATE_JS, [element.text, element.href, _CONTROLS])) if element.text else {}
        if not found.get("ok"):
            if stale is not None:
                raise stale
            return probe  # type: ignore[return-value]
        sx, sy = self._guard(lambda: self.page.evaluate("[scrollX, scrollY]"))
        try:
            return self._probe_point(found["x"] + sx, found["y"] + sy, element.tag, click_action)
        except ToolError:
            # The relocated point can land on a non-control (e.g. a field's <label> when the original probe hit a
            # control whose hit-label differs, like an input's placeholder vs its label). Keep the original probe.
            if stale is not None:
                raise stale
            return probe  # type: ignore[return-value]

    def _click_at(self, probe: dict[str, Any]) -> None:
        """Click at the probed point, re-deriving the viewport position at click time: the page can keep scrolling
        between the probe's scroll jump and the click (scroll anchoring, lazy layout, inner scroll containers),
        and the stale viewport point then lands on a different control. The probe stashes the validated hit on
        `window.__tg_hit`, so prefer its live rect — that survives any scroll, window or container. Fall back to
        document-coordinate compensation when the stash is gone (navigation) and to raw coords for older probes."""
        point = self.page.evaluate(
            """() => {
                const hit = window.__tg_hit;
                if (!hit || !hit.isConnected) return null;
                const r = hit.getBoundingClientRect();
                let x = r.left + r.width / 2, y = r.top + r.height / 2;
                // Inside a same-origin frame the rect is frame-local: add each ancestor frame's offset (+ border)
                // until the point is in the top window's viewport coordinates.
                let win = hit.ownerDocument && hit.ownerDocument.defaultView;
                while (win && win !== window) {
                    const fe = win.frameElement;
                    if (!fe) return null;
                    const fr = fe.getBoundingClientRect();
                    x += fr.left + fe.clientLeft; y += fr.top + fe.clientTop;
                    win = fe.ownerDocument.defaultView;
                }
                return [x, y];
            }"""
        )
        if point is not None:
            self.page.mouse.click(point[0], point[1])
        elif "sx" in probe:
            sx, sy = self.page.evaluate("[scrollX, scrollY]")
            self.page.mouse.click(probe["x"] + probe["sx"] - sx, probe["y"] + probe["sy"] - sy)
        else:
            self.page.mouse.click(probe["x"], probe["y"])

    def click(self, element_id: int) -> Element:
        """Click element `element_id`, refusing risky ones (blocked words, form submits). Returns the element."""
        with self.overhead("safety_ms"):
            element = self._element(element_id)
            if not getattr(self, "allow_actions", False) and (word := ph.blocked_word(element)):
                raise ToolError(f"refused: risky: text or context contains the blocked word {word!r}")
            if not element.enabled:
                raise ToolError("refused: the element is disabled")
            probe = self._locate(element, click_action=True)
        self._guard(lambda: self._click_at(probe))
        self._settle()
        return element

    def type_text(self, element_id: int, text: str, press_enter: bool) -> Element:
        """Focus a text field by clicking it, replace its content with `text`, optionally press Enter. Returns the
        element."""
        with self.overhead("safety_ms"):
            element, probe = self._field(element_id)
        self._type_into(probe, text, press_enter)
        self._settle()
        return element

    def _field(self, element_id: int, kind: str = "text field", select_ok: bool = False) -> tuple[Element, dict[str, Any]]:
        """Resolve `element_id` to a typeable field behind the click-safety probe; returns (element, probe point)."""
        element = self._element(element_id)
        if not (select_ok and element.tag == "select") and element.role not in ph.TYPEABLE_ROLES:
            raise ToolError(f"element {element_id} is a {element.role}, not a {kind}")
        if not element.enabled:
            raise ToolError("refused: the element is disabled")
        return element, self._locate(element, click_action=False)

    def _type_into(self, probe: dict[str, Any], text: str, press_enter: bool = False) -> None:
        def act() -> None:
            self._click_at(probe)
            self.page.keyboard.press("ControlOrMeta+A")  # typing over the selection replaces any existing text
            self.page.keyboard.type(text)
            if press_enter:
                self.page.keyboard.press("Enter")

        self._guard(act)

    def fill_field(self, element_id: int, text: str) -> Element:
        """Fill one form field: type into text fields, `select_option` on native selects.
        Same resolution and safety probing as `type_text`. Returns the element."""
        with self.overhead("safety_ms"):
            element, probe = self._field(element_id, kind="form field", select_ok=True)
        if element.tag == "select":
            def pick() -> None:
                target = self.page.evaluate_handle("() => window.__tg_hit").as_element()
                if target is None or target.evaluate("e => e.tagName") != "SELECT":
                    raise ToolError(f"element {element_id} could not be located as a select")
                target.select_option(label=text)

            self._guard(pick)
        else:
            self._type_into(probe, text)
        self._settle()
        return element

    def go_back(self) -> None:
        if self._guard(lambda: self.page.go_back(wait_until="load")) is None:
            raise ToolError("there is no previous page")
        self._settle()

    def succeeded(self, check: dict[str, str]) -> bool:
        try:
            text = self.text() if any(k.startswith("text_") for k in check) else ""
        except ToolError:
            return False
        return check_success(check, self.url, text)
