"""DeskForge-1M (sample) -> our Linux desktop pages and examples, plus its held-out-app test split.

    uv run --extra datasets-convert python tools/deskforge.py --out data/deskforge
    uv run --extra datasets-convert python tools/deskforge.py --out data/deskforge --splits train --max-examples 200   # dry run

Source: `docling-project/DeskForge-1M` (MIT, Gurbuz et al. 2026, arXiv:2610.02320). Splits written, all from the
`index/instructions/<split>.parquet` click instructions:
  * `train`    from `train`, a seeded sample of at most `--max-examples` rows (see below);
  * `dev`      from `val`, at most `--dev-rows` rows, sampled the same way;
  * `test_app` from `test_app` (scenes containing `gnome-system-monitor`, `pluma` or `xarchiver`), at most
    `--test-rows` rows whose clicked element belongs to one of those three applications. EVALUATION set.

Only the `<key>.leaf.json` member of each before-observation is fetched, with one HTTP range request each: the tars
are uncompressed and `index/shard_members.parquet` gives the byte range of every member (`byte_offset` is where the
member's data starts, after its 512-byte tar header; the range is checked against the member's sha256). Screenshots
and whole tars are never downloaded. The Hub rate limits `resolve` requests (3000 per 5 minutes, anonymous), so each
tar is resolved once to its signed CDN URL and the range requests go there. Downloaded leaves are cached under `--raw` (`leaf/`), so a rerun resumes; the
parquet indexes are cached there too (`index/`).

An action is taken ON the before observation, so one page = one before observation (`deskforge-<hash of key>`),
site `deskforge-<app of the clicked element>`, url `ax://deskforge/<app>`. Elements follow the axdump philosophy
(`tools/ax_tree.py`): candidates are interactive or editable elements; static text and labels are context only. The
box is the union of the `visible_fragments` (the `rect` is amodal: it claims pixels another window covers) and
fully occluded elements are dropped. A row whose clicked element is not a kept element is dropped and counted.

Each row yields a "plain" example (query = `primary_instruction`) and, when the row has a second instruction
style, a "paraphrase" example. Sampling is deterministic (`--seed`): rows are ordered so that distinct screens
come first (at most `--max-per-screen` rows per screen), then taken while no application (of the clicked element)
exceeds `--max-app-share` of the row budget. The application comes from `index/transitions` (`action_target_app`),
so the caps are applied before anything is downloaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import threading
import time
from collections import Counter, defaultdict, deque
from contextlib import closing
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence, TypeVar
from urllib.parse import parse_qs, urlparse

import httpx
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from grounder.data.pages import write_examples, write_page
from grounder.records import MAX_CONTEXT_CHARS, MAX_TEXT_CHARS, Element, Example, Page, clean_text, equiv_ids, validate_element
from tools import convert_common
from tools.ax_collect import slugify
from tools.ax_tree import MAX_ELEMENTS, MAX_GROUP_CHARS, build_context, normalised_box

REPO = "docling-project/DeskForge-1M"
BASE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
SPLITS = {"train": "train", "dev": "val", "test_app": "test_app"}  # our split -> DeskForge split
HELD_OUT_APPS = frozenset({"gnome-system-monitor", "pluma", "xarchiver"})
PAGE_ID = "deskforge-{}"
APP_URL = "ax://deskforge/{}"
SITE = "deskforge-{}"
CONTEXT_LABELS = 3  # preceding label texts (reading order, same window) put in the context
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
MAX_ATTEMPTS = 6
MAX_BACKOFF = 300.0  # the Hub's rate limit window is 5 minutes
URL_MARGIN = 120.0  # seconds before expiry a signed URL is renewed
CHUNK_ROWS = 1024
REPORT_SAMPLES = 5
PROGRESS_ROWS = 1000

# AT-SPI element class (`type`) -> record role; None: never a candidate (context or container). Types not listed
# fall back to ROLE_ROLES by AT-SPI role, which also decides the column headers of the Heading class.
TYPE_ROLES: dict[str, str | None] = {
    "Button": "button", "Link": "link", "Text Input": "textbox", "Search Field": "searchbox", "Checkbox": "checkbox",
    "Radiobox": "radio", "Toggles": "switch", "Select": "combobox", "Menu": "menuitem", "Tab": "tab",
    "Slider": "slider", "Steppers": "textbox", "List Item": "option", "File Icon": "option", "Image": "image",
    "Text": None, "Window": None, "Scroll": None, "Status Bar": None, "Side Bar": None, "List": None, "Alert": None,
    "Calendar": None, "Navigation Bar": None, "Toolbar": None, "Table": None, "Tab Bar": None, "Progress bar": None,
    "Video": None, "Bottom navigation": None,
}
ROLE_ROLES: dict[str, str] = {
    "push button": "button", "toggle button": "switch", "link": "link", "check box": "checkbox",
    "radio button": "radio", "menu item": "menuitem", "check menu item": "menuitem", "radio menu item": "menuitem",
    "page tab": "tab", "combo box": "combobox", "entry": "textbox", "password text": "textbox",
    "spin button": "textbox", "slider": "slider", "list item": "option", "tree item": "option",
    "table column header": "button",
}
LABEL_TYPES = frozenset({"Text", "Heading"})
LABEL_ROLES = frozenset({"label", "static", "heading", "text", "paragraph"})
FRAME_ROLE = "frame"
NO_READING_ORDER = 1 << 40
OBJECT_REPLACEMENT = "\ufffc"  # what the accessibility tree reports as the text of a window or an embedded object

Leaf = list[dict[str, Any]]
WindowKey = tuple[str, int]  # (application, window stack index)
T = TypeVar("T")
R = TypeVar("R")


# --- elements -----------------------------------------------------------------------------------


def map_role(type_: str, role: str) -> str | None:
    """The record role of a leaf element's class and AT-SPI role, or None when it is not a candidate class."""
    if type_ in TYPE_ROLES:
        return TYPE_ROLES[type_]
    return ROLE_ROLES.get(role)


def is_interactive(el: dict[str, Any]) -> bool:
    """DeskForge records what an element can do, so no action is assumed: actionable or editable."""
    interaction = el.get("interaction") or {}
    return bool(interaction.get("actionable") or interaction.get("editable"))


def is_fully_occluded(el: dict[str, Any]) -> bool:
    return bool(el.get("is_occluded")) and not el.get("visible_fragments")


def visible_box(el: dict[str, Any]) -> list[float] | None:
    """[x, y, w, h] in pixels: the union bbox of the visible fragments (the amodal `rect` when there are none).
    None when it has no area."""
    rects = el.get("visible_fragments") or [el["rect"]]
    x0, y0 = min(r["x"] for r in rects), min(r["y"] for r in rects)
    x1, y1 = max(r["x"] + r["w"] for r in rects), max(r["y"] + r["h"] for r in rects)
    return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)] if x1 - x0 >= 1 and y1 - y0 >= 1 else None


def leaf_text(el: dict[str, Any]) -> str:
    """visible_text, else the accessible name."""
    return clean_text((el.get("visible_text") or "").replace(OBJECT_REPLACEMENT, "") or el.get("name"), MAX_TEXT_CHARS)


def is_label(el: dict[str, Any]) -> bool:
    return el["type"] in LABEL_TYPES and el["role"] in LABEL_ROLES


def window_key(el: dict[str, Any]) -> WindowKey:
    return el.get("app_name") or "", el.get("_window_stack_index", -1)


def reading_key(indexed: tuple[int, dict[str, Any]]) -> tuple[int, int]:
    """Sort key of an (index, element) pair: reading order, elements without one last, then leaf order."""
    index, el = indexed
    order = el.get("reading_order_index")
    return (NO_READING_ORDER if order is None else order, index)


def window_titles(leaf: Leaf) -> dict[WindowKey, str]:
    """Window title per (app, stack index): the name of its frame (a dialog of the same window only fills a gap)."""
    titles: dict[WindowKey, str] = {}
    for el in leaf:
        if el["type"] != "Window" or not el.get("name"):
            continue
        key = window_key(el)
        if el["role"] == FRAME_ROLE or key not in titles:
            titles[key] = clean_text(el["name"], MAX_CONTEXT_CHARS)
    return titles


@dataclass
class Screen:
    """One before observation as candidate elements.

    windows[i]:  (app, stack index) of element i.
    rejected:    uid -> why an element with a uid is not a candidate (the clicked element may be one of them).
    dropped:     elements not kept, by reason.
    """

    elements: list[Element] = field(default_factory=list)
    uid_to_id: dict[str, int] = field(default_factory=dict)
    windows: list[WindowKey] = field(default_factory=list)
    titles: dict[WindowKey, str] = field(default_factory=dict)
    rejected: dict[str, str] = field(default_factory=dict)
    dropped: Counter[str] = field(default_factory=Counter)

    def reject(self, el: dict[str, Any], reason: str) -> None:
        self.dropped[reason] += 1
        if el.get("uid"):
            self.rejected[el["uid"]] = reason

    def add(self, el: dict[str, Any], role: str, box: list[float], width: int, height: int, context: str) -> None:
        if len(self.elements) >= MAX_ELEMENTS:
            return self.reject(el, "over_cap")
        interaction = el.get("interaction") or {}
        if el.get("uid"):
            self.uid_to_id[el["uid"]] = len(self.elements)
        self.windows.append(window_key(el))
        self.elements.append(validate_element({
            "id": len(self.elements), "role": role, "text": leaf_text(el), "context": context, "tag": el["role"],
            "box": normalised_box(box, [0, 0, width, height]), "abs_box": box, "in_viewport": True,
            "enabled": bool(interaction.get("enabled", True)), "source": "ax",
        }))


def assess(el: dict[str, Any]) -> tuple[str, list[float]] | str:
    """(role, pixel box) of a candidate, else the reason it is dropped."""
    role = map_role(el["type"], el["role"]) if is_interactive(el) else None
    if role is None:
        return "not_candidate"
    if is_fully_occluded(el):
        return "fully_occluded"
    box = visible_box(el)
    return (role, box) if box else "empty_box"


def build_screen(leaf: Leaf, width: int, height: int) -> Screen:
    """Candidate elements in reading order. Context = owning app, window title and the last labels before the element
    (reading order, same window)."""
    screen = Screen(titles=window_titles(leaf))
    labels: defaultdict[WindowKey, deque[str]] = defaultdict(lambda: deque(maxlen=CONTEXT_LABELS))
    for _, el in sorted(enumerate(leaf), key=reading_key):
        key = window_key(el)
        if is_label(el):
            if not is_fully_occluded(el) and leaf_text(el):
                labels[key].append(leaf_text(el))
            continue
        verdict = assess(el)
        if isinstance(verdict, str):
            screen.reject(el, verdict)
            continue
        role, box = verdict
        title = clean_text(screen.titles.get(key, ""), MAX_GROUP_CHARS)
        screen.add(el, role, box, width, height, build_context(key[0], title, list(labels[key]), leaf_text(el)))
    return screen


def page_for(key: str, screen: Screen, size: tuple[int, int], target_id: int) -> Page:
    """The page of an observation; the app (site, url) and title are those of the clicked element."""
    app, stack = screen.windows[target_id]
    title = screen.titles.get((app, stack)) or app
    return Page(page_id=page_id_for(key), url=APP_URL.format(app), site=SITE.format(slugify(app)), title=title,
                viewport=list(size), doc_size=list(size), elements=screen.elements)


def page_id_for(key: str) -> str:
    return PAGE_ID.format(hashlib.sha1(key.encode("utf-8")).hexdigest()[:12])


# --- instructions -----------------------------------------------------------------------------------


def paraphrase(row: dict[str, Any]) -> str | None:
    """The text of another instruction style than the primary one (the first by style name), if it differs."""
    others = sorted((v["style"], v["text"]) for v in row["instruction_variants"]
                    if v["style"] != row["primary_style"] and v["text"].strip().casefold() != row["primary_instruction"].strip().casefold())
    return others[0][1] if others else None


def examples_for(page: Page, target_id: int, row: dict[str, Any]) -> list[Example]:
    ids = equiv_ids(page.elements, target_id)
    queries = [("plain", row["primary_instruction"]), ("paraphrase", paraphrase(row))]
    return [Example(page_id=page.page_id, kind=kind, mode="do", query=query, target_id=target_id, equiv_ids=ids)
            for kind, query in queries if query]


# --- sampling ---------------------------------------------------------------------------------------


def candidate_order(before_keys: Sequence[str], seed: str, max_per_screen: int) -> list[int]:
    """Row indices in seeded random order, distinct screens first: every screen's first row, then every second row
    (a screen contributes at most `max_per_screen` rows)."""
    shuffled = list(range(len(before_keys)))
    random.Random(seed).shuffle(shuffled)
    seen: Counter[str] = Counter()
    rounds: list[list[int]] = [[] for _ in range(max_per_screen)]
    for i in shuffled:
        if seen[before_keys[i]] < max_per_screen:
            rounds[seen[before_keys[i]]].append(i)
            seen[before_keys[i]] += 1
    return [i for round_ in rounds for i in round_]


@dataclass
class Sampler:
    """Admits rows by the application of their clicked element until `max_rows`.

    app_cap:      most rows of one application (None: no limit).
    allowed_apps: only these applications (None: all).
    """

    max_rows: int
    app_cap: int | None = None
    allowed_apps: frozenset[str] | None = None
    per_app: Counter[str] = field(default_factory=Counter)

    @property
    def full(self) -> bool:
        return sum(self.per_app.values()) >= self.max_rows

    def reject_reason(self, app: str) -> str | None:
        if self.allowed_apps is not None and app not in self.allowed_apps:
            return "app_not_held_out"
        if self.app_cap is not None and self.per_app[app] >= self.app_cap:
            return "app_cap"
        return None

    def admit(self, app: str) -> None:
        self.per_app[app] += 1


def app_row_cap(max_rows: int, max_share: float) -> int:
    return max(1, int(max_rows * max_share))


# --- index and range requests -------------------------------------------------------------------------


@dataclass(frozen=True)
class Member:
    """A tar member: its data is `size` bytes at `offset` of the tar at `tar_path` (after the 512-byte header)."""

    tar_path: str
    offset: int
    size: int
    sha256: str

    @property
    def range_header(self) -> str:
        return f"bytes={self.offset}-{self.offset + self.size - 1}"


class FetchError(Exception):
    pass


def ensure_index(client: httpx.Client, raw: Path, name: str) -> Path:
    """The index parquet `name` (e.g. "instructions/train"), downloaded into `raw/index` on first use."""
    path = raw / "index" / f"{name}.parquet"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        with client.stream("GET", f"{BASE_URL}index/{name}.parquet") as response, tmp.open("wb") as f:
            response.raise_for_status()
            f.writelines(response.iter_bytes())
        tmp.rename(path)
        print(f"downloaded index/{name}.parquet ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def read_leaf_members(path: Path, keys: pa.Array) -> dict[str, Member]:
    table = pq.read_table(path, columns=["observation_key", "kind", "tar_path", "byte_offset", "byte_size", "sha256"],
                          filters=[("kind", "=", "leaf")])
    table = table.filter(pc.is_in(table["observation_key"], value_set=keys))
    cols = [table[c].to_pylist() for c in ("observation_key", "tar_path", "byte_offset", "byte_size", "sha256")]
    return {key: Member(*rest) for key, *rest in zip(*cols)}


def read_sizes(path: Path, keys: pa.Array) -> dict[str, tuple[int, int]]:
    table = pq.read_table(path, columns=["observation_key", "width", "height"])
    table = table.filter(pc.is_in(table["observation_key"], value_set=keys))
    return {k: (w, h) for k, w, h in zip(*(table[c].to_pylist() for c in ("observation_key", "width", "height")))}


class Retryable(Exception):
    """A failed attempt worth repeating after `delay` seconds (None: the caller's backoff)."""

    def __init__(self, reason: str, delay: float | None = None) -> None:
        super().__init__(reason)
        self.delay = delay


def retry_delay(response: httpx.Response) -> float | None:
    """Seconds the server asks to wait: Retry-After, else the reset time `t=` of the Hub's RateLimit header."""
    found = response.headers.get("retry-after") or next(iter(re.findall(r"\bt=(\d+)", response.headers.get("ratelimit", ""))), None)
    return min(float(found), MAX_BACKOFF) if found else None


def url_expiry(url: str) -> float:
    """Unix time a signed CDN URL stops working (its `Expires` parameter); a URL without one is taken as permanent."""
    expires = parse_qs(urlparse(url).query).get("Expires")
    return float(expires[0]) if expires else float("inf")


class Fetcher:
    """Range requests on the tars.

    The Hub's `resolve` endpoint is rate limited (anonymous: 3000 requests per 5 minutes) and redirects to a signed CDN
    URL valid for about an hour, which serves range requests without that limit. So each tar is resolved once (and
    again when its URL expires) and every member is then read from the CDN URL.
    """

    def __init__(self, client: httpx.Client, sleep: Callable[[float], None] = time.sleep) -> None:
        self.client = client
        self.sleep = sleep
        self.urls: dict[str, str] = {}
        self.lock = threading.Lock()

    def cdn_url(self, tar_path: str) -> str:
        with self.lock:
            url = self.urls.get(tar_path)
        if url is None or url_expiry(url) < time.time() + URL_MARGIN:
            url = self.resolve(tar_path)
            with self.lock:
                self.urls[tar_path] = url
        return url

    def resolve(self, tar_path: str) -> str:
        response = self.client.head(BASE_URL + tar_path, follow_redirects=False)
        if response.is_redirect:
            return response.headers["location"]
        if response.status_code in RETRY_STATUS:
            raise Retryable(f"resolve HTTP {response.status_code}", retry_delay(response))
        if response.status_code != 200:
            raise FetchError(f"{tar_path}: resolve HTTP {response.status_code}")
        return BASE_URL + tar_path

    def attempt(self, member: Member) -> bytes:
        with self.client.stream("GET", self.cdn_url(member.tar_path), headers={"Range": member.range_header}) as response:
            if response.status_code in RETRY_STATUS:
                raise Retryable(f"HTTP {response.status_code}", retry_delay(response))
            if response.status_code == 403:  # a signed URL that expired early: resolve again
                with self.lock:
                    self.urls.pop(member.tar_path, None)
                raise Retryable("HTTP 403", 0.0)
            if response.status_code != 206:
                raise FetchError(f"{member.tar_path}: HTTP {response.status_code} (expected 206 partial content)")
            return verified(response.read(), member)

    def fetch(self, member: Member) -> bytes:
        """The member's data, checked against its sha256; 429 / 5xx / transport errors are retried with backoff."""
        reason = ""
        for attempt in range(MAX_ATTEMPTS):
            delay = min(2.0**attempt, MAX_BACKOFF)
            try:
                return self.attempt(member)
            except Retryable as err:
                reason, delay = str(err), err.delay if err.delay is not None else delay
            except httpx.TransportError as err:
                reason = f"{type(err).__name__}: {err}"
            if attempt + 1 < MAX_ATTEMPTS:
                self.sleep(delay)
        raise FetchError(f"{member.tar_path} {member.range_header}: {reason} after {MAX_ATTEMPTS} attempts")


def verified(data: bytes, member: Member) -> bytes:
    if len(data) != member.size or hashlib.sha256(data).hexdigest() != member.sha256:
        raise FetchError(f"{member.tar_path} {member.range_header}: size or sha256 mismatch")
    return data


class LeafCache:
    """`<dir>/<key>.leaf.json`, written atomically and re-verified against the sha256 on every read."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        return self.directory / f"{key}.leaf.json"

    def load(self, key: str, member: Member) -> bytes | None:
        try:
            data = self.path(key).read_bytes()
        except FileNotFoundError:
            return None
        return data if len(data) == member.size and hashlib.sha256(data).hexdigest() == member.sha256 else None

    def save(self, key: str, data: bytes) -> None:
        tmp = self.path(key).with_suffix(".part")
        tmp.write_bytes(data)
        tmp.rename(self.path(key))


@dataclass
class Job:
    row: dict[str, Any]
    size: tuple[int, int] | None
    member: Member | None


@dataclass
class Outcome:
    job: Job
    data: bytes | None = None
    downloaded: bool = False
    error: str = ""


def fetch_leaf(fetcher: Fetcher, cache: LeafCache, job: Job) -> Outcome:
    """The leaf of a job from the cache or the network; a failure is an Outcome with an error, not an exception."""
    key = job.row["before_key"]
    if job.member is None or job.size is None:
        return Outcome(job, error="not_in_index")
    cached = cache.load(key, job.member)
    if cached is not None:
        return Outcome(job, cached)
    try:
        data = fetcher.fetch(job.member)
    except FetchError as err:
        print(f"fetch failed: {err}", flush=True)
        return Outcome(job, error="fetch_failed")
    cache.save(key, data)
    return Outcome(job, data, downloaded=True)


def ordered_parallel(fn: Callable[[T], R], items: Iterable[T], workers: int) -> Iterator[R]:
    """`fn` over `items` on a thread pool with at most 2 x `workers` in flight; results come in input order and the
    items are consumed lazily, so stopping early stops the work."""
    with ThreadPoolExecutor(workers) as pool:
        source = iter(items)
        pending: deque[Future[R]] = deque(pool.submit(fn, item) for item in islice(source, 2 * workers))
        try:
            while pending:
                yield pending.popleft().result()
                for item in islice(source, 1):
                    pending.append(pool.submit(fn, item))
        finally:
            for future in pending:
                future.cancel()


# --- conversion -----------------------------------------------------------------------------------------


@dataclass
class Summary:
    query: str
    role: str
    text: str
    n_elements: int
    app: str


@dataclass
class Conversion:
    rows_read: int = 0
    leaves_downloaded: int = 0
    bytes_downloaded: int = 0
    seconds: float = 0.0
    dropped: Counter[str] = field(default_factory=Counter)
    element_drops: Counter[str] = field(default_factory=Counter)
    apps: Counter[str] = field(default_factory=Counter)  # rows per app of the clicked element
    target_roles: Counter[str] = field(default_factory=Counter)
    element_counts: list[int] = field(default_factory=list)  # per written page
    examples: list[Example] = field(default_factory=list)
    summaries: list[Summary] = field(default_factory=list)
    written: set[str] = field(default_factory=set)

    @property
    def rows(self) -> int:
        return sum(self.apps.values())


class SplitConverter:
    """Turns fetched leaves into pages and examples, one row at a time."""

    def __init__(self, pages_dir: Path) -> None:
        self.pages_dir = pages_dir
        self.conv = Conversion()

    def consume(self, outcome: Outcome) -> None:
        conv, row = self.conv, outcome.job.row
        conv.rows_read += 1
        if outcome.data is None:
            conv.dropped[outcome.error] += 1
            return
        if outcome.downloaded:
            conv.leaves_downloaded += 1
            conv.bytes_downloaded += len(outcome.data)
        width, height = outcome.job.size or (0, 0)
        screen = build_screen(json.loads(outcome.data), width, height)
        target_id = screen.uid_to_id.get(row["target_uid"])
        if target_id is None:
            conv.dropped["target_" + screen.rejected.get(row["target_uid"], "not_in_leaf")] += 1
            return
        self.record(row["before_key"], screen, (width, height), target_id, row)

    def record(self, key: str, screen: Screen, size: tuple[int, int], target_id: int, row: dict[str, Any]) -> None:
        conv = self.conv
        page = page_for(key, screen, size, target_id)
        if page.page_id not in conv.written:
            write_page(page, self.pages_dir)
            conv.written.add(page.page_id)
            conv.element_counts.append(len(page.elements))
            conv.element_drops.update(screen.dropped)
        target = page.elements[target_id]
        conv.apps[screen.windows[target_id][0]] += 1
        conv.target_roles[target.role] += 1
        conv.examples.extend(examples_for(page, target_id, row))
        conv.summaries.append(Summary(row["primary_instruction"], target.role, target.text, len(page.elements), screen.windows[target_id][0]))


def target_apps(rows: pa.Table, transitions: pa.Table) -> list[str | None]:
    """The application of each row's clicked element, from the transitions index (None: transition not found)."""
    found = pc.index_in(rows["transition_id"], value_set=transitions["transition_id"])
    return transitions["action_target_app"].take(found).to_pylist()


def select_rows(order: Iterable[int], apps: Sequence[str | None], sampler: Sampler) -> tuple[list[int], Counter[str]]:
    """The rows of `order` the sampler admits, with the reasons for the skipped ones. The application is known
    before anything is downloaded, so no leaf is fetched for a row the caps reject."""
    selected: list[int] = []
    skipped: Counter[str] = Counter()
    for i in order:
        if sampler.full:
            break
        app = apps[i]
        reason = "no_transition" if app is None else sampler.reject_reason(app)
        if reason:
            skipped[reason] += 1
        else:
            sampler.admit(app)
            selected.append(i)
    return selected, skipped


def jobs_for(table: pa.Table, order: Iterable[int], sizes: dict[str, tuple[int, int]], members: dict[str, Member]) -> Iterator[Job]:
    """Rows of `table` in `order` (read in chunks) as fetch jobs."""
    indices = iter(order)
    while chunk := list(islice(indices, CHUNK_ROWS)):
        for row in table.take(pa.array(chunk)).to_pylist():
            yield Job(row, sizes.get(row["before_key"]), members.get(row["before_key"]))


def convert_split(client: httpx.Client, raw: Path, out: Path, split: str, sampler: Sampler, max_per_screen: int, seed: int, workers: int) -> Conversion:
    """Convert one of our splits; its pages / examples files are replaced."""
    source = SPLITS[split]
    columns = ["transition_id", "before_key", "target_uid", "primary_style", "primary_instruction", "instruction_variants"]
    table = pq.read_table(ensure_index(client, raw, f"instructions/{source}"), columns=columns)
    keys = pc.unique(table["before_key"])
    members = read_leaf_members(ensure_index(client, raw, "shard_members"), keys)
    sizes = read_sizes(ensure_index(client, raw, f"observations/{source}"), keys)
    transitions = pq.read_table(ensure_index(client, raw, f"transitions/{source}"), columns=["transition_id", "action_target_app"])
    order = candidate_order(table["before_key"].to_pylist(), f"{seed}:{split}", max_per_screen)
    selected, skipped = select_rows(order, target_apps(table, transitions), sampler)

    pages_dir = out / "pages" / split
    shutil.rmtree(pages_dir, ignore_errors=True)
    converter = SplitConverter(pages_dir)
    converter.conv.dropped.update(skipped)
    cache, fetcher = LeafCache(raw / "leaf"), Fetcher(client)
    started = time.monotonic()
    with closing(ordered_parallel(lambda job: fetch_leaf(fetcher, cache, job), jobs_for(table, selected, sizes, members), workers)) as outcomes:
        for outcome in outcomes:
            converter.consume(outcome)
            if converter.conv.rows_read % PROGRESS_ROWS == 0:
                print(f"[{split}] {converter.conv.rows_read} / {len(selected)} rows, {time.monotonic() - started:.0f} s", flush=True)
    converter.conv.seconds = time.monotonic() - started
    (out / "examples").mkdir(parents=True, exist_ok=True)
    write_examples(converter.conv.examples, out / "examples" / f"{split}.jsonl")
    return converter.conv


# --- reports and output -------------------------------------------------------------------------------


def traffic_text(conv: Conversion) -> str:
    per_leaf = conv.bytes_downloaded / conv.leaves_downloaded / 1e3 if conv.leaves_downloaded else 0.0
    rate = conv.leaves_downloaded / conv.seconds if conv.seconds else 0.0
    return (f"rows read: {conv.rows_read}  kept rows: {conv.rows}  leaves downloaded: {conv.leaves_downloaded} "
            f"({conv.bytes_downloaded / 1e6:.1f} MB, {per_leaf:.0f} kB per leaf, {rate:.1f} leaves/s)  time: {conv.seconds / 60:.1f} min")


def apps_text(conv: Conversion) -> str:
    return "apps (share of rows): " + ", ".join(f"{a} {n / conv.rows:.1%}" for a, n in conv.apps.most_common())


def samples_text(conv: Conversion, seed: int) -> str:
    picks = random.Random(seed).sample(conv.summaries, min(REPORT_SAMPLES, len(conv.summaries)))
    return "\n".join(f"  {s.query!r} -> {s.role} {s.text!r}  ({s.n_elements} elements, {s.app})" for s in picks)


def report_text(name: str, conv: Conversion, seed: int) -> str:
    return "\n".join([
        f"[{name}] {traffic_text(conv)}",
        f"[{name}] dropped rows: {dict(conv.dropped) or 0}  dropped elements (pages written): {dict(conv.element_drops) or 0}",
        convert_common.stats_text(name, len(conv.examples), len(conv.apps), conv.element_counts, conv.target_roles),
        f"[{name}] {apps_text(conv)}",
        f"[{name}] random examples:\n{samples_text(conv, seed)}",
    ])


SOURCE_NOTE = """DeskForge-1M (sample), converted by tools/deskforge.py.

Source:   https://huggingface.co/datasets/docling-project/DeskForge-1M
License:  MIT.
Use:      train / dev = training and development data for the MIT-licensed release;
          test_app = held-out-application EVALUATION (gnome-system-monitor, pluma, xarchiver; rows whose clicked
          element belongs to one of them).
Citation: Gurbuz, Nassar, Hong, Pollefeys, Staar 2026, "DeskForge: Dense Supervision from Desktop Environments for
          Computer-Use Agents", arXiv:2610.02320.
Sample:   {parameters}
"""


def write_source_note(out: Path, args: argparse.Namespace) -> None:
    parameters = ", ".join(f"{k}={getattr(args, k)}" for k in ("seed", "max_examples", "dev_rows", "test_rows", "max_app_share", "max_per_screen"))
    (out / "SOURCE.txt").write_text(SOURCE_NOTE.format(parameters=parameters), encoding="utf-8")


def samplers(args: argparse.Namespace) -> dict[str, Sampler]:
    return {
        "train": Sampler(args.max_examples, app_row_cap(args.max_examples, args.max_app_share)),
        "dev": Sampler(args.dev_rows, app_row_cap(args.dev_rows, args.max_app_share)),
        "test_app": Sampler(args.test_rows, allowed_apps=HELD_OUT_APPS),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/deskforge"))
    parser.add_argument("--raw", type=Path, default=None, help="cache of the parquet indexes and leaf files (default: OUT/raw)")
    parser.add_argument("--splits", nargs="+", choices=sorted(SPLITS), default=list(SPLITS))
    parser.add_argument("--max-examples", type=int, default=12_000, help="train: rows (each gives up to 2 examples)")
    parser.add_argument("--dev-rows", type=int, default=1_000)
    parser.add_argument("--test-rows", type=int, default=2_000)
    parser.add_argument("--max-app-share", type=float, default=0.12, help="train / dev: most rows of one application")
    parser.add_argument("--max-per-screen", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=32, help="concurrent range requests")
    args = parser.parse_args(argv)

    raw = args.raw or args.out / "raw"
    limits = httpx.Limits(max_connections=args.workers, max_keepalive_connections=args.workers)
    with httpx.Client(follow_redirects=True, timeout=60.0, limits=limits) as client:
        for split, sampler in samplers(args).items():
            if split in args.splits:
                conv = convert_split(client, raw, args.out, split, sampler, args.max_per_screen, args.seed, args.workers)
                print(report_text(split, conv, args.seed))
    write_source_note(args.out, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
