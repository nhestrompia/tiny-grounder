"""Crawl a site list into page files (implementation plan Step 2).

    uv run python -m grounder.data.crawl --sites sites.txt --out data/pages --links 2 --screenshots-for test
    uv run python -m grounder.data.crawl --out data/pages --stats

Layout: `{out}/{split}/{site}/{page_id}.json.gz` (split = train | dev | test, site = registrable domain),
screenshots next to the page as `{page_id}.png`, and `{out}/crawl_log.jsonl`. The split lives in the
path, so `grounder.data.pages.iter_pages(out / "train")` loads one split; `iter_pages(out)` loads all.

Splitting: SHA-1 of the registrable domain modulo 100 (0-89 train, 90-94 dev, 95-99 test), so every
subdomain of a site lands in the same split and the split never depends on crawl order.

Template dedupe (plan-review item 5): a page is dropped when the multiset of its element texts has
Jaccard >= 0.9 with an already kept page. The multiset is turned into a set of (text, occurrence)
pairs, which makes the set Jaccard exact; pages are bucketed by size so only sizes that can reach
0.9 are compared. Which of two near-identical pages is kept depends on crawl order.

Page selection: the homepage, then up to `--links` more pages. If the homepage has a GET search form
whose action robots.txt allows, one result page (the form filled with a word from `SEARCH_WORDS`) is
among them; the rest are same-site links sampled with weights (`link_score` boosts search, product,
shop, settings, contact, help, ... pages). Each log line records `via`: home | link | search.

Resuming: a site with a `site_done` line in the log is skipped; in an unfinished site the homepage is
fetched again (its links are needed) and any page file already on disk is not rewritten.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import logging
import random
import re
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from urllib.parse import urldefrag, urlencode, urlsplit
from urllib.robotparser import RobotFileParser

import httpx
import numpy as np
import tldextract
from playwright.async_api import Browser, BrowserContext, Error as PlaywrightError, async_playwright

from grounder.data.pages import iter_pages, write_page
from grounder.extract.web import extract_page
from grounder.records import Page

logger = logging.getLogger("grounder.crawl")

USER_AGENT = "tiny-grounder-research"
VIEWPORT = {"width": 1280, "height": 800}
SPLITS: tuple[str, ...] = ("train", "dev", "test")
SPLIT_PERCENT: tuple[int, ...] = (90, 95, 100)  # cumulative upper bounds of train, dev, test
TRANCO_URL = "https://tranco-list.eu/top-1m.csv.zip"
LOG_NAME = "crawl_log.jsonl"
MIN_ELEMENTS = 5
DEDUPE_JACCARD = 0.9
NAV_TIMEOUT_MS = 15_000
SETTLE_MS = 500
PAGE_BUDGET_S = 90
ROBOTS_TIMEOUT_S = 10

_EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)  # bundled suffix list, no network
_BLOCKED_PATH_RE = re.compile(
    r"(?:^|[/_.-])(?:log-?in|log-?out|sign-?in|sign-?out|sign-?up|register|cart|basket|checkout|"
    r"account|password|oauth|auth|download)(?:$|[/_.-])",
    re.IGNORECASE,
)
LINK_BOOST = 5.0
_BOOST_RE = re.compile(
    r"search|product|shop|categor|catalog|browse|filter|setting|preference|contact|support|help|feedback|"
    r"newsletter|subscribe|compare|calculator|tools?\b|docs?\b|faq|form",
    re.IGNORECASE,
)
SEARCH_WORDS: tuple[str, ...] = ("shoes", "weather", "python", "news", "recipe", "music", "laptop", "travel")
_SEARCH_NAMES = frozenset({"q", "query", "search", "s", "keyword", "keywords", "term", "text"})
LINKS_JS = "els => els.map(e => [e.href, (e.innerText || e.getAttribute('aria-label') || '').trim()])"
# Attributes, not properties: an input named "action" or "method" shadows form.action / form.method.
FORMS_JS = """forms => forms.map(f => ({
  action: new URL(f.getAttribute('action') || '', document.baseURI).href,
  method: f.getAttribute('method') || 'get',
  fields: [...f.elements].filter(e => e.getAttribute('name') && !e.disabled).map(e => ({
    name: e.getAttribute('name'), type: (e.getAttribute('type') || e.type || '').toLowerCase(),
    value: String(e.value ?? ''), checked: !!e.checked}))}))"""
_FILE_EXTENSIONS = frozenset(
    ".pdf .zip .gz .tar .tgz .rar .7z .exe .dmg .pkg .apk .iso .mp3 .mp4 .avi .mov .wav .jpg .jpeg .png "
    ".gif .svg .webp .ico .xml .json .csv .rss .doc .docx .xls .xlsx .ppt .pptx".split()
)


# --- sites, splits, links ---------------------------------------------------------------------


def registrable_domain(url_or_host: str) -> str:
    """eTLD+1 of a URL or host (the host itself for IPs and unknown suffixes)."""
    parts = urlsplit(url_or_host if "://" in url_or_host else f"//{url_or_host}")
    host = (parts.hostname or "").lower()
    return _EXTRACT(host).top_domain_under_public_suffix or host


def split_for(domain: str) -> str:
    """Deterministic 90/5/5 split of a registrable domain."""
    bucket = int(hashlib.sha1(domain.encode("utf-8")).hexdigest(), 16) % 100
    return next(name for name, bound in zip(SPLITS, SPLIT_PERCENT) if bucket < bound)


def start_url(entry: str) -> str:
    """A site-list entry is a bare domain (crawled over https) or a full URL."""
    return entry if "://" in entry else f"https://{entry}/"


def page_id_for(url: str) -> str:
    return hashlib.sha1(urldefrag(url)[0].encode("utf-8")).hexdigest()[:16]


def is_crawlable_link(url: str, site: str) -> bool:
    """Same-site http(s) page that is not an obvious login/logout/cart/checkout page or a file download."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return False
    if registrable_domain(url) != site:
        return False
    path = parts.path.lower()
    if Path(path).suffix in _FILE_EXTENSIONS:
        return False
    return not _BLOCKED_PATH_RE.search(path)


def link_score(url: str, text: str = "") -> float:
    """Sampling weight of a link: pages whose path or anchor text suggests controls are boosted."""
    return LINK_BOOST if _BOOST_RE.search(f"{urlsplit(url).path} {text}") else 1.0


def choose_links(links: Iterable[tuple[str, str]], home: str, site: str, k: int, seed: int) -> list[str]:
    """Up to `k` distinct crawlable links other than the homepage, seeded per site.

    `links` are (url, anchor text) pairs. Sampling is weighted by `link_score`, without replacement.
    """
    home_key = urldefrag(home)[0].rstrip("/")
    texts: dict[str, str] = {}
    for url, text in links:
        url = urldefrag(url)[0]
        if is_crawlable_link(url, site) and url not in (home_key, home_key + "/"):
            texts[url] = texts.get(url) or text
    candidates = sorted(texts)
    rng = random.Random(f"{seed}:{site}")
    keys = {u: rng.random() ** (1.0 / link_score(u, texts[u])) for u in candidates}
    return sorted(candidates, key=lambda u: -keys[u])[:k]


def _search_rank(field_: dict) -> tuple[bool, bool]:
    """Prefer type=search inputs, then conventional names (q, query, ...)."""
    return field_["type"] == "search", field_["name"].lower() in _SEARCH_NAMES


def search_url(forms: Iterable[dict], site: str, query: str) -> str | None:
    """A same-site result-page URL made by filling the best GET search form with `query`.

    Only forms that submit with GET qualify, never ones with a password field. The query is
    URL-encoded; hidden fields and checked boxes keep their default values.
    """
    best: tuple[tuple[bool, bool], str, list[tuple[str, str]]] | None = None
    for form in forms:
        action, method, fields = form.get("action"), form.get("method", "get"), form.get("fields", [])
        if not isinstance(action, str) or not isinstance(method, str):
            continue
        action = urldefrag(action)[0]
        if method.lower() != "get" or not is_crawlable_link(action, site):
            continue
        if any(f["type"] == "password" for f in fields):
            continue
        search = [f for f in fields if f["type"] in ("search", "text") and f["name"]]
        if not search:
            continue
        field_ = max(search, key=_search_rank)
        params = [(f["name"], f["value"]) for f in fields if f["type"] == "hidden"]
        params += [(f["name"], f["value"]) for f in fields if f["type"] in ("checkbox", "radio") and f["checked"]]
        params.append((field_["name"], query))
        if best is None or _search_rank(field_) > best[0]:
            best = (_search_rank(field_), action, params)
    if best is None:
        return None
    return urlsplit(best[1])._replace(query=urlencode(best[2]), fragment="").geturl()


def read_sites(path: str | Path) -> list[str]:
    """One domain or URL per line; blank lines and `#` comments are ignored."""
    lines = (line.split("#")[0].strip() for line in Path(path).read_text(encoding="utf-8").splitlines())
    return [line for line in lines if line]


def load_tranco(n: int, cache_dir: str | Path, client: httpx.Client | None = None) -> list[str]:
    """Top `n` domains of the Tranco list, downloaded once into `cache_dir`."""
    zip_path = Path(cache_dir) / "tranco-top-1m.csv.zip"
    if not zip_path.exists():
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with (client or httpx.Client(follow_redirects=True, timeout=120)) as c:
            response = c.get(TRANCO_URL)
            response.raise_for_status()
        zip_path.write_bytes(response.content)
    with zipfile.ZipFile(zip_path) as archive, archive.open(archive.namelist()[0]) as member:
        rows = io.TextIOWrapper(member, encoding="utf-8")
        return [line.strip().split(",", 1)[1] for _, line in zip(range(n), rows) if "," in line]


# --- robots.txt -------------------------------------------------------------------------------


class RobotsError(Exception):
    """robots.txt could not be fetched (anything but a 404/410 or a successful response)."""


async def load_robots(client: httpx.AsyncClient, origin: str) -> RobotFileParser:
    """Fetch and parse `origin`/robots.txt. A missing file (404/410) allows everything."""
    try:
        response = await client.get(f"{origin}/robots.txt", headers={"User-Agent": USER_AGENT})
    except httpx.HTTPError as e:
        raise RobotsError(f"robots.txt fetch failed: {type(e).__name__}: {e}") from e
    parser = RobotFileParser()
    if response.status_code in (404, 410):
        parser.parse([])
    elif response.status_code == 200:
        parser.parse(response.text.splitlines())
    else:
        raise RobotsError(f"robots.txt returned HTTP {response.status_code}")
    parser.modified()
    return parser


# --- template dedupe --------------------------------------------------------------------------


def text_signature(page: Page) -> frozenset[tuple[str, int]]:
    """The multiset of non-empty element texts as a set of (text, occurrence index) pairs."""
    seen: Counter[str] = Counter()
    pairs = set()
    for el in page.elements:
        text = el.text.casefold()
        if text:
            pairs.add((text, seen[text]))
            seen[text] += 1
    return frozenset(pairs)


def jaccard(a: frozenset, b: frozenset) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 0.0


class TemplateIndex:
    """Remembers kept pages' signatures and flags near-duplicates (exact Jaccard >= `threshold`)."""

    def __init__(self, threshold: float = DEDUPE_JACCARD) -> None:
        self.threshold = threshold
        self._by_size: dict[int, list[frozenset]] = {}

    def is_duplicate(self, signature: frozenset) -> bool:
        n = len(signature)
        if n == 0:
            return False
        low, high = int(np.ceil(self.threshold * n)), int(n / self.threshold)
        return any(
            jaccard(signature, other) >= self.threshold
            for size in range(low, high + 1)
            for other in self._by_size.get(size, ())
        )

    def add(self, signature: frozenset) -> None:
        self._by_size.setdefault(len(signature), []).append(signature)


# --- log --------------------------------------------------------------------------------------


class CrawlLog:
    """Append-only JSONL log; one line per page attempt plus a `site_done` line per finished site."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, url: str, status: str, n_elements: int = 0, reason: str = "", site: str = "", via: str = "") -> None:
        record = {"url": url, "status": status, "n_elements": n_elements, "reason": reason, "site": site, "via": via}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def finished_sites(self) -> set[str]:
        if not self.path.exists():
            return set()
        with open(self.path, encoding="utf-8") as f:
            records = (json.loads(line) for line in f if line.strip())
            return {r["site"] for r in records if r["status"] == "site_done"}


# --- browser ----------------------------------------------------------------------------------


class BrowserSession:
    """One Chromium instance, relaunched when it has died. Each page gets its own context."""

    def __init__(self, playwright, args: tuple[str, ...]) -> None:
        self._playwright = playwright
        self._args = list(args)
        self._browser: Browser | None = None
        self._lock = asyncio.Lock()

    async def _ensure(self) -> Browser:
        async with self._lock:
            if self._browser is None or not self._browser.is_connected():
                if self._browser is not None:
                    logger.warning("browser died, restarting")
                self._browser = await self._playwright.chromium.launch(args=self._args)
            return self._browser

    async def new_context(self) -> BrowserContext:
        for attempt in range(2):
            browser = await self._ensure()
            try:
                return await browser.new_context(viewport=VIEWPORT, user_agent=USER_AGENT)
            except PlaywrightError as e:
                logger.warning("new_context failed (attempt %d): %s", attempt + 1, e)
                if not browser.is_connected():
                    continue
                raise
        raise PlaywrightError("browser could not be restarted")

    async def close(self) -> None:
        if self._browser is not None and self._browser.is_connected():
            await self._browser.close()


# --- crawling ---------------------------------------------------------------------------------


@dataclass
class CrawlConfig:
    out: Path
    concurrency: int = 8
    links: int = 2
    screenshots_for: tuple[str, ...] = ()
    seed: int = 0
    browser_args: tuple[str, ...] = ()
    settle_ms: int = SETTLE_MS


@dataclass
class Rendered:
    page: Page
    links: list[tuple[str, str]]
    forms: list[dict]


class CrawlError(Exception):
    """A page that is not usable; `status` goes to the crawl log."""

    def __init__(self, status: str, reason: str = "") -> None:
        super().__init__(f"{status}: {reason}")
        self.status = status
        self.reason = reason


@dataclass
class CrawlSummary:
    sites: int = 0
    statuses: Counter = field(default_factory=Counter)


class Crawler:
    def __init__(self, cfg: CrawlConfig, browser: BrowserSession, robots_client: httpx.AsyncClient) -> None:
        self.cfg = cfg
        self.browser = browser
        self.robots_client = robots_client
        self.log = CrawlLog(cfg.out / LOG_NAME)
        self.index = TemplateIndex()
        self.summary = CrawlSummary()
        for page in iter_pages(cfg.out):
            self.index.add(text_signature(page))

    async def run(self, entries: list[str]) -> CrawlSummary:
        finished = self.log.finished_sites()
        queue: asyncio.Queue[str] = asyncio.Queue()
        for entry in entries:
            if registrable_domain(start_url(entry)) not in finished:
                queue.put_nowait(entry)
        workers = [asyncio.create_task(self._worker(queue)) for _ in range(self.cfg.concurrency)]
        await queue.join()
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        return self.summary

    async def _worker(self, queue: asyncio.Queue[str]) -> None:
        while True:
            entry = await queue.get()
            try:
                await self._crawl_site(entry)
            except Exception as e:  # one bad site must not stop the crawl; the reason is logged
                logger.exception("site %s failed", entry)
                self.log.write(start_url(entry), "site_error", reason=f"{type(e).__name__}: {e}")
            finally:
                queue.task_done()

    def _record(self, url: str, site: str, status: str, n_elements: int = 0, reason: str = "", via: str = "") -> None:
        self.summary.statuses[status] += 1
        self.log.write(url, status, n_elements, reason, site, via)

    async def _crawl_site(self, entry: str) -> None:
        home = start_url(entry)
        site = registrable_domain(home)
        parts = urlsplit(home)
        try:
            robots = await load_robots(self.robots_client, f"{parts.scheme}://{parts.netloc}")
        except RobotsError as e:
            self._record(home, site, "robots_error", reason=str(e))
        else:
            if robots.can_fetch(USER_AGENT, home):
                rendered = await self._visit(home, site, "home", is_home=True)
                for url, via in self._plan_visits(rendered, home, site, robots):
                    if robots.can_fetch(USER_AGENT, url):
                        await self._visit(url, site, via)
                    else:
                        self._record(url, site, "robots_disallowed", via=via)
            else:
                self._record(home, site, "robots_disallowed")
        self.log.write(home, "site_done", site=site)
        self.summary.sites += 1

    def _plan_visits(self, home: Rendered | None, home_url: str, site: str, robots: RobotFileParser) -> list[tuple[str, str]]:
        """(url, via) pairs after the homepage: one search-result page when the robots rules allow it, then weighted links."""
        if home is None or self.cfg.links <= 0:
            return []
        plan: list[tuple[str, str]] = []
        word = random.Random(f"{self.cfg.seed}:{site}:search").choice(SEARCH_WORDS)
        search = search_url(home.forms, site, word)
        if search and robots.can_fetch(USER_AGENT, search):
            plan.append((search, "search"))
        budget = self.cfg.links - len(plan)
        return plan + [(u, "link") for u in choose_links(home.links, home_url, site, budget, self.cfg.seed)]

    async def _visit(self, url: str, site: str, via: str, is_home: bool = False) -> Rendered | None:
        """Crawl one page; returns what was rendered (None on failure or when skipped).

        A page already on disk is skipped, except a homepage, which is needed to find links.
        """
        split = split_for(site)
        page_id = page_id_for(url)
        target = self.cfg.out / split / site / f"{page_id}.json.gz"
        if target.exists() and not is_home:
            self._record(url, site, "exists", via=via)
            return None
        shot = target.with_suffix("").with_suffix(".png") if split in self.cfg.screenshots_for else None
        try:
            rendered = await asyncio.wait_for(self._render(url, site, page_id, shot), PAGE_BUDGET_S)
        except CrawlError as e:
            self._record(url, site, e.status, reason=e.reason, via=via)
            return None
        except (PlaywrightError, asyncio.TimeoutError) as e:
            self._record(url, site, "nav_error", reason=f"{type(e).__name__}: {next(iter(str(e).splitlines()), '')}", via=via)
            return None
        page = rendered.page
        n = len(page.elements)
        if target.exists():
            self._record(url, site, "exists", n, via=via)
        elif n < MIN_ELEMENTS:
            self._drop(shot)
            self._record(url, site, "too_few_elements", n, via=via)
        elif self.index.is_duplicate(text_signature(page)):
            self._drop(shot)
            self._record(url, site, "duplicate_template", n, via=via)
        else:
            self.index.add(text_signature(page))
            write_page(page, self.cfg.out / split)
            self._record(url, site, "ok", n, via=via)
        return rendered

    @staticmethod
    def _drop(path: Path | None) -> None:
        if path is not None:
            path.unlink(missing_ok=True)

    async def _render(self, url: str, site: str, page_id: str, screenshot: Path | None) -> Rendered:
        context = await self.browser.new_context()
        try:
            page = await context.new_page()
            response = await page.goto(url, wait_until="load", timeout=NAV_TIMEOUT_MS)
            if response is not None and response.status >= 400:
                raise CrawlError("http_error", f"HTTP {response.status}")
            content_type = (response.headers.get("content-type", "") if response else "text/html")
            if "html" not in content_type:
                raise CrawlError("not_html", content_type)
            await page.wait_for_timeout(self.cfg.settle_ms)
            if registrable_domain(page.url) != site:
                raise CrawlError("cross_site_redirect", page.url)
            extracted = await extract_page(page, page_id=page_id, site=site)
            links = await page.eval_on_selector_all("a[href]", LINKS_JS)
            forms = await page.eval_on_selector_all("form", FORMS_JS)
            if screenshot is not None:
                screenshot.parent.mkdir(parents=True, exist_ok=True)
                await page.screenshot(path=str(screenshot), full_page=True)
            return Rendered(extracted, [(u, t) for u, t in links], forms)
        finally:
            try:
                await context.close()
            except PlaywrightError as e:
                logger.warning("closing context for %s failed: %s", url, e)


async def crawl(entries: list[str], cfg: CrawlConfig, robots_client: httpx.AsyncClient | None = None) -> CrawlSummary:
    """Crawl `entries` (domains or URLs) into `cfg.out`."""
    cfg.out.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as playwright:
        session = BrowserSession(playwright, cfg.browser_args)
        client = robots_client or httpx.AsyncClient(follow_redirects=True, timeout=ROBOTS_TIMEOUT_S)
        try:
            return await Crawler(cfg, session, client).run(entries)
        finally:
            if robots_client is None:
                await client.aclose()
            await session.close()


# --- stats ------------------------------------------------------------------------------------


def format_stats(out: str | Path) -> str:
    """Pages, sites, elements-per-page percentiles, role histogram and split sizes of a crawl directory."""
    lines: list[str] = []
    sizes: list[int] = []
    roles: Counter[str] = Counter()
    sites: set[str] = set()
    for split in SPLITS:
        split_pages = list(iter_pages(Path(out) / split)) if (Path(out) / split).is_dir() else []
        split_sites = {p.site for p in split_pages}
        lines.append(f"split {split:<5} pages={len(split_pages):<6} sites={len(split_sites)}")
        sizes += [len(p.elements) for p in split_pages]
        sites |= split_sites
        for p in split_pages:
            roles.update(e.role for e in p.elements)
    if not sizes:
        return f"no pages under {out}"
    p10, p50, p90 = np.percentile(sizes, [10, 50, 90])
    total = sum(roles.values())
    lines.insert(0, f"pages={len(sizes)} sites={len(sites)} elements/page p10={p10:.0f} median={p50:.0f} p90={p90:.0f}")
    lines.append("roles:")
    lines += [f"  {role:<10} {n:>8} {n / total:6.1%}" for role, n in roles.most_common()]
    return "\n".join(lines)


# --- CLI --------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Crawl sites into page files.")
    parser.add_argument("--sites", help="text file with one domain per line")
    parser.add_argument("--tranco", type=int, help="also use the top N domains of the Tranco list")
    parser.add_argument("--cache-dir", default="data/cache", help="where the Tranco list is cached")
    parser.add_argument("--out", default="data/pages")
    parser.add_argument("--limit", type=int, help="crawl at most this many sites")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--links", type=int, default=2, help="random same-site links per site")
    parser.add_argument("--screenshots-for", nargs="*", default=[], choices=SPLITS, help="splits to screenshot")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stats", action="store_true", help="print statistics of --out and exit")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if args.stats:
        print(format_stats(args.out))
        return 0
    entries = read_sites(args.sites) if args.sites else []
    if args.tranco:
        entries += load_tranco(args.tranco, args.cache_dir)
    if not entries:
        parser.error("give --sites and/or --tranco")
    entries = entries[: args.limit] if args.limit else entries
    cfg = CrawlConfig(Path(args.out), args.concurrency, args.links, tuple(args.screenshots_for), args.seed)
    summary = asyncio.run(crawl(entries, cfg))
    print(f"sites={summary.sites} " + " ".join(f"{k}={v}" for k, v in sorted(summary.statuses.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
