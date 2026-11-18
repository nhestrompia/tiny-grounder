"""Crawler tests. Pages are served by a local HTTP server; Chromium maps every host name to it."""

from __future__ import annotations

import asyncio
import json
import shutil
import threading
import zipfile
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from grounder.data.crawl import (
    LOG_NAME, SPLITS, CrawlConfig, RobotsError, TemplateIndex, choose_links, crawl, format_stats,
    is_crawlable_link, jaccard, load_robots, load_tranco, read_sites, registrable_domain, split_for,
    text_signature,
)
from grounder.data.pages import iter_pages
from grounder.records import Element, Page

FIXTURES = Path(__file__).parent / "fixtures"
SITES = {"shop.alpha-store.com": "alpha-store.com", "docs.beta-news.org": "beta-news.org", "gamma-wiki.net": "gamma-wiki.net"}
HOST_RESOLVER = ("--host-resolver-rules=MAP * 127.0.0.1",)
ALL_LINKS = 7  # every same-site link of the fixture homepages


class FixtureHandler(BaseHTTPRequestHandler):
    """Serves tests/fixtures/crawl/<Host>/<path> and records every request path per host."""

    hits: list[tuple[str, str]]

    def do_GET(self) -> None:
        host = (self.headers.get("Host") or "").split(":")[0]
        self.hits.append((host, self.path))
        name = "index.html" if self.path == "/" else self.path.lstrip("/")
        file = FIXTURES / "crawl" / host / name
        if not file.is_file():
            self.send_error(404)
            return
        body = file.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        return


@pytest.fixture(scope="module")
def server():
    hits: list[tuple[str, str]] = []
    handler = type("Handler", (FixtureHandler,), {"hits": hits})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd.server_address[1], hits
    httpd.shutdown()
    thread.join()


def robots_client(disallowed: str = "gamma-wiki.net") -> httpx.AsyncClient:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.host == disallowed:
            return httpx.Response(200, text="User-agent: *\nDisallow: /\n")
        return httpx.Response(404)

    return httpx.AsyncClient(transport=httpx.MockTransport(handle))


def run_crawl(out: Path, port: int, screenshots_for: tuple[str, ...] = ()):
    entries = [f"http://{host}:{port}/" for host in SITES]
    cfg = CrawlConfig(out, concurrency=3, links=ALL_LINKS, screenshots_for=screenshots_for,
                      browser_args=HOST_RESOLVER, settle_ms=50)
    return asyncio.run(crawl(entries, cfg, robots_client()))


def log_lines(out: Path) -> list[dict]:
    return [json.loads(line) for line in (out / LOG_NAME).read_text().splitlines()]


@pytest.fixture(scope="module")
def crawled(server, tmp_path_factory):
    port, hits = server
    out = tmp_path_factory.mktemp("crawl") / "pages"
    hits.clear()
    summary = run_crawl(out, port, screenshots_for=(split_for("alpha-store.com"),))
    return out, summary, list(hits)


# --- crawl end to end ---------------------------------------------------------------------------


def test_crawl_statuses(crawled):
    _, summary, _ = crawled
    assert summary.sites == 3
    assert summary.statuses["ok"] == 6  # 4 alpha pages, beta index + archive
    assert summary.statuses["duplicate_template"] == 1  # beta copy.html
    assert summary.statuses["too_few_elements"] == 1  # beta tiny.html
    assert summary.statuses["robots_disallowed"] == 1  # gamma
    assert summary.statuses["http_error"] == 4  # beta links without a fixture file


def test_pages_are_stored_by_split_and_site(crawled):
    out, _, _ = crawled
    files = sorted(out.glob("*/*/*.json.gz"))
    assert len(files) == 6
    for f in files:
        assert f.parts[-3] == split_for(f.parts[-2])
    assert {p.site for p in iter_pages(out)} == {"alpha-store.com", "beta-news.org"}
    assert all(len(p.elements) >= 5 for p in iter_pages(out))


def test_only_same_site_crawlable_links_are_followed(crawled):
    _, _, hits = crawled
    alpha_paths = {path for host, path in hits if host == "shop.alpha-store.com"}
    assert alpha_paths == {"/", "/about.html", "/products.html", "/offers.html"}  # no /login, /cart, pdf
    assert not any(host == "gamma-wiki.net" for host, _ in hits)  # robots.txt disallows the whole site


def test_log_has_one_line_per_attempt(crawled):
    out, _, _ = crawled
    lines = log_lines(out)
    assert Counter(r["status"] for r in lines)["site_done"] == 3
    assert all({"url", "status", "n_elements", "reason"} <= r.keys() for r in lines)
    assert any(r["status"] == "robots_disallowed" and r["site"] == "gamma-wiki.net" for r in lines)


def test_screenshots_only_for_requested_split(crawled):
    out, _, _ = crawled
    shot_split = split_for("alpha-store.com")
    for page_file in out.glob("*/*/*.json.gz"):
        png = page_file.with_name(page_file.name.removesuffix(".json.gz") + ".png")
        assert png.exists() == (page_file.parts[-3] == shot_split)


def test_stats_report(crawled):
    out, _, _ = crawled
    report = format_stats(out)
    assert "pages=6 sites=2" in report
    assert "link" in report and "median=" in report
    for split in SPLITS:
        assert f"split {split}" in report


def test_resume_skips_finished_sites(crawled, server, tmp_path):
    port, hits = server
    out = tmp_path / "pages"
    shutil.copytree(crawled[0], out)
    hits.clear()
    summary = run_crawl(out, port)
    assert hits == [] and summary.sites == 0


def test_resume_does_not_refetch_pages_on_disk(crawled, server, tmp_path):
    port, hits = server
    out = tmp_path / "pages"
    shutil.copytree(crawled[0], out)
    (out / LOG_NAME).unlink()  # as if the run was killed before any site finished
    hits.clear()
    run_crawl(out, port)
    alpha = {path for host, path in hits if host == "shop.alpha-store.com"}
    assert alpha == {"/"}  # the homepage is refetched for its links; the other pages are on disk
    assert len(list(iter_pages(out))) == 6


# --- units -------------------------------------------------------------------------------------


@pytest.mark.parametrize("url,site", [("https://www.example.co.uk/a", "example.co.uk"), ("https://a.b.github.io/x", "github.io"),
                                      ("http://127.0.0.1:8000/", "127.0.0.1"), ("example.com", "example.com")])
def test_registrable_domain(url, site):
    assert registrable_domain(url) == site


def test_split_is_deterministic_and_shared_by_subdomains():
    assert split_for("example.com") == split_for("example.com")
    assert registrable_domain("https://shop.example.com") == registrable_domain("https://www.example.com")
    counts = Counter(split_for(f"site{i}.com") for i in range(4000))
    assert 0.87 < counts["train"] / 4000 < 0.93
    assert counts["dev"] > 100 and counts["test"] > 100


@pytest.mark.parametrize("url,ok", [
    ("https://example.com/about", True),
    ("http://blog.example.com/post/1?x=1", True),
    ("https://other.org/about", False),
    ("mailto:a@example.com", False),
    ("javascript:void(0)", False),
    ("https://example.com/login", False),
    ("https://example.com/user/sign-in", False),
    ("https://example.com/logout", False),
    ("https://example.com/cart/", False),
    ("https://example.com/checkout/step-1", False),
    ("https://example.com/files/report.pdf", False),
    ("https://example.com/setup.exe", False),
    ("https://example.com/accounting", True),
    ("https://example.com/carts-and-bikes", True),
])
def test_is_crawlable_link(url, ok):
    assert is_crawlable_link(url, "example.com") == ok


def test_choose_links_is_seeded_distinct_and_skips_home():
    links = [f"https://example.com/p{i}" for i in range(20)] + ["https://example.com/", "https://example.com/p1#top"]
    first = choose_links(links, "https://example.com/", "example.com", 3, seed=1)
    assert first == choose_links(links, "https://example.com/", "example.com", 3, seed=1)
    assert first != choose_links(links, "https://example.com/", "example.com", 3, seed=2)
    assert len(set(first)) == 3 and "https://example.com/" not in first
    assert sorted(choose_links(links[:2], "https://example.com/", "example.com", 5, seed=1)) == ["https://example.com/p0", "https://example.com/p1"]


def robots_for(status: int, text: str = "") -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(status, text=text)))


def test_robots_rules():
    async def run():
        allow = await load_robots(robots_for(404), "https://a.com")
        block = await load_robots(robots_for(200, "User-agent: tiny-grounder-research\nDisallow: /private\n"), "https://a.com")
        everyone = await load_robots(robots_for(200, "User-agent: *\nDisallow: /\n"), "https://a.com")
        return allow, block, everyone

    allow, block, everyone = asyncio.run(run())
    assert allow.can_fetch("tiny-grounder-research", "https://a.com/anything")
    assert block.can_fetch("tiny-grounder-research", "https://a.com/public")
    assert not block.can_fetch("tiny-grounder-research", "https://a.com/private/x")
    assert not everyone.can_fetch("tiny-grounder-research", "https://a.com/")


@pytest.mark.parametrize("status", [403, 500])
def test_robots_error_status_raises(status):
    with pytest.raises(RobotsError, match=str(status)):
        asyncio.run(load_robots(robots_for(status), "https://a.com"))


def test_robots_network_error_raises():
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(boom))
    with pytest.raises(RobotsError, match="ConnectError"):
        asyncio.run(load_robots(client, "https://a.com"))


def make_page(texts: list[str]) -> Page:
    elements = [Element(id=i, role="link", text=t) for i, t in enumerate(texts)]
    return Page("p", "https://a.com", "a.com", "t", [1280, 800], [1280, 800], elements)


def test_signature_counts_repeated_texts():
    sig = text_signature(make_page(["Home", "home", "Cart", ""]))
    assert sig == {("home", 0), ("home", 1), ("cart", 0)}
    assert jaccard(sig, sig) == 1.0


def test_template_index_flags_near_duplicates():
    base = [f"item {i}" for i in range(20)]
    index = TemplateIndex()
    index.add(text_signature(make_page(base)))
    assert index.is_duplicate(text_signature(make_page(base[::-1])))  # order is irrelevant
    assert index.is_duplicate(text_signature(make_page(base[:19] + ["other"])))  # 19/21 = 0.905
    assert not index.is_duplicate(text_signature(make_page(base[:18] + ["x", "y"])))  # 18/22 = 0.82
    assert not index.is_duplicate(text_signature(make_page([f"news {i}" for i in range(20)])))
    assert not index.is_duplicate(text_signature(make_page([""])))


def test_read_sites_skips_comments_and_blanks(tmp_path):
    path = tmp_path / "sites.txt"
    path.write_text("# header\nexample.com\n\nexample.org  # trailing\n")
    assert read_sites(path) == ["example.com", "example.org"]


def test_load_tranco_uses_cached_zip(tmp_path):
    with zipfile.ZipFile(tmp_path / "tranco-top-1m.csv.zip", "w") as z:
        z.writestr("top-1m.csv", "1,google.com\n2,facebook.com\n3,example.org\n")
    no_network = httpx.Client(transport=httpx.MockTransport(lambda r: pytest.fail("downloaded")))
    assert load_tranco(2, tmp_path, no_network) == ["google.com", "facebook.com"]
