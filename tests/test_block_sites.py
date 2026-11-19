from __future__ import annotations

import asyncio
import gzip
from pathlib import Path

import httpx
import pytest

from grounder.data import blocklist
from tools import block_sites


def make_site(pages: Path, split: str, site: str, n: int = 2) -> None:
    d = pages / split / site
    d.mkdir(parents=True)
    for i in range(n):
        (d / f"{site}-{i}.json.gz").write_bytes(gzip.compress(b"{}"))


def test_keyword_block_skips_known_false_positives():
    assert blocklist.keyword_blocked("freeporno.xxx") and blocklist.keyword_blocked("hentaifox.com")
    assert not blocklist.keyword_blocked("dnsexit.com") and not blocklist.keyword_blocked("bbc.com")


def test_blocklist_round_trip_ignores_comments(tmp_path):
    path = tmp_path / "blocked.txt"
    blocklist.write_blocklist({"b.com", "a.com"}, path)
    assert path.read_text().splitlines()[1:] == ["a.com", "b.com"] and path.read_text().startswith("#")
    assert blocklist.read_blocklist(path) == {"a.com", "b.com"}
    assert blocklist.read_blocklist(tmp_path / "missing.txt") == set()


def test_family_resolver_answer_decides(monkeypatch):
    answers = {"bad.com": "0.0.0.0", "good.com": "93.184.216.34"}

    def handler(request: httpx.Request) -> httpx.Response:
        name = request.url.params["name"]
        return httpx.Response(200, json={"Answer": [{"data": answers[name]}]})

    real = httpx.AsyncClient
    monkeypatch.setattr(block_sites.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler)))
    assert asyncio.run(block_sites.check_sites(["bad.com", "good.com"])) == {"bad.com"}


def test_failed_lookup_raises(monkeypatch):
    real = httpx.AsyncClient
    monkeypatch.setattr(block_sites.httpx, "AsyncClient",
                        lambda **kw: real(transport=httpx.MockTransport(lambda r: httpx.Response(503))))
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(block_sites.check_sites(["x.com"]))


def test_main_moves_blocked_sites_and_updates_list(tmp_path, monkeypatch):
    pages = tmp_path / "pages"
    make_site(pages, "train", "ok.com")
    make_site(pages, "train", "sexyshop.com", 3)
    make_site(pages, "test", "dnsbad.com")
    listed = tmp_path / "blocked.txt"
    monkeypatch.setattr(block_sites, "read_blocklist", lambda: blocklist.read_blocklist(listed))
    monkeypatch.setattr(block_sites, "write_blocklist", lambda s: blocklist.write_blocklist(s, listed))

    async def fake_check(sites):
        return {s for s in sites if s == "dnsbad.com"}

    monkeypatch.setattr(block_sites, "check_sites", fake_check)
    assert block_sites.main(["--pages", str(pages)]) == 0
    assert blocklist.read_blocklist(listed) == {"sexyshop.com", "dnsbad.com"}
    assert sorted(p.name for p in (pages / "train").iterdir()) == ["ok.com"]
    assert len(list((tmp_path / "pages_blocked" / "train" / "sexyshop.com").glob("*.json.gz"))) == 3
    assert (tmp_path / "pages_blocked" / "test" / "dnsbad.com").is_dir()
