"""Web extractor: runs dom.js in a Playwright page and returns a validated records.Page."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from grounder.records import Page, validate_element

DOM_JS = (Path(__file__).with_name("dom.js")).read_text(encoding="utf-8")


def page_from_raw(raw: dict[str, Any], url: str, page_id: str, site: str) -> Page:
    """Validate the raw dict returned by dom.js into a records.Page."""
    return Page(
        page_id=page_id,
        url=url,
        site=site,
        title=raw.get("title", ""),
        lang=raw.get("lang", ""),
        viewport=[int(v) for v in raw["viewport"]],
        doc_size=[int(v) for v in raw["doc_size"]],
        elements=[validate_element(e) for e in raw["elements"]],
    )


def _identity(url: str, page_id: str | None, site: str | None) -> tuple[str, str]:
    host = urlparse(url).hostname or ""
    return page_id or url, site if site is not None else host


async def extract_page(page, page_id: str | None = None, site: str | None = None) -> Page:
    """Extract the current state of an async Playwright page."""
    raw = await page.evaluate(DOM_JS)
    pid, site_name = _identity(page.url, page_id, site)
    return page_from_raw(raw, page.url, pid, site_name)


def extract_page_sync(page, page_id: str | None = None, site: str | None = None) -> Page:
    """Extract the current state of a sync Playwright page."""
    raw = page.evaluate(DOM_JS)
    pid, site_name = _identity(page.url, page_id, site)
    return page_from_raw(raw, page.url, pid, site_name)


def format_table(page: Page) -> str:
    rows = [f"{page.url}  title={page.title!r}  viewport={page.viewport}  doc={page.doc_size}  "
            f"elements={len(page.elements)}"]
    for e in page.elements:
        box = " ".join(f"{v:.2f}" for v in e.box)
        flags = ("" if e.in_viewport else " below") + ("" if e.enabled else " disabled")
        rows.append(f"{e.id:4d} {e.role:<9} {e.text[:40]!r:<44} {e.context[:60]!r:<64} [{box}]{flags}")
    return "\n".join(rows)


async def _run(target: str, out: str | None, screenshot: str | None) -> Page:
    from playwright.async_api import async_playwright

    url = target if "://" in target else Path(target).resolve().as_uri()
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1280, "height": 800})
        await page.goto(url, wait_until="load", timeout=30000)
        await page.wait_for_timeout(500)
        result = await extract_page(page)
        if screenshot:
            await page.screenshot(path=screenshot, full_page=True)
        await browser.close()
    if out:
        Path(out).write_text(json.dumps(result.to_dict(), indent=1), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract UI elements from a page and print them.")
    parser.add_argument("target", help="URL or local HTML file")
    parser.add_argument("--out", help="write the Page JSON here")
    parser.add_argument("--screenshot", help="write a full-page PNG here")
    args = parser.parse_args(argv)
    print(format_table(asyncio.run(_run(args.target, args.out, args.screenshot))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
