"""Adapters that turn a live UI into a `records.Page`. The extraction code itself lives in grounder.extract."""

from __future__ import annotations

from grounder.records import Page


def from_playwright(page) -> Page:
    """Extract a sync Playwright page (reuses grounder.extract.web, the extractor the helper uses)."""
    from grounder.extract.web import extract_page_sync

    return extract_page_sync(page)


async def afrom_playwright(page) -> Page:
    """Extract an async Playwright page."""
    from grounder.extract.web import extract_page

    return await extract_page(page)


def from_macos_ax(app: str | None = None, pid: int | None = None, timeout: float = 30.0) -> Page:
    """Accessibility tree of an app's front window (the frontmost app when neither `app` nor `pid` is given).

    macOS only; needs the Swift `axdump` binary (`swift build -c release` in grounder/extract/macos) and the
    Accessibility permission. Raises grounder.extract.ax.AxdumpError with the reason otherwise.
    """
    from grounder.extract import ax

    if app is None and pid is None:
        return ax.page_from_dump(ax.run_axdump(["--frontmost"], timeout=timeout), "ax-frontmost", "frontmost")
    return ax.dump_page(app=app, pid=pid, timeout=timeout)
