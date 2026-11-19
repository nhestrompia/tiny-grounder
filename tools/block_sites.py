"""Keep adult, malware and piracy sites out of the dataset.

    uv run python -m tools.block_sites --pages data/pages            # check every site, update the list, move pages
    uv run python -m tools.block_sites --pages data/pages --no-check # only move pages of sites already listed

A site is blocked when Cloudflare's family resolver (1.1.1.3, DNS over HTTPS) answers 0.0.0.0 for it, or when its
domain matches an adult keyword (minus a few known false positives). Only domain names are sent. The list is kept in
grounder/data/blocked_sites.txt (committed, so the dataset is reproducible) and the crawler skips listed sites.
Blocked pages are moved, not deleted, to `{pages}_blocked/{split}/{site}`.
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
from pathlib import Path

import httpx

from grounder.data.blocklist import keyword_blocked, read_blocklist, write_blocklist
from grounder.data.crawl import SPLITS

FAMILY_DOH = "https://family.cloudflare-dns.com/dns-query"
BLOCKED_ANSWER = "0.0.0.0"
CONCURRENCY = 32


async def family_blocked(site: str, http: httpx.AsyncClient, gate: asyncio.Semaphore) -> bool:
    """True when the family resolver sinkholes `site`. Raises on a failed lookup (retry the run)."""
    async with gate:
        r = await http.get(FAMILY_DOH, params={"name": site, "type": "A"}, headers={"accept": "application/dns-json"})
    r.raise_for_status()
    return any(a.get("data") == BLOCKED_ANSWER for a in r.json().get("Answer", []))


async def check_sites(sites: list[str]) -> set[str]:
    gate = asyncio.Semaphore(CONCURRENCY)
    async with httpx.AsyncClient(timeout=10) as http:
        flags = await asyncio.gather(*(family_blocked(s, http, gate) for s in sites))
    return {s for s, blocked in zip(sites, flags) if blocked}


def site_dirs(pages: Path) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {}
    for split in SPLITS:
        for d in sorted((pages / split).glob("*")):
            if d.is_dir():
                found.setdefault(d.name, []).append(d)
    return found


def move_blocked(pages: Path, dirs: dict[str, list[Path]], blocked: set[str]) -> int:
    """Move the page dirs of blocked sites to `{pages}_blocked/`; returns the number of pages moved."""
    moved = 0
    target_root = pages.with_name(pages.name + "_blocked")
    for site in sorted(blocked & dirs.keys()):
        for d in dirs[site]:
            target = target_root / d.parent.name / site
            target.parent.mkdir(parents=True, exist_ok=True)
            moved += sum(1 for _ in d.glob("*.json.gz"))
            shutil.move(str(d), str(target))
    return moved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pages", type=Path, default=Path("data/pages"))
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=True,
                        help="query the family resolver for sites not yet listed")
    args = parser.parse_args(argv)
    dirs = site_dirs(args.pages)
    blocked = read_blocklist()
    new = {s for s in dirs if s not in blocked and keyword_blocked(s)}
    if args.check:
        new |= asyncio.run(check_sites(sorted(s for s in dirs if s not in blocked)))
    if new:
        write_blocklist(blocked | new)
    moved = move_blocked(args.pages, dirs, blocked | new)
    print(f"sites={len(dirs)} newly_blocked={len(new)} listed={len(blocked | new)} pages_moved={moved}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
