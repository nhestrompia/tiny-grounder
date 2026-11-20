"""Build the held-out site list for the human evaluation set (plan v2.1, step 3).

    uv run python -m tools.heldout_sites --candidates cands.txt --out data/heldout/sites.txt [--target 120]

A candidate is kept when its registrable domain is not a training domain (data/leakage/heldout_domains.txt, which
lists TRAINING domains despite its name), not a demo-task domain (demo/tasks.yaml), not on the blocklist, passes the
adult keyword check and Cloudflare's family DoH check, and answers HTTP 200 within 15 s. Dropped candidates and the
reason are printed. Only domain names go to the family resolver.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path

import httpx

from grounder.data.blocklist import keyword_blocked, read_blocklist
from grounder.data.crawl import USER_AGENT, read_sites, registrable_domain, start_url
from tools.block_sites import check_sites

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DOMAINS = ROOT / "data" / "leakage" / "heldout_domains.txt"
TASKS = ROOT / "demo" / "tasks.yaml"
LOAD_TIMEOUT_S = 15
_URL_RE = re.compile(r"https?://([^/\s\"']+)")


def read_domain_file(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def task_domains(tasks_yaml: str) -> set[str]:
    """Registrable domains of every URL in demo/tasks.yaml (text scan, no YAML dependency)."""
    return {registrable_domain(host) for host in _URL_RE.findall(tasks_yaml)}


def filter_candidates(
    candidates: list[str], train: set[str], demo: set[str], blocked: set[str],
) -> tuple[list[str], dict[str, str]]:
    """(kept entries in order, one per registrable domain; {entry: reason} for the dropped)."""
    kept: list[str] = []
    dropped: dict[str, str] = {}
    seen: set[str] = set()
    for entry in candidates:
        domain = registrable_domain(start_url(entry))
        if domain in seen:
            dropped[entry] = "duplicate domain"
        elif domain in train:
            dropped[entry] = "training domain"
        elif domain in demo:
            dropped[entry] = "demo task domain"
        elif domain in blocked or keyword_blocked(domain):
            dropped[entry] = "blocklist"
        else:
            seen.add(domain)
            kept.append(entry)
    return kept, dropped


async def loads_ok(entry: str, http: httpx.AsyncClient, gate: asyncio.Semaphore) -> str:
    """"" when the start page answers 200 (after redirects, still on the same registrable domain), else the reason."""
    url = start_url(entry)
    async with gate:
        try:
            r = await http.get(url, headers={"User-Agent": USER_AGENT})
        except httpx.HTTPError as e:
            return f"load failed: {type(e).__name__}"
    if r.status_code != 200:
        return f"HTTP {r.status_code}"
    if registrable_domain(str(r.url)) != registrable_domain(url):
        return f"redirects to {registrable_domain(str(r.url))}"
    return ""


async def check_loads(entries: list[str]) -> dict[str, str]:
    gate = asyncio.Semaphore(16)
    async with httpx.AsyncClient(timeout=LOAD_TIMEOUT_S, follow_redirects=True) as http:
        reasons = await asyncio.gather(*(loads_ok(e, http, gate) for e in entries))
    return {e: r for e, r in zip(entries, reasons) if r}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--target", type=int, default=120)
    args = parser.parse_args(argv)
    kept, dropped = filter_candidates(
        read_sites(args.candidates), read_domain_file(TRAIN_DOMAINS),
        task_domains(TASKS.read_text(encoding="utf-8")), read_blocklist())
    blocked = asyncio.run(check_sites([registrable_domain(start_url(e)) for e in kept]))
    for e in kept:
        if registrable_domain(start_url(e)) in blocked:
            dropped[e] = "family DNS block"
    kept = [e for e in kept if e not in dropped]
    for e, reason in asyncio.run(check_loads(kept)).items():
        dropped[e] = reason
    kept = [e for e in kept if e not in dropped][: args.target]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(f"{e}\n" for e in kept), encoding="utf-8")
    for e, reason in dropped.items():
        print(f"drop {e}: {reason}", file=sys.stderr)
    print(f"kept {len(kept)} of {len(kept) + len(dropped)} candidates -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
