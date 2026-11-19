"""Sites kept out of the dataset (adult, malware, piracy); maintained by tools/block_sites.py."""

from __future__ import annotations

import re
from pathlib import Path

BLOCKLIST = Path(__file__).with_name("blocked_sites.txt")
ADULT_RE = re.compile(
    r"porn|xxx|sex|jizz|fap|nude|naked|erotic|literotica|escort|hentai|cam4|chaturbate|xvideo|xhamster|onlyfans"
    r"|bodgirl|milf|nsfw|redtube|brazzers|stripchat|camsoda", re.I)
NOT_ADULT = frozenset({"dnsexit.com", "jeuxvideo.com"})  # keyword false positives


def read_blocklist(path: Path = BLOCKLIST) -> set[str]:
    if not path.exists():
        return set()
    lines = path.read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip() and not line.startswith("#")}


def write_blocklist(sites: set[str], path: Path = BLOCKLIST) -> None:
    header = "# Sites kept out of the dataset (tools/block_sites.py): Cloudflare family DNS block or adult keyword.\n"
    path.write_text(header + "".join(f"{s}\n" for s in sorted(sites)), encoding="utf-8")


def keyword_blocked(site: str) -> bool:
    return site not in NOT_ADULT and bool(ADULT_RE.search(site))
