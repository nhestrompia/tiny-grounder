"""Reading and writing page files (`pages/{site}/{page_id}.json.gz`) and example JSONL files.

Standard library plus `grounder.records` only, so the tokenizer CLI and the runtime
tooling can use it without torch.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from grounder.records import Example, Page


def slugify(name: str) -> str:
    """A stable lowercase file-system name for an app ("Visual Studio Code" -> "visual-studio-code")."""
    return re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-") or "app"


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


def page_id_for(site: str, page: Page) -> str:
    """Deterministic and unique per distinct window content: site plus a hash of the text signature."""
    digest = hashlib.sha1(repr(sorted(text_signature(page))).encode("utf-8")).hexdigest()[:10]
    return f"{site}-{digest}"


def page_paths(pages_dir: str | Path) -> list[Path]:
    return sorted(Path(pages_dir).glob("**/*.json.gz"))


def load_page(path: str | Path) -> Page:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return Page.from_dict(json.load(f))


def iter_pages(pages_dir: str | Path) -> Iterator[Page]:
    for path in page_paths(pages_dir):
        yield load_page(path)


def load_pages(pages_dir: str | Path) -> dict[str, Page]:
    return {p.page_id: p for p in iter_pages(pages_dir)}


def load_pages_from(pages_dirs: Sequence[str | Path]) -> dict[str, Page]:
    """The pages of several directories by page_id; the same page_id with different content is an error."""
    pages: dict[str, Page] = {}
    for pages_dir in pages_dirs:
        for page in iter_pages(pages_dir):
            if pages.setdefault(page.page_id, page) != page:
                raise ValueError(f"page_id {page.page_id!r} appears in {pages_dir} with different content")
    return pages


def write_page(page: Page, pages_dir: str | Path) -> Path:
    path = Path(pages_dir) / page.site / f"{page.page_id}.json.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(page.to_dict(), f)
    return path


def load_examples(path: str | Path) -> list[Example]:
    with open(path, encoding="utf-8") as f:
        return [Example.from_dict(json.loads(line)) for line in f if line.strip()]


def write_examples(examples: Iterable[Example], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict()) + "\n")
