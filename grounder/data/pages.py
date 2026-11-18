"""Reading and writing page files (`pages/{site}/{page_id}.json.gz`) and example JSONL files.

Standard library plus `grounder.records` only, so the tokenizer CLI and the runtime
tooling can use it without torch.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterable, Iterator

from grounder.records import Example, Page


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
