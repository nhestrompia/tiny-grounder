"""Element record schema shared by every extractor, the data pipeline and the runtime.

This module is the contract between components. It must stay dependency-free
(standard library only) because the runtime imports it.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Literal

ROLES: tuple[str, ...] = (
    "button", "link", "textbox", "searchbox", "checkbox", "radio", "switch",
    "combobox", "option", "tab", "menuitem", "slider", "image", "text",
    "other", "unknown",
)
ROLE_TO_ID: dict[str, int] = {r: i for i, r in enumerate(ROLES)}

SOURCES: tuple[str, ...] = ("dom", "ax", "ocr", "det")

MAX_TEXT_CHARS = 80
MAX_CONTEXT_CHARS = 120
MAX_HREF_CHARS = 200
MAX_LANG_CHARS = 35

Mode = Literal["do", "goal"]
ExampleKind = Literal["plain", "paraphrase", "context", "goal", "null"]


@dataclass
class Element:
    """One candidate UI element.

    box:      [x/vw, y/vh, w/vw, h/vh], viewport-relative; y may exceed 1 below the fold.
    abs_box:  [x, y, w, h] in CSS pixels of the document (or screen points for `ax`).
    href:     link target, origin stripped (path + query), never a model input.
    """

    id: int
    role: str
    text: str
    context: str = ""
    tag: str = ""
    box: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    abs_box: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    in_viewport: bool = True
    enabled: bool = True
    conf: float = 1.0
    source: str = "dom"
    href: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Element":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


def drop_surrogates(s: str) -> str:
    """Remove lone UTF-16 surrogates (JavaScript `slice` can cut an emoji in half); they cannot be encoded as UTF-8."""
    return s.encode("utf-8", "ignore").decode("utf-8")


def clean_text(s: str | None, limit: int) -> str:
    """NFKC-normalise, collapse whitespace, strip, cut to `limit` chars."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", drop_surrogates(s))
    s = re.sub(r"\s+", " ", s).strip()
    return s[:limit]


def normalize_role(role: str | None) -> str:
    return role if role in ROLE_TO_ID else "unknown"


def validate_element(d: dict[str, Any]) -> Element:
    """Coerce a raw dict (from any extractor) into a clean Element. Raises ValueError if unusable."""
    if "id" not in d:
        raise ValueError("element without id")
    el = Element.from_dict(d)
    el.role = normalize_role(el.role)
    el.text = clean_text(el.text, MAX_TEXT_CHARS)
    el.context = clean_text(el.context, MAX_CONTEXT_CHARS)
    el.href = (el.href or "")[:MAX_HREF_CHARS]
    if el.source not in SOURCES:
        raise ValueError(f"unknown source {el.source!r}")
    if len(el.box) != 4 or len(el.abs_box) != 4:
        raise ValueError("box and abs_box must have 4 values")
    el.conf = float(min(max(el.conf, 0.0), 1.0))
    return el


def equiv_key(el: Element) -> tuple[str, str, str]:
    """Elements with the same key are interchangeable targets (same role, text and link target)."""
    return (el.role, el.text.casefold(), el.href)


def equiv_ids(elements: Iterable[Element], target_id: int) -> list[int]:
    els = list(elements)
    target = next(e for e in els if e.id == target_id)
    key = equiv_key(target)
    return [e.id for e in els if equiv_key(e) == key]


def format_query(mode: Mode, query: str, history: list[str] | None = None) -> str:
    """The only place a model query string is built. Special tokens are defined in tokenizer.py."""
    q = clean_text(query, 300)
    if mode == "do":
        return f"[DO] {q}"
    if mode == "goal":
        done = " ; ".join(clean_text(h, 100) for h in (history or [])[-3:])
        return f"[GOAL] {q} [DONE] {done}".rstrip()
    raise ValueError(f"unknown mode {mode!r}")


@dataclass
class Example:
    """One training / evaluation example (one line of a JSONL file)."""

    page_id: str
    kind: str
    mode: str
    query: str
    history: list[str] = field(default_factory=list)
    target_id: int | None = None
    equiv_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Example":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Page:
    """One page file: pages/{site}/{page_id}.json.gz

    lang: the `<html lang>` attribute as crawled; empty when absent or unknown (old files, native extractors).
    """

    page_id: str
    url: str
    site: str
    title: str
    viewport: list[int]
    doc_size: list[int]
    elements: list[Element]
    lang: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Page":
        return cls(
            page_id=d["page_id"], url=d["url"], site=d["site"], title=drop_surrogates(d.get("title", "")),
            viewport=list(d["viewport"]), doc_size=list(d["doc_size"]),
            elements=[validate_element(e) for e in d["elements"]], lang=clean_text(d.get("lang"), MAX_LANG_CHARS),
        )
