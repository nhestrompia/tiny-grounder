"""Validation of teacher output (implementation plan Step 4).

`filter_examples` checks the examples of one page; `filter_all` groups examples by page.
Both return the kept examples plus a count of rejections by reason.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping

from grounder.features import content_tokens
from grounder.records import Element, Example, Page

MIN_WORDS = 2
MAX_WORDS = 20

# "element 3", "button #12", "id 7", "no. 4", "#3"
_REFERENCE_RE = re.compile(r"\b(?:element|elem|node|item|id|index|button|link|field|box|number|no)\.?\s*#?\d+\b|#\d+")
# "(120, 340)", "120,340", "120x340"
_COORDINATE_RE = re.compile(r"\(?\b\d+\s*[,x×]\s*\d+\)?")
_NUMBER_RE = re.compile(r"\b\d+\b")
_SPACE_RE = re.compile(r"\s+")


@dataclass
class FilterResult:
    kept: list[Example] = field(default_factory=list)
    rejected: Counter[str] = field(default_factory=Counter)

    def merge(self, other: "FilterResult") -> None:
        self.kept.extend(other.kept)
        self.rejected.update(other.rejected)


def label_counts(elements: Iterable[Element]) -> Counter[str]:
    """How many elements carry each (case-folded, non-empty) text."""
    return Counter(e.text.casefold() for e in elements if e.text)


def is_duplicated(element: Element, counts: Mapping[str, int]) -> bool:
    return bool(element.text) and counts[element.text.casefold()] > 1


def plain_is_unambiguous(element: Element, equiv: Iterable[int], elements: Iterable[Element]) -> bool:
    """A plain instruction naming a duplicated label is only valid if every duplicate is an equivalent target."""
    key = element.text.casefold()
    equiv_set = set(equiv)
    return all(e.id in equiv_set for e in elements if e.text.casefold() == key)


def _normalise_query(query: str) -> str:
    return _SPACE_RE.sub(" ", query.casefold()).strip()


def _check(example: Example, target: Element | None, page: Page, counts: Counter[str]) -> str | None:
    """The reason `example` is invalid, or None."""
    if target is None:
        return "unknown_target"
    n_words = len(example.query.split())
    if not MIN_WORDS <= n_words <= MAX_WORDS:
        return "length"
    if _REFERENCE_RE.search(example.query.lower()) or _COORDINATE_RE.search(example.query):
        return "id_or_coordinate"
    allowed = set(_NUMBER_RE.findall(f"{target.text} {target.context}"))
    if any(n not in allowed for n in _NUMBER_RE.findall(example.query)):
        return "id_or_coordinate"
    duplicated = is_duplicated(target, counts)
    if example.kind == "paraphrase" and content_tokens(example.query) & content_tokens(target.text):
        return "paraphrase_overlap"
    if example.kind == "context":
        if not duplicated:
            return "context_not_duplicated"
        if not content_tokens(example.query) & content_tokens(target.context):
            return "context_unused"
    if example.kind == "plain" and duplicated and not plain_is_unambiguous(target, example.equiv_ids, page.elements):
        return "ambiguous_plain"
    return None


def filter_examples(page: Page, examples: Iterable[Example]) -> FilterResult:
    """Validate the examples of one page and drop repeated queries.

    A query that appears several times with the same target is kept once; a query that
    maps to different targets is ambiguous and all its copies are dropped.
    """
    by_id = {e.id: e for e in page.elements}
    counts = label_counts(page.elements)
    result = FilterResult()
    valid: list[Example] = []
    for ex in examples:
        reason = _check(ex, by_id.get(ex.target_id) if ex.target_id is not None else None, page, counts)
        if reason:
            result.rejected[reason] += 1
        else:
            valid.append(ex)

    targets: dict[tuple[str, str], set[frozenset[int]]] = defaultdict(set)
    for ex in valid:
        targets[(ex.mode, _normalise_query(ex.query))].add(frozenset(ex.equiv_ids) | {ex.target_id})
    seen: set[tuple[str, str]] = set()
    for ex in valid:
        key = (ex.mode, _normalise_query(ex.query))
        if len(targets[key]) > 1:
            result.rejected["conflicting_query"] += 1
        elif key in seen:
            result.rejected["duplicate_query"] += 1
        else:
            seen.add(key)
            result.kept.append(ex)
    return result


def filter_all(pages: Mapping[str, Page], examples: Iterable[Example]) -> FilterResult:
    """Filter examples of many pages; examples whose page is missing are rejected."""
    grouped: dict[str, list[Example]] = defaultdict(list)
    result = FilterResult()
    for ex in examples:
        if ex.page_id in pages:
            grouped[ex.page_id].append(ex)
        else:
            result.rejected["unknown_page"] += 1
    for page_id, group in grouped.items():
        result.merge(filter_examples(pages[page_id], group))
    return result


def format_rejections(rejected: Mapping[str, int]) -> str:
    return ", ".join(f"{reason}={n}" for reason, n in sorted(rejected.items())) or "none"
