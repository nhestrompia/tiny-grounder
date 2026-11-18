"""Rule-based teacher: `plain` and `context` instructions from templates (plan-review item 7).

Also owns the element sampler shared with the LLM teacher (`sample_elements`).
Instructions reuse the element's own words by construction, so this data must never be the
only data a model is compared on; `paraphrase` and `goal` come from the LLM teacher.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from typing import Mapping, Sequence

from grounder.data.filters import is_duplicated, label_counts, plain_is_unambiguous
from grounder.features import content_tokens
from grounder.records import Element, Example, Page, equiv_ids

SAMPLE_SIZE = 20
MAX_PLAIN_WORDS = 8
MAX_CONTEXT_WORDS = 6
CONTEXT_SEPARATOR = " · "
MAX_DUPLICATES_PER_PAGE = 3
MAX_ICONS_PER_PAGE = 2
MAX_ROLE_SHARE = 0.5
ICON_MAX_SIDE = 48
FOOTER_FRACTION = 0.9
FOOTER_MARKERS = ("©", "privacy", "terms of")

PLAIN_TEMPLATES: dict[str, tuple[str, ...]] = {
    "button": ("click {t}", "press {t}", "hit the {t} button", "click the {t} button", "tap {t}"),
    "link": ("open {t}", "go to {t}", "follow the {t} link", "click on {t}", "click the {t} link"),
    "textbox": ("type in the {t} field", "click the {t} field", "focus the {t} input", "enter text in {t}"),
    "searchbox": ("type in the {t} box", "click the {t} search field", "use the {t} search"),
    "checkbox": ("check {t}", "select {t}", "toggle {t}", "tick the {t} box"),
    "radio": ("select {t}", "choose {t}", "pick the {t} option"),
    "switch": ("toggle {t}", "turn on {t}", "switch {t} on"),
    "combobox": ("choose from {t}", "open the {t} dropdown", "pick an option in {t}"),
    "option": ("pick {t}", "choose the {t} option", "select {t}"),
    "tab": ("switch to the {t} tab", "open the {t} tab", "show the {t} tab"),
    "menuitem": ("choose {t} from the menu", "click the {t} menu item", "select {t}"),
    "slider": ("adjust the {t} slider", "move the {t} slider", "drag the {t} slider"),
    "image": ("click the {t} image", "open the {t} picture", "click on {t}"),
}
DEFAULT_TEMPLATES = ("click {t}", "click on {t}", "select {t}")
SOLE_SEARCHBOX_TEMPLATE = "search for something"
# Nouns a template adds after the label; a label that already ends in one skips such templates.
_ROLE_NOUNS = ("button", "link", "field", "tab", "box", "input", "image", "menu", "slider", "dropdown", "option")
_NOUN_RE = re.compile(r"\b(?:" + "|".join(_ROLE_NOUNS) + r")\b", re.I)
_EDGE_PUNCT = " .:;,»«›‹>…|-–—•*"

CONTEXT_VERBS: dict[str, tuple[str, ...]] = {
    "button": ("click", "press"),
    "link": ("open", "click"),
}
DEFAULT_CONTEXT_VERBS = ("click",)
CONTEXT_TEMPLATES = ("{v} {t} for {c}", "{t} next to {c}", "{v} {t} in the {c} section")


def clean_label(text: str) -> str:
    return text.strip(_EDGE_PUNCT)


def is_icon_like(el: Element) -> bool:
    """A small, roughly square labelled element: its text most likely came from an aria-label."""
    w, h = el.abs_box[2], el.abs_box[3]
    return bool(el.text) and 0 < w <= ICON_MAX_SIDE and 0 < h <= ICON_MAX_SIDE and 0.5 <= w / h <= 2


def _weight(el: Element, page: Page, role_counts: Mapping[str, int]) -> float:
    weight = role_counts[el.role] ** -0.5  # tempered inverse role frequency: rare roles are favoured
    if el.in_viewport:
        weight *= 3.0
    if not el.text:
        weight *= 0.2
    doc_h = page.doc_size[1] if len(page.doc_size) > 1 else 0
    if doc_h and el.abs_box[1] / doc_h > FOOTER_FRACTION:
        weight *= 0.2
    if any(m in el.context.casefold() for m in FOOTER_MARKERS):
        weight *= 0.2
    return weight


def _weighted_order(items: Sequence[Element], weights: Sequence[float], rng: random.Random) -> list[Element]:
    """All items in weighted random order (Efraimidis-Spirakis keys)."""
    keyed = sorted(((rng.random() ** (1.0 / w), i) for i, w in enumerate(weights)), reverse=True)
    return [items[i] for _, i in keyed]


def sample_elements(page: Page, rng: random.Random, n: int = SAMPLE_SIZE) -> list[Element]:
    """Pick up to `n` elements: some duplicated labels and icon-like ones guaranteed, the rest weighted.

    Favours in-viewport elements and elements with text, down-weights footer-like elements, and
    balances roles: weights scale with (role frequency on the page)^-0.5 and, when the page has
    other roles to fill it, no role takes more than `MAX_ROLE_SHARE` of the sample.
    """
    counts = label_counts(page.elements)
    role_counts = Counter(e.role for e in page.elements)
    cap = n if len(role_counts) == 1 else max(1, int(MAX_ROLE_SHARE * n))
    chosen: dict[int, Element] = {}

    def take(pool: list[Element], k: int, capped: bool) -> None:
        free = [e for e in pool if e.id not in chosen]
        ordered = _weighted_order(free, [_weight(e, page, role_counts) for e in free], rng)
        added = 0
        for e in ordered:
            if added == k:
                break
            if not capped or sum(c.role == e.role for c in chosen.values()) < cap:
                chosen[e.id] = e
                added += 1

    take([e for e in page.elements if is_duplicated(e, counts)], min(MAX_DUPLICATES_PER_PAGE, n), capped=False)
    take([e for e in page.elements if is_icon_like(e)], min(MAX_ICONS_PER_PAGE, n - len(chosen)), capped=False)
    take(list(page.elements), n - len(chosen), capped=True)
    take(list(page.elements), n - len(chosen), capped=False)  # the cap only applies while other roles remain
    return [chosen[e.id] for e in page.elements if e.id in chosen]


def context_phrase(context: str) -> str:
    """A short salient piece of the context: its first segment, at most 6 words."""
    first = context.split(CONTEXT_SEPARATOR)[0]
    return clean_label(" ".join(first.split()[:MAX_CONTEXT_WORDS]))


def _usable_templates(templates: Sequence[str], label: str) -> list[str]:
    present = {m.lower() for m in _NOUN_RE.findall(label)}
    usable = [t for t in templates if not any(re.search(rf"\b{n}\b", t) for n in present)]
    return usable or list(DEFAULT_TEMPLATES)


def plain_query(el: Element, page: Page, rng: random.Random) -> str | None:
    """A template instruction naming the element's label, or None when it has no usable label."""
    label = clean_label(el.text)
    if not label or len(label.split()) > MAX_PLAIN_WORDS:
        return None
    if el.role == "searchbox" and sum(e.role == "searchbox" for e in page.elements) == 1 and rng.random() < 0.3:
        return SOLE_SEARCHBOX_TEMPLATE
    templates = _usable_templates(PLAIN_TEMPLATES.get(el.role, DEFAULT_TEMPLATES), label)
    return rng.choice(templates).format(t=label)


def context_query(el: Element, page: Page, rng: random.Random) -> str | None:
    """An instruction singling out one of several same-label elements by its context phrase.

    Returns None unless the phrase is absent from the context of every other element with that label.
    """
    label = clean_label(el.text)
    phrase = context_phrase(el.context)
    if not label or len(label.split()) > MAX_PLAIN_WORDS or not content_tokens(phrase):
        return None
    if any(m in phrase.casefold() for m in FOOTER_MARKERS):
        return None
    tokens = content_tokens(phrase)
    if sum(w.isdigit() or any(c.isdigit() for c in w) for w in phrase.split()) * 2 >= len(phrase.split()):
        return None
    others = [e for e in page.elements if e.id != el.id and e.text.casefold() == el.text.casefold()]
    if any(tokens <= content_tokens(e.context) for e in others):
        return None
    verb = rng.choice(CONTEXT_VERBS.get(el.role, DEFAULT_CONTEXT_VERBS))
    return rng.choice(CONTEXT_TEMPLATES).format(v=verb, t=label, c=phrase).strip()


def examples_for_element(page: Page, el: Element, counts: Mapping[str, int], rng: random.Random) -> list[Example]:
    equiv = equiv_ids(page.elements, el.id)
    out: list[Example] = []

    def add(kind: str, query: str | None) -> None:
        if query:
            out.append(Example(page.page_id, kind, "do", query, [], el.id, equiv))

    if not is_duplicated(el, counts) or plain_is_unambiguous(el, equiv, page.elements):
        add("plain", plain_query(el, page, rng))
    if is_duplicated(el, counts) and el.context:
        add("context", context_query(el, page, rng))
    return out


def template_examples(page: Page, seed: int, n: int = SAMPLE_SIZE) -> list[Example]:
    """Deterministic template examples (mode "do") for up to `n` sampled elements of `page`."""
    rng = random.Random(f"{seed}:{page.page_id}")
    counts = label_counts(page.elements)
    examples: list[Example] = []
    for el in sample_elements(page, rng, n):
        examples.extend(examples_for_element(page, el, counts, rng))
    return examples
