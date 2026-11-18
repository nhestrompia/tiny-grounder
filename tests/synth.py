"""Synthetic pages and instructions for tests and for end-to-end CLI checks.

Every page lists 12-40 elements whose labels come from a small lexicon. `plain` queries reuse the label
("click cart"); `paraphrase` queries replace every label word by its synonym ("click basket"), so a
lexical baseline cannot solve them. Duplicate labels differ by context and link target ("click cart in
sidebar"). `null` queries name an element that was removed from the page.

    uv run python tests/synth.py --out /tmp/synth --pages 600
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from grounder.data.pages import write_examples, write_page
from grounder.ranking import example_query
from grounder.records import Element, Example, Page

SYNONYMS: dict[str, str] = {
    "cart": "basket", "login": "signin", "search": "find", "home": "start", "settings": "preferences",
    "profile": "account", "help": "support", "contact": "reach", "delete": "remove", "save": "store",
    "download": "fetch", "upload": "send", "share": "post", "print": "output", "pricing": "cost",
    "news": "updates", "blog": "articles", "careers": "jobs", "about": "company", "logout": "exit",
    "register": "enroll", "subscribe": "join", "checkout": "pay", "wishlist": "favorites",
    "orders": "purchases", "reviews": "ratings", "shipping": "delivery", "returns": "refunds",
    "language": "locale", "theme": "appearance", "alerts": "notifications", "messages": "inbox",
    "calendar": "schedule", "photos": "pictures", "videos": "clips", "music": "songs",
    "maps": "directions", "tickets": "passes", "sale": "discount", "gifts": "presents",
}
FILLER = ["copyright", "privacy", "terms", "cookies", "sitemap", "partners", "press", "investors",
          "accessibility", "feedback", "status", "docs", "community", "events", "stories", "advertise"]
SECTIONS = ["header navigation", "footer links", "sidebar menu", "main content", "product card", "account panel"]
ROLES = ["button", "link", "tab", "menuitem", "button", "link"]
TEMPLATES = ["click {}", "open {}", "go to {}", "press the {} button", "select {}"]
CONTEXT_TEMPLATES = ["click {} in {}", "open {} from {}", "press {} inside {}"]
_LABELS = sorted(SYNONYMS)


def _label(rng: np.random.Generator) -> str:
    words = rng.choice(_LABELS, size=int(rng.integers(1, 3)), replace=False)
    return " ".join(words)


def _paraphrase(text: str) -> str:
    return " ".join(SYNONYMS[w] for w in text.split())


def make_page(rng: np.random.Generator, index: int) -> Page:
    """A page of 12-40 elements; a few labels appear twice in different sections."""
    n = int(rng.integers(12, 41))
    labels = [_label(rng) for _ in range(n - 4)] + list(rng.choice(FILLER, size=4))
    labels += [labels[int(rng.integers(0, len(labels)))] for _ in range(int(rng.integers(0, 3)))]
    elements = []
    for i, text in enumerate(labels):
        section = SECTIONS[int(rng.integers(0, len(SECTIONS)))]
        x, y, w, h = rng.random() * 0.8, rng.random() * 3, 0.05 + rng.random() * 0.2, 0.02 + rng.random() * 0.05
        elements.append(Element(
            id=i, role=str(rng.choice(ROLES)), text=text, context=section, tag="a", href=f"/{text.replace(' ', '-')}/{i}",
            box=[x, y, w, h], abs_box=[x * 1280, y * 800, w * 1280, h * 800], in_viewport=y < 1.0,
        ))
    rng.shuffle(elements)  # type: ignore[arg-type]
    page_id = f"synth-{index:05d}"
    return Page(page_id, f"https://synth.test/{page_id}", "synth", page_id, [1280, 800], [1280, 3200], elements)


def make_examples(rng: np.random.Generator, page: Page) -> list[Example]:
    """3 plain, 3 paraphrase, 1 context (if a label is duplicated) and 1 null example for the page."""
    by_text: dict[str, list[Element]] = {}
    for el in page.elements:
        by_text.setdefault(el.text, []).append(el)
    labelled = [els for text, els in by_text.items() if all(w in SYNONYMS for w in text.split())]
    unique = [els[0] for els in labelled if len(els) == 1]
    duplicated = [els for els in labelled if len(els) > 1]
    out: list[Example] = []

    def example(kind: str, query: str, target: Element | None, equiv: list[int]) -> Example:
        return Example(page.page_id, kind, "do", query, [], target.id if target else None, equiv)

    picks = rng.permutation(len(unique))
    for j, i in enumerate(picks[:7]):
        el = unique[int(i)]
        if j < 3:
            out.append(example("plain", str(rng.choice(TEMPLATES)).format(el.text), el, [el.id]))
        elif j < 6:
            out.append(example("paraphrase", str(rng.choice(TEMPLATES)).format(_paraphrase(el.text)), el, [el.id]))
        else:
            out.append(example("null", str(rng.choice(TEMPLATES)).format(_paraphrase(el.text)), None, [el.id]))
    for els in duplicated[:1]:
        target = els[int(rng.integers(0, len(els)))]
        if sum(e.context == target.context for e in els) == 1:
            out.append(example("context", str(rng.choice(CONTEXT_TEMPLATES)).format(target.text, target.context), target, [target.id]))
    return out


def make_dataset(n_pages: int, seed: int) -> tuple[list[Page], list[Example]]:
    rng = np.random.default_rng(seed)
    pages = [make_page(rng, i) for i in range(n_pages)]
    return pages, [ex for page in pages for ex in make_examples(rng, page)]


def texts(pages: list[Page], examples: list[Example]) -> list[str]:
    """Tokenizer training corpus: element texts, contexts and formatted queries."""
    return [t for p in pages for e in p.elements for t in (e.text, e.context)] + [example_query(e) for e in examples]


def write_dataset(out_dir: str | Path, n_pages: int, seed: int = 0) -> dict[str, Path]:
    """Write pages/, train.jsonl (90% of pages) and dev.jsonl (10%); returns the paths."""
    out = Path(out_dir)
    pages, examples = make_dataset(n_pages, seed)
    for page in pages:
        write_page(page, out / "pages")
    dev_ids = {p.page_id for p in pages[int(0.9 * n_pages):]}
    write_examples([e for e in examples if e.page_id not in dev_ids], out / "train.jsonl")
    write_examples([e for e in examples if e.page_id in dev_ids], out / "dev.jsonl")
    return {"pages": out / "pages", "train": out / "train.jsonl", "dev": out / "dev.jsonl"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pages", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print({k: str(v) for k, v in write_dataset(args.out, args.pages, args.seed).items()})
