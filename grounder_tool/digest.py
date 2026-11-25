"""Goal-aware page digests bounded by ceil(characters / 4) tokens."""

from __future__ import annotations

from math import ceil
import re
from typing import Callable

from grounder.playwright_helper import Passage
from grounder.baselines.exact import rank_elements as lexical_rank
from grounder.features import content_tokens, query_tokens
from grounder.records import Element, Page, equiv_key
from grounder_tool.lines import Hit
from grounder_tool.rankers import Ranker

MIN_BUDGET = 100
MAX_BUDGET = 2000
# A short digit-bearing block within this many CSS pixels of a shown element belongs to it
# (a price, total or count sitting next to the control it describes).
NEAR_Y = 120
NEAR_TEXT_CHARS = 80
# observe() looks for such numbers next to its first few elements only; read() matches more.
NEAR_TOP = 3
NEAR_ELEMENTS = 10
# read() adds element-context passages only when lexical matching found fewer than this many.
READ_FALLBACK_LINES = 3

_DIGIT_RE = re.compile(r"\d")
_NUMBER_RE = re.compile(r"\d[\d.,:]*")

LineSuffix = Callable[[Element], str]


def estimate_tokens(text: str) -> int:
    """Return the shared conservative text-token estimate, ceil(characters / 4)."""
    return ceil(len(text) / 4)


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: max(0, limit - 1)] + ("…" if limit else "")


def _centre_y(element: Element) -> float | None:
    return element.abs_box[1] + element.abs_box[3] / 2 if any(element.abs_box) else None


def _numbers_near(passages: list[Passage], elements: list[Element], near_y: float = NEAR_Y) -> list[Passage]:
    """Short digit-bearing blocks beside `elements` (prices, totals, counts), in document order.

    Blocks whose numbers the shown elements already carry in their text or context are skipped."""
    centres = [y for e in elements if (y := _centre_y(e)) is not None]
    if not centres:
        return []
    known = set(_NUMBER_RE.findall(" ".join(f"{e.text} {e.context}" for e in elements)))
    return [
        p for p in passages
        if len(p.text) <= NEAR_TEXT_CHARS and _DIGIT_RE.search(p.text)
        and not all(n in known for n in _NUMBER_RE.findall(p.text))
        and any(abs(p.y - y) <= near_y for y in centres)
    ]


def _line(e: Element, suffix: LineSuffix | None) -> str:
    return Hit(e.id, _clip(e.role, 24), _clip(e.text, 80), _clip(e.context, 40), 0, e.abs_box).line(False) + (
        suffix(e) if suffix else ""
    )


def _outline(landmarks: list[str], cap: int) -> str:
    names = list(dict.fromkeys(_clip(name, 60) for name in landmarks if name.strip()))
    parts: list[str] = []
    for name in names:
        remaining = len(names) - len(parts) - 1
        candidate = "headings: " + " · ".join([*parts, name])
        reserve = len(f" · +{remaining} more") if remaining else 0
        if len(candidate) + reserve > cap:
            break
        parts.append(name)
    if len(parts) < len(names):
        parts.append(f"+{len(names) - len(parts)} more")
    out = "headings: " + " · ".join(parts) if parts else ""
    return out if len(out) <= cap else ""


def observe(
    page: Page,
    goal: str,
    ranker: Ranker,
    budget: int = 600,
    history: list[str] | None = None,
    line_suffix: LineSuffix | None = None,
    landmarks: list[str] | None = None,
    passages: list[Passage] | None = None,
    adaptive_min: int | None = None,
) -> str:
    """Return header, outline and deduplicated goal-ranked lines within `budget` estimated tokens.

    Empty goals use the title. Coordinate suffixes count toward the budget. Tiny budgets shorten the header
    and omit components that cannot fit; element and footer lines are always complete. Short numbers next to
    the top-ranked elements (prices, totals, counts) are part of the text section, ahead of ranked passages.

    With `adaptive_min`, a ranker that reports a null probability scales the digest between
    max(MIN_BUDGET, adaptive_min) when it is sure (low null_prob) and `budget` when it is not.
    Lexical rankers abstain via `.lexical` and always get `budget`.
    """
    ranked = ranker.rank(goal or page.title, page.elements, "goal", history) if page.elements else None
    if adaptive_min is not None and ranked is not None and not getattr(ranker, "lexical", False):
        floor = max(MIN_BUDGET, min(adaptive_min, budget))  # never grow the digest past the caller's budget
        confidence = 1.0 - min(1.0, max(0.0, ranked.null_prob))
        budget = budget - round((budget - floor) * confidence)
    cap = max(0, budget) * 4
    total = len(page.elements)

    def footer(k: int) -> str:
        return f"showing {k} of {total} elements; call find(query) for others or inspect(id) for detail"

    header_cap = max(0, cap - len(footer(0)) - 1)
    if header_cap < len("url: \ntitle: "):
        return _clip(f"url: {page.url} title: {page.title}", cap)
    url_cap = min(200, (header_cap - 13) // 2)
    title_cap = min(120, header_cap - 13 - url_cap)
    lines = [f"url: {_clip(page.url, url_cap)}", f"title: {_clip(page.title, title_cap)}"]
    used = sum(map(len, lines)) + 1
    outline = _outline(landmarks or [], int(cap * 0.15))
    if outline and used + len(outline) + len(footer(0)) + 2 <= cap:
        lines.append(outline)
        used += len(outline) + 1
    by_id = {e.id: e for e in page.elements}
    top = [by_id[i] for i in (ranked.ids[:NEAR_TOP] if ranked else [])]
    adjacent = _numbers_near(passages or [], top)
    text_used = 0
    seen_text = set()
    for passage in [*adjacent, *(p for p, _ in rank_passages(goal or page.title, passages or [], ranker))]:
        if passage.text in seen_text:
            continue
        seen_text.add(passage.text)
        line = passage_line(passage)
        cost = len(line) + 1
        if text_used + cost > int(cap * 0.30):
            continue
        if used + cost + len(footer(0)) + 1 > cap:
            continue
        lines.append(line)
        used += cost
        text_used += cost
    kept = 0
    if ranked is not None:
        seen = set()
        for identifier in ranked.ids:
            # Even the shortest id|role|text|context line needs four characters and a newline.
            if cap - used - len(footer(kept + 1)) - 1 < 5:
                break
            e = by_id[identifier]
            key = equiv_key(e)
            if key in seen:
                continue
            seen.add(key)
            line = _line(e, line_suffix)
            if used + len(line) + len(footer(kept + 1)) + 2 > cap:
                break
            lines.append(line)
            used += len(line) + 1
            kept += 1
    return "\n".join([*lines, footer(kept)])


def inspect(
    page: Page,
    id: int | None = None,
    region: str = "",
    budget: int = 400,
    line_suffix: LineSuffix | None = None,
    landmarks: list[str] | None = None,
    heading_positions: list[tuple[str, float]] | None = None,
) -> str:
    """Return clipped element details and nearby lines, or a section, within `budget` estimated tokens.

    Omitted detail/element lines are counted in an intact +N more footer. Region matching is case-insensitive
    by heading or context prefix. Heading regions end at the next heading. Missing selectors and unknown ids/regions return bounded usage messages.
    """
    cap = max(0, budget) * 4
    ordered = sorted(page.elements, key=lambda e: (e.abs_box[1] if e.abs_box else 0, e.id))
    details: list[str] = []
    if id is not None:
        target = next((e for e in ordered if e.id == id), None)
        if target is None:
            return _clip(f"Unknown element {id}; call observe() or find(query) for current ids.", cap)
        x, y, w, h = target.abs_box
        details = [_line(target, line_suffix)] + [
            f"{key}: {_clip(str(value), 80)}"
            for key, value in (
                ("role", target.role),
                ("text", target.text),
                ("context", target.context),
                ("tag", target.tag),
                ("href", target.href),
                ("enabled", target.enabled),
                ("in_viewport", target.in_viewport),
                ("center", f"{x + w / 2:g},{y + h / 2:g}"),
            )
        ]
        section = [e for e in ordered if e.context == target.context]
        index = next(i for i, e in enumerate(section) if e.id == id)
        neighbours = sorted(
            ((i, e) for i, e in enumerate(section) if e.id != id), key=lambda pair: abs(pair[0] - index)
        )
        candidates = [e for _, e in sorted(neighbours[:10])]
    elif region:
        headings = heading_positions or []
        match = next((i for i, (text, _) in enumerate(headings)
                      if text.casefold().startswith(region.casefold())), None)
        if match is not None and any(e.abs_box and any(e.abs_box) for e in ordered):
            start = headings[match][1]
            end = headings[match + 1][1] if match + 1 < len(headings) else float("inf")
            candidates = [e for e in ordered if e.abs_box and start <= e.abs_box[1] < end]
        else:
            candidates = [e for e in ordered if e.context.casefold().startswith(region.casefold())]
        if not candidates:
            return _clip(f"Unknown section {region!r}; call observe() for heading names.", cap)
    else:
        return _clip("Usage: inspect(id=element_id) or inspect(region=section_name).", cap)
    pending = details + [_line(e, line_suffix) for e in candidates]
    lines: list[str] = []
    used = 0
    for i, line in enumerate(pending):
        remaining = len(pending) - i - 1
        reserve = len(f"+{remaining} more") + 1 if remaining else 0
        if used + len(line) + reserve > cap:
            break
        lines.append(line)
        used += len(line) + 1
    if len(lines) < len(pending):
        more = f"+{len(pending) - len(lines)} more"
        if used + len(more) <= cap:
            lines.append(more)
    return "\n".join(lines)


def rank_passages(query: str, passages: list[Passage], ranker: Ranker | None = None) -> list[tuple[Passage, float]]:
    """Rank passages for the query; ties retain document order.

    Without a ranker this is exact/BM25 lexical scoring and drops zero-score passages. With one, the ranker
    decides the order (model/blend scores are logits or probabilities, not lexical match counts, so they are
    not thresholded); lexical rankers keep the score > 0 drop."""
    if not passages:
        return []
    elements = [Element(i, "text", p.text, p.heading) for i, p in enumerate(passages)]
    if ranker is None:
        ranked = lexical_rank(query, elements)
        return [(passages[i], score) for i, score in zip(ranked.ids, ranked.scores) if score > 0]
    ranked = ranker.rank(query, elements, "goal")
    if getattr(ranker, "lexical", False):
        # Raw BM25 scores can be negative even on a match, so the no-overlap gate uses the exact scorer's
        # positive match bonus rather than the ranker's own score.
        lex = lexical_rank(query, elements)
        keep = {i for i, s in zip(lex.ids, lex.scores) if s > 0}
        return [(passages[i], score) for i, score in zip(ranked.ids, ranked.scores) if i in keep]
    return [(passages[i], score) for i, score in zip(ranked.ids, ranked.scores)]


def passage_line(passage: Passage) -> str:
    heading = f"  [§ {_clip(passage.heading, 40)}]" if passage.heading else ""
    return f"> {_clip(passage.text, 300)}{heading}"


def context_passages(query: str, passages: list[Passage], elements: list[Element]) -> list[Passage]:
    """Passages taken from the matched elements' context (the row text extraction already records)
    plus short numbers sitting beside those elements."""
    if not elements:
        return []
    by_id = {e.id: e for e in elements}
    ranked = lexical_rank(query, elements)
    # Matches share a content token with the query (the corpus BM25 scores is `text + context`);
    # raw scores are not a usable gate: BM25 can return 0 on a match.
    q_tokens = query_tokens(query)
    matched = [
        by_id[i] for i in ranked.ids
        if content_tokens(f"{by_id[i].text} {by_id[i].context}") & q_tokens
    ][:NEAR_ELEMENTS]
    contexts: dict[str, float] = {}
    for e in matched:
        if e.context:
            contexts.setdefault(e.context, e.abs_box[1])
    near = {id(p) for p in _numbers_near(passages, matched)}
    return [
        *(Passage(context, y=y) for context, y in contexts.items()),
        *(p for p in passages if id(p) in near),
    ]


def read(
    query: str, passages: list[Passage], budget: int = 400, ranker: Ranker | None = None,
    elements: list[Element] | None = None,
) -> str:
    """Return relevant text with an intact count footer within the clamped budget.

    When fewer than READ_FALLBACK_LINES passages match the query, passages from the context of
    elements matching the query (their section, or numbers beside them) are appended after the matches.
    """
    cap = max(MIN_BUDGET, min(MAX_BUDGET, budget)) * 4
    ranked = [p for p, _ in rank_passages(query, passages, ranker)]
    # Model rankers return every passage unscored, so count real matches by token overlap
    # (the same gate context_passages uses for elements).
    q_tokens = query_tokens(query)
    hits = [p for p in ranked if content_tokens(f"{p.text} {p.heading}") & q_tokens]
    extra: list[Passage] = []
    if elements is not None and len(hits) < READ_FALLBACK_LINES:
        shown = {p.text for p in ranked}
        extra = [p for p in context_passages(query, passages, elements) if p.text not in shown]
    total = len(passages) + len(extra)
    lines: list[str] = []
    used = 0
    for passage in [*ranked, *extra]:
        line = passage_line(passage)
        footer = f"showing {len(lines) + 1} of {total} passages"
        if used + len(line) + 1 + len(footer) <= cap:
            lines.append(line)
            used += len(line) + 1
    return "\n".join([*lines, f"showing {len(lines)} of {total} passages"])
