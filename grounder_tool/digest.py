"""Goal-aware page digests bounded by ceil(characters / 4) tokens."""

from __future__ import annotations

from math import ceil
from typing import Callable

from grounder.records import Element, Page, equiv_key
from grounder_tool.lines import Hit
from grounder_tool.rankers import Ranker

MIN_BUDGET = 100
MAX_BUDGET = 2000

LineSuffix = Callable[[Element], str]


def estimate_tokens(text: str) -> int:
    """Return the shared conservative text-token estimate, ceil(characters / 4)."""
    return ceil(len(text) / 4)


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: max(0, limit - 1)] + ("…" if limit else "")


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
) -> str:
    """Return header, outline and deduplicated goal-ranked lines within `budget` estimated tokens.

    Empty goals use the title. Coordinate suffixes count toward the budget. Tiny budgets shorten the header
    and omit components that cannot fit; element and footer lines are always complete.
    """
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
    kept = 0
    if page.elements:
        ranked = ranker.rank(goal or page.title, page.elements, "goal", history)
        by_id = {e.id: e for e in page.elements}
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
