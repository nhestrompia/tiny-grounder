"""`UIGrounder`: compact `find` / `snapshot` over a page or a list of elements."""

from __future__ import annotations

from typing import Sequence

from grounder.playwright_helper import Passage
from grounder.records import Element, Page
from grounder_tool.digest import LineSuffix, inspect, observe
from grounder_tool.lines import Hit, format_lines
from grounder_tool.rankers import Ranker, make_ranker


def _elements(source: Page | Sequence[Element]) -> list[Element]:
    return list(source.elements) if isinstance(source, Page) else list(source)


class UIGrounder:
    """`ranker` is a Ranker or a spec string for make_ranker ("exact", "bm25", "blend:dist/xs-s0:8", "model:dist/xs-s0")."""

    def __init__(self, ranker: Ranker | str = "exact", int8: bool | None = None, threads: int = 1) -> None:
        self.ranker = make_ranker(ranker, int8, threads) if isinstance(ranker, str) else ranker

    def _top(self, source, query, k, mode, history) -> list[Hit]:
        els = _elements(source)
        if not els or k <= 0:
            return []
        ranked = self.ranker.rank(query, els, mode, history)
        by_id = {e.id: e for e in els}
        return [
            Hit(i, by_id[i].role, by_id[i].text, by_id[i].context, float(s), by_id[i].abs_box)
            for i, s in zip(ranked.ids[:k], ranked.scores[:k])
        ]

    def find(self, source: Page | Sequence[Element], query: str, k: int = 5) -> list[Hit]:
        """The `k` elements best matching a short description ("search box"), best first."""
        return self._top(source, query, k, "do", None)

    def snapshot(
        self, source: Page | Sequence[Element], goal: str, k: int = 20, history: list[str] | None = None
    ) -> list[Hit]:
        """The `k` elements most relevant to a goal, best first (goal mode)."""
        return self._top(source, goal, k, "goal", history)

    def observe(
        self, source: Page, goal: str = "", budget: int = 600,
        history: list[str] | None = None, landmarks: list[str] | None = None,
        line_suffix: LineSuffix | None = None, passages: list[Passage] | None = None,
    ) -> str:
        """Return a goal-ranked digest within the estimated token budget."""
        return observe(source, goal, self.ranker, budget, history, line_suffix, landmarks, passages)

    def inspect(
        self, source: Page, id: int | None = None, region: str = "", budget: int = 400,
        landmarks: list[str] | None = None, line_suffix: LineSuffix | None = None,
        heading_positions: list[tuple[str, float]] | None = None,
    ) -> str:
        """Return element or heading-region details within the estimated token budget."""
        return inspect(source, id, region, budget, line_suffix, landmarks, heading_positions)

    format_lines = staticmethod(format_lines)
