"""`UIGrounder`: compact `find` / `snapshot` over a page or a list of elements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from grounder.records import Element, Page
from grounder_tool.rankers import Ranker, make_ranker


@dataclass
class Hit:
    id: int
    role: str
    text: str
    context: str
    score: float
    box: list[float]

    def line(self, with_score: bool = True) -> str:
        base = f"{self.id}|{self.role}|{self.text}|{self.context}"
        return f"{base}|score={self.score:.3f}" if with_score else base


def format_lines(hits: Sequence[Hit], with_score: bool = True) -> str:
    """`id|role|text|context|score=` lines (the format demo/mcp_server.py hands to agents)."""
    return "\n".join(h.line(with_score) for h in hits)


def _elements(source: Page | Sequence[Element]) -> list[Element]:
    return list(source.elements) if isinstance(source, Page) else list(source)


class UIGrounder:
    """`ranker` is a Ranker or a spec string for make_ranker ("bm25", "model:dist/xs-s0", "router:dist/xs-s0:0.8")."""

    def __init__(self, ranker: Ranker | str = "bm25", int8: bool | None = None, threads: int = 1) -> None:
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

    format_lines = staticmethod(format_lines)
