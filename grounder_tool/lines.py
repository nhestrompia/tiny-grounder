"""Shared compact element-line format."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class Hit:
    id: int
    role: str
    text: str
    context: str
    score: float
    box: list[float]

    def line(self, with_score: bool = True) -> str:
        """Return one id|role|text|context line with an optional score."""
        base = f"{self.id}|{self.role}|{self.text}|{self.context}"
        return f"{base}|score={self.score:.3f}" if with_score else base


def format_lines(hits: Sequence[Hit], with_score: bool = True) -> str:
    """`id|role|text|context|score=` lines (the format demo/mcp_server.py hands to agents)."""
    return "\n".join(h.line(with_score) for h in hits)
