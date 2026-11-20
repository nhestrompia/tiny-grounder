"""Compact UI grounding for agents: `find` and `snapshot` with a pluggable ranker."""

from grounder_tool.api import Hit, UIGrounder, format_lines
from grounder_tool.rankers import make_ranker

__all__ = ["Hit", "UIGrounder", "format_lines", "make_ranker"]
