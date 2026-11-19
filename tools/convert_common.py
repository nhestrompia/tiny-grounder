"""Pieces shared by the dataset converters (`tools/guirilla.py`, `tools/deskforge.py`)."""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np


def stats_text(name: str, n_examples: int, n_apps: int, element_counts: Sequence[int], target_roles: Counter[str]) -> str:
    """Three report lines: sizes, elements per page (median / p10 / p90) and the target role distribution."""
    p10, median, p90 = np.percentile(element_counts, [10, 50, 90]) if len(element_counts) else (0, 0, 0)
    return "\n".join([
        f"[{name}] examples: {n_examples}  pages: {len(element_counts)}  apps: {n_apps}",
        f"[{name}] elements per page: median {median:.0f}  p10 {p10:.0f}  p90 {p90:.0f}",
        f"[{name}] target roles: " + ", ".join(f"{r} {n}" for r, n in target_roles.most_common()),
    ])
