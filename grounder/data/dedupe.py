"""Dependency-free page-signature deduplication helpers shared by crawlers and native collectors."""

from __future__ import annotations

import math

DEDUPE_JACCARD = 0.9


def jaccard(a: frozenset, b: frozenset) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 0.0


class TemplateIndex:
    """Remembers kept pages' signatures and flags near-duplicates (exact Jaccard >= `threshold`)."""

    def __init__(self, threshold: float = DEDUPE_JACCARD) -> None:
        self.threshold = threshold
        self._by_size: dict[int, list[frozenset]] = {}

    def is_duplicate(self, signature: frozenset) -> bool:
        n = len(signature)
        if n == 0:
            return False
        low, high = int(math.ceil(self.threshold * n)), int(n / self.threshold)
        return any(
            jaccard(signature, other) >= self.threshold
            for size in range(low, high + 1)
            for other in self._by_size.get(size, ())
        )

    def add(self, signature: frozenset) -> None:
        self._by_size.setdefault(len(signature), []).append(signature)
