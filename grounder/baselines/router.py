"""Router: exact+BM25 when the query is lexically covered by some element, else the model.

coverage(query, elements) = max over elements of (query content tokens found in the element's text + context)
                            / (query content tokens), using `grounder.features` tokenisation (0 for an empty query).
"""

from __future__ import annotations

from grounder.features import content_tokens, query_tokens
from grounder.ranking import Ranked, example_elements, example_query
from grounder.records import Element, Example, Page


def coverage(query: str, elements: list[Element]) -> float:
    q = query_tokens(query)
    if not q:
        return 0.0
    return max((len(q & content_tokens(f"{e.text} {e.context}")) / len(q) for e in elements), default=0.0)


def example_coverage(page: Page, example: Example) -> float:
    return coverage(example_query(example), example_elements(page, example))


def route(cov: float, tau: float, lexical: Ranked, model: Ranked) -> Ranked:
    """The lexical (exact+BM25) ranking when coverage >= tau, else the model's."""
    return lexical if cov >= tau else model
