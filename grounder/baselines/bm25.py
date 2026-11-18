"""BM25 baseline: rank elements by Okapi BM25 of the query against `text + context`."""

from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi

from grounder.features import content_tokens, query_tokens
from grounder.ranking import Ranked, example_elements, example_query
from grounder.records import Element, Example, Page


def rank_elements(query: str, elements: list[Element]) -> Ranked:
    """BM25 ranking (ties keep page order). It never abstains: null_prob is 0."""
    q_tokens = sorted(query_tokens(query))
    docs = [sorted(content_tokens(f"{e.text} {e.context}")) for e in elements]
    if not elements or not q_tokens or not any(docs):
        scores = np.zeros(len(elements))
    else:
        scores = BM25Okapi(docs).get_scores(q_tokens)
    order = np.argsort(-scores, kind="stable")
    return Ranked([elements[i].id for i in order], [float(scores[i]) for i in order], 0.0)


def rank(page: Page, example: Example) -> Ranked:
    return rank_elements(example_query(example), example_elements(page, example))
