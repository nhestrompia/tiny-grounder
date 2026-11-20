"""Exact-label match + BM25: elements whose label equals or nests with the query rank first, BM25 orders the rest.

Normalisation is `grounder.features.content_tokens`: NFKC, lowercase, words, light stemming, with stop words and
generic UI words ("click", "button", "the", ...) removed, so "click the Save button" is {"save"}. The label is the
element's own `text` only (context is not a label). Tiers, best first:

  2  the label's token set equals the query's token set          ("click Sign in"  vs "Sign in")
  1  one contains the other, both non-empty, ignoring order        ("sign in to your account" vs "Sign in",
                                                                    "save" vs "Save as draft")
  0  anything else

Within a tier (and for the whole page when no element matches) the order is BM25, ties keep page order. The
reported score is `tier * 1000 + bm25`, so the scores sort the same way as the ids. It never abstains.
"""

from __future__ import annotations


from grounder.baselines.bm25 import rank_elements as bm25_rank_elements
from grounder.features import content_tokens, query_tokens
from grounder.ranking import Ranked, example_elements, example_query
from grounder.records import Element, Example, Page

TIER_WEIGHT = 1000.0


def match_tier(q_tokens: frozenset[str], label: str) -> int:
    """2 for an equal token set, 1 for containment either way, else 0 (an empty query or label never matches)."""
    t = content_tokens(label)
    if not q_tokens or not t:
        return 0
    if t == q_tokens:
        return 2
    return 1 if (t <= q_tokens or q_tokens <= t) else 0


def rank_elements(query: str, elements: list[Element]) -> Ranked:
    q_tokens = query_tokens(query)
    bm25 = bm25_rank_elements(query, elements)
    by_id = dict(zip(bm25.ids, bm25.scores))
    tiers = {e.id: match_tier(q_tokens, e.text) for e in elements}
    order = sorted(range(len(elements)), key=lambda i: (-tiers[elements[i].id], -by_id[elements[i].id], i))
    ids = [elements[i].id for i in order]
    return Ranked(ids, [tiers[i] * TIER_WEIGHT + by_id[i] for i in ids], 0.0)


def rank(page: Page, example: Example) -> Ranked:
    return rank_elements(example_query(example), example_elements(page, example))
