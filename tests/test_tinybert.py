"""TinyBERT baseline: pure sampling logic always runs; model tests skip when the base model cannot be downloaded."""

from __future__ import annotations

import numpy as np
import pytest

from grounder.baselines import tinybert
from grounder.eval import evaluate, rank_with
from grounder.records import Element, Example, Page


def page_with(n: int) -> Page:
    els = [Element(id=i, role="button" if i % 2 else "link", text=f"item {i}", context="list") for i in range(n)]
    return Page("p", "u", "s", "t", [1, 1], [1, 1], els)


def test_element_and_query_strings():
    assert tinybert.element_string(Element(id=0, role="link", text="Cart", context="Header")) == "link | Cart | Header"
    ex = Example("p", "goal", "goal", "buy a lamp", ["[a] x", "[b] y"], 0, [0])
    assert tinybert.query_string(ex) == "buy a lamp | done: [a] x; [b] y"
    assert tinybert.query_string(Example("p", "plain", "do", "click cart", [], 0, [0])) == "click cart"


def test_hard_negatives_exclude_positives_and_prefer_the_targets_role():
    page = page_with(60)
    ex = Example("p", "plain", "do", "item 1", [], 1, [1])
    negs = tinybert.hard_negatives(page.elements, ex, np.random.default_rng(0))
    assert len(negs) == 7 and all(e.id != 1 for e in negs)
    assert all(e.role == "button" for e in negs)  # 29 same-role negatives fill the pool of 20


def test_hard_negatives_repeat_when_the_page_is_tiny():
    page = page_with(4)
    ex = Example("p", "plain", "do", "item 0", [], 0, [0])
    assert len(tinybert.hard_negatives(page.elements, ex, np.random.default_rng(0))) == 7


def test_training_group_puts_the_positive_first_and_skips_null():
    page = page_with(20)
    q, group = tinybert.training_group(page, Example("p", "plain", "do", "item 3", [], 3, [3]), np.random.default_rng(0))
    assert q == "item 3" and group[0] == "button | item 3 | list" and len(group) == 8
    assert tinybert.training_group(page, Example("p", "null", "do", "item 3", [], None, [3]), np.random.default_rng(0)) is None


@pytest.fixture(scope="module")
def base_ranker():
    pytest.importorskip("transformers")
    try:
        return tinybert.TinyBertRanker()
    except OSError as e:
        pytest.skip(f"cannot download {tinybert.BASE_MODEL}: {e}")


def test_fine_tune_saves_a_model_that_ranks_like_the_baseline_interface(base_ranker, synth_data, tmp_path):
    pages, examples = synth_data
    plain = [e for e in examples if e.kind == "plain"]
    out = tinybert.fine_tune(pages, plain, tmp_path / "tb", steps=2, batch_size=2)
    ranker = tinybert.TinyBertRanker(out)
    rankings = rank_with(ranker.rank, pages, plain[:5])
    assert all(len(r.ids) > 1 and r.null_prob == 0.0 and abs(sum(r.scores) - 1) < 1e-4 for r in rankings)
    assert evaluate(plain[:5], rankings)["overall"]["n"] == 5
