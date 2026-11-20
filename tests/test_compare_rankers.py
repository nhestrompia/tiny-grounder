from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grounder.baselines import exact
from grounder.baselines.blend import blend
from grounder.baselines.router import coverage, example_coverage, route
from grounder.ranking import Ranked
from grounder.records import Element, Example, Page
from tools import compare_rankers as cr


def _page(*texts_ctx: tuple[str, str]) -> Page:
    els = [Element(id=i, role="button", text=t, context=c) for i, (t, c) in enumerate(texts_ctx)]
    return Page("p", "u", "site", "t", [1, 1], [1, 1], els)


def _ex(query: str, target: int = 0) -> Example:
    return Example("p", "plain", "do", query, [], target, [target])


def test_exact_ranks_equal_label_first_then_containment_then_bm25():
    page = _page(("Save as draft", ""), ("Cancel", ""), ("Save", ""), ("Discard the save file draft", ""))
    r = exact.rank(page, _ex("click the Save button"))
    assert r.ids[0] == 2          # equal label {"save"} (UI words and stopwords dropped)
    assert r.ids[1] == 0          # label contains the query tokens
    assert r.ids[-1] == 1         # no match, BM25 0
    assert r.scores == sorted(r.scores, reverse=True)
    # label contained in a longer query ranks above non-matches
    r2 = exact.rank(_page(("Pricing", ""), ("Sign in", "")), _ex("go to your sign in page"))
    assert r2.ids[0] == 1


def test_exact_falls_back_to_bm25_without_match():
    page = _page(("Home", "main menu navigation"), ("Orders", "your recent orders list"), ("Help", "faq"), ("Cart", "basket"))
    r = exact.rank(page, _ex("recent purchases history", 1))
    assert r.ids[0] == 1  # no label match; BM25 on context ("recent")


def test_coverage_and_router_switching():
    page = _page(("Sign in", "account"), ("Pricing", ""))
    assert coverage("[DO] sign in", page.elements) == 1.0
    assert example_coverage(page, _ex("log into my profile")) == 0.0
    assert coverage("[DO] sign in to billing", page.elements) == pytest.approx(0.5)
    lex, mod = Ranked([0, 1], [1.0, 0.0], 0.0), Ranked([1, 0], [0.9, 0.1], 0.1)
    assert route(0.5, 0.5, lex, mod) is lex
    assert route(0.49, 0.5, lex, mod) is mod
    assert route(0.0, 0.0, lex, mod) is lex and route(1.0, 1.01, lex, mod) is mod


def test_blend_math():
    model = Ranked([1, 0, 2], [0.6, 0.3, 0.1], 0.2)
    bm = Ranked([0, 2, 1], [4.0, 2.0, 0.0], 0.0)
    assert blend(model, bm, 0.0).ids == [1, 0, 2]
    b = blend(model, bm, 1.0)
    assert b.scores == pytest.approx([np.log(0.3) + 1.0, np.log(0.6), np.log(0.1) + 0.5][:1] + [np.log(0.6), np.log(0.1) + 0.5])
    assert b.ids == [0, 1, 2] and b.null_prob == 0.2
    assert blend(model, bm, 100.0).ids == [0, 2, 1]  # large alpha -> BM25 order
    zero = Ranked([0, 1], [0.0, 0.0], 0.0)
    assert blend(Ranked([0, 1], [0.2, 0.8], 0.0), zero, 5.0).ids == [1, 0]  # all-zero BM25: no division blow-up


def test_metrics():
    ranks = np.array([[1, 3, cr.MISSING, 7]])
    m = cr.mean_metrics(ranks)
    assert m["top1"] == 0.25 and m["recall5"] == 0.5 and m["mrr"] == pytest.approx((1 + 1 / 3 + 0 + 1 / 7) / 4)


def test_bootstrap_identical_rankers_ci_is_zero():
    rng = np.random.default_rng(0)
    r = rng.integers(1, 10, size=(1, 200))
    out = cr.paired_bootstrap({"a": r, "b": r.copy()}, {"a": None, "b": None}, np.arange(200) % 20, [("a", "b")], draws=300)
    for m in cr.METRICS:
        d = out[("a", "b")][m]
        assert d["diff"] == 0 and d["lo"] == 0 and d["hi"] == 0


def test_bootstrap_better_ranker_excludes_zero():
    rng = np.random.default_rng(1)
    worse = rng.integers(2, 12, size=(1, 300))
    better = worse.copy()
    better[0, :200] = 1
    out = cr.paired_bootstrap({"good": better, "bad": worse}, {"good": None, "bad": None}, np.arange(300) % 30, [("good", "bad")], draws=500)
    d = out[("good", "bad")]["top1"]
    assert d["lo"] > 0 and d["diff"] > 0.5
    rev = cr.paired_bootstrap({"good": better, "bad": worse}, {}, np.arange(300) % 30, [("bad", "good")], draws=500)
    assert rev[("bad", "good")]["top1"]["hi"] < 0


def test_bootstrap_resamples_groups_and_seeds():
    # one group is all-good, others all-bad: group resampling must widen the CI vs. per-example resampling
    n = 100
    groups = np.repeat(np.arange(5), 20)
    good = np.where(groups[None] == 0, 1, 9)
    bad = np.full((1, n), 9)
    out = cr.paired_bootstrap({"a": good, "b": bad}, {}, groups, [("a", "b")], draws=1000)["a", "b"]["top1"]
    assert out["lo"] == 0.0 and out["hi"] >= 0.4  # draws without group 0 give 0, draws with it up to 1 group-share
    # seeds: a neural ranker with one perfect and one useless seed has a CI spanning both
    seeds = np.stack([np.ones(n, dtype=int), np.full(n, 9)])
    out = cr.paired_bootstrap({"m": seeds, "b": bad}, {"m": "m"}, np.arange(n) % 10, [("m", "b")], draws=500)["m", "b"]["top1"]
    assert out["diff"] == pytest.approx(0.5) and out["lo"] < 0.1 and out["hi"] > 0.9
    # rankers of one family share the seed draw
    same = cr.paired_bootstrap({"m": seeds, "r": seeds}, {"m": "m", "r": "m"}, np.arange(n) % 10, [("m", "r")], draws=200)["m", "r"]["top1"]
    assert same["lo"] == same["hi"] == 0


def test_refuses_to_tune_on_guirilla():
    with pytest.raises(SystemExit):
        cr.check_tune_name("guirilla_test")
    with pytest.raises(SystemExit):
        cr.check_tune_name("GUIrilla_dev")
    cr.check_tune_name("dev")


def test_tune_and_report_end_to_end():
    pages = {"p": _page(("Sign in", ""), ("Pricing", ""), ("Help", ""))}
    exs = [_ex("sign in"), _ex("pricing", 1)]
    s = cr.EvalSet("dev", pages, exs)
    cr.prepare_lexical(s)
    s.model["xs"] = [[Ranked([2, 0, 1], [0.8, 0.1, 0.1], 0.0)] * 2] * 2
    tau, alpha, _ = cr.tune(s)
    assert tau["xs"] <= 1.0 and alpha["xs"] > 0  # lexical is perfect, the fake model is not
    res = cr.report_set(s, tau, alpha, "page_site", draws=50)
    assert res["slices"]["all"]["rankers"]["exact"]["top1"] == 1.0
    assert res["slices"]["all"]["rankers"]["xs"]["top1"] == 0.0
    assert "router[xs]" in res["slices"]["all"]["rankers"]
