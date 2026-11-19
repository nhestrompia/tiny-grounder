"""Several data sources: merged page directories, the example `source`, the mix, per-source model selection."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from grounder.data import build
from grounder.data.pages import load_examples, load_pages_from, write_examples, write_page
from grounder.eval import evaluate, select_score, to_markdown
from grounder.ranking import Ranked
from grounder.records import Element, Example, Page
from tools import mix
from tools.pilot_report import projection
from tools.results_table import config_name, summary_table


def page(page_id: str, site: str, n: int = 3, title: str = "t") -> Page:
    words = ["Pricing", "Contact", "Careers", "Search", "Docs", "Blog", "About", "Login"]
    els = [Element(id=i, role="button", text=words[i], href=f"/{i}", box=[0.1, 0.1 * i, 0.1, 0.05], abs_box=[0, 0, 1, 1])
           for i in range(n)]
    return Page(page_id, f"https://{site}/", site, title, [1280, 800], [1280, 800], els)


def example(page_id: str, target: int = 0, kind: str = "plain", source: str = "") -> Example:
    return Example(page_id, kind, "do", f"click Item {target}", [], target, [target], source)


# --- merged page directories ---------------------------------------------------------------------


def test_load_pages_from_merges_directories(tmp_path):
    write_page(page("a1", "a.test"), tmp_path / "x")
    write_page(page("b1", "b.test"), tmp_path / "y")
    write_page(page("a1", "a.test"), tmp_path / "y")  # the same page twice is fine
    assert sorted(load_pages_from([tmp_path / "x", tmp_path / "y"])) == ["a1", "b1"]


def test_load_pages_from_rejects_a_page_id_with_different_content(tmp_path):
    write_page(page("a1", "a.test", title="one"), tmp_path / "x")
    write_page(page("a1", "a.test", title="two"), tmp_path / "y")
    with pytest.raises(ValueError, match="a1"):
        load_pages_from([tmp_path / "x", tmp_path / "y"])


def test_build_merges_pages_directories_per_split(tmp_path):
    write_page(page("a1", "a.test", n=4), tmp_path / "ax" / "train")
    write_page(page("b1", "b.test", n=4), tmp_path / "s2ax" / "train")
    argv = ["--pages", str(tmp_path / "ax"), str(tmp_path / "s2ax"), "--out", str(tmp_path / "out"), "--min-english", "0"]
    assert build.main(argv) == 0
    assert {ex.page_id for ex in load_examples(tmp_path / "out" / "train.jsonl")} == {"a1", "b1"}
    assert load_examples(tmp_path / "out" / "dev.jsonl") == []


# --- the example source --------------------------------------------------------------------------


def test_example_source_defaults_to_empty_and_round_trips():
    ex = Example("p", "plain", "do", "q", [], 0, [0])
    assert ex.source == "" and Example.from_dict(ex.to_dict()) == ex
    assert Example.from_dict({k: v for k, v in ex.to_dict().items() if k != "source"}).source == ""
    assert Example.from_dict(replace(ex, source="web").to_dict()).source == "web"


# --- model selection -----------------------------------------------------------------------------


def ranked(first: int) -> Ranked:
    return Ranked([first, 1 - first], [0.9, 0.1], 0.0)


def test_select_score_is_the_mean_over_sources():
    examples = [example("p", 0, source="big")] * 8 + [example("p", 0, source="small")] * 2
    rankings = [ranked(0)] * 8 + [ranked(1)] * 2  # big: all right, small: all wrong
    metrics = evaluate(examples, rankings)
    assert metrics["overall"]["top1"] == pytest.approx(0.8)
    assert select_score(metrics) == pytest.approx(0.5)  # not dominated by the large source
    assert set(metrics["by_source"]) == {"big", "small"} and "| big | 8 |" in to_markdown(metrics, "t")


def test_select_score_without_sources_is_the_paraphrase_top1():
    examples = [example("p", 0, "paraphrase"), example("p", 0, "paraphrase"), example("p", 0, "plain")]
    metrics = evaluate(examples, [ranked(0), ranked(1), ranked(0)])
    assert "by_source" not in metrics and select_score(metrics) == pytest.approx(0.5)


# --- the mix -------------------------------------------------------------------------------------


def test_resample_up_and_down_is_deterministic():
    items = [example("p", i % 3) for i in range(10)]
    up = mix.resample(items, 25, seed=1)
    assert len(up) == 25 and up == mix.resample(items, 25, seed=1)
    assert Counter(map(id, up)).most_common(1)[0][1] == 3  # two whole copies plus a sample
    down = mix.resample(items, 4, seed=1)
    assert len(down) == 4 and len({id(e) for e in down}) == 4 and down != mix.resample(items, 4, seed=2)
    assert mix.resample(items, 0, seed=1) == []
    with pytest.raises(ValueError):
        mix.resample([], 3, seed=0)


def test_source_targets_split_a_group_by_natural_size():
    targets = mix.source_targets({"web": 700, "macos": 100, "deskforge": 300, "mind2web": 50}, mix.DEFAULT_WEIGHTS, 1000)
    assert targets == {"web": 700, "macos": 50, "deskforge": 150, "mind2web": 100}
    with pytest.raises(ValueError, match="no examples"):
        mix.source_targets({"web": 5, "macos": 0, "deskforge": 0, "mind2web": 5}, mix.DEFAULT_WEIGHTS, 100)


def test_parse_weights():
    assert mix.parse_weights(["web=0.5", "desktop=0.3", "mind2web=0.2"]) == {"web": 0.5, "desktop": 0.3, "mind2web": 0.2}
    for bad in (["web=1"], ["web=0.5", "desktop=0.3", "mind2web=0.3"], ["web=0.7", "desktop=0.2", "other=0.1"]):
        with pytest.raises(ValueError):
            mix.parse_weights(bad)


def test_holdout_sites_moves_whole_sites_to_dev():
    pages = {f"p{i}": page(f"p{i}", f"s{i % 5}.test") for i in range(20)}
    examples = [example(f"p{i}") for i in range(20)]
    train, dev = mix.holdout_sites(examples, pages, 0.2, seed=0)
    train_sites, dev_sites = {pages[e.page_id].site for e in train}, {pages[e.page_id].site for e in dev}
    assert len(dev_sites) == 1 and not train_sites & dev_sites and len(train) + len(dev) == 20
    assert mix.holdout_sites(examples, pages, 0.2, seed=0) == (train, dev)
    one_site = {"p0": pages["p0"], "p5": pages["p5"]}
    assert mix.holdout_sites([example("p0"), example("p5")], one_site, 0.2, 0)[1] == []  # a single site is never held out


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    """A miniature data/ with every source of the mix."""
    def make(name: str, examples: list[Example], dev: list[Example] | None = None) -> None:
        write_examples(examples, tmp_path / f"{name}.jsonl")
        if dev is not None:
            write_examples(dev, tmp_path / f"{name}-dev.jsonl")

    pages = {f"{src}{i}": page(f"{src}{i}", f"{src}{i % 4}.test") for src in ("m2w", "ax") for i in range(8)}
    for p in pages.values():
        write_page(p, tmp_path / ("mind2web/pages/train" if p.page_id.startswith("m2w") else "pages_ax/train"))
    (tmp_path / "deskforge/examples").mkdir(parents=True)
    (tmp_path / "web").mkdir()
    (tmp_path / "macos").mkdir()
    write_examples([example("w", i % 3) for i in range(40)], tmp_path / "web/train.jsonl")
    write_examples([example("w", 0)] * 5, tmp_path / "web/dev.jsonl")
    write_examples([example(f"ax{i}") for i in range(8)], tmp_path / "macos/train.jsonl")
    write_examples([example("d", 0)] * 20, tmp_path / "deskforge/examples/train.jsonl")
    write_examples([example("d", 1)] * 3, tmp_path / "deskforge/examples/dev.jsonl")
    write_examples([example(f"m2w{i}", kind="goal") for i in range(8)], tmp_path / "mind2web/train.jsonl")
    return tmp_path


def run_mix(root: Path, *extra: str) -> list[Example]:
    argv = ["--data", str(root), "--web-examples", str(root / "web"), "--macos-examples", str(root / "macos"),
            "--out", str(root / "mix"), *extra]
    assert mix.main(argv) == 0
    return load_examples(root / "mix" / "train.jsonl")


def test_mix_writes_weighted_train_and_per_source_dev(data_root, capsys):
    train = run_mix(data_root, "--total", "100")
    shares = Counter(ex.source for ex in train)
    assert len(train) == 100 and shares["web"] == 70 and shares["mind2web"] == 10
    assert shares["macos"] + shares["deskforge"] == 20 and shares["macos"] > 0 < shares["deskforge"]
    mixed_dev = load_examples(data_root / "mix" / "dev.jsonl")
    per_source = {s: load_examples(data_root / "mix" / f"dev_{s}.jsonl") for s in ("web", "macos", "deskforge", "mind2web")}
    assert [len(per_source[s]) for s in ("web", "deskforge")] == [5, 3]
    assert per_source["macos"] and per_source["mind2web"]
    assert len(mixed_dev) == sum(map(len, per_source.values()))
    assert all(ex.source == s for s, v in per_source.items() for ex in v)
    train_pages = {ex.page_id for ex in train if ex.source == "mind2web"}
    assert not train_pages & {ex.page_id for ex in per_source["mind2web"]}  # held-out sites are not trained on
    assert "natural" in capsys.readouterr().out


def test_mix_is_deterministic_and_defaults_to_the_natural_total(data_root):
    first = run_mix(data_root)
    natural = 40 + 20 + len(load_examples(data_root / "macos/train.jsonl")) + len(load_examples(data_root / "mind2web/train.jsonl"))
    held_out = sum(len(load_examples(data_root / "mix" / f"dev_{s}.jsonl")) for s in ("macos", "mind2web"))
    assert len(first) == natural - held_out
    assert first == run_mix(data_root)
    assert first != run_mix(data_root, "--seed", "1")


# --- reports -------------------------------------------------------------------------------------


def test_results_table_has_a_column_group_per_eval_set():
    def metrics(top1: float, top5: float) -> dict:
        return {"by_kind": {}, "overall": {"top1": top1, "top5": top5}}

    rows = {"xs-s0": (1356929, {"dev": metrics(0.5, 0.8), "guirilla_test": metrics(0.25, 0.5)}), "bm25": (None, {"dev": metrics(0.1, 0.2)})}
    table = summary_table(["dev", "guirilla_test"], rows).splitlines()
    assert table[0] == "| model | params | dev top-1 | dev top-5 | guirilla_test top-1 | guirilla_test top-5 |"
    assert table[2] == "| xs-s0 | 1,356,929 | 0.500 | 0.800 | 0.250 | 0.500 |"
    assert table[3] == "| bm25 | - | 0.100 | 0.200 | - | - |"
    assert [config_name(n) for n in ("m-s1", "s-s0", "xs-s2", "bm25")] == ["m", "s", "xs", "bm25"]


def test_pilot_projection_scales_calls_cost_and_time():
    web = {"live_calls": 10, "failed": 0, "prompt_tokens": 10_000, "completion_tokens": 5_000, "cost_usd": 0.01,
           "seconds": 20.0, "items_expected": 100, "items_received": 99}
    native = {**web, "live_calls": 10, "items_received": 100, "failed": 1}
    text = projection([web, native], full_calls=2000)
    assert "20 live calls" in text and "cost $0.0200" in text and "item coverage 99.50% (199/200), failed batches 1" in text
    assert "~$2.00, ~1.1 h" in text and "2.0M prompt and 1.0M completion" in text
    assert "nothing to project" in projection([{**web, "live_calls": 0}], full_calls=2000)
