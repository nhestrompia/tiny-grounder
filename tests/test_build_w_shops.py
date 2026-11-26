"""Target resolution and frozen-file helpers for the W shops evaluation builder."""

import json

from grounder.data.pages import load_examples, write_examples
from grounder.records import Element, Example
from tools.build_w_shops import action_history, canonical_site, leakage_drop, match_target_fingerprint, write_manifest


def fp(box, tag="button", href="", label="Continue"):
    return {"tag": tag, "abs_box": box, "href": href, "label": label, "visible_text": label, "nonvisual_text": ""}


def test_match_target_fingerprint_by_exact_geometry_tag_and_href():
    elements = [Element(1, "button", "Continue", tag="button", abs_box=[10, 20, 100, 30], href="/next"),
                Element(2, "button", "Cancel", tag="button", abs_box=[10, 60, 100, 30], href="/cancel")]
    match = match_target_fingerprint(elements, fp([10.0, 20.0, 100.0, 30.0], href="/next"))
    assert match.ids == [1]
    assert match.method == "exact_geometry"
    assert match.equivalent_ids == [1]


def test_match_target_fingerprint_accepts_all_equivalent_targets():
    elements = [Element(1, "link", "Next", tag="a", abs_box=[10, 20, 100, 30], href="/next"),
                Element(2, "link", "Next", tag="a", abs_box=[50, 20, 100, 30], href="/next")]
    match = match_target_fingerprint(elements, fp([10.0, 20.0, 100.0, 30.0], tag="a", href="/next", label="Next"))
    assert match.ids == [1]
    assert match.equivalent_ids == [1, 2]


def test_match_target_fingerprint_drops_ambiguous_overlapping_controls():
    elements = [Element(1, "button", "Yes", tag="button", abs_box=[10, 20, 100, 30]),
                Element(2, "button", "No", tag="button", abs_box=[10, 20, 100, 30])]
    match = match_target_fingerprint(elements, fp([10.0, 20.0, 100.0, 30.0]))
    assert match.ids == []
    assert match.reason == "ambiguous_extraction_match"


def test_match_target_fingerprint_allows_subpixel_live_differences():
    elements = [Element(1, "button", "Continue", tag="button", abs_box=[10, 20, 100, 30])]
    match = match_target_fingerprint(elements, fp([10.4, 20.4, 99.7, 30.4]))
    assert match.ids == [1]


def test_action_history_extends_action_repr_for_selects():
    el = Element(3, "combobox", "Sort products", tag="select")
    assert action_history(el, "click") == "[select]  Sort products -> CLICK"
    assert action_history(el, "type", "Ada") == "[select]  Sort products -> TYPE: Ada"
    assert action_history(el, "select", "Price (low to high)") == "[select]  Sort products -> SELECT: Price (low to high)"
    assert action_history(el, "press", "K") == "[select]  Sort products -> PRESS: K"


def test_example_tags_roundtrip_in_standard_jsonl(tmp_path):
    example = Example("p", "goal", "goal", "buy", [], 1, [1], "task", ["goal-only"])
    path = tmp_path / "examples.jsonl"
    write_examples([example], path)
    assert load_examples(path) == [example]
    assert json.loads(path.read_text().splitlines()[0])["tags"] == ["goal-only"]


def test_manifest_hashes_every_output_file(tmp_path):
    (tmp_path / "pages").mkdir()
    a = tmp_path / "examples.jsonl"
    b = tmp_path / "pages" / "p.json.gz"
    a.write_text("a")
    b.write_bytes(b"b")
    out = write_manifest(tmp_path, [a, b], {"examples": 1})
    manifest = json.loads(out.read_text())
    assert sorted(manifest["files"]) == ["examples.jsonl", "pages/p.json.gz"]
    assert all(set(row) == {"bytes", "sha256"} for row in manifest["files"].values())
    assert manifest["summary"] == {"examples": 1}


def test_leakage_drop_checks_training_domain_and_benchmark_host():
    raw = {"start_url": "https://the-internet.herokuapp.com/login", "path": [{"goto": "https://the-internet.herokuapp.com/x"}]}
    assert leakage_drop(raw, set()) == ""
    assert leakage_drop(raw, {"herokuapp.com"}) == "training_domain:herokuapp.com"
    raw["path"].append({"goto": "https://example.com/"})
    assert leakage_drop(raw, set()) == "outside_benchmark_sites:example.com"


def test_heroku_benchmark_site_stays_distinct_from_heroku_platform():
    assert canonical_site("https://the-internet.herokuapp.com/login") == "the-internet.herokuapp.com"
    assert canonical_site("https://www.saucedemo.com/") == "saucedemo.com"
