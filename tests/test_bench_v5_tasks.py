"""The frozen v5 slice: same sites as v4, harder sessions. No network."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
import yaml

from demo.browser import load_tasks
from grounder.data.crawl import registrable_domain
from tools.heldout_sites import read_domain_file, task_domains

ROOT = Path(__file__).resolve().parent.parent
TASKS_V5 = ROOT / "demo" / "tasks_v5.yaml"
pytestmark = pytest.mark.skipif(not TASKS_V5.exists(), reason="demo/tasks_v5.yaml not generated")

DOCS_ONLY_TAGS = {"icon", "vague", "long", "multistep", "normal", "app"}


def raw() -> list[dict]:
    return yaml.safe_load(TASKS_V5.read_text(encoding="utf-8"))


def test_tasks_parse_and_are_frozen_replays() -> None:
    tasks = load_tasks(TASKS_V5)
    assert len(tasks) >= 20 and len({t.id for t in tasks}) == len(tasks)
    for t in tasks:
        assert t.har == f"data/bench_v5/har/{t.id}.zip"
        assert t.tags and set(t.tags) <= DOCS_ONLY_TAGS
        assert t.check and t.max_steps >= 3


def test_ids_do_not_overlap_v4() -> None:
    v4 = ROOT / "demo" / "tasks_v4.yaml"
    if not v4.exists():
        pytest.skip("no v4 task file")
    v4_ids = {t.id for t in load_tasks(v4)}
    for t in load_tasks(TASKS_V5):
        assert t.id not in v4_ids


def test_domains_are_not_dev_docs() -> None:
    """The slice exists because v2/v3 are all developer docs — shop/app domains must differ."""
    tasks = load_tasks(TASKS_V5)
    docs = {registrable_domain(t.start_url)
            for f in ("tasks_v2.yaml", "tasks_v3.yaml")
            if (ROOT / "demo" / f).exists()
            for t in load_tasks(ROOT / "demo" / f)}
    domains = {registrable_domain(t.start_url) for t in tasks}
    assert domains.isdisjoint(docs)
    assert len(domains) >= 4


def test_domains_are_held_out() -> None:
    train_file = ROOT / "data" / "leakage" / "heldout_domains.txt"
    if not train_file.exists():
        pytest.skip("no training-domain list")
    train = read_domain_file(train_file)
    demo = task_domains((ROOT / "demo" / "tasks.yaml").read_text(encoding="utf-8"))
    for t in raw():
        urls = [t["start_url"]] + [s["goto"] for s in t.get("path", []) if "goto" in s]
        for u in urls:
            d = registrable_domain(u)
            assert d not in train, f"{t['id']}: {d} is a training domain"
            assert d not in demo, f"{t['id']}: {d} is a v1 task domain"


def test_hard_slice_properties() -> None:
    """Plan v4 step 2, made harder: more long sessions, icon-only controls, vague goals."""
    tags = Counter(tag for t in load_tasks(TASKS_V5) for tag in t.tags)
    assert tags["icon"] >= 6 and tags["vague"] >= 6
    long_sessions = [t for t in raw() if len(t.get("path") or []) >= 8]
    assert len(long_sessions) >= 7, f"need >=7 sessions of 8+ steps, got {len(long_sessions)}"
    for t in raw():
        assert len(t.get("path") or []) <= 15, f"{t['id']}: path longer than 15 steps"


def test_alternative_routes_are_recorded() -> None:
    """Archives must cover plausible wrong routes, not only the scripted path."""
    wrong = {t["id"]: t.get("wrong") or [] for t in raw()}
    assert sum(len(w) for w in wrong.values()) >= 5
    joined = yaml.safe_dump(wrong)
    assert "product/search" in joined or "search" in joined  # tutorialsninja search results
    assert "ASUS Full HD" in joined  # demoblaze monitor product page
    assert "btn-primary" in joined  # blazedemo purchase form -> confirmation.php


def test_verified_and_manifested() -> None:
    results = ROOT / "data" / "bench_v5" / "verify.json"
    manifest = ROOT / "data" / "bench_v5" / "manifest.json"
    if not results.exists() or not manifest.exists():
        pytest.skip("data/bench_v5 not recorded on this machine")
    import json
    verified = json.loads(results.read_text(encoding="utf-8"))
    tasks = load_tasks(TASKS_V5)
    assert all(verified.get(t.id, {}).get("ok") for t in tasks)
    assert len(verified) == len(tasks)
