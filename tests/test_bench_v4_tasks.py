"""The frozen v4 hard-slice task file: non-docs sites, long sessions, icon-only controls. No network."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
import yaml

from demo.browser import load_tasks
from grounder.data.crawl import registrable_domain
from tools.heldout_sites import read_domain_file, task_domains

ROOT = Path(__file__).resolve().parent.parent
TASKS_V4 = ROOT / "demo" / "tasks_v4.yaml"
pytestmark = pytest.mark.skipif(not TASKS_V4.exists(), reason="demo/tasks_v4.yaml not generated")

DOCS_ONLY_TAGS = {"icon", "vague", "long", "multistep", "normal", "app"}


def raw() -> list[dict]:
    return yaml.safe_load(TASKS_V4.read_text(encoding="utf-8"))


def test_tasks_parse_and_are_frozen_replays() -> None:
    tasks = load_tasks(TASKS_V4)
    assert len(tasks) >= 20 and len({t.id for t in tasks}) == len(tasks)
    for t in tasks:
        assert t.har == f"data/bench_v4/har/{t.id}.zip"
        assert t.tags and set(t.tags) <= DOCS_ONLY_TAGS
        assert t.check and t.max_steps >= 3


def test_domains_are_not_dev_docs() -> None:
    """The slice exists because v2/v3 are all developer docs — shop/app domains must differ."""
    tasks = load_tasks(TASKS_V4)
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
    """Plan v4 step 2: 8-15-step sessions, icon-only controls, vague goals."""
    tags = Counter(tag for t in load_tasks(TASKS_V4) for tag in t.tags)
    assert tags["icon"] >= 5 and tags["vague"] >= 5
    long_sessions = [t for t in raw() if len(t.get("path") or []) >= 8]
    assert len(long_sessions) >= 5, f"need >=5 sessions of 8+ steps, got {len(long_sessions)}"
    for t in raw():
        assert len(t.get("path") or []) <= 15, f"{t['id']}: path longer than 15 steps"


def test_verified_and_manifested() -> None:
    results = ROOT / "data" / "bench_v4" / "verify.json"
    manifest = ROOT / "data" / "bench_v4" / "manifest.json"
    if not results.exists() or not manifest.exists():
        pytest.skip("data/bench_v4 not recorded on this machine")
    import json
    verified = json.loads(results.read_text(encoding="utf-8"))
    tasks = load_tasks(TASKS_V4)
    assert all(verified.get(t.id, {}).get("ok") for t in tasks)
    assert len(verified) == len(tasks)
