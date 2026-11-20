"""The frozen v2 task file: held-out domains, category quotas, well-formed checks. No network."""

from __future__ import annotations

import glob
from collections import Counter
from pathlib import Path

import pytest
import yaml

from demo.browser import load_tasks, parse_tasks
from grounder.data.crawl import registrable_domain
from tools.heldout_sites import read_domain_file, task_domains

ROOT = Path(__file__).resolve().parent.parent
TASKS_V2 = ROOT / "demo" / "tasks_v2.yaml"
pytestmark = pytest.mark.skipif(not TASKS_V2.exists(), reason="demo/tasks_v2.yaml not generated")


def raw() -> list[dict]:
    return yaml.safe_load(TASKS_V2.read_text(encoding="utf-8"))


def humaneval_sites() -> set[str]:
    """Registrable domains of the pages in the human-eval select and confirm splits (kept disjoint from the benchmark)."""
    pages = {Path(p).name.split(".")[0]: Path(p).parent.name for p in glob.glob(str(ROOT / "data/heldout/pages/test/*/*.json.gz"))}
    ids: list[str] = []
    for name in ("select_pages.txt", "confirm_pages.txt"):
        f = ROOT / "data" / "heldout" / name
        if f.exists():
            ids += [l.strip() for l in f.read_text().splitlines() if l.strip()]
    return {registrable_domain(pages[i]) for i in ids if i in pages}


def test_tasks_parse_and_are_frozen_replays() -> None:
    tasks = load_tasks(TASKS_V2)
    assert len(tasks) >= 40 and len({t.id for t in tasks}) == len(tasks)
    for t in tasks:
        assert t.har == f"data/bench_v2/har/{t.id}.zip" and t.tags and set(t.tags) <= {"icon", "vague", "long", "multistep", "normal"}
        assert t.check and t.max_steps >= 3


def test_start_and_path_domains_are_held_out() -> None:
    train = read_domain_file(ROOT / "data" / "leakage" / "heldout_domains.txt")  # lists TRAINING domains
    demo = task_domains((ROOT / "demo" / "tasks.yaml").read_text(encoding="utf-8"))
    human = humaneval_sites()
    for t in raw():
        urls = [t["start_url"]] + [s["goto"] for s in t.get("path", []) if "goto" in s]
        for u in urls:
            d = registrable_domain(u)
            assert d not in train, f"{t['id']}: {d} is a training domain"
            assert d not in demo, f"{t['id']}: {d} is a v1 task domain"
            assert d not in human, f"{t['id']}: {d} is a human-eval site"


def test_category_quotas() -> None:
    n = Counter(tag for t in raw() for tag in t["tags"])
    assert n["icon"] >= 15 and n["vague"] >= 15 and n["long"] >= 10 and n["multistep"] >= 10, n


def test_answer_tasks_have_a_scripted_answer_and_checks_are_strict() -> None:
    for t in raw():
        if t.get("answer_regex"):
            assert t.get("answer"), t["id"]
        # a bare url_contains of the site's own root would pass on the start page
        for key, val in t["check"].items():
            assert val not in ("/", "http", "https", ""), t["id"]
    assert len(parse_tasks(raw())) == len(raw())
