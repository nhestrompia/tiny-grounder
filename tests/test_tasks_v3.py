"""Integrity and leakage gates for the frozen multistep benchmark. No live requests."""
from __future__ import annotations

import glob
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pytest
import yaml

from demo.browser import check_success, parse_tasks
from grounder.data.crawl import registrable_domain
from tools.heldout_sites import read_domain_file, task_domains

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/bench_v3"
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="v3 snapshots are not installed")


@pytest.fixture(scope="module")
def frozen():
    tasks_file = ROOT / "demo/tasks_v3.yaml"
    raw = yaml.safe_load(tasks_file.read_text())
    return raw, json.loads((DATA / "manifest.json").read_text()), tasks_file


def test_v3_task_shape_and_quotas(frozen):
    raw, _, _ = frozen
    tasks = parse_tasks(raw)
    assert 90 <= len(tasks) <= 110
    sites = Counter(registrable_domain(t.start_url) for t in tasks)
    assert len(sites) >= 30 and max(sites.values()) <= 4
    assert sum("vague" in t.tags for t in tasks) >= len(tasks) / 2
    assert .50 <= sum(bool(t.answer_regex) for t in tasks) / len(tasks) <= .70
    for r, t in zip(raw, tasks):
        assert "multistep" in t.tags
        assert set(t.tags) <= {"multistep", "vague", "long", "icon", "normal"}
        assert 2 <= len(r["path"]) <= 5
        assert t.max_steps == len(r["path"]) + 3
        assert all(set(step) == {"click"} for step in r["path"])
        assert t.har == f"data/bench_v3/har/{t.id}.zip"
        assert not check_success(t.check, t.start_url, "")
        if t.answer_regex:
            assert re.search(t.answer_regex, r["answer"], re.I)
            assert not r.get("no_evidence")


def test_v3_navigation_domains_are_held_out(frozen):
    raw, _, _ = frozen
    excluded = read_domain_file(ROOT / "data/leakage/heldout_domains.txt")
    for name in ("tasks.yaml", "tasks_v2.yaml"):
        excluded |= task_domains((ROOT / "demo" / name).read_text())
    pages = {Path(p).name.split(".")[0]: Path(p).parent.name
             for p in glob.glob(str(ROOT / "data/heldout/pages/test/*/*.json.gz"))}
    for name in ("select_pages.txt", "confirm_pages.txt"):
        ids = (ROOT / "data/heldout" / name).read_text().splitlines()
        excluded |= {registrable_domain(pages[i.strip()]) for i in ids if i.strip() in pages}
    audit = json.loads((DATA / "authoring.json").read_text())
    for t in raw:
        entry = audit[t["id"]]
        assert entry["ok"] and len(entry["pages"]) == len(t["path"]) + 1
        urls = [p["url"] for p in entry["pages"]]
        assert len(set(urls)) == len(urls)
        assert check_success(t["check"], urls[-1], "")
        for url in [t["start_url"], *urls]:
            assert registrable_domain(url) not in excluded, (t["id"], url)
        assert entry["robots"]


def test_v3_manifest_covers_exact_artifacts(frozen):
    raw, manifest, tasks_file = frozen
    assert manifest["tasks_file"]["tasks"] == len(raw)
    assert manifest["tasks_file"]["sha256"] == hashlib.sha256(tasks_file.read_bytes()).hexdigest()
    assert set(manifest["snapshots"]) == {t["id"] for t in raw}
    for t in raw:
        snapshot = ROOT / t["har"]
        entry = manifest["snapshots"][t["id"]]
        assert snapshot.stat().st_size == entry["bytes"]
        assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == entry["sha256"]


def test_v3_all_scripted_offline_replays_pass(frozen):
    raw, _, _ = frozen
    verification = json.loads((DATA / "verify.json").read_text())
    assert set(verification) == {t["id"] for t in raw}
    for t in raw:
        r = verification[t["id"]]
        assert all(r[k] for k in ("ok", "do_nothing_fails", "check_ok", "answer_ok", "evidence_ok")), r
        assert r["network_failures"] == 0, r


def test_v3_scripted_accessibility_baseline_passes(frozen):
    raw, _, _ = frozen
    verification = json.loads((DATA / "a11y_verify.json").read_text())
    assert set(verification) == {t["id"] for t in raw}
    for t in raw:
        r = verification[t["id"]]
        assert r["ok"] and r["aria_evidence_ok"], r
        assert r["network_failures"] == 0 and len(r["refs"]) == len(t["path"]), r


def test_v3_manifest_cli_accepts_relative_tasks_file(tmp_path, monkeypatch):
    from tools import bench_v2

    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(bench_v2, "HAR_DIR", bench_v2.HAR_DIR)
    assert bench_v2.main(["manifest", "--tasks-file", "demo/tasks_v3.yaml", "--out", str(tmp_path)]) == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["tasks_file"]["path"] == "demo/tasks_v3.yaml"
    assert len(manifest["snapshots"]) == len(yaml.safe_load((ROOT / "demo/tasks_v3.yaml").read_text()))
    assert (tmp_path / "har").is_dir()
