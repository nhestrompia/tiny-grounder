"""Hand-computed milestone statistics and pre-registered gate boundaries."""
import json
import math
from collections import Counter

import pytest

from demo.cli_report import cluster_bootstrap
from tools import milestone_report as mr

AGENTS = ("claude:claude-haiku-5-5", "codex:gpt-6-luna")
LABELS = ("a11y-cap", "find+observe[exact]", "find+observe[exact+blend:dist/xs-s0:8]")


@pytest.fixture
def runs():
    # Each agent has three sites; the first site has two tasks.
    # Baseline successes: t0,t1,t2. B successes: t0,t2,t3.
    # Baseline tokens/success = 400/3; B = 80/3, C = 40/4.
    # B latency ratios: 1,2,3,4; both-success ratios: 1,3.
    rows = []
    for agent in AGENTS:
        for i in range(4):
            for j, label in enumerate(LABELS):
                rows.append({"task_id": f"t{i}", "site": ("s0", "s0", "s1", "s2")[i],
                             "agent": agent, "arm": label,
                             "config": "a11y-cap" if j == 0 else "find+observe",
                             "ranker": ("exact", "exact", "exact+blend:dist/xs-s0:8")[j],
                             "success": (i < 3, i != 1, True)[j],
                             "tool_tokens_est": (100, 20, 10)[j],
                             "input_tokens": (1000, 200, 100)[j],
                             "uncached_input_tokens": (500, 100, 50)[j],
                             "cost_usd": .01 if agent.startswith("claude:") else None,
                             "wall_seconds": (10, 10 * (i + 1), 5)[j],
                             "agent_duration_ms": (1000, 500, 250)[j],
                             "tool_calls": 3, "turns": 2, "grounding_ms": 100,
                             "timed_out": i == 3, "agent_error": "error" if i == 3 else None})
    return rows


@pytest.fixture
def tasks():
    return [{"id": f"t{i}", "tags": ["multistep", "vague"] if i < 2 else ["long"],
             "har": f"data/bench_v{2 if i < 2 else 3}/har/t{i}.zip"} for i in range(4)]


def test_hand_computed_pooled_metrics(runs):
    r = mr.compare(mr.index_runs(runs), "B", "baseline")
    assert r["paired_units"] == 8
    assert r["sites"] == 3
    assert r["success_delta_pp"][0] == 0
    tokens = r["resources_per_success"]["tool_tokens_est"]
    assert tokens["arm"] == pytest.approx(80 / 3)
    assert tokens["reference"] == pytest.approx(400 / 3)
    assert tokens["ratio"][0] == pytest.approx(.2)
    assert r["latency_ratio"][0] == 2.5
    assert r["both_success_units"] == 4
    assert r["both_success_latency_ratio"][0] == 2
    wall = r["resources_per_success"]["wall_seconds"]
    assert wall["arm"] == pytest.approx(100 / 3)
    assert wall["reference"] == pytest.approx(40 / 3)
    assert wall["ratio"][0] == 2.5
    assert r["resources_per_success"]["input_tokens"]["arm"] == pytest.approx(800 / 3)
    assert r["resources_per_success"]["uncached_input_tokens"]["arm"] == pytest.approx(400 / 3)
    # Cost includes only the Claude agent, including its failed run.
    assert r["resources_per_success"]["cost_usd"]["arm"] == pytest.approx(.04 / 3)
    assert r["agent_duration_ms_ratio"][0] == .5


def test_missing_pair_is_excluded(runs):
    rows = [r for r in runs if not (r["agent"] == AGENTS[0] and r["task_id"] == "t3" and mr.arm_kind(r) == "B")]
    r = mr.compare(mr.index_runs(rows), "B", "baseline")
    assert r["paired_units"] == 7
    assert r["missing_pairs"] == r["reference_only"] == 1
    assert r["arm_only"] == 0
    assert r["success_delta_pp"][0] == pytest.approx(-100 / 7)
    assert r["resources_per_success"]["tool_tokens_est"]["arm"] == 140 / 5
    assert r["resources_per_success"]["tool_tokens_est"]["reference"] == 700 / 6
    assert r["latency_ratio"][0] == 2


def test_bootstrap_preserves_sites_and_agents(runs):
    indexed = mr.index_runs(runs)
    units = [(mr.site_of(r), key) for key, r in indexed["B"].items()]
    seen = []

    def statistic(draw):
        counts = Counter(draw)
        # Both tasks and both agents from s0 always have the same multiplicity.
        assert len({counts[(t, a)] for t in ("t0", "t1") for a in AGENTS}) == 1
        assert counts[("t2", AGENTS[0])] == counts[("t2", AGENTS[1])]
        assert counts[("t3", AGENTS[0])] == counts[("t3", AGENTS[1])]
        assert len(draw) in (6, 8, 10, 12)  # three whole sites of size 4,2,2
        seen.append(len(draw))
        return len(draw)

    first = cluster_bootstrap(units, statistic, reps=100, seed=0)
    assert first == cluster_bootstrap(units, statistic, reps=100, seed=0)
    assert set(seen) == {6, 8, 10, 12}


@pytest.mark.parametrize("reference,success,tokens,latency,expected", [
    ("baseline", -9.999, .4999, 1.10, True),
    ("baseline", -10, .4, 1, False),
    ("baseline", -9, .5, 1, False),
    ("baseline", -9, .4, 1.10001, False),
    ("B", -1.999, 1, 1, True),
    ("B", -2, 1, 1, False),
    ("B", 0, 1.00001, 1, False),
    ("B", 0, 1, 1.00001, False),
    ("baseline", float("nan"), .4, 1, False),
])
def test_gate_boundaries(reference, success, tokens, latency, expected):
    result = mr.gates([0, success, 1], [.2, .1, tokens], [1, .9, latency], reference)
    assert result["overall"] is expected


def test_secondary_minus_five_is_strict():
    assert not mr.gates([0, -5, 1], [.2] * 3, [1] * 3, "baseline")["success_minus5_secondary"]
    assert mr.gates([0, -4.999, 1], [.2] * 3, [1] * 3, "baseline")["success_minus5_secondary"]


@pytest.mark.parametrize("label", ["find[exact]", "find+observe[model]", "vision", "mystery", "find+observe[exact+router:x]", "find+observe[exact"])
def test_unknown_arm(label):
    with pytest.raises(ValueError, match="Unknown arm"):
        mr.arm_kind({"arm": label})


@pytest.mark.parametrize("ranker,expected", [("exact", "B"), ("exact+exact", "B"), ("exact+blend:foo:8", "C")])
def test_config_ranker_mapping(ranker, expected):
    assert mr.arm_kind({"config": "find+observe", "ranker": ranker}) == expected


def test_duplicate_and_site_mismatch(runs):
    with pytest.raises(ValueError, match="Duplicate"):
        mr.index_runs([*runs, runs[0]])
    indexed = mr.index_runs(runs)
    indexed["B"][("t0", AGENTS[0])]["site"] = "wrong"
    with pytest.raises(ValueError, match="Site mismatch"):
        mr.compare(indexed, "B", "baseline")


def test_empty_and_no_successes(runs):
    r = mr.compare({}, "B", "baseline")
    assert r["paired_units"] == 0
    assert not r["gates"]["overall"]
    for row in runs:
        row["success"] = False
    r = mr.compare(mr.index_runs(runs), "B", "baseline")
    assert math.isnan(r["token_ratio"][0])
    assert not r["gates"]["overall"]
    assert math.isnan(r["both_success_latency_ratio"][0])


def test_secondary_and_task_file_breakdown(runs, tasks):
    rows = mr.secondary(runs)
    base = next(r for r in rows if r["agent"] == "pooled" and r["arm"] == "baseline")
    assert base["successes"] == 6
    assert base["timeouts"] == base["agent_errors"] == 2
    assert base["median_grounding_share"] == .01
    assert base["p50_wall_seconds"] == base["p90_wall_seconds"] == 10
    # Derive tags from YAML metadata, not the run's tags.
    tags = mr.breakdown(runs, tasks)
    b = next(r for r in tags if r["tag"] == "v2" and r["arm"] == "B")
    assert b["success_rate"] == .5
    assert b["tokens_per_success"] == 40
    assert {r["tag"] for r in tags} == {"multistep", "vague", "long", "v2", "v3"}
    assert math.isnan(mr.per_success([r for r in runs if r["agent"].startswith("codex:")], "cost_usd"))


def test_cli_outputs_and_ship_decision(tmp_path, runs, tasks, monkeypatch):
    # The shared helper still runs, with fewer draws for this output-format test.
    monkeypatch.setattr(mr, "BOOTSTRAP", 20)
    runfile = tmp_path / "runs.jsonl"
    runfile.write_text("\n".join(json.dumps(r) for r in runs))
    import yaml
    taskfile = tmp_path / "tasks.yaml"
    taskfile.write_text(yaml.safe_dump(tasks))
    out = tmp_path / "milestone.md"
    mr.main(["--runs", str(runfile), "--out", str(out), "--tasks", str(taskfile)])
    report = json.loads(out.with_suffix(".json").read_text())
    assert report["model_ships"] is True
    assert len(report["per_agent"]) == 2
    text = out.read_text()
    assert text.index("## Gate summary") < text.index("## claude:") < text.index("## Paired secondary")
    assert "Success -5 secondary" in text
    # C can pass the baseline gates while failing non-inferiority to B.
    for r in runs:
        if mr.arm_kind(r) == "B":
            r["wall_seconds"] = 4
    changed = mr.analyze(runs, tasks)
    assert changed["pooled"][1]["gates"]["overall"]
    assert not changed["model_ships"]
