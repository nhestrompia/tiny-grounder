"""Synthetic checks for the pre-registered competitor analysis."""
import json
import math

import pytest
import yaml

from tools import competitor_report as cr


@pytest.fixture
def runs():
    rows = []
    for agent in cr.AGENTS:
        for i in range(3):
            for arm, config in zip(cr.ARMS, ("find+observe", "pw-mcp+observe", "pw-mcp-compact", "agent-browser")):
                rows.append(dict(task_id=f"t{i}", agent=agent, site=f"s{i // 2}", config=config,
                                 ranker="exact" if arm == "B" else "grounder", arm="arbitrary display label",
                                 success=True, has_server_summary=True, input_tokens=50 if arm in ("B", "M") else 100,
                                 tool_tokens_est=200 if arm == "B" else 10, uncached_input_tokens=20,
                                 cost_usd=.01 if agent.startswith("claude:") else None,
                                 wall_seconds=10, wall_seconds_without_harness=9, tool_calls=3, turns=2,
                                 safety_ms=1, settle_ms=2, proxy_ms=3, harness_overhead_ms=6,
                                 replay_live_requests=0, foreign_tools=[], timed_out=False, agent_error=""))
    return rows


@pytest.fixture(autouse=True)
def fast_bootstrap(monkeypatch):
    # Real shared bootstrap, fewer draws except in the explicit reproducibility check.
    monkeypatch.setattr(cr, "BOOTSTRAP", 20)


def test_mapping(runs):
    assert {cr.arm_kind(r) for r in runs} == set(cr.ARMS)
    assert cr.arm_kind(dict(config="pw-mcp", ranker="anything")) == "pw-mcp"
    with pytest.raises(ValueError, match="Unknown arm"):
        cr.arm_kind(dict(config="find+observe", ranker="blend"))


def test_pairing_and_input_gate(runs):
    result = cr.compare(cr.index_runs(list(reversed(runs))), "B", "pw-mcp-compact",
                        {(f"t{i}", a) for i in range(3) for a in cr.AGENTS})
    assert result["paired_units"] == 12
    assert result["sites"] == 2
    assert result["token_ratio"] == [.5] * 3  # Tool output ratio is 20, not the primary gate.
    assert result["latency_ratio"] == [1.] * 3
    assert result["gates"]["overall"]
    assert result["resources_per_success"]["cost_usd"]["arm"] == .01
    assert result["adjusted_latency_ratio"] == [1.] * 3
    runs[0]["site"] = "other"
    with pytest.raises(ValueError, match="Site mismatch"):
        cr.compare(cr.index_runs(runs), "B", "pw-mcp-compact", set())
    with pytest.raises(ValueError, match="Duplicate"):
        cr.index_runs([*runs, runs[0]])


def test_missing_pairs_and_global_provisional(runs):
    result = cr.analyze(runs[:-1])
    assert result["provisional"]
    assert result["pooled"][1]["missing_pairs"] == 1
    assert not result["beats_existing_compact_tools"]
    text = cr.markdown(result)
    assert "PROVISIONAL" in text and "PASS" not in text
    assert "PENDING" in text
    # Entire absent tasks and agents must also prevent a pass.
    result = cr.analyze(runs, tasks=[{"id": f"t{i}"} for i in range(4)])
    assert all(r["missing_pairs"] == 4 for r in result["pooled"])
    assert not any(v for rows in result["per_agent"].values() for r in rows for v in r["gates"].values())


def test_harness_failures_are_pairwise(runs):
    runs[0]["agent_error"] = "NO SERVER SUMMARY; failure"
    runs[1]["has_server_summary"] = False
    result = cr.analyze(runs)
    assert not result["provisional"]  # Records exist, but their paired observations are excluded.
    assert [r["harness_excluded_pairs"] for r in result["pooled"]] == [1, 1, 1]
    assert all(r["paired_units"] == 11 for r in result["pooled"])
    pooled_b = next(r for r in result["chart_data"] if r["agent"] == "pooled" and r["arm"] == "B")
    assert pooled_b["harness_failures"] == 1 and pooled_b["n"] == 11
    assert pooled_b["agent_errors"] == 1
    assert pooled_b["total_safety_ms"] == 12
    assert cr.harness_failure({})


def test_both_rule_and_optional_stock(runs):
    report = cr.analyze(runs)
    assert report["beats_existing_compact_tools"] and report["middleware_passes"]
    for r in runs:
        if cr.arm_kind(r) == "agent-browser":
            r["input_tokens"] = 25
    report = cr.analyze(runs)
    assert report["pooled"][0]["gates"]["overall"]
    assert not report["pooled"][1]["gates"]["overall"]
    assert not report["beats_existing_compact_tools"]
    stock = dict(runs[0], config="pw-mcp")
    report = cr.analyze([*runs, stock])
    assert len(report["pooled"]) == 3
    assert any(r["arm"] == "pw-mcp" for r in report["chart_data"])
    assert not report["provisional"]


@pytest.mark.parametrize("success,tokens,latency,passed", [
    (-9.999, .9999, 1.10, True), (-10, .9, 1., False),
    (-9, 1., 1., False), (-9, .9, 1.100001, False),
    (float("nan"), .9, 1., False), (-9, float("nan"), 1., False),
    (-9, .9, float("nan"), False),
])
def test_boundaries(success, tokens, latency, passed):
    assert cr.gates([success] * 3, [tokens] * 3, [latency] * 3)["overall"] is passed
    assert not cr.gates([success] * 3, [tokens] * 3, [latency] * 3, provisional=True)["overall"]


def test_failed_runs_and_missing_telemetry(runs):
    # Failed attempts still spend input tokens and still count in latency.
    for r in runs:
        r["success"] = r["task_id"] != "t2"
        if cr.arm_kind(r) == "B":
            r["wall_seconds"] = {"t0": 10, "t1": 20, "t2": 40}[r["task_id"]]
    result = cr.analyze(runs)
    assert result["pooled"][0]["resources_per_success"]["input_tokens"]["arm"] == 75
    assert result["pooled"][0]["latency_ratio"][0] == 2
    runs[0]["wall_seconds_without_harness"] = None
    result = cr.analyze(runs)
    assert math.isnan(result["pooled"][0]["adjusted_latency_ratio"][0])
    for r in runs:
        r["success"] = False
    assert not cr.analyze(runs)["beats_existing_compact_tools"]


def test_loader_cli_json(tmp_path, runs, capsys):
    logs = tmp_path / "logs"
    logs.mkdir()
    for i, r in enumerate(runs):
        path = logs / f"{i}.json"
        path.write_text(json.dumps(r))
        path.with_suffix(".server.jsonl").write_text('{"type":"summary"}\n{')
    (logs / "incomplete.json").write_text("{")
    loaded = cr.load_runs(tmp_path)
    assert len(loaded) == 48 and all(r["has_server_summary"] for r in loaded)
    (logs / "0.server.jsonl").unlink()
    assert sum(cr.harness_failure(r) for r in cr.load_runs(tmp_path)) == 1
    tasks = tmp_path / "tasks.yaml"
    tasks.write_text(yaml.safe_dump([{"id": f"t{i}"} for i in range(4)]))
    out, data = tmp_path / "report.md", tmp_path / "chart.json"
    cr.main(["--runs-dir", str(tmp_path), "--out", str(out), "--json", str(data), "--tasks", str(tasks)])
    report = json.loads(data.read_text())
    assert report["seed"] == 0 and report["confidence"] == .95
    assert report["provisional"] and "PASS" not in capsys.readouterr().out
    assert "PROVISIONAL" in out.read_text()
    assert len(report["chart_data"]) == 20
    required = {"agent", "arm", "success_rate", "input_tokens_per_success", "claude_cost_per_success", "median_wall_seconds"}
    assert all(required <= row.keys() for row in report["chart_data"])
    assert all(len(row["success_rate"]) == 3 for row in report["chart_data"])
    assert next(row for row in report["chart_data"] if row["agent"].startswith("codex:"))["claude_cost_per_success"] is None
    assert "NaN" not in data.read_text()


def test_bootstrap_uses_shared_method(monkeypatch):
    monkeypatch.setattr(cr, "BOOTSTRAP", 4000)
    units = [("s0", 1), ("s0", 3), ("s1", 9)]
    stat = lambda xs: sum(xs) / len(xs)
    assert cr.bootstrap(units, stat) == list(cr.cluster_bootstrap(units, stat, reps=4000, seed=0))
