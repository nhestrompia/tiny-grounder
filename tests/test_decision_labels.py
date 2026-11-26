"""Decision-label rules on small synthetic server logs."""
import json
from pathlib import Path
from types import SimpleNamespace

from grounder.records import Example
from tools.decision_labels import (
    commitment, done_label, enough_label, rows_from_next_examples, same_target_retry,
    scan_states, state_records, worked_label,
)
from tools.harvest_queries import read_run
from tools.next_step_recall import parse_steps

FIXTURES = Path(__file__).parent / "fixtures" / "decisions"


def events(name):
    return read_run(str(FIXTURES / name))[0]


def run(success=True):
    return {
        "agent": "claude:test", "config": "fixture", "arm": "fixture", "task_id": "fixture-task",
        "repeat": 0, "success": success, "site": "example.org",
    }


def task():
    return SimpleNamespace(id="fixture-task", start_url="https://example.org/", goal="fixture goal", har="fixture.zip")


def records(name, success=True):
    return state_records(run(success), "fixture-runs", task(), events(name))[0]


def test_scan_states_groups_observations_before_the_next_commitment():
    states = scan_states(events("success_text.server.jsonl"), "https://example.org/")
    assert len(states) == 3
    first_commit, first_obs = commitment(states[0])
    assert first_commit["name"] == "click"
    assert [o["name"] for o in first_obs] == ["snapshot"]
    assert states[1]["step"] == 1
    assert states[1]["history"] == ["[a]  Docs -> CLICK"]
    assert states[2]["history"][-1] == "[input]  Search -> TYPE: install"


def test_show_and_next_rules_convert_successful_next_action_examples():
    example = Example("page-1", "goal", "goal", "fixture goal", ["[a] Docs -> CLICK"], 4, [4, 5], "fixture-task")
    prov = {
        "task": "fixture-task", "url": "https://example.org/docs", "page_id": "page-1", "run_id": "agent__arm__task__0",
        "step": 2, "call_n": 8, "agent": "claude:test", "arm": "fixture", "run_success": True,
        "site": "example.org", "action": "type", "type_text": "install",
    }
    rows, duplicates = rows_from_next_examples([example], [prov], "fixture-runs")
    assert duplicates == 0
    assert rows["show"][0]["label"] == {"target_id": 4, "target_ids": [4, 5]}
    assert rows["show"][0]["rule"] == "show.next_action_success_v1"
    assert rows["next"][0]["label"] == {
        "action_type": "type", "target_id": 4, "target_ids": [4, 5], "text": "install",
    }
    assert rows["next"][0]["provenance"]["run_success"] is True


def test_parse_steps_reads_click_and_type_targets_from_fixture_log():
    steps = parse_steps(events("success_text.server.jsonl"), "https://example.org/")
    assert [s["action"] for s in steps[:2]] == [("click", "a", "Docs"), ("type", "input", "Search")]
    assert steps[0]["action_step"] == 1
    assert steps[1]["action_step"] == 2


def test_enough_rule_uses_text_success_as_negative_and_failure_without_screenshot_as_positive():
    success_records = records("success_text.server.jsonl", success=True)
    assert all(not rec["labels"]["enough"][0] for rec in success_records)
    assert success_records[0]["labels"]["enough"][1] == "succeeded_without_screenshot"

    failure_records = records("failure_retry.server.jsonl", success=False)
    assert all(rec["labels"]["enough"][0] for rec in failure_records)
    assert failure_records[0]["labels"]["enough"][1] == "failed_without_screenshot"


def test_enough_rule_counts_explicit_and_automatic_screenshots_as_positive():
    recs = records("vision_success.server.jsonl", success=True)
    assert [rec["labels"]["enough"][1] for rec in recs] == [
        "screenshot_before_commitment",
        "vision_observation_returned_by_previous_action",
        "screenshot_before_commitment",
    ]


def test_worked_rule_marks_rechecks_errors_retries_and_progress():
    success_records = records("success_text.server.jsonl", success=True)
    assert success_records[1]["labels"]["worked"] == (False, "immediate_recheck")
    assert success_records[2]["labels"]["worked"] == (True, "proceeded")

    failure_records = records("failure_retry.server.jsonl", success=False)
    assert failure_records[1]["labels"]["worked"] == (False, "tool_error")

    vision_records = records("vision_success.server.jsonl", success=True)
    assert vision_records[2]["labels"]["worked"] == (False, "immediate_recheck")


def test_worked_retry_detection_uses_element_id_and_nearby_coordinates():
    previous = {"name": "click", "args": {"id": 7}}
    assert same_target_retry(previous, {"name": "click", "args": {"id": 7}})
    assert not same_target_retry(previous, {"name": "click", "args": {"id": 8}})
    previous_xy = {"name": "click_xy", "args": {"x": 10.0, "y": 20.0}}
    assert same_target_retry(previous_xy, {"name": "click_xy", "args": {"x": 11.0, "y": 21.0}})
    assert not same_target_retry(previous_xy, {"name": "click_xy", "args": {"x": 30.0, "y": 20.0}})


def test_done_rule_positive_only_for_done_in_a_successful_run():
    assert done_label({"name": "done"}, True) == (True, "done_called_and_run_successful")
    assert done_label({"name": "done"}, False) == (False, "done_called_but_run_failed")
    assert done_label({"name": "click"}, True) == (False, "agent_committed_to_more_work")

    success_records = records("success_text.server.jsonl", success=True)
    assert [rec["labels"]["done"][0] for rec in success_records] == [False, False, True]
    failure_records = records("failure_retry.server.jsonl", success=False)
    assert [rec["labels"]["done"][0] for rec in failure_records] == [False, False, False]


def test_worked_label_is_not_applicable_without_a_previous_action_or_commitment():
    assert worked_label(None, {"name": "click"}, []) is None
    assert worked_label({"name": "click", "is_error": False, "args": {}}, None, []) is None


def test_binary_metrics_and_auroc_handle_scores_and_ties():
    from tools.decision_baselines import auroc, binary_metrics

    y = [True, True, False, False]
    assert auroc(y, [0.9, 0.8, 0.2, 0.1]) == 1.0
    assert auroc(y, [0.5, 0.5, 0.5, 0.5]) == 0.5
    row = binary_metrics(y, [True, False, True, False], [0.9, 0.8, 0.2, 0.1])
    assert row["precision"] == 0.5
    assert row["recall"] == 0.5
    assert row["base_rate"] == 0.5


def test_assign_sites_keeps_sites_disjoint_and_covers_all():
    from tools.decision_labels import assign_sites

    assignment = assign_sites({f"site-{i}": 100 - i for i in range(20)})
    assert len(assignment) == 20
    assert set(assignment.values()) == {"train", "dev", "test"}
    assert len(assignment) == len(set(assignment))
