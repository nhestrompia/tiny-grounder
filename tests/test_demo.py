"""Agent demo tests: the full loop against local fixture pages with a scripted planner (no network, no API key)."""

from __future__ import annotations

import functools
import http.server
import json
import re
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from demo import agent
from demo.agent import BM25Ranker, ClaudePlanner, RunResult, run_task
from demo.browser import Task, ToolError, check_success, load_tasks, parse_tasks
from demo.report import load_runs, results_table, write_results
from demo.run import run_one

FIXTURES = Path(__file__).parent / "fixtures" / "demo"


@pytest.fixture(scope="module")
def site() -> str:
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            pass

    handler = functools.partial(Quiet, directory=str(FIXTURES))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def make_task(site: str, goal: str = "Open the install guide in the documentation", **check: str) -> Task:
    return Task("local", f"{site}/index.html", goal, check or {"url_contains": "install.html"}, max_steps=6)


def element_id(text: str, label: str) -> int:
    """Id of the element whose text is `label` in a snapshot / find result."""
    match = re.search(rf"^(\d+)\|[a-z]+\|{re.escape(label)}\|", text, re.MULTILINE)
    assert match, f"{label!r} not in:\n{text}"
    return int(match.group(1))


def response(*calls: tuple[str, dict], stop_reason: str = "tool_use", usage: tuple[int, int, int, int] = (100, 10, 50, 5)):
    """A fake API message. usage = (input, output, cache_read, cache_creation)."""
    blocks = [SimpleNamespace(type="tool_use", id=f"toolu_{i}", name=n, input=a) for i, (n, a) in enumerate(calls)]
    i, o, r, c = usage
    return SimpleNamespace(
        content=blocks, stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=i, output_tokens=o, cache_read_input_tokens=r, cache_creation_input_tokens=c),
    )


class ScriptedPlanner:
    """Each script step is a function of the previous tool results (one string) returning a fake response."""

    def __init__(self, *steps: Callable[[str], Any]) -> None:
        self.steps, self.requests = list(steps), []

    def create(self, system: str, tools: list[dict], messages: list[dict]) -> Any:
        last = messages[-1]["content"]
        results = "\n".join(r["content"] for r in last) if isinstance(last, list) else last
        self.requests.append({"system": system, "tools": [t["name"] for t in tools], "results": results})
        return self.steps[len(self.requests) - 1](results)


def run(task: Task, config: str, planner: ScriptedPlanner, ranker=None) -> tuple[RunResult, list[dict]]:
    events: list[dict] = []
    return run_task(task, config, planner, ranker or BM25Ranker(), events.append), events


def navigate_script(finder: str) -> list[Callable[[str], Any]]:
    """Home -> Documentation -> Install guide -> done, listing elements with `finder` ('snapshot' or 'find')."""

    def look(query: str) -> tuple[str, dict]:
        return (finder, {"query": query} if finder == "find" else {})

    return [
        lambda _: response(look("documentation link")),
        lambda r: response(("click", {"id": element_id(r, "Documentation")})),
        lambda _: response(look("install guide link")),
        lambda r: response(("click", {"id": element_id(r, "Install guide")})),
        lambda _: response(("done", {"answer": "pip install acme"})),
    ]


# ---- tasks ----------------------------------------------------------------------------------------------


def test_tasks_yaml_is_valid() -> None:
    tasks = load_tasks()
    assert len(tasks) >= 20
    assert len({t.id for t in tasks}) == len(tasks)
    for t in tasks:
        assert t.start_url.startswith("https://") and t.goal and t.check and t.max_steps > 0


@pytest.mark.parametrize("patch", [
    {"check": None}, {"check": {}}, {"check": {"title_is": "x"}}, {"check": {"url_regex": "("}},
    {"start_url": "ftp://x"}, {"max_steps": 0}, {"goal": ""},
])
def test_parse_tasks_rejects_bad_task(patch: dict) -> None:
    task = {"id": "t", "start_url": "https://a.b", "goal": "g", "check": {"url_contains": "a"}, "max_steps": 3}
    with pytest.raises(ValueError):
        parse_tasks([{**task, **patch}])


def test_parse_tasks_rejects_duplicates_and_empty() -> None:
    task = {"id": "t", "start_url": "https://a.b", "goal": "g", "check": {"url_contains": "a"}, "max_steps": 3}
    with pytest.raises(ValueError):
        parse_tasks([task, task])
    with pytest.raises(ValueError):
        parse_tasks([])


def test_check_success_requires_every_condition() -> None:
    check = {"url_contains": "/docs", "url_regex": r"v\d+", "text_contains": "Hello World"}
    assert check_success(check, "https://x/docs/v3", "say hello world")
    assert not check_success(check, "https://x/docs/vX", "say hello world")
    assert not check_success(check, "https://x/docs/v3", "nothing")


# ---- the loop, one test per configuration ---------------------------------------------------------------


@pytest.mark.parametrize("config, finder", [("full", "snapshot"), ("top20", "snapshot"), ("find", "find")])
def test_full_loop(site: str, config: str, finder: str) -> None:
    planner = ScriptedPlanner(*navigate_script(finder))
    result, events = run(make_task(site), config, planner)

    assert result.success and result.done and result.stop == ""
    assert result.answer == "pip install acme"
    assert result.final_url.endswith("/install.html")
    assert (result.calls, result.steps) == (5, 2)
    assert (result.input_tokens, result.output_tokens) == (500, 50)
    assert (result.cache_read_tokens, result.cache_creation_tokens, result.tokens_in) == (250, 25, 775)
    assert result.seconds > 0
    offered = planner.requests[0]["tools"]
    assert offered == (["find"] if config == "find" else ["snapshot"]) + ["click", "type", "go_back", "done"]
    assert [e["type"] for e in events].count("response") == 5
    assert [e["name"] for e in events if e["type"] == "tool"] == [
        finder, "click", finder, "click", "done"]
    assert all(e["usage"]["input_tokens"] == 100 for e in events if e["type"] == "response")


def test_text_check_on_final_page(site: str) -> None:
    result, _ = run(make_task(site, text_contains="PIP INSTALL ACME"), "full", ScriptedPlanner(*navigate_script("snapshot")))
    assert result.success


def test_failure_when_final_page_is_wrong(site: str) -> None:
    steps = navigate_script("snapshot")[:2] + [lambda _: response(("done", {"answer": "gave up"}))]
    result, _ = run(make_task(site), "full", ScriptedPlanner(*steps))
    assert result.done and not result.success and result.final_url.endswith("/docs.html")


def test_top20_uses_goal_mode_with_history(site: str) -> None:
    seen: list[tuple] = []

    class Spy(BM25Ranker):
        def rank(self, query, elements, k=5, mode="do", history=None):
            seen.append((query, k, mode, list(history or [])))
            return super().rank(query, elements, k, mode, history)

    run(make_task(site), "top20", ScriptedPlanner(*navigate_script("snapshot")), Spy())
    assert seen[0] == ("Open the install guide in the documentation", 20, "goal", [])
    assert seen[1][3] == ["[a]  Documentation -> CLICK"]  # Mind2Web's action_reprs format


def test_find_uses_do_mode_and_shows_scores(site: str) -> None:
    seen: list[tuple] = []

    class Spy(BM25Ranker):
        def rank(self, query, elements, k=5, mode="do", history=None):
            seen.append((query, k, mode, history))
            return super().rank(query, elements, k, mode, history)

    planner = ScriptedPlanner(*navigate_script("find"))
    run(make_task(site), "find", planner, Spy())
    assert seen[0] == ("documentation link", 5, "do", None)
    first = planner.requests[1]["results"]
    assert re.search(r"^\d+\|link\|Documentation\|.*\|score=[\d.]+$", first, re.MULTILINE)
    assert len([ln for ln in first.splitlines() if "score=" in ln]) == 5


def test_snapshot_lists_every_element_in_full_config(site: str) -> None:
    planner = ScriptedPlanner(lambda _: response(("snapshot", {})), lambda _: response(("done", {"answer": ""})))
    run(make_task(site), "full", planner)
    snapshot = planner.requests[1]["results"]
    assert f"{site}/index.html" in snapshot and "title: Acme Home" in snapshot
    for label in ("Documentation", "Pricing", "Search the site", "Go", "Delete account"):
        element_id(snapshot, label)


def test_type_and_press_enter(site: str) -> None:
    planner = ScriptedPlanner(
        lambda _: response(("snapshot", {})),
        lambda r: response(("type", {"id": element_id(r, "Search the site"), "text": "install", "press_enter": True})),
        lambda _: response(("done", {"answer": ""})),
    )
    result, _ = run(make_task(site, url_contains="search.html?q=install"), "full", planner)
    assert result.success and result.steps == 1


def test_typing_replaces_existing_text(site: str) -> None:
    box: dict[str, int] = {}

    def type_draft(snapshot: str) -> Any:
        box["id"] = element_id(snapshot, "Search the site")
        return response(("type", {"id": box["id"], "text": "draft"}))

    planner = ScriptedPlanner(
        lambda _: response(("snapshot", {})),
        type_draft,
        lambda _: response(("type", {"id": box["id"], "text": "install", "press_enter": True})),
        lambda _: response(("done", {"answer": ""})),
    )
    result, _ = run(make_task(site, url_regex=r"search\.html\?q=install$"), "full", planner)
    assert result.success


def test_typing_into_a_button_is_an_error(site: str) -> None:
    planner = ScriptedPlanner(
        lambda _: response(("snapshot", {})),
        lambda r: response(("type", {"id": element_id(r, "Go"), "text": "x"})),
        lambda _: response(("done", {"answer": ""})),
    )
    _, events = run(make_task(site), "full", planner)
    tool = [e for e in events if e["type"] == "tool" and e["name"] == "type"][0]
    assert tool["is_error"] and "not a text field" in tool["result"]


def test_go_back(site: str) -> None:
    script = navigate_script("snapshot")[:2] + [
        lambda _: response(("go_back", {})), lambda _: response(("done", {"answer": ""}))]
    result, _ = run(make_task(site, url_contains="index.html"), "full", ScriptedPlanner(*script))
    assert result.success and result.steps == 2


# ---- click safety ---------------------------------------------------------------------------------------


def test_risky_clicks_are_refused_with_error_results(site: str) -> None:
    planner = ScriptedPlanner(
        lambda _: response(("snapshot", {})),
        lambda r: response(("click", {"id": element_id(r, "Delete account")}), ("click", {"id": element_id(r, "Go")}),
                           ("click", {"id": element_id(r, "Join newsletter")})),
        lambda _: response(("done", {"answer": ""})),
    )
    result, events = run(make_task(site, url_contains="/index.html"), "full", planner)
    clicks = [e for e in events if e["type"] == "tool" and e["name"] == "click"]
    assert [c["is_error"] for c in clicks] == [True, True, True]
    assert "blocked word 'delete'" in clicks[0]["result"]
    assert "submits a form" in clicks[1]["result"] and "submits a form" in clicks[2]["result"]
    assert result.success and result.steps == 3  # refused clicks still spend budget; the page did not change


def test_error_flag_is_sent_back_to_the_model(site: str) -> None:
    sent: list[dict] = []

    class Recording(ScriptedPlanner):
        def create(self, system, tools, messages):
            sent.append(messages[-1])
            return super().create(system, tools, messages)

    planner = Recording(
        lambda _: response(("snapshot", {})),
        lambda r: response(("click", {"id": element_id(r, "Delete account")}), ("click", {"id": 99999})),
        lambda _: response(("done", {"answer": ""})),
    )
    run(make_task(site), "full", planner)
    results = sent[2]["content"]  # one user message carries both tool results
    assert [r["is_error"] for r in results] == [True, True]
    assert [r["tool_use_id"] for r in results] == ["toolu_0", "toolu_1"]


def test_click_before_listing_elements_is_an_error(site: str) -> None:
    planner = ScriptedPlanner(lambda _: response(("click", {"id": 0})), lambda _: response(("done", {"answer": ""})))
    _, events = run(make_task(site), "full", planner)
    assert events[1]["is_error"] and "fresh element list" in events[1]["result"]


def test_stale_ids_after_navigation_are_refused(site: str) -> None:
    planner = ScriptedPlanner(
        lambda _: response(("snapshot", {})),
        lambda r: response(("click", {"id": element_id(r, "Documentation")})),
        lambda _: response(("click", {"id": 0})),
        lambda _: response(("done", {"answer": ""})),
    )
    _, events = run(make_task(site), "full", planner)
    stale = [e for e in events if e["type"] == "tool"][2]
    assert stale["is_error"] and "changed" in stale["result"]


def test_invalid_arguments_and_unknown_tools_are_errors(site: str) -> None:
    planner = ScriptedPlanner(
        lambda _: response(("click", {}), ("find", {"query": "x"}), ("nope", {})),
        lambda _: response(("done", {"answer": ""})),
    )
    _, events = run(make_task(site), "full", planner)
    tools = [e for e in events if e["type"] == "tool"]
    assert [t["is_error"] for t in tools] == [True, True, True, False]


# ---- stop conditions ------------------------------------------------------------------------------------


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_refused_or_truncated_turn_runs_no_tools(site: str, stop_reason: str) -> None:
    planner = ScriptedPlanner(lambda _: response(("click", {"id": 0}), stop_reason=stop_reason))
    result, events = run(make_task(site), "full", planner)
    assert result.stop == stop_reason and result.steps == 0 and not result.done
    assert [e["type"] for e in events] == ["response"]


def test_end_turn_without_tool_call_stops(site: str) -> None:
    text = SimpleNamespace(content=[SimpleNamespace(type="text", text="I am done")], stop_reason="end_turn",
                           usage=SimpleNamespace(input_tokens=7, output_tokens=3))
    result, _ = run(make_task(site), "full", ScriptedPlanner(lambda _: text))
    assert (result.stop, result.answer, result.input_tokens, result.cache_read_tokens) == ("end_turn", "I am done", 7, 0)


def test_step_budget_ends_the_run(site: str) -> None:
    task = make_task(site)
    task.max_steps = 2
    planner = ScriptedPlanner(lambda _: response(("snapshot", {})), *[lambda _: response(("go_back", {}))] * 5)
    result, _ = run(task, "full", planner)
    assert result.stop == "max_steps" and result.steps == 2 and not result.done


# ---- logs, resume, report -------------------------------------------------------------------------------


def test_run_one_logs_resumes_and_reports(site: str, tmp_path: Path) -> None:
    task = make_task(site)
    results = {}
    for config, finder in (("full", "snapshot"), ("find", "find")):
        for repeat in range(2):
            planner = ScriptedPlanner(*navigate_script(finder))
            results[config, repeat] = run_one(task, config, repeat, planner, BM25Ranker(), tmp_path)
    assert all(r.success for r in results.values())
    assert run_one(task, "full", 0, ScriptedPlanner(), BM25Ranker(), tmp_path) is None  # already logged

    lines = (tmp_path / "logs" / "local__full__0.jsonl").read_text().splitlines()
    events = [json.loads(line) for line in lines]
    assert [e["type"] for e in events] == ["response", "tool", "response", "tool", "response", "tool", "response",
                                           "tool", "response", "tool", "summary"]
    assert events[0]["usage"]["cache_read_input_tokens"] == 50 and events[1]["result"].startswith("url: ")
    assert events[-1]["tokens_in"] == 775 and events[-1]["success"] is True

    runs = load_runs(tmp_path)
    assert len(runs) == 4
    table = results_table(runs)
    assert "| full | 2 | 100.0% | 775 (500 / 250 / 25) | 50 |" in table
    assert "| find | 2 | 100.0% | 775 (500 / 250 / 25) | 50 |" in table
    assert "| local | 2/2 | 2/2 |" in table
    assert write_results(tmp_path) == table and (tmp_path / "results.md").read_text() == table + "\n"


def test_failed_run_is_recorded_and_counted(site: str, tmp_path: Path) -> None:
    class Broken:
        def create(self, *args: Any) -> Any:
            raise RuntimeError("api down")

    result = run_one(make_task(site), "full", 0, Broken(), BM25Ranker(), tmp_path)
    assert not result.success and result.stop == "error" and "api down" in result.error
    assert "| full | 1 | 0.0% |" in results_table(load_runs(tmp_path))


def test_unfinished_logs_are_ignored(tmp_path: Path) -> None:
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "a__full__0.jsonl").write_text('{"type": "response"}\n')
    assert load_runs(tmp_path) == []


def test_run_result_dict_has_totals() -> None:
    r = RunResult("t", "full", 0)
    r.add_usage(SimpleNamespace(input_tokens=1, output_tokens=2, cache_read_input_tokens=None, cache_creation_input_tokens=4))
    assert r.to_dict()["tokens_in"] == 5 and r.output_tokens == 2


# ---- the Claude planner request -------------------------------------------------------------------------


def test_claude_planner_request_shape() -> None:
    calls: list[dict] = []
    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: calls.append(kw) or "msg")))
    planner = ClaudePlanner(client=client)
    tools = [agent.TOOLS["click"]]
    assert planner.create("sys", tools, [{"role": "user", "content": "hi"}]) == "msg"
    kw = calls[0]
    assert kw["model"] == "claude-opus-5-5" and kw["output_config"] == {"effort": "medium"}
    assert kw["betas"] == ["server-side-fallback-2026-07-01"] and kw["fallbacks"] == "default"
    assert kw["system"] == [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}]
    assert kw["tools"] == tools
    assert not {"temperature", "top_p", "thinking", "tool_choice"} & set(kw)


def test_unknown_config_is_rejected(site: str) -> None:
    with pytest.raises(ValueError):
        agent.ToolBox("nope", None, BM25Ranker(), "goal")  # type: ignore[arg-type]
