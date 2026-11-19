"""CLI benchmark tests: MCP server handlers on the local fixture site, report aggregation, agent command lines.

No network and no agent CLI is ever started.
"""

from __future__ import annotations

import asyncio
import functools
import http.server
import json
import re
import threading
from pathlib import Path

import pytest

from demo import cli_bench, cli_report, mcp_server
from demo.agent import BM25Ranker
from demo.browser import BrowserSession, Task, ToolError
from demo.cli_bench import Agent
from demo.mcp_server import BenchServer

FIXTURES = Path(__file__).parent / "fixtures" / "demo"
CHECK = {"url_contains": "install.html"}


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


@pytest.fixture
def make(site: str, tmp_path: Path):
    """Factory for a BenchServer on the fixture home page; closes every browser it opened."""
    opened: list[BrowserSession] = []

    def factory(config: str, **kw) -> BenchServer:
        session = BrowserSession(f"{site}/index.html")
        opened.append(session)
        ranker = None if config in ("a11y", "vision") else BM25Ranker()
        return BenchServer(config, session, ranker, "Open the install guide", CHECK, tmp_path / f"{config}.jsonl", "t", **kw)

    yield factory
    for s in opened:
        s.close()


def log_events(server: BenchServer) -> list[dict]:
    path = server.log_path
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def line_id(text: str, label: str) -> int:
    m = re.search(rf"^(\d+)\|[a-z]+\|{re.escape(label)}\|", text, re.MULTILINE)
    assert m, f"{label!r} not in:\n{text}"
    return int(m.group(1))


def ref_of(snapshot: str, name: str) -> str:
    m = re.search(rf'"{re.escape(name)}"[^\n]*\[ref=(\w+)\]', snapshot)
    assert m, f"{name!r} not in:\n{snapshot}"
    return m.group(1)


# ---- server: grounder-family configs ------------------------------------------------------------------------


@pytest.mark.parametrize("config", ["full", "top20", "find", "find+top20"])
def test_grounder_configs_navigate_and_log(make, config: str) -> None:
    server = make(config)
    lookup = (lambda q: server.invoke("find", query=q)) if config == "find" else (lambda q: server.invoke("snapshot"))
    docs = server.invoke("click", id=line_id(lookup("documentation link"), "Documentation"))
    assert "docs.html" in docs
    server.invoke("click", id=line_id(lookup("install guide"), "Install guide"))
    server.invoke("done", answer="pip install acme")

    events = log_events(server)
    summary = events[-1]
    assert summary["type"] == "summary" and summary["reason"] == "done"
    assert summary["success"] is True and summary["done_called"] is True
    assert summary["final_url"].endswith("/install.html")
    assert (summary["calls"], summary["steps"]) == (5, 2)
    calls = [e for e in events if e["type"] == "call"]
    assert [c["name"] for c in calls] == [("find" if config == "find" else "snapshot"), "click"] * 2 + ["done"]
    assert summary["tool_chars"] == sum(c["chars"] for c in calls) > 0
    assert all("image_bytes" not in c for c in calls)


def test_find_config_has_no_snapshot(make) -> None:
    with pytest.raises(ToolError):
        make("find").invoke("snapshot")


def test_find_plus_top20_has_both_tools(make) -> None:
    both = make("find+top20")
    assert "elements shown" in both.invoke("snapshot") and "score=" in both.invoke("find", query="docs")


def test_find_lines_carry_scores(make) -> None:
    assert "|score=" in make("find").invoke("find", query="search box")


def test_top20_is_capped(make) -> None:
    n = make("top20").invoke("snapshot").splitlines()
    assert len(n) - 3 <= 20  # url, title, count line


def test_refusals_are_logged_as_errors_and_count_in_chars(make) -> None:
    server = make("full")
    snap = server.invoke("snapshot")
    with pytest.raises(ToolError, match="refused"):
        server.invoke("click", id=line_id(snap, "Delete account"))
    summary = (server.finish("exit"), log_events(server))[1][-1]
    err = next(e for e in log_events(server) if e.get("is_error"))
    assert err["chars"] == len(err["result"]) > 0
    assert summary["errors"] == 1 and summary["success"] is False and summary["done_called"] is False and summary["reason"] == "exit"


def test_type_and_enter_searches(make) -> None:
    server = make("full")
    box = line_id(server.invoke("snapshot"), "Search the site")
    assert "search.html" in server.invoke("type", id=box, text="acme", press_enter=True)


def test_call_budget_and_finish_once(make) -> None:
    server = make("full", max_calls=2)
    server.invoke("snapshot")
    server.invoke("snapshot")
    with pytest.raises(ToolError, match="budget"):
        server.invoke("snapshot")
    server.invoke("done", answer="gave up")  # done is always allowed
    with pytest.raises(ToolError):
        server.invoke("snapshot")
    server.finish("exit")  # second finish is a no-op
    assert sum(e["type"] == "summary" for e in log_events(server)) == 1


# ---- server: a11y ---------------------------------------------------------------------------------------------


def test_a11y_snapshot_click_by_ref_and_observation(make) -> None:
    server = make("a11y")
    snap = server.invoke("snapshot")
    assert "[ref=" in snap and "Documentation" in snap
    after = server.invoke("click", id=ref_of(snap, "Documentation"))
    assert "docs.html" in after and "Install guide" in after  # the action returns the new snapshot
    server.invoke("click", id=ref_of(after, "Install guide"))
    server.invoke("done", answer="ok")
    assert log_events(server)[-1]["success"] is True


def test_a11y_type_refusals_and_bad_refs(make) -> None:
    server = make("a11y")
    snap = server.invoke("snapshot")
    with pytest.raises(ToolError, match="refused"):
        server.invoke("click", id=ref_of(snap, "Delete account"))
    with pytest.raises(ToolError, match="submits a form"):
        server.invoke("click", id=ref_of(snap, "Go"))
    with pytest.raises(ToolError, match="not a ref"):
        server.invoke("click", id="e1; drop")
    with pytest.raises(ToolError, match="no element"):
        server.invoke("click", id="e99999")
    with pytest.raises(ToolError, match="not a text field"):
        server.invoke("type", id=ref_of(snap, "Documentation"), text="x")
    typed = server.invoke("type", id=ref_of(snap, "Search the site"), text="acme", press_enter=True)
    assert "search.html" in typed


def test_a11y_no_observe_returns_only_status(make) -> None:
    server = make("a11y", observe=False)
    snap = server.invoke("snapshot")
    out = server.invoke("click", id=ref_of(snap, "Documentation"))
    assert "[ref=" not in out and "docs.html" in out


# ---- server: vision -------------------------------------------------------------------------------------------


def test_vision_screenshot_click_type_scroll(make) -> None:
    server = make("vision")
    text, png = server.invoke("screenshot")
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and mcp_server.png_size(png) == (1280, 800)
    box = server.session.page.get_by_label("Search the site").bounding_box()
    with pytest.raises(ToolError, match="no text field"):
        server.invoke("type_text", text="x")
    server.invoke("click_xy", x=box["x"] + 5, y=box["y"] + 5)
    out, shot = server.invoke("type_text", text="acme", press_enter=True)
    assert "search.html" in out and shot[:4] == b"\x89PNG"
    server.invoke("scroll", dy=300)

    calls = [e for e in log_events(server) if e["type"] == "call"]
    shots = [c for c in calls if "image_bytes" in c]
    assert len(shots) == 4 and all((c["width"], c["height"]) == (1280, 800) for c in shots)
    server.finish("exit")
    summary = log_events(server)[-1]
    assert summary["images"] == 4 and summary["image_bytes"] == sum(c["image_bytes"] for c in shots)


def test_vision_refusals_and_bounds(make) -> None:
    server = make("vision")
    page = server.session.page
    delete = page.get_by_role("button", name="Delete account").bounding_box()
    with pytest.raises(ToolError, match="refused"):
        server.invoke("click_xy", x=delete["x"] + 5, y=delete["y"] + 5)
    go = page.get_by_role("button", name="Go").bounding_box()
    with pytest.raises(ToolError, match="submits a form"):
        server.invoke("click_xy", x=go["x"] + 3, y=go["y"] + 3)
    with pytest.raises(ToolError, match="outside"):
        server.invoke("click_xy", x=5000, y=5)
    with pytest.raises(ToolError):
        server.invoke("snapshot")


# ---- MCP surface ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("config", mcp_server.BENCH_CONFIGS)
def test_mcp_exposes_exactly_the_config_tools(config: str) -> None:
    app = mcp_server.build_mcp(config, lambda name, **kw: "ok")
    names = [t.name for t in asyncio.run(app.list_tools())]
    assert sorted(names) == sorted(mcp_server.tools_for(config))


def test_mcp_image_result_is_image_content() -> None:
    png = b"\x89PNG\r\n\x1a\n" + bytes(8) + (4).to_bytes(4, "big") + (2).to_bytes(4, "big") + bytes(16)
    app = mcp_server.build_mcp("vision", lambda name, **kw: ("status", png) if name == "screenshot" else "ok")
    content = asyncio.run(app.call_tool("screenshot", {}))
    blocks = content[0] if isinstance(content, tuple) else content
    assert [b.type for b in blocks] == ["text", "image"]


# ---- agent command lines ------------------------------------------------------------------------------------------


def task() -> Task:
    return Task("wiki-search-turing", "https://en.wikipedia.org/wiki/Main_Page", "Open the Alan Turing article.",
                {"url_contains": "/wiki/Alan_Turing"}, 8)


def test_parse_agent() -> None:
    assert cli_bench.parse_agent("codex:gpt-6.2-sol") == Agent("codex", "gpt-6.2-sol")
    for bad in ("claude", "gemini:x", "claude:"):
        with pytest.raises(ValueError):
            cli_bench.parse_agent(bad)


def test_prompt_mentions_goal_tools_and_done() -> None:
    p = cli_bench.build_prompt(task(), "find")
    assert "Alan Turing" in p and "find, click, type, go_back, done" in p and "done(answer)" in p and task().start_url in p


def test_server_argv_roundtrips_through_the_server_parser(tmp_path: Path) -> None:
    argv = cli_bench.server_argv(task(), "find+top20", "bm25", tmp_path, tmp_path / "s.jsonl", 40, observe=False)
    args = mcp_server.parse_args(argv[3:])  # drop: python -m demo.mcp_server
    assert (args.config, args.ranker, args.max_calls, args.no_observe) == ("find+top20", "bm25", 40, True)
    assert args.check == {"url_contains": "/wiki/Alan_Turing"} and args.goal == task().goal


def test_claude_command_is_locked_down(tmp_path: Path) -> None:
    cmd = cli_bench.claude_command(Agent("claude", "claude-haiku-5-5"), "PROMPT", tmp_path / "m.json", effort="low")
    assert cmd[:3] == ["claude", "-p", "PROMPT"] and "--strict-mcp-config" in cmd
    assert cmd[cmd.index("--tools") + 1] == ""  # every built-in tool removed
    assert cmd[cmd.index("--allowedTools") + 1] == "mcp__ui__*"
    assert cmd[cmd.index("--model") + 1] == "claude-haiku-5-5" and cmd[cmd.index("--effort") + 1] == "low"
    assert cmd[cmd.index("--setting-sources") + 1] == "" and "--no-session-persistence" in cmd
    cfg = cli_bench.claude_mcp_config(["/py", "-m", "demo.mcp_server", "--config", "full"])
    assert cfg["mcpServers"]["ui"]["command"] == "/py" and cfg["mcpServers"]["ui"]["args"][0] == "-m"


def test_codex_command_is_locked_down(tmp_path: Path) -> None:
    argv = ["/py", "-m", "demo.mcp_server", "--goal", 'say "hi"', "--check", '{"url_contains": "x"}']
    cmd = cli_bench.codex_command(Agent("codex", "gpt-6.1-terra"), "PROMPT", argv, tmp_path)
    assert cmd[:3] == ["codex", "exec", "--json"] and cmd[-1] == "PROMPT"
    assert cmd[cmd.index("-s") + 1] == "read-only" and "--ignore-user-config" in cmd
    assert cmd[cmd.index("-C") + 1] == str(tmp_path) and cmd[cmd.index("-m") + 1] == "gpt-6.1-terra"
    overrides = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-c"]
    assert 'web_search="disabled"' in overrides and 'mcp_servers.ui.command="/py"' in overrides
    args_line = next(o for o in overrides if o.startswith("mcp_servers.ui.args="))
    assert json.loads(args_line.split("=", 1)[1]) == argv[1:]  # TOML array == JSON array here, quotes escaped
    disabled = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--disable"]
    assert {"shell_tool", "unified_exec", "plugins", "browser_use", "computer_use"} <= set(disabled)


def test_parse_claude_stream() -> None:
    events = [
        {"type": "system", "subtype": "init", "tools": ["mcp__ui__find", "mcp__ui__done"]},
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "mcp__ui__find"},
                                                      {"type": "text", "text": "x"}]}},
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Bash"}]}},
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "content": "<persisted-output>\nOutput too large"},
            {"type": "tool_result", "content": [{"type": "text", "text": "ok"}]}]}},
        {"type": "result", "is_error": False, "num_turns": 4, "total_cost_usd": 0.0123, "duration_ms": 5000,
         "usage": {"input_tokens": 10, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 5, "output_tokens": 7}},
    ]
    got = cli_bench.parse_claude("\n".join(json.dumps(e) for e in events))
    assert (got["input_tokens"], got["uncached_input_tokens"], got["output_tokens"], got["turns"]) == (115, 10, 7, 4)
    assert got["cost_usd"] == 0.0123 and got["tools_offered"] == ["mcp__ui__find", "mcp__ui__done"]
    assert cli_bench.foreign(got["agent_tools"]) == ["Bash"]
    assert got["oversized_results"] == 1 and got["model_visible_chars"] == len("<persisted-output>\nOutput too large") + 2
    assert cli_bench.parse_claude("garbage")["agent_error"]


def test_parse_codex_events() -> None:
    events = [
        {"type": "item.completed", "item": {"type": "agent_message", "text": "hi"}},
        {"type": "item.completed", "item": {"type": "mcp_tool_call", "server": "ui", "tool": "find"}},
        {"type": "item.completed", "item": {"type": "command_execution", "command": "ls"}},
        {"type": "turn.completed", "usage": {"input_tokens": 1000, "cached_input_tokens": 600, "output_tokens": 30,
                                             "reasoning_output_tokens": 5}},
    ]
    got = cli_bench.parse_codex("\n".join(json.dumps(e) for e in events) + "\nnot json")
    assert (got["input_tokens"], got["uncached_input_tokens"], got["cache_read_tokens"], got["output_tokens"]) == (1000, 400, 600, 30)
    assert got["turns"] == 1 and got["agent_tools"] == ["command_execution", "mcp:ui.find"]
    assert cli_bench.foreign(got["agent_tools"]) == ["command_execution"]


def test_read_server_log(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    path.write_text('{"type": "call", "chars": 5}\nbroken\n{"type": "summary", "success": true}\n', encoding="utf-8")
    summary, calls = cli_bench.read_server_log(path)
    assert summary == {"type": "summary", "success": True} and len(calls) == 1
    assert cli_bench.read_server_log(tmp_path / "none.jsonl") == (None, [])


# ---- report ---------------------------------------------------------------------------------------------------------


def fake_run(agent: str, config: str, task_id: str, ok: bool, repeat: int = 0, **kw) -> dict:
    return {"agent": agent, "config": config, "task_id": task_id, "repeat": repeat, "success": ok, "done_called": ok,
            "wall_seconds": 10.0, "tool_calls": 4, "tool_chars": 1000, "image_bytes": 0, "input_tokens": 5000,
            "output_tokens": 100, "cost_usd": None, "foreign_tools": [], "timed_out": False, "agent_error": "", **kw}


def test_report_aggregates_and_writes_files(tmp_path: Path) -> None:
    runs = [
        fake_run("claude:m", "find", "t1", True, wall_seconds=10, tool_chars=100, cost_usd=0.01),
        fake_run("claude:m", "find", "t1", False, repeat=1, wall_seconds=30, tool_chars=300, cost_usd=0.03),
        fake_run("claude:m", "find", "t2", True, wall_seconds=20, tool_chars=200, cost_usd=0.02),
        fake_run("claude:m", "full", "t1", False, tool_chars=90_000),
        fake_run("codex:g", "vision", "t1", True, image_bytes=204_800),
    ]
    table = cli_report.results_table(runs)
    rows = {tuple(c.strip() for c in l.split("|")[1:3]): [c.strip() for c in l.split("|")[3:-1]] for l in table.splitlines()
            if l.startswith("| claude") or l.startswith("| codex")}
    assert rows[("claude:m", "find")][:6] == ["67% (2/3)", "2/3", "20.0", "4", "200", "0"]
    assert rows[("claude:m", "find")][-1] == "0.0200"
    assert rows[("claude:m", "full")][0] == "0% (0/1)" and rows[("claude:m", "full")][4] == "90,000"
    assert rows[("codex:g", "vision")][5] == "200"  # median image KB
    # per-task matrix: columns ordered by agent, then by the bench's config order (full before find)
    assert "| t1 | 0/1 | 1/2 | 1/1 |" in table and "| t2 | - | 1/1 | - |" in table
    assert "outside the `ui`" not in table

    (tmp_path / cli_bench.LOGS_DIR).mkdir(exist_ok=True)
    for i, r in enumerate(runs):
        (tmp_path / cli_bench.LOGS_DIR / f"{i}.json").write_text(json.dumps(r), encoding="utf-8")
    cli_report.write_results(tmp_path)
    assert (tmp_path / "results.md").read_text().startswith("| agent |")
    assert len((tmp_path / "runs.jsonl").read_text().splitlines()) == 5


def test_report_flags_foreign_tools() -> None:
    table = cli_report.results_table([fake_run("claude:m", "full", "t1", True, foreign_tools=["WebFetch"], oversized_results=2)])
    assert "WebFetch" in table and "outside the `ui`" in table and "2 oversized" in table
