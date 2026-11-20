"""Benchmark coding-agent CLIs (Claude Code, Codex) on the demo tasks, with our MCP browser server as their only tool.

    uv run --extra demo python -m demo.cli_bench --out runs/cli \
        --agents claude:claude-sonnet-5-5 codex:gpt-6.2-sol --configs full a11y vision top20 find \
        --repeats 2 --parallel 2 --model-dir dist/s-s0

Agents run through the subscriptions of the locally logged-in CLIs (no API keys). Per run, `demo.mcp_server` starts as
a stdio MCP server inside the agent CLI, owns one browser for the task and writes `OUT/logs/<run>.server.jsonl`;
the agent CLI's own output goes to `OUT/logs/<run>.agent.jsonl`; `OUT/logs/<run>.json` is the merged run record.
Success, tool calls and context volume come from the server log, never from the agent's claims.

A run whose `.json` exists is skipped, so an interrupted sweep resumes (delete the file to repeat a run).

Lock-down. Claude: `--tools ""` removes every built-in tool, `--strict-mcp-config` loads only our server,
`--setting-sources ""` skips user/project settings and CLAUDE.md, no skills, empty temp cwd.
Codex: `--ignore-user-config` (no user plugins/MCP), read-only sandbox, empty temp cwd, web search, shell, browser,
computer-use, plugins, apps and sub-agents switched off. (Codex still offers a JavaScript `exec` sandbox with no fetch,
no require and no filesystem; probed, not removable.) Every run records the tools the agent
used (`agent_tools`) and lists any outside our `ui` MCP server as `foreign_tools`.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from demo.browser import Task, load_tasks
from demo.mcp_server import BENCH_CONFIGS, CONFIG_HINTS, SERVER_NAME, tools_for

REPO = Path(__file__).resolve().parent.parent
DEFAULT_TIMEOUT_S = 360
LOGS_DIR = "logs"
FOREIGN_NOTE = "foreign_tools"


@dataclass(frozen=True)
class Agent:
    kind: str  # "claude" | "codex"
    model: str

    @property
    def name(self) -> str:
        return f"{self.kind}:{self.model}"


def parse_agent(spec: str) -> Agent:
    kind, _, model = spec.partition(":")
    if kind not in ("claude", "codex") or not model:
        raise ValueError(f"agent must look like claude:<model> or codex:<model>, got {spec!r}")
    return Agent(kind, model)


def slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "-" for c in text)


def run_id(agent: Agent, config: str, task_id: str, repeat: int) -> str:
    return f"{slug(agent.name)}__{slug(config)}__{task_id}__{repeat}"


def build_prompt(task: Task, config: str) -> str:
    tools = ", ".join(tools_for(config))
    return (
        f"Goal: {task.goal}\n\n"
        f"A web browser is already open on {task.start_url}. You can only act through the browser tools provided by "
        f"the `{SERVER_NAME}` MCP server ({tools}); do not use any other tool, shell, file or web access, and do not "
        "answer from memory. Some clicks are refused for safety (buttons that submit forms, or that mention paying, "
        "buying, deleting, sending...); pick another way. You can only read and navigate; never log in or change "
        "anything.\n\n"
        f"{CONFIG_HINTS[config]}\n\n"
        "Work efficiently. When the goal is reached, you MUST call done(answer) with a short answer; if the goal cannot "
        "be reached, call done with the reason."
    )


def server_argv(task: Task, config: str, ranker: str, model_dir: Path, log: Path, max_calls: int, observe: bool) -> list[str]:
    argv = [
        sys.executable, "-m", "demo.mcp_server", "--config", config, "--goal", task.goal, "--start-url", task.start_url,
        "--log", str(log), "--check", json.dumps(task.check), "--task-id", task.id, "--ranker", ranker,
        "--model-dir", str(model_dir), "--max-calls", str(max_calls),
    ]
    return argv if observe else argv + ["--no-observe"]


def claude_command(agent: Agent, prompt: str, mcp_config: Path, effort: str | None = None) -> list[str]:
    cmd = [
        "claude", "-p", prompt, "--model", agent.model, "--output-format", "stream-json", "--verbose",
        "--mcp-config", str(mcp_config), "--strict-mcp-config",
        "--tools", "",  # no built-in tools at all: no Bash, WebFetch, WebSearch, Read, Write, Task...
        "--allowedTools", f"mcp__{SERVER_NAME}__*",
        "--permission-mode", "dontAsk",
        "--setting-sources", "", "--disable-slash-commands", "--no-session-persistence",
    ]
    return cmd + ["--effort", effort] if effort else cmd


def claude_mcp_config(argv: list[str]) -> dict[str, Any]:
    return {"mcpServers": {SERVER_NAME: {"command": argv[0], "args": argv[1:], "env": {"PYTHONPATH": str(REPO)}}}}


def toml_str(value: str) -> str:
    return json.dumps(value)  # JSON strings are valid TOML basic strings


def codex_command(agent: Agent, prompt: str, argv: list[str], workdir: Path, effort: str | None = None) -> list[str]:
    args = "[" + ", ".join(toml_str(a) for a in argv[1:]) + "]"
    cmd = [
        "codex", "exec", "--json", "-m", agent.model, "-s", "read-only", "--skip-git-repo-check", "--ephemeral",
        "--ignore-user-config", "--ignore-rules", "-C", str(workdir),
        "-c", f"mcp_servers.{SERVER_NAME}.command={toml_str(argv[0])}",
        "-c", f"mcp_servers.{SERVER_NAME}.args={args}",
        "-c", f"mcp_servers.{SERVER_NAME}.env={{PYTHONPATH={toml_str(str(REPO))}}}",
        "-c", f"mcp_servers.{SERVER_NAME}.default_tools_approval_mode=\"approve\"",
        "-c", f"mcp_servers.{SERVER_NAME}.tool_timeout_sec=120",
        "-c", 'web_search="disabled"', "-c", 'approval_policy="never"',
    ]
    for feature in ("shell_tool", "unified_exec", "apps", "plugins", "browser_use", "browser_use_external",
                    "computer_use", "in_app_browser", "view_image", "image_generation", "skill_search", "multi_agent",
                    "multi_agent_v2", "goals", "sleep_tool"):
        cmd += ["--disable", feature]
    if effort:
        cmd += ["-c", f"model_reasoning_effort={toml_str(effort)}"]
    return cmd + [prompt]


# ---- parsing the agents' own output -----------------------------------------------------------------------------


def parse_claude(stdout: str) -> dict[str, Any]:
    """Fields of Claude Code's `--output-format stream-json --verbose` events: the final result's usage / turns /
    cost, the tools offered in the init event and every tool the model called."""
    result: dict[str, Any] = {}
    offered: list[str] = []
    used: set[str] = set()
    seen_chars = oversized = 0
    for line in stdout.splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "system" and ev.get("subtype") == "init":
            offered = list(ev.get("tools") or [])
        elif ev.get("type") == "assistant":
            for block in (ev.get("message") or {}).get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    used.add(str(block.get("name")))
        elif ev.get("type") == "user":
            for block in (ev.get("message") or {}).get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    content = block.get("content")
                    text = content if isinstance(content, str) else "".join(
                        b.get("text", "") for b in content or [] if isinstance(b, dict))
                    seen_chars += len(text)
                    # Claude Code refuses / persists to disk MCP results over its token cap; the model sees a stub.
                    oversized += text.startswith(("<persisted-output>", "Error: result ("))
        elif ev.get("type") == "result":
            result = ev
    extra = {"model_visible_chars": seen_chars, "oversized_results": oversized}
    if not result:
        return {"agent_error": "no result event", "agent_tools": sorted(used), "tools_offered": offered, **extra}
    usage = result.get("usage") or {}
    in_new, cache_r, cache_w = (usage.get(k, 0) or 0 for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
    return {
        "input_tokens": in_new + cache_r + cache_w, "uncached_input_tokens": in_new, "cache_read_tokens": cache_r,
        "cache_creation_tokens": cache_w, "output_tokens": usage.get("output_tokens", 0) or 0,
        "turns": result.get("num_turns"), "cost_usd": result.get("total_cost_usd"),
        "agent_duration_ms": result.get("duration_ms"),
        "agent_error": str(result.get("result", ""))[:300] if result.get("is_error") else "",
        "agent_tools": sorted(used), "tools_offered": offered, **extra,
    }


def parse_codex(stdout: str) -> dict[str, Any]:
    """Fields of `codex exec --json` events: token usage, turns and every tool item the agent used."""
    tokens = {"input_tokens": 0, "cached_input_tokens": 0, "cache_write_input_tokens": 0, "output_tokens": 0,
              "reasoning_output_tokens": 0}
    turns, tools, error = 0, set(), ""
    for line in stdout.splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = ev.get("type", "")
        if kind == "turn.completed":
            turns += 1
            for key in tokens:
                tokens[key] += (ev.get("usage") or {}).get(key, 0) or 0
        elif kind in ("error", "turn.failed"):
            error = str(ev.get("message") or ev.get("error") or ev)[:300]
        item = ev.get("item") or {}
        if kind == "item.completed" and item.get("type") not in (None, "agent_message", "reasoning", "error"):
            if item.get("type") == "mcp_tool_call":
                tools.add(f"mcp:{item.get('server')}.{item.get('tool')}")
            else:
                tools.add(str(item.get("type")))
    return {
        "input_tokens": tokens["input_tokens"], "uncached_input_tokens": tokens["input_tokens"] - tokens["cached_input_tokens"],
        "cache_read_tokens": tokens["cached_input_tokens"], "cache_creation_tokens": tokens["cache_write_input_tokens"],
        "output_tokens": tokens["output_tokens"], "reasoning_tokens": tokens["reasoning_output_tokens"],
        "turns": turns, "cost_usd": None, "agent_error": error, "agent_tools": sorted(tools),
    }


def foreign(agent_tools: list[str]) -> list[str]:
    """Tools the agent used that are not our MCP server's."""
    ours = (f"mcp__{SERVER_NAME}__", f"mcp:{SERVER_NAME}.")
    return [t for t in agent_tools if not t.startswith(ours)]


def read_server_log(path: Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """(summary or None, call events) of a server log."""
    summary, calls = None, []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") == "summary":
                summary = rec
            elif rec.get("type") == "call":
                calls.append(rec)
    return summary, calls


# ---- running -----------------------------------------------------------------------------------------------------


def kill_group(proc: subprocess.Popen) -> None:
    """SIGTERM the agent's process group (so the MCP server writes its summary), then SIGKILL stragglers."""
    for sig, wait in ((signal.SIGTERM, 8), (signal.SIGKILL, 3)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


def kill_server(server_log: Path | None) -> None:
    """Kill the run's MCP server by its unique log path: Codex starts it in its own session, out of our group."""
    if server_log is not None:
        subprocess.run(["pkill", "-f", str(server_log)], check=False)


def run_agent(cmd: list[str], cwd: Path, timeout: int, env: dict[str, str],
              server_log: Path | None = None) -> tuple[str, str, int | None, bool]:
    """(stdout, stderr, returncode, timed_out). Runs in its own process group; the group and the MCP server are
    killed on timeout, and reading the output after that is bounded too (a stray child may hold the pipes)."""
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        out, err = proc.communicate(timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        kill_group(proc)
        kill_server(server_log)
        try:
            out, err = proc.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = "", "output unreadable after timeout"
        timed_out = True
    else:
        try:  # stray children (the MCP server) must not outlive the run
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        kill_server(server_log)
    return out or "", err or "", proc.returncode, timed_out


def run_path(out: Path, rid: str) -> Path:
    return out / LOGS_DIR / f"{rid}.json"


def execute(agent: Agent, task: Task, config: str, repeat: int, out: Path, ranker: str, model_dir: Path,
            timeout: int, max_calls: int, observe: bool, effort: str | None) -> dict[str, Any]:
    """Run one (agent, task, config, repeat) and return its merged record (also written to OUT/logs)."""
    rid = run_id(agent, config, task.id, repeat)
    logs = out / LOGS_DIR
    logs.mkdir(parents=True, exist_ok=True)
    server_log = (logs / f"{rid}.server.jsonl").resolve()
    server_log.unlink(missing_ok=True)
    argv = server_argv(task, config, ranker, model_dir.resolve(), server_log, max_calls, observe)
    prompt = build_prompt(task, config)
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="cli-bench-") as tmp:
        workdir = Path(tmp)
        if agent.kind == "claude":
            cfg = workdir.parent / f"{workdir.name}.mcp.json"
            cfg.write_text(json.dumps(claude_mcp_config(argv)), encoding="utf-8")
            cmd = claude_command(agent, prompt, cfg, effort)
        else:
            cmd = codex_command(agent, prompt, argv, workdir, effort)
        stdout, stderr, code, timed_out = run_agent(cmd, workdir, timeout, env, server_log)
        if agent.kind == "claude":
            cfg.unlink(missing_ok=True)
    wall = time.perf_counter() - started
    (logs / f"{rid}.agent.jsonl").write_text(stdout, encoding="utf-8")
    parsed = parse_claude(stdout) if agent.kind == "claude" else parse_codex(stdout)
    summary, calls = read_server_log(server_log)
    if summary is None:  # the server died without writing one (e.g. never started)
        summary = {"success": False, "done_called": False, "final_url": "", "calls": len(calls), "steps": 0,
                   "tool_chars": sum(c.get("chars", 0) for c in calls), "image_bytes": 0, "images": 0, "errors": 0}
        parsed["agent_error"] = parsed.get("agent_error") or f"no server summary; stderr: {stderr.strip()[-300:]}"
    tools_used = parsed.pop("agent_tools", [])
    record = {
        "agent": agent.name, "agent_kind": agent.kind, "model": agent.model, "config": config, "task_id": task.id,
        "repeat": repeat, "ranker": ranker, "success": bool(summary["success"]), "done_called": bool(summary["done_called"]),
        "answer": summary.get("answer", ""), "final_url": summary["final_url"], "wall_seconds": round(wall, 1),
        "tool_calls": summary["calls"], "steps": summary["steps"], "tool_errors": summary.get("errors", 0),
        "tool_chars": summary["tool_chars"], "image_bytes": summary.get("image_bytes", 0), "images": summary.get("images", 0),
        "timed_out": timed_out, "returncode": code, "agent_tools": tools_used, FOREIGN_NOTE: foreign(tools_used),
        **parsed,
    }
    run_path(out, rid).write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    return record


def load_runs(out: str | Path) -> list[dict[str, Any]]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((Path(out) / LOGS_DIR).glob("*.json"))]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--agents", nargs="+", required=True, help="claude:<model> / codex:<model>")
    p.add_argument("--configs", nargs="+", choices=BENCH_CONFIGS, default=list(BENCH_CONFIGS))
    p.add_argument("--tasks", nargs="+", help="task ids (default: all)")
    p.add_argument("--tasks-file")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--parallel", type=int, default=1)
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S, help="seconds per run")
    p.add_argument("--ranker", choices=["grounder", "bm25"], default="grounder")
    p.add_argument("--model-dir", default="dist/s-s0")
    p.add_argument("--max-calls", type=int, default=60, help="tool-call budget enforced by the server")
    p.add_argument("--no-observe", action="store_true", help="a11y / vision actions return no new observation")
    p.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"], help="reasoning effort, passed to the CLI")
    args = p.parse_args(argv)

    agents = [parse_agent(a) for a in args.agents]
    for agent in agents:
        if shutil.which(agent.kind) is None:
            raise SystemExit(f"`{agent.kind}` CLI not found on PATH")
    tasks = load_tasks(args.tasks_file) if args.tasks_file else load_tasks()
    if args.tasks:
        unknown = set(args.tasks) - {t.id for t in tasks}
        if unknown:
            raise SystemExit(f"unknown task ids: {', '.join(sorted(unknown))}")
        tasks = [t for t in tasks if t.id in args.tasks]
    out = Path(args.out)
    model_dir = Path(args.model_dir)
    if args.ranker == "grounder" and any(c in args.configs for c in ("top20", "find", "find+top20")) and not model_dir.exists():
        raise SystemExit(f"model dir {model_dir} not found")

    jobs = [(a, t, c, r) for t in tasks for c in args.configs for a in agents for r in range(args.repeats)
            if not run_path(out, run_id(a, c, t.id, r)).exists()]
    total = len(jobs)
    print(f"{total} runs to do", file=sys.stderr)

    def work(job: tuple[Agent, Task, str, int]) -> dict[str, Any]:
        a, t, c, r = job
        return execute(a, t, c, r, out, args.ranker, model_dir, args.timeout, args.max_calls, not args.no_observe, args.effort)

    done = 0
    with cf.ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        for fut in cf.as_completed([pool.submit(work, j) for j in jobs]):
            rec = fut.result()
            done += 1
            print(f"[{done}/{total}] {rec['agent']} {rec['config']} {rec['task_id']} #{rec['repeat']}: "
                  f"{'ok' if rec['success'] else 'FAIL'} calls={rec['tool_calls']} chars={rec['tool_chars']} "
                  f"in={rec.get('input_tokens')} {rec['wall_seconds']}s {rec.get('agent_error', '')[:80]}"
                  f"{' FOREIGN=' + ','.join(rec[FOREIGN_NOTE]) if rec[FOREIGN_NOTE] else ''}", file=sys.stderr)
    from demo.cli_report import write_results

    print(write_results(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
