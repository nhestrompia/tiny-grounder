"""Let an interactive agent (e.g. a Devin session) be the benchmark agent: one task, one arm, the real MCP server.

    uv run python -m tools.drive start --tasks-file demo/tasks_v4.yaml --task tn-search-imac --arm agent-browser \
        --out runs/devin-lite --agent devin:lite
    uv run python -m tools.drive call snapshot '{}'
    uv run python -m tools.drive call click '{"ref": "@e3"}'
    uv run python -m tools.drive stop

`start` prints the same prompt and tool list `demo.cli_bench` gives Claude/Codex, then keeps the stdio MCP server
alive in a background daemon. `stop` ends the server (it writes its summary) and writes OUT/logs/<run>.json in the
cli_bench record format, so `demo.cli_report` / `tools.competitor_report` read it unchanged. Success, calls and
tokens come from the server log. `wall_seconds` includes the driving agent's thinking time between calls.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from demo.browser import load_tasks
from demo.cli_bench import (LOGS_DIR, REPO, Agent, build_prompt, parse_arm, read_server_log, resolve_ranker_spec,
                            run_id, run_overhead, run_path, server_argv, site_of, task_har)
from demo.mcp_server import DEFAULT_MAX_CALLS, DEFAULT_MODEL_DIR

STATE = REPO / ".cache" / "drive" / "current.json"


def send(sock_path: str, request: dict, timeout: float = 300) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect(sock_path)
        s.sendall((json.dumps(request) + "\n").encode())
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(1 << 16)
            if not chunk:
                break
            buf += chunk
    return json.loads(buf or b"{}")


def render(result) -> str:
    parts = []
    for c in result.content:
        if c.type == "text":
            parts.append(c.text)
        elif c.type == "image":
            parts.append(f"[image {c.mimeType}]")
        else:
            parts.append(f"[{c.type}]")
    return ("ERROR: " if result.isError else "") + "\n".join(parts)


async def daemon(state: dict) -> None:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    argv = state["argv"]
    params = StdioServerParameters(command=argv[0], args=argv[1:], cwd=state["workdir"],
                                   env={**os.environ, "PYTHONPATH": str(REPO)})
    done = asyncio.Event()
    async with stdio_client(params) as streams, ClientSession(*streams) as client:
        await client.initialize()
        tools = (await client.list_tools()).tools
        listing = [{"name": t.name, "description": t.description, "input_schema": t.inputSchema} for t in tools]
        lock = asyncio.Lock()

        async def handle(reader, writer):
            req = json.loads(await reader.readline())
            async with lock:
                if req["op"] == "tools":
                    resp = {"tools": listing}
                elif req["op"] == "call":
                    try:
                        resp = {"text": render(await client.call_tool(req["name"], req.get("args") or {}))}
                    except Exception as exc:  # noqa: BLE001 - reported to the driving agent
                        resp = {"text": f"ERROR: {exc}"}
                else:
                    resp = {"ok": True}
                    done.set()
            writer.write((json.dumps(resp) + "\n").encode())
            await writer.drain()
            writer.close()

        os.umask(0o077)
        server = await asyncio.start_unix_server(handle, path=state["sock"])
        Path(state["ready"]).write_text("1")
        async with server:
            await done.wait()


def finish(state: dict, stopped: float) -> dict:
    task, agent = state["task"], Agent(*state["agent"].split(":", 1))
    log = Path(state["server_log"])
    summary, calls = read_server_log(log)
    error = None
    if summary is None:
        summary = {"success": False, "done_called": False, "final_url": "", "calls": len(calls), "steps": 0,
                   "tool_chars": sum(c.get("chars", 0) for c in calls), "image_bytes": 0, "images": 0, "errors": 0}
        error = "no server summary"
    wall = stopped - state["started"]
    image_tokens = summary.get("image_tokens_est") or sum(
        -(-c["width"] * c["height"] // 750) for c in calls if c.get("width") and c.get("height"))
    record = {
        "agent": agent.name, "agent_kind": agent.kind, "model": agent.model, "config": state["config"],
        "arm": state["arm"], "task_id": task["id"], "site": task["site"], "tags": task["tags"], "repeat": state["repeat"],
        "ranker": state["ranker"], "success": bool(summary["success"]),
        "check_ok": bool(summary.get("check_ok", summary["success"])), "answer_ok": bool(summary.get("answer_ok", True)),
        "done_called": bool(summary["done_called"]), "replay": bool(summary.get("replay")),
        "replay_requests": summary.get("replay_requests", 0), "replay_misses": summary.get("replay_misses", 0),
        "replay_network_failures": summary.get("replay_network_failures", 0),
        "replay_live_requests": summary.get("replay_live_requests", 0),
        **({"competitor": summary["competitor"]} if "competitor" in summary else {}),
        "tool_tokens_est": -(-summary["tool_chars"] // 4) + image_tokens,
        "answer": summary.get("answer", ""), "final_url": summary["final_url"], "wall_seconds": round(wall, 1),
        **run_overhead(summary, calls, wall),
        "grounding_ms": sum(c.get("extract_ms", 0) + c.get("rank_ms", 0) for c in calls),
        "tool_calls": summary["calls"], "steps": summary["steps"], "tool_errors": summary.get("errors", 0),
        "tool_chars": summary["tool_chars"], "image_bytes": summary.get("image_bytes", 0),
        "images": summary.get("images", 0), "timed_out": False, "returncode": 0, "agent_tools": [],
        "foreign_tools": [], "agent_error": error, "driver": "tools.drive",
    }
    run_path(Path(state["out"]), state["rid"]).write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    return record


def start(args) -> int:
    if STATE.exists():
        print(f"a run is active ({json.loads(STATE.read_text())['rid']}); call `stop` first", file=sys.stderr)
        return 1
    task = next((t for t in load_tasks(args.tasks_file) if t.id == args.task), None)
    if task is None:
        print(f"unknown task {args.task}", file=sys.stderr)
        return 1
    arm = parse_arm(args.arm, "exact")
    agent = Agent(*args.agent.split(":", 1))
    out = Path(args.out).resolve()
    rid = run_id(agent, arm.config, task.id, args.repeat, arm.label)
    logs = out / LOGS_DIR
    logs.mkdir(parents=True, exist_ok=True)
    if run_path(out, rid).exists():
        print(f"{rid} already recorded; delete {run_path(out, rid)} to repeat it", file=sys.stderr)
        return 1
    server_log = (logs / f"{rid}.server.jsonl").resolve()
    server_log.unlink(missing_ok=True)
    har = task_har(task, live=False)
    if har is not None and not har.exists():
        print(f"snapshot {har} missing", file=sys.stderr)
        return 1
    argv = server_argv(task, arm.config, resolve_ranker_spec(arm.ranker or "exact"), (REPO / DEFAULT_MODEL_DIR).resolve(),
                       server_log, args.max_calls, True, har)
    run_dir = STATE.parent / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    state = {"rid": rid, "argv": argv, "workdir": str(run_dir), "sock": f"/tmp/tg-drive-{hashlib.sha1(rid.encode()).hexdigest()[:12]}.sock",
             "ready": str(run_dir / "ready"), "server_log": str(server_log), "out": str(out), "config": arm.config,
             "arm": arm.label, "ranker": arm.ranker or "exact", "repeat": args.repeat, "agent": args.agent,
             "task": {"id": task.id, "site": task.site or site_of(task.start_url), "tags": list(task.tags)},
             "started": time.time()}
    for p in (state["sock"], state["ready"]):
        Path(p).unlink(missing_ok=True)
    STATE.write_text(json.dumps(state))
    with open(run_dir / "daemon.log", "w") as err:
        subprocess.Popen([sys.executable, "-m", "tools.drive", "_daemon"], cwd=REPO, stdin=subprocess.DEVNULL,
                         stdout=err, stderr=err, start_new_session=True, env={**os.environ, "PYTHONPATH": str(REPO)})
    deadline = time.time() + 120
    while not Path(state["ready"]).exists():
        if time.time() > deadline:
            STATE.unlink(missing_ok=True)
            print(f"server did not start; see {run_dir / 'daemon.log'}", file=sys.stderr)
            return 1
        time.sleep(0.2)
    state["started"] = time.time()
    STATE.write_text(json.dumps(state))
    print(f"run: {rid}\n\n=== PROMPT ===\n{build_prompt(task, arm.config)}\n\n=== TOOLS ===")
    print(json.dumps(send(state["sock"], {"op": "tools"})["tools"], indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start")
    s.add_argument("--tasks-file", default="demo/tasks_v4.yaml")
    s.add_argument("--task", required=True)
    s.add_argument("--arm", required=True, help="config or config@ranker, as in demo.cli_bench --arms")
    s.add_argument("--out", default="runs/devin")
    s.add_argument("--agent", default="devin:lite", help="label recorded as the agent, kind:model")
    s.add_argument("--repeat", type=int, default=0)
    s.add_argument("--max-calls", type=int, default=DEFAULT_MAX_CALLS)
    c = sub.add_parser("call")
    c.add_argument("name")
    c.add_argument("args", nargs="?", default="{}", help="tool arguments as a JSON object")
    sub.add_parser("tools")
    sub.add_parser("stop")
    sub.add_parser("_daemon")
    args = p.parse_args(argv)
    if args.cmd == "start":
        return start(args)
    if not STATE.exists():
        print("no active run; call `start` first", file=sys.stderr)
        return 1
    state = json.loads(STATE.read_text())
    if args.cmd == "_daemon":
        asyncio.run(daemon(state))
        return 0
    if args.cmd == "tools":
        print(json.dumps(send(state["sock"], {"op": "tools"})["tools"], indent=1))
    elif args.cmd == "call":
        print(send(state["sock"], {"op": "call", "name": args.name, "args": json.loads(args.args)})["text"])
    else:
        stopped = time.time()
        try:
            send(state["sock"], {"op": "stop"}, timeout=60)
        except OSError:
            pass
        deadline = time.time() + 30
        while time.time() < deadline and read_server_log(Path(state["server_log"]))[0] is None:
            time.sleep(0.3)
        record = finish(state, stopped)
        STATE.unlink(missing_ok=True)
        keys = ("task_id", "arm", "success", "check_ok", "answer_ok", "tool_calls", "steps", "tool_tokens_est",
                "replay_live_requests", "agent_error")
        print(json.dumps({k: record[k] for k in keys}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
