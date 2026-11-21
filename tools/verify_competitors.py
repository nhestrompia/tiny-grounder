"""No-LLM checks through real pinned tools on five frozen milestone paths.

uv run python -m tools.verify_competitors
"""
from __future__ import annotations
import argparse
import asyncio
import json
from pathlib import Path
import re
import math
from types import SimpleNamespace

from demo.browser import parse_tasks
from demo.competitors import competitor_session, MiddlewareSession
from demo.mcp_server import BenchServer, SpecRanker
from demo.competitor_specs import COMPETITOR_CONFIGS
from tools.bench_v2 import load_raw, har_path, locate

TASK_IDS = ("open-mpi-index-3", "asciidoctor-document-attributes-ref", "fastapi-tutorial-devserver", "fastapi-async-explainer", "fastapi-deploy-vague")


def text(result):
    if result.isError:
        raise RuntimeError(str(result.content))
    return "\n".join(c.text for c in result.content if c.type == "text")


async def verify(raw, arm, logs):
    task = parse_tasks([raw])[0]
    log = logs / f"{task.id}-{arm}.jsonl"
    log.unlink(missing_ok=True)
    args = SimpleNamespace(config=arm, start_url=task.start_url, har=har_path(task), replay_miss="stub", goal=task.goal,
                           check=task.check, log=log, task_id=task.id, max_calls=200, answer_regex=task.answer_regex)
    record = dict(task=task.id, arm=arm, success=False)
    try:
        async with competitor_session(args) as proxy:
            record["exposed_tools"] = sorted(proxy.allowed)
            async def call(name, **kwargs):
                return text(await proxy.call(name, kwargs))
            observation = ("snapshot", {}) if arm == "agent-browser" else (("observe", {}) if arm == "pw-mcp+observe" else ("browser_snapshot", {"depth": 3} if arm == "pw-mcp-compact" else {}))
            first = await call(observation[0], **observation[1])
            first_event = json.loads(log.read_text().splitlines()[-1])
            record["first_observation_tokens"] = -(-first_event["chars"] // 4) + first_event.get("image_tokens_est", 0)
            record["path"] = []
            for step in raw["path"]:
                # The oracle supplies only the next label. The action uses a ref/label actually exposed by the arm.
                def label():
                    target = locate(proxy.bench.session.page, step["click"])
                    return " ".join((target.inner_text().strip() or target.get_attribute("aria-label")).split())
                name = await proxy.owner(label)
                if arm == "pw-mcp+observe":
                    found = await call("find", query=name)
                    if name.casefold() not in found.casefold():
                        raise ValueError(f"label absent from find: {name}")
                    candidates = [line for line in found.splitlines() if line.startswith('tg-') and name.casefold() in line.casefold()]
                    if not candidates:
                        raise ValueError(f'no middleware id for {name}')
                    identifier = candidates[0].split('|')[0]
                    # Check the exact oracle node, including duplicate labels, before forwarding its public id.
                    selector = await proxy.owner(proxy.bench.session.middleware_target, identifier)
                    same = await proxy.owner(lambda: locate(proxy.bench.session.page, step['click']).evaluate(
                        '(el, selector) => el === document.querySelector(selector)', selector))
                    if not same:
                        raise ValueError(f'middleware id selected a different duplicate: {name}')
                    await call('browser_click', target=identifier, element=name)
                else:
                    if arm == "pw-mcp-compact":
                        snapshot = await call("browser_find", text=name)
                    else:
                        snapshot = first
                    candidates = [line for line in snapshot.splitlines() if any(f'{role} "{name}"'.casefold() in line.casefold() for role in ("link", "button")) and re.search(r'\[ref=([a-z0-9]+)\]', line)]
                    if not candidates and arm != "agent-browser":
                        snapshot = await call("browser_find", text=name)
                        candidates = [line for line in snapshot.splitlines() if any(f'{role} "{name}"'.casefold() in line.casefold() for role in ("link", "button")) and re.search(r'\[ref=([a-z0-9]+)\]', line)]
                    if not candidates:
                        raise ValueError(f"label absent from snapshot: {name}: {snapshot[:5000]}")
                    ref = re.search(r'\[ref=([a-z0-9]+)\]', candidates[0]).group(1)
                    await call("click" if arm == "agent-browser" else "browser_click", **({"ref": "@"+ref} if arm == "agent-browser" else {"target": ref, "element": name}))
                event = json.loads(log.read_text().splitlines()[-1])
                # Navigation cost includes the action response and an explicit view when actions lack one.
                cost = -(-event["chars"] // 4) + event.get("image_tokens_est", 0)
                if arm in ("agent-browser", "pw-mcp-compact"):
                    first = await call(observation[0], **observation[1])
                    event = json.loads(log.read_text().splitlines()[-1])
                    cost += -(-event["chars"] // 4) + event.get("image_tokens_est", 0)
                else:
                    # Use the action snapshot already logged and returned, with no unmeasured call.
                    events = [json.loads(line) for line in log.read_text().splitlines()]
                    first = "\n".join(c["text"] for c in events[-1]["forwarded_result"]["content"] if c["type"] == "text")
                record["path"].append(dict(label=name, navigation_tokens=cost))
            if task.id == "open-mpi-index-3":
                # A missing navigation must hit the local stub, including in a tool-created tab.
                missing = "https://competitor-replay.invalid/unrecorded"
                final_url = await proxy.owner(lambda: proxy.bench.session.url)
                if arm == "agent-browser":
                    await call("open", url=missing)
                else:
                    refused = await proxy.call("browser_tabs", {"action": "new", "url": missing})
                    if not refused.isError:
                        raise ValueError("single-tab guard did not reject new tab")
                    await call("browser_evaluate", function=f"() => {{ window.open({json.dumps(missing)}); }}")
                stub = await proxy.owner(lambda: proxy.bench.session.text())
                if "Page not archived" not in stub:
                    raise ValueError("external navigation escaped the replay stub")
                record["isolation_probe"] = dict(stub=True, single_tab=True, popup_redirected=arm != "agent-browser")
                if arm == "agent-browser":
                    await call("open", url=final_url)
                else:
                    await call("browser_navigate", url=final_url)
            await call("done", answer=raw.get("answer", ""))
            summary = json.loads(log.read_text().splitlines()[-1])
            record.update(success=summary["success"], live_requests=summary["replay_live_requests"], network_failures=summary["replay_network_failures"], replay_requests=summary["replay_requests"], replay_misses=summary["replay_misses"], competitor=summary["competitor"], tool_calls=summary["calls"], tool_tokens_est=-(-summary["tool_chars"]//4)+summary["image_tokens_est"])
            record.update({k: summary[k] for k in ("safety_ms", "settle_ms", "proxy_ms", "harness_overhead_ms")})
            record["ok"] = record["success"] and not record["live_requests"] and not record["network_failures"]
    except Exception as exc:
        def errors(error):
            if isinstance(error, BaseExceptionGroup):
                return "; ".join(errors(e) for e in error.exceptions)
            return str(error)
        record.update(ok=False, error=errors(exc))
    return record


def verify_native(raw, logs):
    task = parse_tasks([raw])[0]
    log = logs / f"{task.id}-find+observe.jsonl"
    log.unlink(missing_ok=True)
    with MiddlewareSession(task.start_url, har=har_path(task), replay_miss="stub") as session:
        bench = BenchServer("find+observe", session, SpecRanker("exact"), task.goal, task.check, log,
                            task.id, answer_regex=task.answer_regex)
        try:
            bench.invoke("observe")
            for step in raw["path"]:
                target = locate(session.page, step["click"])
                name = " ".join((target.inner_text().strip() or target.get_attribute("aria-label")).split())
                bench.invoke("find", query=name)
                identifier = target.get_attribute("data-tg")
                element_id = next(n for n, value in session.middleware_ids.items() if value == identifier)
                bench.invoke("click", id=element_id)
            bench.invoke("done", answer=raw.get("answer", ""))
            summary = json.loads(log.read_text().splitlines()[-1])
            return dict(task=task.id, arm="find+observe", ok=summary["success"], success=summary["success"],
                        **{k: summary[k] for k in ("safety_ms", "settle_ms", "proxy_ms", "harness_overhead_ms")})
        finally:
            bench.finish("exit")


def latency(logs):
    groups = {}
    for path in sorted(logs.glob("*.jsonl")):
        events = [json.loads(line) for line in path.read_text().splitlines()]
        summary = next((e for e in reversed(events) if e.get("type") == "summary"), {})
        arm = summary.get("config", path.stem)
        for e in events:
            if e.get("type") == "call":
                groups.setdefault(arm, {}).setdefault(e["name"], []).append(e)
    def stats(events):
        def quantile(key, q):
            values = sorted(e.get(key, 0.0) for e in events)
            return round(values[max(0, math.ceil(q * len(values)) - 1)], 2)
        return dict(n=len(events), p50_ms=quantile("ms", .5), p95_ms=quantile("ms", .95),
                    safety_p95_ms=quantile("safety_ms", .95), child_p50_ms=quantile("child_ms", .5),
                    child_p95_ms=quantile("child_ms", .95))
    return {arm: {name: stats(events) for name, events in tools.items()} for arm, tools in groups.items()}


async def run(args):
    raws = {r["id"]: r for r in load_raw(args.tasks_file)}
    records = []
    logs = Path(".cache/competitors/verification")
    before = latency(Path(".cache/competitors/verification-before-parity"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for arm in args.arms:
        for task in args.tasks:
            logs = Path(".cache/competitors/verification")
            logs.mkdir(parents=True, exist_ok=True)
            record = await verify(raws[task], arm, logs)
            records.append(record)
            print(json.dumps(record), flush=True)
            args.out.write_text(json.dumps({"runs": records, "means": means(records), "latency_before": before, "latency_after": latency(logs)}, indent=2)+"\n")
    for task in args.tasks:
        record = await asyncio.to_thread(verify_native, raws[task], logs)
        records.append(record)
        print(json.dumps(record), flush=True)
    args.out.write_text(json.dumps({"runs": records, "means": means(records), "latency_before": before,
                                    "latency_after": latency(logs)}, indent=2)+"\n")
    return int(any(not r["ok"] for r in records))


def means(records):
    out = {}
    for arm in dict.fromkeys(r["arm"] for r in records):
        rows = [r for r in records if r["arm"] == arm and r.get("ok") and "path" in r]
        nav = [p["navigation_tokens"] for r in rows for p in r["path"]]
        out[arm] = dict(tasks=len(rows), first_observation_tokens=sum(r["first_observation_tokens"] for r in rows)/len(rows) if rows else None,
                        navigation_tokens=sum(nav)/len(nav) if nav else None)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks-file", type=Path, default=Path("demo/tasks_milestone.yaml"))
    p.add_argument("--out", type=Path, default=Path("data/bench_v3/verify_competitors.json"))
    p.add_argument("--tasks", nargs="+", default=TASK_IDS)
    p.add_argument("--arms", nargs="+", choices=COMPETITOR_CONFIGS, default=COMPETITOR_CONFIGS)
    return asyncio.run(run(p.parse_args()))

if __name__ == "__main__":
    raise SystemExit(main())
