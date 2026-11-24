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

from demo.browser import ToolError, PlaywrightError, parse_tasks
from demo.competitors import competitor_session, MiddlewareSession
from demo.mcp_server import BenchServer, SpecRanker
from demo.competitor_specs import COMPETITOR_CONFIGS
from tools.bench_v2 import load_raw, har_path, locate, scroll_through

TASK_IDS = ("open-mpi-index-3", "asciidoctor-document-attributes-ref", "fastapi-tutorial-devserver", "fastapi-async-explainer", "fastapi-deploy-vague")

ELEMENT_STEP = ("click", "type", "select")
ELEMENT_ROLES = ("link", "button", "textbox", "searchbox", "combobox", "checkbox", "option", "menuitem", "tab", "radio")


def text(result):
    if result.isError:
        raise RuntimeError(str(result.content))
    return "\n".join(c.text for c in result.content if c.type == "text")


def step_locator(step):
    kind = next((k for k in ELEMENT_STEP if k in step), None)
    return kind, step[kind] if kind else None


_CANDIDATES_JS = """el => {
  const out = [];
  const push = (t) => { t = (t || '').replace(/\\s+/g, ' ').trim();
                        if (t && !out.includes(t)) out.push(t); };
  push(el.getAttribute('aria-label'));
  if (el.id) { const l = document.querySelector(`label[for="${CSS.escape(el.id)}"]`);
               if (l) push(l.innerText); }
  const wrap = el.closest('label');
  if (wrap) push(wrap.innerText);
  push(el.getAttribute('placeholder'));
  push(el.innerText);
  push(el.getAttribute('title'));
  push(el.getAttribute('data-original-title'));
  push(el.getAttribute('value'));
  push(el.getAttribute('name'));
  push(el.id);
  const img = el.querySelector('img');
  if (img && img.alt) push(img.alt);
  const q = el.querySelector('i, svg, span, img, use') || el;
  const c = (getComputedStyle(q, '::before').content || '').replace(/^\\"|\\"$/g, '');
  if (c && c !== 'none') push(c);
  return out; }"""


def element_labels(target, kind):
    """Ordered candidate names for a step's element, any of which may be the
    accessible name an arm's snapshot exposes.

    Comboboxes are named by their selected option. For other elements the
    snapshot name may come from the a11y name (aria-label, associated label,
    placeholder), from visible text, from a tooltip, or — for icon-only
    controls — from the icon glyph, so all are tried in turn."""
    if kind == "select":
        return [" ".join((target.evaluate("el => (el.options[el.selectedIndex] || {text: ''}).text") or "").split()) or
                target.get_attribute("name") or ""]
    return target.evaluate(_CANDIDATES_JS) or []


def _norm(value):
    """Comparable form of a snapshot/oracle label: edges stripped, icon glyphs kept."""
    stripped = re.sub(r'^[^\w]+|[^\w]+$', '', value).casefold()
    return stripped or value.strip().casefold()


def element_label(target, kind):
    """The first (most likely) candidate name from `element_labels`."""
    labels = element_labels(target, kind)
    return labels[0] if labels else ""


def snapshot_ref(snapshot, name, kind, ordinal=0):
    """The [ref=] whose a11y node matches the oracle label, or None.

    Comboboxes carry no name in these snapshots, so a select step matches the
    combobox whose nested options include the label (each select's option set
    differs). Other steps match role lines carrying the name; when several
    elements share it (e.g. per-row 'Choose This Flight' buttons), `ordinal`
    picks the oracle's position among them — snapshot order is DOM order."""
    if kind == "select":
        lines = snapshot.splitlines()
        candidates = []
        for i, line in enumerate(lines):
            if "combobox" not in line or "[ref=" not in line:
                continue
            indent = len(line) - len(line.lstrip())
            for line2 in lines[i + 1:]:
                if len(line2) - len(line2.lstrip()) <= indent:
                    break
                if "option" in line2 and f'"{name}"' in line2:
                    candidates.append(line)
                    break
        if not candidates:
            return None
        pick = candidates[ordinal] if 0 <= ordinal < len(candidates) else candidates[0]
        return re.search(r'\[ref=([a-z0-9]+)\]', pick).group(1)
    want = _norm(name)
    candidates = []
    for line in snapshot.splitlines():
        if not re.search(r'\[ref=([a-z0-9]+)\]', line):
            continue
        for role, quoted in re.findall(r'(\w+)\s+"([^"]*)"', line):
            # Icon glyphs prefix some snapshot names (e.g. a shopping-cart button);
            # compare the text part only.
            got = _norm(quoted)
            if role in ELEMENT_ROLES and got == want:
                candidates.append(line)
                break
        else:
            # Unnamed fields render their content after the ref: `textbox [ref=..]: "1"`.
            m = re.search(r'\b(textbox|searchbox|spinbutton)\s*\[ref=([a-z0-9]+)\]\s*:\s*"([^"]*)"', line)
            if m and _norm(m.group(3)) == want:
                candidates.append(line)
    if not candidates:
        lines = snapshot.splitlines()
        for i, line in enumerate(lines):
            # An image-only link/button carries no name but shows its img's alt below it.
            if not re.match(r'\s*-?\s*(link|button)\s*\[ref=[a-z0-9]+\]\s*(\[cursor=pointer\])?\s*:?\s*$', line):
                continue
            indent = len(line) - len(line.lstrip())
            for line2 in lines[i + 1:]:
                if len(line2) - len(line2.lstrip()) <= indent:
                    break
                # Descendant node names (`img "alt"`) or trailing leaf text
                # (`strong [ref=..]: HP LP3065`) name the unnamed parent.
                m = re.search(r'"([^"]*)"', line2) or re.search(r':\s*(.+)$', line2)
                if m and _norm(m.group(1)) == want:
                    candidates.append(line)
                    break
        if not candidates and not want:
            # Fields with no accessible name at all render as bare `textbox [disabled] [ref=..]`.
            candidates = [line for line in lines
                          if re.search(r'\b(textbox|searchbox|spinbutton)\b[^"\n]*\[ref=[a-z0-9]+\]', line)
                          and not re.search(r'\w\s+"[^"]*"', line)]
    if not candidates:
        return None
    pick = candidates[ordinal] if 0 <= ordinal < len(candidates) else candidates[0]
    return re.search(r'\[ref=([a-z0-9]+)\]', pick).group(1)


def oracle_ordinal(target, name, kind):
    """The oracle element's position among DOM elements sharing its label, or -1.

    Mirrors `element_label` in-page so a step's `nth`-style intent survives the
    label-only hand-off to snapshot refs (labels, not selectors, are compared)."""
    if kind == "select":
        return target.evaluate("""(el, name) => {
          const has = (s) => [...s.options].some(o => o.text.trim() === name);
          return [...document.querySelectorAll('select')].filter(has).indexOf(el);}""", name)
    want = _norm(name)
    return target.evaluate("""(el, want) => {
      const candidates = """ + _CANDIDATES_JS + """;
      const lab = (e) => candidates(e).map(t => t.replace(/^[^\\w]+|[^\\w]+$/g, '').toLowerCase() || t.toLowerCase());
      const sel = 'a,button,input,textarea,select,summary,label,[role]';
      if (want === '')
        return [...document.querySelectorAll('input,textarea')].filter(e => lab(e).length === 0).indexOf(el);
      return [...document.querySelectorAll(sel)].filter(e => lab(e).includes(want)).indexOf(el);}""", want)


async def verify(raw, arm, logs):
    task = parse_tasks([raw])[0]
    log = logs / f"{task.id}-{arm}.jsonl"
    log.unlink(missing_ok=True)
    args = SimpleNamespace(config=arm, start_url=task.start_url, har=har_path(task), replay_miss="stub", goal=task.goal,
                           check=task.check, log=log, task_id=task.id, max_calls=200, answer_regex=task.answer_regex,
                           allow_actions=task.allow_actions)
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
                if "wait" in step:
                    await proxy.owner(proxy.bench.session.page.wait_for_timeout, int(step["wait"]))
                    continue
                if "scroll" in step:
                    await proxy.owner(scroll_through, proxy.bench.session.page)
                    continue
                if "goto" in step:
                    nav = await call("open" if arm == "agent-browser" else "browser_navigate", url=step["goto"])
                    if "[ref=" not in nav:
                        # Offloaded or no snapshot: fetch a usable view for the next step.
                        nav = await call(observation[0], **observation[1])
                    first = nav
                    continue
                if "press" in step:
                    nav = await call("press" if arm == "agent-browser" else "browser_press_key", key=step["press"])
                    if "[ref=" in nav:
                        first = nav
                    continue
                kind, spec = step_locator(step)
                if kind is None:
                    raise ValueError(f"unsupported step {step}")
                # The oracle supplies only the next label. The action uses a ref/label actually exposed by the arm.
                names = await proxy.owner(lambda: element_labels(locate(proxy.bench.session.page, spec), kind))
                name = names[0] if names else ""
                if kind == "select":
                    option = dict(spec.get("option", {}))
                    value = await proxy.owner(lambda: locate(proxy.bench.session.page, spec).evaluate(
                        """(el, opt) => {
                          const opts = [...el.options];
                          const o = opt.label != null ? opts.find(o => o.label.trim() === opt.label)
                                  : opt.value != null ? opts.find(o => o.value === opt.value) : opts[opt.index];
                          return o ? o.value : null;}""", option))
                    if value is None:
                        raise ValueError(f"no option {option} on select")
                if arm == "pw-mcp+observe":
                    # Check each duplicate until the middleware id maps to the oracle node itself.
                    identifier = None
                    oracle = await proxy.owner(lambda: locate(proxy.bench.session.page, spec))
                    for cand in names:
                        if not cand:
                            continue
                        found = await call("find", query=cand)
                        for candidate in (line for line in found.splitlines()
                                          if line.startswith('tg-') and cand.casefold() in line.casefold()):
                            selector = await proxy.owner(proxy.bench.session.middleware_target, candidate.split('|')[0])
                            same = selector and await proxy.owner(lambda s=selector: oracle.evaluate(
                                '(el, selector) => el === document.querySelector(selector)', s))
                            if same:
                                identifier = candidate.split('|')[0]
                                name = cand
                                break
                        if identifier:
                            break
                    if identifier is None:
                        raise ValueError(f'middleware id selected a different duplicate: {names}')
                    if kind == "click":
                        await call('browser_click', target=identifier, element=name)
                    elif kind == "type":
                        await call('browser_type', target=identifier, element=name, text=step["text"],
                                   submit=bool(step.get("enter")))
                    else:
                        await call('browser_select_option', target=identifier, element=name, values=[value])
                else:
                    ref = None
                    snapshot = first
                    for cand in names or [""]:
                        ordinal = await proxy.owner(lambda c=cand: oracle_ordinal(
                            locate(proxy.bench.session.page, spec), c, kind))
                        if arm == "pw-mcp-compact" and cand:
                            snapshot = await call("browser_find", text=cand)
                        ref = snapshot_ref(snapshot, cand, kind, ordinal)
                        if ref is None and arm != "agent-browser" and cand:
                            snapshot = await call("browser_find", text=cand)
                            ref = snapshot_ref(snapshot, cand, kind, ordinal)
                        if ref is None and arm == "pw-mcp-compact":
                            # The depth-capped view truncates; a full snapshot is the last resort.
                            snapshot = await call("browser_snapshot")
                            ref = snapshot_ref(snapshot, cand, kind, ordinal)
                        if ref is not None:
                            name = cand
                            break
                    if ref is None:
                        raise ValueError(f"label absent from snapshot: {names}: {snapshot[:5000]}")
                    if arm == "agent-browser":
                        if kind == "click":
                            await call("click", ref="@" + ref)
                        elif kind == "type":
                            await call("fill", ref="@" + ref, text=step["text"])
                            if step.get("enter"):
                                await call("press", key="Enter")
                        else:
                            await call("select", ref="@" + ref, value=value)
                    elif kind == "click":
                        await call("browser_click", target=ref, element=name)
                    elif kind == "type":
                        await call("browser_type", target=ref, element=name, text=step["text"],
                                   submit=bool(step.get("enter")))
                    else:
                        await call("browser_select_option", target=ref, element=name, values=[value])
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
                    if "[ref=" not in first:
                        # Big pages offload the snapshot to a file link; fetch a usable view.
                        first = await call(observation[0], **observation[1])
                        event = json.loads(log.read_text().splitlines()[-1])
                        cost += -(-event["chars"] // 4) + event.get("image_tokens_est", 0)
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
        session.allow_actions = task.allow_actions
        bench = BenchServer("find+observe", session, SpecRanker("exact"), task.goal, task.check, log,
                            task.id, answer_regex=task.answer_regex)
        try:
            return _walk_native(bench, session, task, raw, log)
        finally:
            bench.finish("exit")


def _walk_native(bench, session, task, raw, log):
    try:
        bench.invoke("observe")
        for step in raw["path"]:
            if "wait" in step:
                session.page.wait_for_timeout(int(step["wait"]))
                continue
            if "scroll" in step:
                scroll_through(session.page)
                continue
            if "goto" in step:
                session.page.goto(step["goto"], wait_until="load")
                session._settle()
                continue
            if "press" in step:
                session.page.keyboard.press(step["press"])
                session._settle()
                continue
            kind, spec = step_locator(step)
            if kind is None:
                raise ValueError(f"unsupported step {step}")
            target = locate(session.page, spec)
            if kind == "select":
                # find+observe has no select tool; drive the element the oracle named.
                # TODO: PR #9 adds a `fill` tool that handles selects — route through it once merged.
                target.select_option(**dict(spec.get("option", {})))
                session._settle()
                continue
            names = element_labels(target, kind)
            element_id = None
            for cand in names or [""]:
                bench.invoke("find", query=cand)
                element_id = next((n for n, value in session.middleware_ids.items()
                                   if value == target.get_attribute("data-tg")), None)
                if element_id is not None:
                    break
            if element_id is None:
                raise ValueError(f"no middleware id for {names}")
            if kind == "click":
                bench.invoke("click", id=element_id)
            else:
                bench.invoke("type", id=element_id, text=step["text"],
                             press_enter=bool(step.get("enter")))
        bench.invoke("done", answer=raw.get("answer", ""))
        summary = json.loads(log.read_text().splitlines()[-1])
        return dict(task=task.id, arm="find+observe", ok=summary["success"], success=summary["success"],
                    **{k: summary[k] for k in ("safety_ms", "settle_ms", "proxy_ms", "harness_overhead_ms")})
    except (ToolError, ValueError, PlaywrightError) as exc:  # a refused or unmatched step is a verdict, not a crash
        return dict(task=task.id, arm="find+observe", ok=False, success=False,
                    error=(str(exc).splitlines() or [type(exc).__name__])[0][:300])


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
