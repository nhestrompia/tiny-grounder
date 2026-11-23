"""Real competitor binaries behind the benchmark MCP server and its server-side log."""
from __future__ import annotations

import asyncio
import base64
import copy
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack, asynccontextmanager
import io
import json
import os
from pathlib import Path
import re
import signal
import tempfile
import time
import uuid

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from demo.browser import BrowserSession, ToolError
from grounder import playwright_helper as ph
from grounder.extract.web import DOM_JS, page_from_raw
from demo.mcp_server import _FACTS_JS
from demo.competitor_specs import AB_VERSION, BIN_DIR, PW_VERSION, pw_args, page_changing, PW_PAGE_CHANGING, AB_PAGE_CHANGING
from demo.mcp_server import BenchServer, CONFIG_HINTS, RESULT_LOG_CHARS, SpecRanker, build_mcp


def strip_snapshots(result: types.CallToolResult) -> types.CallToolResult:
    """Fallback for versions without snapshot-mode; preserve all other content blocks."""
    result = result.model_copy(deep=True)
    for block in result.content:
        if block.type == "text":
            block.text = re.sub(r"(?ms)^### (?:Page snapshot|Snapshot)\n.*?(?=^### |\Z)", "", block.text)
    return result


def result_volume(result):
    chars = image_bytes = image_tokens = images = 0
    for block in result.content:
        if block.type == "text":
            chars += len(block.text)
        elif block.type == "image":
            from PIL import Image
            raw = base64.b64decode(block.data)
            with Image.open(io.BytesIO(raw)) as img:
                width, height = img.size
            image_bytes += len(raw)
            image_tokens += -(-width * height // 750)
            images += 1
        else:
            # Resource links and embedded resource metadata are visible too.
            chars += len(block.model_dump_json(exclude_none=True))
    if result.structuredContent is not None:
        chars += len(json.dumps(result.structuredContent, ensure_ascii=False))
    return dict(chars=chars, image_bytes=image_bytes, images=images, image_tokens_est=image_tokens)


class MiddlewareSession(BrowserSession):
    """Stamp the exact extractor nodes, preserving identity across extractions of one document."""

    def extract(self):
        stamp = """if (described) {
          const state = window.__tgAdapter ||= {prefix: crypto.randomUUID().slice(0,8), next: 0, ids: new WeakMap()};
          let id = state.ids.get(entry.el);
          if (!id) { id = `tg-${state.prefix}-${state.next++}`; state.ids.set(entry.el, id); }
          entry.el.setAttribute('data-tg', id);
          elements.push({ id: elements.length, ...described, middleware_id: id });
        }"""
        raw = self._guard(lambda: self.page.evaluate(DOM_JS.replace(
            'if (described) elements.push({ id: elements.length, ...described });', stamp)))
        self.middleware_ids = {e['id']: e.pop('middleware_id') for e in raw['elements']}
        page = page_from_raw(raw, self.url, self.url, '')
        self.landmarks, self.heading_positions = ph.extract_landmarks(self.page)
        self.elements, self._extracted_url = page.elements, page.url
        return page

    def middleware_target(self, target):
        if not isinstance(target, str) or not re.fullmatch(r'tg-[a-f0-9]{8}-\d+', target):
            return target
        if target not in getattr(self, 'middleware_ids', {}).values() or self._extracted_url != self.url:
            return target
        selector = f'[data-tg="{target}"]'
        return selector if self.page.locator(selector).count() == 1 else target


def check_label(label, click_action, allow=False):
    if not allow and click_action and (word := ph.blocked_text(label)):
        raise ToolError(f"refused: risky: text or context contains the blocked word {word!r}")


_SAFETY_JS = """el => {
  const facts = (""" + _FACTS_JS + """)(el);
  return {...facts, disabled: el.matches(':disabled,[aria-disabled=true]'),
    context: el.closest('section,article,form,fieldset')?.querySelector('h1,h2,h3,legend')?.textContent || ''};
}"""


def check_facts(facts, click_action, allow=False):
    check_label(facts['label'] + ' ' + facts.get('context', ''), click_action, allow)
    if facts.get('disabled'):
        raise ToolError('refused: the element is disabled')
    if not allow and (reason := ph.verdict(dict(ok=True, **facts), click_action)):
        raise ToolError(f'refused: {reason}')


def check_locator(session, loc, click_action):
    # One non-waiting DOM read. The child retains its native actionability checks.
    check_facts(loc.evaluate(_SAFETY_JS, timeout=100), click_action, getattr(session, 'allow_actions', False))


def check_target(session, target, label, click_action):
    allow = getattr(session, 'allow_actions', False)
    check_label(label, click_action, allow)
    # Child refs belong to another client. Resolve available labels in one DOM read,
    # conservatively checking every exact-label match rather than guessing a duplicate.
    facts = session.page.evaluate("""([target, label]) => {
      const facts = """ + _SAFETY_JS + """;
      if (target && !/^(?:@|f\\d+)?e\\d+$/.test(target)) {
        try { return [...document.querySelectorAll(target)].map(facts); } catch { return []; }
      }
      const norm = s => (s || '').replace(/\\s+/g, ' ').trim();
      return [...document.querySelectorAll('a,button,input,textarea,select,[role]')]
        .filter(el => norm(el.getAttribute('aria-label') || el.innerText || el.value || el.title) === norm(label))
        .map(facts);
    }""", [target, label])
    for item in facts:
        check_facts(item, click_action, allow)
    return 'checked' if facts else 'label checked; DOM facts unavailable'


class AgentBrowser:
    def __init__(self, endpoint, directory):
        self.endpoint = endpoint
        self.session = "b" + uuid.uuid4().hex[:10]
        self.directory = str(directory)
        self.env = {k: v for k, v in os.environ.items() if not k.startswith("AGENT_BROWSER_")}
        self.env.update(AGENT_BROWSER_NAMESPACE=self.session, AGENT_BROWSER_NO_VERSION_CHECK="1", AGENT_BROWSER_SOCKET_DIR=str(directory))
        self.base = [str(BIN_DIR / "agent-browser"), "--config", str(Path(directory) / "agent-browser.json"),
                     "--session", self.session, "--cdp", endpoint]
        (Path(directory) / "agent-browser.json").write_text("{}")

    async def command(self, args):
        proc = await asyncio.create_subprocess_exec(*self.base, *args, stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.PIPE, env=self.env, cwd=self.directory)
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), 90)
        except BaseException:
            proc.kill()
            await proc.wait()
            raise
        text = stdout.decode()
        if proc.returncode:
            return types.CallToolResult(content=[types.TextContent(type="text", text=text or stderr.decode())], isError=True)
        return types.CallToolResult(content=[types.TextContent(type="text", text=text)])

    async def call_tool(self, name, args):
        if name == "snapshot":
            command = ["snapshot", "-i", "-c"]
        elif name in ("click", "fill", "get_text", "select"):
            ref = args["ref"]
            if not re.fullmatch(r"@e\d+", ref):
                raise ToolError("use an @ref from snapshot")
            if name == "select":
                command = ["select", ref, args["value"]]
            else:
                command = ["get", "text", ref] if name == "get_text" else [name, ref]
                if name == "fill":
                    command.append(args["text"])
        elif name == "press":
            command = ["press", args["key"]]
        elif name == "open":
            command = ["open", args["url"]]
        elif name == "back":
            command = ["back"]
        else:
            raise ToolError(f"unknown tool {name}")
        return await self.command(command)

    async def list_tools(self):
        def tool(name, description, props=None):
            return types.Tool(name=name, description=description, inputSchema={"type": "object", "properties": props or {}, "required": list(props or {}), "additionalProperties": False})
        string = {"type": "string"}
        return types.ListToolsResult(tools=[
            tool("snapshot", "Interactive compact snapshot: agent-browser snapshot -i -c."),
            tool("click", "Click an @ref.", {"ref": string}),
            tool("fill", "Clear and fill an @ref.", {"ref": string, "text": string}),
            tool("select", "Select an option in an @ref dropdown.", {"ref": string, "value": string}),
            tool("press", "Press a key.", {"key": string}),
            tool("open", "Navigate to a URL.", {"url": string}),
            tool("back", "Go back."), tool("get_text", "Get text of an @ref.", {"ref": string}),
        ])


class CompetitorBench(BenchServer):
    def _ids(self, text):
        return re.sub(r'(?m)^(\d+)\|', lambda m: self.session.middleware_ids[int(m[1])] + '|', text)

    def _t_find(self, query):
        return self._ids(super()._t_find(query))

    def _t_observe(self, goal='', budget=600):
        # Account for the longer public ids inside the same estimated-token budget.
        inner_budget = budget
        while True:
            out = self._ids(super()._t_observe(goal, inner_budget))
            if len(out) <= budget * 4 or inner_budget <= 100:
                return out
            inner_budget = max(100, inner_budget - (len(out) - budget * 4 + 3) // 4)

    def _t_inspect(self, id=None, region=''):
        self.session.extract()
        if isinstance(id, str):
            id = next((n for n, value in self.session.middleware_ids.items() if value == id), -1)
        return self._ids(super()._t_inspect(id=id, region=region))

    def summary(self, reason):
        return {**super().summary(reason), "competitor": self.identity,
                "tool_tokens_est": -(-self.chars // 4) + self.image_tokens}


# Arbitrary code would bypass the element safety checks that every arm shares.
HIDDEN_TOOLS = {"browser_run_code", "browser_run_code_unsafe"}


class Proxy:
    def __init__(self, bench, pool, child, local_tools):
        self.bench, self.pool, self.child, self.local_tools = bench, pool, child, local_tools
        self.allowed = set()
        self.target_tools = set()
        self.ref_labels = {}

    async def owner(self, fn, *args, **kwargs):
        return await asyncio.get_running_loop().run_in_executor(self.pool, lambda: fn(*args, **kwargs))

    async def list_tools(self):
        remote = [t for t in (await self.child.list_tools()).tools if t.name not in HIDDEN_TOOLS]
        # Middleware keeps actions and utilities, but perception comes from tiny-grounder.
        if self.bench.config == "pw-mcp+observe":
            remote = [t for t in remote if t.name not in {"browser_snapshot", "browser_find", "browser_take_screenshot"}]
        self.target_tools = {t.name for t in remote if 'target' in t.inputSchema.get('properties', {})}
        self.allowed = {t.name for t in remote}
        return remote + self.local_tools

    async def call(self, name, args):
        if name in {t.name for t in self.local_tools}:
            try:
                out = await self.owner(self.bench.invoke, name, **args)
                return types.CallToolResult(content=[types.TextContent(type="text", text=out)])
            except ToolError as exc:
                return types.CallToolResult(content=[types.TextContent(type="text", text=str(exc))], isError=True)
        start = time.perf_counter()
        b = self.bench
        if b.finished:
            return types.CallToolResult(content=[types.TextContent(type="text", text="the task is over")], isError=True)
        b.calls += 1
        forwarded_args = copy.deepcopy(args)
        safety = "not applicable"
        b.extract_ms = b.rank_ms = 0.0
        action = page_changing(name, args)
        before = b.overhead_totals()
        child_ms = digest_ms = 0.0
        b.steps += action
        try:
            if name not in self.allowed:
                raise ToolError(f"unknown tool {name}")
            if b.calls > b.max_calls:
                raise ToolError(f"call budget of {b.max_calls} used up; call done(answer) now")
            if name == "browser_tabs" and args.get("action") == "new":
                raise ToolError("benchmark replay is single-tab; use browser_navigate in the existing tab")
            if name == "browser_tabs" and args.get("isolatedContext", "default") != "default":
                raise ToolError("benchmark replay uses one context; isolatedContext is unavailable")
            target_slots = []
            def collect_targets(value):
                if isinstance(value, dict):
                    for key, item in value.items():
                        if key in {'target', 'startTarget', 'endTarget', 'ref'} and isinstance(item, str):
                            target_slots.append((value, key))
                        else:
                            collect_targets(item)
                elif isinstance(value, list):
                    for item in value:
                        collect_targets(item)
            if name.startswith('browser_'):
                collect_targets(forwarded_args)
            safety_start = time.perf_counter()
            try:
                if b.config == 'pw-mcp+observe':
                    for obj, key in target_slots:
                        obj[key] = await self.owner(b.session.middleware_target, obj[key])
                if action and (target_slots or name in {'click', 'fill', 'select'}):
                    click_action = name not in {'fill', 'select', 'browser_type', 'browser_hover', 'browser_fill_form',
                                                'browser_select_option'}
                    slots = target_slots or [(args, 'ref')]
                    def safety_check():
                        checks = []
                        for obj, key in slots:
                            target = obj[key]
                            label_key = {'startTarget': 'startElement', 'endTarget': 'endElement'}.get(key, 'element')
                            label = obj.get(label_key, args.get(label_key, self.ref_labels.get(target, '')))
                            # Middleware ids check their cached text; context comes from the same DOM read as other arms.
                            identifier = next((v for v in getattr(b.session, 'middleware_ids', {}).values()
                                               if v in target), None)
                            if identifier:
                                element = next((e for e in b.session.elements
                                                if b.session.middleware_ids.get(e.id) == identifier), None)
                                if element:
                                    check_label(element.text, click_action,
                                                getattr(b.session, 'allow_actions', False))
                            checks.append(check_target(b.session, target, label, click_action))
                        return '; '.join(checks)
                    safety = await self.owner(safety_check)
            finally:
                if action and (target_slots or name in {'click', 'fill'}):
                    b.session.safety_ms += (time.perf_counter() - safety_start) * 1000
            child_start = time.perf_counter()
            try:
                result = await self.child.call_tool(name, forwarded_args)
            finally:
                child_ms = (time.perf_counter() - child_start) * 1000
            if name == 'snapshot' and not result.isError:
                self.ref_labels = {}
                for block in result.content:
                    if block.type == 'text':
                        for label, ref in re.findall(r'"([^"\n]*)"[^\n]*?\[ref=(e\d+)\]', block.text):
                            self.ref_labels['@' + ref] = label
            elif action:
                self.ref_labels.clear()
            def select_page():
                pages = self.bench.session.context.pages
                if name == "browser_tabs" and args.get("action") == "select":
                    self.bench.session.page = pages[args["index"]]
                elif self.bench.session.page.is_closed() and pages:
                    self.bench.session.page = pages[-1]
            if action:
                await self.owner(select_page)
            if b.config == "pw-mcp+observe":
                result = strip_snapshots(result)
            if action and not result.isError:
                await self.owner(b.session._settle)
            if b.config == 'pw-mcp+observe' and action and not result.isError:
                if name in self.target_tools and b.box is not None:
                    b.box.history.append(f"{name} {args.get('target', '')}")
                digest_start = time.perf_counter()
                digest = await self.owner(b._t_observe)
                digest_ms = (time.perf_counter() - digest_start) * 1000
                result.content.append(types.TextContent(type='text', text=digest))
        except Exception as exc:
            result = types.CallToolResult(content=[types.TextContent(type="text", text=str(exc))], isError=True)
        elapsed_ms = (time.perf_counter() - start) * 1000
        overhead = {k: v - before[k] for k, v in b.overhead_totals().items()}
        proxy_ms = max(0.0, elapsed_ms - child_ms - digest_ms - overhead['safety_ms'] - overhead['settle_ms'])
        b.proxy_ms += proxy_ms
        overhead = {k: v - before[k] for k, v in b.overhead_totals().items()}
        volume = result_volume(result)
        b.chars += volume["chars"]
        b.image_bytes += volume["image_bytes"]
        b.images += volume["images"]
        b.image_tokens += volume["image_tokens_est"]
        b.errors += bool(result.isError)
        event = dict(type="call", n=b.calls, name=name, args=args, is_error=bool(result.isError),
                     ms=round((time.perf_counter()-start)*1000), extract_ms=b.extract_ms, rank_ms=b.rank_ms, forwarded_args=forwarded_args, safety=safety, page_changing=action,
                     unknown_tool=name not in PW_PAGE_CHANGING and name not in AB_PAGE_CHANGING, child_ms=child_ms, **overhead, **volume)
        event["result"] = "\n".join(c.text for c in result.content if c.type == "text")[:RESULT_LOG_CHARS]
        # Full forwarded content is retained so binary outputs can be audited.
        event["forwarded_result"] = result.model_dump(mode="json", exclude_none=True)
        await self.owner(b._log, event)
        return result


@asynccontextmanager
async def competitor_session(args):
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="replay-owner")
    loop = asyncio.get_running_loop()
    async def owner(fn):
        return await loop.run_in_executor(pool, fn)
    def build():
        session = MiddlewareSession(args.start_url, har=args.har, replay_miss=args.replay_miss, cdp=True)
        session.allow_actions = getattr(args, 'allow_actions', False)
        bench = CompetitorBench("find+observe", session, SpecRanker("exact"), args.goal, args.check, args.log,
                                args.task_id, max_calls=args.max_calls, answer_regex=args.answer_regex)
        bench.config = args.config
        bench.identity = {"package": "agent-browser" if args.config == "agent-browser" else "@playwright/mcp",
                          "version": AB_VERSION if args.config == "agent-browser" else PW_VERSION}
        session.context.on("page", lambda page: setattr(session, "page", page))
        return bench
    bench = await owner(build)
    pumping = True
    async def pump():
        # Sync Playwright must dispatch HAR route events while a different CDP client drives it.
        while pumping:
            def tick():
                if bench.session.page.is_closed():
                    pages = bench.session.context.pages
                    if not pages:
                        return
                    bench.session.page = pages[-1]
                bench.session.page.wait_for_timeout(1)
            await owner(tick)
            await asyncio.sleep(0)
    pump_task = asyncio.create_task(pump())
    try:
        with tempfile.TemporaryDirectory(prefix="tg-", dir="/tmp") as tmp:
            async with AsyncExitStack() as stack:
                if args.config == "agent-browser":
                    child = AgentBrowser(bench.session.cdp_endpoint, tmp)
                    version = await asyncio.create_subprocess_exec(str(BIN_DIR / "agent-browser"), "--version", stdout=asyncio.subprocess.PIPE)
                    stdout, _ = await version.communicate()
                    if not re.search(r"(?<![\d.])" + re.escape(AB_VERSION) + r"(?![\d.])", stdout.decode()):
                        raise RuntimeError("agent-browser version mismatch")
                    bench.identity = dict(package="agent-browser", version=AB_VERSION, flags=child.base[1:], snapshot_flags=["-i", "-c"])
                    stack.push_async_callback(child.command, ["close"])
                else:
                    package = json.loads((BIN_DIR.parent / "@playwright/mcp/package.json").read_text())
                    if package["version"] != PW_VERSION:
                        raise RuntimeError("Playwright MCP version mismatch")
                    flags = pw_args(args.config, bench.session.cdp_endpoint)
                    bench.identity = dict(package="@playwright/mcp", version=PW_VERSION, flags=flags)
                    streams = await stack.enter_async_context(stdio_client(StdioServerParameters(command=str(BIN_DIR / "playwright-mcp"), args=flags, cwd=tmp)))
                    child = await stack.enter_async_context(ClientSession(*streams))
                    await child.initialize()
                local_app = build_mcp("find+observe", None)
                local = await local_app.list_tools()
                names = {"done", "find", "observe", "inspect", "read"} if args.config == "pw-mcp+observe" else {"done"}
                local = [t.model_copy(deep=True) for t in local if t.name in names]
                for tool in local:
                    if tool.name == 'inspect':
                        tool.inputSchema['properties']['id'] = {'anyOf': [{'type': 'string'}, {'type': 'integer'}, {'type': 'null'}]}
                proxy = Proxy(bench, pool, child, local)
                await proxy.list_tools()
                try:
                    yield proxy
                finally:
                    # Write the summary before closing the child tools: a slow child shutdown must not lose it
                    # (bench.finish writes once; the later call in the outer finally is then a no-op).
                    await owner(lambda: bench.finish(getattr(args, "shutdown_reason", "exit")))
    finally:
        pumping = False
        await pump_task
        await owner(lambda: (bench.finish(getattr(args, "shutdown_reason", "exit")), bench.session.close()))
        pool.shutdown()


async def serve(args):
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    def stop():
        args.shutdown_reason = "signal"
        task.cancel()
    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, stop)
    async with competitor_session(args) as proxy:
        server = Server("ui", instructions=CONFIG_HINTS[args.config])
        server.list_tools()(proxy.list_tools)
        server.call_tool(validate_input=False)(proxy.call)
        async with stdio_server() as streams:
            await server.run(*streams, server.create_initialization_options())


def main(args):
    try:
        asyncio.run(serve(args))
    except asyncio.CancelledError:
        pass
    return 0
