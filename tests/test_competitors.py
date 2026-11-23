"""Proxy contracts, binary invocation, and compatibility with existing benchmark arms."""
import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mcp import types

from demo.cli_bench import build_prompt, parse_arm
from demo.competitor_specs import COMPETITOR_CONFIGS, PW_VERSION, AB_VERSION, pw_args
from demo.competitors import AgentBrowser, CompetitorBench, Proxy, result_volume, strip_snapshots
from demo.mcp_server import tools_for


def result(text):
    return types.CallToolResult(content=[types.TextContent(type="text", text=text)])


@pytest.mark.parametrize("arm", COMPETITOR_CONFIGS)
def test_arms(arm):
    assert parse_arm(arm).config == arm
    assert parse_arm(arm).ranker == ""
    with pytest.raises(ValueError):
        parse_arm(arm + "@exact")
    assert "done" in tools_for(arm)
    assert pw_args("pw-mcp", "endpoint") == ["--cdp-endpoint", "endpoint", "--timeout-settle", "0"]
    assert pw_args("pw-mcp-compact", "endpoint")[-2:] == ["--snapshot-mode", "none"]


def test_old_arms_unchanged():
    assert tools_for("a11y-cap") == ("snapshot", "click", "type", "go_back", "done")
    assert tools_for("find+observe") == ("find", "observe", "inspect", "read", "click", "type", "go_back", "done")
    assert parse_arm("find+observe@exact").label == "find+observe[exact]"


def test_strip_only_snapshot_section():
    original = result("### Ran Playwright code\nclick()\n### Page snapshot\n```yaml\n- secret\n```\n### Events\nkept")
    stripped = strip_snapshots(original)
    assert "secret" not in stripped.content[0].text
    assert "click()" in stripped.content[0].text
    assert "### Events\nkept" in stripped.content[0].text
    assert "secret" in original.content[0].text
    assert strip_snapshots(result("### Error\nfailed")).content[0].text == "### Error\nfailed"


def test_images_counted():
    from PIL import Image
    import io
    stream = io.BytesIO()
    Image.new("RGB", (30, 50)).save(stream, format="PNG")
    raw = stream.getvalue()
    r = result("hello")
    r.content.append(types.ImageContent(type="image", mimeType="image/png", data=base64.b64encode(raw).decode()))
    assert result_volume(r) == dict(chars=5, image_bytes=len(raw), images=1, image_tokens_est=2)


def test_proxy_passthrough_and_log(tmp_path):
    async def run():
        page = SimpleNamespace(is_closed=lambda: False, evaluate=lambda *args: [])
        session = SimpleNamespace(url="https://example.test/", replay=None, succeeded=lambda check: True,
                                  _settle=lambda: None, extract=lambda: None, page=page, context=SimpleNamespace(pages=[page]))
        b = CompetitorBench("a11y", session, None, "goal", {}, tmp_path / "log")
        b.config = "pw-mcp"
        b.identity = dict(version=PW_VERSION)
        tool = types.Tool(name="browser_click", description="stock", inputSchema={"type": "object", "properties": {"target": {"type": "string"}}})
        output = result("unchanged")
        child = SimpleNamespace(list_tools=AsyncMock(return_value=types.ListToolsResult(tools=[tool])), call_tool=AsyncMock(return_value=output))
        with ThreadPoolExecutor(max_workers=1) as pool:
            p = Proxy(b, pool, child, [])
            assert (await p.list_tools())[0] == tool
            assert await p.call("browser_click", {"target": "e1"}) is output
            assert child.call_tool.await_args_list[-1].args == ("browser_click", {"target": "e1"})
            assert child.call_tool.await_count == 1
            logged = json.loads((tmp_path / "log").read_text())
            assert logged["chars"] == len("unchanged")
            assert logged["forwarded_result"]["content"][0]["text"] == "unchanged"
            assert b.calls == b.steps == 1
            assert b.summary("test")["competitor"]["version"] == PW_VERSION
            b.finish("test")
    asyncio.run(run())


@pytest.mark.parametrize("name,args,expected", [
    ("snapshot", {}, ["snapshot", "-i", "-c"]),
    ("click", {"ref": "@e12"}, ["click", "@e12"]),
    ("fill", {"ref": "@e3", "text": "words"}, ["fill", "@e3", "words"]),
    ("get_text", {"ref": "@e4"}, ["get", "text", "@e4"]),
    ("open", {"url": "https://test/"}, ["open", "https://test/"]),
    ("back", {}, ["back"]),
])
def test_agent_browser_invocation(monkeypatch, tmp_path, name, args, expected):
    async def run():
        proc = SimpleNamespace(communicate=AsyncMock(return_value=(b"verbatim\n", b"")), returncode=0)
        spawn = AsyncMock(return_value=proc)
        monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
        cli = AgentBrowser("http://localhost:9222", tmp_path)
        out = await cli.call_tool(name, args)
        assert out.content[0].text == "verbatim\n"
        argv = spawn.call_args.args
        assert argv[0].endswith(".cache/competitors/node_modules/.bin/agent-browser")
        assert list(argv[-len(expected):]) == expected
        assert argv[argv.index("--cdp")+1] == "http://localhost:9222"
        assert spawn.call_args.kwargs["env"]["AGENT_BROWSER_NO_VERSION_CHECK"] == "1"
    asyncio.run(run())


def test_middleware_identity_and_stale_ids(tmp_path):
    from demo.competitors import MiddlewareSession
    html = tmp_path / 'page.html'
    html.write_text('<a href="#one">Duplicate</a><a href="#two">Duplicate</a>')
    with MiddlewareSession(html.as_uri()) as session:
        page = session.extract()
        links = [e for e in page.elements if e.role == 'link']
        ids = [session.middleware_ids[e.id] for e in links]
        assert len(set(ids)) == 2
        for identifier, href in zip(ids, ['#one', '#two']):
            selector = session.middleware_target(identifier)
            assert selector == f'[data-tg="{identifier}"]'
            assert session.page.locator(selector).count() == 1
            assert session.page.locator(selector).get_attribute('href') == href
        session.extract()
        assert session.middleware_target(ids[1]) != ids[1]
        assert session.middleware_target('e5') == 'e5'
        assert session.middleware_target('f1e48') == 'f1e48'
        session.page.locator(f'[data-tg="{ids[0]}"]').evaluate('el => el.remove()')
        assert session.middleware_target(ids[0]) == ids[0]
        session.page.reload()
        assert session.middleware_target(ids[1]) == ids[1]
        session.extract()
        assert session.middleware_target(ids[1]) == ids[1]


@pytest.mark.parametrize('html,selector,click,reason', [
    ('<button>Buy</button>', 'button', True, 'blocked word'),
    ('<form><button>Continue</button></form>', 'button', True, 'submits a form'),
    ('<input type="password">', 'input', False, 'password'),
    ('<button disabled>Continue</button>', 'button', True, 'disabled'),
    ('<section><h2>Delete account</h2><button>Continue</button></section>', 'button', True, 'blocked word'),
])
def test_competitor_element_safety(tmp_path, html, selector, click, reason):
    from demo.competitors import MiddlewareSession, check_locator
    from demo.browser import ToolError
    file = tmp_path / 'page.html'
    file.write_text(html)
    with MiddlewareSession(file.as_uri()) as session:
        with pytest.raises(ToolError, match=reason):
            check_locator(session, session.page.locator(selector), click)


def test_proxy_id_adapter_digest_and_safety(tmp_path):
    from demo.competitors import MiddlewareSession
    from demo.mcp_server import SpecRanker
    html = tmp_path / 'page.html'
    html.write_text('<a href="#one">Duplicate</a><a href="#two">Duplicate</a><button>Buy</button>')
    async def run():
        with ThreadPoolExecutor(max_workers=1) as pool:
            loop = asyncio.get_running_loop()
            def build():
                session = MiddlewareSession(html.as_uri())
                b = CompetitorBench('find+observe', session, SpecRanker('exact'), 'Duplicate', {}, tmp_path / 'calls')
                b.config = 'pw-mcp+observe'
                b.identity = {'version': PW_VERSION}
                return b
            b = await loop.run_in_executor(pool, build)
            tool = types.Tool(name='browser_click', inputSchema={'type': 'object', 'properties': {'target': {'type': 'string'}}})
            child = SimpleNamespace(list_tools=AsyncMock(return_value=types.ListToolsResult(tools=[tool])), call_tool=AsyncMock(return_value=result('### Page\nstock summary\n### Snapshot\nold snapshot')))
            p = Proxy(b, pool, child, [])
            try:
                await p.list_tools()
                view = await p.owner(b._t_observe)
                ids = [line.split('|')[0] for line in view.splitlines() if line.startswith('tg-') and 'Duplicate' in line]
                assert len(ids) == 2
                out = await p.call('browser_click', {'target': ids[1]})
                child.call_tool.assert_awaited_once_with('browser_click', {'target': f'[data-tg="{ids[1]}"]'})
                assert out.content[0].text.startswith('### Page\nstock summary')
                assert 'old snapshot' not in out.content[0].text
                assert 'Duplicate' in out.content[-1].text
                assert len(out.content[-1].text) <= 2400
                event = json.loads((tmp_path / 'calls').read_text().splitlines()[-1])
                assert event['args']['target'] == ids[1]
                assert event['forwarded_args']['target'] == f'[data-tg="{ids[1]}"]'
                assert event['chars'] == sum(len(c.text) for c in out.content)
                assert event['extract_ms'] > 0 and event['rank_ms'] > 0
                assert event['safety'] == 'checked'
                # A dangerous action must never reach the child.
                rejected = await p.call('browser_click', {'target': 'button'})
                assert rejected.isError and 'blocked word' in rejected.content[0].text
                assert child.call_tool.await_count == 1
                await p.call('browser_click', {'target': 'e5'})
                assert child.call_tool.call_args.args[1]['target'] == 'e5'
            finally:
                await p.owner(b.finish, 'test')
                await p.owner(b.session.close)
    asyncio.run(run())


@pytest.mark.parametrize("name,args,expected", [
    ("browser_snapshot", {}, False), ("browser_find", {}, False),
    ("browser_console_messages", {}, False), ("browser_network_requests", {}, False),
    ("browser_take_screenshot", {}, False), ("snapshot", {}, False), ("get_text", {}, False),
    ("browser_click", {}, True), ("browser_type", {}, True), ("open", {}, True),
    ("browser_evaluate", {}, True), ("new_future_tool", {}, True),
    ("browser_tabs", {"action": "list"}, False), ("browser_tabs", {"action": "select"}, True),
])
def test_settle_classification(name, args, expected):
    from demo.competitor_specs import page_changing
    assert page_changing(name, args) is expected


def test_pinned_inventory():
    import re
    from demo.competitor_specs import PW_PAGE_CHANGING, BIN_DIR
    bundle = BIN_DIR.parent / "playwright-core/lib/coreBundle.js"
    if not bundle.exists():
        pytest.skip("pinned binary is not installed")
    assert set(re.findall(r'name: "(browser_[^"]+)"', bundle.read_text())) == set(PW_PAGE_CHANGING)


def test_cheap_label_ref_safety():
    from demo.competitors import check_target
    from demo.browser import ToolError
    from unittest.mock import Mock
    page = SimpleNamespace(evaluate=Mock(return_value=[]))
    session = SimpleNamespace(page=page)
    assert 'label checked' in check_target(session, 'e14', 'Documentation', True)
    page.evaluate.assert_called_once()
    with pytest.raises(ToolError, match="blocked word"):
        check_target(session, '@e2', 'Buy now', True)
    assert page.evaluate.call_count == 1


def test_overhead_run_record():
    from demo.cli_bench import run_overhead
    assert run_overhead(dict(safety_ms=2, settle_ms=500, proxy_ms=3), [], 1) == dict(
        safety_ms=2, settle_ms=500, proxy_ms=3, harness_overhead_ms=505,
        wall_seconds_without_harness=.495)
    assert run_overhead({}, [dict(safety_ms=4, settle_ms=500)], 2)['harness_overhead_ms'] == 504


@pytest.mark.parametrize("arm,read_tool,action_tool,action_args", [
    ("pw-mcp", "browser_snapshot", "browser_click", {"target": "e1", "element": "Documentation"}),
    ("agent-browser", "snapshot", "click", {"ref": "@e1"}),
])
def test_proxy_settle_and_cached_label(tmp_path, arm, read_tool, action_tool, action_args):
    from unittest.mock import Mock
    async def run():
        page = SimpleNamespace(is_closed=lambda: False, evaluate=Mock(return_value=[]))
        session = SimpleNamespace(url="test:", replay=None, succeeded=lambda check: True,
                                  _settle=Mock(), page=page, context=SimpleNamespace(pages=[page]))
        b = CompetitorBench("a11y", session, None, "goal", {}, tmp_path / "calls")
        b.config, b.identity = arm, {}
        names = [read_tool, action_tool]
        child = SimpleNamespace(list_tools=AsyncMock(return_value=types.ListToolsResult(tools=[
            types.Tool(name=n, inputSchema={"type": "object"}) for n in names])),
            call_tool=AsyncMock(return_value=result('- button "Buy now" [ref=e1]')))
        with ThreadPoolExecutor(max_workers=1) as pool:
            p = Proxy(b, pool, child, [])
            await p.list_tools()
            await p.call(read_tool, {})
            session._settle.assert_not_called()
            event = json.loads((tmp_path / 'calls').read_text().splitlines()[-1])
            assert event['settle_ms'] == event['safety_ms'] == 0
            out = await p.call(action_tool, action_args)
            if arm == "agent-browser":
                assert out.isError and 'blocked word' in out.content[0].text
                assert child.call_tool.await_count == 1
                session._settle.assert_not_called()
            else:
                session._settle.assert_called_once()
                assert child.call_tool.await_count == 2
            assert b.summary('test')['safety_ms'] > 0
            b.finish('test')
    asyncio.run(run())


@pytest.mark.parametrize("arm", ["pw-mcp+observe", "agent-browser"])
def test_summary_written_when_agent_exits_without_done(tmp_path, arm):
    """Pilot regression: a run that ended without done() lost its summary while the child tool shut down."""
    from pathlib import Path
    import yaml
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from demo.browser import parse_tasks
    from demo.cli_bench import REPO, server_argv, task_har
    from demo.competitor_specs import BIN_DIR
    exe = BIN_DIR / ("agent-browser" if arm == "agent-browser" else "playwright-mcp")
    task = next(t for t in parse_tasks(yaml.safe_load((REPO / "demo/tasks_milestone.yaml").read_text())) if t.id == "elv-str")
    har = task_har(task, live=False)
    if not exe.exists() or har is None or not har.exists():
        pytest.skip("pinned competitor binaries or frozen snapshot not installed")
    log = tmp_path / "server.jsonl"
    argv = server_argv(task, arm, "exact", Path("dist/xs-s0"), log, 60, True, har=har)

    async def run():
        params = StdioServerParameters(command=argv[0], args=argv[1:], cwd=str(REPO), env={**__import__("os").environ, "PYTHONPATH": str(REPO)})
        async with stdio_client(params) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                assert (await session.list_tools()).tools
        # client exits without calling done: stdin EOF must still produce exactly one summary

    asyncio.run(run())
    events = [json.loads(line) for line in log.read_text().splitlines()]
    summaries = [e for e in events if e.get("type") == "summary"]
    assert len(summaries) == 1 and summaries[0]["done_called"] is False


def test_proxy_hides_arbitrary_code_tools(tmp_path):
    async def run():
        b = SimpleNamespace(config="pw-mcp")
        names = ["browser_click", "browser_run_code", "browser_run_code_unsafe", "browser_evaluate"]
        child = SimpleNamespace(list_tools=AsyncMock(return_value=types.ListToolsResult(tools=[
            types.Tool(name=n, inputSchema={"type": "object"}) for n in names])))
        p = Proxy(b, None, child, [])
        listed = {t.name for t in await p.list_tools()}
        assert listed == p.allowed == {"browser_click", "browser_evaluate"}
    asyncio.run(run())
