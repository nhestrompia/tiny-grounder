from __future__ import annotations

import asyncio
import socket
import json
from pathlib import Path

import pytest

from grounder.records import Element, Page
from grounder_tool.api import UIGrounder
from grounder_tool.digest import estimate_tokens


def page(n=100):
    return Page(
        "p",
        "https://example.test",
        "example",
        "Install docs",
        [1280, 800],
        [1280, 5000],
        [
            Element(i, "link", f"Install {i}", context=f"Section {i % 30}", href=f"/{i}", abs_box=[10, i * 20, 100, 20])
            for i in range(n)
        ],
    )


@pytest.mark.parametrize("n", [0, 5, 500, 2000])
@pytest.mark.parametrize("budget", [0, 1, 50, 120, 300, 600])
def test_budget(n, budget):
    assert estimate_tokens(UIGrounder().observe(page(n), budget=budget)) <= budget


def test_dedup_outline_and_details():
    p = page()
    p.elements.append(Element(999, "link", "INSTALL 0", context="Section 0", href="/0"))
    g = UIGrounder()
    out = g.observe(p, "Install", 600, landmarks=[f"Heading {i}" for i in range(30)])
    assert "headings:" in out and " more" in out
    assert not ("0|link|" in out and "999|link|" in out)
    detail = g.inspect(p, 0)
    assert "center: 60,10" in detail and "enabled: True" in detail
    assert "Unknown element" in g.inspect(p, 9999)
    assert "Unknown section" in g.inspect(p, region="absent")
    assert "0|link|Install 0" in g.inspect(p, region="section 0")
    assert estimate_tokens(g.inspect(p, region="Section", budget=50)) <= 50


@pytest.mark.parametrize("spec", ["exact", "blend:dist/xs-s0:8"])
def test_goal_history(spec):
    if spec.startswith("blend:") and not Path("dist/xs-s0/model.int8.onnx").exists():
        pytest.skip("dist/xs-s0 not present")
    g = UIGrounder(spec)
    original = g.ranker.rank
    calls = []

    def rank(query, elements, mode="do", history=None):
        calls.append((query, mode, history))
        return original(query, elements, mode, history)

    g.ranker.rank = rank
    g.observe(page(5), history=["[a] Install -> CLICK"])
    assert calls == [("Install docs", "goal", ["[a] Install -> CLICK"])]


def test_cdp_attach(tmp_path):
    pytest.importorskip("playwright")
    from concurrent.futures import ThreadPoolExecutor
    from playwright.sync_api import sync_playwright
    from grounder_tool.browser import ToolError
    from grounder_tool.mcp import ToolServer, build_mcp

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    html = tmp_path / "page.html"
    html.write_text(
        '<title>Install</title><h1>Docs</h1><div style="height:1000px"></div><a href="/install">Install guide</a>'
    )
    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=[f"--remote-debugging-port={port}"])
        tab = browser.new_page()
        tab.goto(html.as_uri())
        server = ToolServer(UIGrounder(), cdp_endpoint=f"http://127.0.0.1:{port}", cdp_page_url_substring="page.html")

        def check():
            server._need()
            attached = server.session.page
            attached.evaluate("scrollTo(0, 700)")
            assert server.observe(budget=50) == server.observe(budget=100)
            assert "headings: Docs" in server.observe()
            assert "Install guide" in server.observe()
            found = server.find("Install")
            line = next(line for line in found.splitlines() if "|link|" in line)
            center = attached.locator("a").bounding_box()
            xy = [float(v) for v in line.rsplit("@", 1)[1].split(",")]
            assert xy == pytest.approx([center["x"] + center["width"] / 2, center["y"] + center["height"] / 2], abs=0.1)
            with pytest.raises(ToolError, match="perception-only"):
                server.open("https://example.test")
            with pytest.raises(ToolError, match="perception-only"):
                server.session.go_back()
            attached.locator("a").evaluate("el => el.textContent = 'Updated guide'")
            assert "Updated guide" in server.find("guide")
            detail = server.inspect(region="Docs")
            assert "Updated guide" in detail and "@" in detail
            assert attached.url == html.as_uri()

        with ThreadPoolExecutor(max_workers=1) as worker:
            try:
                worker.submit(check).result()
                mcp, pool = build_mcp(server)
                try:
                    names = pool.submit(lambda: {t.name for t in asyncio.run(mcp.list_tools())}).result()
                    assert not names & {"open", "click", "type", "go_back"}
                finally:
                    pool.shutdown()
            finally:
                worker.submit(server.close).result()
        assert tab.title() == "Install"
        browser.close()


def test_arm():
    from demo.cli_bench import parse_arm, resolve_ranker_spec

    arm = parse_arm("find+observe@exact+blend:dist/xs-s0:8")
    assert arm.config == "find+observe"
    assert resolve_ranker_spec(arm.ranker).startswith("exact+blend:/")
    assert arm.label == "find+observe[exact+blend]"
    assert parse_arm("find+observe@exact").label == "find+observe[exact]"
    for bad in ("find@exact+bm25", "find+observe@exact+", "find+observe@exact+bm25+exact"):
        with pytest.raises(ValueError):
            parse_arm(bad)


def test_har_observe(tmp_path):
    from demo.browser import BrowserSession, load_tasks
    from demo.mcp_server import BenchServer, make_ranker

    tasks = load_tasks("demo/tasks_v2.yaml")
    task = next(t for t in tasks if t.id == "fastapi-tutorial-devserver")
    if not Path(task.har).exists():
        pytest.skip("bench_v2 HAR not present")
    session = BrowserSession(task.start_url, har=task.har)
    server = BenchServer(
        "find+observe", session, make_ranker("exact", ""), task.goal, task.check, tmp_path / "run.jsonl"
    )
    try:
        out = server.invoke("observe")
        assert "showing" in out and estimate_tokens(out) <= 600
        assert server.invoke("find", query="documentation")
        assert "Unknown element" in server.invoke("inspect", id=-1)
        assert server._t_observe() == server._t_observe(task.goal)
        from grounder.records import action_repr

        found = server.invoke("find", query="Learn")
        identifier = int(next(line.split("|")[0] for line in found.splitlines() if "|Learn|" in line))
        element = next(e for e in session.elements if e.id == identifier)
        action = server.invoke("click", id=identifier)
        assert action.startswith("clicked") and "showing" in action and "url:" in action
        assert server.box.history == [action_repr(element)]
    finally:
        server.finish("test")
        session.close()
    calls = [
        e for line in (tmp_path / "run.jsonl").read_text().splitlines() if (e := json.loads(line))["type"] == "call"
    ]
    assert all("extract_ms" in e and "rank_ms" in e for e in calls)
    assert next(e for e in calls if e["name"] == "find")["rank_ms"] > 0


@pytest.mark.parametrize("budget", [0, 1, 10, 50, 120, 300, 600])
def test_long_header_and_inspect_lines(budget):
    p = page(500)
    p.url = "https://example.test/" + "x" * 10000
    p.title = "Title " * 10000
    for e in p.elements:
        e.text = "Long text " * 100
        e.context = "Region " + str(e.id) + "x" * 1000
    g = UIGrounder()
    out = g.observe(p, budget=budget)
    assert estimate_tokens(out) <= budget
    assert len(out.splitlines()[0]) <= 205 if budget >= 120 else True
    for kwargs in ({"id": 0}, {"region": "Region"}, {}, {"id": -1}, {"region": "missing"}):
        detail = g.inspect(p, budget=budget, **kwargs)
        assert estimate_tokens(detail) <= budget
        if kwargs.get("id") == 0 or kwargs.get("region") == "Region":
            assert not detail.endswith("…")
        for line in detail.splitlines():
            if line.startswith("+"):
                assert line.endswith(" more")
    assert "Usage: inspect" in g.inspect(p)


def test_outline_is_bounded_and_reports_omissions():
    out = UIGrounder().observe(page(2000), budget=50, landmarks=[f"Heading {i}" for i in range(30)])
    outline = next(line for line in out.splitlines() if line.startswith("headings:"))
    assert len(outline) <= 50 * 4 * 0.15
    assert outline.endswith("more")
    assert out.splitlines()[-1].endswith("inspect(id) for detail")


def test_blend_forwards_goal_and_history_to_cpu_model():
    if not Path("dist/xs-s0/model.int8.onnx").exists():
        pytest.skip("dist/xs-s0 not present")
    from grounder_tool.rankers import make_ranker

    ranker = make_ranker("blend:dist/xs-s0:8")
    runtime = ranker.model.grounder
    assert runtime.session.get_providers() == ["CPUExecutionProvider"]
    original = runtime.rank
    calls = []

    def capture(query, elements, **kwargs):
        calls.append((query, kwargs["mode"], kwargs["history"]))
        return original(query, elements, **kwargs)

    runtime.rank = capture
    history = ["[link] Docs -> CLICK"]
    ranked = ranker.rank("Install package", page(5).elements, "goal", history)
    assert len(ranked.ids) == 5
    assert calls == [("Install package", "goal", history)]


def test_keyword_goal_matches_exact_baseline():
    from grounder_tool.rankers import make_ranker, exact_bm25

    p = page(5)
    assert make_ranker("exact").rank("Install 3", p.elements, "goal", ["previous"]) == exact_bm25(
        "Install 3", p.elements
    )


def test_stored_pages_omit_outline_and_clip_context():
    p = page(1)
    p.elements[0].text = "t" * 100
    p.elements[0].context = "c" * 120
    for landmarks in (None, []):
        out = UIGrounder().observe(p, landmarks=landmarks)
        assert "headings:" not in out and "sections:" not in out
        line = next(line for line in out.splitlines() if "|link|" in line)
        assert len(line.split("|")[2]) == 80
        assert len(line.split("|")[3]) == 40


def test_heading_regions_and_fallback():
    g = UIGrounder()
    p = page(10)
    headings = [("Install", 40), ("Configure", 100), ("Finish", 160)]
    out = g.inspect(p, region="iNsT", heading_positions=headings)
    assert [int(line.split("|")[0]) for line in out.splitlines()] == [2, 3, 4]
    assert "9|link|" in g.inspect(p, region="Finish", heading_positions=headings)
    assert "0|link|" in g.inspect(p, region="Section 0", heading_positions=headings)
    assert "0|link|" in g.inspect(p, region="Section 0")
    for e in p.elements:
        e.abs_box = []
    assert "0|link|" in g.inspect(p, region="Section 0", heading_positions=[("Section 0", 40)])


@pytest.mark.parametrize("server_kind", ["tool", "demo"])
@pytest.mark.parametrize("budget,expected", [(-10, 100), (0, 100), (300, 300), (50000, 2000)])
def test_mcp_budget_clamp_and_shared_api(server_kind, budget, expected, monkeypatch):
    from types import SimpleNamespace
    from grounder_tool.mcp import ToolServer
    from demo.mcp_server import BenchServer

    monkeypatch.setattr(ToolServer, "_passages", lambda self: [])
    monkeypatch.setattr(BenchServer, "_passages", lambda self: [])
    calls = []

    class Capture:
        def observe(self, *args, **kwargs):
            calls.append((args, kwargs))
            return "ok"

    session = SimpleNamespace(extract=lambda: page(5), landmarks=["Install"], heading_positions=[("Install", 0)])
    if server_kind == "tool":
        server = ToolServer(UIGrounder(), observe_grounder=Capture())
        server.session = session
        assert server.observe(budget=budget) == "ok"
    else:
        server = BenchServer.__new__(BenchServer)
        server.session, server.digest = session, Capture()
        server.goal, server.box = "Install", SimpleNamespace(history=["a", "b", "c", "d"])
        server.extract_ms = server.rank_ms = 0
        assert server._t_observe(budget=budget) == "ok"
    args, kwargs = calls[0]
    assert args[2] == expected
    assert kwargs["landmarks"] == ["Install"]
    assert server.extract_ms > 0 and server.rank_ms > 0


def test_live_landmarks(tmp_path):
    from demo.browser import BrowserSession as DemoSession
    from grounder_tool.browser import BrowserSession as ToolSession

    html = tmp_path / "headings.html"
    html.write_text('''<title>Docs</title><nav aria-label="Navigation"><a>Home</a></nav>
        <main aria-labelledby="title"><h1 id="title">Install</h1>
        <h2 style="display:none">Hidden</h2><h2 style="visibility:hidden">Invisible</h2>
        <div aria-hidden="true"><h2>Aria hidden</h2></div>
        <h2>Packages</h2><a href="#">Package link</a><h2>Packages</h2>
        <div style="height:1000px"></div><h3>Below fold</h3></main>
        <aside>Unnamed</aside><footer aria-label="Footer"></footer>''')
    for cls in (DemoSession, ToolSession):
        session = cls(html.as_uri())
        try:
            session.extract()
            assert session.landmarks == ["Navigation", "Install", "Packages", "Below fold"]
            assert [name for name, _ in session.heading_positions] == ["Install", "Packages", "Packages", "Below fold"]
            assert session.heading_positions[-1][1] > 800
            session.page.evaluate("scrollTo(0, 700)")
            session.extract()
            assert session.heading_positions[-1][1] > 800
            session.page.set_content("".join(f"<h2>Heading {i} {'x' * 80}</h2>" for i in range(40)))
            session.extract()
            assert len(session.landmarks) == 30
            assert all(len(name) <= 60 for name in session.landmarks)
            assert len(session.heading_positions) == 40
        finally:
            session.close()
