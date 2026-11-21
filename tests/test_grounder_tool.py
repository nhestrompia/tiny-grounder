"""grounder_tool: rankers, api formatting, router, and the MCP tool handlers against local fixture pages."""

from __future__ import annotations

import functools
import http.server
import threading
from pathlib import Path

import pytest

from grounder.records import Element, Page
from grounder_tool import UIGrounder, format_lines, make_ranker
from grounder_tool.rankers import BM25Ranker, ModelRanker, RouterRanker, exact_bm25, lexical_coverage

FIXTURES = Path(__file__).parent / "fixtures" / "demo"
MODEL_DIR = Path(__file__).parent.parent / "dist" / "xs-s0"


def els() -> list[Element]:
    rows = [("link", "Pricing", "nav"), ("button", "Sign in", "header"), ("textbox", "Search docs", "header"),
            ("link", "Download the app", "footer"), ("button", "Save", "dialog")]
    return [Element(id=i, role=r, text=t, context=c, abs_box=[10 * i, 5, 40, 20]) for i, (r, t, c) in enumerate(rows)]


def test_bm25_and_exact():
    r = BM25Ranker().rank("pricing link", els())
    assert r.ids[0] == 0 and sorted(r.ids) == list(range(5))
    assert exact_bm25("click the Save button", els()).ids[0] == 4


def test_make_ranker_specs():
    assert isinstance(make_ranker("bm25"), BM25Ranker)
    for bad in ("", "model", "model:", "bm26", "bm25:x"):
        with pytest.raises(ValueError):
            make_ranker(bad)


def test_api_formats():
    g = UIGrounder("bm25")
    hits = g.find(els(), "search box", k=2)
    assert len(hits) == 2 and hits[0].id == 2
    line = format_lines(hits).splitlines()[0]
    assert line.startswith("2|textbox|Search docs|header|score=")
    assert g.snapshot(Page("p", "u", "s", "t", [1, 1], [1, 1], els()), "sign in", k=3)[0].id == 1
    assert g.find([], "x") == []
    assert "score=" not in hits[0].line(with_score=False)


def test_router_switches():
    class Stub:
        model_dir = Path("stub")
        calls = 0

        def rank(self, q, e, mode="do", history=None):
            from grounder.ranking import Ranked
            Stub.calls += 1
            return Ranked([e_.id for e_ in e][::-1], [0.0] * len(e), 0.2)

    router = RouterRanker(Stub(), tau=0.8)
    router.rank("Pricing", els())
    assert router.last_route == "lexical" and Stub.calls == 0
    router.rank("what does it cost", els())
    assert router.last_route == "model" and Stub.calls == 1
    assert lexical_coverage("pricing page", els()) == 0.5


@pytest.mark.skipif(not (MODEL_DIR / "model.int8.onnx").exists(), reason="dist/xs-s0 not present")
def test_model_ranker_and_router_spec():
    m = make_ranker(f"model:{MODEL_DIR}")
    assert isinstance(m, ModelRanker)
    r = UIGrounder(m).find(els(), "open the cart", k=3)
    assert len(r) == 3
    fp32 = make_ranker(f"model:{MODEL_DIR}", int8=False)
    assert len(fp32.rank("sign in", els()).ids) == 5
    router = make_ranker(f"router:{MODEL_DIR}:0.7")
    assert isinstance(router, RouterRanker) and router.tau == 0.7


# ---- MCP handlers --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def site() -> str:
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(FIXTURES)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


@pytest.fixture
def server():
    pytest.importorskip("playwright")
    from grounder_tool.mcp import ToolServer

    s = ToolServer(UIGrounder("bm25"))
    yield s
    s.close()


def first_id(text: str, role: str, label: str) -> int:
    for line in text.splitlines():
        parts = line.split("|")
        if len(parts) >= 3 and parts[1] == role and label.lower() in parts[2].lower():
            return int(parts[0])
    raise AssertionError(f"{label!r} not in:\n{text}")


def test_mcp_handlers(site, server):
    from grounder_tool.browser import ToolError

    with pytest.raises(ToolError):
        server.find("anything")
    with pytest.raises(ToolError):
        server.open("file:///etc/passwd")
    assert "url:" in server.open(f"{site}/index.html")
    found = server.find("documentation link", k=3)
    assert "score=" in found and len(found.splitlines()) == 3 + 3
    link = first_id(server.snapshot("read the documentation", k=20), "link", "documentation")
    assert "clicked" in server.click(link)
    assert server.session.url.endswith("docs.html")
    assert server.read_text(10_000)
    assert "truncated" in server.read_text(5)
    server.go_back()
    assert server.session.url.endswith("index.html")
    with pytest.raises(ToolError):
        server.click(99999)


def test_mcp_click_after_navigation_is_stale(site, server):
    from grounder_tool.browser import ToolError

    server.open(f"{site}/index.html")
    link = first_id(server.find("pricing", 5), "link", "pricing")
    server.open(f"{site}/docs.html")
    with pytest.raises(ToolError, match="changed"):
        server.click(link)


def test_build_mcp_registers_tools(server):
    from grounder_tool.mcp import build_mcp

    mcp, pool = build_mcp(server)
    try:
        import asyncio
        names = {t.name for t in asyncio.run(mcp.list_tools())}
    finally:
        pool.shutdown()
    assert names == {"open", "find", "snapshot", "click", "type", "go_back", "read_text", "observe", "inspect", "read"}


def test_blend_ranker_spec_and_probabilities():
    from grounder_tool.rankers import BlendRanker, make_ranker
    from grounder.records import Element

    r = make_ranker("blend:dist/xs-s0:8")
    assert isinstance(r, BlendRanker) and r.alpha == 8.0
    els = [Element(id=1, role="button", text="Save"), Element(id=2, role="link", text="Pricing"),
           Element(id=3, role="link", text="Contact us")]
    out = r.rank("open the pricing page", els)
    assert out.ids[0] == 2 and abs(sum(out.scores) - 1.0) < 1e-6 and all(0 <= s <= 1 for s in out.scores)
