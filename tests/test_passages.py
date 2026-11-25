from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

from grounder.playwright_helper import Passage, extract_passages
from grounder.records import Element, Page
from grounder_tool.api import UIGrounder
from grounder_tool.digest import estimate_tokens, rank_passages, read


def test_extract_local(tmp_path):
    html = tmp_path / 'text.html'
    html.write_text('''<nav><p>Navigation noise</p></nav><main><h1>Install</h1>
    <p>Manual pages live in <code>$prefix/share/man</code>.</p>
    <p><a href="/other">Other guide</a></p>
    <p hidden>Hidden</p><p style="display:none">Display hidden</p>
    <p style="visibility:hidden">Invisible</p><p style="opacity:0">Transparent</p>
    <div aria-hidden="true"><p>Aria hidden</p></div>
    <table><tr><th>Name</th><th>Path</th></tr><tr><td>Man</td><td><p>/usr/man</p></td></tr></table>
    <h4>Configuration</h4><pre><code>set prefix=/usr</code></pre>
    <dl><dt>Prefix</dt><dd>Install root</dd></dl><blockquote><p>A quotation</p></blockquote>
    <figure><figcaption>A caption</figcaption></figure>
    <ul><li><p>List paragraph</p></li></ul></main><footer><p>Footer noise</p></footer>''')
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        tab = browser.new_page()
        tab.goto(html.as_uri())
        passages = extract_passages(tab)
        texts = [p.text for p in passages]
        assert 'Navigation noise' not in texts and 'Footer noise' not in texts
        assert not any('hidden' in text.lower() or text in ('Invisible', 'Transparent') for text in texts)
        assert 'Other guide' not in texts
        assert texts.count('set prefix=/usr') == 1
        assert texts.count('A quotation') == 1
        assert texts.count('List paragraph') == 1
        assert sum('/usr/man' in text for text in texts) == 1
        assert any('Name' in text and 'Path' in text for text in texts)
        assert next(p for p in passages if '$prefix' in p.text).heading == 'Install'
        assert next(p for p in passages if p.text == 'Install root').heading == 'Configuration'
        assert [p.y for p in passages] == sorted(p.y for p in passages)
        tab.set_content('<nav><p>Only navigation content</p></nav>')
        assert extract_passages(tab)[0].text == 'Only navigation content'
        tab.set_content('<p>' + 'word ' * 200 + '</p>' + '<p>Block</p>' * 2100)
        passages = extract_passages(tab)
        assert len(passages) == 2000 and len(passages[0].text) == 300
        browser.close()


def test_rank_ties_and_no_match():
    passages = [Passage('manual installation', y=10), Passage('manual installation', y=20)]
    assert [p.y for p, _ in rank_passages('manual installation', passages)] == [10, 20]
    assert rank_passages('unrelated zebras', passages) == []
    assert rank_passages('', passages) == []


class _ReverseRanker:
    """Non-lexical stub: returns every element in reverse order; stands in for a model ranker."""
    name = "model:stub"

    def rank(self, query, elements, mode="do", history=None):
        from grounder.ranking import Ranked
        ids = [e.id for e in elements][::-1]
        return Ranked(ids, [1.0] * len(ids), 0.0)


def test_read_and_observe_use_ranker_for_passages():
    passages = [Passage('manual installation', y=1), Passage('zebra text unrelated to query', y=2)]
    # Lexical ranking puts the matching passage first; the ranker's ordering wins when one is supplied.
    assert read('manual installation', passages, 400, _ReverseRanker()).splitlines()[0].startswith('> zebra')
    # The default exact ranker keeps lexical ordering and the zero-score drop: identical output to no ranker.
    assert read('manual installation', passages, 400, UIGrounder().ranker) == read('manual installation', passages, 400)
    # BM25 scores can be negative on matches; a lexical ranker must not drop them or keep non-matches.
    from grounder_tool.rankers import BM25Ranker
    bm25 = BM25Ranker()
    assert [p.y for p, _ in rank_passages('manual installation', passages, bm25)]
    assert rank_passages('kubernetes pods', passages, bm25) == []
    assert rank_passages('q', [], bm25) == []
    page = Page('p', 'https://example.test', 'example', 'Manual installation', [1280, 800], [1280, 5000],
                [Element(i, 'link', f'Manual installation {i}') for i in range(50)])
    out = UIGrounder(_ReverseRanker()).observe(page, 'manual installation', budget=600, passages=passages)
    assert out.splitlines()[2].startswith('> zebra')


@pytest.mark.parametrize('budget', [0, 1, 10, 50, 100, 400, 600, 2000])
def test_observe_passage_budget(budget):
    page = Page('p', 'https://example.test', 'example', 'Manual installation', [1280,800], [1280,5000],
                [Element(i, 'link', f'Manual installation {i}') for i in range(500)])
    passages = [Passage('Manual installation ' + 'text ' * 100, 'Installation ' * 20) for _ in range(100)]
    g = UIGrounder()
    out = g.observe(page, budget=budget, passages=passages, landmarks=['Installation'])
    assert estimate_tokens(out) <= budget
    assert sum(len(line)+1 for line in out.splitlines() if line.startswith('>')) <= budget * 4 * .3
    assert g.observe(page, budget=budget) == g.observe(page, budget=budget, passages=None)
    if budget >= 400:
        assert '> Manual installation' in out


@pytest.mark.parametrize('budget', [-100, 0, 100, 400, 2000, 50000])
def test_read_budget_clamp(budget):
    passages = [Passage('Manual installation ' + 'text ' * 80, 'Install') for _ in range(100)]
    out = read('Manual installation', passages, budget)
    assert estimate_tokens(out) <= max(100, min(2000, budget))
    assert out.splitlines()[-1].endswith('of 100 passages')
    assert '[§ Install]' in out
    assert read('manual', passages, -1) == read('manual', passages, 100)
    assert read('manual', passages, 99999) == read('manual', passages, 2000)
    assert read('zebras', passages) == 'showing 0 of 100 passages'
    assert read('manual', []) == 'showing 0 of 0 passages'


def test_servers_expose_read():
    from demo.mcp_server import build_mcp, tools_for
    from grounder_tool.mcp import ToolServer, build_mcp as tool_mcp
    assert 'read' in tools_for('find+observe')
    assert 'read' not in tools_for('find')
    app = build_mcp('find+observe', lambda name, **kw: 'ok')
    assert 'read' in {t.name for t in asyncio.run(app.list_tools())}
    for endpoint in ('', 'http://localhost:9222'):
        app, pool = tool_mcp(ToolServer(UIGrounder(), cdp_endpoint=endpoint))
        try:
            assert 'read' in {t.name for t in asyncio.run(app.list_tools())}
        finally:
            pool.shutdown()


@pytest.mark.parametrize('task_id,url,question,answer', [
    ('open-mpi-index-3', 'https://docs.open-mpi.org/en/v5.0.x/man-openmpi/index.html',
     'where are man pages installed', '$prefix/share/man'),
    ('htslib-htsfile', 'https://www.htslib.org/doc/htsfile.html',
     'Which three sequencing data formats can htsfile identify?', 'CRAM'),
])
def test_har_facts(tmp_path, task_id, url, question, answer):
    from demo.browser import BrowserSession, load_tasks
    from demo.mcp_server import BenchServer, make_ranker
    task = next(t for t in load_tasks('demo/tasks_milestone.yaml') if t.id == task_id)
    if not Path(task.har).exists():
        pytest.skip('HAR absent')
    session = BrowserSession(url, har=task.har)
    server = BenchServer('find+observe', session, make_ranker('exact',''), task.goal, task.check, tmp_path/'log.jsonl')
    try:
        observed = server.invoke('observe', budget=600)
        result = server.invoke('read', query=question)
        assert answer in observed or answer in result
        assert estimate_tokens(observed) <= 600 and estimate_tokens(result) <= 400
        assert result == server.invoke('read', query=question, budget=400)
    finally:
        server.finish('test')
        session.close()
    events = [json.loads(line) for line in (tmp_path/'log.jsonl').read_text().splitlines()]
    assert all(e['extract_ms'] > 0 and e['rank_ms'] > 0 for e in events if e['type'] == 'call')


@pytest.mark.parametrize('endpoint', ['', 'http://localhost:9222'])
def test_tool_live_read_and_refresh(tmp_path, endpoint):
    from grounder_tool.browser import BrowserSession
    from grounder_tool.mcp import ToolServer
    html = tmp_path/'facts.html'
    html.write_text('<h1>Installation</h1><p>Manual pages are installed in $prefix/share/man.</p>')
    session = BrowserSession(html.as_uri())
    server = ToolServer(UIGrounder(), cdp_endpoint=endpoint)
    server.session = session
    try:
        assert '$prefix/share/man' in server.observe('manual pages installed')
        assert '$prefix/share/man' in server.read('manual pages installed')
        assert server.read('manual', 0) == server.read('manual', 100)
        assert server.read('manual', 99999) == server.read('manual', 2000)
        session.page.set_content('<p>Manual pages are installed in /changed/man.</p>')
        assert '/changed/man' in server.read('manual pages installed')
        assert '$prefix' not in server.read('manual pages installed')
    finally:
        server.close()


def test_leaf_numbers(tmp_path):
    """Short standalone numbers (prices, totals, counts) in leaf nodes become passages."""
    html = tmp_path / 'numbers.html'
    html.write_text('''<main><h1>Item</h1>
    <div class="price">$15.99</div>
    <p>Free shipping</p>
    <div><a href="/buy">Buy now $15.99</a></div>
    <div><span>only <b>3</b> left</span></div>
    <span>no digits here</span>
    <div style="display:none">$0.00</div>
    <p> Paragraph has a digit: 42</p></main>
    <footer><div>$9.99</div></footer>''')
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        tab = browser.new_page()
        tab.goto(html.as_uri())
        texts = [p.text for p in extract_passages(tab)]
        assert texts.count('$15.99') == 1
        assert '3' in texts
        assert 'Buy now $15.99' not in texts  # control text, covered by element extraction
        assert 'no digits here' not in texts
        assert '$0.00' not in texts and '$9.99' not in texts
        assert ' Paragraph has a digit: 42' in texts or 'Paragraph has a digit: 42' in texts
        browser.close()
