"""Agent benchmark v2: HAR replay wiring, answer_regex grading, the capped / paginated a11y arm, ranker specs and the
paired report. No network (HAR files are generated here; hosts like bench.test do not resolve) and no agent CLI."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest

from demo import cli_bench, cli_report, mcp_server
from demo.agent import BM25Ranker
from demo.browser import BrowserSession, Task, ToolError, answer_matches, parse_tasks
from demo.mcp_server import BenchServer, SpecRanker, paginate

HOME = "http://bench.test/"


def har_entry(url: str, body: str, mime: str = "text/html; charset=utf-8", status: int = 200) -> dict:
    return {
        "startedDateTime": "2026-01-01T00:00:00.000Z", "time": 0,
        "request": {"method": "GET", "url": url, "httpVersion": "HTTP/1.1", "headers": [], "queryString": [], "cookies": [],
                    "headersSize": -1, "bodySize": 0},
        "response": {"status": status, "statusText": "OK", "httpVersion": "HTTP/1.1", "headers": [{"name": "Content-Type", "value": mime}],
                     "cookies": [], "content": {"size": len(body), "mimeType": mime, "text": body}, "redirectURL": "",
                     "headersSize": -1, "bodySize": len(body)},
        "cache": {}, "timings": {"send": 0, "wait": 0, "receive": 0},
    }


@pytest.fixture
def har(tmp_path: Path) -> Path:
    """A two-page archive: home links to /next (archived) and /gone (not archived) and loads an unarchived image."""
    home = ('<html><title>Home</title><body><h1>Home</h1><a href="/next">Next page</a> <a href="/gone">Gone page</a>'
            '<img src="/missing.png" alt="x"></body></html>')
    nxt = "<html><title>Next</title><body><h1>The answer is 42</h1><a href='/'>Back home</a></body></html>"
    path = tmp_path / "t.har"
    path.write_text(json.dumps({"log": {"version": "1.2", "creator": {"name": "test", "version": "1"},
                                        "entries": [har_entry(HOME, home), har_entry(HOME + "next", nxt)]}}), encoding="utf-8")
    return path


def server(har: Path, tmp_path: Path, config: str = "find", check: dict | None = None, **kw) -> BenchServer:
    session = BrowserSession(HOME, har=har)
    ranker = None if config in ("a11y", "a11y-cap", "vision") else BM25Ranker()
    return BenchServer(config, session, ranker, "Open the next page", check or {"url_contains": "/next"}, tmp_path / "log.jsonl",
                       "t", **kw)


def summary_of(srv: BenchServer) -> dict:
    srv.finish("exit")
    return [json.loads(line) for line in srv.log_path.read_text().splitlines()][-1]


# ---- HAR replay ------------------------------------------------------------------------------------------------------


def test_har_replay_serves_the_archive_and_stubs_the_rest(har: Path, tmp_path: Path) -> None:
    srv = server(har, tmp_path)
    try:
        assert srv.session.url == HOME and srv.session.page.title() == "Home"
        found = srv.invoke("find", query="next page")
        line = next(l for l in found.splitlines() if "Next page" in l)
        srv.invoke("click", id=int(line.split("|")[0]))
        assert srv.session.url == HOME + "next" and "answer is 42" in srv.session.text()
        # an unarchived page: the navigation lands on a "page not archived" stub, not on the network
        srv.session.page.goto(HOME + "gone")
        assert "not archived" in srv.session.text() and srv.session.url == HOME + "gone"
        replay = srv.session.replay
        assert replay is not None and replay.failed == []  # nothing reached the (dead) network
        assert any(u.endswith("/missing.png") for u in replay.misses) and HOME + "gone" in replay.misses
        assert replay.requests >= 4
    finally:
        srv.session.close()


def test_har_replay_abort_mode_fails_missing_navigations(har: Path) -> None:
    s = BrowserSession(HOME, har=har, replay_miss="abort")
    try:
        with pytest.raises(Exception):
            s.page.goto(HOME + "gone")
    finally:
        s.close()


def test_har_replay_needs_the_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        BrowserSession(HOME, har=tmp_path / "nope.har")


def test_replay_summary_reports_misses(har: Path, tmp_path: Path) -> None:
    srv = server(har, tmp_path)
    try:
        srv.invoke("find", query="next")
        s = srv.summary("exit")
        assert s["replay"] is True and s["replay_misses"] >= 1 and s["replay_network_failures"] == 0
        assert s["success"] is False  # the do-nothing run fails the strict check
    finally:
        srv.session.close()


def test_task_har_resolution() -> None:
    t = Task("a", "https://x.test/", "g", {"url_contains": "x"}, 3, har="data/bench_v2/har/a.zip")
    assert cli_bench.task_har(t, live=True) is None
    assert cli_bench.task_har(t, live=False) == cli_bench.REPO / "data/bench_v2/har/a.zip"
    assert cli_bench.task_har(Task("b", "https://x.test/", "g", {"url_contains": "x"}, 3), live=False) is None


# ---- answer_regex grading --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "ok"),
    [("It says the answer is 42.", True), ("THE ANSWER IS 42", True), ("forty-two", False), ("", False)])
def test_answer_regex_is_part_of_success(har: Path, tmp_path: Path, answer: str, ok: bool) -> None:
    srv = server(har, tmp_path, answer_regex=r"answer is 42")
    try:
        srv.invoke("click", id=int(next(l for l in srv.invoke("find", query="next").splitlines() if "Next page" in l).split("|")[0]))
        srv.invoke("done", answer=answer)
        events = [json.loads(line) for line in srv.log_path.read_text().splitlines()]
        s = events[-1]
        assert s["check_ok"] is True and s["answer_ok"] is ok and s["success"] is ok
        assert s["answer"] == answer and s["answer_regex"] == r"answer is 42"
    finally:
        srv.session.close()


def test_answer_without_the_page_check_fails_and_no_regex_means_check_only(har: Path, tmp_path: Path) -> None:
    srv = server(har, tmp_path, answer_regex="42")
    try:
        srv.invoke("done", answer="42")  # right answer, wrong page
        s = summary_of(srv)
        assert s["answer_ok"] is True and s["check_ok"] is False and s["success"] is False
    finally:
        srv.session.close()
    assert answer_matches("", "anything") and not answer_matches("x", "") and answer_matches("A+", "aaa")


def test_not_calling_done_fails_an_answer_task(har: Path, tmp_path: Path) -> None:
    srv = server(har, tmp_path, answer_regex="")
    try:
        assert summary_of(srv)["success"] is False
    finally:
        srv.session.close()
    srv = server(har, tmp_path / "b", answer_regex="42")
    try:
        srv.session.page.goto(HOME + "next")
        s = srv.summary("exit")
        assert s["check_ok"] is True and s["answer_ok"] is False and s["success"] is False  # on the page, never answered
    finally:
        srv.session.close()


def test_parse_tasks_reads_v2_fields_and_rejects_bad_regex() -> None:
    raw = {"id": "a", "start_url": "https://x.test/", "goal": "g", "check": {"url_contains": "x"}, "max_steps": 3,
           "answer_regex": r"4\d", "tags": ["vague", "long"], "har": "data/bench_v2/har/a.zip", "path": [], "answer": "42"}
    t = parse_tasks([raw])[0]
    assert (t.answer_regex, t.tags, t.har) == (r"4\d", ("vague", "long"), "data/bench_v2/har/a.zip")
    with pytest.raises(ValueError, match="answer_regex"):
        parse_tasks([{**raw, "answer_regex": "("}])
    with pytest.raises(ValueError, match="tags"):
        parse_tasks([{**raw, "tags": "vague"}])


# ---- capped, paginated a11y --------------------------------------------------------------------------------------------


def test_paginate_is_lossless_and_capped() -> None:
    text = "\n".join(f"line {i} " + "x" * (i % 7) for i in range(200))
    pages = paginate(text, 300)
    assert "\n".join(pages) == text and all(len(p) <= 300 for p in pages) and len(pages) > 5
    assert paginate("short", 300) == ["short"]
    assert paginate("a" * 250, 100) == ["a" * 100, "a" * 100, "a" * 50]  # a line longer than the cap is cut hard


def test_a11y_cap_snapshot_is_capped_and_paginated(har: Path, tmp_path: Path) -> None:
    srv = server(har, tmp_path, "a11y-cap", a11y_cap=120)
    try:
        p1 = srv.invoke("snapshot")
        assert re.search(r"snapshot page 1 of (\d+)", p1) and "snapshot(page=2)" in p1
        total = int(re.search(r"page 1 of (\d+)", p1).group(1))
        assert total >= 2
        body1 = p1.split("]\n", 1)[1]
        assert len(body1) <= 120
        p2 = srv.invoke("snapshot", page=2)
        assert f"page 2 of {total}" in p2 and p2.split("]\n", 1)[1] != body1
        last = srv.invoke("snapshot", page=total)
        assert "last page" in last
        with pytest.raises(ToolError, match="does not exist"):
            srv.invoke("snapshot", page=total + 1)
        with pytest.raises(ToolError):
            srv.invoke("snapshot", page=0)
        assert all(r["name"] == "snapshot" for r in [json.loads(l) for l in srv.log_path.read_text().splitlines()] if r["type"] == "call")
    finally:
        srv.session.close()


def test_a11y_cap_pages_belong_to_one_snapshot_and_actions_return_page_one(har: Path, tmp_path: Path) -> None:
    srv = server(har, tmp_path, "a11y-cap", a11y_cap=150)
    try:
        full = "\n".join(srv.invoke("snapshot", page=p).split("]\n", 1)[1] for p in range(1, 3))
        ref = re.search(r'link "Next page"[^\n]*\[ref=(\w+)\]', srv._aria_cache[1])
        assert ref, srv._aria_cache[1]
        out = srv.invoke("click", id=ref.group(1))  # ref taken from the whole snapshot, whichever page it was on
        assert "clicked" in out and "/next" in out and "snapshot page 1 of" in out
        assert srv.session.url == HOME + "next"
        assert full  # pages concatenate
        assert summary_of(srv)["success"] is True
    finally:
        srv.session.close()


def test_snapshot_page_is_only_for_a11y_cap(har: Path, tmp_path: Path) -> None:
    srv = server(har, tmp_path, "full")
    try:
        with pytest.raises(ToolError):
            srv.invoke("snapshot", page=2)
    finally:
        srv.session.close()


def test_mcp_a11y_cap_tool_states_cap_and_pagination() -> None:
    app = mcp_server.build_mcp("a11y-cap", lambda name, **kw: "ok")
    tools = {t.name: t for t in asyncio.run(app.list_tools())}
    assert sorted(tools) == sorted(mcp_server.tools_for("a11y-cap"))
    desc = tools["snapshot"].description
    assert "20,000" in desc and "page" in desc and "snapshot(page=2)" in desc
    assert "page" in tools["snapshot"].inputSchema["properties"]
    assert "20,000" in mcp_server.CONFIG_HINTS["a11y-cap"] and "page" in mcp_server.CONFIG_HINTS["a11y-cap"]
    assert "page" not in {p for t in asyncio.run(mcp_server.build_mcp("a11y", lambda n, **k: "ok").list_tools())
                          if t.name == "snapshot" for p in t.inputSchema.get("properties", {})}


# ---- arms and rankers -------------------------------------------------------------------------------------------------


def test_parse_arm_and_labels() -> None:
    assert cli_bench.parse_arm("a11y-cap") == cli_bench.Arm("a11y-cap", "")
    arm = cli_bench.parse_arm("find@blend:dist/xs-s0:8")
    assert arm == cli_bench.Arm("find", "blend:dist/xs-s0:8") and arm.label == "find[blend]"
    assert cli_bench.parse_arm("find@exact").label == "find[exact]"
    assert cli_bench.parse_arm("find", "bm25") == cli_bench.Arm("find", "bm25")
    assert cli_bench.arm_label(cli_bench.parse_arm("find", "bm25"), explicit=False) == "find"
    for bad in ("nope", "vision@exact", "find@"):
        if bad == "find@":
            assert cli_bench.parse_arm(bad, "x").ranker == ""  # empty ranker after @: server falls back to the default
            continue
        with pytest.raises(ValueError):
            cli_bench.parse_arm(bad)


def test_resolve_ranker_spec_makes_model_dirs_absolute(tmp_path: Path) -> None:
    assert cli_bench.resolve_ranker_spec("exact") == "exact" and cli_bench.resolve_ranker_spec("bm25") == "bm25"
    assert cli_bench.resolve_ranker_spec("blend:dist/xs-s0:8", tmp_path) == f"blend:{tmp_path / 'dist/xs-s0'}:8"
    assert cli_bench.resolve_ranker_spec("model:dist/xs", tmp_path) == f"model:{tmp_path / 'dist/xs'}"
    assert cli_bench.resolve_ranker_spec(f"router:{tmp_path}/m:0.5", tmp_path) == f"router:{tmp_path}/m:0.5"


def test_server_argv_carries_replay_and_answer(tmp_path: Path) -> None:
    t = Task("a", "https://x.test/", "g", {"url_contains": "x"}, 3, answer_regex=r"4\d")
    argv = cli_bench.server_argv(t, "find", "exact", tmp_path, tmp_path / "s.jsonl", 40, True, tmp_path / "a.zip", "abort")
    args = mcp_server.parse_args(argv[3:])
    assert (args.har, args.replay_miss, args.answer_regex, args.ranker) == (str(tmp_path / "a.zip"), "abort", r"4\d", "exact")
    live = mcp_server.parse_args(cli_bench.server_argv(t, "find", "exact", tmp_path, tmp_path / "s.jsonl", 40, True)[3:])
    assert live.har is None


def test_spec_ranker_prints_the_same_find_lines_for_every_ranker(har: Path, tmp_path: Path) -> None:
    outputs = {}
    for spec in ("exact", "bm25"):
        srv = BenchServer("find", BrowserSession(HOME, har=har), SpecRanker(spec), "g", {"url_contains": "x"}, tmp_path / f"{spec}.jsonl")
        try:
            outputs[spec] = srv.invoke("find", query="next page")
        finally:
            srv.session.close()
    for spec, text in outputs.items():
        rows = text.splitlines()[3:]  # url, title, count line
        assert rows and len(rows) <= 5
        for r in rows:
            assert re.fullmatch(r"\d+\|[a-z]+\|.*\|.*\|score=\d\.\d{3}", r), r
        scores = [float(r.rsplit("score=", 1)[1]) for r in rows]
        assert scores == sorted(scores, reverse=True) and 0.0 <= min(scores) and max(scores) <= 1.0
    assert "Next page" in outputs["exact"].splitlines()[3]


def test_spec_ranker_blend_uses_the_shipped_code() -> None:
    model_dir = Path(__file__).resolve().parent.parent / "dist" / "xs-s0"
    if not model_dir.exists():
        pytest.skip("dist/xs-s0 not built")
    r = SpecRanker(f"blend:{model_dir}:8")
    from grounder.records import Element

    els = [Element(0, "link", "Home"), Element(1, "link", "Contact us"), Element(2, "button", "Search")]
    out = r.rank("contact", els, k=2)
    assert out["results"][0]["id"] == 1 and len(out["results"]) == 2 and all(0 <= x["score"] <= 1 for x in out["results"])


def test_make_ranker_legacy_names_still_work() -> None:
    assert isinstance(mcp_server.make_ranker("bm25", "x"), BM25Ranker)
    assert isinstance(mcp_server.make_ranker("exact", "x"), SpecRanker)


# ---- report: paired analysis ----------------------------------------------------------------------------------------


def run(agent: str, arm: str, task_id: str, ok: bool, site: str, repeat: int = 0, tokens: float = 400, **kw) -> dict:
    return {"agent": agent, "config": arm.split("[")[0], "arm": arm, "task_id": task_id, "site": site, "repeat": repeat,
            "success": ok, "done_called": True, "wall_seconds": 5.0, "tool_calls": 3, "tool_chars": 1000, "image_bytes": 0,
            "tool_tokens_est": tokens, "foreign_tools": [], "timed_out": False, "agent_error": "", "tags": kw.pop("tags", []), **kw}


def synthetic_runs() -> list[dict]:
    runs = []
    for i in range(24):
        site = f"s{i // 2}"
        tid = f"t{i}"
        for rep in range(2):
            runs.append(run("claude:m", "find[blend]", tid, True, site, rep, tokens=300, tags=["vague"] if i % 2 else ["icon"]))
            runs.append(run("claude:m", "find[exact]", tid, i % 4 != 0 or rep == 0, site, rep, tokens=300, tags=["vague"] if i % 2 else ["icon"]))
            runs.append(run("claude:m", "a11y-cap", tid, i % 3 != 0, site, rep, tokens=2000, tags=["vague"] if i % 2 else ["icon"]))
            runs.append(run("claude:m", "vision", tid, True, site, rep, tokens=1500, tags=["vague"] if i % 2 else ["icon"]))
    return runs


def test_paired_analysis_differences_cis_and_tokens() -> None:
    runs = synthetic_runs()
    text = cli_report.paired_analysis(runs)
    assert "## Paired analysis" in text and "### claude:m" in text
    exact = cli_report.task_table(runs, "claude:m", "find[exact]")
    blend = cli_report.task_table(runs, "claude:m", "find[blend]")
    assert len(exact) == 24 and exact["t0"]["rate"] == 0.5 and exact["t1"]["rate"] == 1.0 and exact["t0"]["site"] == "s0"
    n, d, lo, hi = cli_report.paired_success(blend, exact)
    # 6 of 24 tasks (i % 4 == 0) have exact at 50%: mean diff = 6 * 0.5 / 24 = 12.5 points
    assert n == 24 and d == pytest.approx(12.5) and lo <= d <= hi and lo > 0
    assert cli_report.paired_success(blend, exact) == cli_report.paired_success(blend, exact)  # seeded: deterministic
    assert cli_report.paired_success(blend, exact, cluster="task")[1] == pytest.approx(12.5)
    ta, tb, ratio, rlo, rhi = cli_report.paired_tokens(blend, cli_report.task_table(runs, "claude:m", "a11y-cap"))
    # blend: 300 tokens per success; a11y-cap: 2000 tokens x 48 runs / 32 successes = 3000 per success
    assert (ta, tb, ratio) == pytest.approx((300, 3000, 0.1)) and rlo <= ratio <= rhi
    assert cli_report.best_baseline(runs, "claude:m") == "vision"  # 100% success beats a11y-cap's 67%
    assert "find[blend] vs find[exact]" in text and "find[blend] vs vision" in text
    assert "non-inferior" in text and ">=25% fewer tokens: YES" in text
    assert "| icon |" in text and "| vague |" in text  # per-tag table


def test_paired_analysis_skipped_without_blend_arm_and_results_table_uses_arms() -> None:
    v1 = [{"agent": "a", "config": "find", "task_id": "t1", "repeat": 0, "success": True, "done_called": True, "wall_seconds": 1,
           "tool_calls": 1, "tool_chars": 40, "image_bytes": 0, "foreign_tools": []}]
    assert cli_report.paired_analysis(v1) == ""
    table = cli_report.results_table(synthetic_runs())
    assert "find[blend]" in table and "tool tokens / success" in table and "a11y-cap" in table


def test_cluster_bootstrap_resamples_whole_sites() -> None:
    import numpy as np

    units = [("a", 1.0)] * 8 + [("b", 0.0)] * 8  # perfectly clustered
    p, lo, hi = cli_report.cluster_bootstrap(units, lambda xs: float(np.mean(xs)), "site", reps=500)
    p2, lo2, hi2 = cli_report.cluster_bootstrap(units, lambda xs: float(np.mean(xs)), "task", reps=500)
    assert p == p2 == 0.5 and (lo, hi) == (0.0, 1.0) and 0.0 < lo2 and hi2 < 1.0  # site clustering widens the interval


def test_write_results_includes_paired_section(tmp_path: Path) -> None:
    (tmp_path / cli_bench.LOGS_DIR).mkdir()
    for i, r in enumerate(synthetic_runs()):
        (tmp_path / cli_bench.LOGS_DIR / f"{i}.json").write_text(json.dumps(r), encoding="utf-8")
    assert "## Paired analysis" in cli_report.write_results(tmp_path)


# ---- click robustness on long pages with sticky, inner-scrolling sidebars (docs tables of contents) ---------------------


def test_click_finds_an_element_that_moved_with_a_sticky_scroller(tmp_path: Path) -> None:
    links = "".join(f'<a href="#x{i}" style="position:absolute;top:{i * 40}px">Filler {i}</a>' for i in range(100))
    (tmp_path / "long.html").write_text(
        "<html><body style='margin:0'><aside style='position:sticky;top:0;float:right;width:200px;height:300px;overflow:auto'>"
        f"<div style='position:relative;height:6000px'>{links}"
        '<a href="target.html" style="position:absolute;top:5000px">Deep target</a></div></aside>'
        "<main style='height:30000px'>text</main></body></html>", encoding="utf-8")
    (tmp_path / "target.html").write_text("<html><title>arrived</title><body>arrived</body></html>", encoding="utf-8")
    s = BrowserSession((tmp_path / "long.html").as_uri())
    try:
        page = s.extract()
        el = next(e for e in page.elements if e.text == "Deep target")
        assert el.abs_box[1] > 4000  # extracted far down the document, inside the sticky scroller
        s.click(el.id)
        assert s.url.endswith("/target.html")
    finally:
        s.close()


def test_replay_detects_live_responses_and_sees_none_when_offline(har: Path, tmp_path: Path) -> None:
    import functools
    import http.server
    import threading

    from demo.browser import Replay

    srv = server(har, tmp_path)
    try:
        srv.invoke("find", query="next")
        assert srv.summary("exit")["replay_live_requests"] == 0 and srv.session.replay.live == []
    finally:
        srv.session.close()
    # control: the same detector on a session that really uses the network (a local server) must fire
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(Path(__file__).parent / "fixtures" / "demo"))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    live = BrowserSession(f"http://127.0.0.1:{httpd.server_port}/index.html")
    try:
        probe = Replay()
        live.context.on("response", probe.on_response)
        live.page.reload()
        live.page.wait_for_timeout(300)  # response events are delivered asynchronously
        assert probe.live, "a real network response must be detected"
    finally:
        live.close()
        httpd.shutdown()
