from __future__ import annotations

import hashlib
import io
import json
import tarfile
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from grounder.data.pages import load_examples, load_pages
from tools import deskforge
from tools.deskforge import Job, LeafCache, Member, Outcome, Sampler


def el(uid: str | None, type_: str, role: str, rect: tuple[int, int, int, int], name: str = "", app: str = "calc",
       order: int = 0, stack: int = 1, actionable: bool = True, **kw: Any) -> dict[str, Any]:
    x, y, w, h = rect
    r = {"x": x, "y": y, "w": w, "h": h}
    base = {"type": type_, "role": role, "name": name, "visible_text": kw.pop("visible_text", name), "rect": r,
            "visible_fragments": kw.pop("visible_fragments", [r]), "is_occluded": kw.pop("is_occluded", False),
            "app_name": app, "reading_order_index": order, "_window_stack_index": stack, "uid": uid,
            "interaction": {"actionable": actionable, "editable": kw.pop("editable", False), "enabled": True}}
    return {**base, **kw}


def leaf() -> list[dict[str, Any]]:
    return [
        el("w1", "Window", "frame", (0, 0, 1000, 500), "Calculator", order=0, actionable=False),
        el("lbl", "Text", "label", (10, 10, 100, 20), "Result", order=1, actionable=False),
        el("ok", "Button", "push button", (10, 40, 80, 30), "OK", order=2),
        el("txt", "Text", "table cell", (10, 80, 80, 20), "static cell", order=3),
        el("hid", "Button", "push button", (10, 120, 80, 30), "Hidden", order=4, is_occluded=True, visible_fragments=[]),
        el("part", "Button", "push button", (100, 40, 100, 50), "Part", order=5, is_occluded=True,
           visible_fragments=[{"x": 100, "y": 40, "w": 40, "h": 50}, {"x": 120, "y": 80, "w": 60, "h": 10}]),
        el("sw", "Toggles", "toggle button", (300, 40, 40, 20), "", order=6),
        el("inp", "Text Input", "entry", (300, 80, 100, 20), "", order=7, actionable=False, editable=True, visible_text="hello"),
        el("early", "Link", "link", (400, 40, 50, 20), "Docs", order=-1),
    ]


def test_role_mapping_table() -> None:
    cases = [("Button", "push button", "button"), ("Link", "link", "link"), ("Text Input", "entry", "textbox"),
             ("Search Field", "entry", "searchbox"), ("Checkbox", "check box", "checkbox"), ("Radiobox", "radio button", "radio"),
             ("Toggles", "toggle button", "switch"), ("Select", "combo box", "combobox"), ("Menu", "menu", "menuitem"),
             ("Tab", "page tab", "tab"), ("Slider", "slider", "slider"), ("List Item", "tree item", "option"),
             ("File Icon", "icon", "option"), ("Image", "drawing area", "image"),
             ("Heading", "table column header", "button"), ("Heading", "title bar", None),
             ("Text", "table cell", None), ("Window", "frame", None), ("Scroll", "scroll bar", None),
             ("Unseen Class", "push button", "button"), ("Unseen Class", "section", None)]
    for type_, role, expected in cases:
        assert deskforge.map_role(type_, role) == expected, (type_, role)


def test_roles_are_valid_records_roles() -> None:
    from grounder.records import ROLES

    assert {r for r in [*deskforge.TYPE_ROLES.values(), *deskforge.ROLE_ROLES.values()] if r} <= set(ROLES)


def test_screen_keeps_interactive_visible_elements_in_reading_order() -> None:
    screen = deskforge.build_screen(leaf(), 1000, 500)
    assert [(e.text, e.role) for e in screen.elements] == [("Docs", "link"), ("OK", "button"), ("Part", "button"), ("", "switch"), ("hello", "textbox")]
    assert [e.id for e in screen.elements] == [0, 1, 2, 3, 4]
    assert screen.uid_to_id == {"early": 0, "ok": 1, "part": 2, "sw": 3, "inp": 4}
    assert screen.elements[1].source == "ax" and screen.elements[1].tag == "push button"


def test_drops_are_counted_and_attributed_to_uids() -> None:
    screen = deskforge.build_screen(leaf(), 1000, 500)
    assert screen.rejected["hid"] == "fully_occluded" and screen.rejected["txt"] == "not_candidate"
    assert screen.dropped == {"not_candidate": 2, "fully_occluded": 1}  # the window and the static cell; the label is context


def test_box_is_the_union_of_visible_fragments_not_the_amodal_rect() -> None:
    screen = deskforge.build_screen(leaf(), 1000, 500)
    part = screen.elements[2]
    assert part.abs_box == [100.0, 40.0, 80.0, 50.0]  # rect says 100 wide; the fragments span x 100..180
    assert part.box == [0.1, 0.08, 0.08, 0.1]
    assert deskforge.visible_box(el("a", "Button", "push button", (5, 6, 7, 8))) == [5.0, 6.0, 7.0, 8.0]
    assert deskforge.visible_box(el("a", "Button", "push button", (5, 6, 0, 8))) is None


def test_empty_box_and_element_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    flat = el("flat", "Button", "push button", (0, 0, 0, 0), "Flat")
    assert deskforge.build_screen([flat], 100, 100).rejected == {"flat": "empty_box"}
    monkeypatch.setattr(deskforge, "MAX_ELEMENTS", 2)
    screen = deskforge.build_screen(leaf(), 1000, 500)
    assert len(screen.elements) == 2 and screen.dropped["over_cap"] == 3


def test_text_prefers_visible_text_then_name_and_drops_object_replacement() -> None:
    assert deskforge.leaf_text(el("a", "Button", "push button", (0, 0, 1, 1), "Name", visible_text="Shown")) == "Shown"
    assert deskforge.leaf_text(el("a", "Button", "push button", (0, 0, 1, 1), "Name", visible_text="")) == "Name"
    assert deskforge.leaf_text(el("a", "Window", "frame", (0, 0, 1, 1), "Title", visible_text="￼")) == "Title"


def test_context_is_app_window_title_and_preceding_labels_in_the_same_window() -> None:
    labels = [el(None, "Text", "label", (0, 0, 9, 9), f"L{i}", order=i + 1, actionable=False) for i in range(4)]
    other = el(None, "Text", "label", (0, 0, 9, 9), "Elsewhere", order=2, stack=9, actionable=False)
    frame = el("w", "Window", "frame", (0, 0, 100, 100), "Calculator", order=0, actionable=False)
    button = el("b", "Button", "push button", (0, 0, 9, 9), "Go", order=10)
    screen = deskforge.build_screen([button, other, *labels, frame], 100, 100)
    assert screen.elements[0].context == "calc · Calculator · L1 L2 L3"


def test_page_takes_site_title_and_viewport_from_the_clicked_element() -> None:
    elements = [el("a", "Button", "push button", (0, 0, 10, 10), "A", app="calc"), el("b", "Button", "push button", (0, 0, 10, 10), "B", app="Sys Mon", stack=2)]
    screen = deskforge.build_screen([el("w", "Window", "frame", (0, 0, 9, 9), "Monitor", app="Sys Mon", stack=2, actionable=False), *elements], 1366, 768)
    page = deskforge.page_for("shard-0__scene-1-step00", screen, (1366, 768), screen.uid_to_id["b"])
    assert (page.site, page.url, page.title, page.viewport, page.doc_size) == ("deskforge-sys-mon", "ax://deskforge/Sys Mon", "Monitor", [1366, 768], [1366, 768])
    assert page.page_id == deskforge.page_id_for("shard-0__scene-1-step00") and page.page_id.startswith("deskforge-")
    untitled = deskforge.page_for("k", screen, (1, 1), screen.uid_to_id["a"])
    assert untitled.title == "calc"


def row(key: str = "k1", uid: str = "ok", n: int = 0, variants: list[tuple[str, str]] | None = None) -> dict[str, Any]:
    variants = variants if variants is not None else [("standard", f"Press OK {n}"), ("detailed_contextual", f"In the calculator press OK {n}")]
    return {"transition_id": f"t{n}", "before_key": key, "target_uid": uid, "primary_style": "standard",
            "primary_instruction": variants[0][1] if variants else "x", "instruction_variants": [{"style": s, "text": t} for s, t in variants]}


def outcome(r: dict[str, Any], data: list[dict[str, Any]] | None = None, **kw: Any) -> Outcome:
    return Outcome(Job(r, (1000, 500), None), json.dumps(data if data is not None else leaf()).encode(), **kw)


def test_paraphrase_is_the_other_style_when_it_differs() -> None:
    assert deskforge.paraphrase(row()) == "In the calculator press OK 0"
    assert deskforge.paraphrase(row(variants=[("standard", "Press OK")])) is None
    assert deskforge.paraphrase(row(variants=[("standard", "Press OK"), ("detailed_contextual", "press ok ")])) is None


def test_target_matching_by_uid_and_drop_counting(tmp_path: Path) -> None:
    converter = deskforge.SplitConverter(tmp_path / "pages")
    for r in (row(uid="ok", n=1), row(uid="hid", n=2), row(uid="txt", n=3), row(uid="nope", n=4), row(uid="part", n=5)):
        converter.consume(outcome(r))
    converter.consume(Outcome(Job(row(n=6), None, None), error="not_in_index"))
    conv = converter.conv
    assert dict(conv.dropped) == {"target_fully_occluded": 1, "target_not_candidate": 1, "target_not_in_leaf": 1, "not_in_index": 1}
    assert conv.rows_read == 6 and conv.rows == 2
    assert [(e.kind, e.target_id) for e in conv.examples] == [("plain", 1), ("paraphrase", 1), ("plain", 2), ("paraphrase", 2)]
    assert conv.examples[0].equiv_ids == [1] and conv.written == {conv.examples[0].page_id}
    assert dict(conv.target_roles) == {"button": 2} and conv.apps == {"calc": 2}
    assert conv.element_counts == [5]  # one page for the two rows of the same screen


def test_outputs_load_with_the_eval_readers(tmp_path: Path) -> None:
    converter = deskforge.SplitConverter(tmp_path / "pages" / "train")
    converter.consume(outcome(row()))
    deskforge.write_examples(converter.conv.examples, tmp_path / "examples.jsonl")
    pages = load_pages(tmp_path / "pages" / "train")
    examples = load_examples(tmp_path / "examples.jsonl")
    assert list(pages) == [examples[0].page_id] and pages[examples[0].page_id].elements[examples[0].target_id].text == "OK"


# --- sampling ----------------------------------------------------------------------------------------


def test_candidate_order_is_deterministic_caps_rows_per_screen_and_prefers_distinct_screens() -> None:
    keys = ["a"] * 5 + ["b"] * 2 + ["c"]
    order = deskforge.candidate_order(keys, "0:train", 3)
    assert order == deskforge.candidate_order(keys, "0:train", 3) != deskforge.candidate_order(keys, "1:train", 3)
    assert sorted(keys[i] for i in order) == ["a", "a", "a", "b", "b", "c"]
    assert sorted(keys[i] for i in order[:3]) == ["a", "b", "c"]  # one row of every screen first
    assert sorted(keys[i] for i in order[3:5]) == ["a", "b"] and keys[order[5]] == "a"


def test_select_rows_applies_the_app_share_and_the_row_budget() -> None:
    apps: list[str | None] = ["x"] * 30 + ["y"] * 8 + ["z"] * 2 + [None]
    sampler = Sampler(max_rows=20, app_cap=deskforge.app_row_cap(20, 0.3))
    selected, skipped = deskforge.select_rows(range(len(apps)), apps, sampler)
    assert sampler.per_app == {"x": 6, "y": 6, "z": 2} and len(selected) == 14
    assert dict(skipped) == {"app_cap": 26, "no_transition": 1}
    again, _ = deskforge.select_rows(range(len(apps)), apps, Sampler(max_rows=20, app_cap=6))
    assert again == selected
    capped, _ = deskforge.select_rows(range(len(apps)), apps, Sampler(max_rows=8, app_cap=6))
    assert len(capped) == 8


def test_select_rows_held_out_apps_and_missing_transitions() -> None:
    apps: list[str | None] = ["xarchiver", "calc", None, "pluma", "xarchiver"]
    selected, skipped = deskforge.select_rows(range(5), apps, Sampler(max_rows=10, allowed_apps=deskforge.HELD_OUT_APPS))
    assert selected == [0, 3, 4] and dict(skipped) == {"app_not_held_out": 1, "no_transition": 1}


def test_app_row_cap() -> None:
    assert deskforge.app_row_cap(30_000, 0.12) == 3600 and deskforge.app_row_cap(3, 0.12) == 1


def test_target_apps_follow_the_row_order() -> None:
    rows = pa.table({"transition_id": ["b", "z", "a"]})
    transitions = pa.table({"transition_id": ["a", "b"], "action_target_app": ["app-a", "app-b"]})
    assert deskforge.target_apps(rows, transitions) == ["app-b", None, "app-a"]


# --- range requests ---------------------------------------------------------------------------------------


def make_tar(members: dict[str, bytes]) -> tuple[bytes, dict[str, Member]]:
    """A small uncompressed tar and, per member, the Member the shard index would give for it (data offset, size)."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    tar_bytes = buffer.getvalue()
    with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tar:
        found = {i.name: Member("data/train/part-00000.tar", i.offset_data, i.size, hashlib.sha256(members[i.name]).hexdigest()) for i in tar}
    return tar_bytes, found


CDN = "https://cdn.test/"


def range_server(tar_bytes: bytes, log: list[str] | None = None, failures: list[int] | None = None, expires: float = 4e9) -> httpx.Client:
    """A client whose requests are answered like the Hub's: `resolve` redirects to a signed CDN URL, which answers
    Range requests with 206 (after the given failure statuses, which carry a zero Retry-After)."""
    pending = list(failures or [])

    def handler(request: httpx.Request) -> httpx.Response:
        if log is not None:
            log.append(request.method if request.method == "HEAD" else request.headers.get("range", ""))
        if request.method == "HEAD":
            return httpx.Response(302, headers={"location": f"{CDN}{request.url.path}?Expires={int(expires)}"})
        if pending:
            return httpx.Response(pending.pop(0), headers={"retry-after": "0"})
        start, end = (int(v) for v in request.headers["range"].removeprefix("bytes=").split("-"))
        return httpx.Response(206, content=tar_bytes[start:end + 1])

    return httpx.Client(transport=httpx.MockTransport(handler))


def no_sleep(_: float) -> None:
    return None


def test_member_offset_is_the_data_start_not_the_tar_header() -> None:
    payload = b'[{"type": "Button"}]'
    tar_bytes, members = make_tar({"a.leaf.json": b"x" * 700, "b.leaf.json": payload})
    member = members["b.leaf.json"]
    assert member.offset % 512 == 0 and member.offset == 512 + 1024 + 512  # header + 700 bytes padded to 1024, then b's header
    assert tar_bytes[member.offset:member.offset + member.size] == payload
    assert member.range_header == f"bytes={member.offset}-{member.offset + member.size - 1}"
    assert deskforge.Fetcher(range_server(tar_bytes)).fetch(member) == payload


def test_each_tar_is_resolved_once_and_members_are_read_from_the_cdn() -> None:
    tar_bytes, members = make_tar({"a.leaf.json": b"first", "b.leaf.json": b"second"})
    log: list[str] = []
    fetcher = deskforge.Fetcher(range_server(tar_bytes, log))
    assert [fetcher.fetch(m) for m in members.values()] == [b"first", b"second"]
    assert [entry for entry in log if entry == "HEAD"] == ["HEAD"] and len(log) == 3
    assert fetcher.urls["data/train/part-00000.tar"].startswith(CDN)


def test_expired_signed_urls_are_resolved_again() -> None:
    tar_bytes, members = make_tar({"a.leaf.json": b"first"})
    log: list[str] = []
    fetcher = deskforge.Fetcher(range_server(tar_bytes, log, expires=time.time() + 10))  # inside the renewal margin
    fetcher.fetch(members["a.leaf.json"])
    fetcher.fetch(members["a.leaf.json"])
    assert log.count("HEAD") == 2
    log.clear()
    forgetful = deskforge.Fetcher(range_server(tar_bytes, log, failures=[403]), sleep=no_sleep)  # rejected early: resolve and retry
    assert forgetful.fetch(members["a.leaf.json"]) == b"first" and log.count("HEAD") == 2


def test_fetch_checks_sha256_and_requires_partial_content() -> None:
    tar_bytes, members = make_tar({"a.leaf.json": b"payload"})
    member = members["a.leaf.json"]
    bad = Member(member.tar_path, member.offset, member.size, "0" * 64)
    with pytest.raises(deskforge.FetchError, match="mismatch"):
        deskforge.Fetcher(range_server(tar_bytes)).fetch(bad)
    ignores_range = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=tar_bytes)))
    with pytest.raises(deskforge.FetchError, match="expected 206"):
        deskforge.Fetcher(ignores_range).fetch(member)
    with pytest.raises(deskforge.FetchError, match="resolve HTTP 404"):
        deskforge.Fetcher(httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(404)))).fetch(member)


def test_fetch_retries_throttling_and_server_errors_with_backoff() -> None:
    tar_bytes, members = make_tar({"a.leaf.json": b"payload"})
    member, slept = members["a.leaf.json"], []
    assert deskforge.Fetcher(range_server(tar_bytes, failures=[429, 503]), sleep=slept.append).fetch(member) == b"payload"
    assert slept == [0.0, 0.0]  # Retry-After: 0 is honoured
    with pytest.raises(deskforge.FetchError, match="after 6 attempts"):
        deskforge.Fetcher(range_server(tar_bytes, failures=[500] * 6), sleep=no_sleep).fetch(member)
    with pytest.raises(deskforge.FetchError, match="HTTP 404"):
        deskforge.Fetcher(range_server(tar_bytes, failures=[404]), sleep=no_sleep).fetch(member)


def test_retry_delay_reads_retry_after_then_the_hub_rate_limit_reset() -> None:
    assert deskforge.retry_delay(httpx.Response(429, headers={"retry-after": "7"})) == 7.0
    assert deskforge.retry_delay(httpx.Response(429, headers={"ratelimit": '"resolvers";r=0;t=250'})) == 250.0
    assert deskforge.retry_delay(httpx.Response(429, headers={"retry-after": "9999"})) == deskforge.MAX_BACKOFF
    assert deskforge.retry_delay(httpx.Response(503)) is None


def test_url_expiry() -> None:
    assert deskforge.url_expiry("https://cdn.test/x?a=1&Expires=1795050476&b=2") == 1795050476.0
    assert deskforge.url_expiry("https://huggingface.co/x") == float("inf")


def test_fetch_retries_transport_errors() -> None:
    tar_bytes, members = make_tar({"a.leaf.json": b"payload"})
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "HEAD":
            return httpx.Response(200)
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectError("reset")
        return httpx.Response(206, content=b"payload")

    fetcher = deskforge.Fetcher(httpx.Client(transport=httpx.MockTransport(handler)), sleep=no_sleep)
    assert fetcher.fetch(members["a.leaf.json"]) == b"payload"


# --- cache and resume ---------------------------------------------------------------------------------------


def test_cache_resumes_and_rejects_corrupt_entries(tmp_path: Path) -> None:
    tar_bytes, members = make_tar({"k.leaf.json": b"payload"})
    member, log = members["k.leaf.json"], []
    fetcher, cache = deskforge.Fetcher(range_server(tar_bytes, log)), LeafCache(tmp_path / "leaf")
    job = Job({"before_key": "k"}, (10, 10), member)
    first = deskforge.fetch_leaf(fetcher, cache, job)
    assert (first.data, first.downloaded) == (b"payload", True) and log == ["HEAD", f"bytes={member.offset}-{member.offset + member.size - 1}"]
    second = deskforge.fetch_leaf(fetcher, cache, job)
    assert (second.data, second.downloaded) == (b"payload", False) and len(log) == 2  # served from disk
    cache.path("k").write_bytes(b"corrupt")
    third = deskforge.fetch_leaf(fetcher, cache, job)
    assert (third.data, third.downloaded) == (b"payload", True) and cache.load("k", member) == b"payload"
    assert not list((tmp_path / "leaf").glob("*.part"))


def test_fetch_leaf_reports_failures_as_outcomes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _, members = make_tar({"k.leaf.json": b"payload"})
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(403)))
    fetcher = deskforge.Fetcher(client)
    out = deskforge.fetch_leaf(fetcher, LeafCache(tmp_path), Job({"before_key": "k"}, (1, 1), members["k.leaf.json"]))
    assert (out.data, out.error) == (None, "fetch_failed") and "HTTP 403" in capsys.readouterr().out
    assert deskforge.fetch_leaf(fetcher, LeafCache(tmp_path), Job({"before_key": "k"}, None, None)).error == "not_in_index"


def test_ordered_parallel_keeps_order_and_stops_lazily() -> None:
    started: list[int] = []
    lock = threading.Lock()

    def slow(i: int) -> int:
        with lock:
            started.append(i)
        time.sleep(0.05 if i % 2 else 0.0)
        return i * 2

    assert list(deskforge.ordered_parallel(slow, range(10), 4)) == [i * 2 for i in range(10)]
    started.clear()
    stream = deskforge.ordered_parallel(slow, range(1000), 2)
    assert next(stream) == 0
    stream.close()
    assert len(started) <= 8


# --- end to end ----------------------------------------------------------------------------------------------


def write_index(raw: Path, name: str, table: pa.Table) -> None:
    path = raw / "index" / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def test_convert_split_end_to_end_and_rerun_uses_the_cache(tmp_path: Path) -> None:
    keys = [f"shard-0__scene-{i}-step00" for i in range(4)]
    leaves = {k: json.dumps(leaf()).encode() for k in keys}
    tar_bytes, members = make_tar({f"{k}.leaf.json": d for k, d in leaves.items()})
    raw, out = tmp_path / "raw", tmp_path / "out"
    rows = [row(key=k, n=i) for i, k in enumerate(keys)] + [row(key=keys[0], uid="hid", n=9)]
    write_index(raw, "instructions/val", pa.table({
        "transition_id": [r["transition_id"] for r in rows], "before_key": [r["before_key"] for r in rows],
        "target_uid": [r["target_uid"] for r in rows], "primary_style": [r["primary_style"] for r in rows],
        "primary_instruction": [r["primary_instruction"] for r in rows], "instruction_variants": [r["instruction_variants"] for r in rows]}))
    write_index(raw, "observations/val", pa.table({"observation_key": keys, "width": [1000] * 4, "height": [500] * 4}))
    write_index(raw, "transitions/val", pa.table({"transition_id": [r["transition_id"] for r in rows], "action_target_app": ["calc"] * 5}))
    write_index(raw, "shard_members", pa.table({
        "observation_key": keys, "kind": ["leaf"] * 4, "tar_path": [m.tar_path for m in members.values()],
        "byte_offset": [m.offset for m in members.values()], "byte_size": [m.size for m in members.values()],
        "sha256": [m.sha256 for m in members.values()]}))

    log: list[str] = []
    client = range_server(tar_bytes, log)
    conv = deskforge.convert_split(client, raw, out, "dev", Sampler(max_rows=10), max_per_screen=3, seed=0, workers=2)
    assert log.count("HEAD") == 1 and len(log) == 5 and conv.leaves_downloaded == 4 and conv.bytes_downloaded == sum(len(d) for d in leaves.values())
    assert conv.rows == 4 and conv.dropped == {"target_fully_occluded": 1} and len(conv.written) == 4
    assert len(load_examples(out / "examples" / "dev.jsonl")) == 8 and len(list(load_pages(out / "pages" / "dev"))) == 4

    again = deskforge.convert_split(client, raw, out, "dev", Sampler(max_rows=10), max_per_screen=3, seed=0, workers=2)
    assert len(log) == 5 and again.leaves_downloaded == 0 and again.examples == conv.examples
