"""Template teacher, filters, null examples and the LLM teacher (against a mocked endpoint)."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
from pathlib import Path

import httpx
import pytest

from grounder.data.build import add_null_examples, count_table
from grounder.data.filters import filter_all, filter_examples
from grounder.data.teacher import (
    BATCH_SIZE, MAX_TOKENS, RESPONSE_FORMAT, BadResponse, Stats, TeacherConfig, build_messages, generate, items_to_examples,
    json_object, neighbour_texts, parse_items, pick_pages,
)
from grounder.data.template_teacher import context_phrase, sample_elements, template_examples
from grounder.ranking import example_elements, positive_ids, removed_ids
from grounder.data.pages import write_page
from grounder.records import Element, Example, Page, equiv_ids


def el(i: int, text: str, role: str = "link", context: str = "", href: str = "", box=(0.1, 0.1, 0.2, 0.05), **kw) -> Element:
    x, y, w, h = box
    return Element(id=i, role=role, text=text, context=context, href=href or f"/{text.lower().replace(' ', '-')}",
                   box=[x, y, w, h], abs_box=[x * 1280, y * 800, w * 1280, h * 800], **kw)


def page_of(elements: list[Element], title: str = "Acme Shop", url: str = "https://acme.com/") -> Page:
    return Page("pg", url, "acme.com", title, [1280, 800], [1280, 4000], elements)


@pytest.fixture
def shop() -> Page:
    return page_of([
        el(0, "Pricing", box=(0.1, 0.05, 0.1, 0.04)),
        el(1, "Sign in", "button", box=(0.8, 0.05, 0.1, 0.04)),
        el(2, "Search", "searchbox", box=(0.4, 0.05, 0.3, 0.04)),
        el(3, "Add to cart", "button", "Red lamp · $20", href="/cart/1", box=(0.1, 0.4, 0.1, 0.04)),
        el(4, "Add to cart", "button", "Blue chair · $90", href="/cart/2", box=(0.1, 0.6, 0.1, 0.04)),
        el(5, "Contact", box=(0.1, 0.9, 0.1, 0.04), context="© 2026 Acme · Privacy"),
        el(6, "Home", box=(0.3, 0.05, 0.1, 0.04), href="/"),
        el(7, "Home", box=(0.3, 0.97, 0.1, 0.04), href="/"),
        el(8, "Close", "button", box=(0.9, 0.02, 0.02, 0.03)),
        el(9, "", "button"),
        el(10, "one two three four five six seven eight nine", "link"),
    ])


# --- template teacher --------------------------------------------------------------------------


def test_template_examples_are_valid(shop):
    examples = template_examples(shop, seed=0, n=50)
    assert examples
    by_id = {e.id: e for e in shop.elements}
    for ex in examples:
        assert ex.mode == "do" and ex.kind in ("plain", "context") and ex.page_id == shop.page_id
        assert ex.target_id in by_id and ex.history == []
        assert ex.equiv_ids == equiv_ids(shop.elements, ex.target_id)
    assert filter_examples(shop, examples).kept == examples


def test_template_skips_empty_and_long_labels(shop):
    targets = {ex.target_id for ex in template_examples(shop, seed=0, n=50)}
    assert 9 not in targets and 10 not in targets
    assert {0, 1, 2, 8} <= targets


def test_template_duplicated_labels(shop):
    examples = template_examples(shop, seed=0, n=50)
    cart = [ex for ex in examples if ex.target_id in (3, 4)]
    assert {ex.kind for ex in cart} == {"context"}  # different hrefs: plain would be ambiguous
    assert any("Red lamp" in ex.query for ex in cart if ex.target_id == 3)
    assert any("Blue chair" in ex.query for ex in cart if ex.target_id == 4)
    home = [ex for ex in examples if ex.target_id in (6, 7)]
    assert home and all(ex.kind == "plain" and ex.equiv_ids == [6, 7] for ex in home)  # same href: equivalent


def test_template_wording_per_role(shop):
    examples = {ex.target_id: [e.query for e in template_examples(shop, seed=s, n=50) if e.target_id == ex.target_id and e.kind == "plain"]
                for s in range(30) for ex in template_examples(shop, seed=s, n=50)}
    assert all(q.split()[0] in ("click", "press", "hit", "tap") for q in examples[1])
    assert all(q.split()[0] in ("open", "go", "follow", "click") for q in examples[0])
    assert any("Search" in q for q in examples[2])


def test_template_label_noun_is_not_repeated():
    page = page_of([el(0, "Login button", "button"), el(1, "Help link", "link"), el(2, "Other", "link")])
    for seed in range(30):
        for ex in template_examples(page, seed, n=10):
            assert "button button" not in ex.query.lower() and "link link" not in ex.query.lower()


def test_template_is_deterministic(shop):
    assert template_examples(shop, 3) == template_examples(shop, 3)
    assert template_examples(shop, 3) != template_examples(shop, 4)


def test_sampler_includes_duplicates_and_icons(shop):
    for seed in range(10):
        picked = {e.id for e in sample_elements(shop, random.Random(seed), n=5)}
        assert picked & {3, 4, 6, 7}  # a duplicated label
        assert 8 in picked  # the only icon-sized element
        assert len(picked) == 5


def test_sampler_downweights_footer_and_unlabelled():
    page = page_of([el(0, "Alpha"), el(1, "Beta", context="© 2026 Acme · Privacy"), el(2, "Gamma", box=(0.1, 4.75, 0.2, 0.05), in_viewport=False),
                    el(3, "", "link"), el(4, "Delta"), el(5, "Epsilon")])
    freq = dict.fromkeys(range(6), 0)
    for seed in range(600):
        for e in sample_elements(page, random.Random(seed), n=2):
            freq[e.id] += 1
    assert max(freq[1], freq[2], freq[3]) * 2 < min(freq[0], freq[4], freq[5])


def test_sampler_balances_roles():
    elements = [el(i, f"Link {i}") for i in range(45)]
    elements += [el(45 + i, f"Go {i}", "button") for i in range(3)] + [el(48 + i, f"Field {i}", "textbox") for i in range(2)]
    page = page_of(elements)
    non_link = 0
    for seed in range(50):
        picked = sample_elements(page, random.Random(seed), n=20)
        assert len(picked) == 20
        non_link += sum(e.role != "link" for e in picked)
    assert non_link == 50 * 5  # every button and textbox is sampled; a uniform sample would average 2 of 5


def test_sampler_caps_a_role_while_others_remain():
    elements = [el(i, f"Link {i}") for i in range(45)] + [el(45 + i, f"Go {i}", "button") for i in range(15)]
    page = page_of(elements)
    for seed in range(20):
        picked = sample_elements(page, random.Random(seed), n=20)
        assert sum(e.role == "link" for e in picked) <= 10


def test_sampler_without_other_roles_is_not_capped():
    page = page_of([el(i, f"Link {i}") for i in range(30)])
    assert len(sample_elements(page, random.Random(0), n=20)) == 20


def test_context_phrase_is_short_first_segment():
    assert context_phrase("Blue cotton shirt and more words here today · $24") == "Blue cotton shirt and more words"
    assert context_phrase("Single") == "Single"


# --- null examples -----------------------------------------------------------------------------


def test_null_examples_follow_ranking_convention(shop):
    plain = [Example("pg", "plain", "do", f"open {t}", [], i, equiv_ids(shop.elements, i)) for i, t in [(0, "Pricing"), (1, "Sign in"), (2, "Search"), (8, "Close")]]
    dup = Example("pg", "paraphrase", "do", "put it in my basket", [], 3, [3])  # "Add to cart" still on the page at id 4
    out = add_null_examples(plain + [dup], {"pg": shop}, fraction=0.5, seed=0)
    nulls = [ex for ex in out if ex.kind == "null"]
    assert out[:5] == plain + [dup] and len(nulls) == 2
    for ex in nulls:
        assert ex.target_id is not None and ex.target_id in ex.equiv_ids
        assert ex.target_id not in {e.id for e in example_elements(shop, ex)}
        assert removed_ids(ex) == set(ex.equiv_ids) and not positive_ids(ex)
        assert ex.target_id != 3
    assert add_null_examples(plain, {"pg": shop}, 0.5, 0) == add_null_examples(plain, {"pg": shop}, 0.5, 0)


def test_count_table_lists_kinds_per_split():
    from collections import Counter

    table = count_table({"train": Counter(plain=3, null=1), "dev": Counter(plain=1)})
    assert "plain" in table and "null" in table and table.splitlines()[1].split()[-1] == "4"


# --- filters -----------------------------------------------------------------------------------


def ex(kind: str, query: str, target: int, equiv: list[int] | None = None, mode: str = "do") -> Example:
    return Example("pg", kind, mode, query, [], target, equiv if equiv is not None else [target])


@pytest.mark.parametrize("example,reason", [
    (ex("plain", "pricing", 0), "length"),
    (ex("plain", " ".join(["open pricing"] * 11), 0), "length"),
    (ex("plain", "click element 3", 0), "id_or_coordinate"),
    (ex("plain", "click button #4", 1), "id_or_coordinate"),
    (ex("plain", "click at 120, 340", 0), "id_or_coordinate"),
    (ex("plain", "click the 640x480 thing", 0), "id_or_coordinate"),
    (ex("plain", "open 7", 0), "id_or_coordinate"),
    (ex("paraphrase", "view the pricing page", 0), "paraphrase_overlap"),
    (ex("paraphrase", "tap sign in please", 1), "paraphrase_overlap"),
    (ex("context", "click Contact for the footer", 5), "context_not_duplicated"),
    (ex("context", "click Add to cart for something nice", 3), "context_unused"),
    (ex("plain", "click Add to cart", 3), "ambiguous_plain"),
    (ex("plain", "open pricing", 99), "unknown_target"),
])
def test_filter_rejects(shop, example, reason):
    result = filter_examples(shop, [example])
    assert result.kept == [] and result.rejected == {reason: 1}


def test_filter_accepts_good_examples(shop):
    good = [
        ex("plain", "open Pricing", 0),
        ex("paraphrase", "see what it costs", 0),
        ex("plain", "go home", 6, [6, 7]),
        ex("context", "click Add to cart for the red lamp", 3),
        ex("goal", "buy a lamp for my living room", 3, mode="goal"),
        ex("plain", "open Contact", 5),
    ]
    result = filter_examples(shop, good)
    assert result.kept == good and not result.rejected


def test_filter_allows_numbers_that_are_in_the_element():
    page = page_of([el(0, "Go to slide 5", "button"), el(1, "Other")])
    assert filter_examples(page, [ex("plain", "press Go to slide 5", 0)]).kept
    assert not filter_examples(page, [ex("plain", "press Go to slide 6", 0)]).kept


def test_filter_dedupes_queries_per_page(shop):
    same = [ex("plain", "open Pricing", 0), ex("plain", "Open  pricing", 0)]
    assert filter_examples(shop, same).rejected == {"duplicate_query": 1}
    conflict = [ex("plain", "open Pricing", 0), ex("plain", "open pricing", 2)]
    result = filter_examples(shop, conflict)
    assert result.kept == [] and result.rejected == {"conflicting_query": 2}


def test_filter_all_groups_by_page(shop):
    other = Example("other", "plain", "do", "open Pricing", [], 0, [0])
    result = filter_all({"pg": shop}, [ex("plain", "open Pricing", 0), other])
    assert len(result.kept) == 1 and result.rejected == {"unknown_page": 1}


# --- LLM teacher -------------------------------------------------------------------------------

GOOD_REPLY = {"items": [
    {"i": 0, "plain": "open Pricing", "paraphrase": "see what it costs", "context": None, "goal": "compare plans before buying"},
    {"i": 1, "plain": "click Sign in", "paraphrase": "log into my account", "context": None, "goal": "check my past orders"},
    {"i": 2, "plain": "type in the Search box", "paraphrase": "look for a product", "context": None, "goal": "find a lamp"},
    {"i": 3, "plain": "click Add to cart", "paraphrase": "put the lamp in my basket", "context": "add to cart for the red lamp", "goal": "buy the red lamp"},
    {"i": 99, "plain": "ignored, bad index", "paraphrase": "x", "context": None, "goal": "y"},
]}


def chat_response(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": content}}]})


class FakeEndpoint:
    """Replays `replies` (callables or responses) in order and records every request body."""

    def __init__(self, replies: list) -> None:
        self.replies = list(replies)
        self.bodies: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        reply = self.replies.pop(0)
        return reply(request) if callable(reply) else reply

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url="http://llm.test/v1", transport=httpx.MockTransport(self))


@pytest.fixture
def small_page() -> Page:
    return page_of([el(0, "Pricing"), el(1, "Sign in", "button"), el(2, "Search", "searchbox"),
                    el(3, "Add to cart", "button", "Red lamp · $20", href="/c/1"), el(4, "Add to cart", "button", "Blue chair · $90", href="/c/2")])


def run_generate(pages, endpoint: FakeEndpoint, cache: Path, **kw):
    cfg = TeacherConfig(model="m", cache_dir=cache, backoff_s=0, per_page=len(pages[0].elements), **kw)

    async def go():
        async with endpoint.client() as http:
            return await generate(pages, http, cfg)

    return asyncio.run(go())


def test_prompt_lists_title_host_neighbours_and_duplicates(small_page):
    messages = build_messages(small_page, small_page.elements)
    user = messages[1]["content"]
    assert messages[0]["role"] == "system" and "Example output" in messages[0]["content"]
    assert 'Page: "Acme Shop" (acme.com)' in user
    assert '[3] role=button text="Add to cart" context="Red lamp · $20"' in user
    assert "same_text_elsewhere=1" in user and "same_text_elsewhere=0" in user
    assert neighbour_texts(small_page.elements[0], small_page.elements) == ["Sign in", "Search", "Add to cart"]


def test_parse_items_variants():
    assert parse_items('{"items": [{"i": 0}]}') == [{"i": 0}]
    assert parse_items('```json\n{"items": [{"i": 1}]}\n```') == [{"i": 1}]
    assert parse_items('[{"i": 2}, 5]') == [{"i": 2}]
    for bad in ("not json", '{"items": 3}', "{}"):
        with pytest.raises(BadResponse):
            parse_items(bad)


def test_items_to_examples_modes_and_equivalents(small_page):
    examples = items_to_examples(small_page, small_page.elements, GOOD_REPLY["items"])
    assert {ex.kind for ex in examples} == {"plain", "paraphrase", "context", "goal"}
    goals = [ex for ex in examples if ex.kind == "goal"]
    assert all(ex.mode == "goal" and ex.history == [] for ex in goals)
    assert all(ex.mode == "do" for ex in examples if ex.kind != "goal")
    assert len([ex for ex in examples if ex.kind == "context"]) == 1
    assert all(ex.target_id in range(4) for ex in examples)  # item 99 is ignored


def test_generate_retries_malformed_reply_and_filters(small_page, tmp_path):
    endpoint = FakeEndpoint([chat_response("Sure! here you go"), chat_response(json.dumps(GOOD_REPLY))])
    raw, failed = run_generate([small_page], endpoint, tmp_path)
    assert failed == 0 and len(endpoint.bodies) == 2
    assert endpoint.bodies[0]["response_format"] == RESPONSE_FORMAT and endpoint.bodies[0]["seed"] == 0
    result = filter_all({small_page.page_id: small_page}, raw)
    # "click Add to cart" targets one of two different buttons -> ambiguous; the rest is valid
    assert result.rejected == {"ambiguous_plain": 1}
    kinds = {ex.kind for ex in result.kept}
    assert kinds == {"plain", "paraphrase", "context", "goal"}


def test_generate_retries_server_errors_then_gives_up(small_page, tmp_path):
    endpoint = FakeEndpoint([httpx.Response(503)] * 3)
    raw, failed = run_generate([small_page], endpoint, tmp_path, retries=2)
    assert raw == [] and failed == 1 and len(endpoint.bodies) == 3


def test_generate_does_not_retry_auth_errors(small_page, tmp_path):
    endpoint = FakeEndpoint([httpx.Response(401, text="bad key")])
    raw, failed = run_generate([small_page], endpoint, tmp_path)
    assert raw == [] and failed == 1 and len(endpoint.bodies) == 1


def test_generate_falls_back_when_schema_unsupported(small_page, tmp_path):
    endpoint = FakeEndpoint([httpx.Response(400, text="response_format unsupported"), chat_response(json.dumps(GOOD_REPLY))])
    raw, failed = run_generate([small_page], endpoint, tmp_path)
    assert failed == 0 and raw
    assert "response_format" in endpoint.bodies[0] and "response_format" not in endpoint.bodies[1]


def test_generate_reads_cache_on_rerun(small_page, tmp_path):
    first = FakeEndpoint([chat_response(json.dumps(GOOD_REPLY))])
    raw, _ = run_generate([small_page], first, tmp_path)
    second = FakeEndpoint([])
    again, failed = run_generate([small_page], second, tmp_path)
    assert second.bodies == [] and failed == 0 and again == raw


def test_malformed_replies_are_not_cached(small_page, tmp_path):
    run_generate([small_page], FakeEndpoint([chat_response("nope")] * 5), tmp_path, retries=1)
    assert list(tmp_path.glob("*.json")) == []


def test_generate_batches_ten_elements_per_call(tmp_path):
    big = page_of([el(i, f"Item {i}") for i in range(25)])
    endpoint = FakeEndpoint([chat_response('{"items": []}')] * 3)
    run_generate([big], endpoint, tmp_path)
    sizes = sorted(body["messages"][1]["content"].count("\n[") for body in endpoint.bodies)
    assert sizes == [5, BATCH_SIZE, BATCH_SIZE]


def test_extra_body_and_max_tokens_reach_the_request(small_page, tmp_path):
    endpoint = FakeEndpoint([chat_response(json.dumps(GOOD_REPLY))])
    extra = {"chat_template_kwargs": {"enable_thinking": False}}
    run_generate([small_page], endpoint, tmp_path, extra_body=extra, max_tokens=700)
    body = endpoint.bodies[0]
    assert body["chat_template_kwargs"] == {"enable_thinking": False} and body["max_tokens"] == 700
    assert TeacherConfig(model="m", cache_dir=tmp_path).max_tokens == MAX_TOKENS


def test_extra_body_changes_the_cache_key(small_page, tmp_path):
    run_generate([small_page], FakeEndpoint([chat_response(json.dumps(GOOD_REPLY))]), tmp_path)
    other = FakeEndpoint([chat_response(json.dumps(GOOD_REPLY))])
    run_generate([small_page], other, tmp_path, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    assert len(other.bodies) == 1


def test_json_object_argument():
    assert json_object('{"a": 1}') == {"a": 1}
    with pytest.raises(argparse.ArgumentTypeError):
        json_object("[1]")


def test_stats_line_reports_rates_cache_hits_and_eta():
    stats = Stats(total=100, calls=20, cache_hits=10, failed=1, prompt_tokens=6000, completion_tokens=3000, started=0.0)
    line = stats.line(now=60.0)
    assert "20/100 calls" in line and "100 prompt tok/s" in line and "50 completion tok/s" in line
    assert "cache hits 10" in line and "failed 1" in line and "ETA 8.0 min" in line  # 80 left at 10 live calls/min
    assert "ETA n/a" in Stats(total=5, calls=2, cache_hits=2).line()


def test_generate_logs_progress_and_counts_tokens(small_page, tmp_path, caplog):
    reply = httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(GOOD_REPLY)}}],
                                      "usage": {"prompt_tokens": 400, "completion_tokens": 150}})
    big = page_of([el(i, f"Item {i}") for i in range(25)])
    with caplog.at_level(logging.INFO, logger="grounder.teacher"):
        run_generate([big], FakeEndpoint([reply] * 3), tmp_path, log_every=2)
    lines = [r.getMessage() for r in caplog.records if "calls," in r.getMessage()]
    assert len(lines) == 2 and lines[0].startswith("2/3 calls") and lines[1].startswith("3/3 calls")
    again = FakeEndpoint([])
    with caplog.at_level(logging.INFO, logger="grounder.teacher"):
        run_generate([big], again, tmp_path, log_every=3)
    assert "cache hits 3" in caplog.records[-1].getMessage()


def test_pick_pages_samples_a_seeded_subset(tmp_path):
    for i in range(10):
        write_page(Page(f"p{i}", "https://a.test/", "a.test", "t", [1280, 800], [1280, 800], []), tmp_path / "train")
    everything = pick_pages(tmp_path, None, 0, 0)
    assert len(everything) == 10
    picked = [p.page_id for p in pick_pages(tmp_path / "train", 4, seed=1, min_english=0)]
    assert len(picked) == 4 and picked == [p.page_id for p in pick_pages(tmp_path / "train", 4, seed=1, min_english=0)]
    assert picked != [p.page_id for p in pick_pages(tmp_path / "train", 4, seed=2, min_english=0)]


def test_fake_teacher_replies_survive_the_filters(shop):
    from tools.fake_teacher import completion

    batch = [e for e in shop.elements if e.text and len(e.text.split()) < 9][:10]
    request = {"messages": build_messages(shop, batch)}
    items = parse_items(completion(request)["choices"][0]["message"]["content"])
    assert [i["i"] for i in items] == list(range(len(batch)))
    raw = items_to_examples(shop, batch, items)
    result = filter_all({shop.page_id: shop}, raw)
    assert {ex.kind for ex in result.kept} == {"plain", "paraphrase", "context", "goal"}


def test_pilot_report_samples_each_kind_with_its_target(shop):
    from tools.pilot_report import report

    examples = [ex("plain", "open Pricing", 0), ex("plain", "click Sign in", 1), ex("paraphrase", "see what it costs", 0),
                ex("goal", "compare plans before buying", 0, mode="goal")]
    text = report({shop.page_id: shop}, examples, per_kind=1, seed=0)
    assert text.count("===") == 6 and "=== plain: 1 of 2 kept ===" in text
    assert "'see what it costs'" in text and "[link] 'Pricing'" in text


def test_lone_surrogates_in_crawled_text_are_dropped_on_load():
    from grounder.records import validate_element

    raw = el(0, "Like \ud83d").to_dict()
    assert validate_element(raw).text == "Like"
    page = Page.from_dict({**page_of([]).to_dict(), "title": "Shop \ud83d", "elements": [raw]})
    json.dumps(build_messages(page, page.elements)).encode("utf-8")
    assert page.title == "Shop "


def test_results_table_lists_models_and_kinds():
    from tools.results_table import split_table

    def metrics(paraphrase: float, overall: float) -> dict:
        return {"by_kind": {"plain": {"top1": 0.9}, "paraphrase": {"top1": paraphrase}}, "overall": {"top1": overall}}

    table = split_table("dev", {"xs": (1356929, metrics(0.5, 0.7)), "bm25": (None, metrics(0.1, 0.5))}).splitlines()
    assert table[0] == "| dev top-1 | params | paraphrase | plain | overall |"
    assert table[2] == "| xs | 1,356,929 | 0.500 | 0.900 | 0.700 |"
    assert table[3] == "| bm25 | - | 0.100 | 0.900 | 0.500 |"
