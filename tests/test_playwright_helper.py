"""Playwright helper tests: a tiny XS model is trained on synthetic data (about 30 s) and driven against fixture pages."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from argparse import Namespace
from pathlib import Path

import pytest

from grounder import playwright_helper as ph
from grounder.data.pages import load_pages
from grounder.export import export_model_dir
from grounder.model import load_checkpoint
from grounder.records import Element
from grounder.runtime import Grounder
from grounder.tokenizer import train_tokenizer
from grounder.train import train
from synth import make_dataset, texts, write_dataset

FIXTURE = (Path(__file__).parent / "fixtures" / "helper" / "shop.html").as_uri()
CONFIG = Path(__file__).parent.parent / "configs" / "xs.yaml"
VIEWPORT = {"width": 1280, "height": 800}
CLICK_THRESHOLD = 0.5


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("helper-data")
    write_dataset(out, 400, seed=3)
    pages, examples = make_dataset(400, seed=3)
    train_tokenizer(texts(pages, examples), 8192, out / "tok.json")
    return out


@pytest.fixture(scope="module")
def model_dir(data_dir) -> Path:
    args = Namespace(
        config=str(CONFIG), pages=[str(data_dir / "pages")], train=str(data_dir / "train.jsonl"),
        dev=str(data_dir / "dev.jsonl"), tokenizer=str(data_dir / "tok.json"), out=str(data_dir / "xs"),
        epochs=12, batch_size=64, lr=1e-3, warmup=50, weight_decay=0.01, grad_clip=1.0, device="cpu",
        max_steps=None, max_batch_elements=12288, eval_batch_size=32, workers=0, seed=0, log_every=10_000,
    )
    checkpoint = train(args)
    model, _, ckpt = load_checkpoint(checkpoint)
    return export_model_dir(model, ckpt["tokenizer"], data_dir / "dist")


@pytest.fixture(scope="module")
def grounder(model_dir) -> Grounder:
    return Grounder(model_dir)


@pytest.fixture(scope="module")
def browser():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


@pytest.fixture
def page(browser):
    pg = browser.new_page(viewport=VIEWPORT)
    pg.goto(FIXTURE)
    yield pg
    pg.close()


def clicks(page) -> dict[str, int]:
    return page.evaluate("window.clicks")


def test_tune_threshold_picks_lowest_threshold_with_target_precision():
    scores = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5]
    correct = [True, True, True, False, True, False]
    assert ph.tune_threshold(scores, correct, 1.0) == 0.8
    assert ph.tune_threshold(scores, correct, 0.8) == 0.6
    assert ph.tune_threshold(scores, correct, 0.5) == 0.5


def test_tune_threshold_handles_ties_and_unreachable_targets():
    assert ph.tune_threshold([0.9, 0.9, 0.5], [True, False, True], 1.0) is None
    assert ph.tune_threshold([], [], 0.9) is None
    with pytest.raises(ValueError):
        ph.tune_threshold([0.5], [True, False])


@pytest.mark.parametrize("text,context,word", [
    ("Place Order", "", "place order"), ("Pay now", "", "pay"), ("share", "send it to a friend", "send"),
    ("Delete account", "", "delete"), ("cart", "Checkout summary", "checkout"),
])
def test_blocked_word_matches_whole_words_in_text_or_context(text, context, word):
    assert ph.blocked_word(Element(id=0, role="button", text=text, context=context)) == word


@pytest.mark.parametrize("text", ["payment methods", "border", "reorder soon", "resend", "cart"])
def test_blocked_word_ignores_substrings(text):
    assert ph.blocked_word(Element(id=0, role="link", text=text)) is None


def test_find_returns_the_exact_label_first_with_geometry(page, grounder):
    found = ph.find(page, grounder, "click cart", k=3)
    assert [c["text"] for c in found.candidates][0] == "cart"
    assert len(found.candidates) == 3 and 0.0 <= found.null_prob <= 1.0
    top = found.candidates[0]
    x, y, w, h = top["abs_box"]
    assert top["centre"] == [pytest.approx(x + w / 2, abs=0.1), pytest.approx(y + h / 2, abs=0.1)]
    assert set(top) == {"id", "score", "abs_box", "role", "text", "centre"}
    assert found.candidates[0]["score"] > found.candidates[1]["score"]


@pytest.mark.parametrize("label", ["home", "login", "settings"])
def test_click_clicks_the_named_element(page, grounder, label):
    result = ph.click(page, grounder, f"click {label}", CLICK_THRESHOLD)
    assert result.done, result.reason
    assert result.target["text"] == label
    assert clicks(page) == {label: 1}


def test_click_scrolls_an_offscreen_element_into_view(page, grounder):
    assert page.evaluate("scrollY") == 0
    result = ph.click(page, grounder, "click contact", CLICK_THRESHOLD)
    assert result.done, result.reason
    assert clicks(page) == {"contact": 1} and page.evaluate("scrollY") > 0


def test_click_refuses_below_threshold(page, grounder):
    result = ph.click(page, grounder, "click cart", 1.01)
    assert not result.done and "below the threshold" in result.reason
    assert result.candidates[0]["text"] == "cart" and clicks(page) == {}


@pytest.mark.parametrize("instruction,word", [("click delete", "delete"), ("click share", "send")])
def test_click_refuses_blocked_words_in_text_or_context(page, grounder, instruction, word):
    result = ph.click(page, grounder, instruction, 0.0)
    assert not result.done and word in result.reason and result.candidates
    assert clicks(page) == {}


def test_click_refuses_a_form_submit_button(page, grounder):
    result = ph.click(page, grounder, "click save", 0.0)
    assert result.target["text"] == "save"
    assert not result.done and "submits a form" in result.reason
    assert clicks(page) == {}


def test_click_refuses_input_submit(browser, grounder):
    pg = browser.new_page(viewport=VIEWPORT)
    pg.set_content('<form onsubmit="return false"><input type="submit" value="share"></form>')
    result = ph.click(pg, grounder, "click share", 0.0)
    assert not result.done and "submits a form" in result.reason
    pg.close()


def test_click_allows_a_plain_button_inside_a_form(browser, grounder):
    pg = browser.new_page(viewport=VIEWPORT)
    pg.set_content('<form><button type="button" onclick="window.n=1">share</button></form>')
    result = ph.click(pg, grounder, "click share", 0.0)
    assert result.done and pg.evaluate("window.n") == 1
    pg.close()


def test_click_refuses_when_the_page_changed_since_extraction(page):
    probe = page.evaluate(ph.PROBE_JS, [60, 40, "textarea"])
    assert not probe["ok"] and "page changed" in probe["reason"]


def test_type_text_types_into_the_textbox_and_never_into_other_elements(page, grounder):
    result = ph.type_text(page, grounder, "find in search box", "hello", CLICK_THRESHOLD)
    assert result.done, result.reason
    assert page.input_value("input[type=search]") == "hello"
    refused = ph.type_text(page, grounder, "click cart", "x", 0.0)
    assert not refused.done and "not one of textbox" in refused.reason


def test_type_text_refuses_password_fields(page, grounder):
    result = ph.type_text(page, grounder, "click help", "secret", 0.0)
    assert result.target["text"] == "help" and not result.done and "password" in result.reason
    assert page.input_value("input[type=password]") == ""


def test_async_api_matches_sync(grounder):
    from playwright.async_api import async_playwright

    async def run() -> tuple[str, bool, int]:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            pg = await browser.new_page(viewport=VIEWPORT)
            await pg.goto(FIXTURE)
            found = await ph.afind(pg, grounder, "click cart")
            result = await ph.aclick(pg, grounder, "click cart", CLICK_THRESHOLD)
            typed = await ph.atype_text(pg, grounder, "find in search box", "ab", CLICK_THRESHOLD)
            count = (await pg.evaluate("window.clicks")).get("cart", 0)
            await browser.close()
            return found.candidates[0]["text"], result.done and typed.done, count

    with ThreadPoolExecutor(1) as pool:  # the module's sync Playwright already owns this thread's event loop
        assert pool.submit(asyncio.run, run()).result() == ("cart", True, 1)


def test_tune_cli_writes_thresholds_next_to_the_model(model_dir, data_dir, capsys):
    ph._main(["tune", "--model-dir", str(model_dir), "--pages", str(data_dir / "pages"),
              "--examples", str(data_dir / "dev.jsonl"), "--precision", "0.9"])
    report = json.loads((model_dir / "thresholds.json").read_text())
    assert report["precision"] >= 0.9 and report["answered"] > 0
    assert ph.load_threshold(model_dir) == report["click"]
    assert load_pages(data_dir / "pages")
