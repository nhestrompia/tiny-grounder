"""Extractor tests: each fixture is loaded in headless Chromium and checked against exact expectations."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from grounder.extract.web import DOM_JS, extract_page_sync, page_from_raw
from grounder.records import ROLES, Element, Page, validate_element

FIXTURES = Path(__file__).parent / "fixtures"
VIEWPORT = {"width": 1280, "height": 800}


@pytest.fixture(scope="module")
def browser_page():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT)
        yield page
        browser.close()


@pytest.fixture
def extract(browser_page):
    def run(name: str) -> Page:
        browser_page.goto((FIXTURES / f"{name}.html").as_uri())
        return extract_page_sync(browser_page)

    return run


def by_text(page: Page, text: str) -> list[Element]:
    return [e for e in page.elements if e.text == text]


def one(page: Page, text: str) -> Element:
    found = by_text(page, text)
    assert len(found) == 1, f"{text!r}: {[e.text for e in page.elements]}"
    return found[0]


def texts(page: Page) -> list[str]:
    return [e.text for e in page.elements]


ALL_FIXTURES = sorted(p.stem for p in FIXTURES.glob("*.html"))


def test_fixture_count():
    assert len(ALL_FIXTURES) >= 10


@pytest.mark.parametrize("name", ALL_FIXTURES)
def test_invariants(extract, name):
    page = extract(name)
    assert page.viewport == [1280, 800]
    assert [e.id for e in page.elements] == list(range(len(page.elements)))
    vw, vh = page.viewport
    for e in page.elements:
        validate_element(e.to_dict())
        assert e.role in ROLES
        assert e.source == "dom" and e.conf == 1.0
        assert len(e.text) <= 80 and len(e.context) <= 120 and len(e.href) <= 200
        x, y, w, h = e.abs_box
        assert w >= 1 and h >= 1
        assert e.box == pytest.approx([x / vw, y / vh, w / vw, h / vh], abs=1e-3)
        if e.in_viewport:
            assert y < vh and y + h > 0 and x < vw and x + w > 0


def test_icon_only_and_alt_text(extract):
    page = extract("icon_buttons")
    assert texts(page) == [
        "Open menu", "Search", "Settings", "Acme home", "docs", "Close dialog", "Go", "Reset all",
    ]
    assert one(page, "Open menu").role == "button"
    assert one(page, "Settings").role == "button"  # svg <title>
    home = one(page, "Acme home")
    assert home.role == "link" and home.href == "/home"
    docs = one(page, "docs")  # visible text beats the title tooltip
    assert docs.href == "https://other.example.org/x?y=1"
    assert one(page, "Close dialog").role == "button"
    assert one(page, "Go").tag == "input" and one(page, "Reset all").role == "button"
    assert all("Toolbar" in e.context for e in page.elements)


def test_duplicate_labels_differ_by_context(extract):
    page = extract("product_cards")
    adds = by_text(page, "Add to cart")
    assert len(adds) == 3
    assert [w in e.context for e, w in zip(adds, ["Aurora X1", "Borealis Pro", "Cirrus Mini"])] == [True] * 3
    assert "$99" in adds[0].context and "Add to cart" not in adds[0].context
    assert len({e.context for e in adds}) == 3
    details = by_text(page, "Details")
    assert "Row one" in details[0].context and "Row two" in details[1].context
    assert "Headphones" in details[0].context  # heading of the enclosing main, not a card heading
    assert "Invoice 7" in one(page, "Download").context  # table row


def test_form_labels_and_legends(extract):
    page = extract("form")
    assert texts(page) == [
        "Email address", "Full name", "Referral code", "Password", "Search help",
        "Standard", "Express", "Subscribe to newsletter", "Age", "Comments", "Notes editor", "Create account",
    ]
    roles = {e.text: e.role for e in page.elements}
    assert roles["Email address"] == "textbox" and roles["Full name"] == "textbox"
    assert roles["Search help"] == "searchbox"
    assert roles["Standard"] == "radio" and roles["Subscribe to newsletter"] == "checkbox"
    assert roles["Age"] == "slider" and roles["Comments"] == "textbox" and roles["Notes editor"] == "textbox"
    assert roles["Create account"] == "button"
    assert "Create account" in one(page, "Email address").context  # form heading
    assert "Email address" not in one(page, "Email address").context  # own text removed
    assert "Shipping method" in one(page, "Standard").context
    assert "Shipping method" in one(page, "Express").context
    assert all("hunter2" not in e.text for e in page.elements)  # password value never used
    assert "csrf" not in " ".join(texts(page)) and "not editable" not in texts(page)


def test_native_select(extract):
    page = extract("native_select")
    dest = one(page, "Destination")
    assert dest.role == "combobox" and dest.tag == "select"
    country = one(page, "Country")  # wrapping label, options excluded
    assert country.role == "combobox"
    assert one(page, "Confirm").role == "button"
    assert "Harbor Research" not in " ".join(texts(page))
    assert len(page.elements) == 3


def test_shadow_roots(extract):
    page = extract("shadow_root")
    assert texts(page) == [
        "Light button", "Shadow declarative", "Shadow link", "Shadow input", "Nested shadow", "After shadow",
    ]
    assert one(page, "Shadow link").href == "/shadow-link"
    assert "Card" in one(page, "Shadow declarative").context  # heading inside the shadow root
    assert one(page, "Shadow input").role == "textbox"
    assert "Closed shadow" not in texts(page)
    nested = one(page, "Nested shadow")
    assert nested.abs_box[2] > 0 and nested.in_viewport


def test_same_origin_iframe_offsets(extract, browser_page):
    page = extract("iframe_same_origin")
    assert texts(page) == ["Host button", "Frame button", "Frame link", "Frame far below"]
    first_frame = browser_page.locator("iframe").first.bounding_box()
    btn = browser_page.frame_locator("iframe").first.get_by_role("button", name="Frame button")
    expected = btn.bounding_box()  # page coordinates incl. frame offset
    got = one(page, "Frame button")
    assert got.abs_box[0] == pytest.approx(expected["x"], abs=1.5)
    assert got.abs_box[1] == pytest.approx(expected["y"], abs=1.5)
    assert got.abs_box[0] > first_frame["x"] + 5
    assert one(page, "Frame link").href == "/in-frame"
    far = one(page, "Frame far below")
    assert not far.in_viewport  # scrolled out of the frame's own viewport
    assert one(page, "Frame button").in_viewport
    assert "Sandboxed cross-origin" not in texts(page) and "Hidden frame button" not in texts(page)


def test_cross_origin_frame_is_skipped(extract):
    page = extract("isolated_frame")
    assert texts(page) == ["Host control"]


def test_hidden_elements_are_dropped(extract):
    page = extract("hidden")
    assert texts(page) == ["Visible button", "Free below", "Overlay passes through"]
    assert one(page, "Free below").in_viewport


def test_nested_interactive(extract):
    page = extract("nested")
    assert texts(page) == [
        "Button in same link", "Outer link Span button", "Clickable card Inner linkInner button",
        "Inner link", "Same-link widget",
    ]
    assert one(page, "Button in same link").role == "link"
    assert one(page, "Button in same link").href == "/same"
    assert "Span button" in one(page, "Outer link Span button").text
    assert one(page, "Inner link").href == "/inner-link"  # different target kept
    assert one(page, "Same-link widget").href == "/wrap"  # outer link wins over the inner widget
    assert not any(e.text == "Inner button" for e in page.elements)


def test_pointer_and_aria_roles(extract):
    page = extract("pointer")
    assert texts(page) == [
        "Pointer div", "Pointer span", "Nested pointer leaf", "Focusable div", "Tab one", "Check item",
        "Spin", "Tree node", "Dialog box", "Dark mode", "Click image", "Onclick span", "More info",
    ]
    roles = {e.text: e.role for e in page.elements}
    assert roles["Pointer div"] == "other" and roles["Focusable div"] == "other"
    assert roles["Tab one"] == "tab" and roles["Check item"] == "menuitem"
    assert roles["Spin"] == "textbox" and roles["Tree node"] == "option"
    assert roles["Dialog box"] == "other" and roles["Dark mode"] == "switch"
    assert roles["Click image"] == "image" and roles["More info"] == "button"
    assert "Negative tabindex" not in texts(page)
    assert not any("Pointer container" in t for t in texts(page))  # container with many children
    assert len([e for e in page.elements if "Nested pointer" in e.text]) == 1


def test_long_page_below_the_fold(extract):
    page = extract("long_page")
    assert page.doc_size[0] == 1280 and page.doc_size[1] > 6000
    assert texts(page) == ["Top button", "Deep link", "Bottom button"]
    top, deep, bottom = page.elements
    assert top.in_viewport and not deep.in_viewport and not bottom.in_viewport
    assert 3.5 < deep.box[1] < 4.5 and bottom.box[1] > 7
    assert deep.abs_box[1] == pytest.approx(deep.box[1] * 800, abs=0.5)
    assert deep.context == "Far below"  # heading of its own section
    assert bottom.context == "Top" and top.context == "Top"  # the section's heading is out of scope


def test_disabled_controls(extract):
    page = extract("disabled")
    enabled = {e.text: e.enabled for e in page.elements}
    assert enabled == {
        "Enabled button": True, "Disabled button": False, "Disabled input": False,
        "Fieldset input": False, "Aria disabled": False, "Disabled select": False,
    }
    assert "Locked" in one(page, "Fieldset input").context


def test_aria_labelledby(extract):
    page = extract("aria_labelledby")
    assert texts(page) == ["Shipping address", "Shipping", "Fallback text", "Explicit", "Hidden label text"]
    assert one(page, "Shipping address").role == "button"


def test_page_from_raw_and_json_roundtrip(browser_page):
    browser_page.goto((FIXTURES / "form.html").as_uri())
    raw = browser_page.evaluate(DOM_JS)
    page = page_from_raw(raw, "u", "pid", "site")
    assert page.page_id == "pid" and page.site == "site" and page.title == "Sign up"
    assert Page.from_dict(page.to_dict()).elements == page.elements


def test_robust_and_fast(browser_page):
    html = "<title>Big</title>" + "".join(
        f'<div><a href="/p{i}">Link {i}</a><button>Btn {i}</button><span style="cursor:pointer">S{i}</span></div>'
        for i in range(1500)
    )
    browser_page.set_content(html)
    start = time.perf_counter()
    raw = browser_page.evaluate(DOM_JS)
    elapsed = time.perf_counter() - start
    assert len(raw["elements"]) == 3000
    assert elapsed < 1.5  # generous bound; typical pages take far less
    browser_page.set_content("")
    assert browser_page.evaluate(DOM_JS)["elements"] == []
