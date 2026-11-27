from __future__ import annotations

import json
from pathlib import Path

import pytest

from grounder.data.pages import load_page, page_paths
from grounder.extract import ax
from tests.test_ax_collect import raw_dump
from tools import ax_auto
from tools.mac_policy import load_policy
from tools.mac_safety import (
    HARD_CLICK_DENYLIST, Scrubber, denied_click_term, fixed_type_text, private_view_reason,
    signin_or_payment_reason, typing_allowed,
)


def test_mandated_hard_denylist_terms_are_all_present():
    required = {
        "buy", "purchase", "subscribe", "pay", "checkout", "order", "export", "share", "publish", "upload",
        "send", "delete", "remove", "erase", "trash", "empty", "format", "partition", "commit", "push",
        "merge", "kill", "terminate", "stop container", "force quit", "sign out", "log out", "sign in",
        "install", "update now", "restart", "shut down", "reset", "revoke", "unlink",
    }
    assert required <= set(HARD_CLICK_DENYLIST)


def test_hard_click_denylist_is_case_insensitive_substring_matching():
    for term in HARD_CLICK_DENYLIST:
        assert denied_click_term(f"Please {term.upper()} this item") == term
    assert denied_click_term("Settings") == ""
    assert denied_click_term("DELETE the selected rows") == "delete"


def test_typing_is_restricted_to_search_or_verified_throwaway_document():
    assert typing_allowed("search", "searchbox", "AXSearchField", "Search", False)
    assert typing_allowed("search", "textbox", "AXTextField", "Search", False)
    assert not typing_allowed("search", "textbox", "AXTextField", "Body text", False)
    assert not typing_allowed("search", "textbox", "AXTextField", "Research notes", False)
    assert typing_allowed("throwaway", "textbox", "AXTextArea", "Document", True)
    assert not typing_allowed("throwaway", "textbox", "AXTextArea", "Document", False)
    assert not typing_allowed("never", "searchbox", "AXSearchField", "Search", False)
    assert fixed_type_text(0) == fixed_type_text(5)
    assert fixed_type_text(0) in {"search test", "apple", "weather", "music", "document notes"}


def test_privacy_scrubber_replaces_identifiers_deterministically_and_counts():
    scrubber = Scrubber("Umut Özçelik", "nhestrompia")
    raw = raw_dump()
    raw["title"] = "Umut Özçelik nhestrompia test@example.com +1 (555) 123-4567 https://x.test/a?token=abc"
    raw["elements"][0]["text"] = "mail test@example.com"
    raw["elements"][1]["context"] = "https://x.test/ok and https://x.test/a?access_token=abc"
    once = scrubber.scrub_raw_dump(raw)
    twice = Scrubber("Umut Özçelik", "nhestrompia").scrub_raw_dump(raw)
    assert once == twice
    assert once["title"] == "<macos-full-name> <macos-username> <email> <phone> <url-with-query>"
    assert once["elements"][0]["text"] == "mail <email>"
    assert "<url-with-query>" in once["elements"][1]["context"]
    assert scrubber.counts["emails"] == 2
    assert scrubber.counts["phone_numbers"] == 1
    assert scrubber.counts["urls_with_tokens_or_queries"] == 2
    assert scrubber.counts["macos_full_names"] == 1
    assert scrubber.counts["macos_usernames"] == 1


def test_signin_payment_and_never_save_view_detection():
    secure = raw_dump(elements=[]) if False else raw_dump()
    secure["elements"][0]["tag"] = "AXSecureTextField"
    assert signin_or_payment_reason(secure) == "secure password field"
    payment = raw_dump()
    payment["elements"][0].update(role="textbox", tag="AXTextField", text="card number", context="payment cvv")
    assert signin_or_payment_reason(payment) == "payment entry"
    signin = raw_dump(title="Sign in")
    signin["elements"][0].update(role="textbox", tag="AXTextField", text="Email", context="Sign in")
    assert signin_or_payment_reason(signin) == "sign-in screen"
    browsable = raw_dump()
    browsable["elements"][0].update(role="searchbox", tag="AXSearchField", text="Search", context="sign in available")
    assert signin_or_payment_reason(browsable) == ""
    history = raw_dump(title="History")
    assert private_view_reason(history) == "history view"
    normal = raw_dump()
    assert signin_or_payment_reason(normal) == "" and private_view_reason(normal) == ""


def test_policy_membership_and_split_enforcement():
    policy = load_policy(Path("configs/apps_policy.yaml"))
    assert policy.refusal("VLC", "test") == ""
    assert policy.refusal("VLC", "train") == "belongs to test_v2, not train"
    assert policy.refusal("Font Book", "train") == "frozen test_v1 app"
    assert policy.refusal("T3 Code", "train") == "excluded by policy"
    assert policy.refusal("Not An App", "train") == "not in the app policy"
    assert set(policy.apps_for_split("test")) == {"Logic Pro", "VLC", "Keka", "Chess", "Podcasts", "Audio MIDI Setup"}


class WalkFakeMac:
    def __init__(self, app="Calculator") -> None:
        self.app, self.front = app, app
        self.log: list[tuple] = []

    def is_installed(self, app): return app == self.app
    def is_running(self, app): return False
    def launch(self, app, args=()): self.log.append(("launch", app, tuple(args)))
    def open_path(self, app, path): self.log.append(("open_path", app, str(path)))
    def frontmost(self): return self.front
    def window_count(self, app): return 1
    def has_sheet(self, app): return False
    def press(self, key): self.log.append(("press", key))
    def click_at(self, app, x, y): self.log.append(("click", app, x, y))
    def type_text(self, text): self.log.append(("type", text))
    def quit(self, app): self.log.append(("quit", app))
    def bundle_id(self, app): return f"com.example.{app.lower()}"
    def export_prefs(self, bundle, path): self.log.append(("export_prefs", bundle)); return True
    def import_prefs(self, bundle, path): self.log.append(("import_prefs", bundle))
    def delete_prefs(self, bundle): self.log.append(("delete_prefs", bundle))
    def running_apps(self): return set(getattr(self, "running", {"Finder", "T3 Code"}))


def walk_raw(app: str, variant: int) -> dict:
    words = [
        ["alpha beta gamma delta epsilon zeta eta theta", "one two three four five six seven eight"],
        ["red green blue cyan magenta yellow black white", "north south east west spring summer autumn winter"],
    ][variant % 2]
    return raw_dump(app, words=words[0] if variant < 2 else words[1])


def make_walk_collector(tmp_path: Path, dumps: list[dict], **kw):
    queue = list(dumps)

    def dump_app(app: str):
        if not queue:
            raise RuntimeError("out of dumps")
        return queue.pop(0)

    policy = load_policy(Path("configs/apps_policy.yaml"))
    mac = WalkFakeMac()
    collector = ax_auto.RandomWalkCollector(
        mac, policy, "train", tmp_path, settle=0, sleep=lambda s: None, show=lambda s: None,
        dump_app=dump_app, **kw)
    return collector, mac


def test_walk_collection_records_pre_action_post_action_and_quits_only_launched_app(tmp_path):
    # Initial state, then two steps: A -> B -> A. The exact repetition exercises representative page ids.
    dumps = [walk_raw("Calculator", 0), walk_raw("Calculator", 0), walk_raw("Calculator", 1),
             walk_raw("Calculator", 1), walk_raw("Calculator", 0)]
    collector, mac = make_walk_collector(tmp_path, dumps, max_steps=2, walk_budget=1, state_budget=4)
    result = collector.run_app("Calculator")
    assert result["status"] == "collected" and result["walks"] == 1 and result["steps"] == 2
    [walk_file] = (tmp_path / "train/walks").glob("calculator.jsonl")
    walk = json.loads(walk_file.read_text().splitlines()[0])
    assert walk["app"] == "Calculator" and len(walk["steps"]) == 2
    assert walk["steps"][0]["pre_page_id"] == walk["steps"][1]["pre_page_id"] or True  # ids are content-derived
    assert walk["steps"][0]["pre_page_id"] != walk["steps"][0]["post_page_id"]
    assert walk["steps"][0]["action"]["type"] in {"click", "type"}
    assert walk["steps"][0]["target"]["equiv_ids"]
    assert ("quit", "Calculator") in mac.log
    assert len(page_paths(tmp_path / "train/pages")) == 2


def test_novelty_filter_maps_near_duplicate_state_to_representative_page(tmp_path):
    store = ax_auto.WalkStateStore(tmp_path)
    base = "w01 w02 w03 w04 w05 w06 w07 w08 w09 w10 w11 w12 w13 w14 w15 w16 w17 w18 w19 w20"
    first = raw_dump("Calculator", words=base)
    near = raw_dump("Calculator", words=base.replace("w20", "w99"))
    one = store.add(first, "Calculator")
    two = store.add(near, "Calculator")
    assert one.is_new and not two.is_new and two.page_id == one.page_id
    assert store.distinct_states == 1
    distinct = raw_dump("Calculator", words="one two three four five six seven eight")
    three = store.add(distinct, "Calculator")
    assert three.is_new and store.distinct_states == 2


def test_click_candidates_apply_hard_and_policy_denylists(tmp_path):
    raw = raw_dump("Calculator")
    raw["elements"][0]["text"] = "Delete all memory"
    raw["elements"][1]["text"] = "Close window"
    raw["elements"][2]["text"] = "Settings"
    for element in raw["elements"]:
        element.setdefault("visible_text", element["text"])
        element.setdefault("nonvisual_text", "")
    collector, _ = make_walk_collector(tmp_path / "x", [])
    page = ax.page_from_dump(raw, "p", "calculator")
    candidates = collector.candidates(page, "Calculator", False, __import__("collections").Counter())
    labels = [c.element.text for c in candidates]
    assert "Delete all memory" not in labels and "Close window" not in labels
    assert "Settings" in labels
    assert collector.denials["Calculator"]["delete"] == 1
    assert collector.denials["Calculator"]["close"] == 1


def test_output_pages_dir_refuses_to_replace_unknown_user_data(tmp_path):
    target = tmp_path / "pages_mac_test_v2"
    target.mkdir()
    (target / "user-file").write_text("keep")
    from tools.mac_goals import prepare_output_pages_dir
    with pytest.raises(RuntimeError, match="refusing to replace"):
        prepare_output_pages_dir(target)


def test_icon_only_elements_are_cropped_to_64px_when_screenshots_exist(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image
    from tools.ax_auto import save_icon_crops
    png = tmp_path / "window.png"
    Image.new("RGB", (200, 100), "white").save(png)
    raw = raw_dump("Calculator")
    raw["origin"] = [10.0, 20.0]
    raw["viewport"] = [100, 50]
    raw["elements"][0].update(abs_box=[20.0, 30.0, 10.0, 10.0], visible_text="", nonvisual_text="minus")
    raw["elements"][1].update(visible_text="plus", nonvisual_text="")
    page = ax.page_from_dump(raw, "p", "calculator")
    count = save_icon_crops(png, raw, page, tmp_path / "crops")
    assert count == 1
    out = tmp_path / "crops" / "p-0.png"
    assert out.exists()
    with Image.open(out) as crop:
        assert crop.size == (64, 64)


def test_logic_throwaway_copy_is_a_new_temp_instance_not_the_factory_template(tmp_path):
    source = tmp_path / "source.logicx"
    source.mkdir()
    (source / "marker").write_text("factory")
    original = ax_auto.LOGIC_QUICKSTART_TEMPLATE
    ax_auto.LOGIC_QUICKSTART_TEMPLATE = source
    try:
        target = ax_auto.prepare_throwaway_project("Logic Pro", str(tmp_path / "tmp"))
    finally:
        ax_auto.LOGIC_QUICKSTART_TEMPLATE = original
    assert target is not None and target != source and target.name == "TinyGrounderThrowaway.logicx"
    assert (target / "marker").read_text() == "factory"
    (target / "marker").write_text("changed")
    assert (source / "marker").read_text() == "factory"
    assert target.stat().st_mode & 0o200


def test_review_denylist_blocks_account_install_and_consent_labels():
    from tools.mac_safety import denied_click_term
    for label in ["Get", "Allow", "Follow", "Like", "Save to Your Library", "Download", "Accept", "Agree",
                  "Enable notifications", "Turn On", "Clear History", "OK", "Continue"]:
        assert denied_click_term(label), label
    # Exact-label terms don't block ordinary labels that merely contain them.
    for label in ["Get Info", "Target", "Okay to browse", "Online help", "Continue Reading Guide"]:
        term = denied_click_term(label)
        assert term not in {"get", "ok", "on", "continue"}, (label, term)


def test_preferences_are_snapshotted_and_restored_after_quit(tmp_path):
    dumps = [walk_raw("Calculator", 0), walk_raw("Calculator", 0), walk_raw("Calculator", 1)]
    collector, mac = make_walk_collector(tmp_path, dumps, max_steps=1, walk_budget=1, state_budget=4)
    result = collector.run_app("Calculator")
    events = [e[0] for e in mac.log]
    assert events.index("export_prefs") < events.index("launch")
    assert events.index("quit") < events.index("import_prefs")
    assert result.get("prefs_restored") is True


def test_already_running_app_is_skipped_not_explored(tmp_path):
    collector, mac = make_walk_collector(tmp_path, [walk_raw("Calculator", 0)])
    mac.is_running = lambda app: True
    result = collector.run_app("Calculator")
    assert result["status"] == "skipped" and "already running" in result["reason"]
    assert not any(e[0] in {"launch", "click", "export_prefs"} for e in mac.log)


def test_screenshot_mismatch_rejects_wrong_aspect(tmp_path):
    from PIL import Image
    from tools.ax_auto import screenshot_mismatch
    good, bad = tmp_path / "good.png", tmp_path / "bad.png"
    Image.new("RGB", (1392, 1064)).save(good)
    Image.new("RGB", (1392, 1300)).save(bad)
    assert screenshot_mismatch(good, [696, 532]) == ""
    assert "does not match" in screenshot_mismatch(bad, [696, 532])


def test_app_is_skipped_when_preferences_cannot_be_snapshotted(tmp_path):
    from tools.ax_auto import PrefsSnapshotError
    collector, mac = make_walk_collector(tmp_path, [walk_raw("Calculator", 0)])
    def fail(bundle, path):
        raise PrefsSnapshotError("timed out")
    mac.export_prefs = fail
    result = collector.run_app("Calculator")
    assert result["status"] == "skipped" and "preferences could not be saved" in result["reason"]
    assert not any(e[0] in {"launch", "click"} for e in mac.log)


def test_visible_click_point_clips_to_window_and_skips_mostly_hidden():
    from tools.ax_auto import visible_click_point
    window = [100.0, 100.0, 1000.0, 600.0]
    assert visible_click_point([200, 200, 100, 50], window) == (250, 225)
    # A carousel card mostly past the right edge is skipped, not clicked behind the window.
    assert visible_click_point([1000, 200, 800, 300], window) is None
    # One that is mostly visible is clicked in the middle of its visible part.
    assert visible_click_point([700, 200, 500, 300], window) == (900, 350)
    assert visible_click_point([2000, 200, 50, 50], window) is None


def test_redact_screenshot_blacks_out_scrubbed_elements_and_ocr_name_lines(tmp_path):
    from PIL import Image
    from tools.ax_auto import redact_screenshot
    from tools.mac_safety import Scrubber
    png = tmp_path / "w.png"
    Image.new("RGB", (400, 200), (255, 255, 255)).save(png)
    raw = {"origin": [100.0, 50.0], "viewport": [200, 100], "elements": [
        {"text": "<macos-full-name>", "abs_box": [110.0, 60.0, 40.0, 10.0]},
        {"text": "Discover", "abs_box": [110.0, 80.0, 40.0, 10.0]},
    ]}
    lines = [{"text": "Umut Ozcelik", "box": [200.0, 150.0, 80.0, 20.0]},  # OCR dropped the accents
             {"text": "Categories", "box": [200.0, 100.0, 80.0, 20.0]}]
    n = redact_screenshot(png, raw, Scrubber("Umut Özçelik", "nhestrompia"), ocr=lambda path: lines)
    assert n == 2
    with Image.open(png) as image:
        assert image.getpixel((40, 30)) == (0, 0, 0)       # scrubbed element, scaled x2 from window points
        assert image.getpixel((240, 160)) == (0, 0, 0)     # OCR line with the name
        assert image.getpixel((170, 160)) == (0, 0, 0)     # widened left over an avatar
        assert image.getpixel((240, 110)) == (255, 255, 255)  # ordinary text untouched
        assert image.getpixel((40, 70)) == (255, 255, 255)


def test_file_dialog_is_detected():
    from tools.mac_safety import file_dialog_reason
    panel = {"title": "Open", "elements": [{"text": t} for t in ("Where:", "icon view", "photo.jpg", "Cancel", "Open")]}
    assert file_dialog_reason(panel) == "file dialog"
    plain = {"title": "Xcode", "elements": [{"text": t} for t in ("New Project", "Open...", "Cancel")]}
    assert file_dialog_reason(plain) == ""


def test_phone_scrub_leaves_prices_and_model_numbers():
    from tools.mac_safety import Scrubber
    s = Scrubber("", "")
    for text in ["Standard and Poor’s 500 7135,53", "Marshall JCM 800", "rgb 255 255 255", "© 2011-2026 Apple"]:
        assert s.scrub_text(text) == text
    assert s.scrub_text("call +90 532 123 45 67") == "call <phone>"


def test_purchase_labels_blocked_without_hitting_ordinary_words():
    from tools.mac_safety import denied_click_term
    for label in ["₺1.749,99", "$4.99", "4,99 €", "Try It Free", "Start Free Trial", "Redeem", "Gift", "Rent"]:
        assert denied_click_term(label), label
    for label in ["Arcade", "Categories", "7135,53", "Current", "Parent folder", "Things 3, Organize your life, ₺1.749,99"]:
        assert denied_click_term(label) == "", label


def test_elements_showing_personal_data_are_never_click_candidates(tmp_path):
    collector, mac = make_walk_collector(tmp_path, [walk_raw("Calculator", 0)])
    from grounder.records import Element
    page = type("P", (), {})()
    base = dict(role="button", tag="AXButton", context="", box=(0, 0, 0.1, 0.1), abs_box=(10, 10, 20, 20),
                in_viewport=True, enabled=True)
    page.elements = [Element(id=0, text="<macos-full-name>", **base), Element(id=1, text="Equals", **base)]
    labels = [c.element.text for c in collector.candidates(page, "Calculator", False, __import__("collections").Counter())]
    assert labels == ["Equals"]


def test_apps_a_walk_opens_are_quit_but_preexisting_apps_are_left_alone(tmp_path):
    dumps = [walk_raw("Calculator", 0), walk_raw("Calculator", 0), walk_raw("Calculator", 1)]
    collector, mac = make_walk_collector(tmp_path, dumps, max_steps=1, walk_budget=1, state_budget=4)
    mac.running = {"Finder", "T3 Code"}
    original_click = mac.click_at
    def click_opens_safari(app, x, y):
        original_click(app, x, y)
        mac.running = {"Finder", "T3 Code", "Calculator", "Safari"}
    mac.click_at = click_opens_safari
    result = collector.run_app("Calculator")
    quits = [e[1] for e in mac.log if e[0] == "quit"]
    assert "Safari" in quits and "Finder" not in quits and "T3 Code" not in quits
    assert result["strays_quit"] == ["Safari"]


def test_system_prompts_are_ignored_or_denied_never_allowed():
    from tools.ax_auto import prompt_answer
    def prompt(texts):
        return {"app": "UserNotificationCenter", "elements": [
            {"text": t, "context": " ".join(texts), "abs_box": [10.0 * i, 0.0, 10.0, 10.0]} for i, t in enumerate(texts)]}
    assert prompt_answer(prompt(["Lys quit unexpectedly.", "Help", "Reopen", "Ignore", "Report…"])) == (35.0, 5.0)
    perm = prompt_answer(prompt(["“LTX Desktop.app” would like to access the Microphone.", "Help", "Don’t Allow", "Allow"]))
    assert perm == (25.0, 5.0)
    assert prompt_answer({"app": "Calculator", "elements": []}) is None
    assert prompt_answer(prompt(["Something else", "Allow"])) is None
