from grounder.records import Element
from tools.nextstep_tags import GOAL_ONLY, ICON, LITERAL, is_icon_only, tag_target


def test_literal_needs_at_least_half_the_visible_label_tokens():
    save = Element(0, "button", "Save file")
    assert tag_target(save, "save the document") == {LITERAL}
    assert tag_target(save, "put the document away") == {GOAL_ONLY}
    assert tag_target(save, "use the file") == {LITERAL}  # one of two stems is exactly half
    long_label = Element(1, "button", "Save file permanently")
    assert tag_target(long_label, "save the document") == set()  # one of three: neither literal nor goal-only


def test_goal_only_uses_goal_and_history_and_is_disjoint_from_literal():
    close = Element(0, "button", "Close")
    assert tag_target(close, "dismiss the panel") == {GOAL_ONLY}
    assert tag_target(close, "dismiss the panel", ["[AXButton]  Close -> CLICK"]) == {LITERAL}
    tags = tag_target(close, "close the panel")
    assert LITERAL in tags and GOAL_ONLY not in tags


def test_icon_requires_accessibility_provenance_and_a_non_text_entry_role():
    compile_button = Element(0, "button", "Compile")
    script_area = Element(1, "textbox", "Script")
    assert tag_target(compile_button, "check the syntax", text_source="description") == {GOAL_ONLY, ICON}
    assert tag_target(compile_button, "check the syntax", text_source="label") == {GOAL_ONLY}
    assert not is_icon_only(script_area, "description")


def test_empty_visible_label_is_goal_only_but_never_literal():
    assert tag_target(Element(0, "button", ""), "press the control") == {GOAL_ONLY}
