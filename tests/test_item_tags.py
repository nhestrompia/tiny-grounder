"""Automatic item tags fixed by docs/model-uniqueness-prereg.md."""

from grounder.item_tags import TargetLabel, item_tags


def test_literal_needs_half_of_visible_label_tokens_in_goal_or_history():
    label = TargetLabel(text="Add to cart", visible_text="Add to cart")
    assert item_tags("put it in my basket", [], label) == ["goal-only"]
    assert item_tags("add this product", [], label) == ["literal"]
    assert item_tags("buy the thing", ["[button]  Cart -> CLICK"], label) == ["literal"]


def test_goal_only_uses_the_effective_target_label():
    label = TargetLabel(text="View details for Sauce Labs Backpack", visible_text="Sauce Labs Backpack")
    assert item_tags("buy the bag", [], label) == ["goal-only"]
    assert "goal-only" not in item_tags("open the backpack details", [], label)


def test_icon_requires_no_visible_text_and_a_nonvisual_description():
    assert item_tags("open the cart", [], TargetLabel(text="Cart, empty", nonvisual_text="Cart, empty")) == ["icon"]
    assert "icon" not in item_tags("open the basket", [], TargetLabel(text="Cart", visible_text="Cart"))
    assert "icon" not in item_tags("continue", [], TargetLabel(text=""))


def test_symbol_visible_text_is_visible_but_has_no_literal_content_tokens():
    tags = item_tags("dismiss it", [], TargetLabel(text="Close dialog", visible_text="×", nonvisual_text="Close dialog"))
    assert "icon" not in tags
    assert "literal" not in tags
    assert "goal-only" in tags
