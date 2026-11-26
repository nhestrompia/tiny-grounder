"""Automatic per-item tags for the model-uniqueness evaluation.

The definitions are the pre-registered ones in docs/model-uniqueness-prereg.md. Tokenisation is the project's
shared `content_tokens`: lowercase alphanumeric words, without function/generic UI words, lightly stemmed.
"""

from __future__ import annotations

from dataclasses import dataclass

from grounder.features import content_tokens


@dataclass(frozen=True)
class TargetLabel:
    """What the extraction/UI knows about a target's label.

    text:           the effective extracted label (`Element.text`), whatever its source.
    visible_text:   label text a user can see rendered for the control itself (its own text, a visible HTML label,
                    placeholder, button value or selected option); empty when there is none.
    nonvisual_text: accessibility description, tooltip or image alt text available when the control has no visible text.
    """

    text: str
    visible_text: str = ""
    nonvisual_text: str = ""


def item_tags(goal: str, history: list[str], target: TargetLabel) -> list[str]:
    """Return the automatic pre-registered tags (`literal`, `goal-only`, `icon`) for one target.

    `literal` is measured against visible-label tokens only. `goal-only` uses the target's effective extracted label,
    so an icon control's accessibility name still participates in the lexical-overlap analysis.
    """
    context = content_tokens(" ".join([goal, *history]))
    visible = content_tokens(target.visible_text)
    label = content_tokens(target.text)
    tags: list[str] = []
    if visible and len(visible & context) / len(visible) >= 0.5:
        tags.append("literal")
    if not (label & context):
        tags.append("goal-only")
    if not target.visible_text.strip() and bool(target.nonvisual_text.strip()):
        tags.append("icon")
    return tags
