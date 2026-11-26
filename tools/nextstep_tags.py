"""Automatic item tags for the pre-registered next-step evaluation.

The tags are computed, never hand-assigned per example:

* ``literal``: at least half of the target's visible-label content tokens occur in the goal or history;
* ``goal-only``: no visible-label content token occurs in the goal or history;
* ``icon``: a non-text-entry control had no visible text, so its label came only from an accessibility
  description, tooltip, help text or image alt text.

The serialized ``Element.text`` field deliberately merges visible and accessibility text, so callers pass
``text_source`` as provenance evidence. The desktop recorder gets that evidence from which oracle selector
(``label`` or ``description``) resolved the target; web converters can pass their extractor's provenance.
``label``/unknown provenance is treated as visible text because visible text is the common case in the
extractors, while accessibility-only provenance is explicit and conservative.
"""

from __future__ import annotations

from collections.abc import Iterable

from grounder.features import content_tokens
from grounder.records import Element

LITERAL = "literal"
GOAL_ONLY = "goal-only"
ICON = "icon"
TAGS = (LITERAL, GOAL_ONLY, ICON)

VISIBLE_SOURCES = frozenset({"label", "visible", "own", "text", "value", "placeholder", "unknown", ""})
ACCESSIBILITY_SOURCES = frozenset({"description", "aria", "aria-label", "tooltip", "help", "alt", "icon"})
# Text-entry and value controls can have accessibility descriptions without being icon-only controls.
NON_ICON_ROLES = frozenset({"textbox", "searchbox", "combobox", "option", "slider", "checkbox", "radio", "switch", "tab"})


def visible_label_text(element: Element, text_source: str = "label") -> str:
    """The part of ``element.text`` that should count as a visible label under the pre-registration."""
    return element.text if text_source in VISIBLE_SOURCES else ""


def is_icon_only(element: Element, text_source: str = "label") -> bool:
    """Whether the target is a control whose only label is accessibility metadata.

    Role matters as well as provenance: a blank text area whose accessible description is "Script" is not
    an icon-only control, while a toolbar button whose accessible description is "Compile" is.
    """
    return text_source in ACCESSIBILITY_SOURCES and element.role not in NON_ICON_ROLES


def tag_target(element: Element, goal: str, history: Iterable[str] = (), text_source: str = "label") -> set[str]:
    """Compute the pre-registered tags for one target element.

    Tokenisation and stemming are exactly ``grounder.features.content_tokens``. A target with no visible
    tokens is never ``literal`` (zero overlap cannot be half of a non-empty label), but it is ``goal-only``
    in the pre-registered sense: no visible-label token appears in the goal or history.
    """
    label_tokens = content_tokens(visible_label_text(element, text_source))
    prompt_tokens = content_tokens(" ".join([goal, *history]))
    shared = label_tokens & prompt_tokens
    tags: set[str] = set()
    if label_tokens and len(shared) * 2 >= len(label_tokens):
        tags.add(LITERAL)
    if not shared:
        tags.add(GOAL_ONLY)
    if is_icon_only(element, text_source):
        tags.add(ICON)
    return tags
