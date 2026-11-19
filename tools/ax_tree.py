"""Accessibility tree (a nested dict dump of one macOS window) -> `Page`, with the axdump rules.

Shared by `tools/guirilla.py` and `tools/screen2ax.py`. Ports `grounder/extract/macos/Sources/AXCore/Logic.swift`:
same role map, text priority, group / sibling context, window-relative `box`, screen-point `abs_box`, visibility
test, depth and element caps. The dumps carry no AX action lists, so "has an action" is assumed for the interactive
roles of the role map (buttons, links, checkboxes, radios, pop-ups, menu items, sliders, ...) and for no other:
static text, images and the group / cell / row containers (actionable only when they have AXPress) are dropped.

A node is expected to carry `role`, `children` and `name` / `description` / `value` / `role_description`, plus
`absolute_position` ("x;y" in screen points) and `size` ("w;h"); `id` and `enabled` are optional.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from grounder.records import MAX_CONTEXT_CHARS, MAX_TEXT_CHARS, Element, Page, clean_text, validate_element

MAX_DEPTH = 40
MAX_ELEMENTS = 2000
MAX_GROUP_CHARS = 40

# Ported from AXCore/Logic.swift `roleMap`; AXStaticText and AXImage are context only, so never kept here.
ROLE_MAP: dict[str, str] = {
    "AXButton": "button", "AXLink": "link", "AXTextField": "textbox", "AXTextArea": "textbox",
    "AXSearchField": "searchbox", "AXCheckBox": "checkbox", "AXRadioButton": "radio",
    "AXPopUpButton": "combobox", "AXComboBox": "combobox", "AXMenuItem": "menuitem",
    "AXMenuButton": "button", "AXSlider": "slider", "AXIncrementor": "slider",
    "AXDisclosureTriangle": "button",
}
GROUP_ROLES = frozenset({"AXGroup", "AXToolbar", "AXScrollArea", "AXTabGroup", "AXSplitGroup"})
EMPTY_NAME = re.compile(r"empty-name-AX\w*")  # GUIrilla's placeholder for a node without a name

Node = dict[str, Any]


@dataclass
class ConvertedPage:
    page: Page
    node_to_element: dict[str, int]  # source node id -> our element id (nodes without an id are not listed)


def map_role(ax_role: str, parent_role: str | None) -> str | None:
    """The record role of a kept element, or None (see the module docstring for the action assumption)."""
    if ax_role == "AXRadioButton" and parent_role == "AXTabGroup":
        return "tab"
    return ROLE_MAP.get(ax_role)


def parse_pair(s: str) -> list[float]:
    """"x;y" or "w;h" -> [a, b]."""
    return [float(v) for v in s.split(";")]


def node_rect(node: Node) -> list[float] | None:
    """[x, y, w, h] in screen points, or None for a node without a frame (a few groups have an empty size)."""
    if not node.get("absolute_position") or not node.get("size"):
        return None
    return [*parse_pair(node["absolute_position"]), *parse_pair(node["size"])]


def is_visible(rect: list[float], window: list[float]) -> bool:
    """Port of `isVisible`: the rect has area and overlaps the window frame."""
    if rect[2] < 1 or rect[3] < 1:
        return False
    return (rect[0] < window[0] + window[2] and rect[0] + rect[2] > window[0]
            and rect[1] < window[1] + window[3] and rect[1] + rect[3] > window[1])


def normalised_box(rect: list[float], window: list[float]) -> list[float]:
    """Port of `normalisedBox`: [x/ww, y/wh, w/ww, h/wh] relative to the window, rounded to 4 decimals."""
    box = [(rect[0] - window[0]) / window[2], (rect[1] - window[1]) / window[3], rect[2] / window[2], rect[3] / window[3]]
    return [round(v, 4) for v in box]


def node_name(node: Node) -> str:
    name = node.get("name") or ""
    return "" if EMPTY_NAME.fullmatch(name) else name


def pick_text(node: Node) -> str:
    """First non-empty of name (title), description, value; a non-string or secure-field value is skipped."""
    secure = "secure" in (node.get("role_description") or "")
    value = node.get("value")
    candidates = [node_name(node), node.get("description"), value if isinstance(value, str) and not secure else None]
    return next((t for t in (clean_text(c, MAX_TEXT_CHARS) for c in candidates) if t), "")


def build_context(window_title: str, group_title: str, sibling_texts: list[str], own_text: str) -> str:
    """Port of `buildContext`: window title, nearest titled group, sibling static texts; own text removed."""
    siblings = [t for t in (clean_text(s, MAX_TEXT_CHARS) for s in sibling_texts) if t and t != own_text]
    parts = [clean_text(p, MAX_CONTEXT_CHARS) for p in (window_title, group_title, " ".join(siblings))]
    return " · ".join(p for p in parts if p and p != own_text)[:MAX_CONTEXT_CHARS]


def sibling_texts(children: list[Node]) -> list[str]:
    return [c["value"] if isinstance(c.get("value"), str) else node_name(c) for c in children if c["role"] == "AXStaticText"]


def group_title(node: Node, inherited: str) -> str:
    """The node's own short title when it is a titled group-like container, else the inherited one."""
    if node["role"] not in GROUP_ROLES:
        return inherited
    return clean_text(node_name(node) or node.get("description") or "", MAX_GROUP_CHARS) or inherited


class _Walker:
    def __init__(self, window: list[float], window_title: str) -> None:
        self.window = window
        self.window_title = window_title
        self.elements: list[Element] = []
        self.node_to_element: dict[str, int] = {}

    def visit(self, node: Node, parent_role: str | None, group: str, siblings: list[str], depth: int) -> None:
        if len(self.elements) >= MAX_ELEMENTS or depth > MAX_DEPTH:
            return
        role = map_role(node["role"], parent_role)
        rect = node_rect(node)
        if role is not None and rect is not None and is_visible(rect, self.window):
            text = pick_text(node)
            if node.get("id") is not None:
                self.node_to_element[node["id"]] = len(self.elements)
            self.elements.append(validate_element({
                "id": len(self.elements), "role": role, "text": text,
                "context": build_context(self.window_title, group, siblings, text), "tag": node["role"],
                "box": normalised_box(rect, self.window), "abs_box": rect, "in_viewport": True,
                "enabled": bool(node.get("enabled", True)), "source": "ax",
            }))
        children = node.get("children") or []
        next_group, kid_texts = group_title(node, group), sibling_texts(children)
        for child in children:
            self.visit(child, node["role"], next_group, kid_texts, depth + 1)


def flatten_tree(tree: Node, page_id: str, url: str, site: str) -> ConvertedPage:
    """Flatten one accessibility tree (rooted at the window) into a Page with the axdump rules."""
    window = node_rect(tree)
    if window is None:
        raise ValueError(f"{page_id}: the window node has no frame")
    title = clean_text(node_name(tree), MAX_CONTEXT_CHARS)
    walker = _Walker(window, title)
    walker.visit(tree, None, "", [], 0)
    size = [int(window[2]), int(window[3])]
    page = Page(page_id=page_id, url=url, site=site, title=title, viewport=size, doc_size=size, elements=walker.elements)
    return ConvertedPage(page, walker.node_to_element)


def element_at(elements: list[Element], point: tuple[float, float]) -> int | None:
    """Id of the smallest element whose abs_box contains the point."""
    x, y = point
    hits = [e for e in elements if e.abs_box[0] <= x <= e.abs_box[0] + e.abs_box[2] and e.abs_box[1] <= y <= e.abs_box[1] + e.abs_box[3]]
    return min(hits, key=lambda e: e.abs_box[2] * e.abs_box[3]).id if hits else None


def tree_key(page: Page) -> str:
    """Hash of the page's element (role, text) list: equal for two dumps of the same screen."""
    signature = repr([(e.role, e.text) for e in page.elements])
    return hashlib.sha1(signature.encode("utf-8")).hexdigest()
