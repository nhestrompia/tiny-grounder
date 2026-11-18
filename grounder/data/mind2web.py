"""Mind2Web -> our records and examples (goal mode).

Source: `osunlp/Mind2Web` (cleaned_html, pos/neg candidates). Train is downloaded from the dataset repo's raw
files `data/train/train_*.json` (11 files, about 6 GB, each a JSON list of rows): listed through the hub API,
fetched resumably into `--cache` (complete files are skipped, partial ones continue with an HTTP Range request)
and parsed as a stream, so memory stays at one row. With `--limit N` the first N rows are streamed straight
from the hub without caching (quick runs). The test splits (test_task, test_website, test_domain) are
distributed separately as a password-protected zip: unzip it yourself and pass the JSON files (a JSON list
of rows, a JSONL file, or a directory of those) with `--input`.

    uv run python -m grounder.data.mind2web --split train --out data/mind2web --cache data/mind2web_raw
    uv run python -m grounder.data.mind2web --split train --limit 50 --out data/mind2web_small
    uv run python -m grounder.data.mind2web --split train --only train_10.json --out ...   # one raw file
    uv run python -m grounder.data.mind2web --split test_task --input test_task/ --out data/mind2web

Writes `{out}/pages/{split}/{site}/{page_id}.json.gz` and `{out}/{split}.jsonl`. One page and one Example
(kind "goal", mode "goal", query = the task, history = the last 3 earlier action_reprs) per action step.

Conversion rules (same spirit as extract/dom.js, approximated on the cleaned HTML tree):
  * elements = positive + negative candidates, ordered by backend_node_id (document order, so the target
    is not first); target_id = first positive, equiv_ids = all positives.
  * role: explicit role attribute, else tag/type (a link, button, select combobox, input by type, ...);
    other tags are "text" (headings, p, label, ...) or "other"; a clickable text tag is "other".
  * text: aria_label, alt, placeholder, button value, visible text of the node, title, selected option.
  * context: text of the nearest row-like ancestor (li, tr, article, role listitem/row, or a div of at most
    300 chars) minus the element's own text, plus the nearest preceding heading inside the closest
    section/form/landmark. Containers keep their (clipped) visible text, so generic divs look different
    from real controls only by role and text, as in the benchmark's candidate set.
  * box: bounding_box_rect is "x,y,w,h" in page pixels. The viewport is not recorded: assumed 1280x720, at
    scroll 0, so box = [x/1280, y/720, w/1280, h/720] and in_viewport means y < 720. No href (not in the data).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Iterator, Sequence

import httpx
import ijson
from lxml import etree

from grounder.data.pages import write_examples, write_page
from grounder.records import (
    MAX_CONTEXT_CHARS, MAX_TEXT_CHARS, ROLES, Element, Example, Page, clean_text,
)

VIEWPORT = (1280, 720)
HISTORY = 3
ROW_TAGS = {"li", "tr", "article"}
ROW_ROLES = {"listitem", "row"}
MAX_ROW_TEXT = 300
SECTION_TAGS = {"section", "form", "main", "nav", "aside", "header", "footer", "dialog"}
HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")
TEXT_TAGS = {"p", "label", "td", "th", "span", "strong", "em", "b", "i", "legend", "caption", "time", "small",
             "blockquote", "pre", "code", *HEADING_TAGS}
ROLE_ALIAS = {"menuitemcheckbox": "menuitem", "menuitemradio": "menuitem", "spinbutton": "textbox",
              "treeitem": "option", "img": "image"}
BUTTON_INPUT_TYPES = {"button", "submit", "reset", "image", "file", "color"}
HUB = "https://huggingface.co"
TRAIN_DIR = "data/train"
CHUNK_BYTES = 1 << 20
_HEADING_XPATH = "preceding::*[" + " or ".join(f"self::{t}" for t in HEADING_TAGS) + "]"
_PARSER = etree.XMLParser(recover=True, huge_tree=True)


def flat(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def input_role(kind: str) -> str:
    if kind in BUTTON_INPUT_TYPES:
        return "button"
    if kind in ("checkbox", "radio"):
        return kind
    return {"range": "slider", "search": "searchbox"}.get(kind, "textbox")


def role_of(tag: str, attrs: dict[str, Any]) -> str:
    explicit = flat(str(attrs.get("role", ""))).split(" ")[0].lower()
    if explicit and explicit not in ("presentation", "none"):
        role = ROLE_ALIAS.get(explicit, explicit)
        return role if role in ROLES else "other"
    if tag == "a":
        return "link"
    if tag in ("button", "summary"):
        return "button"
    if tag == "select":
        return "combobox"
    if tag == "textarea":
        return "textbox"
    if tag in ("img", "svg"):
        return "image"
    if tag == "input":
        return input_role(str(attrs.get("type", "")).lower())
    if tag in TEXT_TAGS:
        return "other" if attrs.get("is_clickable") == "true" else "text"
    return "other"


class Tree:
    """A parsed cleaned_html with cached per-node visible text, looked up by backend_node_id."""

    def __init__(self, cleaned_html: str) -> None:
        self.root = etree.fromstring(cleaned_html.encode("utf-8"), _PARSER)
        self.by_id = {el.get("backend_node_id"): el for el in self.root.iter() if isinstance(el.tag, str)}
        self._text: dict[int, str] = {}

    def text(self, node: Any) -> str:
        key = id(node)
        if key not in self._text:
            self._text[key] = flat(" ".join(node.itertext()))
        return self._text[key]

    def row_text(self, node: Any, own: str) -> str:
        for anc in node.iterancestors():
            tag = anc.tag
            if tag in ROW_TAGS or anc.get("role") in ROW_ROLES:
                return self.text(anc)
            if tag == "div":
                text = self.text(anc)
                if len(text) > MAX_ROW_TEXT:
                    return ""
                if len(text) > len(own) and text != own:
                    return text
        return ""

    def heading_before(self, node: Any) -> str:
        scope = next((a for a in node.iterancestors() if a.tag in SECTION_TAGS), self.root)
        for heading in reversed(node.xpath(_HEADING_XPATH)):
            if next((a for a in heading.iterancestors() if a.tag in SECTION_TAGS), self.root) is scope:
                return self.text(heading)
        return ""


def _without(text: str, part: str) -> str:
    rest = text.replace(part, " ") if part else text
    return flat(re.sub(r"^[\s|·•]+|[\s|·•]+$", "", rest))


def element_text(tag: str, attrs: dict[str, Any], tree: Tree, node: Any) -> tuple[str, str]:
    """(label shown to the model, the node's own visible text)."""
    own = "" if tag in ("input", "select", "textarea") or node is None else tree.text(node)
    button_value = attrs.get("value", "") if tag == "input" and attrs.get("type") in ("button", "submit", "reset") else ""
    selected = ""
    if tag == "select" and node is not None:
        selected = next((tree.text(o) for o in node.iter("option") if o.get("option_selected")), "")
    candidates = [attrs.get("aria_label"), attrs.get("alt"), attrs.get("placeholder"), button_value, own,
                  attrs.get("title"), selected]
    return clean_text(next((c for c in candidates if c), ""), MAX_TEXT_CHARS), own


def element_context(tree: Tree, node: Any, text: str, own: str) -> str:
    if node is None:
        return ""
    row = _without(_without(tree.row_text(node, own), own), text)
    if row == text:
        row = ""
    heading = tree.heading_before(node)
    if heading == text or (heading and heading in row):
        heading = ""
    heading = clean_text(heading, 40)
    budget = MAX_CONTEXT_CHARS - (len(heading) + 3 if heading else 0)
    return " · ".join(p for p in (clean_text(row, budget), heading) if p)


def candidate_attrs(cand: dict[str, Any]) -> dict[str, Any]:
    raw = cand["attributes"]
    return raw if isinstance(raw, dict) else json.loads(raw)


def to_element(eid: int, cand: dict[str, Any], tree: Tree) -> Element:
    attrs, tag = candidate_attrs(cand), cand["tag"]
    node = tree.by_id.get(str(cand["backend_node_id"]))
    text, own = element_text(tag, attrs, tree, node)
    x, y, w, h = (float(v) for v in attrs["bounding_box_rect"].split(","))
    vw, vh = VIEWPORT
    return Element(
        id=eid, role=role_of(tag, attrs), text=text, context=element_context(tree, node, text, own), tag=tag,
        box=[round(x / vw, 4), round(y / vh, 4), round(w / vw, 4), round(h / vh, 4)],
        abs_box=[x, y, w, h], in_viewport=y < vh and y + h > 0, source="dom",
    )


def convert_action(row: dict[str, Any], index: int) -> tuple[Page, Example] | None:
    """One action step -> (page, goal example); None when the step has no positive candidate."""
    action = row["actions"][index]
    if not action["pos_candidates"]:
        return None
    tree = Tree(action["cleaned_html"])
    cands = sorted(action["pos_candidates"] + action["neg_candidates"], key=lambda c: int(c["backend_node_id"]))
    positive = {c["backend_node_id"] for c in action["pos_candidates"]}
    elements = [to_element(i, c, tree) for i, c in enumerate(cands)]
    pos_ids = [i for i, c in enumerate(cands) if c["backend_node_id"] in positive]
    target = next(i for i, c in enumerate(cands) if c["backend_node_id"] == action["pos_candidates"][0]["backend_node_id"])
    bottom = max([VIEWPORT[1]] + [int(e.abs_box[1] + e.abs_box[3]) for e in elements])
    page = Page(f"m2w-{action['action_uid']}", f"mind2web://{row['website']}", row["website"], row["confirmed_task"][:80],
                list(VIEWPORT), [VIEWPORT[0], bottom], elements)
    history = row["action_reprs"][max(0, index - HISTORY):index]
    example = Example(page.page_id, "goal", "goal", row["confirmed_task"], history, target,
                      sorted(pos_ids))
    return page, example


def convert_row(row: dict[str, Any]) -> Iterator[tuple[Page, Example]]:
    for i in range(len(row["actions"])):
        if (pair := convert_action(row, i)) is not None:
            yield pair


def stream_json_list(f: BinaryIO) -> Iterator[dict[str, Any]]:
    """Rows of a top-level JSON list, parsed incrementally."""
    return ijson.items(f, "item", use_float=True)


def read_rows(path: Path) -> Iterator[dict[str, Any]]:
    """Rows from a JSON list file, a JSONL file, or a directory of such files."""
    if path.is_dir():
        for child in sorted(path.glob("*.json*")):
            yield from read_rows(child)
    elif path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            yield from (json.loads(line) for line in f if line.strip())
    else:
        with path.open("rb") as f:
            yield from stream_json_list(f)


def list_train_files(hub: str = HUB) -> list[tuple[str, int]]:
    """(repo path, size in bytes) of the raw train files, in name order (train_0, train_1, ..., train_10)."""
    resp = httpx.get(f"{hub}/api/datasets/osunlp/Mind2Web/tree/main/{TRAIN_DIR}", timeout=60)
    resp.raise_for_status()
    files = [(e["path"], int(e["size"])) for e in resp.json() if e["type"] == "file" and e["path"].endswith(".json")]
    return sorted(files, key=lambda f: int(re.search(r"(\d+)\.json$", f[0]).group(1)))  # type: ignore[union-attr]


def download(path: str, size: int, cache: Path, hub: str = HUB) -> Path:
    """Fetch one repo file into `cache`. A complete file is reused; a `.part` file continues with a Range request."""
    dest, part = cache / Path(path).name, cache / (Path(path).name + ".part")
    if dest.exists() and dest.stat().st_size == size:
        return dest
    cache.mkdir(parents=True, exist_ok=True)
    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with httpx.stream("GET", f"{hub}/datasets/osunlp/Mind2Web/resolve/main/{path}", headers=headers,
                      follow_redirects=True, timeout=60) as resp:
        resp.raise_for_status()
        resumed = resp.status_code == 206
        with part.open("ab" if resumed else "wb") as f:  # a plain 200 means the server ignored Range: start over
            for chunk in resp.iter_bytes(CHUNK_BYTES):
                f.write(chunk)
    if part.stat().st_size != size:
        raise RuntimeError(f"{path}: downloaded {part.stat().st_size} of {size} bytes; run again to resume")
    part.rename(dest)
    return dest


class RemoteFile:
    """A forward-only binary file over an HTTP stream, so ijson can read the first rows without a full download."""

    def __init__(self, response: httpx.Response) -> None:
        self._chunks = response.iter_bytes(CHUNK_BYTES)
        self._buffer = b""

    def read(self, n: int = -1) -> bytes:
        while n < 0 or len(self._buffer) < n:
            chunk = next(self._chunks, b"")
            if not chunk:
                break
            self._buffer += chunk
        out, self._buffer = (self._buffer, b"") if n < 0 else (self._buffer[:n], self._buffer[n:])
        return out


def fetch_train_rows(
    cache: Path, limit: int | None = None, only: Sequence[str] = (), hub: str = HUB
) -> Iterator[dict[str, Any]]:
    """Train rows from the raw hub files. With `limit`, stream the first rows without caching."""
    files = [f for f in list_train_files(hub) if not only or Path(f[0]).name in only]
    yielded = 0
    for path, size in files:
        if limit is None:
            with download(path, size, cache, hub).open("rb") as f:
                yield from stream_json_list(f)
            continue
        with httpx.stream("GET", f"{hub}/datasets/osunlp/Mind2Web/resolve/main/{path}", follow_redirects=True, timeout=60) as resp:
            resp.raise_for_status()
            for row in stream_json_list(RemoteFile(resp)):  # type: ignore[arg-type]
                yield row
                yielded += 1
                if yielded >= limit:
                    return


def convert(rows: Iterable[dict[str, Any]], out: Path, split: str) -> dict[str, int]:
    """Write pages and `{split}.jsonl`; returns counts."""
    examples: list[Example] = []
    stats = {"tasks": 0, "steps": 0, "elements": 0, "skipped_steps": 0}
    for row in rows:
        stats["tasks"] += 1
        pairs = list(convert_row(row))
        stats["skipped_steps"] += len(row["actions"]) - len(pairs)
        for page, example in pairs:
            write_page(page, out / "pages" / split)
            examples.append(example)
            stats["steps"] += 1
            stats["elements"] += len(page.elements)
    out.mkdir(parents=True, exist_ok=True)
    write_examples(examples, out / f"{split}.jsonl")
    return stats


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", required=True, help="name used for the output files; 'train' downloads from HF")
    parser.add_argument("--input", help="local JSON/JSONL file or directory (required unless --split train)")
    parser.add_argument("--limit", type=int, help="train: stream only the first N rows, no caching")
    parser.add_argument("--only", nargs="*", default=[], help="train: raw file names to use, e.g. train_10.json")
    parser.add_argument("--cache", default="data/mind2web_raw", help="train: where raw files are downloaded")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.input:
        rows: Iterable[dict[str, Any]] = read_rows(Path(args.input))
    elif args.split == "train":
        rows = fetch_train_rows(Path(args.cache), args.limit, args.only)
    else:
        parser.error("test splits are not on the hub: pass --input")
    stats = convert(rows, Path(args.out), args.split)
    print(json.dumps(stats) + f"  avg elements/step {stats['elements'] / max(stats['steps'], 1):.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
