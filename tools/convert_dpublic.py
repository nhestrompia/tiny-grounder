"""Convert the D-public desktop trajectory set into the frozen next-step format.

    uv run python tools/convert_dpublic.py \
      --train-apps /path/to/data/leakage/train_apps.txt

Source: `ash256/look-before-you-click` (MIT), the public Ubuntu OSWorld/OS-Harm trajectory dataset of
"Look Before You Click: Sparse Critical-Action Review for Computer-Use Agent Control" (Chew, Wang and Akunuri,
2026). Only `better_log.json` files are downloaded: JSON containing the goal, the pre-action linearized AT-SPI tree
and the model's PyAutoGUI code. Screenshots, videos and every other artifact are left on the Hub. Embedded action
code is parsed with `ast` and a small arithmetic evaluator; it is never executed.

The converter uses the environmental cohorts (`final_env_*` and `final_a2_env*`) at a pinned revision. The scheming
cohorts are excluded because their actions can pursue a hidden side task rather than the stated user goal. There is no
upstream test split; the pinned environmental sample has 559 source steps, below the 2,000-step cap. Steps in the
`chrome` and `vs_code` domains are dropped as training-app overlap (`google-chrome`/`chromium-browser` and
`code`/`vscode`); `thunderbird` and `libreoffice_writer` remain.

Output is the custom-set layout

    d_public|data/nextstep/d_public/examples.jsonl|data/nextstep/d_public/pages

with the pre-registered tags embedded in `Example.tags`, plus `../../d_public.manifest.json` with SHA-256 hashes
of every frozen file. Raw downloads are cached under `data/d_public_raw/` and are not part of the frozen set.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from grounder.data.pages import write_examples, write_page
from grounder.item_tags import TargetLabel, item_tags
from grounder.records import MAX_CONTEXT_CHARS, MAX_TEXT_CHARS, Element, Example, Page, action_repr, clean_text, equiv_ids, validate_element
from tools.ax_tree import MAX_ELEMENTS, element_at, normalised_box

REPO = "ash256/look-before-you-click"
REVISION = "0663309bd34404d6202f9e1433f191f17d99072f"
API_URL = "https://huggingface.co/api/datasets/{repo}/revision/{revision}"
RESOLVE_URL = "https://huggingface.co/datasets/{repo}/resolve/{revision}/"
ENV_COHORT_RE = re.compile(r"final_(?:env_b\d_(?:honest|attack)|a2_env\d_b\d)$")
USER_AGENT = "tiny-grounder-dpublic-converter/1.0"

CLICK_CALLS = frozenset({"click", "doubleClick", "rightClick", "tripleClick", "middleClick"})
WRITE_CALLS = frozenset({"write", "typewrite"})
CONTROL_CALLS = frozenset({"press", "hotkey", "keyDown", "keyUp"})

# Linearized AT-SPI tag -> the shared record role set. Static labels, sections, paragraphs, headings and images are
# context only. `document-web` is normally web content, not a desktop control; the Thunderbird compose body is the
# one editable web document this dataset relies on.
ROLE_MAP: dict[str, str] = {
    "push-button": "button", "toggle-button": "switch", "radio-button": "radio", "check-box": "checkbox",
    "link": "link", "entry": "textbox", "password-text": "textbox", "text": "textbox",
    "combo-box": "combobox", "spin-button": "textbox", "slider": "slider", "scroll-bar": "slider",
    "menu": "menuitem", "menu-item": "menuitem", "check-menu-item": "menuitem", "radio-menu-item": "menuitem",
    "page-tab": "tab", "tree-item": "option", "list-item": "option", "terminal": "textbox",
    "document-text": "textbox",
}
STATIC_TAGS = frozenset({"label", "static", "section", "paragraph", "heading"})
APP_ALIASES: dict[str, set[str]] = {
    "chrome": {"chrome", "google-chrome", "chromium-browser", "chromium"},
    "vs_code": {"vs-code", "vscode", "code", "visual-studio-code"},
    "libreoffice_writer": {"libreoffice-writer"},
    "thunderbird": {"thunderbird", "thunderbird-mail"},
}
APP_TITLES = {"thunderbird": "Thunderbird", "libreoffice_writer": "LibreOffice Writer"}
ROW_END_RE = re.compile(r"\t\((-?\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?)\)\t\((\d+(?:\.\d+)?), (\d+(?:\.\d+)?)\)$")


# ---------------------------------------------------------------- linearized AT-SPI trees


@dataclass
class TreeRow:
    tag: str
    name: str
    text: str
    class_name: str
    description: str
    rect: list[float]

    @property
    def accessibility_label(self) -> str:
        return self.name or self.description

    @property
    def label(self) -> str:
        return self.name or self.description or self.text


@dataclass
class ParsedTree:
    rows: list[TreeRow]
    malformed_rows: int


@dataclass
class ConvertedScreen:
    page: Page
    row_by_element: dict[int, TreeRow]


def unquote_text(text: str) -> str:
    """Undo OSWorld's CSV-style quoting of the linearized tree's text column."""
    if text == '""':
        return ""
    if len(text) >= 2 and text[0] == text[-1] == '"':
        return text[1:-1].replace('""', '"')
    return text


def parse_a11y_tsv(text: str) -> ParsedTree:
    """Parse OSWorld's linearized AT-SPI table.

    The format is tab-separated except that element text may contain literal newlines and tabs. A row is complete
    only when its final position and size columns arrive, so multiline text is accumulated until that terminator.
    Extra tabs inside text are folded back into the text column; rows too short even then are counted malformed.
    """
    rows: list[TreeRow] = []
    malformed = 0
    buf: str | None = None
    for line in text.splitlines()[1:]:
        end = ROW_END_RE.search(line)
        # The final line of a multiline row often contains only its structural class/description/position
        # columns. Joining that line with "\n" would invent a trailing newline in the element text.
        structural_end = bool(end and re.match(r"^\t+\(", line))
        buf = line if buf is None else buf + ("" if structural_end else "\n") + line
        if not end:
            continue
        body = buf[: -len(end.group(0))]
        parts = body.split("\t")
        if len(parts) >= 5:
            x, y, w, h = (float(v) for v in end.groups())
            rows.append(TreeRow(
                tag=parts[0], name=parts[1], text=unquote_text("\t".join(parts[2:-2])),
                class_name=parts[-2], description=parts[-1], rect=[x, y, w, h],
            ))
        else:
            malformed += 1
        buf = None
    if buf is not None:
        malformed += 1
    return ParsedTree(rows, malformed)


def map_role(tag: str, name: str = "") -> str | None:
    if tag == "document-web":
        return "textbox" if "message body" in name.casefold() else None
    return ROLE_MAP.get(tag)


def row_text(row: TreeRow) -> str:
    return clean_text(row.label, MAX_TEXT_CHARS)


def spatial_context(row: TreeRow, rows: list[TreeRow], title: str) -> str:
    """A small spatial context: the application title plus up to two nearby visible static strings.

    The linearized tree has no parent pointers, so exact axdump group/sibling context cannot be reconstructed.
    This uses overlap or an 80-pixel gap as a conservative approximation and removes duplicates of the target label.
    """
    x0, y0, w, h = row.rect
    x1, y1 = x0 + w, y0 + h
    own = row_text(row)
    scored: list[tuple[float, int, str]] = []
    for index, other in enumerate(rows):
        if other is row or other.tag not in STATIC_TAGS:
            continue
        text = clean_text(other.text or other.name, MAX_TEXT_CHARS)
        if not text or text == own:
            continue
        ox0, oy0, ow, oh = other.rect
        ox1, oy1 = ox0 + ow, oy0 + oh
        dx = max(x0 - ox1, ox0 - x1, 0.0)
        dy = max(y0 - oy1, oy0 - y1, 0.0)
        distance = dx + dy
        if distance <= 80.0:
            scored.append((distance, index, text))
    pieces = [clean_text(title, MAX_CONTEXT_CHARS)]
    for _, _, text in sorted(scored)[:2]:
        piece = clean_text(text, 60)
        if piece and piece not in pieces:
            pieces.append(piece)
    return " · ".join(p for p in pieces if p)[:MAX_CONTEXT_CHARS]


def flatten_atspi(text: str, page_id: str, site: str) -> tuple[ConvertedScreen, int]:
    """Convert one linearized Ubuntu AT-SPI tree to a Page and a row index for every kept element."""
    parsed = parse_a11y_tsv(text)
    valid = [r for r in parsed.rows if r.rect[2] >= 1 and r.rect[3] >= 1]
    width = max((int(r.rect[0] + r.rect[2] + 0.999) for r in valid), default=1)
    height = max((int(r.rect[1] + r.rect[3] + 0.999) for r in valid), default=1)
    title = APP_TITLES.get(site, site.replace("_", " "))
    elements: list[Element] = []
    row_by_element: dict[int, TreeRow] = {}
    for row in valid:
        if len(elements) >= MAX_ELEMENTS:
            break
        role = map_role(row.tag, row.name)
        if role is None:
            continue
        element_id = len(elements)
        row_by_element[element_id] = row
        elements.append(validate_element({
            "id": element_id, "role": role, "text": row_text(row),
            "context": spatial_context(row, valid, title), "tag": row.tag,
            "box": normalised_box(row.rect, [0.0, 0.0, float(width), float(height)]),
            "abs_box": row.rect, "in_viewport": True, "enabled": True, "source": "ax",
        }))
    page = Page(page_id=page_id, url=f"atspi://look-before-you-click/{site}", site=site, title=title,
                viewport=[width, height], doc_size=[width, height], elements=elements)
    return ConvertedScreen(page, row_by_element), parsed.malformed_rows


def page_id_for_tree(text: str) -> str:
    return "dpublic-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


# ---------------------------------------------------------------- PyAutoGUI parsing (never execution)


@dataclass
class GuiCall:
    name: str
    args: list[Any]
    kwargs: dict[str, Any]


@dataclass
class ParsedActions:
    calls: list[GuiCall]
    syntax_error: bool = False


@dataclass
class TargetAction:
    kind: str
    x: float
    y: float
    typed_text: str | None


def numeric_value(node: ast.AST, variables: dict[str, Any]) -> float | None:
    """Evaluate only numeric constants and + - * / arithmetic; names resolve solely to simple assignments."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.Name) and isinstance(variables.get(node.id), (int, float)):
        return float(variables[node.id])
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = numeric_value(node.operand, variables)
        return value if value is not None and isinstance(node.op, ast.UAdd) else (-value if value is not None else None)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        left, right = numeric_value(node.left, variables), numeric_value(node.right, variables)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        return left / right if right else None
    return None


def literal_value(node: ast.AST, variables: dict[str, Any]) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        pass
    if isinstance(node, ast.Name) and node.id in variables:
        return variables[node.id]
    return numeric_value(node, variables)


def parse_action_code(code: str) -> ParsedActions:
    """AST-parse one untrusted PyAutoGUI snippet and extract literal call arguments in source order."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ParsedActions([], True)
    variables: dict[str, Any] = {}
    assignments = sorted((n for n in ast.walk(tree) if isinstance(n, ast.Assign)), key=lambda n: (n.lineno, n.col_offset))
    for node in assignments:
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            variables[node.targets[0].id] = literal_value(node.value, variables)
    call_nodes = sorted((
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Name) and n.func.value.id == "pyautogui"
    ), key=lambda n: (n.lineno, n.col_offset))
    calls = []
    for node in call_nodes:
        assert isinstance(node.func, ast.Attribute)
        calls.append(GuiCall(
            node.func.attr,
            [literal_value(arg, variables) for arg in node.args],
            {kw.arg: literal_value(kw.value, variables) for kw in node.keywords if kw.arg},
        ))
    return ParsedActions(calls)


def coordinate(call: GuiCall) -> tuple[float, float] | None:
    if len(call.args) >= 2 and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in call.args[:2]):
        return float(call.args[0]), float(call.args[1])
    x, y = call.kwargs.get("x"), call.kwargs.get("y")
    if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool) and not isinstance(y, bool):
        return float(x), float(y)
    return None


def typed_after(calls: list[GuiCall], start: int) -> str | None:
    for call in calls[start:]:
        if call.name in WRITE_CALLS:
            return call.args[0] if call.args and isinstance(call.args[0], str) else ""
        if call.name in CLICK_CALLS or call.name in CONTROL_CALLS:
            return None
    return None


def first_target_action(calls: list[GuiCall]) -> tuple[TargetAction | None, str | None]:
    """The element-targeted first action, including the no-op moveTo(x,y)+click() idiom."""
    if not calls:
        return None, "no_pyautogui_action"
    first = calls[0]
    if first.name in CLICK_CALLS:
        point = coordinate(first)
        if point is None:
            return None, "coordinate_unrecoverable"
        return TargetAction(first.name, *point, typed_after(calls, 1)), None
    if first.name == "moveTo" and len(calls) > 1 and calls[1].name == "click":
        point = coordinate(calls[1]) or coordinate(first)
        if point is None:
            return None, "coordinate_unrecoverable"
        return TargetAction("click", *point, typed_after(calls, 2)), None
    return None, "first_action_not_element"


def key_text(values: Iterable[Any]) -> str:
    return "+".join(str(v) for v in values if isinstance(v, (str, int, float)))


def fallback_history(calls: list[GuiCall]) -> str:
    """A compact history entry for an action whose element cannot be labelled."""
    if not calls:
        return ""
    first = calls[0]
    if first.name in CLICK_CALLS or first.name == "moveTo":
        return "[unknown]  -> CLICK"
    if first.name in WRITE_CALLS:
        text = first.args[0] if first.args and isinstance(first.args[0], str) else ""
        return "[unknown]  -> TYPE: " + clean_text(text, 100)
    if first.name == "hotkey":
        return "[keyboard]  -> HOTKEY: " + key_text(first.args)
    if first.name == "press":
        return "[keyboard]  -> PRESS: " + key_text(first.args[:1])
    if first.name in {"scroll", "hscroll"}:
        return "[mouse]  -> SCROLL"
    return f"[{first.name}]  -> ACTION"


# ---------------------------------------------------------------- conversion


@dataclass
class ConversionResult:
    pages: dict[str, Page] = field(default_factory=dict)
    examples: list[Example] = field(default_factory=list)
    dropped: Counter[str] = field(default_factory=Counter)
    source_steps: int = 0
    trajectories: int = 0
    apps: set[str] = field(default_factory=set)
    malformed_rows: int = 0

    @property
    def tag_counts(self) -> Counter[str]:
        return Counter(tag for example in self.examples for tag in example.tags)


def normalise_app(name: str) -> str:
    return re.sub(r"[\s_]+", "-", name.strip().casefold())


def app_names(domain: str) -> set[str]:
    return {normalise_app(domain), *APP_ALIASES.get(domain, set())}


def is_train_app(domain: str, train_apps: set[str]) -> bool:
    return bool(app_names(domain) & train_apps)


def log_parts(path: str) -> tuple[str, str, str]:
    parts = path.split("/")
    if len(parts) < 7:
        raise ValueError(f"unexpected trajectory path {path!r}")
    return parts[0], parts[4], parts[5]


def environmental_logs(siblings: Iterable[dict[str, Any]]) -> list[str]:
    """Canonical final environmental-cohort logs; `results/` copies and rerun/smoke diagnostics are excluded."""
    out = []
    for sibling in siblings:
        path = sibling.get("rfilename", "")
        if path.endswith("/better_log.json") and ENV_COHORT_RE.fullmatch(path.split("/", 1)[0]):
            out.append(path)
    return sorted(out)


def convert_step(screen: ConvertedScreen, target: TargetAction) -> tuple[int | None, TreeRow | None]:
    element_id = element_at(screen.page.elements, (target.x, target.y))
    return (element_id, screen.row_by_element.get(element_id)) if element_id is not None else (None, None)


def convert_trajectory(path: str, data: dict[str, Any], result: ConversionResult, screens: dict[str, ConvertedScreen],
                       malformed: Counter[str], seen: set[tuple[Any, ...]]) -> None:
    cohort, domain, task_id = log_parts(path)
    goal = clean_text((data.get("task") or {}).get("instruction"), 300)
    history: list[str] = []
    result.trajectories += 1
    result.apps.add(domain)
    if not goal:
        result.dropped["missing_goal"] += len(data.get("steps") or [])
        return
    for step_index, step in enumerate(data.get("steps") or []):
        result.source_steps += 1
        code = "\n".join(step.get("actions") or [])
        parsed = parse_action_code(code) if code.strip() else ParsedActions([])
        target, reason = first_target_action(parsed.calls)
        tree_text = step.get("a11y_tree") or ""
        element = None
        row = None
        if target is not None:
            if not tree_text.strip():
                reason = "tree_empty"
            else:
                page_id = page_id_for_tree(tree_text)
                if page_id not in screens:
                    screen, bad_rows = flatten_atspi(tree_text, page_id, domain)
                    screens[page_id] = screen
                    malformed["rows"] += bad_rows
                screen = screens[page_id]
                if not screen.page.elements:
                    reason = "no_elements"
                else:
                    target_id, target_row = convert_step(screen, target)
                    if target_id is None or target_row is None:
                        reason = "target_not_in_tree"
                    else:
                        element = screen.page.elements[target_id]
                        row = target_row
        if element is None or row is None or target is None:
            result.dropped[reason or "target_unrecoverable"] += 1
            entry = fallback_history(parsed.calls)
        else:
            page = screen.page
            # AT-SPI does not say whether the accessible name is visibly rendered. Follow the axdump
            # convention and treat the name as the visible-label fallback; `description` remains
            # accessibility-only evidence for the icon tag.
            visible = row.name or row.text
            nonvisual = row.description or ("" if visible else row.name)
            tags = item_tags(goal, list(history), TargetLabel(element.text, visible, nonvisual))
            example = Example(page_id=page.page_id, kind="goal", mode="goal", query=goal,
                              history=list(history), target_id=element.id,
                              equiv_ids=equiv_ids(page.elements, element.id), source=task_id, tags=tags)
            key = (example.page_id, example.query, tuple(example.history), example.target_id)
            if key in seen:
                result.dropped["duplicate"] += 1
            else:
                seen.add(key)
                result.pages.setdefault(page.page_id, page)
                result.examples.append(example)
            entry = action_repr(element, target.typed_text)
        if entry:
            history.append(entry)


def convert_logs(paths: list[str], raw_root: Path, train_apps: set[str], max_steps: int = 2000) -> ConversionResult:
    result = ConversionResult()
    screens: dict[str, ConvertedScreen] = {}
    malformed = Counter()
    seen: set[tuple[Any, ...]] = set()
    for path in paths:
        _, domain, _ = log_parts(path)
        data = json.loads((raw_root / path).read_text(encoding="utf-8"))
        step_count = len(data.get("steps") or [])
        if is_train_app(domain, train_apps):
            result.dropped["train_app_step"] += step_count
            continue
        if result.source_steps + step_count > max_steps:
            result.dropped["over_step_budget"] += step_count
            continue
        convert_trajectory(path, data, result, screens, malformed, seen)
    result.malformed_rows = malformed["rows"]
    return result


# ---------------------------------------------------------------- download, freeze and CLI


def urlopen_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def dataset_listing(revision: str) -> tuple[str, list[str]]:
    meta = urlopen_json(API_URL.format(repo=REPO, revision=revision))
    paths = environmental_logs(meta.get("siblings") or [])
    if not paths:
        raise ValueError(f"{REPO}@{revision}: no final environmental better_log.json files")
    return meta.get("sha") or revision, paths


def download_logs(raw_root: Path, revision: str) -> tuple[str, list[str]]:
    """Download only the small better_log JSON files at the pinned revision; existing cache files are reused."""
    revision, paths = dataset_listing(revision)
    raw_root.mkdir(parents=True, exist_ok=True)
    base = RESOLVE_URL.format(repo=REPO, revision=revision)
    for index, path in enumerate(paths, 1):
        dest = raw_root / path
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = base + urllib.parse.quote(path, safe="/")
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_bytes(payload)
        tmp.replace(dest)
        print(f"downloaded {index}/{len(paths)} {path}", file=sys.stderr, flush=True)
    return revision, paths


def load_train_apps(path: Path) -> set[str]:
    return {normalise_app(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_manifest(paths: list[Path], manifest_path: Path, result: ConversionResult, revision: str, max_steps: int) -> dict[str, Any]:
    files = {}
    for path in sorted(paths):
        files[str(path.relative_to(manifest_path.parent))] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    return {
        "dataset": {
            "repo": REPO, "revision": revision, "license": "MIT", "use": "evaluation only",
            "citation": "Chew, Wang and Akunuri 2026, Look Before You Click: Sparse Critical-Action Review for Computer-Use Agent Control",
        },
        "sample": {
            "cohorts": "final environmental cohorts only; results/ duplicates, scheming, rerun and smoke cohorts excluded",
            "max_steps": max_steps, "source_steps": result.source_steps, "trajectories": result.trajectories,
        },
        "counts": {
            "examples": len(result.examples), "pages": len(result.pages), "apps": sorted(result.apps),
            "malformed_tree_rows": result.malformed_rows,
        },
        "dropped": dict(sorted(result.dropped.items())),
        "tags": dict(sorted(result.tag_counts.items())),
        "files": files,
    }


def convert_and_freeze(raw_root: Path, out: Path, manifest_path: Path, train_apps_path: Path,
                       revision: str = REVISION, max_steps: int = 2000, download: bool = True) -> dict[str, Any]:
    if download:
        revision, paths = download_logs(raw_root, revision)
    else:
        _, paths = dataset_listing(revision)
    train_apps = load_train_apps(train_apps_path)
    result = convert_logs(paths, raw_root, train_apps, max_steps)
    if not result.examples:
        raise ValueError("conversion produced no examples")

    out.mkdir(parents=True, exist_ok=True)
    examples_path = out / "examples.jsonl"
    report_path = out / "report.json"
    write_examples(result.examples, examples_path)
    written = [examples_path]
    for page in sorted(result.pages.values(), key=lambda p: p.page_id):
        written.append(write_page(page, out / "pages"))
    report = {
        "dataset": REPO, "revision": revision, "examples": len(result.examples), "pages": len(result.pages),
        "source_steps": result.source_steps, "trajectories": result.trajectories, "apps": sorted(result.apps),
        "dropped": dict(sorted(result.dropped.items())), "tags": dict(sorted(result.tag_counts.items())),
        "malformed_tree_rows": result.malformed_rows,
    }
    report_path.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    written.append(report_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = frozen_manifest(written, manifest_path, result, revision, max_steps)
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return {"report": report, "manifest": manifest,
            "paths": {"examples": str(examples_path), "pages": str(out / "pages"), "manifest": str(manifest_path)}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=Path("data/d_public_raw"))
    parser.add_argument("--out", type=Path, default=Path("data/nextstep/d_public"))
    parser.add_argument("--manifest", type=Path, default=Path("data/nextstep/d_public.manifest.json"))
    parser.add_argument("--train-apps", type=Path, default=Path("data/leakage/train_apps.txt"))
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--max-steps", type=int, default=2000)
    args = parser.parse_args(argv)
    result = convert_and_freeze(args.raw, args.out, args.manifest, args.train_apps, args.revision, args.max_steps)
    print(json.dumps(result["report"], indent=1, sort_keys=True))
    print(f"custom set: d_public|{result['paths']['examples']}|{result['paths']['pages']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
