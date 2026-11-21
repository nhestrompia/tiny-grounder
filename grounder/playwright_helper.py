"""Playwright helper: find, click and type into UI elements by instruction, with click safety.

Sync functions (`extract`, `find`, `click`, `type_text`) take a sync Playwright page; the `a`-prefixed
async twins (`aextract`, `afind`, `aclick`, `atype_text`) take an async page. Both share the pure logic.

    from grounder.playwright_helper import find, click, load_threshold
    from grounder.runtime import Grounder
    grounder = Grounder("dist/m")
    result = click(page, grounder, "open the cart", load_threshold("dist/m"))

Click safety (docs/plan-review.md item 9). `click` only clicks when the top score reaches the threshold AND
the action is not risky. Risky means one of:
  * the element's text or context contains a blocked word (BLOCKED_WORDS, whole word, any case);
  * the element at the click point is a submit control: `input[type=submit|image]`, or a `<button>` that
    submits a form (checked in the page at click time, because the extractor records the tag but not the
    button's type or form).
A refused click returns the candidates and a reason so the caller can decide.

Why coordinates instead of a locator: extractor records carry no selector, and shadow roots / same-origin
iframes make selectors fragile. The click point is the element's centre, the same point the extractor
hit-tested. Before clicking, an in-page probe scrolls that point into the viewport, resolves the element
there (through shadow roots and same-origin iframes), and refuses when the element no longer matches the
record's tag (the page changed since extraction) or lies under an overlay in an inner scroller.

Thresholds are tuned on a dev set for a target precision and stored beside the model:

    uv run python -m grounder.playwright_helper tune --model-dir dist/m --pages data/pages/dev \
        --examples dev.jsonl --precision 0.98        # writes dist/m/thresholds.json
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from grounder.data.pages import load_examples, load_pages
from grounder.extract.web import extract_page, extract_page_sync
from grounder.ranking import example_elements, positive_ids
from grounder.records import Element, Mode
from grounder.runtime import Grounder

# Headings include content below the fold. Hidden/mobile copies are excluded.
LANDMARKS_JS = r"""() => {
  const norm = s => (s || "").replace(/\s+/g, " ").trim();
  const clip = s => s.length > 60 ? s.slice(0, 59) + "…" : s;
  const landmarks = [], headings = [], seen = new Set();
  const selector = 'h1,h2,h3,nav,main,aside,footer,[role=navigation],[role=main],'
    + '[role=complementary],[role=contentinfo]';
  for (const el of document.querySelectorAll(selector)) {
    const style = getComputedStyle(el), box = el.getBoundingClientRect();
    if (!el.getClientRects().length || !box.width || !box.height || style.visibility !== 'visible'
        || el.closest('[hidden],[aria-hidden="true"]') || style.opacity === '0') continue;
    const heading = /^H[123]$/.test(el.tagName);
    const labelledby = norm(el.getAttribute('aria-labelledby')).split(' ')
      .map(id => document.getElementById(id)?.textContent || '').join(' ');
    const name = norm(labelledby) || norm(el.getAttribute('aria-label'));
    // Docs sites append permalink anchors ("Install¶", "Install #"); drop them.
    const text = clip(heading ? norm(el.innerText).replace(/\s*[¶#§🔗]+$/u, "") : name);
    if (!text) continue;
    if (heading) headings.push([text, box.top + scrollY]);
    const key = text.toLocaleLowerCase();
    if (!seen.has(key) && landmarks.length < 30) {
      seen.add(key);
      landmarks.push(text);
    }
  }
  return {landmarks, headings};
}"""


def extract_landmarks(page: Any) -> tuple[list[str], list[tuple[str, float]]]:
    """Return visible landmark names and heading document positions from one live page."""
    result = page.evaluate(LANDMARKS_JS)
    return result["landmarks"], [(text, float(y)) for text, y in result["headings"]]


BLOCKED_WORDS: tuple[str, ...] = (
    "submit", "pay", "buy", "order", "purchase", "checkout", "delete", "remove", "send", "confirm",
    "transfer", "place order",
)
_BLOCKED_RE = re.compile(r"\b(?:" + "|".join(re.escape(w) for w in BLOCKED_WORDS) + r")\b", re.IGNORECASE)
TYPEABLE_ROLES = ("textbox", "searchbox", "combobox")  # comboboxes: search fields with suggestions (MDN, GitHub)
THRESHOLDS_FILE = "thresholds.json"
DEFAULT_PRECISION = 0.98

# Runs in the page: bring the document point (cx, cy) into the viewport, find the element there and
# report whether clicking it is safe. `tag` is the extractor's tag for the element we mean to hit.
PROBE_JS = """([cx, cy, tag]) => {
  const jump = (left, top) => scrollTo({ left: Math.max(0, left), top: Math.max(0, top), behavior: "instant" });
  if (cy < scrollY || cy >= scrollY + innerHeight) jump(scrollX, cy - innerHeight / 2);
  if (cx < scrollX || cx >= scrollX + innerWidth) jump(cx - innerWidth / 2, scrollY);
  const x = cx - scrollX, y = cy - scrollY;
  const up = (n) => n.parentElement || (n.getRootNode() && n.getRootNode().host) || null;
  const closest = (n, sel) => { for (; n; n = up(n)) if (n.matches && n.matches(sel)) return n; return null; };
  let hit = document.elementFromPoint(x, y), ox = 0, oy = 0;
  while (hit) {
    let inner = null;
    if (hit.shadowRoot) inner = hit.shadowRoot.elementFromPoint(x - ox, y - oy);
    else if (hit.contentDocument) {
      const r = hit.getBoundingClientRect();
      ox += r.left + hit.clientLeft; oy += r.top + hit.clientTop;
      inner = hit.contentDocument.elementFromPoint(x - ox, y - oy);
    }
    if (!inner || inner === hit) break;
    hit = inner;
  }
  if (!hit) return { ok: false, reason: "the element is not reachable at its position", x, y };
  if (!closest(hit, tag)) return { ok: false, reason: `the element at that point is not a <${tag}>: the page changed`, x, y };
  const control = closest(hit, "button,input");
  const submit = !!control && (
    control.matches("input[type=submit],input[type=image]") ||
    (control.tagName === "BUTTON" && !!control.form && control.type === "submit"));
  const field = closest(hit, "input,textarea");
  return { ok: true, reason: "", x, y, submit, password: !!field && field.type === "password" };
}"""


@dataclass
class FindResult:
    """Top candidates (dicts: id, role, text, score, abs_box, centre) and P(no element matches)."""

    candidates: list[dict[str, Any]]
    null_prob: float


@dataclass
class ClickResult:
    """Outcome of `click` / `type_text`: `done` is True only when the action was performed."""

    done: bool
    reason: str
    candidates: list[dict[str, Any]] = field(default_factory=list)
    null_prob: float = 0.0
    target: dict[str, Any] | None = None


def blocked_text(text: str) -> str | None:
    """The first blocked word found in `text`, else None."""
    match = _BLOCKED_RE.search(text)
    return match.group(0).lower() if match else None


def blocked_word(element: Element) -> str | None:
    """The first blocked word found in the element's text or context, else None."""
    return blocked_text(f"{element.text} | {element.context}")


def centre(box: Sequence[float]) -> list[float]:
    return [round(box[0] + box[2] / 2, 1), round(box[1] + box[3] / 2, 1)]


def _find_result(
    grounder: Grounder, elements: list[Element], query: str, k: int, mode: Mode, history: list[str] | None
) -> FindResult:
    ranked = grounder.rank(query, elements, k=k, mode=mode, history=history)
    candidates = [{**r, "centre": centre(r["abs_box"])} for r in ranked["results"]]
    return FindResult(candidates, ranked["null_prob"])


def _screen(
    found: FindResult, elements: list[Element], threshold: float, roles: tuple[str, ...] | None,
) -> tuple[Element | None, str]:
    """Pre-browser checks on the top candidate: threshold, role, enabled, blocked words."""
    if not found.candidates:
        return None, "no elements on the page"
    top = found.candidates[0]
    if top["score"] < threshold:
        return None, f"top score {top['score']:.3f} is below the threshold {threshold:.3f}"
    element = next(e for e in elements if e.id == top["id"])
    if roles is not None and element.role not in roles:
        return None, f"the top element is a {element.role}, not one of {', '.join(roles)}"
    if not element.enabled:
        return None, "the top element is disabled"
    if roles is None and (word := blocked_word(element)):
        return None, f"risky: text or context contains the blocked word {word!r}"
    return element, ""


def verdict(probe: dict[str, Any], click_action: bool) -> str:
    """Reason to refuse after the in-page probe, or '' when the action may go ahead."""
    if not probe["ok"]:
        return probe["reason"]
    if click_action and probe["submit"]:
        return "risky: the element submits a form"
    if not click_action and probe["password"]:
        return "the element is a password field"
    return ""


def _outcome(done: bool, reason: str, found: FindResult, target: dict[str, Any] | None) -> ClickResult:
    return ClickResult(done, reason, found.candidates, found.null_prob, target)


# ---- sync ---------------------------------------------------------------------------------------------


def extract(page) -> list[Element]:
    return extract_page_sync(page).elements


def find(
    page, grounder: Grounder, query: str, k: int = 5, mode: Mode = "do", history: list[str] | None = None
) -> FindResult:
    return _find_result(grounder, extract(page), query, k, mode, history)


def _act(page, grounder: Grounder, instruction: str, threshold: float, roles: tuple[str, ...] | None, text: str | None) -> ClickResult:
    elements = extract(page)
    found = _find_result(grounder, elements, instruction, 5, "do", None)
    element, reason = _screen(found, elements, threshold, roles)
    if element is None:
        return _outcome(False, reason, found, None)
    target = found.candidates[0]
    cx, cy = target["centre"]
    probe = page.evaluate(PROBE_JS, [cx, cy, element.tag])
    if reason := verdict(probe, click_action=text is None):
        return _outcome(False, reason, found, target)
    page.mouse.click(probe["x"], probe["y"])
    if text is not None:
        page.keyboard.type(text)
    return _outcome(True, "", found, target)


def click(page, grounder: Grounder, instruction: str, threshold: float) -> ClickResult:
    """Click the top element for `instruction` if it clears `threshold` and is not risky."""
    return _act(page, grounder, instruction, threshold, None, None)


def type_text(page, grounder: Grounder, instruction: str, text: str, threshold: float) -> ClickResult:
    """Focus the top textbox for `instruction` by clicking it, then type `text` (Enter is never pressed).

    Blocked words do not apply: typing cannot trigger the action the words name. Password fields are refused.
    """
    return _act(page, grounder, instruction, threshold, TYPEABLE_ROLES, text)


# ---- async --------------------------------------------------------------------------------------------


async def aextract(page) -> list[Element]:
    return (await extract_page(page)).elements


async def afind(
    page, grounder: Grounder, query: str, k: int = 5, mode: Mode = "do", history: list[str] | None = None
) -> FindResult:
    return _find_result(grounder, await aextract(page), query, k, mode, history)


async def _aact(
    page, grounder: Grounder, instruction: str, threshold: float, roles: tuple[str, ...] | None, text: str | None
) -> ClickResult:
    elements = await aextract(page)
    found = _find_result(grounder, elements, instruction, 5, "do", None)
    element, reason = _screen(found, elements, threshold, roles)
    if element is None:
        return _outcome(False, reason, found, None)
    target = found.candidates[0]
    cx, cy = target["centre"]
    probe = await page.evaluate(PROBE_JS, [cx, cy, element.tag])
    if reason := verdict(probe, click_action=text is None):
        return _outcome(False, reason, found, target)
    await page.mouse.click(probe["x"], probe["y"])
    if text is not None:
        await page.keyboard.type(text)
    return _outcome(True, "", found, target)


async def aclick(page, grounder: Grounder, instruction: str, threshold: float) -> ClickResult:
    return await _aact(page, grounder, instruction, threshold, None, None)


async def atype_text(page, grounder: Grounder, instruction: str, text: str, threshold: float) -> ClickResult:
    return await _aact(page, grounder, instruction, threshold, TYPEABLE_ROLES, text)


# ---- thresholds ---------------------------------------------------------------------------------------


def tune_threshold(
    scores: Sequence[float], correct: Sequence[bool], target_precision: float = DEFAULT_PRECISION
) -> float | None:
    """Lowest threshold t such that, among predictions with score >= t, the share that is correct is
    at least `target_precision`. Candidates are the observed scores. None when no threshold gets there."""
    if len(scores) != len(correct):
        raise ValueError("scores and correct must have the same length")
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    s = np.asarray(scores, dtype=float)[order]
    hits = np.cumsum(np.asarray(correct, dtype=bool)[order])
    precision = hits / np.arange(1, len(s) + 1)
    # A threshold admits every prediction tied with it, so only the last index of each score group counts.
    group_end = np.append(s[1:] != s[:-1], True) if len(s) else np.array([], dtype=bool)
    ok = np.flatnonzero((precision >= target_precision) & group_end)
    return float(s[ok[-1]]) if len(ok) else None


def load_threshold(model_dir: str | Path) -> float:
    """The click threshold stored by `tune` in `{model_dir}/thresholds.json`."""
    return float(json.loads((Path(model_dir) / THRESHOLDS_FILE).read_text())["click"])


def dev_scores(grounder: Grounder, pages_dir: str | Path, examples_path: str | Path, limit: int | None) -> tuple[list[float], list[bool]]:
    """Top-1 score and whether the top-1 element is a positive, for every dev example.

    `null` examples count as incorrect: clicking anything on a page where the target is missing is wrong.
    """
    pages = load_pages(pages_dir)
    scores: list[float] = []
    correct: list[bool] = []
    for ex in load_examples(examples_path)[:limit]:
        els = example_elements(pages[ex.page_id], ex)
        top = grounder.rank(ex.query, els, k=1, mode=ex.mode, history=ex.history)["results"]  # type: ignore[arg-type]
        if top:
            scores.append(top[0]["score"])
            correct.append(top[0]["id"] in positive_ids(ex))
    return scores, correct


def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    tune = sub.add_parser("tune", help="pick the click threshold on a dev set and write thresholds.json")
    tune.add_argument("--model-dir", required=True)
    tune.add_argument("--pages", required=True)
    tune.add_argument("--examples", required=True)
    tune.add_argument("--precision", type=float, default=DEFAULT_PRECISION)
    tune.add_argument("--max-examples", type=int)
    args = parser.parse_args(argv)

    scores, correct = dev_scores(Grounder(args.model_dir), args.pages, args.examples, args.max_examples)
    threshold = tune_threshold(scores, correct, args.precision)
    if threshold is None:
        raise SystemExit(f"no threshold reaches precision {args.precision} on {len(scores)} examples")
    kept = [c for s, c in zip(scores, correct) if s >= threshold]
    report = {"click": threshold, "target_precision": args.precision, "examples": len(scores),
              "answered": len(kept), "precision": sum(kept) / len(kept)}
    (Path(args.model_dir) / THRESHOLDS_FILE).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    _main()
