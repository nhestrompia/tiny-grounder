"""Dev-split keyword and heuristic baselines for the decision-label datasets.

This script intentionally reads only `dev.jsonl` files. The test split is frozen for later model evaluation.

    uv run python tools/decision_baselines.py --decisions data/decisions
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grounder.records import Example
from grounder.data.pages import load_page
from grounder_tool.rankers import make_ranker
from tools.next_step_recall import hit_at_distinct

DECISIONS = ("show", "next", "enough", "worked", "done")
VISUAL_PATTERNS = (
    "icon", "magnifier", "logo", "image", "picture", "visual", "screenshot", "top bar", "bottom",
    "far right", "small menu", "button in the header", "footer",
)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for", "from", "how", "i", "in", "is",
    "it", "me", "of", "on", "open", "or", "the", "to", "use", "using", "what", "when", "where", "which", "with",
    "site", "page", "docs", "documentation", "tell", "find", "according", "there",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def load_pages(decisions: Path, rows: list[dict[str, Any]]):
    wanted = {row["page_id"] for row in rows}
    pages = {}
    for path in (decisions / "pages").glob("*/*.json.gz"):
        pid = path.name.removesuffix(".json.gz")
        if pid in wanted:
            pages[pid] = load_page(path)
    return pages


def mean(values: Iterable[float | bool]) -> float:
    vals = list(values)
    return sum(bool(v) for v in vals) / len(vals) if vals else 0.0


def binary_metrics(y: list[bool], pred: list[bool], scores: list[float] | None = None) -> dict[str, Any]:
    tp = sum(a and b for a, b in zip(y, pred))
    fp = sum((not a) and b for a, b in zip(y, pred))
    fn = sum(a and not b for a, b in zip(y, pred))
    tn = sum((not a) and (not b) for a, b in zip(y, pred))
    out: dict[str, Any] = {
        "n": len(y), "base_rate": mean(y), "accuracy": (tp + tn) / len(y) if y else 0.0,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
    if scores is not None and any(y) and not all(y):
        out["auroc"] = auroc(y, scores)
    else:
        out["auroc"] = None
    return out


def auroc(y: list[bool], scores: list[float]) -> float:
    """Mann-Whitney AUROC with average ranks for ties."""
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    pos = sum(y)
    neg = len(y) - pos
    pos_rank_sum = sum(rank for rank, label in zip(ranks, y) if label)
    return (pos_rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def target_rank_baselines(rows: list[dict[str, Any]], pages: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for name in ("exact", "bm25"):
        ranker = make_ranker(name)
        recall3, top1 = [], []
        for row in rows:
            page = pages.get(row["page_id"])
            if page is None:
                continue
            example = Example(
                row["page_id"], "goal", "goal", row["goal"], row["history"],
                row["label"]["target_id"], row["label"]["target_ids"], row["provenance"]["task_id"],
            )
            ranked = ranker.rank(row["goal"] or page.title, page.elements, "goal", row["history"])
            recall3.append(hit_at_distinct(ranked, example, page.elements, 3))
            top1.append(hit_at_distinct(ranked, example, page.elements, 1))
        result[name] = {"n": len(recall3), "recall@3": mean(recall3), "top1": mean(top1)}
    return result


def action_type_baselines(rows: list[dict[str, Any]], pages: dict[str, Any]) -> dict[str, Any]:
    y = [row["label"]["action_type"] for row in rows]
    majority = Counter(y).most_common(1)[0][0] if y else "click"
    majority_pred = [majority] * len(y)
    rule_pred = []
    for row in rows:
        page = pages.get(row["page_id"])
        target = None
        if page is not None:
            target = next((e for e in page.elements if e.id == row["label"]["target_id"]), None)
        rule_pred.append("type" if target is not None and target.role in {"textbox", "searchbox"} else "click")
    return {
        "action_base_rates": dict(Counter(y)),
        "majority": {"class": majority, "accuracy": mean(a == b for a, b in zip(y, majority_pred))},
        "textbox_type_rule": {"accuracy": mean(a == b for a, b in zip(y, rule_pred))},
    }


def visual_score(row: dict[str, Any], page: Any) -> float:
    goal = row["goal"].casefold()
    phrase_hit = any(pattern in goal for pattern in VISUAL_PATTERNS)
    interactive = [e for e in page.elements if e.role not in {"text", "image"}] or page.elements
    empty_ratio = sum(not e.text.strip() for e in interactive) / len(interactive) if interactive else 0.0
    return min(1.0, (0.7 if phrase_hit else 0.0) + 0.3 * empty_ratio)


def tokens(text: str) -> set[str]:
    return {
        tok for tok in re.findall(r"[a-z0-9]+", text.casefold())
        if len(tok) >= 3 and tok not in STOPWORDS
    }


def answer_overlap_score(row: dict[str, Any], page: Any) -> float:
    goal_tokens = tokens(row["goal"])
    if not goal_tokens:
        return 0.0
    lines = [page.title]
    for element in page.elements:
        lines.extend([element.text, element.context])
    best = 0.0
    for line in lines:
        line_tokens = tokens(line)
        if not line_tokens:
            continue
        best = max(best, len(goal_tokens & line_tokens) / len(goal_tokens))
    return best


def binary_baseline_report(rows: list[dict[str, Any]], pages: dict[str, Any], decision: str) -> dict[str, Any]:
    y = [bool(row["label"]) for row in rows]
    majority_label = mean(y) >= 0.5
    majority_pred = [majority_label] * len(y)
    scores: list[float] = []
    if decision == "enough":
        for row in rows:
            page = pages.get(row["page_id"])
            scores.append(visual_score(row, page) if page is not None else 0.0)
        threshold = 0.5
        heuristic = "visual_terms_and_empty_text_ratio"
    elif decision == "worked":
        scores = [1.0 if row["features"]["url_changed_after_previous_action"] else 0.0 for row in rows]
        threshold = 0.5
        heuristic = "url_changed_after_action"
    elif decision == "done":
        for row in rows:
            page = pages.get(row["page_id"])
            scores.append(answer_overlap_score(row, page) if page is not None else 0.0)
        threshold = 0.5
        heuristic = "goal_line_token_overlap"
    else:
        raise ValueError(decision)
    pred = [score >= threshold for score in scores]
    report = {
        "majority": binary_metrics(y, majority_pred),
        heuristic: {
            **binary_metrics(y, pred, scores),
            "threshold": threshold,
            "score_min": min(scores) if scores else 0.0,
            "score_max": max(scores) if scores else 0.0,
        },
    }
    if decision == "enough":
        arm_scores = [1.0 if row["features"]["arm"] == "vision" else 0.0 for row in rows]
        arm_pred = [bool(score) for score in arm_scores]
        report["vision_arm"] = binary_metrics(y, arm_pred, arm_scores)
    return report


def evaluate(dev_decisions: Path) -> dict[str, Any]:
    report: dict[str, Any] = {"split": "dev", "decisions": {}}
    for decision in DECISIONS:
        rows = read_jsonl(dev_decisions / decision / "dev.jsonl")
        pages = load_pages(dev_decisions, rows)
        entry: dict[str, Any] = {"n": len(rows), "pages": len(pages)}
        if decision in {"show", "next"}:
            entry["target_ranking"] = target_rank_baselines(rows, pages)
        if decision == "next":
            entry["action_type"] = action_type_baselines(rows, pages)
        if decision in {"enough", "worked", "done"}:
            entry["base_rate"] = mean([bool(row["label"]) for row in rows])
            entry["binary"] = binary_baseline_report(rows, pages, decision)
        report["decisions"][decision] = entry
    return report


def markdown_tables(report: dict[str, Any]) -> str:
    lines = [
        "| decision | baseline | metric | value |",
        "|---|---|---:|---:|",
    ]
    for decision, entry in report["decisions"].items():
        if "target_ranking" in entry:
            for ranker, row in entry["target_ranking"].items():
                lines.append(f"| {decision.upper()} | {ranker} | recall@3 | {row['recall@3']:.3f} |")
                lines.append(f"| {decision.upper()} | {ranker} | top-1 | {row['top1']:.3f} |")
        if decision == "next":
            action = entry["action_type"]
            lines.append(f"| NEXT | majority action ({action['majority']['class']}) | accuracy | {action['majority']['accuracy']:.3f} |")
            lines.append(f"| NEXT | textbox → type rule | accuracy | {action['textbox_type_rule']['accuracy']:.3f} |")
        if "binary" in entry:
            lines.append(f"| {decision.upper()} | base rate | positive share | {entry['base_rate']:.3f} |")
            for name, row in entry["binary"].items():
                auroc_text = "n/a" if row.get("auroc") is None else f"{row['auroc']:.3f}"
                precision = "n/a" if row.get("precision") is None else f"{row['precision']:.3f}"
                recall = "n/a" if row.get("recall") is None else f"{row['recall']:.3f}"
                lines.append(f"| {decision.upper()} | {name} | precision / recall / AUROC | {precision} / {recall} / {auroc_text} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=Path, default=ROOT / "data/decisions")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    report = evaluate(args.decisions)
    out = args.out or args.decisions / "dev-baselines.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (args.decisions / "dev-baselines.md").write_text(markdown_tables(report))
    print(markdown_tables(report))


if __name__ == "__main__":
    main()
