"""Evaluation: top-k / MRR per example kind, abstain precision/recall for null examples."""

from __future__ import annotations

import argparse
import json
from typing import Callable, Sequence

import numpy as np
import torch
from tokenizers import Tokenizer

from grounder.baselines import bm25
from grounder.data.dataset import collate
from grounder.data.pages import load_examples, load_pages
from grounder.features import CHUNK, MODEL_INPUTS, combine_chunk_logits, encode_page
from grounder.model import Grounder, load_checkpoint
from grounder.ranking import Ranked, example_elements, example_query, positive_ids, ranked_from_logits
from grounder.records import Element, Example, Page

KS = (1, 5, 10, 20)
NULL_THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
Ranker = Callable[[Page, Example], Ranked]


def _forward(model: Grounder, batch: dict[str, torch.Tensor]) -> np.ndarray:
    device = next(model.parameters()).device
    return model(*(batch[k].to(device) for k in MODEL_INPUTS)).cpu().numpy()


@torch.no_grad()
def rank_model(
    model: Grounder, tokenizer: Tokenizer, pages: dict[str, Page], examples: Sequence[Example],
    batch_size: int = 32,
) -> list[Ranked]:
    """Rank every example with the model, batching pages of similar size (same collate as training).

    Pages over `CHUNK` elements go through one at a time, in chunks, exactly like the runtime.
    """
    model.eval()
    cfg = model.cfg
    limits = {"max_q": cfg.max_q, "max_text": cfg.max_text, "max_ctx": cfg.max_ctx}
    queries = [example_query(ex) for ex in examples]
    elements = [example_elements(pages[ex.page_id], ex) for ex in examples]
    results: list[Ranked | None] = [None] * len(examples)

    def encode(i: int, els: list[Element]) -> dict[str, np.ndarray]:
        return encode_page(tokenizer, queries[i], els, **limits)

    small = sorted((i for i, els in enumerate(elements) if len(els) <= CHUNK), key=lambda i: len(elements[i]))
    for start in range(0, len(small), batch_size):
        idx = small[start : start + batch_size]
        logits = _forward(model, collate([encode(i, elements[i]) for i in idx]))
        for row, i in zip(logits, idx):
            results[i] = ranked_from_logits([e.id for e in elements[i]], row[: len(elements[i]) + 1])
    for i, els in enumerate(elements):
        if len(els) > CHUNK:
            chunks = [_forward(model, collate([encode(i, els[j : j + CHUNK])]))[0] for j in range(0, len(els), CHUNK)]
            results[i] = ranked_from_logits([e.id for e in els], combine_chunk_logits(chunks))
    return results  # type: ignore[return-value]


def rank(model: Grounder, tokenizer: Tokenizer, page: Page, example: Example) -> Ranked:
    """Rank one example with the model."""
    return rank_model(model, tokenizer, {page.page_id: page}, [example])[0]


def rank_with(ranker: Ranker, pages: dict[str, Page], examples: Sequence[Example]) -> list[Ranked]:
    """Rank every example with a per-example ranker such as the BM25 baseline."""
    return [ranker(pages[ex.page_id], ex) for ex in examples]


def _first_hit(ranked: Ranked, positives: set[int]) -> int | None:
    """1-based rank of the best-ranked positive, or None."""
    return next((i + 1 for i, eid in enumerate(ranked.ids) if eid in positives), None)


def _rank_metrics(ranks: list[int | None]) -> dict[str, float]:
    n = max(len(ranks), 1)
    out: dict[str, float] = {"n": len(ranks)}
    for k in KS:
        out[f"top{k}"] = sum(r is not None and r <= k for r in ranks) / n
    out["mrr"] = sum(1.0 / r for r in ranks if r is not None) / n
    return out


def _null_metrics(rows: list[tuple[bool, float]]) -> dict[str, dict[str, float | None]]:
    """rows: (example is null, null_prob). Abstain when null_prob >= threshold."""
    n_null = sum(is_null for is_null, _ in rows)
    out: dict[str, dict[str, float | None]] = {}
    for t in NULL_THRESHOLDS:
        abstain = [is_null for is_null, p in rows if p >= t]
        tp = sum(abstain)
        precision = tp / len(abstain) if abstain else None
        recall = tp / n_null
        f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0
        out[f"{t:g}"] = {"precision": precision, "recall": recall, "f1": f1, "abstained": len(abstain)}
    return out


def evaluate(examples: Sequence[Example], rankings: Sequence[Ranked]) -> dict[str, dict]:
    """Metrics per kind (null examples excluded from top-k), "overall" over all non-null, and "null"."""
    ranks_by_kind: dict[str, list[int | None]] = {}
    null_rows: list[tuple[bool, float]] = []
    for ex, ranked in zip(examples, rankings, strict=True):
        null_rows.append((ex.kind == "null", ranked.null_prob))
        if ex.kind != "null":
            ranks_by_kind.setdefault(ex.kind, []).append(_first_hit(ranked, positive_ids(ex)))
    result: dict[str, dict] = {"by_kind": {k: _rank_metrics(v) for k, v in sorted(ranks_by_kind.items())}}
    if ranks_by_kind:
        result["overall"] = _rank_metrics([r for v in ranks_by_kind.values() for r in v])
    if any(is_null for is_null, _ in null_rows):
        result["null"] = _null_metrics(null_rows)
    return result


def select_score(metrics: dict[str, dict]) -> float:
    """Checkpoint selection: top-1 on `paraphrase`, else overall top-1."""
    by_kind = metrics["by_kind"]
    return by_kind["paraphrase"]["top1"] if "paraphrase" in by_kind else metrics["overall"]["top1"]


def _fmt(v: float | None) -> str:
    return "-" if v is None else f"{v:.3f}"


def to_markdown(metrics: dict[str, dict], title: str) -> str:
    lines = [f"### {title}", "", "| kind | n | " + " | ".join(f"top-{k}" for k in KS) + " | MRR |",
             "|---|---|" + "---|" * (len(KS) + 1)]
    rows = dict(metrics["by_kind"])
    if "overall" in metrics:
        rows["overall"] = metrics["overall"]
    for kind, m in rows.items():
        lines.append(f"| {kind} | {int(m['n'])} | " + " | ".join(_fmt(m[f"top{k}"]) for k in KS) + f" | {_fmt(m['mrr'])} |")
    if "null" in metrics:
        lines += ["", "| null_prob threshold | abstain precision | abstain recall | F1 | abstained |", "|---|---|---|---|---|"]
        for t, m in metrics["null"].items():
            lines.append(f"| {t} | {_fmt(m['precision'])} | {_fmt(m['recall'])} | {_fmt(m['f1'])} | {m['abstained']} |")
    return "\n".join(lines)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint or the BM25 baseline.")
    parser.add_argument("--pages", required=True)
    parser.add_argument("--examples", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--baseline", choices=["bm25"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--out", help="write metrics as JSON")
    args = parser.parse_args()
    if bool(args.checkpoint) == bool(args.baseline):
        parser.error("give exactly one of --checkpoint or --baseline")

    pages = load_pages(args.pages)
    examples = load_examples(args.examples)[: args.max_examples]
    if args.checkpoint:
        model, tokenizer, _ = load_checkpoint(args.checkpoint, args.device)
        rankings = rank_model(model, tokenizer, pages, examples)
        title = f"{args.checkpoint} ({model.count_params():,} params)"
    else:
        rankings, title = rank_with(bm25.rank, pages, examples), "BM25"
    metrics = evaluate(examples, rankings)
    print(to_markdown(metrics, title))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    _main()
