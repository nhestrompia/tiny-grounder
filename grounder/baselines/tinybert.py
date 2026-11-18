"""TinyBERT-L2 cross-encoder baseline: score (query, "role | text | context") for one element at a time.

Fine-tune `cross-encoder/ms-marco-TinyBERT-L2-v2` with listwise cross-entropy over 1 positive + 7 negatives
per example (hard negatives: same role as the target and high BM25 preferred), then rank every element of a
page with it. `rank(...)` has the same interface as `baselines.bm25.rank`, so eval.py uses it as a baseline:

    uv run python -m grounder.baselines.tinybert --pages P --train train.jsonl --out runs/tinybert --steps 2000
    uv run python -m grounder.eval --pages P --examples dev.jsonl --baseline tinybert \
        --baseline-model runs/tinybert --max-examples 2000

It scores elements one by one, so it is slow: evaluate on a 2k-example subsample (`--max-examples`).
Needs the `baselines` extra (transformers). torch and transformers are imported inside the functions that
use them, so the sampling and string helpers import without them.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np

from grounder.baselines.bm25 import rank_elements
from grounder.data.pages import load_examples, load_pages
from grounder.ranking import Ranked, example_elements, positive_ids
from grounder.records import Element, Example, Page

if TYPE_CHECKING:
    import torch

BASE_MODEL = "cross-encoder/ms-marco-TinyBERT-L2-v2"
N_NEGATIVES = 7
HARD_POOL = 20  # negatives are sampled from the best HARD_POOL by (same role, BM25)
MAX_LENGTH = 128
SCORE_BATCH = 64


def element_string(e: Element) -> str:
    return f"{e.role} | {e.text} | {e.context}"


def query_string(example: Example) -> str:
    """The plain-language query: the instruction or goal, plus the recent actions in goal mode."""
    done = f" | done: {'; '.join(example.history)}" if example.history else ""
    return f"{example.query}{done}"


def hard_negatives(elements: list[Element], example: Example, rng: np.random.Generator) -> list[Element]:
    """`N_NEGATIVES` non-positive elements drawn from the HARD_POOL best by (same role as target, BM25).

    Pages with fewer negatives than that repeat some.
    """
    positives = positive_ids(example)
    target_role = next(e.role for e in elements if e.id == example.target_id)
    negatives = [e for e in elements if e.id not in positives]
    by_id = {e.id: e for e in negatives}
    bm25_order = rank_elements(query_string(example), negatives).ids
    ordered = sorted(bm25_order, key=lambda i: by_id[i].role != target_role)  # stable: BM25 order inside each group
    pool = [by_id[i] for i in ordered[:HARD_POOL]]
    picks = rng.choice(len(pool), N_NEGATIVES, replace=len(pool) < N_NEGATIVES)  # tiny pages repeat negatives
    return [pool[i] for i in picks]


def training_group(page: Page, example: Example, rng: np.random.Generator) -> tuple[str, list[str]] | None:
    """(query, [positive, negatives...]) or None when the example has no positive or no negative."""
    if example.kind == "null" or example.target_id is None:
        return None
    elements = example_elements(page, example)
    if len(elements) < 2:
        return None
    target = next(e for e in elements if e.id == example.target_id)
    negatives = hard_negatives(elements, example, rng)
    return query_string(example), [element_string(target)] + [element_string(e) for e in negatives]


class TinyBertRanker:
    """A fine-tuned cross-encoder. `rank(page, example)` matches the `eval.Ranker` signature."""

    def __init__(self, model_dir: str | Path = BASE_MODEL, device: str = "cpu") -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), num_labels=1).to(device).eval()
        self.device = torch.device(device)

    def logits(self, queries: Sequence[str], passages: Sequence[str]) -> torch.Tensor:
        batch = self.tokenizer(list(queries), list(passages), truncation=True, max_length=MAX_LENGTH,
                               padding=True, return_tensors="pt").to(self.device)
        return self.model(**batch).logits.squeeze(-1)

    def rank(self, page: Page, example: Example) -> Ranked:
        """Elements best-first; scores are a softmax over the page's elements; null_prob is always 0."""
        import torch

        elements = example_elements(page, example)
        if not elements:
            return Ranked([], [], 0.0)
        query = query_string(example)
        with torch.no_grad():
            parts = [
                self.logits([query] * len(chunk), [element_string(e) for e in chunk])
                for chunk in (elements[i : i + SCORE_BATCH] for i in range(0, len(elements), SCORE_BATCH))
            ]
            scores = torch.cat(parts).float().cpu().numpy()
        probs = np.exp(scores - scores.max())
        probs /= probs.sum()
        order = np.argsort(-probs, kind="stable")
        return Ranked([elements[i].id for i in order], [float(probs[i]) for i in order], 0.0)


def fine_tune(
    pages: dict[str, Page], examples: Sequence[Example], out: str | Path, steps: int, batch_size: int = 8,
    lr: float = 3e-5, seed: int = 0, device: str = "cpu", base_model: str = BASE_MODEL,
) -> Path:
    """Listwise fine-tuning: each example's 8 (query, element) pairs are one softmax group, target first."""
    import torch

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    ranker = TinyBertRanker(base_model, device)
    ranker.model.train()
    optimizer = torch.optim.AdamW(ranker.model.parameters(), lr=lr, weight_decay=0.01)
    for step in range(steps):
        groups = [g for i in rng.integers(0, len(examples), batch_size * 2)
                  if (g := training_group(pages[examples[i].page_id], examples[i], rng))][:batch_size]
        queries = [q for q, passages in groups for _ in passages]
        passages = [p for _, group in groups for p in group]
        width = N_NEGATIVES + 1
        logits = ranker.logits(queries, passages).view(len(groups), width)
        loss = torch.nn.functional.cross_entropy(logits, torch.zeros(len(groups), dtype=torch.long, device=ranker.device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(f"step {step} loss {loss.item():.4f}")
    out_dir = Path(out)
    ranker.model.save_pretrained(out_dir)
    ranker.tokenizer.save_pretrained(out_dir)
    return out_dir


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pages", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    out = fine_tune(load_pages(args.pages), load_examples(args.train), args.out, args.steps, args.batch_size,
                    args.lr, device=args.device)
    print(f"saved to {out}")


if __name__ == "__main__":
    _main()
