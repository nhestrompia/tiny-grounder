"""Ranking result type and the example -> candidate-set rules shared by training, eval and baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from grounder.records import Element, Example, Page, format_query


@dataclass
class Ranked:
    """Element ids best-first with their scores, and the probability that nothing matches."""

    ids: list[int]
    scores: list[float]
    null_prob: float


def removed_ids(example: Example) -> set[int]:
    """For `null` examples: the target and its equivalents are deleted from the page."""
    if example.kind != "null":
        return set()
    return set(example.equiv_ids) | ({example.target_id} if example.target_id is not None else set())


def positive_ids(example: Example) -> set[int]:
    if example.kind == "null":
        return set()
    return set(example.equiv_ids) | ({example.target_id} if example.target_id is not None else set())


def example_elements(page: Page, example: Example) -> list[Element]:
    """The page's candidate elements for this example (null examples lose their target)."""
    gone = removed_ids(example)
    return [e for e in page.elements if e.id not in gone]


def example_query(example: Example) -> str:
    return format_query(example.mode, example.query, example.history)  # type: ignore[arg-type]


def ranked_from_logits(element_ids: list[int], logits: np.ndarray) -> Ranked:
    """Softmax over [NULL ; elements] logits, elements sorted best-first (ties keep page order)."""
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    order = np.argsort(-probs[1:], kind="stable")
    return Ranked([element_ids[i] for i in order], [float(probs[1 + i]) for i in order], float(probs[0]))
