"""Torch dataset, augmentation, collate and size-bucketed batch sampler."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterator, Sequence

import numpy as np
import torch
from tokenizers import Tokenizer
from torch.utils.data import Dataset, Sampler

from grounder.features import MODEL_INPUTS, encode_page
from grounder.ranking import example_elements, example_query, positive_ids
from grounder.records import Element, Example, Page

MAX_ELEMENTS = 512
_OCR_SUBSTITUTIONS = {"0": "o", "o": "0", "l": "1", "1": "l", "i": "l"}


@dataclass(frozen=True)
class AugmentConfig:
    """Probabilities (per example unless noted) and magnitudes of the training augmentations."""

    shuffle: float = 1.0
    drop_role: float = 0.15       # per element: role becomes "unknown"
    drop_context: float = 0.15    # per element: context is cleared
    text_noise: float = 0.1       # per element: OCR-style noise, conf lowered
    ocr_page: float = 0.05        # OCR-like page: unknown roles, no context, noisy low-conf text
    ocr_page_noise: float = 0.5   # per element share of noisy text on an OCR-like page
    box_jitter: float = 0.01      # sigma of the gaussian box jitter (0 disables)
    conf_range: tuple[float, float] = (0.6, 0.9)


def ocr_noise(text: str, rng: np.random.Generator) -> str:
    """One or two OCR-style edits: adjacent swap, dropped char, flipped case, 0/o and l/1 confusions."""
    chars = list(text)
    for _ in range(int(rng.integers(1, 3))):
        op = rng.choice(["swap", "drop", "case", "subst"])
        i = int(rng.integers(0, max(len(chars), 1)))
        if not chars:
            break
        if op == "swap" and len(chars) > 1:
            i = min(i, len(chars) - 2)
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
        elif op == "drop" and len(chars) > 2:
            del chars[i]
        elif op == "case":
            chars[i] = chars[i].swapcase()
        else:
            candidates = [j for j, c in enumerate(chars) if c.lower() in _OCR_SUBSTITUTIONS]
            if candidates:
                j = candidates[int(rng.integers(0, len(candidates)))]
                chars[j] = _OCR_SUBSTITUTIONS[chars[j].lower()]
    return "".join(chars)


def _noisy(el: Element, rng: np.random.Generator, conf_range: tuple[float, float]) -> Element:
    conf = float(rng.uniform(*conf_range))
    return replace(el, text=ocr_noise(el.text, rng), conf=min(el.conf, conf))


def augment_elements(elements: list[Element], rng: np.random.Generator, cfg: AugmentConfig) -> list[Element]:
    """Augment copies of `elements`. Role/context drops and text noise are per element; an OCR-like
    page (see `AugmentConfig.ocr_page`) applies its source conditions to every element consistently."""
    ocr = rng.random() < cfg.ocr_page
    noise_p = cfg.ocr_page_noise if ocr else cfg.text_noise
    out = []
    for el in elements:
        if ocr or rng.random() < cfg.drop_role:
            el = replace(el, role="unknown")
        if ocr or rng.random() < cfg.drop_context:
            el = replace(el, context="")
        if el.text and rng.random() < noise_p:
            el = _noisy(el, rng, cfg.conf_range)
        if cfg.box_jitter > 0:
            jitter = rng.normal(0.0, cfg.box_jitter, size=4)
            box = [float(v) for v in np.asarray(el.box) + jitter]
            el = replace(el, box=[box[0], box[1], max(box[2], 0.0), max(box[3], 0.0)])
        out.append(el)
    if rng.random() < cfg.shuffle:
        out = [out[i] for i in rng.permutation(len(out))]
    return out


def cap_elements(
    elements: list[Element], positives: set[int], rng: np.random.Generator, limit: int = MAX_ELEMENTS
) -> list[Element]:
    """Keep every positive and a random sample of the rest, `limit` elements in total."""
    if len(elements) <= limit:
        return elements
    rest = [i for i, e in enumerate(elements) if e.id not in positives]
    n_keep = max(limit - (len(elements) - len(rest)), 0)
    sampled = set(rng.choice(rest, size=n_keep, replace=False).tolist())
    return [e for i, e in enumerate(elements) if e.id in positives or i in sampled]


@dataclass
class Prepared:
    """A training example ready for featurisation: `positive[0]` is NULL, `positive[i+1]` is elements[i]."""

    query: str
    elements: list[Element]
    positive: np.ndarray


def prepare_example(
    page: Page, example: Example, rng: np.random.Generator, augment: AugmentConfig | None,
    max_elements: int | None = MAX_ELEMENTS,
) -> Prepared:
    """Candidate set, positives and (optionally) augmentation for one example."""
    positives = positive_ids(example)
    elements = example_elements(page, example)
    if example.kind != "null" and not positives & {e.id for e in elements}:
        raise ValueError(f"{page.page_id}: example {example.query!r} has no positive element on the page")
    if max_elements is not None:
        elements = cap_elements(elements, positives, rng, max_elements)
    if augment is not None:
        elements = augment_elements(elements, rng, augment)
    mask = np.zeros(len(elements) + 1, dtype=bool)
    mask[0] = example.kind == "null"
    mask[1:] = [e.id in positives for e in elements]
    return Prepared(example_query(example), elements, mask)


class GrounderDataset(Dataset):
    """One item per example. Randomness is a function of (seed, epoch, index), so runs are reproducible."""

    def __init__(
        self, pages: dict[str, Page], examples: Sequence[Example], tokenizer: Tokenizer,
        augment: AugmentConfig | None = None, seed: int = 0,
        max_q: int = 48, max_text: int = 16, max_ctx: int = 24,
    ) -> None:
        self.pages, self.examples, self.tokenizer = pages, list(examples), tokenizer
        self.augment, self.seed = augment, seed
        self.limits = {"max_q": max_q, "max_text": max_text, "max_ctx": max_ctx}
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def sizes(self) -> list[int]:
        """Approximate number of elements of each example (for bucketing)."""
        return [min(len(self.pages[ex.page_id].elements), MAX_ELEMENTS) for ex in self.examples]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        ex = self.examples[index]
        rng = np.random.default_rng((self.seed, self.epoch, index))
        prep = prepare_example(self.pages[ex.page_id], ex, rng, self.augment)
        item = encode_page(self.tokenizer, prep.query, prep.elements, **self.limits)
        item["positive"] = prep.positive
        return item


def collate(batch: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
    """Pad a list of per-page arrays to the batch max query length and element count.

    Items from `GrounderDataset` carry a `positive` mask that is padded too; eval items do not."""
    b = len(batch)
    lq = max(len(item["query_ids"]) for item in batch)
    n = max(len(item["el_role"]) for item in batch)
    first = batch[0]
    out = {
        "query_ids": np.zeros((b, lq), dtype=np.int64),
        "el_text_ids": np.zeros((b, n, first["el_text_ids"].shape[1]), dtype=np.int64),
        "el_ctx_ids": np.zeros((b, n, first["el_ctx_ids"].shape[1]), dtype=np.int64),
        "el_role": np.zeros((b, n), dtype=np.int64),
        "el_feats": np.zeros((b, n, first["el_feats"].shape[1]), dtype=np.float32),
        "el_mask": np.zeros((b, n), dtype=bool),
    }
    if "positive" in first:
        out["positive"] = np.zeros((b, n + 1), dtype=bool)
    for i, item in enumerate(batch):
        out["query_ids"][i, : len(item["query_ids"])] = item["query_ids"]
        k = len(item["el_role"])
        for name in MODEL_INPUTS[1:]:
            out[name][i, :k] = item[name]
        if "positive" in item:
            out["positive"][i, : k + 1] = item["positive"]
    return {name: torch.from_numpy(arr) for name, arr in out.items()}


class BucketBatchSampler(Sampler[list[int]]):
    """Batches of examples with similar page size. Shuffled: sort inside random mega-chunks, shuffle batches."""

    def __init__(
        self, sizes: Sequence[int], batch_size: int, shuffle: bool, seed: int = 0, chunk_batches: int = 50
    ) -> None:
        self.sizes, self.batch_size, self.shuffle = np.asarray(sizes), batch_size, shuffle
        self.seed, self.chunk = seed, batch_size * chunk_batches
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        n = len(self.sizes)
        if not self.shuffle:
            return math.ceil(n / self.batch_size)
        full, rest = divmod(n, self.chunk)
        return full * math.ceil(self.chunk / self.batch_size) + math.ceil(rest / self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng((self.seed, self.epoch))
        order = rng.permutation(len(self.sizes)) if self.shuffle else np.arange(len(self.sizes))
        chunk = self.chunk if self.shuffle else len(order)
        batches: list[list[int]] = []
        for start in range(0, len(order), chunk):
            part = order[start : start + chunk]
            part = part[np.argsort(self.sizes[part], kind="stable")]
            batches += [part[i : i + self.batch_size].tolist() for i in range(0, len(part), self.batch_size)]
        if self.shuffle:
            batches = [batches[i] for i in rng.permutation(len(batches))]
        return iter(batches)
