"""Torch dataset, augmentation, collate and size-bucketed batch sampler."""

from __future__ import annotations

import math
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from tokenizers import Tokenizer
from torch.utils.data import Dataset, Sampler

from grounder.features import MODEL_INPUTS, TokenCache, encode_page
from grounder.ranking import example_elements, example_query, positive_ids
from grounder.records import ROLE_TO_ID, ROLES, Element, Example, Page, normalize_role
from grounder.tokenizer import PAD_ID, encode_ids

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


def _noisy(el: Element, rng: np.random.Generator, conf_range: tuple[float, float]) -> dict[str, Any]:
    """Field changes for OCR-style noise on `el`: edited text and a lowered confidence."""
    conf = float(rng.uniform(*conf_range))
    return {"text": ocr_noise(el.text, rng), "conf": min(el.conf, conf)}


def augment_elements(elements: list[Element], rng: np.random.Generator, cfg: AugmentConfig) -> list[Element]:
    """Augment copies of `elements`. Role/context drops and text noise are per element; an OCR-like
    page (see `AugmentConfig.ocr_page`) applies its source conditions to every element consistently."""
    ocr = rng.random() < cfg.ocr_page
    noise_p = cfg.ocr_page_noise if ocr else cfg.text_noise
    out = []
    for el in elements:
        changes: dict[str, Any] = {}  # one `replace` per element: dataclass copies are the costly part
        if ocr or rng.random() < cfg.drop_role:
            changes["role"] = "unknown"
        if ocr or rng.random() < cfg.drop_context:
            changes["context"] = ""
        if el.text and rng.random() < noise_p:
            changes |= _noisy(el, rng, cfg.conf_range)
        if cfg.box_jitter > 0:
            jitter = rng.normal(0.0, cfg.box_jitter, size=4)
            box = [float(v) for v in np.asarray(el.box) + jitter]
            changes["box"] = [box[0], box[1], max(box[2], 0.0), max(box[3], 0.0)]
        out.append(replace(el, **changes) if changes else el)
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


class CompactPages:
    """Columnar, packed store of the pages a training set uses, holding only the fields training reads.

    All pages share a few big numpy arrays (one row per element) plus two de-duplicated string tables
    (texts, contexts) packed as one UTF-8 blob each, so pickling to a worker is cheap and the object count
    is tiny. Token ids of every unique string are precomputed. `get` rebuilds one `Page` (and the matching
    `TokenCache`) lazily. `tag`, `source`, `href`, `abs_box` and the page metadata are never model inputs
    and are not kept.
    """

    def __init__(
        self, pages: dict[str, Page], page_ids: Sequence[str], tokenizer: Tokenizer, max_text: int, max_ctx: int,
    ) -> None:
        self.page_ids = list(dict.fromkeys(page_ids))
        self.index = {pid: i for i, pid in enumerate(self.page_ids)}
        counts = [len(pages[pid].elements) for pid in self.page_ids]
        self.offsets = np.zeros(len(counts) + 1, dtype=np.int64)
        np.cumsum(counts, out=self.offsets[1:])
        n = int(self.offsets[-1])
        self.ids = np.empty(n, dtype=np.int32)
        self.role = np.empty(n, dtype=np.uint8)
        self.box = np.empty((n, 4), dtype=np.float64)  # float64: augmentation arithmetic stays bit-identical
        self.conf = np.empty(n, dtype=np.float64)
        self.flags = np.empty(n, dtype=np.uint8)  # bit 0: in_viewport, bit 1: enabled
        self.text_idx = np.empty(n, dtype=np.int32)
        self.ctx_idx = np.empty(n, dtype=np.int32)
        texts: dict[str, int] = {"": 0}
        contexts: dict[str, int] = {"": 0}
        for pid, start in zip(self.page_ids, self.offsets[:-1].tolist()):
            els = pages[pid].elements
            sl = slice(start, start + len(els))
            self.ids[sl] = [e.id for e in els]
            self.role[sl] = [ROLE_TO_ID[normalize_role(e.role)] for e in els]
            if els:
                self.box[sl] = [e.box for e in els]
            self.conf[sl] = [e.conf for e in els]
            self.flags[sl] = [int(e.in_viewport) | int(e.enabled) << 1 for e in els]
            self.text_idx[sl] = [texts.setdefault(e.text, len(texts)) for e in els]
            self.ctx_idx[sl] = [contexts.setdefault(e.context, len(contexts)) for e in els]
        self.text_blob, self.text_off = self._pack(texts)
        self.ctx_blob, self.ctx_off = self._pack(contexts)
        self.text_tok, self.text_len = self._tokenize(tokenizer, texts, max_text)
        self.ctx_tok, self.ctx_len = self._tokenize(tokenizer, contexts, max_ctx)

    @staticmethod
    def _pack(table: dict[str, int]) -> tuple[bytes, np.ndarray]:
        encoded = [s.encode("utf-8") for s in table]  # dicts keep insertion order = index order
        off = np.zeros(len(encoded) + 1, dtype=np.int64)
        np.cumsum([len(b) for b in encoded], out=off[1:])
        return b"".join(encoded), off

    @staticmethod
    def _tokenize(tokenizer: Tokenizer, table: dict[str, int], max_len: int) -> tuple[np.ndarray, np.ndarray]:
        ids = encode_ids(tokenizer, list(table), max_len)
        dtype = np.int16 if tokenizer.get_vocab_size() < 2**15 else np.int32
        tok = np.full((len(ids), max(max_len, 1)), PAD_ID, dtype=dtype)
        lens = np.zeros(len(ids), dtype=np.int32)
        for i, row in enumerate(ids):
            tok[i, : len(row)] = row
            lens[i] = len(row)
        return tok, lens

    _ARRAYS = ("offsets", "ids", "role", "box", "conf", "flags", "text_idx", "ctx_idx", "text_off", "ctx_off",
               "text_tok", "text_len", "ctx_tok", "ctx_len")
    _BLOBS = ("text_blob", "ctx_blob")
    _mmap_dir: str | None = None

    def share(self, directory: str | Path) -> None:
        """Move the big arrays to `directory` (as .npy files) and memory-map them from there.

        Pickling then carries only the directory name; every DataLoader worker maps the same files, so the
        OS page cache is shared instead of each worker holding a private copy. The directory must outlive
        every process that uses this store."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name in self._BLOBS:
            setattr(self, name, np.frombuffer(getattr(self, name), dtype=np.uint8))
        for name in self._ARRAYS + self._BLOBS:
            np.save(directory / f"{name}.npy", getattr(self, name))
        self._mmap_dir = str(directory)
        self._map()

    def _map(self) -> None:
        for name in self._ARRAYS + self._BLOBS:
            setattr(self, name, np.load(Path(self._mmap_dir) / f"{name}.npy", mmap_mode="r"))

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        if self._mmap_dir is not None:
            for name in self._ARRAYS + self._BLOBS:
                del state[name]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        if self._mmap_dir is not None:
            self._map()

    def _string(self, blob: np.ndarray | bytes, off: np.ndarray, j: int) -> str:
        return bytes(blob[off[j] : off[j + 1]]).decode("utf-8")

    def __contains__(self, page_id: str) -> bool:
        return page_id in self.index

    def n_elements(self, page_id: str) -> int:
        i = self.index[page_id]
        return int(self.offsets[i + 1] - self.offsets[i])

    def get(self, page_id: str) -> tuple[Page, TokenCache]:
        """The page's elements (training fields only) and a TokenCache for exactly their strings."""
        i = self.index[page_id]
        a, b = int(self.offsets[i]), int(self.offsets[i + 1])
        t_idx, c_idx = self.text_idx[a:b].tolist(), self.ctx_idx[a:b].tolist()
        texts = {j: self._string(self.text_blob, self.text_off, j) for j in set(t_idx)}
        ctxs = {j: self._string(self.ctx_blob, self.ctx_off, j) for j in set(c_idx)}
        flags = self.flags[a:b].tolist()
        els = [
            Element(id=eid, role=ROLES[role], text=texts[t], context=ctxs[c], box=box, in_viewport=bool(f & 1),
                    enabled=bool(f & 2), conf=conf)
            for eid, role, t, c, box, f, conf in zip(
                self.ids[a:b].tolist(), self.role[a:b].tolist(), t_idx, c_idx, self.box[a:b].tolist(), flags,
                self.conf[a:b].tolist())
        ]
        cache = TokenCache(
            {texts[j]: self.text_tok[j, : self.text_len[j]].tolist() for j in texts},
            {ctxs[j]: self.ctx_tok[j, : self.ctx_len[j]].tolist() for j in ctxs},
        )
        return Page(page_id, "", "", "", [], [], els), cache


class GrounderDataset(Dataset):
    """One item per example. Randomness is a function of (seed, epoch, index), so runs are reproducible.

    `pages` may be the full page dict or a prebuilt `CompactPages`; only the pages the examples use are kept,
    in compact form, so the dataset stays small when it is pickled to DataLoader workers."""

    def __init__(
        self, pages: dict[str, Page] | CompactPages, examples: Sequence[Example], tokenizer: Tokenizer,
        augment: AugmentConfig | None = None, seed: int = 0,
        max_q: int = 48, max_text: int = 16, max_ctx: int = 24,
    ) -> None:
        self.examples, self.tokenizer = list(examples), tokenizer
        self.augment, self.seed = augment, seed
        self.limits = {"max_q": max_q, "max_text": max_text, "max_ctx": max_ctx}
        if isinstance(pages, CompactPages):
            self.pages = pages
        else:
            self.pages = CompactPages(pages, sorted({ex.page_id for ex in self.examples}), tokenizer, max_text, max_ctx)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def sizes(self) -> list[int]:
        """Approximate number of elements of each example (for bucketing)."""
        return [min(self.pages.n_elements(ex.page_id), MAX_ELEMENTS) for ex in self.examples]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        ex = self.examples[index]
        rng = np.random.default_rng((self.seed, self.epoch, index))
        page, cache = self.pages.get(ex.page_id)
        prep = prepare_example(page, ex, rng, self.augment)
        item = encode_page(self.tokenizer, prep.query, prep.elements, **self.limits, cache=cache)
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
    """Batches of examples with similar page size. Shuffled: sort inside random mega-chunks, shuffle batches.

    With `max_batch_elements`, a batch also stops growing once batch length x largest page size would exceed it,
    so batches of very large pages are smaller (attention memory grows with that product).
    """

    def __init__(
        self, sizes: Sequence[int], batch_size: int, shuffle: bool, seed: int = 0, chunk_batches: int = 50,
        max_batch_elements: int | None = None,
    ) -> None:
        self.sizes, self.batch_size, self.shuffle = np.asarray(sizes), batch_size, shuffle
        self.seed, self.chunk = seed, batch_size * chunk_batches
        self.max_batch_elements = max_batch_elements
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self._batches())

    def _split(self, part: np.ndarray) -> list[list[int]]:
        """Consecutive batches of the size-sorted `part`."""
        if self.max_batch_elements is None:
            return [part[i : i + self.batch_size].tolist() for i in range(0, len(part), self.batch_size)]
        batches: list[list[int]] = []
        current: list[int] = []
        for idx in part.tolist():
            size = max(int(self.sizes[idx]), 1)
            if current and (len(current) >= self.batch_size or (len(current) + 1) * size > self.max_batch_elements):
                batches.append(current)
                current = []
            current.append(idx)
        return batches + ([current] if current else [])

    def _batches(self) -> list[list[int]]:
        rng = np.random.default_rng((self.seed, self.epoch))
        order = rng.permutation(len(self.sizes)) if self.shuffle else np.arange(len(self.sizes))
        chunk = self.chunk if self.shuffle else len(order)
        batches: list[list[int]] = []
        for start in range(0, len(order), chunk):
            part = order[start : start + chunk]
            batches += self._split(part[np.argsort(self.sizes[part], kind="stable")])
        if self.shuffle:
            batches = [batches[i] for i in rng.permutation(len(batches))]
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self._batches())
