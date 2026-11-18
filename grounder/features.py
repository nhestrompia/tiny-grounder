"""Featurisation shared by training (collate) and the runtime. Pure Python + numpy, no torch.

`encode_page` is the single place that turns (query, elements) into model inputs; the
training collate only pads and stacks its output.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np
from tokenizers import Tokenizer

from grounder.records import ROLE_TO_ID, Element, normalize_role
from grounder.tokenizer import PAD_ID, encode_ids

N_FEATS = 10
CHUNK = 512  # pages are scored in chunks of at most this many elements (the training cap)
MODEL_INPUTS: tuple[str, ...] = (
    "query_ids", "el_text_ids", "el_ctx_ids", "el_role", "el_feats", "el_mask",
)

_MARKER_RE = re.compile(r"\[(?:DO|GOAL|DONE)\]")
_WORD_RE = re.compile(r"[a-z0-9]+")
_ES_STEMS = ("s", "x", "z", "ch", "sh")
_MIN_STEM = 3

# Function words plus generic UI words that never identify an element ("click the login button").
STOP_WORDS = frozenset(
    "a an the of to in on at for from by with and or but not is are was be do does this that these "
    "those it its my me i you your our we please then so as into onto up out all any can will "
    "click tap press button link icon".split()
)


def stem(word: str) -> str:
    """Light deterministic stemming: strip one of -ing, -ed, -es (after s/x/z/ch/sh), -s.

    A suffix is only removed when at least 3 letters remain; "-ss" words keep their s.
    """
    for suffix in ("ing", "ed"):
        if word.endswith(suffix) and len(word) - len(suffix) >= _MIN_STEM:
            return word[: -len(suffix)]
    if word.endswith("es") and len(word) - 2 >= _MIN_STEM and word[:-2].endswith(_ES_STEMS):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) - 1 >= _MIN_STEM:
        return word[:-1]
    return word


@lru_cache(maxsize=1 << 18)
def content_tokens(text: str) -> frozenset[str]:
    """Lowercase words of `text` without stop words, stemmed (memoised: element texts repeat across examples)."""
    words = _WORD_RE.findall(unicodedata.normalize("NFKC", text).lower())
    return frozenset(stem(w) for w in words if w not in STOP_WORDS)


def query_tokens(query: str) -> frozenset[str]:
    """Content tokens of a formatted query, ignoring the [DO]/[GOAL]/[DONE] markers."""
    return content_tokens(_MARKER_RE.sub(" ", query))


def _share(part: frozenset[str], whole: frozenset[str]) -> float:
    """Fraction of `part` that is in `whole` (0 for an empty `part`)."""
    return len(part & whole) / len(part) if part else 0.0


def overlap(query: str, text: str) -> float:
    """Share of the query's content tokens that also occur in `text`."""
    return _share(query_tokens(query), content_tokens(text))


def _element_features(el: Element, q_tokens: frozenset[str]) -> np.ndarray:
    text_tokens = content_tokens(el.text)
    x, y, w, h = el.box
    return np.array(
        [
            min(max(x, 0.0), 1.0), min(max(y, 0.0), 4.0) / 4.0,
            min(max(w, 0.0), 1.0), min(max(h, 0.0), 1.0),
            float(el.in_viewport), float(el.enabled), el.conf,
            _share(q_tokens, text_tokens), _share(q_tokens, content_tokens(el.context)),
            _share(text_tokens, q_tokens),
        ],
        dtype=np.float32,
    )


def element_features(el: Element, query: str) -> np.ndarray:
    """[x, y/4, w, h, in_viewport, enabled, conf, overlap(query, text), overlap(query, context), coverage].

    coverage is the share of the element's own text tokens found in the query; together with the text
    overlap it separates "subscribe" from "subscribe messages" for the query "open subscribe".
    """
    return _element_features(el, query_tokens(query))


def combine_chunk_logits(chunks: list[np.ndarray]) -> np.ndarray:
    """Merge per-chunk logits [n_c + 1] (index 0 = NULL) into one [N + 1] vector.

    Element logits are concatenated as they are; the NULL logit is the mean of the chunks' NULL logits.
    One softmax over the result then gives probabilities that sum to 1 across the whole page.
    """
    null = np.mean([c[0] for c in chunks])
    return np.concatenate([[null], *[c[1:] for c in chunks]])


def _pad(rows: list[list[int]], width: int) -> np.ndarray:
    out = np.full((len(rows), width), PAD_ID, dtype=np.int64)
    for i, row in enumerate(rows):
        out[i, : len(row)] = row
    return out


@dataclass(frozen=True)
class TokenCache:
    """Token ids of known element texts and contexts, so a training dataset tokenizes each string once.

    Ids are truncated to the lengths the cache was built with; `encode_page` must use the same limits.
    """

    text: dict[str, list[int]]
    context: dict[str, list[int]]

    @classmethod
    def build(cls, tokenizer: Tokenizer, elements: Iterable[Element], max_text: int, max_ctx: int) -> "TokenCache":
        els = list(elements)
        texts = sorted({e.text for e in els} | {""})
        contexts = sorted({e.context for e in els} | {""})
        return cls(dict(zip(texts, encode_ids(tokenizer, texts, max_text))),
                   dict(zip(contexts, encode_ids(tokenizer, contexts, max_ctx))))


def _lookup_ids(tokenizer: Tokenizer, strings: list[str], max_len: int, known: dict[str, list[int]] | None) -> list[list[int]]:
    """Ids of `strings`; strings missing from `known` (e.g. augmented text) are tokenized, not stored."""
    if known is None:
        return encode_ids(tokenizer, strings, max_len)
    missing = sorted({s for s in strings if s not in known})
    fresh = dict(zip(missing, encode_ids(tokenizer, missing, max_len)))
    return [known[s] if s in known else fresh[s] for s in strings]


def encode_page(
    tokenizer: Tokenizer, query: str, elements: list[Element],
    max_q: int = 48, max_text: int = 16, max_ctx: int = 24, cache: TokenCache | None = None,
) -> dict[str, np.ndarray]:
    """Model input arrays for ONE page. `query` is the string built by `records.format_query`.

    With a `cache`, only strings it does not know are tokenized; the result is identical either way.
    """
    q_tokens = query_tokens(query)
    query_ids = encode_ids(tokenizer, [query], max_q)[0] or [PAD_ID]
    feats = [_element_features(el, q_tokens) for el in elements]
    return {
        "query_ids": np.array(query_ids, dtype=np.int64),
        "el_text_ids": _pad(_lookup_ids(tokenizer, [e.text for e in elements], max_text, cache and cache.text), max_text),
        "el_ctx_ids": _pad(_lookup_ids(tokenizer, [e.context for e in elements], max_ctx, cache and cache.context), max_ctx),
        "el_role": np.array([ROLE_TO_ID[normalize_role(e.role)] for e in elements], dtype=np.int64),
        "el_feats": np.stack(feats) if feats else np.zeros((0, N_FEATS), dtype=np.float32),
        "el_mask": np.ones(len(elements), dtype=bool),
    }
