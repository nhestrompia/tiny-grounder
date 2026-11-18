"""8k BPE tokenizer (Hugging Face `tokenizers`). No torch: the runtime imports this module."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Iterator

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers

from grounder.data.pages import iter_pages, load_examples
from grounder.ranking import example_query
from grounder.records import Example, Page

SPECIAL_TOKENS: tuple[str, ...] = ("[PAD]", "[UNK]", "[DO]", "[GOAL]", "[DONE]")
PAD_ID = 0
UNK_ID = 1
_BASE_ALPHABET = [chr(c) for c in range(32, 127)]


def train_tokenizer(texts: Iterable[str], vocab_size: int, out_path: str | Path) -> Tokenizer:
    """Train a lowercase BPE tokenizer on `texts` and save it to `out_path`."""
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase()])
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=_BASE_ALPHABET, min_frequency=2, show_progress=False,
    )
    tok.train_from_iterator(texts, trainer)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_path))
    return tok


def load_tokenizer(path: str | Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def encode_ids(tok: Tokenizer, texts: list[str], max_len: int) -> list[list[int]]:
    """Token ids for each text, truncated to `max_len` (not padded)."""
    if not texts:
        return []
    return [enc.ids[:max_len] for enc in tok.encode_batch(texts)]


def tokenizer_stats(tok: Tokenizer, texts: list[str]) -> dict[str, float]:
    """Average tokens per text and share of [UNK] among all produced tokens."""
    ids = [enc.ids for enc in tok.encode_batch(texts)] if texts else []
    total = sum(len(i) for i in ids)
    unk = sum(i.count(UNK_ID) for i in ids)
    return {
        "texts": len(texts),
        "avg_tokens": total / max(len(texts), 1),
        "unk_rate": unk / max(total, 1),
    }


def page_texts(pages: Iterable[Page]) -> Iterator[str]:
    """Element texts and contexts (non-empty) of the given pages."""
    for page in pages:
        for el in page.elements:
            yield from (t for t in (el.text, el.context) if t)


def example_queries(examples: Iterable[Example]) -> Iterator[str]:
    for ex in examples:
        yield example_query(ex)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    train = sub.add_parser("train", help="train a tokenizer on page texts and example queries")
    train.add_argument("--pages", required=True)
    train.add_argument("--examples", nargs="*", default=[])
    train.add_argument("--vocab-size", type=int, default=8192)
    train.add_argument("--out", required=True)
    stats = sub.add_parser("stats", help="avg tokens per element text and UNK rate")
    stats.add_argument("--tokenizer", required=True)
    stats.add_argument("--pages", required=True)
    args = parser.parse_args()

    if args.cmd == "train":
        texts = list(page_texts(iter_pages(args.pages)))
        for path in args.examples:
            texts.extend(example_queries(load_examples(path)))
        tok = train_tokenizer(texts, args.vocab_size, args.out)
        print(f"trained vocab={tok.get_vocab_size()} on {len(texts)} texts -> {args.out}")
    else:
        tok = load_tokenizer(args.tokenizer)
        texts = [el.text for page in iter_pages(args.pages) for el in page.elements if el.text]
        s = tokenizer_stats(tok, texts)
        print(f"element texts: {s['texts']}  avg tokens: {s['avg_tokens']:.2f}  unk rate: {s['unk_rate']:.4%}")


if __name__ == "__main__":
    _main()
