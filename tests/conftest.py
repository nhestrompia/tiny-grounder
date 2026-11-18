from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from grounder.model import GrounderConfig
from grounder.records import Example, Page
from grounder.tokenizer import train_tokenizer
from synth import make_dataset, texts


@pytest.fixture(scope="session")
def synth_data() -> tuple[dict[str, Page], list[Example]]:
    pages, examples = make_dataset(n_pages=60, seed=1)
    return {p.page_id: p for p in pages}, examples


@pytest.fixture(scope="session")
def tokenizer(synth_data, tmp_path_factory) -> Tokenizer:
    pages, examples = synth_data
    path = tmp_path_factory.mktemp("tok") / "tok.json"
    return train_tokenizer(texts(list(pages.values()), examples), 8192, path)


@pytest.fixture(scope="session")
def tokenizer_path(tokenizer, tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("tokfile") / "tokenizer.json"
    tokenizer.save(str(path))
    return path


@pytest.fixture(scope="session")
def xs_config(tokenizer) -> GrounderConfig:
    return replace(GrounderConfig.from_yaml(Path(__file__).parent.parent / "configs" / "xs.yaml"),
                   vocab=tokenizer.get_vocab_size(), dropout=0.0)
