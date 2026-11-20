"""Rank with a trained checkpoint (CPU by default), a whole example list at a time."""

from __future__ import annotations

from typing import Sequence

import torch

from grounder.eval import rank_model
from grounder.model import load_checkpoint
from grounder.ranking import Ranked
from grounder.records import Example, Page


class ModelRanker:
    def __init__(self, checkpoint: str, device: str = "cpu", batch_size: int = 32):
        self.checkpoint, self.batch_size = checkpoint, batch_size
        self.model, self.tokenizer, _ = load_checkpoint(checkpoint, device)

    def rank_all(self, pages: dict[str, Page], examples: Sequence[Example]) -> list[Ranked]:
        with torch.no_grad():
            return rank_model(self.model, self.tokenizer, pages, examples, self.batch_size)

    def rank(self, page: Page, example: Example) -> Ranked:
        return self.rank_all({page.page_id: page}, [example])[0]
