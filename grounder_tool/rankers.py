"""Pluggable rankers for the grounding tool: `make_ranker("bm25" | "exact" | "model:<dir>" | "blend:<dir>:<alpha>" | "router:<dir>:<tau>")`.

Every ranker implements `rank(query, elements, mode, history) -> grounder.ranking.Ranked` (all ids, best first).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

from grounder.ranking import Ranked
from grounder.records import Element, Mode

DEFAULT_TAU = 1.0  # router: lexical coverage at or above this goes to the exact-match + BM25 path


class Ranker(Protocol):
    name: str

    def rank(
        self, query: str, elements: Sequence[Element], mode: Mode = "do", history: list[str] | None = None
    ) -> Ranked: ...


def exact_bm25(query: str, elements: Sequence[Element]) -> Ranked:
    """Exact label matches first, BM25 for the rest: the same ranker the evaluation uses
    (`grounder.baselines.exact`), so what is measured is what ships."""
    from grounder.baselines.exact import rank_elements

    return rank_elements(query, list(elements))


class BM25Ranker:
    name = "bm25"
    lexical = True  # scores are match counts: passages with no term overlap are dropped by rank_passages

    def rank(self, query, elements, mode: Mode = "do", history=None) -> Ranked:
        from grounder.baselines.bm25 import rank_elements

        return rank_elements(query, list(elements))


class ExactRanker:
    """Keyword ranking against the supplied query or goal; lexical scores do not use history."""

    name = "exact"
    lexical = True  # scores are match counts: passages with no term overlap are dropped by rank_passages

    def rank(self, query, elements, mode: Mode = "do", history=None) -> Ranked:
        return exact_bm25(query, elements)


class ModelRanker:
    """The ONNX model in `model_dir` (int8 when present unless `int8=False`)."""

    def __init__(self, model_dir: str | Path, int8: bool | None = None, threads: int = 1) -> None:
        from grounder.runtime import Grounder

        self.model_dir = Path(model_dir)
        self.grounder = Grounder(self.model_dir, quantized=int8, threads=threads)
        self.name = f"model:{self.model_dir.name}"

    def rank(self, query, elements, mode: Mode = "do", history=None) -> Ranked:
        els = list(elements)
        out = self.grounder.rank(query, els, k=len(els), mode=mode, history=history)
        return Ranked([r["id"] for r in out["results"]], [r["score"] for r in out["results"]], out["null_prob"])


def lexical_coverage(query: str, elements: Sequence[Element]) -> float:
    """Best share of the query's content words found in a single element (`grounder.baselines.router.coverage`)."""
    from grounder.baselines.router import coverage

    return coverage(query, list(elements))


DEFAULT_ALPHA = 8.0  # blend weight chosen on the human select split (plan v2.1 step 4); see runs/compare-select


class BlendRanker:
    """log p_model + alpha * bm25/max_bm25 (`grounder.baselines.blend`), the step-4 winner. Scores are turned back
    into a softmax over the page's elements so the agent sees probabilities between 0 and 1."""

    def __init__(self, model: ModelRanker, alpha: float = DEFAULT_ALPHA) -> None:
        self.model, self.alpha = model, alpha
        self.name = f"blend:{model.model_dir.name}:{alpha:g}"

    def rank(self, query, elements, mode: Mode = "do", history=None) -> Ranked:
        import numpy as np

        from grounder.baselines.blend import blend
        from grounder.baselines.bm25 import rank_elements

        els = list(elements)
        mixed = blend(self.model.rank(query, els, mode, history), rank_elements(query, els), self.alpha)
        z = np.array(mixed.scores, dtype=float)
        p = np.exp(z - z.max()) if len(z) else z
        p = p / p.sum() if len(z) else p
        return Ranked(mixed.ids, [float(x) for x in p], mixed.null_prob)


class RouterRanker:
    """Lexical coverage >= tau -> exact match + BM25, otherwise the model."""

    def __init__(self, model: ModelRanker, tau: float = DEFAULT_TAU) -> None:
        self.model, self.tau = model, tau
        self.name = f"router:{model.model_dir.name}:{tau:g}"
        self.last_route = ""
        self.lexical = False  # per-call: True while the last rank() took the lexical route

    def rank(self, query, elements, mode: Mode = "do", history=None) -> Ranked:
        if lexical_coverage(query, elements) >= self.tau:
            self.last_route = "lexical"
            self.lexical = True  # the hardcoded null_prob=0 is not model confidence; observe must not shrink
            return exact_bm25(query, elements)
        self.last_route = "model"
        self.lexical = False
        return self.model.rank(query, elements, mode, history)


def make_ranker(spec: str = "exact", int8: bool | None = None, threads: int = 1) -> Ranker:
    """Build a lexical, model, blend or router ranker; all accept goal mode and history.

    Model paths use CPU ONNX inference. `int8=False` forces fp32; lexical rankers ignore history."""
    kind, _, rest = spec.partition(":")
    if kind == "bm25" and not rest:
        return BM25Ranker()
    if kind == "exact" and not rest:
        return ExactRanker()
    if kind == "model" and rest:
        return ModelRanker(rest, int8, threads)
    if kind == "blend" and rest:
        path, sep, tail = rest.rpartition(":")
        try:
            path, alpha = (path, float(tail)) if sep else (rest, DEFAULT_ALPHA)
        except ValueError:
            path, alpha = rest, DEFAULT_ALPHA
        return BlendRanker(ModelRanker(path, int8, threads), alpha)
    if kind == "router" and rest:
        path, sep, tail = rest.rpartition(":")
        try:
            path, tau = (path, float(tail)) if sep else (rest, DEFAULT_TAU)
        except ValueError:
            path, tau = rest, DEFAULT_TAU
        return RouterRanker(ModelRanker(path, int8, threads), tau)
    raise ValueError(f"bad ranker {spec!r}: use bm25, exact, model:<dir>, blend:<dir>:<alpha> or router:<dir>:<tau>")
