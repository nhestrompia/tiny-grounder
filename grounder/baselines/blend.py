"""Model + BM25 blend: score = log p_model + alpha * bm25 / max_bm25 (max over the page's elements; 0 if all are 0).

alpha = 0 is the model alone; large alpha approaches BM25 with the model breaking ties.
"""

from __future__ import annotations

import numpy as np

from grounder.ranking import Ranked

EPS = 1e-12


def blend(model: Ranked, bm25: Ranked, alpha: float) -> Ranked:
    """Blend two rankings of the same elements. `null_prob` is the model's."""
    by_id = dict(zip(bm25.ids, bm25.scores))
    top = max(by_id.values(), default=0.0)
    norm = top if top > 0 else 1.0
    scores = np.array([np.log(max(p, EPS)) + alpha * by_id.get(i, 0.0) / norm for i, p in zip(model.ids, model.scores)])
    order = np.argsort(-scores, kind="stable")
    return Ranked([model.ids[i] for i in order], [float(scores[i]) for i in order], model.null_prob)
