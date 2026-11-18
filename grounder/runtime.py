"""Inference: rank UI elements for an instruction with onnxruntime.

Imports only numpy, onnxruntime, tokenizers and grounder.{records,features,ranking,tokenizer}; no torch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import onnxruntime as ort

from grounder.features import CHUNK, MODEL_INPUTS, combine_chunk_logits, encode_page
from grounder.ranking import ranked_from_logits
from grounder.records import Element, Mode, format_query, validate_element
from grounder.tokenizer import load_tokenizer


def new_session(path: str | Path, threads: int = 4) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def session_logits(session: ort.InferenceSession, arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Logits [N+1] (index 0 = NULL) for ONE page given as `features.encode_page` arrays."""
    feeds = {name: arrays[name][None] for name in MODEL_INPUTS}
    return session.run(["logits"], feeds)[0][0]


class Grounder:
    """Loads `model.int8.onnx` (or `model.onnx`), `tokenizer.json` and `config.json` from `model_dir`."""

    def __init__(self, model_dir: str | Path, quantized: bool | None = None, threads: int = 4) -> None:
        root = Path(model_dir)
        int8, fp32 = root / "model.int8.onnx", root / "model.onnx"
        use_int8 = int8.exists() if quantized is None else quantized
        self.session = new_session(int8 if use_int8 else fp32, threads)
        self.tokenizer = load_tokenizer(root / "tokenizer.json")
        self.limits = {k: v for k, v in json.loads((root / "config.json").read_text()).items()
                       if k in ("max_q", "max_text", "max_ctx")}

    def rank(
        self, query: str, elements: Sequence[Element | dict[str, Any]], k: int = 5,
        mode: Mode = "do", history: list[str] | None = None,
    ) -> dict[str, Any]:
        """Top-`k` elements with softmax scores and the probability that nothing matches.

        Scores and null_prob come from one softmax over [NULL ; all elements], so they sum to 1.
        Pages over 512 elements (the training cap) are run in chunks of 512 in page order that share
        the query; the raw element logits are concatenated and the NULL logit is the mean of the
        chunks' NULL logits (features.combine_chunk_logits). That is an approximation: each chunk's
        attention only sees its own elements.
        """
        els = [e if isinstance(e, Element) else validate_element(e) for e in elements]
        if not els:
            return {"results": [], "null_prob": 1.0}
        text = format_query(mode, query, history)
        chunks = [
            session_logits(self.session, encode_page(self.tokenizer, text, els[i : i + CHUNK], **self.limits))
            for i in range(0, len(els), CHUNK)
        ]
        ranked = ranked_from_logits([e.id for e in els], combine_chunk_logits(chunks))
        by_id = {e.id: e for e in els}
        results = [
            {"id": eid, "score": score, "abs_box": by_id[eid].abs_box, "role": by_id[eid].role, "text": by_id[eid].text}
            for eid, score in zip(ranked.ids[:k], ranked.scores[:k])
        ]
        return {"results": results, "null_prob": ranked.null_prob}
