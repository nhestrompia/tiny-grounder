"""Export a checkpoint to ONNX (fp32 + int8) and check parity with PyTorch.

uv run python -m grounder.export --checkpoint runs/m/model.pt --out dist/m \
    [--pages DIR... --examples dev.jsonl --max-examples 200]
"""

from __future__ import annotations

import argparse
import json
import shutil
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from tokenizers import Tokenizer

from grounder.data.pages import load_examples, load_pages_from
from grounder.features import MODEL_INPUTS, encode_page
from grounder.model import Grounder, load_checkpoint
from grounder.ranking import example_elements, example_query
from grounder.records import Example, Page
from grounder.runtime import new_session, session_logits

OPSET = 17
_DYNAMIC_AXES = {
    "query_ids": {0: "batch", 1: "lq"},
    "el_text_ids": {0: "batch", 1: "n"},
    "el_ctx_ids": {0: "batch", 1: "n"},
    "el_role": {0: "batch", 1: "n"},
    "el_feats": {0: "batch", 1: "n"},
    "el_mask": {0: "batch", 1: "n"},
    "logits": {0: "batch", 1: "n_plus_1"},
}


def _example_inputs(model: Grounder) -> tuple[torch.Tensor, ...]:
    """Dummy batch (B=2, Lq=7, N=5) with shapes different from every other dim so axes stay dynamic."""
    cfg, b, lq, n = model.cfg, 2, 7, 5
    return (
        torch.randint(2, cfg.vocab, (b, lq)),
        torch.randint(0, cfg.vocab, (b, n, cfg.max_text)),
        torch.randint(0, cfg.vocab, (b, n, cfg.max_ctx)),
        torch.randint(0, cfg.n_roles, (b, n)),
        torch.rand(b, n, cfg.n_feats),
        torch.ones(b, n, dtype=torch.bool),
    )


def export_onnx(model: Grounder, path: str | Path) -> None:
    """fp32 ONNX with dynamic batch, query length and element count."""
    model = model.cpu().eval()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # the legacy exporter is deliberate: it takes dynamic_axes
        torch.onnx.export(
            model, _example_inputs(model), str(path), input_names=list(MODEL_INPUTS), output_names=["logits"],
            dynamic_axes=_DYNAMIC_AXES, opset_version=OPSET, dynamo=False,
        )


def quantize_int8(fp32_path: str | Path, int8_path: str | Path) -> None:
    """Dynamic int8 quantisation of the MatMul/Gemm weights and the embedding tables (Gather).

    The embeddings are half of the parameters, so quantising them is what brings M from ~11 MB to ~4.5 MB.
    """
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul", "Gemm", "Gather"])


def export_model_dir(model: Grounder, tokenizer_path: str | Path, out_dir: str | Path) -> Path:
    """Write model.onnx, model.int8.onnx, tokenizer.json and config.json: everything the runtime loads."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    export_onnx(model, out / "model.onnx")
    quantize_int8(out / "model.onnx", out / "model.int8.onnx")
    shutil.copyfile(tokenizer_path, out / "tokenizer.json")
    (out / "config.json").write_text(json.dumps(asdict(model.cfg), indent=2))
    return out


@torch.no_grad()
def parity_check(
    model: Grounder, tokenizer: Tokenizer, model_dir: str | Path, items: Sequence[tuple[Page, Example]]
) -> dict[str, float]:
    """PyTorch vs ONNX fp32 (max abs logit diff) and vs int8 (top-1 agreement), one page at a time."""
    model = model.cpu().eval()
    fp32 = new_session(Path(model_dir) / "model.onnx")
    int8 = new_session(Path(model_dir) / "model.int8.onnx")
    cfg, max_diff, agree = model.cfg, 0.0, 0
    for page, ex in items:
        arrays = encode_page(tokenizer, example_query(ex), example_elements(page, ex), cfg.max_q, cfg.max_text, cfg.max_ctx)
        ref = model(*(torch.from_numpy(arrays[k])[None] for k in MODEL_INPUTS))[0].numpy()
        max_diff = max(max_diff, float(np.abs(ref - session_logits(fp32, arrays)).max()))
        agree += int(ref.argmax() == session_logits(int8, arrays).argmax())
    return {
        "examples": len(items), "fp32_max_abs_diff": max_diff, "int8_top1_agreement": agree / max(len(items), 1),
        "fp32_mb": (Path(model_dir) / "model.onnx").stat().st_size / 1e6,
        "int8_mb": (Path(model_dir) / "model.int8.onnx").stat().st_size / 1e6,
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pages", nargs="+", help="page directories, merged by page_id")
    parser.add_argument("--examples")
    parser.add_argument("--max-examples", type=int, default=200)
    args = parser.parse_args()

    model, tokenizer, ckpt = load_checkpoint(args.checkpoint)
    out = export_model_dir(model, ckpt["tokenizer"], args.out)
    print(f"exported to {out}")
    if args.pages and args.examples:
        pages = load_pages_from(args.pages)
        items = [(pages[e.page_id], e) for e in load_examples(args.examples)[: args.max_examples]]
        for name, value in parity_check(model, tokenizer, out, items).items():
            print(f"{name}: {value:.6g}")


if __name__ == "__main__":
    _main()
