"""CPU latency benchmark of `runtime.Grounder.rank` (features + tokenisation + ONNX) on synthetic pages.

    uv run python tools/bench.py --model-dir dist/m --threads 1 4 --iters 30

Prints a markdown table: p50 / p95 milliseconds per call for N = 100, 250, 500, 1000 elements, fp32 vs
int8, for each thread count, and the model file sizes. Pages over 512 elements run as several chunks.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from grounder.records import Element
from grounder.runtime import Grounder

SIZES = (100, 250, 500, 1000)
WORDS = (
    "home cart login search settings help profile orders account sign in out shop deals new sale about "
    "contact careers press blog news pricing download share save print delete add remove next back"
).split()
ROLES = ("button", "link", "textbox", "checkbox", "tab", "menuitem")
QUERY = "open the shopping cart"


def synthetic_elements(n: int, seed: int = 0) -> list[Element]:
    """`n` elements with 1-3 word labels, a short context and random boxes."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        text = " ".join(rng.choice(WORDS, size=int(rng.integers(1, 4))))
        context = " ".join(rng.choice(WORDS, size=int(rng.integers(0, 8))))
        x, y, w, h = rng.random() * 0.8, rng.random() * 4, 0.05 + rng.random() * 0.2, 0.02 + rng.random() * 0.05
        out.append(Element(id=i, role=str(rng.choice(ROLES)), text=text, context=context, tag="a",
                           box=[x, y, w, h], abs_box=[x * 1280, y * 800, w * 1280, h * 800], in_viewport=y < 1))
    return out


def percentiles(samples_ms: Sequence[float]) -> tuple[float, float]:
    return float(np.percentile(samples_ms, 50)), float(np.percentile(samples_ms, 95))


def time_rank(call: Callable[[], object], iters: int, warmup: int = 3) -> tuple[float, float]:
    """(p50, p95) wall-clock milliseconds of `call`, after `warmup` untimed runs."""
    for _ in range(warmup):
        call()
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        call()
        samples.append((time.perf_counter() - start) * 1000)
    return percentiles(samples)


def model_sizes_mb(model_dir: Path) -> dict[str, float]:
    return {p.name: p.stat().st_size / 1e6 for p in sorted(model_dir.glob("model*.onnx"))}


def benchmark(model_dir: Path, threads: Sequence[int], sizes: Sequence[int], iters: int) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for t in threads:
        for quantized in (False, True):
            grounder = Grounder(model_dir, quantized=quantized, threads=t)
            for n in sizes:
                elements = synthetic_elements(n)
                p50, p95 = time_rank(lambda: grounder.rank(QUERY, elements, k=5), iters)
                rows.append({"threads": t, "precision": "int8" if quantized else "fp32", "n": n, "p50": p50, "p95": p95})
    return rows


def to_markdown(rows: Sequence[dict[str, float | int | str]], sizes_mb: dict[str, float]) -> str:
    sizes = sorted({int(r["n"]) for r in rows})
    lines = ["| threads | precision | " + " | ".join(f"N={n} p50 / p95 (ms)" for n in sizes) + " |",
             "|---|---|" + "---|" * len(sizes)]
    for t, prec in dict.fromkeys((r["threads"], r["precision"]) for r in rows):
        cells = {int(r["n"]): f"{r['p50']:.1f} / {r['p95']:.1f}" for r in rows if (r["threads"], r["precision"]) == (t, prec)}
        lines.append(f"| {t} | {prec} | " + " | ".join(cells[n] for n in sizes) + " |")
    lines += ["", "| file | size (MB) |", "|---|---|"]
    lines += [f"| {name} | {mb:.2f} |" for name, mb in sizes_mb.items()]
    return "\n".join(lines)


def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--threads", type=int, nargs="+", default=[4])
    parser.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args(argv)
    model_dir = Path(args.model_dir)
    print(to_markdown(benchmark(model_dir, args.threads, args.sizes, args.iters), model_sizes_mb(model_dir)))


if __name__ == "__main__":
    _main()
