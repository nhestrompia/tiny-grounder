from __future__ import annotations

from grounder.export import export_model_dir
from grounder.model import Grounder
from tools import bench


def test_synthetic_elements_are_deterministic_and_valid():
    a, b = bench.synthetic_elements(30, seed=1), bench.synthetic_elements(30, seed=1)
    assert [e.to_dict() for e in a] == [e.to_dict() for e in b]
    assert [e.id for e in a] == list(range(30)) and all(e.text for e in a)


def test_percentiles():
    assert bench.percentiles(list(range(1, 101))) == (50.5, 95.05)


def test_time_rank_runs_warmup_plus_iters():
    calls = []
    p50, p95 = bench.time_rank(lambda: calls.append(1), iters=5, warmup=2)
    assert len(calls) == 7 and 0 <= p50 <= p95


def test_benchmark_table_on_an_exported_model(xs_config, tokenizer_path, tmp_path):
    model_dir = export_model_dir(Grounder(xs_config).eval(), tokenizer_path, tmp_path / "dist")
    rows = bench.benchmark(model_dir, [1], [10, 20], iters=2)
    table = bench.to_markdown(rows, bench.model_sizes_mb(model_dir))
    assert "| 1 | fp32 |" in table and "| 1 | int8 |" in table and "model.int8.onnx" in table
    assert len(rows) == 4 and all(r["p50"] > 0 for r in rows)
