from __future__ import annotations

import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest
import torch

from grounder.data.dataset import GrounderDataset, collate
from grounder.eval import rank, rank_model
from grounder.export import export_model_dir, parity_check
from grounder.features import MODEL_INPUTS, combine_chunk_logits, encode_page
from grounder.model import Grounder
from grounder.ranking import example_elements, ranked_from_logits
from grounder.records import Element, Example, Page
from grounder.runtime import Grounder as Runtime
from grounder.runtime import new_session, session_logits


@pytest.fixture(scope="module")
def exported(xs_config, tokenizer_path, tmp_path_factory):
    torch.manual_seed(0)
    model = Grounder(xs_config).eval()
    return model, export_model_dir(model, tokenizer_path, tmp_path_factory.mktemp("dist"))


def _items(synth_data, count=12):
    pages, examples = synth_data
    return [(pages[e.page_id], e) for e in examples[:count]]


def test_parity_fp32_and_int8(exported, synth_data, tokenizer):
    model, out = exported
    report = parity_check(model, tokenizer, out, _items(synth_data))
    assert report["fp32_max_abs_diff"] < 1e-4
    assert report["int8_top1_agreement"] >= 0.5
    assert report["int8_mb"] < report["fp32_mb"]


def test_onnx_dynamic_axes_with_padded_batches(exported, synth_data, tokenizer):
    model, out = exported
    pages, examples = synth_data
    ds = GrounderDataset(pages, examples[:8], tokenizer, augment=None)
    batch = collate([ds[i] for i in range(8)])
    feeds = {k: batch[k].numpy() for k in MODEL_INPUTS}
    with torch.no_grad():
        expected = model(*(batch[k] for k in MODEL_INPUTS)).numpy()
    got = new_session(out / "model.onnx").run(["logits"], feeds)[0]
    real = batch["positive"].numpy().shape
    assert got.shape == expected.shape == real
    np.testing.assert_allclose(got[expected > -1e8], expected[expected > -1e8], atol=1e-4)


def test_runtime_matches_pytorch_top1(exported, synth_data, tokenizer):
    model, out = exported
    runtime = Runtime(out, quantized=False)
    for page, ex in _items(synth_data):
        expected = rank(model, tokenizer, page, ex)
        got = runtime.rank(ex.query, example_elements(page, ex), k=3, mode=ex.mode, history=ex.history)
        assert got["results"][0]["id"] == expected.ids[0]
        assert got["null_prob"] == pytest.approx(expected.null_prob, abs=1e-4)
        assert [r["id"] for r in got["results"]] == expected.ids[:3]


def test_runtime_accepts_dicts_and_int8(exported, synth_data):
    _, out = exported
    pages, examples = synth_data
    page, ex = pages[examples[0].page_id], examples[0]
    result = Runtime(out).rank(ex.query, [e.to_dict() for e in page.elements], k=2)
    assert len(result["results"]) == 2 and set(result["results"][0]) == {"id", "score", "abs_box", "role", "text"}
    assert 0.0 <= result["null_prob"] <= 1.0
    assert Runtime(out).rank("click cart", [])["results"] == []


def test_runtime_chunks_large_pages(exported, synth_data):
    _, out = exported
    pages, examples = synth_data
    base = pages[examples[0].page_id].elements
    els = [replace(base[i % len(base)], id=i) for i in range(1100)]
    result = Runtime(out, quantized=False).rank("click cart", els, k=5)
    assert len(result["results"]) == 5 and all(0 <= r["id"] < 1100 for r in result["results"])
    assert 0.0 <= result["null_prob"] <= 1.0


def test_runtime_does_not_import_torch(exported, tmp_path):
    _, out = exported
    code = (
        "import sys; from grounder.runtime import Grounder; "
        f"g = Grounder({str(out)!r}); g.rank('click cart', [{{'id': 1, 'role': 'link', 'text': 'cart'}}]); "
        "assert 'torch' not in sys.modules, 'runtime imported torch'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_chunked_scores_form_one_distribution_and_find_tail_target(exported, tokenizer):
    model, out = exported
    runtime = Runtime(out, quantized=False)
    els = [Element(id=i, role="link", text=f"filler {i % 7}", context="footer") for i in range(600)]
    els[580] = Element(id=580, role="button", text="cart", context="footer")
    chunks = [els[:512], els[512:]]
    result = runtime.rank("click cart", els, k=600)
    scores = {r["id"]: r["score"] for r in result["results"]}
    assert len(scores) == 600
    assert sum(scores.values()) + result["null_prob"] == pytest.approx(1.0, abs=1e-5)
    arrays = [encode_page(tokenizer, "[DO] click cart", c, **runtime.limits) for c in chunks]
    logits = combine_chunk_logits([session_logits(runtime.session, a) for a in arrays])
    expected = ranked_from_logits([e.id for e in els], logits)
    assert [r["id"] for r in result["results"][:5]] == expected.ids[:5]
    assert scores[580] == pytest.approx(expected.scores[expected.ids.index(580)], rel=1e-5)


def test_batched_eval_matches_one_at_a_time_and_chunked_runtime(exported, synth_data, tokenizer):
    model, out = exported
    pages, examples = synth_data
    batched = rank_model(model, tokenizer, pages, examples[:20], batch_size=7)
    for ex, got in zip(examples[:20], batched):
        single = rank_model(model, tokenizer, pages, [ex])[0]
        assert got.ids == single.ids and got.null_prob == pytest.approx(single.null_prob, abs=1e-5)
    big = Page("big", "u", "s", "t", [1, 1], [1, 1],
               [Element(id=i, role="link", text=f"cart {i % 9}", context="x") for i in range(700)])
    ex = Example("big", "plain", "do", "click cart", [], 0, [0])
    expected = Runtime(out, quantized=False).rank("click cart", big.elements, k=3)
    got = rank_model(model, tokenizer, {"big": big}, [ex])[0]
    assert got.ids[:3] == [r["id"] for r in expected["results"]]
    assert got.null_prob == pytest.approx(expected["null_prob"], abs=1e-4)
