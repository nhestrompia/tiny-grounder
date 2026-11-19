from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from grounder.data.dataset import AugmentConfig, GrounderDataset, collate
from grounder.features import MODEL_INPUTS
from grounder.model import NEG_INF, Grounder, GrounderConfig, listwise_loss

CONFIGS = Path(__file__).parent.parent / "configs"
EXPECTED_PARAMS = {"xs": 1_356_929, "s": 2_553_601, "m": 4_357_889, "l": 8_208_001}


def random_batch(cfg: GrounderConfig, b: int = 3, lq: int = 6, n: int = 7) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    return {
        "query_ids": torch.randint(2, cfg.vocab, (b, lq), generator=g),
        "el_text_ids": torch.randint(0, cfg.vocab, (b, n, cfg.max_text), generator=g),
        "el_ctx_ids": torch.randint(0, cfg.vocab, (b, n, cfg.max_ctx), generator=g),
        "el_role": torch.randint(0, cfg.n_roles, (b, n), generator=g),
        "el_feats": torch.rand(b, n, cfg.n_feats, generator=g),
        "el_mask": torch.ones(b, n, dtype=torch.bool),
    }


@pytest.fixture
def model(xs_config) -> Grounder:
    torch.manual_seed(0)
    return Grounder(xs_config).eval()


@pytest.mark.parametrize("name", sorted(EXPECTED_PARAMS))
def test_param_counts(name):
    cfg = GrounderConfig.from_yaml(CONFIGS / f"{name}.yaml")
    assert Grounder(cfg).count_params() == EXPECTED_PARAMS[name]


def test_output_shape_and_padded_logits(model, xs_config):
    batch = random_batch(xs_config)
    batch["el_mask"][1, 4:] = False
    logits = model(**batch)
    assert logits.shape == (3, 8)
    assert (logits[1, 5:] == NEG_INF).all() and (logits[1, :5] > NEG_INF).all()


def test_masked_mean_handles_empty_rows(model, xs_config):
    batch = random_batch(xs_config)
    batch["el_ctx_ids"].zero_()
    batch["el_text_ids"][:, 0] = 0
    logits = model(**batch)
    assert torch.isfinite(logits).all()
    assert (model._masked_mean(torch.zeros(2, 3, 5, dtype=torch.long)) == 0).all()


def test_padding_invariance(model, xs_config):
    cfg = xs_config
    small = random_batch(cfg, b=1, lq=5, n=6)
    big = {k: v.clone() for k, v in small.items()}
    pad = random_batch(replace(cfg), b=1, lq=5, n=4)
    for k in ("el_text_ids", "el_ctx_ids", "el_role", "el_feats", "el_mask"):
        big[k] = torch.cat([small[k], pad[k]], dim=1)
    big["el_mask"][:, 6:] = False
    big["query_ids"] = torch.cat([small["query_ids"], torch.zeros(1, 3, dtype=torch.long)], dim=1)
    with torch.no_grad():
        a, b = model(**small), model(**big)
    torch.testing.assert_close(b[:, :7], a, atol=1e-5, rtol=1e-4)


def test_permutation_equivariance(model, xs_config):
    batch = random_batch(xs_config, b=2, n=9)
    perm = torch.randperm(9, generator=torch.Generator().manual_seed(3))
    shuffled = {k: (v[:, perm] if k.startswith("el_") else v) for k, v in batch.items()}
    with torch.no_grad():
        a, b = model(**batch), model(**shuffled)
    torch.testing.assert_close(b[:, 1:], a[:, 1:][:, perm], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(b[:, 0], a[:, 0], atol=1e-5, rtol=1e-4)


def test_listwise_loss():
    logits = torch.tensor([[0.0, 1.0, 2.0, NEG_INF]])
    pos = torch.tensor([[False, True, True, False]])
    expected = -torch.log(torch.softmax(logits, -1)[0, 1:3].sum())
    torch.testing.assert_close(listwise_loss(logits, pos), expected)
    assert listwise_loss(logits, torch.tensor([[True, False, False, False]])) > 2.0


def test_overfits_100_examples(synth_data, tokenizer, xs_config):
    pages, examples = synth_data
    subset = [e for e in examples if e.kind in ("plain", "paraphrase")][:100]
    ds = GrounderDataset(pages, subset, tokenizer, augment=None)
    batch = collate([ds[i] for i in range(len(ds))])
    torch.manual_seed(0)
    model = Grounder(xs_config)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    for _ in range(250):
        loss = listwise_loss(model(**{k: batch[k] for k in random_batch(xs_config)}), batch["positive"])
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(**{k: batch[k] for k in random_batch(xs_config)})
    hit = batch["positive"].gather(1, logits.argmax(1, keepdim=True)).squeeze(1)
    assert hit.float().mean() >= 0.97, hit.float().mean()


def test_rounded_collate_gives_same_loss(model, synth_data, tokenizer, xs_config):
    pages, examples = synth_data
    ds = GrounderDataset(pages, examples[:8], tokenizer, None, max_q=xs_config.max_q,
                         max_text=xs_config.max_text, max_ctx=xs_config.max_ctx)
    items = [ds[i] for i in range(len(ds))]
    exact, rounded = collate(items), collate(items, round_q=8, round_n=32)
    assert rounded["query_ids"].shape[1] % 8 == 0 and rounded["el_role"].shape[1] % 32 == 0
    assert rounded["el_role"].shape[1] >= exact["el_role"].shape[1]
    with torch.no_grad():
        losses = [listwise_loss(model(*(b[k] for k in MODEL_INPUTS)), b["positive"]) for b in (exact, rounded)]
    torch.testing.assert_close(losses[1], losses[0], atol=1e-5, rtol=1e-4)
