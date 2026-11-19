from __future__ import annotations

import numpy as np

from grounder.data.dataset import (
    AugmentConfig, BucketBatchSampler, GrounderDataset, cap_elements, collate, ocr_noise, prepare_example,
)
from grounder.records import Element, Example, Page

ALWAYS = AugmentConfig(shuffle=1.0, drop_role=0.5, drop_context=0.5, text_noise=0.5, ocr_page=0.3)


def _examples(examples, kind, count=20):
    return [e for e in examples if e.kind == kind][:count]


def test_augmentation_keeps_positives_correct(synth_data):
    pages, examples = synth_data
    for i, ex in enumerate(_examples(examples, "plain") + _examples(examples, "context")):
        rng = np.random.default_rng(i)
        prep = prepare_example(pages[ex.page_id], ex, rng, ALWAYS)
        chosen = [e.id for e, is_pos in zip(prep.elements, prep.positive[1:]) if is_pos]
        assert chosen == [ex.target_id]
        assert not prep.positive[0]
        assert len(prep.positive) == len(prep.elements) + 1
        assert prep.query.startswith("[DO] ")


def test_shuffle_changes_order_but_not_content(synth_data):
    pages, examples = synth_data
    ex = examples[0]
    original = [e.id for e in pages[ex.page_id].elements]
    shuffled = prepare_example(pages[ex.page_id], ex, np.random.default_rng(0), AugmentConfig(shuffle=1.0))
    assert [e.id for e in shuffled.elements] != original
    assert sorted(e.id for e in shuffled.elements) == sorted(original)


def test_null_examples_remove_target_and_make_null_positive(synth_data):
    pages, examples = synth_data
    for ex in _examples(examples, "null"):
        prep = prepare_example(pages[ex.page_id], ex, np.random.default_rng(0), ALWAYS)
        ids = {e.id for e in prep.elements}
        assert not ids & set(ex.equiv_ids)
        assert len(ids) == len(pages[ex.page_id].elements) - len(ex.equiv_ids)
        assert prep.positive[0] and prep.positive.sum() == 1


def test_augmentation_does_not_mutate_the_page(synth_data):
    pages, examples = synth_data
    ex = examples[0]
    before = [e.to_dict() for e in pages[ex.page_id].elements]
    prepare_example(pages[ex.page_id], ex, np.random.default_rng(0), ALWAYS)
    assert [e.to_dict() for e in pages[ex.page_id].elements] == before


def test_ocr_page_applies_source_conditions_to_every_element(synth_data):
    pages, examples = synth_data
    ex = examples[0]
    cfg = AugmentConfig(ocr_page=1.0, ocr_page_noise=1.0, box_jitter=0.0)
    prep = prepare_example(pages[ex.page_id], ex, np.random.default_rng(0), cfg)
    assert all(e.role == "unknown" and e.context == "" and 0.6 <= e.conf <= 0.9 for e in prep.elements)


def test_role_and_context_drops_are_per_element(synth_data):
    pages, examples = synth_data
    ex = max(examples, key=lambda e: len(pages[e.page_id].elements))
    cfg = AugmentConfig(drop_role=0.5, drop_context=0.5, text_noise=0.0, ocr_page=0.0, box_jitter=0.0)
    prep = prepare_example(pages[ex.page_id], ex, np.random.default_rng(0), cfg)
    roles = [e.role == "unknown" for e in prep.elements]
    contexts = [e.context == "" for e in prep.elements]
    assert 0 < sum(roles) < len(roles) and 0 < sum(contexts) < len(contexts)
    assert all(e.conf == 1.0 for e in prep.elements)


def test_ocr_noise_edits_text():
    rng = np.random.default_rng(0)
    assert any(ocr_noise("hello world", rng) != "hello world" for _ in range(10))
    assert all(ocr_noise("a", rng) for _ in range(20))


def test_large_pages_keep_all_positives():
    els = [Element(id=i, role="link", text=f"t{i}") for i in range(1000)]
    page = Page("p", "u", "s", "t", [1, 1], [1, 1], els)
    ex = Example("p", "plain", "do", "click t5", [], 5, [5, 900, 901])
    prep = prepare_example(page, ex, np.random.default_rng(0), None)
    ids = {e.id for e in prep.elements}
    assert len(ids) == 512 and {5, 900, 901} <= ids and prep.positive[1:].sum() == 3
    assert len(cap_elements(els[:10], {1}, np.random.default_rng(0))) == 10



def test_pages_with_thousands_of_positives_are_capped():
    els = [Element(id=i, role="link", text="reply") for i in range(3600)]
    page = Page("p", "u", "s", "t", [1, 1], [1, 1], els)
    ex = Example("p", "plain", "do", "click reply", [], 0, list(range(3000)))
    prep = prepare_example(page, ex, np.random.default_rng(0), None)
    assert len(prep.elements) == 512 and prep.positive[1:].sum() == 256

def test_dataset_is_reproducible_and_collate_pads(synth_data, tokenizer):
    pages, examples = synth_data
    ds = GrounderDataset(pages, examples[:6], tokenizer, AugmentConfig(), seed=3)
    first = [ds[i] for i in range(6)]
    again = [ds[i] for i in range(6)]
    assert all(np.array_equal(a["el_feats"], b["el_feats"]) for a, b in zip(first, again))
    ds.set_epoch(1)
    assert not all(np.array_equal(a["el_feats"], ds[i]["el_feats"]) for i, a in enumerate(first)
                   if a["el_feats"].shape == ds[i]["el_feats"].shape)

    batch = collate(first)
    n = max(len(i["el_role"]) for i in first)
    assert batch["el_mask"].shape == (6, n) and batch["positive"].shape == (6, n + 1)
    assert batch["el_mask"].sum(1).tolist() == [len(i["el_role"]) for i in first]
    assert not batch["positive"][:, 1:][~batch["el_mask"]].any()
    assert batch["positive"].any(dim=1).all()


def test_bucket_sampler_groups_similar_sizes():
    rng = np.random.default_rng(0)
    sizes = rng.integers(10, 500, size=640)
    sampler = BucketBatchSampler(sizes, batch_size=16, shuffle=True, seed=0)
    batches = list(sampler)
    assert len(batches) == len(sampler) == 40
    assert sorted(i for b in batches for i in b) == list(range(640))
    spread = np.mean([sizes[b].max() - sizes[b].min() for b in batches])
    assert spread < 0.2 * (sizes.max() - sizes.min())
    sampler.set_epoch(1)
    assert list(sampler) != batches


def test_bucket_sampler_caps_elements_per_batch():
    sizes = [10] * 100 + [500] * 40
    sampler = BucketBatchSampler(sizes, batch_size=64, shuffle=True, seed=1, max_batch_elements=2000)
    batches = list(sampler)
    assert sorted(i for b in batches for i in b) == list(range(140)) and len(sampler) == len(batches)
    for b in batches:
        assert len(b) <= 64 and (len(b) == 1 or len(b) * max(sizes[i] for i in b) <= 2000)
    assert any(len(b) == 64 for b in batches)  # small pages still get full batches


def _reference_item(pages, ex, tokenizer, augment, seed, epoch, index):
    """The pre-compaction path: original Page objects, no packed store."""
    from grounder.features import TokenCache, encode_page
    rng = np.random.default_rng((seed, epoch, index))
    prep = prepare_example(pages[ex.page_id], ex, rng, augment)
    els = [e for pid in sorted({ex.page_id}) for e in pages[pid].elements]
    cache = TokenCache.build(tokenizer, els, 16, 24)
    item = encode_page(tokenizer, prep.query, prep.elements, max_q=48, max_text=16, max_ctx=24, cache=cache)
    item["positive"] = prep.positive
    return item


def test_compact_dataset_matches_page_objects(synth_data, tokenizer):
    pages, examples = synth_data
    pages = dict(pages)
    odd = [Element(id=i, role="weird" if i % 7 == 0 else "link", text=f"Zażółć ünï {i % 40} \U0001F600" if i % 3 else "",
                   context="ctx é" if i % 5 else "", box=[i / 1000, i / 300, 0.1 + i / 9999, 0.03],
                   in_viewport=i % 2 == 0, enabled=i % 4 != 0, conf=0.123456789 + i / 5000,
                   tag="a", source="ocr", href="/x", abs_box=[1.0, 2.0, 3.0, 4.0]) for i in range(900)]
    pages["big"] = Page("big", "u", "s", "t", [1, 1], [1, 1], odd)
    chosen = list(examples[:8]) + [
        Example("big", "plain", "do", "click Zażółć 5", [], 5, [5, 800]),
        Example("big", "null", "do", "click Zażółć 6", [], 6, [6]),
    ]
    for augment in (AugmentConfig(), None):
        ds = GrounderDataset(pages, chosen, tokenizer, augment, seed=5)
        for epoch in (0, 1):
            ds.set_epoch(epoch)
            for i, ex in enumerate(chosen):
                got, want = ds[i], _reference_item(pages, ex, tokenizer, augment, 5, epoch, i)
                assert got.keys() == want.keys()
                for k in want:
                    assert got[k].dtype == want[k].dtype and np.array_equal(got[k], want[k]), (k, i, epoch)
    assert ds.sizes() == [min(len(pages[ex.page_id].elements), 512) for ex in chosen]
    import pickle
    clone = pickle.loads(pickle.dumps(ds))
    assert all(np.array_equal(clone[i]["el_feats"], ds[i]["el_feats"]) for i in range(len(chosen)))


def test_shared_compact_pages_match_and_pickle_small(synth_data, tokenizer, tmp_path):
    import pickle
    pages, examples = synth_data
    ds = GrounderDataset(pages, examples[:8], tokenizer, AugmentConfig(), seed=2)
    want = [ds[i] for i in range(8)]
    full_size = len(pickle.dumps(ds))
    ds.pages.share(tmp_path / "pages")
    clone = pickle.loads(pickle.dumps(ds))
    assert len(pickle.dumps(ds)) < full_size
    for i, w in enumerate(want):
        for k, v in w.items():
            assert np.array_equal(ds[i][k], v) and np.array_equal(clone[i][k], v)


def test_bucketed_sampler_batches_fit_static_shapes():
    from grounder.data.dataset import StaticShapes, bucket_size
    rng = np.random.default_rng(0)
    sizes = [bucket_size(int(n)) for n in rng.integers(1, 513, size=3000)]
    shapes = StaticShapes(48, batch_size=64, max_batch_elements=12288)
    sampler = BucketBatchSampler(sizes, 64, shuffle=True, seed=0, max_batch_elements=12288)
    for batch in sampler:
        assert len(batch) <= shapes.rows(max(sizes[i] for i in batch))
    assert bucket_size(1) == 32 and bucket_size(97) == 128 and bucket_size(512) == 512 and bucket_size(513) == 1024
