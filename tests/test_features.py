from __future__ import annotations

import numpy as np

from grounder.features import combine_chunk_logits, content_tokens, element_features, encode_page, overlap, stem
from grounder.records import Element, format_query
from grounder.tokenizer import PAD_ID, UNK_ID, load_tokenizer, tokenizer_stats


def test_stem_is_light_and_deterministic():
    assert [stem(w) for w in ("carts", "boxes", "shipping", "added", "class", "bus", "is")] == [
        "cart", "box", "shipp", "add", "class", "bus", "is"]
    assert stem("cart") == stem("carts")


def test_content_tokens_drop_stop_words_and_stem():
    assert content_tokens("Click the Shopping Carts button!") == {"shopp", "cart"}
    assert content_tokens("") == set()


def test_overlap():
    assert overlap("[DO] click login", "Login") == 1.0
    assert overlap("[DO] my shopping cart", "Cart") == 0.5
    assert overlap("[DO] click login", "") == 0.0
    assert overlap("[DO] click the button", "anything") == 0.0  # no content tokens in the query


def test_markers_are_not_content():
    q = format_query("goal", "buy a cart", ["opened home"])
    assert overlap(q, "do goal done") == 0.0
    assert overlap(q, "buy cart") == 0.5  # buy, cart of buy, cart, open, home


def test_element_features():
    el = Element(id=1, role="button", text="Login", context="Header", box=[-0.2, 9.0, 1.5, 0.1],
                 in_viewport=False, enabled=True, conf=0.7)
    f = element_features(el, "[DO] click login")
    assert f.shape == (10,) and f.dtype == np.float32
    np.testing.assert_allclose(f, [0.0, 1.0, 1.0, 0.1, 0.0, 1.0, 0.7, 1.0, 0.0, 1.0], atol=1e-6)


def test_tokenizer_special_tokens(tokenizer):
    enc = tokenizer.encode("[DO] click Login [GOAL] x [DONE]")
    assert enc.ids[0] == tokenizer.token_to_id("[DO]") and PAD_ID == 0 and UNK_ID == 1
    assert tokenizer.token_to_id("[PAD]") == 0 and tokenizer.token_to_id("[UNK]") == 1
    assert tokenizer.token_to_id("[GOAL]") in enc.ids and tokenizer.token_to_id("[DONE]") in enc.ids
    assert tokenizer.encode("LOGIN").ids == tokenizer.encode("login").ids


def test_tokenizer_stats_and_reload(tokenizer, tokenizer_path):
    stats = tokenizer_stats(load_tokenizer(tokenizer_path), ["click login", "open cart"])
    assert stats["avg_tokens"] <= 6 and stats["unk_rate"] == 0.0


def test_encode_page_shapes_and_truncation(tokenizer):
    els = [Element(id=i, role="link", text="login " * 30, context="") for i in range(3)]
    out = encode_page(tokenizer, format_query("do", "click login"), els, max_q=8, max_text=4, max_ctx=5)
    assert out["query_ids"].shape[0] <= 8
    assert out["el_text_ids"].shape == (3, 4) and out["el_ctx_ids"].shape == (3, 5)
    assert (out["el_ctx_ids"] == PAD_ID).all()
    assert out["el_role"].shape == (3,) and out["el_feats"].shape == (3, 10) and out["el_mask"].all()


def test_encode_page_no_elements(tokenizer):
    out = encode_page(tokenizer, format_query("do", "click login"), [])
    assert out["el_feats"].shape == (0, 10) and out["el_text_ids"].shape == (0, 16)


def test_coverage_separates_exact_label_from_longer_label():
    exact = Element(id=1, role="link", text="Subscribe")
    longer = Element(id=2, role="link", text="Subscribe messages")
    assert element_features(exact, "[DO] open subscribe")[9] == 1.0
    assert element_features(longer, "[DO] open subscribe")[9] == 0.5
    assert element_features(Element(id=3, role="link", text=""), "[DO] open subscribe")[9] == 0.0


def test_combine_chunk_logits_concatenates_and_averages_null():
    out = combine_chunk_logits([np.array([1.0, 5.0, 6.0]), np.array([3.0, 7.0])])
    np.testing.assert_allclose(out, [2.0, 5.0, 6.0, 7.0])
