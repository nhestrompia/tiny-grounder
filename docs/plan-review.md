# Review of implementation-plan.md

The plan is sound: one narrow job, small listwise model, synthetic data, honest baselines. The changes below fix gaps that would have caused bugs or wasted a training run. They **override** `implementation-plan.md` where they conflict.

## Changes

1. **Coordinates are viewport-relative, not document-relative.** On a 12,000 px page, a document-normalised `y` squashes everything above the fold into 0.00–0.07, so position stops meaning anything. `box = [x/vw, y/vh, w/vw, h/vh]` (y may exceed 1 below the fold). The model feature clips `y` to [0, 4] and divides by 4.

2. **Records carry `href`.** `equiv_ids` and the nested-link rule both need the link target, but the record had no field for it. `href` is stored (origin-stripped path + query, max 200 chars) and is never a model input.

3. **One featurisation module, shared by training and runtime.** Overlap features computed slightly differently in collate and in the runtime would silently cost accuracy. `grounder/features.py` is pure Python + numpy, no torch, and both `dataset.py` and `runtime.py` call it.

4. **Queries are structured, encoded in one place.** Examples store `mode`, `query`, `history` as separate fields; `records.format_query()` is the only code that turns them into the model string (`[DO] …` / `[GOAL] … [DONE] … ; …`). No hand-built `"do: "` strings anywhere else.

5. **Split by registrable domain (eTLD+1), and dedupe templates.** `shop.example.com` and `www.example.com` must land in the same split. Pages whose sorted element-text set is ≥ 90% identical to an earlier page are dropped, so one CMS theme can't leak between train and test.

6. **Own attention, not `nn.TransformerEncoder`.** Its fast path and nested tensors make ONNX export with dynamic N fragile. Explicit pre-norm layers with masked matmul–softmax attention export cleanly and are as fast at this size.

7. **A rule-based template teacher exists alongside the LLM teacher.** It writes `plain` and `context` instructions from templates ("click {text}", "open {text}", "the {text} button next to {context}"). Uses: the whole pipeline runs end to end on a laptop with no GPU (crawl → examples → train → export → runtime), tests have real data, and it adds cheap volume. The LLM teacher stays the source of `paraphrase` and `goal`.

8. **The LLM teacher talks to any OpenAI-compatible endpoint** (`--base-url`, `--model`), with a recorded-response mock in tests. Same code on the rented 4090 (vLLM) and anywhere else.

9. **Click safety also checks the element, not just its words.** Block auto-click when the element is `input[type=submit]`, a `button` inside a `form` whose submit it triggers, or its text/context matches the blocked list. Returns candidates instead.

10. **Python 3.12 pinned via uv** (machine default is 3.14, which PyTorch wheels lag on).

## Doubts to settle with data, not now

- **Mean-pooled element text loses word order** ("sign in" vs "in sign" is fine; "not now" vs "now not" isn't an issue in UI text). Keep it; revisit only if failure reading shows order mattering.
- **Goal mode at 4M params** may be weak on Mind2Web. The plan already reports it honestly.
- **Template teacher bias:** its instructions share words with the label by construction, so it must never be the only data the hand-written set is compared against. Report `paraphrase` separately, as planned.

## Build phases (this repo)

| Phase | Owner | Content |
|---|---|---|
| A | agent | Extractor (`dom.js`, `web.py`), fixtures, tests |
| B | agent | Tokenizer, features, model, dataset, train, eval, BM25, export, runtime |
| C | agent | Crawl, template teacher, LLM teacher + filters, end-to-end smoke run |
| D | agent | Playwright helper, label tool, bench, Mind2Web converter, TinyBERT baseline, `axdump` |

GPU-dependent steps (LLM teacher at scale, full training, size sweep, agent demo) are run later; the code for them is built and tested here on small data.
