# Tiny UI grounder — build plan

**Goal:** the smallest, fastest on-device model that finds the right UI element, without giving up accuracy. English only for now.

Same mindset as tiny-voca: one narrow job, one open teacher, synthetic data, small models trained from scratch, one rented GPU, everything open. Background research is in `tiny-ui-grounding-plan.md`; this file is what we build.

## The shape of the system

```
Accessibility tree / DOM ──► extractor ─────────────┐   (fast path, default)
                                                     ▼
                                      unified element records
                          role · text · context · box · confidence
                                                     ▲   │
Screenshot ──► OCR + element detector ───────────────┘   ▼
               (fallback, v1.5 / v2)              ~4M ranker
                                                         │
                              instruction or goal ──► ranked elements + screen coordinates
```

One ranker serves every input source. Vision is only used when the tree is missing or incomplete.

## Roadmap

| Stage | Deliverable | Adds |
|---|---|---|
| **V1** | Structured ranker: browser + macOS apps, instruction mode + goal mode | ~4M params |
| **V1.5** | Screenshot fallback using the operating system's built-in OCR; each text line is a candidate | no new model |
| **V2** | Nano element detector + small icon classifier | ~5M params (target) |
| **V2 eval** | Screenshot-only grounding on ScreenSpot | — |

V1 gets finished and measured before any screenshot work starts.

---

## V1 — the ranker

### The job

**Input:** a query + the page's element records. **Output:** elements ranked by score, a confidence, and each element's screen coordinates.

The query is one of two things, handled by the same model:

- **An instruction** ("open the cart") → the element to act on.
- **A goal** ("buy running shoes"), optionally with the last few actions → the ~20 elements most likely needed next. An agent can use this without changing its loop.

Screenshot-based computer-use agents benefit even in V1: they ask "where is the Save button?" and click the returned coordinates instead of guessing pixels.

### The model

```
[query tokens] + [one vector per element] + [NULL slot]
        → 4 transformer layers (d=256, 4 heads, FFN 2×)
        → one score per element (softmax over the page)
```

- **Tokenizer:** our own BPE, 8k vocab, trained on the crawled UI text.
- **Element vector** = mean token embedding of its text + mean token embedding of its context (nearest heading / row / label) + role embedding + a tiny MLP over box position and size + a source-confidence value + one word-overlap number against the query.
- **NULL slot** wins when the target isn't on the page; its probability is the abstain signal.
- **Loss:** cross-entropy over the page's elements.
- **Size:** embeddings 8k × 256 = 2.1M, layers ≈ 2.1M → **~4.2M params, ~4 MB int8**.
- **Speed (est., to be measured):** up to 512 elements in one pass, roughly 10–30 ms on a laptop CPU.

### Finding the smallest model that doesn't lose accuracy

Each model trains in about an hour, so run a sweep instead of guessing:

| Variant | Width / layers | Params (approx.) |
|---|---|---|
| XS | 128 / 2 | ~1.2M |
| S | 192 / 3 | ~2.2M |
| **M (default)** | 256 / 4 | ~4.2M |
| L | 384 / 4 | ~7.5M |

**Rule:** ship the smallest variant within 1 point of the best on the hand-written test set. L exists only to show where the curve flattens.

### The extractor

One JS snippet run through Playwright, plus a macOS accessibility-tree reader (starting from Veil's Swift AX code), both producing the same record:

`role, text, context, x, y, w, h, enabled, confidence, source`

Visible, interactable elements only. Walks open shadow roots and same-origin iframes. macOS roles are mapped to the web role vocabulary. Tested on the antiagent fixtures. This piece deserves the most care: the model can't pick what the extractor drops.

### Data (all synthetic except Mind2Web)

1. **Crawl** ~6k pages from ~3k sites with Playwright (~1 hour). Split **by site**: 90% train, 10% held out.
2. **Teacher:** an open instruct model on one rented RTX 4090 (vLLM). For ~20 sampled elements per page it writes:
   - a plain instruction ("click Sign in")
   - an instruction that may **not** reuse the element's own words ("log into my account")
   - for duplicated labels, one that uses the context ("add the blue shirt to the cart")
   - a **goal** for which this element is the next step ("you want to buy running shoes" → the search box)
3. **NULL examples:** for ~10%, delete the target from the page.
4. **Real data for goal mode:** Mind2Web's training split (~7.8k human actions), converted to our records.
5. **Desktop:** accessibility trees of ~30 macOS apps, run through the same teacher.
6. **Noise, so the same ranker works on screenshots later:** during training, randomly drop role and context, jitter boxes, add OCR-style text errors, and lower the confidence value accordingly.

Target: ~300k examples, a few GPU hours of teacher time (est.).

### Evaluation

- **Held-out sites:** top-1 and top-5, with the "no shared words" instructions reported separately.
- **200 instructions written by hand** on ~20 unseen pages, plus 50 on unseen macOS apps. The numbers to trust.
- **Baselines on the same data:** BM25, and a fine-tuned TinyBERT-L2 (4.4M, pretrained).
- **Goal mode:** Mind2Web unseen-website and unseen-domain splits, top-10 and top-20. For reference, an 86M pretrained ranker gets about 93% top-10. Report whatever we get.
- **Speed and size:** ms per page on a laptop CPU; file size after int8.
- **Agent demo:** one real agent on ~20 browser tasks three ways (full page / top-20 from goal mode / `find` tool): task success, input tokens, time per task.

**Ship if:** it clearly beats BM25 on the "no shared words" set, is close to TinyBERT-L2 in accuracy while much faster per page, and the agent demo keeps its success rate with far fewer tokens.

### Using it

- `find(query) → top elements + confidence + coordinates`
- `click(instruction)` acts directly when confidence is high. Never auto-acts on submit / pay / delete.
- Ships as an ONNX file + a small Playwright helper. MCP wrapper if time allows.

### Schedule

| When | What |
|---|---|
| Day 1 morning | Extractor + start the crawl |
| Day 1 afternoon | Teacher generation; tokenizer; model + training loop |
| Day 1 evening | First training run; baselines |
| Day 2 morning | Hand-written test set; fix data; retrain; size sweep |
| Day 2 afternoon | int8 ONNX, speed test, Playwright helper, agent demo |
| Day 3 | Mind2Web conversion + goal-mode numbers; macOS extractor + desktop data; retrain; release |

**Cost (est.):** ~10–20 GPU hours on a 4090 (about $5–10), plus a few dollars of API calls for the agent demo.

---

## V1.5 — screenshot fallback with system OCR

- Use the OS text reader (Apple Vision on macOS, Windows OCR on Windows). No model to ship.
- **Each recognised text line becomes a candidate record**: text, box, OCR confidence, role unknown. Clicking the centre of the word "Save" hits the Save button, so no detector is needed yet.
- The ranker already handles these records because of the training noise in V1.
- **Covers:** any target with visible text. **Doesn't cover:** icon-only buttons, unlabelled inputs.
- **Measure:** ScreenSpot text-target subset; ms per screenshot.

About half a day.

## V2 — nano detector + icon classifier

- **Detector:** a nano-size detector (~3M params) trained on the **ScreenParse** dataset (1.45M web screenshots, 25.6M labelled elements, CC-BY-4.0). Collapse its 55 classes to about five: clickable, input, text, icon, image.
  - The existing ScreenParser model is YOLO11-Large at 1280px: too big to ship, but useful as a benchmark and as a teacher.
  - Check the licence of the detector training code before building on it.
- **Joining OCR and boxes:** text inside a detected box becomes that element's text; text outside any box stays its own candidate.
- **Icon classifier:** ~1–2.5M params on small crops, trained on rendered open icon sets (Material, Font Awesome, Lucide). Its top few guesses and their probabilities go to the ranker as uncertain features; context decides whether a magnifying glass means search or zoom.
- **Targets (hypotheses, to be measured):** ~10M params total across the pipeline; 0.3–1 s per screenshot on a laptop CPU.

### V2 evaluation

- **ScreenSpot** (screenshots + human instructions across web, macOS, Windows, Android, iOS), reported per platform and separately for text vs. icon targets.
- Compare against the full ScreenParser detector in the same pipeline, to see what the nano detector costs in accuracy.
- Expect to land well below the 2B+ vision-language models, especially on icons and on desktop (ScreenParse is web-only). The claim to make is accuracy per megabyte and per millisecond on a CPU.

---

## Risks

- **From-scratch semantics.** The ranker only knows the synonyms the teacher wrote. If the hand-written set is much worse than the synthetic one, generate more varied instructions before changing the architecture.
- **Goal mode at this size.** It may be weak on Mind2Web. Instruction mode is the dependable one; goal mode is reported honestly either way.
- **Crawl bias.** Public pages only, no logged-in apps. Say so in the README.
- **Web-to-desktop gap** in both the ranker's training data and the V2 detector's.
- **If time slips:** cut the MCP wrapper and the L variant, not the hand-written test sets.
