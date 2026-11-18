# Tiny UI grounder — implementation plan

Implements `build-plan.md`. V1 is specified in full; V1.5 and V2 are outlined at the end. Hyperparameters and time estimates are starting points, not measured values. `tiny-grounder` is a working name.

## Stack

- Python 3.12, `uv`, PyTorch, Hugging Face `tokenizers` and `datasets`
- Playwright (Python) for crawling and the runtime helper
- vLLM (OpenAI-compatible server) for the teacher, on one rented RTX 4090
- `onnx` + `onnxruntime` for export, int8 quantisation and inference
- `rank_bm25` and `sentence-transformers` for baselines
- Swift (small CLI) for the macOS accessibility extractor

Crawling, extraction, labelling and the agent demo run on your Mac. The teacher and training run on the rented GPU.

## Repo layout

```
tiny-grounder/
  pyproject.toml
  configs/            xs.yaml  s.yaml  m.yaml  l.yaml
  grounder/
    records.py        record schema, role vocabulary, normalisation
    extract/
      dom.js          in-page extractor
      web.py          Playwright wrapper around dom.js
      macos/          Swift CLI: axdump
    data/
      crawl.py        site list → pages/*.json.gz
      teacher.py      pages → raw instructions (vLLM)
      filters.py      validation of teacher output
      mind2web.py     Mind2Web → our records + examples
      dataset.py      torch Dataset, collate, augmentation
    tokenizer.py      train / load BPE
    model.py
    train.py
    eval.py
    baselines/        bm25.py  tinybert.py
    export.py         ONNX + int8 + parity check
    runtime.py        Grounder class (onnxruntime + tokenizers + numpy only)
    playwright_helper.py   extract / find / click
  tools/
    label.py          hand-labelling tool
    bench.py          CPU latency benchmark
  demo/
    agent_demo.py  tasks.yaml
  tests/
```

## Data formats

**Element record** (one per element, every source):

```json
{"id": 17, "role": "button", "text": "Add to cart", "context": "Blue cotton shirt · $24",
 "tag": "button", "box": [0.62, 0.41, 0.09, 0.03], "abs_box": [794, 328, 115, 24],
 "in_viewport": true, "enabled": true, "conf": 1.0, "source": "dom"}
```

- `box` is normalised to the document (0–1); `abs_box` is in pixels for clicking.
- `source` ∈ `dom | ax | ocr | det`. `conf` is 1.0 for tree sources.
- `role` comes from a fixed vocabulary of ~16: `button, link, textbox, searchbox, checkbox, radio, switch, combobox, option, tab, menuitem, slider, image, text, other, unknown`.

**Page file** (`pages/{site}/{page_id}.json.gz`): `{page_id, url, site, title, viewport, doc_size, elements[]}`

**Example** (JSONL):

```json
{"page_id": "...", "kind": "plain|paraphrase|context|goal|null",
 "query": "log into my account", "history": [], "target_id": 17, "equiv_ids": [17, 203]}
```

- `equiv_ids`: elements with the same role, text and link target. All count as correct in the loss and in evaluation.
- `kind: null` has `target_id: null` (target deleted from the page).

Query strings carry a mode prefix: `do: <instruction>` or `goal: <goal> | done: <last 3 actions>`.

---

## Step 1 — Extractor (`extract/dom.js`, `extract/web.py`)

**Candidates:** `a[href]`, `button`, `input` (not hidden), `select`, `textarea`, `summary`, `[contenteditable]`, elements with an interactive ARIA role, `[onclick]`, `[tabindex]` ≥ 0, and leaf elements with `cursor: pointer`.

**Keep only if:** non-zero box, not `display:none` / `visibility:hidden` / opacity 0, and (when in the viewport) `elementFromPoint` at its centre returns it or a descendant.

**Text:** first non-empty of `aria-label`, `aria-labelledby` target, `<label for>`, `alt`, `title`, `placeholder`, `value` (buttons), trimmed `innerText` (max 80 chars).

**Context** (max 120 chars, own text removed):
- inputs: label and `fieldset > legend`
- otherwise: text of the nearest "row" ancestor (`li`, `tr`, `article`, `[role=listitem]`, card-like `div` with ≤ 300 chars) + the nearest preceding heading inside the closest `section` / `form` / landmark

**Other rules:**
- Nested interactive elements: keep the outermost unless the inner one has a different link target.
- Walk open shadow roots and same-origin iframes (offset boxes by the frame position).
- Drop hashed `class` / `id` values; they are never used as text.

**Done when:**
- Unit tests pass on 10 saved HTML fixtures, including the antiagent fixtures (native select, isolated frame, open shadow root).
- On 20 real pages checked by eye, nothing you would want to click is missing.

**Known gap for v1:** closed shadow roots and cross-origin iframes. Fix later by reading through CDP.

## Step 2 — Crawl (`data/crawl.py`)

- Site list: top ~5k of the Tranco list; skip sites that disallow crawling in `robots.txt`.
- Per site: homepage + up to 2 random same-site links. Viewport 1280×800, 15 s timeout, 8 concurrent tabs.
- Save the page file. For held-out sites also save a full-page screenshot (used by the labelling tool).
- Split by site with a fixed hash: 90% train / 5% dev / 5% test.

**Done when:** ≥ 6k pages from ≥ 3k sites, median 50–400 elements per page, and a printed histogram of roles looks sane.

## Step 3 — Tokenizer (`tokenizer.py`)

- BPE, 8k vocab, lowercase, trained on all element text + context from the train split, plus the teacher queries once they exist (retrain after step 4).
- Special tokens: `[PAD] [UNK] [DO] [GOAL] [DONE]`.

**Done when:** average tokens per element text ≤ 6 and unknown-token rate on the dev split < 0.5%.

## Step 4 — Teacher data (`data/teacher.py`, `data/filters.py`)

**Setup:** vLLM serving an open instruct model that fits in 24 GB. Pick the best current one at build time; the model name is a config value. Use JSON-constrained output.

**Sampling ~20 elements per page:** favour in-viewport elements and elements with text; down-weight footer links; always include some icon-only elements (text only from `aria-label`) and some duplicated labels.

**Prompt (10 elements per call):** page title and host, then for each element: role, text, context, the text of 3 neighbours, and how many other elements share its text. Ask for:

```json
{"plain": "...", "paraphrase": "...", "context": "... or null", "goal": "..."}
```

- `plain`: a short natural instruction.
- `paraphrase`: must not reuse the element's words.
- `context`: only when the label is duplicated; must use the context to single it out.
- `goal`: what a user wants such that this element is the next step.

**Filters:**
- 2–20 words; valid JSON; no element ids or coordinates in the text.
- `paraphrase` shares no content word (stop words removed, stemmed) with the element text.
- `plain` on a duplicated label is dropped unless all duplicates are in `equiv_ids`.
- Optional if time allows: round-trip check — the teacher must pick the same element from 10 candidates given only the instruction.

**NULL examples:** copy 10% of examples, delete the target and its equivalents from the page, set `target_id: null`.

**Done when:** ~300k examples after filtering; 100 random examples read by hand, with ≥ 90% judged correct and natural.

## Step 5 — Model (`model.py`)

**Inputs** (B pages, N elements):

| Tensor | Shape | Notes |
|---|---|---|
| `query_ids` | B × ≤48 | with mode prefix |
| `el_text_ids` | B × N × ≤16 | |
| `el_ctx_ids` | B × N × ≤24 | |
| `el_role` | B × N | |
| `el_feats` | B × N × 9 | x, y, w, h, in_viewport, enabled, conf, overlap(query, text), overlap(query, ctx) |
| `el_mask` | B × N | |

**Forward:**

```
q   = E(query_ids) + pos + type_q                       # B × Lq × d
el  = mean(E(el_text_ids)) + Wc · mean(E(el_ctx_ids))
      + R(el_role) + MLP(el_feats) + type_e             # B × N × d
seq = [q ; NULL ; el]                                    # no positions on elements
h   = 4 × pre-norm transformer layer(seq)
scores = Linear(h[NULL and elements]) → B × (N+1)
```

- Loss: `-log Σ p(positives)` over `equiv_ids`; the NULL slot is the positive for null examples.
- `overlap` = share of the query's content tokens that appear in the element's text / context. Computed in the collate function and identically in the runtime.
- Sizes come from `configs/*.yaml`: XS 128/2, S 192/3, M 256/4, L 384/4 (width / layers), FFN 2×, 4 heads.

**Done when:** a shape test passes, parameter counts print as expected (M ≈ 4.2M), and the model overfits 100 examples to ~100% top-1.

## Step 6 — Training (`data/dataset.py`, `train.py`)

- AdamW, lr 1e-3, 1k warmup steps, cosine decay, weight decay 0.01, dropout 0.1, batch 64 pages bucketed by N, mixed precision, ~10 epochs.
- Pages with more than 512 elements: keep the target and sample the rest.
- **Augmentation:** shuffle element order; drop role (p 0.15); drop context (p 0.15); jitter boxes; OCR-style text noise (p 0.1) with `conf` lowered to match.
- Log dev top-1 / top-5 per `kind` every epoch; keep the best checkpoint by dev top-1 on `paraphrase`.

**Done when:** the M model finishes a run and beats BM25 on dev `paraphrase`.

## Step 7 — Evaluation and baselines (`eval.py`, `baselines/`)

- **Metrics:** top-1, top-5, top-10, top-20, MRR, each per `kind`; for NULL, precision/recall of abstaining across thresholds.
- **BM25:** rank elements by BM25 of the query against `text + context`.
- **TinyBERT-L2:** fine-tune `cross-encoder/ms-marco-TinyBERT-L2-v2` on `(query, "role | text | context")`, 1 positive + 7 negatives per example; evaluate on a 2k-example subsample (it scores one element at a time, so it is slow).
- **Hand-written set (`tools/label.py`):** generates a local HTML page showing a held-out screenshot with numbered boxes; you type an instruction and the element number; it appends to `hand_test.jsonl`. Target: 200 instructions on 20 pages.

**Done when:** one table with our model, BM25 and TinyBERT-L2 on test-site and hand-written sets.

## Step 8 — Size sweep

Train XS, S, M, L with the same data and seed. Record hand-set top-1, test `paraphrase` top-1, params, and CPU latency.

**Rule:** ship the smallest within 1 point of the best on the hand-written set.

## Step 9 — Export and runtime (`export.py`, `runtime.py`, `tools/bench.py`)

- Export to ONNX with dynamic N and query length; int8 dynamic quantisation.
- **Parity:** int8 top-1 within 0.5 points of the PyTorch model on the test set; otherwise ship fp16 or quantise only the linear layers.
- `runtime.Grounder(model_dir).rank(query, elements, k)` → `[{id, score, abs_box}]`, plus `null_prob`. Dependencies: `onnxruntime`, `tokenizers`, `numpy`.
- `bench.py`: p50 / p95 for N = 100, 250, 500, 1000 on your laptop CPU, 4 threads.

**Done when:** model file size and latency are recorded, and a runtime-vs-PyTorch test passes.

## Step 10 — Playwright helper (`playwright_helper.py`)

- `extract(page)` → records
- `find(page, query, k=5)` → top elements with coordinates and confidence
- `click(page, instruction)` → clicks only if the top score clears a threshold tuned on dev for ≥ 98% precision **and** the element's text doesn't match a blocked list (submit, pay, buy, order, delete, remove, send, confirm). Otherwise returns the candidates to the caller.

## Step 11 — Agent demo (`demo/`)

- ~20 tasks in `tasks.yaml` on stable public sites, each with a programmatic success check (URL or page-text condition).
- One planner model through its API with tool use, run three ways:
  1. `snapshot()` returns the full element list
  2. `snapshot()` returns the top 20 from goal mode
  3. `find(query)` + `click(id)` / `type(id, text)`
- Record per task: success, input and output tokens (from the API's usage fields), wall-clock time, number of model calls. Run each task 3 times per configuration.

**Done when:** a results table and the raw logs are saved.

## Step 12 — Goal mode on Mind2Web (`data/mind2web.py`)

- Source: `osunlp/Multimodal-Mind2Web` (train, test_task, test_website, test_domain).
- Per action: elements = positive + negative candidates. Text and context come from the cleaned HTML by node id, using the same rules as step 1. Box from the candidate's bounding-box attribute.
- Query: `goal: {task} | done: {last 3 action descriptions}`.
- Add the train split to the training mix and retrain. Evaluate on all candidates per page (no 512 cap); any positive candidate counts.

**Done when:** top-1 / top-10 / top-20 on the three test splits are in the results table.

## Step 13 — macOS extractor (`extract/macos/`)

- Swift CLI `axdump --pid <pid>` → JSON records. Needs the Accessibility permission. Start from Veil's AX traversal code.
- Role map: `AXButton→button`, `AXLink→link`, `AXTextField/AXTextArea→textbox`, `AXSearchField→searchbox`, `AXCheckBox→checkbox`, `AXRadioButton→radio`, `AXPopUpButton/AXComboBox→combobox`, `AXMenuItem→menuitem`, `AXSlider→slider`, `AXStaticText→text`, `AXImage→image`.
- Text: first non-empty of `AXTitle`, `AXDescription`, `AXValue`, `AXHelp`. Context: window title + nearest group title + sibling static text. Box from `AXPosition` / `AXSize`.
- Dump ~30 apps (a few windows each), run through the teacher, add to training, retrain.
- Hand-write 50 instructions on apps not in training.

**Done when:** desktop hand-set numbers are in the table and `find` works on a live app.

## Step 14 — Release

README with the results table, what it doesn't do (pixels, non-English, logged-in apps in training), model card with the confidence-vs-accuracy curve, weights, dataset, and training recipe.

---

## Order and timing

| Day | Steps |
|---|---|
| 1 | 1 Extractor → 2 Crawl (runs in background) → 3 Tokenizer → 5 Model → 4 Teacher data → 6 First training run |
| 2 | 7 Evaluation, baselines, hand-written set → fix data, retrain → 8 Size sweep → 9 Export → 10 Helper → 11 Agent demo |
| 3 | 12 Mind2Web → 13 macOS → retrain → 14 Release |

**Decision points:**
- After step 6: if the model doesn't beat BM25 on `paraphrase`, look at the data (read 100 failures) before touching the architecture.
- After step 7: if the hand-written set is far below the synthetic test set, make the teacher's instructions more varied and regenerate.
- After step 11: if success drops in configuration 2 or 3, check whether the right element was in the returned list. That separates a ranking problem from a protocol problem.

---

## Later stages (outline)

**V1.5 — system OCR fallback**
1. `extract/ocr_macos` (Apple Vision) → one record per text line: `role: unknown`, `source: ocr`, `conf` = OCR confidence.
2. `find_on_screenshot(image, query)` using the same ranker.
3. Evaluate on ScreenSpot's text targets; record ms per screenshot.

**V2 — detector + icons**
1. Download ScreenParse; map its 55 classes to five (clickable, input, text, icon, image).
2. Train a nano detector; compare against the full ScreenParser model on a held-out slice. Check the training code's licence first.
3. Merge step: OCR text inside a detected box becomes that element's text; other text lines stay as candidates.
4. Icon classifier on rendered open icon sets; pass the top 3 guesses and probabilities as extra element features (requires a small ranker retrain).
5. Evaluate the full pipeline on ScreenSpot per platform, text vs. icon targets; record total params and ms per screenshot.
