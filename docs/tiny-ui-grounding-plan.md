# Tiny UI Grounding Model — Research & Plan

*Research date: 2026-11-16. Numbers come from the cited sources; anything marked **(est.)** is my estimate.*

---

## 0. TL;DR

- **The gap is real.** Today's element-grounding models for web agents range from **86M** (DeBERTa rankers) to **322M** (laya-browser) to **1.7B–27B** (OpenJev / Jev). Nothing credible exists at **≤10M parameters that runs in-process on a CPU, or inside a browser via WASM**. That is the niche.
- **Build it as a ranker with an abstain option, not a picker.** The model returns the top-k elements, a calibrated confidence, and an "abstain / not on page" signal. The planner sees only those k elements instead of the full page. When confidence is very high, the action runs without calling the LLM at all.
- **The architectural bet: a listwise "set" model, not a pointwise cross-encoder.** Encode each element cheaply, then let all elements attend to each other and to the query in a few small transformer layers. This is about 50–100× cheaper than running a cross-encoder once per element, and it is the only way a tiny model can tell apart 20 identical "Add to cart" buttons.
- **"From scratch" holds for the weights. The knowledge comes from distillation and free web supervision**: link text → destination page title, `<label for>` → input, icon class name → aria-label. With an open-weight teacher on one rented GPU, the whole project should cost **about $10–30** (tiny-voca, a 4.5M TTS trained on synthetic data, cost $9.56).
- **Be honest about the economics.** Haiku 5.5 costs **$0.10/MTok** input, so "replace a cheap-LLM grounding call" is a weak pitch. The strong pitch is **removing 5–10k tokens per step from the *frontier planner's* context** (Opus 5.5 is $4/MTok, Sonnet 5.5 is $2/MTok), plus latency (<20 ms local vs. a network round-trip), privacy, offline use, and determinism.

---

## 1. Landscape (what already exists)

| Work | Size | What it does | Key numbers |
|---|---|---|---|
| **MindAct** (Mind2Web, NeurIPS'23) | DeBERTa-v3-base, 86M | Pointwise cross-encoder ranks DOM elements; an LLM picks from the top-50 | Recall@50 ≈ 85%. Pages average **1,135 elements** (580 after heuristic filtering, keeping 94.7% of targets) |
| **mind2web-candidate-ranker** (HF, 2026) | 86M | Same recipe, re-trained | Cross-domain: **acc@1 55.4%, R@5 87.1%, R@10 92.8%**. Cross-website: acc@1 50.9%, R@10 92.6%. Recall plateaus at k≈20 |
| **WebLINX DMR** (ICML'24) | MiniLM bi-encoder (~22M) | Dense markup ranker | R@10 74.3 in-domain, **~52 out-of-domain**. About 5× faster than the MindAct cross-encoder (186 ms vs 916 ms per turn) |
| **Prune4Web** (AAAI'26) | 0.5B–3B | LLM writes Python scoring programs that prune the DOM | Grounding accuracy 46.8% → 88.3%. Candidates cut 25–50×. Recall >90% @3, 97.8% @20 |
| **Filtering paper** (arXiv 2609.27770) | DeBERTa / T5 / ColBERT | Filters the accessibility tree before the LLM | ~81% R@10 on Mind2Web. **Caution:** aggressive filtering *lowered* GPT-3.5's WebArena success (6.65 → 5.84%) |
| **laya-browser** (HF, Sep 2026) | **322M** (mmBERT-based) | "System-1" browser decision model | Trained on Mind2Web, nnetnav-live, webchain |
| **Jev / OpenJev** (browser-use + TypeSafe, 2026) | hosted; open model **27B** (community 1.7B/4B/8B) | Picks the next operation and target from an element table; a small LLM only generates typed text | 87.4% next-action accuracy on unseen sites. ~210 ms per decision on an H100 |
| **OmniParser v2** (Microsoft) | YOLO detector + Florence-2 captioner | Screenshot → element list for pixel-only agents | 0.6 s/frame on an A100. 39.6 on ScreenSpot-Pro |
| **Stagehand benchmark** (2026) | — | Compares page representations | Accessibility-tree tokens: 134 (login) / 6.5k (listing) / **80k** (500-row table). Indexed interactive DOM reaches **97% act accuracy**. Screenshot is ~1.3k tokens flat |
| **MDPI 2026: cost-efficient observations** | — | Which element attributes to serialize | Dropping volatile `class`/`id` values: **+9.5 pp** success. Best profile: **+10 pp with 34% fewer tokens** |

**Size reference points for a ~5M budget**
- `cross-encoder/ms-marco-TinyBERT-L2-v2`: **4.39M** params (BERT-Tiny, 2 layers, d=128). This is your **size-matched baseline**.
- `cross-encoder/ms-marco-MiniLM-L2-v2`: 15.6M.
- `minishlab/potion-base-2M` / `potion-base-8M` (static embeddings): **1.9M / 7.6M**. Encoding takes microseconds on a CPU. Good as a zero-shot baseline and as a prefilter.

**What follows from this:** a 5M model has to beat a fine-tuned TinyBERT-L2 (same size) and approach the 86M DeBERTa (R@10 ≈ 93% cross-domain). The project is worth doing if it hits **R@10 ≥ 88–90% at ≤5M** with **~10–20 ms CPU latency**.

---

## 2. Where the savings actually come from

### 2.1 Token math (est.)
- A typical serialized interactive element costs about 10–20 tokens. A Mind2Web-like page has ~580 candidates after heuristics, so **~6k–12k tokens per observation** (Stagehand measured 4–7.5k for a product listing, 80k for big tables).
- With the ranker, the planner gets the **top-10 elements (~200 tokens) plus a compact page outline (~300 tokens)**, which saves **~5k–11k input tokens per step**.

| Planner | Input $/MTok | Saved per step (8k tok) | Per 25-step task | Per 10k tasks |
|---|---|---|---|---|
| Fable 5.2 | $8 | $0.064 | $1.60 | $16,000 |
| Opus 5.5 | $4 | $0.032 | $0.80 | $8,000 |
| Sonnet 5.5 | $2 | $0.016 | $0.40 | $4,000 |
| Haiku 5.5 | $0.10 | $0.0008 | $0.02 | $200 |

Agents that keep the last N observations in their history multiply these savings by N. Prompt caching does **not** rescue full-tree observations, because the page changes every step.

### 2.2 Honest caveats
- **Cheap LLMs are already cheap.** Compared with a Haiku-class grounder, the win is latency (~10 ms vs. hundreds of ms), offline and private use, rate limits, and determinism, not dollars.
- **Pixel-based computer use is a different case.** A screenshot is ~1.3k tokens flat, so the token savings for pixel agents are smaller. The value there is reliability: return an element reference instead of guessing pixel coordinates, which means fewer misclicks and retries. That requires an element tree (AX API / UIA / AT-SPI, or OmniParser-style detection).
- **Filtering can hurt end-to-end** (see the filtering paper). Mitigations: a recall-oriented k, an abstain path, and always measuring task success, not only ranking metrics.

---

## 3. System design

```
               ┌──────────── frontier planner (LLM) ────────────┐
               │ sees: task, history, page outline, top-k elems │
               │ emits: subgoal / low-level intent              │
               └───────────────▲───────────────┬────────────────┘
                               │ top-k + conf   │ intent ("click Add to cart for blue shirt")
┌──────────┐   element list   ┌┴───────────────▼┐   conf ≥ τ_auto    ┌──────────┐
│ extractor│ ───────────────► │  tiny ranker    │ ─────────────────► │ executor │
│ CDP/AX   │  (≤5k elements)  │  ~5M, CPU/WASM  │  (skip LLM call)   │Playwright│
└──────────┘                  └─────────────────┘                    └──────────┘
                                abstain / not-on-page → planner gets full tree or scrolls
```

**Three operating modes, all from one model**
1. **Observation filter:** each step, rank by `task + subgoal + history` and show the planner the top-k. Biggest token savings.
2. **Intent resolver:** the planner says *what* to do in words, and the ranker resolves *which* element. If confidence ≥ τ_auto, act without another LLM round-trip.
3. **Abstain:** a "target not present" head tells the agent to scroll, open a menu, or fall back to the full tree.

**The extractor matters as much as the model.** Stagehand found every representation loses something: iframes, closed shadow roots, single-character pagination links, `<select>` state. Use CDP `DOMSnapshot` + `Accessibility.getFullAXTree`, walk frames and shadow roots, keep form state, and drop volatile `class`/`id` hashes (that alone gave +9.5 pp in the MDPI study). A ranker cannot find an element the extractor dropped, so measure **extractor recall** separately.

**Element record (model input):**
`role, tag, accessible name, visible text (trimmed), aria-*/placeholder/title/alt/value, label text, nearest heading, parent/landmark role, icon class tokens, href path tokens, bbox (normalized), viewport visibility, DOM depth, sibling index, interactable/enabled/checked flags`.

---

## 4. Model architecture (~5M params)

### 4.1 Why listwise
- **Pointwise cross-encoder:** one forward pass per (query, element) pair. With a ~3M non-embedding model and ~48 tokens per pair, that is ~290 MFLOP per element and **~150 GFLOP for 500 elements, roughly 0.5–1.5 s on CPU**. Too slow, and each element is scored blind to its neighbors.
- **Listwise set model:** encode each element cheaply once, then run a few transformer layers over the set of ~256 element vectors plus the query tokens. That is about 1–2 GFLOP, **~10–20 ms on a laptop CPU** (est.). Elements see each other, so duplicates, "the second result", and "price next to X" become learnable.

### 4.2 Proposed architecture: "SetRanker-5M"

| Block | Design | Params (est.) |
|---|---|---|
| Tokenizer | **UI-domain BPE, 8k vocab**, trained on crawled UI strings plus a byte fallback. UI text is narrow, and a small vocab frees up budget | — |
| Token embeddings | 8192 × 256, tied, int8 at inference | 2.10M |
| Query encoder | 2 transformer layers, d=256, 4 heads, FFN 2×, ≤64 tokens (task + subgoal + last 2 actions) | ~1.05M |
| Element encoder | Per field: mean+max pool of token embeddings ⊕ field-type embedding; role embedding (~80 ARIA roles); numeric features (bbox, depth, flags) via MLP; **lexical-match features** (BM25 / exact / fuzzy overlap with the query per field) → 256-d | ~0.35M |
| Listwise interaction | 2 blocks × (element self-attention with **structural bias**: tree distance and spatial adjacency buckets + cross-attention to query tokens + FFN 2×) | ~1.6M |
| Heads | score (per element), operation {click, type, select, hover}, **not-on-page** (via a learned [NULL] element), confidence calibration via temperature | ~0.05M |
| **Total** | | **~5.2M** (~5 MB int8) |

**Prefilter for huge pages:** interactable + visible heuristics, then lexical/static-embedding top-256 → listwise stage. Report the recall of this stage separately.

**Why lexical features:** small rankers benefit a lot from explicit match signals (classic learning-to-rank). They also make the model robust to proper nouns it has never seen, like product names and cities.

### 4.3 Scaling ladder (for the research question)
- **XS ~1.5M:** potion-style static embeddings + lexical features + MLP (no attention). Shows how far "no transformer" gets.
- **S ~5M:** the main model above.
- **M ~15M:** d=384, 4 listwise layers. Shows where returns diminish.

Plot R@10 and acc@1 against params and against CPU latency. The curve is a publishable result in itself.

---

## 5. Data plan

### 5.1 Human-labelled (supervised)
- **Mind2Web / Multimodal-Mind2Web:** 1,009 train tasks / 73 sites, ~7.8k actions. Official cross-task, cross-website, and cross-domain test splits (cross-domain: 694 tasks / 4,060 actions). The primary benchmark.
- **WebLINX:** multi-turn, real sites, with out-of-distribution splits.
- **AndroidControl / AITW:** accessibility trees on mobile, for cross-platform transfer ("computer use" beyond the browser).
- **nnetnav-live, webchain, GoBrowse:** extra trajectories (used by laya-browser).
- ⚠️ Verify each dataset's license before releasing weights.

### 5.2 Free self-supervision (pretraining, ~10M+ pairs)
Crawl ~50k sites (e.g., the Tranco top list) with Playwright, snapshot them, and mine pairs:
- link text / aria-label → **destination page `<title>`/`<h1>`** (teaches semantics like "Sign in" ≈ "Log in" ≈ "Account")
- `<label for>` / `aria-labelledby` → input
- icon class tokens (`fa-cart`, `icon-search`, svg `<title>`) ↔ accessible name (icon-only grounding)
- `button[type=submit]` ↔ form heading / action URL tokens
- translated page versions (`hreflang`) → multilingual alignment, at no labelling cost

Objective: contrastive (InfoNCE) query↔element with in-page hard negatives, plus masked-field prediction.

### 5.3 Teacher-generated instructions (distillation)
- Sample elements from crawled pages, and have a teacher LLM write instructions at **five levels**: literal ("click Checkout"), paraphrase ("proceed to payment"), goal-level ("buy what's in my bag"), relational ("the 2nd result's price"), and icon-only.
- **Anti-leakage rule:** for ≥50% of samples, the instruction may not reuse any content word from the element's own fields. Otherwise the model only learns string matching.
- Teacher scores for the top-20 candidates feed **listwise KL distillation**, giving soft labels and hard negatives.
- **Teacher = an open-weight LLM on one rented RTX 4090 (~$0.40/h) with vLLM, not an API.** Keep prompts short (element plus local context, ~300 tokens, not the whole page) and ask for instructions for ~10 elements per page in one call. Start with ~200k examples and scale only if the learning curve is still climbing. Cost (est.): **~$5–15**.

### 5.4 Deployment flywheel
Every time the ranker abstains and the planner picks, log `(query, page, chosen element)` (opt-in). This is free on-policy data for the next version.

---

## 6. Training recipe

1. **Tokenizer:** BPE 8k on crawled UI strings.
2. **Pretrain** (§5.2), contrastive + masked-field: ~10M pairs, a few hours on one consumer GPU (est.).
3. **Distill** (§5.3), listwise KL against teacher scores + CE on gold.
4. **Fine-tune** on Mind2Web + WebLINX (+ AndroidControl) with listwise softmax CE over the page, plus the [NULL] target for "not on page" (create these by deleting the gold element from the page).
5. **Calibrate** with temperature scaling on a held-out site set. Choose τ_auto for ≥98% precision.
6. **Quantize** to int8 (dynamic PTQ, then QAT if accuracy drops >1 pp), and export to ONNX → ONNX Runtime (Python/Node/Rust) and **ONNX Runtime Web/WASM** for browser extensions.

Augmentations: shuffle the element order, drop attributes (simulating poor accessibility), use obfuscated class names, perturb text, inject cookie banners and modals.

---

## 7. Evaluation protocol

### 7.1 Offline ranking
- Mind2Web test_task / test_website / **test_domain**: acc@1, R@{1,3,5,10,20}, MRR.
- WebLINX OOD splits. AndroidControl, for transfer.
- **Baselines, in this order:** BM25 + heuristics → potion-8M zero-shot → **fine-tuned TinyBERT-L2 (4.4M, size-matched)** → DMR-MiniLM → DeBERTa-86M ranker → Haiku 5.5 / Sonnet 5.5 choosing directly from the full list.
- **Slices:** icon-only, duplicate labels, relational instructions, non-English, forms, huge pages (>2k elements), **hostile pages** (closed shadow DOM, canvas-only UI, decoys/honeypots, withheld AX tree). On hostile pages the metric is whether the model correctly abstains.

### 7.2 Calibration
ECE, plus a **selective-accuracy curve** (accuracy vs. coverage). The key number: *% of steps that can auto-act at ≥98% precision.*

### 7.3 End-to-end (the result that matters)
**Online-Mind2Web** (300 tasks, 136 live sites), with a fixed frontier planner in three configurations:
1. Full tree (baseline)
2. Ranker top-k filter
3. Ranker filter + auto-act

Report success rate, **input tokens per task, $ per task**, wall-clock, LLM calls per task, and escalation rate. Also run WebArena (self-hosted) for reproducibility.

**Cheap version first:** measure token reduction offline for free (tokenize the full tree vs. top-k on Mind2Web pages). For success rate, run a 50-task pilot with Haiku 5.5 as the planner: 50 × 3 configs × 20 steps × 10k tokens ≈ 30M tokens ≈ **$3**, or $0 with an open planner on the rented GPU. Run the full 300-task frontier-planner eval (~$360 on Sonnet 5.5) only if you want a paper-grade headline number.

### 7.4 Systems
p50/p95 latency for 100, 500, and 2k elements on an Apple M-series chip, x86 AVX2, a Raspberry Pi 5, and WASM in Chrome. Also peak RAM and cold-start time.

---

## 8. Targets & kill criteria

| Metric | Target | Kill / pivot if |
|---|---|---|
| R@10, Mind2Web cross-domain | **≥ 90%** (DeBERTa-86M: 92.8%) | < 85% after distillation → move to the 15M tier |
| acc@1, cross-domain | ≥ 50% | — |
| Beats TinyBERT-L2 (size-matched) | +5 pp R@10 | Doesn't beat it → the listwise design isn't helping |
| CPU latency, 500 elements | p95 **< 30 ms** | > 100 ms |
| Auto-act coverage at 98% precision | ≥ 40% of steps | < 15% → filter mode only |
| End-to-end success vs. full tree | within **−2 pp** | worse than −5 pp |
| Planner input tokens | **−60% or more** | < −30% |
| Model file | ≤ 6 MB int8 | — |

---

## 9. Packaging (usability)

- `pip install` / `npm install` library: `rank(query, elements, k) → [{ref, score, op}], confidence, abstain`.
- **MCP server** exposing `find_elements(query, k)` and `resolve_and_act(intent)`, so any MCP-capable agent can use it without code changes.
- Adapters: Playwright (Python/TS), browser-use, Stagehand, and Chrome-extension WASM.
- Desktop: macOS AX / Windows UIA / Linux AT-SPI extractors feed the same model. Pixel-only apps (v2) pair with OmniParser's MIT-licensed detector plus OCR.
- Ship the model card with the selective-accuracy curve so users can pick τ_auto themselves.

---

## 10. Timeline (solo, ~12 weeks, est.)

| Weeks | Work | Exit gate |
|---|---|---|
| 1–2 | Extractor (CDP + AX, frames, shadow DOM; tested against antiagent fixtures, see §13), Mind2Web/WebLINX loaders, **all baselines** | Extractor recall ≥ 98% on Mind2Web **and** on the hostile-fixture suite; baseline table done |
| 3–4 | SetRanker v1 on supervised data; ablations: pointwise vs listwise, lexical features on/off, vocab 8k vs 30k | Beats TinyBERT-L2 |
| 5–7 | Crawl + self-supervised pretraining + teacher distillation; XS/S/M ladder | R@10 cross-domain ≥ 88% |
| 8–9 | NULL head, op head, calibration, int8/QAT, ONNX + WASM, latency suite | p95 < 30 ms; ≤ 6 MB |
| 10–12 | MCP + Playwright adapters; Online-Mind2Web 3-config eval; write-up and release | End-to-end targets in §8 |

**Compute:** training is trivial (a single consumer GPU or Apple Silicon; the 5M model trains in hours). **Cash (est.):** crawl ≈ free (your machine), teacher labels on a rented 4090 ≈ $5–15, training ≈ $1–5 (or free on a Mac), offline evals free, live pilot ≈ $3. **Total ≈ $10–30.** The optional full frontier-planner eval adds ~$360.

---

## 11. Risks

1. **Extractor gaps** dominate failures. Mitigation: measure extractor recall as its own metric.
2. **Synthetic instruction leakage** inflates scores. Mitigation: the anti-leak rule and testing on human-written splits only.
3. **Relational / multi-hop instructions** exceed a tiny model. Mitigation: abstain and let the planner handle them.
4. **Competition:** Jev/OpenJev and laya-browser target "fast decisions." Differentiation: **60× smaller than laya and >300× smaller than OpenJev-1.7B**, in-process, WASM, calibrated, open.
5. **Platform shift:** frontier pixel-native computer use improves. Mitigation: element references stay more reliable than coordinates, and accessibility trees stay cheaper than repeated screenshots on long tasks.
6. **Dataset licenses.** Verify before releasing weights.

---

## 12. First three things to do this week

1. Build the extractor and load Mind2Web. Measure **extractor recall** and **tokens per page**.
2. Run the baselines: BM25, potion-8M, and **fine-tune TinyBERT-L2 (4.4M)** on Mind2Web. That gives the bar a 5M model has to clear.
3. Prototype SetRanker with lexical features on Mind2Web only, and compare to step 2. If listwise doesn't beat pointwise at equal size by week 4, revisit the architecture before investing in the crawl.

---

## 13. Reusing your existing work (veil, antiagent)

Both projects study the *defense* side: what agents can see and do, and how to stop them. That maps directly onto this plan's biggest risk (the extractor) and onto its eval harness.

| From | What to reuse | Where it goes in this plan |
|---|---|---|
| **antiagent** route research (closed shadow DOM, native `<select>`, isolated frames, canvas UIs, password masking) | Its finding that closed shadow roots block role locators but **not** CDP/AX-level observation means: build the extractor on CDP `DOMSnapshot` + `getFullAXTree` with piercing, never on Playwright locators | §3 extractor design |
| **antiagent** `public/*.html` fixtures | An **extractor test suite**: each fixture has a known target, so extractor recall becomes a CI check | §10 weeks 1–2 gate |
| **antiagent** route-matrix benchmarks (`benchmark-*.mjs`: 3 reps per condition, JSON evidence, pilot vs final runs) | Same runner shape for the end-to-end configs (full tree / top-k / auto-act) | §7.3 |
| **antiagent** decoy gate + hidden/trap elements | **Hard negatives** and a "human-visible" feature set (opacity, off-screen, zero-size, `aria-hidden`, occluded). New eval slice: *decoys and honeypots* | §4.2 features, §7.1 slices |
| **veil** Swift AX traversal (`AXBoundary`, `RendererAXProbe`, `tools/run_native.py --routes ax`) | **macOS desktop extractor** for "computer use" beyond the browser. Veil also shows apps can return empty AX children, which the abstain head must handle (fall back to pixels) | §9 desktop path |
| **veil** `tools/agent_browser.py` task-file protocol | Hand the same task to any agent client and log partial actions, failures and cleanup. Reuse it for live pilots across clients | §7.3 |

**Not relevant to the ranker:** DRM, compositor and capture-suppression work, and HID input policy. Those matter only for pixel-based agents, which this model doesn't serve in v1.

**Positioning:** one portfolio covering both sides: "here is how to block agents" and "here is a well-behaved agent component." Make that explicit in the model card. The ranker grounds only to elements a human can see and use, abstains on pages that withhold their accessibility tree or present decoys, and does nothing for detection evasion. Keep evasion out of this repo entirely.

---

### Sources
- Mind2Web: arxiv.org/abs/2306.06070 · github.com/OSU-NLP-Group/Mind2Web
- DeBERTa ranker results: huggingface.co/torontodeveloper/mind2web-candidate-ranker
- WebLINX / DMR: arxiv.org/abs/2402.05930
- Prune4Web (AAAI'26): ojs.aaai.org/index.php/AAAI/article/view/40772
- Filtering for web agents: arxiv.org/abs/2609.27770
- Online-Mind2Web: arxiv.org/abs/2504.01382
- Stagehand representation benchmark: stagehand.dev/blog/html-vs-markdown-vs-accessibility-tree-llm
- Cost-efficient observations (MDPI Appl. Sci. 16(22) 11180, 2026)
- OmniParser v2: huggingface.co/microsoft/OmniParser-v2.0
- Jev Ultrafast: github.com/browser-use/jev-ultrafast · OpenJev: huggingface.co/openjev/openjev · laya-browser: huggingface.co/cklxx/laya-browser
- TinyBERT-L2 cross-encoder: huggingface.co/cross-encoder/ms-marco-TinyBERT-L2-v2 · Model2Vec/potion: github.com/MinishLab/model2vec
- AndroidControl: github.com/google-research/google-research/tree/master/android_control
- Claude API pricing: platform.claude.com/docs/en/about-claude/pricing
