# Improvement plan after the first full evaluation (2026-11-20)

## What we measured

**Offline ranking** (top-1; dev/web test sets are written by the same teacher as the training data):

| Model | Web | Web, no shared words | DeskForge | GUIrilla (real macOS) | Mind2Web task |
|---|---|---|---|---|---|
| XS 1.4M | 81.6 | 61.1 | 70.4 | 57.3 | 23.8 |
| S 2.6M | 82.8 | 63.6 | 68.9 | 58.7 | 25.0 |
| M 4.4M | 83.2 | 64.9 | 71.7 | 56.7 | 24.3 |
| BM25 | 49.0 | 6.2 | 73.0 | 60.7 | 6.0 |
| TinyBERT-L2 14M (fine-tuned, web) | 67.9 | 29.3 | 62.7 | 59.9 | 1.8 |

- Model size has hit a ceiling: XS→M adds about 1.5 points on web and nothing on desktop.
- On desktop apps, when the instruction shares words with the target, BM25 beats the model: GUIrilla 77% for BM25 against 65% for M.
- **The model ignores literal labels too much, because of the instruction style it was trained on.** Blending BM25 into M's score never helps on teacher-written dev, but raises GUIrilla from 57% to 68%.
- The teacher-written test sets therefore overstate real-world accuracy.

**CPU cost**, for 500 elements on one thread: XS 6 ms, S 10 ms, M 14 ms. int8 file sizes are 1.4, 2.7 and 4.5 MB. int8 is not faster on Apple Silicon.

**Agent benchmark**: 38 read-only web tasks, Claude Code and Codex on subscriptions, 1,824 runs, success checked from the final page.

| Agent | full list | ARIA outline | screenshot | top20 (M) | find (M) | find (BM25) |
|---|---|---|---|---|---|---|
| Claude Sonnet 5.5 | 79% | 78% | 93% | 86% | 93% | 93% |
| GPT-6.2 Sol | 100% | 97% | 97% | 95% | 99% | 100% |
| Claude Haiku 5.5 | 87% | – | 92% | 89% | 100% | 100% |
| GPT-6.1 Terra | 97% | – | 97% | 97% | 100% | 100% |

- `find` returns about 0.8–1.2k characters, against 16–46k for the full list and 180–240k for the ARIA outline.
- `find` halves Claude's input tokens and cost.
- Claude Code's MCP output cap breaks the full and ARIA configurations on long pages: 117 results were replaced by a stub, and raising `MAX_MCP_OUTPUT_TOKENS` did not help.
- **The ranker does not change agent success on these tasks.**
  - 94–100% of the agents' `find` queries share words with the target.
  - M puts the clicked element first 69–75% of the time, against 61–65% for BM25.
  - Agents recover by reading the 5 results, so success comes out the same.
- **The tasks are too easy to tell rankers apart.** They are label-driven navigation with a median of 3–4 tool calls.

## Caveats raised in review (council round 1)

- **The tool interface drives the benchmark result, not the ranker.**
  - `find` (BM25) equals `find` (M) in every cell. Today the evidence supports shipping a compact `find` tool; the neural ranker still has to beat BM25 inside it.
  - `find` cuts input tokens about in half for Claude (39k → 20k), but by only about 2% for Codex, whose fixed context dominates.
  - Claude's full and ARIA baselines are handicapped by Claude Code's MCP output-cap stubs, so the honest competitor is the screenshot view: 93% success at 37k tokens.
- **The fusion sweep is exploratory.**
  - With the blend weight chosen on dev (α=0), fusion gives nothing.
  - The 57→68% on GUIrilla is the best weight read off the test set, not a result.
- **Statistics.**
  - n=76 is 38 tasks × 2 runs, with correlated runs and shared sites, so effective n is about 38 or less.
  - Differences under about 8 points are not significant.
  - The size sweep is one seed: "XS ≈ M", not "M is better".
- **Leakage.**
  - 37 of 38 agent tasks start on a training-crawl domain, and 10 start on an exact training URL.
  - Finder, TextEdit and System Settings appear in the macOS training data.
  - So the agent benchmark is not a held-out test.
- **Clicked element is a weak label.** The element an agent clicked is its own choice, influenced by the list it was shown, so it isn't ground-truth top-1.
- **GUIrilla is report-only.** Its licence excludes commercial benchmarking, so it must never pick weights, thresholds or the shipped model size.

## Plan v2.1 (ordered; steps 1–5 and 7 gate the release, while steps 6 and 8 are optional)

| # | Item | Success metric / gate |
|---|---|---|
| 1 | **Evaluation hygiene.** Mark the fusion numbers exploratory. Audit train/test overlap (domains, URLs, apps, near-duplicates) for every test set and the agent tasks. Train 3 seeds each of XS and M. | Overlap report in the repo; confidence interval across seeds for XS and M. |
| 2 | **Real-query set from the agent logs (in-distribution).** Sample `find` queries from successful and failed runs, deduplicate them, and hand-label the target against the task and the page. Never rely on the clicked element alone. | At least 500 labelled queries; BM25 vs XS vs M reported with a bootstrap that resamples by task and site. Diagnostic only: it does not decide step 4. |
| 3 | **Human set, frozen before any choice is made.** About 400 items written by people who haven't seen the model. Distinct sites and apps not in training. Fixed quotas: web/macOS, literal/paraphrase, text/icon, absent target, Split into **select** (about 250, holding at least 200 hard items: paraphrase, icon, goal) and **confirm** (about 150, untouched until the release decision). Extractor recall reported separately. | Confidence intervals per tag; the confirm split is the release estimate. |
| 4 | **Cheap rankers compared on the human select split.** BM25; exact label match plus BM25; XS; M; a router (BM25 when lexical coverage ≥ τ, otherwise the model); a model+BM25 blend. Each neural option is scored as the mean over its 3 seeds. τ and the blend weight are tuned on dev or the select split, never on GUIrilla. | A neural option advances only if the paired 95% confidence interval of its recall@5 gain on the hard slice excludes 0, and it loses ≤1 point on literal items. Otherwise BM25, or exact match plus BM25, is the candidate. |
| 5 | **Package the candidate.** `find(query, k=5)` and `snapshot(goal, k=20)` as an MCP server and Python API, with whichever ranker step 4 picked. M is chosen over XS only if, in step 4, its recall@5 on the hard slice of the human select split beats XS with a paired 95% confidence interval (over items and seeds) entirely above 0. | p95 `find` ≤150 ms on 500 elements including extraction; package ≤3 MB. Not released yet. |
| 6 | *(optional)* **Retrain only if the labelled failures on select justify it.** One XS ablation with literal and desktop-style teacher data (desktop share about 30%), then a learned BM25 and exact-match feature on that data. | Beats the step-4 candidate on select (paired confidence interval excludes 0) with ≤1 point loss on literal items; otherwise drop it. |
| 7 | **Agent benchmark v2 decides the release.** About 60 frozen-snapshot tasks on held-out domains and apps (icon-only, vague goals, long pages, macOS AX). Run a quick power calculation before freezing the task count. 3 runs each. Strict end-state and answer checks. Arms: a capped snapshot, the screenshot view, `find` (BM25) and `find` (candidate), all through the same interface. Analysis paired by task and grouped by site. | Per agent: ≥25% fewer **tool-output tokens** per success than the best baseline, and a paired confidence interval ruling out a success loss of more than 5 points. The ranker ships over BM25 only if it is non-inferior: the paired confidence interval rules out a success loss of more than 2 points, and its tool-output tokens are no higher. Then report the human confirm split once. |
| 8 | *(optional)* **Gated `act(query)`.** Replay and reversible actions first; measure wrong-action rate against coverage on real missing and ambiguous targets. | 95% upper bound below 1% wrong actions at useful coverage; otherwise `act` is not shipped. |
| 9 | **Deferred** until the human set or step 7 shows a gap: bulk paraphrases, pretrained word vectors, a size sweep, Mind2Web action-history data, OCR or screenshot path. | – |

Licensing checks before release:
- terms of the teacher endpoint for synthetic training data;
- private data in the macOS AX dumps;
- robots.txt and no page text in `dist/`;
- Mind2Web attribution in the model card;
- the source licence of any pretrained vectors.

## Plan v1 (superseded, kept for the record)

### P0: measure what matters (about $0, 1–2 days)
1. **Human test set.** About 300 instructions written by people, not by the teacher:
   - about 150 web, about 100 macOS through our AX extractor, about 50 goal-style;
   - each instruction tagged literal or paraphrase, and text target or icon/unlabelled target;
   - the main release metric from now on.
2. **Agent benchmark v2**, with tasks where ranking matters:
   - icon-only and unlabelled targets;
   - vague goals ("change how dark the screen is");
   - deep menus and long pages;
   - macOS apps through the AX extractor (System Settings, Finder, TextEdit; read-only);
   - a **one-shot `act(instruction)` mode** that executes the grounder's top-1 when its confidence is high, with no list shown to the LLM;
   - success, tokens and LLM calls per task, compared across BM25, M and the hybrid.

### P1: fix the instruction-style shift (about $5–15, 1–2 days of compute)
3. **Learned hybrid.** Add the element's BM25 score (normalised per page) and an exact-label-match flag as element features, and retrain. The model then learns when to trust literal matches. Fixed score blending is the fallback.
4. **Literal and desktop-style instructions.**
   - Teacher pass on the macOS, Screen2AX and DeskForge pages for label-quoting instructions ("click the Save button", "the Wi-Fi toggle") and short imperative ones.
   - Raise the desktop share of the mix from 20% to about 30%. GUIrilla train stays excluded (non-commercial licence).
5. **More instruction variety.** 3–5 paraphrases and goals per element instead of 1, on a subset of about 100k elements, at about $5–10 with Gemma 4 on OpenRouter.
6. **Pretrained word vectors**, only if the user approves:
   - initialise the 8k token embeddings from a distilled static embedding model (Model2Vec "potion", MIT);
   - same parameter count;
   - it targets the "no shared words" gap (65% web, 33–37% desktop).

### P2: product (1 day)
7. Ship the **tool interface**, not just the model:
   - `find(query, k=5)` with calibrated scores;
   - `act(query)` that clicks or types the top-1 when its probability is at least a threshold, and otherwise returns the list;
   - `snapshot(goal, k=20)`;
   - an MCP server and a Python API, with the hybrid ranker by default.
8. **Choose the shipped size after P1.** Today XS (1.4 MB, 6 ms) is within about 1.5 points of M. Retrain XS, S and M on the new data, and ship the smallest within 1 point of the best on the human test set.
9. **Abstain calibration.** Temperature-scale the NULL probability on dev, and report precision and recall on the human set.

### P3: later
10. **Multi-step goal data** with action history (Mind2Web is 16–24%; target top-20 recall of at least 85% as a pre-filter for agents).
11. **Screenshot-only path** (detector plus OCR) for apps without accessibility trees, following build-plan V2.

### Not doing
- Bigger models: the size sweep shows no gain.
- Training on GUIrilla train data (licence).
- Further tuning of the current teacher-written test sets.
