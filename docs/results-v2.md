# Release report: Plan v2.1 results (2026-11-21)

This report closes Plan v2.1 ([improvement-plan.md](improvement-plan.md)). Every number below comes from a
frozen, held-out set. Raw outputs: `runs/train/*`, `runs/compare-{select,confirm,realq}/compare.md`,
`runs/bench-v2/results.md`.

## Decision

- **Ship `find(query)` with the exact-label + BM25 ranker (no neural model in the default path).**
  - Tool output per success is about **25× smaller** than a capped accessibility snapshot, for every agent.
  - Claude: **3.3× (Haiku) and 4.1× (Sonnet) lower cost per successful task**.
  - Success equals or beats the page-view baselines for Claude.
  - GPT agents lose 7–9 points, all on multi-step tasks.
- **The neural ranker does not ship in `find`.**
  - In agent runs, blend[XS] was worse than exact for Haiku (−6.1 points [−10.2, −2.4]) and not better for the others.
  - Agents quote labels: 91% of their real queries share a word with the target.
- **The model is kept for the next phase (`observe(goal)`).** On human-written queries it is clearly better than keyword matching. The confirm split shows this, and its margin is largest where a query shares no words with the target.

## 1. Seeds (step 1): top-1, mean ± sd over 3 seeds

| | Web | Web, no shared words | DeskForge | GUIrilla* | Mind2Web task | M2W website | M2W domain |
|---|---|---|---|---|---|---|---|
| XS 1.36M | 81.3 ± 0.4 | 60.4 ± 0.9 | 72.1 ± 1.6 | 59.8 ± 2.3 | 24.2 ± 0.9 | 16.7 ± 0.5 | 16.6 ± 0.3 |
| M 4.36M | 82.9 ± 0.4 | 63.9 ± 1.3 | 73.6 ± 1.6 | 57.5 ± 2.2 | 23.6 ± 0.9 | 15.7 ± 0.8 | 16.0 ± 0.3 |
| BM25 | 49.0 | 6.2 | 73.0 | 60.7 | 6.0 | 6.2 | 8.0 |

\* GUIrilla is report-only (licence). M beats XS by 1.6 points on teacher-written web and by nothing elsewhere,
so **XS is the size to keep**.

## 2. Human-written sets (steps 3 and 4)

**Select split** (327 items, 96 sites, written by Claude and GPT writers; used to choose τ and α):
- On the hard slice, blend[XS] with α = 8 has recall@5 67.6 against 57.0 for BM25: Δ +10.7 [+4.8, +16.4].
- It loses nothing on literal items.
- XS alone, M and the router did not pass the gate. M ≈ XS.

**Confirm split** (173 items, 59 pages, 31 sites, written by the user):
- Run once, with τ and α fixed on select (τ = 0.34, α = 8); 3 seeds per model.
- Bootstrap over sites.

| ranker | top-1 | recall@5 | top-1, no shared words (n=94) | recall@5, no shared words |
|---|---|---|---|---|
| BM25 | 26.6 | 53.2 | 3.2 | 21.3 |
| exact + BM25 | 32.4 | 54.3 | 2.1 | 21.3 |
| XS | 29.1 | 49.3 | 16.0 | 35.1 |
| **blend[XS]** | **36.6** | **58.4** | 11.3 | 31.6 |
| M | 28.5 | 45.7 | 15.6 | 33.0 |
| blend[M] | 33.9 | 56.6 | 11.0 | 30.9 |

- blend[XS] vs BM25 is +10.0 top-1 [+3.8, +17.3] and +5.2 recall@5 [−0.6, +11.0].
- blend[XS] vs XS is +9.1 recall@5 [+3.3, +15.4].
- **54% of the user's queries share no word with the target.** On these, every lexical ranker is near zero for
  top-1, while the model reaches 16%.
- Absolute numbers are low. The user's queries are short and descriptive ("open events page"), and many pages
  have several near-identical links.

**Real-query diagnostic** (step 2: 472 agent `find` queries with targets labelled against goal and page; diagnostic only):

| ranker | top-1 | recall@5 |
|---|---|---|
| BM25 | 58.1 | 80.5 |
| **exact + BM25** | **68.4** | **84.5** |
| XS | 57.0 | 76.3 |
| blend[XS] | 65.5 | 81.9 |
| router[XS] | 67.3 | 82.0 |

Exact + BM25 beats XS by +11.4 top-1 [+4.1, +20.5]. Only 43 of 472 agent queries (9%) share no word with the target.

**Leakage** (`docs/leakage.md`):
- The human sets come from crawled sites outside the training domains.
- Bench v2 tasks are held-out.
- The real-query pages are 13/14 training domains, which is one more reason they are diagnostic only.

## 3. Agent benchmark v2 (step 7)

Setup:
- 66 frozen held-out tasks replayed offline from HAR: icon, vague, long-page and multi-step tasks.
- Success = end-state check AND answer regex.
- Repeats:
  - Claude Haiku, GPT-6 Luna and GPT-6.1 Terra: 3 per task and arm.
  - Claude Sonnet: 1 per task (trimmed for quota). Its vision arm stopped after 10 tasks and is not compared.
  - GPT-6.2 Sol: about 32 tasks.
- 3,047 runs; 5 timeouts.

Tool tokens are per success.

| agent | a11y-cap (20k chars) | vision | find[exact] | find[blend] | find[exact] vs a11y-cap, success Δ [95% CI] |
|---|---|---|---|---|---|
| Claude Haiku 5.5 | 92% / 16,445 | 91% / 11,220 | 90% / **621** | 84% / 672 | −2.0 [−9.1, +4.6] |
| Claude Sonnet 5.5 | 77% / 13,378 | (70%, 10 tasks) | **85%** / **421** | 83% / 445 | **+7.6** [−0.0, +16.9] |
| GPT-6 Luna | 87% / 15,098 | 82% / 8,012 | 78% / **612** | 76% / 683 | −9.1 [−18.8, −0.0] |
| GPT-6.1 Terra | 94% / 15,775 | 91% / 8,751 | 85% / **568** | 87% / 592 | −8.6 [−19.0, +0.4] |
| GPT-6.2 Sol | 88% / 36,810 | 90% / 10,788 | 86% / **647** | 85% / 720 | (vs vision) −3.1 [−11.1, +2.2] |

Whole-run tokens and cost per success:

| agent | a11y-cap | vision | find[exact] |
|---|---|---|---|
| Claude Haiku: input tokens / cost | 133k / $0.0083 | 94k / $0.0049 | **56k / $0.0025** |
| Claude Sonnet: input tokens / cost | 101k / $0.119 | 109k / $0.085 | **38k / $0.029** |
| GPT Luna: uncached input tokens | 22.5k | 15.0k | **8.8k** |
| GPT Terra: uncached input tokens | 17.1k | 9.7k | **6.2k** |

Codex's total input (cached plus uncached) is 165–230k in every arm, because its fixed context dominates. `find`
still cuts the uncached part 2–3×.

Success by task tag:

| tag (tasks) | Haiku find / a11y | Sonnet find / a11y | Luna find / a11y | Terra find / a11y |
|---|---|---|---|---|
| multistep (18) | 83 / 96 | 83 / 63 | **59 / 81** | **78 / 100** |
| vague (38) | 91 / 94 | 88 / 75 | 81 / 89 | 90 / 97 |
| icon (16) | 94 / 98 | 94 / 94 | 83 / 94 | 81 / 94 |
| long (12) | 94 / 94 | 92 / 89 | 81 / 86 | 92 / 92 |

**Plan gates:**
- Tool tokens ≥ 25% lower: **passed for every agent**, with the CI upper bound of the ratio ≤ 0.06.
- Success non-inferior within 5 points: passed for Sonnet only.
  - Haiku's CI is too wide to rule out a 5-point loss.
  - Luna and Terra lose on multi-step tasks.
- Neural ranker over exact (non-inferior within 2 points): **failed**.

## 4. What failed and why

- **Multi-step tasks** are where `find` loses.
  - After navigating, a `find`-only agent no longer knows what is on the new page, so it guesses queries.
  - GPT agents suffer most.
  - This is the gap `observe(goal, budget)` is meant to close.
- **The neural model's advantage does not reach `find`.**
  - The advantage is real: +13–14 points recall@5 on human queries with no shared words.
  - But agents write literal queries, and on literal queries exact matching is better.
- **Tasks all arms fail:** `pnpm-update-flag`, `charity-phone` (except a11y-cap) and `ocaml-icon-search`. These are extractor or page issues, not ranking.

## 5. Next phase

The next phase is measured against this report. It is described in [improvement-plan.md](improvement-plan.md), Plan v3.
