# Playbook: the most efficient local UI-grounding tool for LLM agents

Written 2026-11-20, after the first full evaluation. The improvement plan, reviewed by the council, is in
[improvement-plan.md](improvement-plan.md). Raw results are under `runs/train/` and `runs/cli*/results.md`.

## 1. Bottom line

- **The win is a compact `find(query)` tool. The model alone is not the win yet.**
  - Instead of showing an agent the whole page, give it `find("search box")`, which returns the 5 best elements in about 1k characters.
  - On 38 live web tasks this matched or beat every default page view for every agent tested.
  - For Claude it halved input tokens and cost.
- **In that tool, our ranker has not yet beaten BM25 on agent success.**
  - Frontier agents write literal queries: 94–100% share a word with the target.
  - Our model ranks the right element first more often (69–75% vs 61–65% of clicks), but agents recover from rank 2–5.
  - The model's measured edge is on meaning-only queries (65% vs 6% top-1 offline) and multi-step goals (Mind2Web 16–24% vs 6–8%). The current tasks rarely need either.
- **Model size is not the lever.**
  - XS (1.4M parameters, 1.4 MB int8, 6 ms per 500 elements) is within about 1.5 points of M (4.4M).
  - The levers are the instruction data (literal and desktop style), honest test sets, and the tool interface.

## 2. Evidence

### Offline ranking (top-1)
The web and DeskForge tests are written by the same teacher as the training data, so they are optimistic.

| | Web | Web, no shared words | DeskForge | GUIrilla (real macOS) | Mind2Web task |
|---|---|---|---|---|---|
| XS 1.4M | 81.6 | 61.1 | 70.4 | 57.3 | 23.8 |
| S 2.6M | 82.8 | 63.6 | 68.9 | 58.7 | 25.0 |
| M 4.4M | **83.2** | **64.9** | 71.7 | 56.7 | 24.3 |
| BM25 | 49.0 | 6.2 | **73.0** | **60.7** | 6.0 |
| TinyBERT-L2 14M (fine-tuned) | 67.9 | 29.3 | 62.7 | 59.9 | 1.8 |

- On desktop instructions that quote the label, BM25 beats the model: GUIrilla 77% for BM25 against 65% for M. The model learned the teacher's style, which favours rewording.
- A fixed BM25 blend tuned on dev gives no gain. The GUIrilla gain seen in a sweep is exploratory, because that weight was read off the test set.

### Agent benchmark
Claude Code (`claude -p`) and Codex (`codex exec`) ran on subscriptions, locked to our MCP browser tools: 38 tasks, 1,824 runs, success checked from the final page.

| Agent | Full list | ARIA outline (Playwright-MCP style) | Screenshot | `find` (M) | `find` (BM25) | `top20` (M) |
|---|---|---|---|---|---|---|
| Claude Sonnet 5.5 | 79% | 78% | 93% | 93% | 93% | 86% |
| GPT-6.2 Sol | 100% | 97% | 97% | 99% | 100% | 95% |
| Claude Haiku 5.5 | 87% | – | 92% | **100%** | 100% | 89% |
| GPT-6.1 Terra | 97% | – | 97% | **100%** | 100% | 97% |

Tool output per run:

| | Full list | ARIA outline | Screenshot | `find` |
|---|---|---|---|---|
| Median tool output | 16–46k chars | 180–240k chars | 0.5–0.8 MB of images | **0.8–1.2k chars** |

Claude median input tokens and cost per task:

| Claude model | Full list | `find` |
|---|---|---|
| Sonnet | 39k tokens, $0.041 | 20k tokens, $0.019 |
| Haiku | 55k tokens | 30k tokens |

Caveats:
- Each cell is about 38 effective samples, so differences under about 8 points are noise.
- 37 of the 38 task domains appear in the training crawl, so this is not a held-out test.
- The tasks are label-driven and too easy to separate rankers. Plan v2 fixes all three.

## 3. How to use it now (recipe)

1. **Expose `find`, not page dumps.** Use `demo/mcp_server.py --config find` (or `find+top20`):
   - `find(query)`: top 5 elements with scores;
   - `click(id)`, `type(id, text, press_enter)`, `go_back()`, `done(answer)`.
   - Tell the agent: *"You cannot see the whole page; call find with a short description of the element you want."* (`CONFIG_HINTS` in `demo/mcp_server.py`.)
2. **Ranker:**
   - Use **BM25** for frontier agents on label-driven UIs. It is free and as good on today's tasks.
   - Use the **model**, XS by default, when queries are vague or goal-like (`snapshot(goal)` and Mind2Web-style goals), when there is no query (the goal-mode top-20 pre-filter), or when the agent is small or local.
   - The router (BM25 for literal queries, the model otherwise) is the next thing to measure; it is step 4 of the plan.
3. **Never feed a full DOM or ARIA dump to Claude Code.**
   - Its MCP output cap swaps large results for a stub the model cannot read: 117 times in our runs.
   - Raising `MAX_MCP_OUTPUT_TOKENS` did not help.
   - If the agent needs the whole page, use a screenshot: 93% success at 37k tokens.
4. **Codex:** `find` cuts tool output about 20×, but total input tokens barely move. Codex carries about 110–150k tokens of fixed context, so measure tool-output tokens, not total tokens.
5. **`act(query)`** (auto-click the top-1 with no LLM look) is **not safe to ship yet**. It needs a wrong-action rate under 1% at 95% confidence, which is step 8 of the plan.
6. **Run it on CPU:**
   - Load with `grounder.runtime`, and `uv run python tools/bench.py --model-dir dist/<m>` for timing.
   - Ship int8 for size (4× smaller). It is not faster on Apple Silicon.

## 4. What to do next
Plan v2.1, with a council score of 8.5–10/10. Details and gates are in [improvement-plan.md](improvement-plan.md).

1. **Evaluation hygiene:** a leakage audit and 3 seeds each of XS and M.
2. **Real-query set** from the agent logs (diagnostic only).
3. **Human test set** of about 400 items on held-out sites and apps: a select split of about 250 (at least 200 hard items) and a confirm split of about 150 kept untouched.
4. **Compare on the select split:** BM25, exact match plus BM25, XS, M, a router and a blend.
5. **Package** the winning `find` + `snapshot` configuration: p95 at most 150 ms including extraction, and at most 3 MB.
6. *(optional)* **Retrain XS** with literal and desktop-style teacher data, plus BM25 and exact-match features, if the failures call for it.
7. **Agent benchmark v2:** about 60 frozen held-out tasks with icon, vague, long-page and macOS cases, and BM25 against the model in the same `find` interface. This step decides the release.
8. *(optional)* **Gated `act()`**.
9. **Deferred:** bulk paraphrases, pretrained word vectors, size sweeps, action-history data, OCR.

## 5. Operational lessons from this run

| Lesson | Detail |
|---|---|
| MPS memory | Training memory grew to more than 40 GB from three causes, all now fixed: per-worker page copies (now a compact memory-mapped store), per-shape MPS buffers (now static shapes plus watermarks), and pages with thousands of positives (positives now capped). `train.log` reports MPS memory every `log_every` steps. |
| Keep the Mac awake | Use `caffeinate -dimsu`. `caffeinate -i` did not stop idle sleep at 04:30, and the background jobs then ran only during maintenance wakes. |
| Codex MCP servers | Codex starts MCP servers in their own session, so `cli_bench` kills them by log path. |
| Test sets | Teacher-written test sets flatter a teacher-trained model, so the human set is the release metric. |
