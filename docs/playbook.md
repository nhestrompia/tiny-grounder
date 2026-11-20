# Playbook: cutting frontier-agent tokens with a local UI-grounding tool

Updated 2026-11-21, after Plan v2.1. Full numbers are in [results-v2.md](results-v2.md); the next plan is in
[improvement-plan.md](improvement-plan.md).

## 1. Bottom line

- **The tool is the win: a compact `find(query)` instead of page dumps or screenshots.**
  - On 66 frozen, held-out web tasks, it returned about **600 tool tokens per successful task**, against 11–16k for screenshots or a capped accessibility snapshot. That is roughly 25× less.
  - For Claude, whole-task cost per success fell **3.3× (Haiku) and 4.1× (Sonnet)**, with equal or better success.
  - For Codex (GPT), uncached input fell 2–3×. Total input barely moved, because Codex's fixed context is 110–150k tokens.
- **Use exact-label + BM25 ranking inside `find`, not the neural model.**
  - Agents quote labels: 91% of their real queries share a word with the target.
  - For those queries, exact matching wins: 68% vs 57% top-1 on 472 real agent queries.
  - In agent runs, the neural blend lost 6 points for Haiku.
- **The tiny model earns its place where no literal query exists.**
  - On the user's own descriptions (the confirm split), more than half share no word with the target. There the blend reaches 36.6% top-1 against 26.6% for BM25, and the model alone is far ahead of every keyword ranker.
  - That is the job of the next tool, `observe(goal)`.
- **`find` alone hurts on multi-step tasks:** 59–83% success vs 81–100% for a page snapshot. Agents lose track after navigating. `observe` is the fix to build next.
- **Size is not the lever.**
  - XS (1.36M parameters, 1.4 MB int8, 6 ms per 500 elements on CPU) is within 1.6 points of M on every set (3 seeds each), and equal off the teacher-written web set.
  - Ship XS.

## 2. Recipe (today)

1. **Expose perception, not page dumps.** Package `grounder_tool` (`grounder-tool` MCP server):
   - `find(query, k=5)`: top elements as `id|role|text|context`, about 1k characters;
   - `snapshot(goal, k=20)`: when the agent has no specific target yet.
   - Tell the agent: *"You cannot see the whole page; call find with a short description of the element you want."*
2. **Ranker:** `exact` (exact label match, then BM25). It is the default.
   - Use `blend:<xs-dir>:8` only for human-typed or vague, descriptive queries, and in `snapshot(goal)`.
   - Do not use the router or the model alone.
3. **Never feed a full DOM or ARIA dump to Claude Code.** Its MCP output cap replaces large results with a stub the model cannot read. Compact results are the only kind that reliably arrive.
4. **For multi-step tasks, today:** let the agent fall back to a capped snapshot after each navigation. This trades tokens for success until `observe` exists.
5. **Codex:** measure tool tokens and uncached input, not total input tokens. Its fixed overhead dominates.
6. **`act(query)`** (auto-click the top-1 with no model looking) stays unshipped. It is not measured.
7. **CPU:** load with `grounder.runtime`, and ship int8 for size. The p95 for `find`, including extraction, is about 110 ms at 500 elements.

## 3. What to build next (Plan v3)

Perception only; the frontier model does every action. The central question:

> How much of a computer-use agent's perception workload can be moved to a tiny local model without losing task success?

1. **`observe(goal, budget)`:** a compact, goal-aware page digest. It gives page landmarks plus the top elements for the goal, within B tokens.
   **`inspect(id | region)`:** more detail on demand.
   **CDP attach:** to the executor's own browser, so handles match what the executor clicks.
2. **Offline metric:** next-step recall at a token budget, on Mind2Web plus the v2 logs, for the model against keywords.
3. **Re-train XS** on goal plus action history, only if (2) shows headroom.
4. **A/B/C agent re-run** on multi-step and vague tasks, with Claude Haiku and Sonnet:
   - A: `find` only;
   - B: keyword digest;
   - C: model digest.
   If C beats B at the same budget, the model ships in `observe`. If not, B ships.

## 4. Operational lessons

| Lesson | Detail |
|---|---|
| MPS memory | Training memory grew to more than 40 GB from three causes, all now fixed: per-worker page copies (now a compact memory-mapped store), per-shape MPS buffers (now static shapes plus watermarks), and pages with thousands of positives (now capped at half the element limit). |
| Keep the Mac awake | `caffeinate -dimsu`; `-i` did not stop idle sleep. |
| Subscription agents | `claude -p --tools "" --strict-mcp-config` and `codex exec --ignore-user-config -s read-only` lock agents to our tools. Drivers stop and purge runs on a usage-limit error. |
| Codex MCP servers | Codex starts them in their own session, so the benchmark kills them by log path. |
| Frozen pages | Replay from HAR so reruns see identical pages. |
| Test sets | Teacher-written tests flatter a teacher-trained model. Only human-written held-out sets decide. Keep a confirm split untouched until the decision. |
