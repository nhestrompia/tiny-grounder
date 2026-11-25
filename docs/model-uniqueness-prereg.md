# Is the model unique on web and desktop? Pre-registration (2026-11-26, before any data is collected)

The goal of tiny-grounder's model is to be the grounding layer that SOTA agents use on both web pages and desktop
apps. So far keywords have matched it in every agent run (v2, v3), but it beats keywords by 21–33 points on
Mind2Web's goal-level next-step recall ([nextstep-recall.md](nextstep-recall.md)). This test decides whether that
advantage holds on **held-out shop/app sites and held-out macOS apps**, where the goal is not an instruction.
Everything below is fixed before the evaluation sets exist.

## Question

Given the user's goal, the actions so far and the current page or window, does the observe digest built by the model
contain the element the next action uses more often than the digest built by keywords, at the same token budget?

## Evaluation sets (frozen before any ranker is run on them)

- **W: web shops and apps.** The scripted oracle paths of `demo/tasks_v4.yaml` (22 tasks) and `demo/tasks_v5.yaml`
  (30 tasks), run live. Each click, type or select step yields one example: the task goal, the previous steps as
  history text, the extracted page before the step and the target element ids (with equivalent ids). Every domain must
  pass the training-domain check (`data/leakage/heldout_domains.txt`).
- **D: held-out macOS apps.** 30–40 multi-step tasks on at least 6 installed apps that are not in
  `data/leakage/train_apps.txt`. Personal data apps (messaging, mail, password, banking, camera) are excluded. Each step
  is an `axdump` of the front window before the action, plus the target element. Goals are written the way a user
  would ask, before any page is dumped.
  - Quotas per set: at least 30% of tasks with a goal that names no on-screen label, and at least 20% that need an
    icon-only control.
- **D-public (secondary, report only).** At most one public dataset of goal-level desktop trajectories with
  accessibility trees, licence permitting evaluation, with apps outside `train_apps.txt`. It is not used for the
  decision.

Separation: whoever collects or writes W or D runs no ranker on it. The sets are committed with a SHA-256 manifest
before evaluation, and the evaluation reads only the frozen files.

## Item tags (computed automatically, not by hand)

- **literal:** at least half of the target's visible label content tokens appear in the goal or history.
- **goal-only:** no target label content token appears in the goal or history.
- **icon:** the target has no visible text, only an accessibility description, tooltip or image alt text.

## Arms

`exact`, `bm25`, `xs`, `m`, `blend[xs]` and `blend[m]` (α = 8, fixed), through the real `observe`, with the
checkpoints and three-seed averaging of `tools/next_step_recall.py`.

**Primary arm: `blend[xs]`.** It is the cheapest model, and the blend keeps literal labels.

## Metric and statistics

- **Primary:** recall at B = 300 estimated tokens: the target is among the ids the digest lists.
- Secondary: B = 600 and 1200, recall@5, mean elements shown, and per-tag recall.
- Paired bootstrap, 2,000 draws, seed 0. **The primary cluster is the task**, because W has only 5 sites. Site- or
  app-clustered intervals are also reported.

## Gate, per set (W and D separately)

The model **passes** on a set if both hold:

1. `blend[xs]` − max(`exact`, `bm25`) at B = 300 has a 95% CI lower bound above 0;
2. on literal items, `blend[xs]` − `exact` at B = 300 is at least −1 point (point estimate).

## Decision

| W | D | Decision |
|---|---|---|
| pass | pass | The model is unique on web and desktop. Run the agent test (gate 2): a SOTA computer-use agent with screenshots only against the same agent plus the digest. |
| pass | fail | Web only. The desktop claim waits for the retrain below. |
| fail | pass | Desktop only. Desktop becomes the product focus. |
| fail | fail | One pre-registered retrain (below). If it still fails both, ship keywords and stop model work. |

**The one retrain.** XS, 3 seeds, trained for next-step selection from the goal and history, not from instructions.
Labels come from Mind2Web train, successful v3/v4 agent runs (`tools/next_step_recall.py --set bench:...`) and
desktop trajectories on **training** apps only. It is evaluated once on the same frozen W and D with the same gate.
No other retrain, threshold or α is tuned on W or D.
