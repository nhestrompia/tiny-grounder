# Model uniqueness test: results

Pre-registration: [model-uniqueness-prereg.md](model-uniqueness-prereg.md), including Amendment 1 (recall@3 is
primary). Command and full tables: `runs/uniqueness-w_shops/report.md`.

## W: web shops and apps (368 items, 52 tasks, 5 sites): **PASS**

| recall@3 (task-clustered 95% CI) | exact | bm25 | xs | blend[xs] | m | blend[m] |
|---|---|---|---|---|---|---|
| all (368) | 53.0 | 48.9 | 63.0 | **60.3** | 62.2 | 61.1 |
| literal (148) | 66.2 | 58.8 | 76.1 | 77.0 | 71.4 | 77.7 |
| goal-only (178) | 40.4 | 43.8 | 60.3 | 51.3 | 61.8 | 52.1 |
| icon (16) | 87.5 | 18.8 | 39.6 | 31.2 | 29.2 | 25.0 |
| untagged (26) | 42.3 | 46.2 | 21.8 | 44.9 | 33.3 | 51.3 |

- **Gate:** `blend[xs]` − exact = **+7.3 pp [+1.7, +12.8]**. On literal items, `blend[xs]` − exact = +10.8 pp, against a
  threshold of −1. Both conditions pass.
- **Site-clustered (secondary):** +7.3 pp [−6.1, +17.2]. With only 5 sites this interval includes 0. The web claim
  rests on task-level clustering.
- **Goal-only items** are where the model is distinctive: XS alone reaches 60.3 against 40.4 for exact (+20 pp). The
  α = 8 blend gives up half of that gain to keep literal labels. A router (keywords when lexical coverage is high,
  otherwise the model) is the obvious next arm. It was not pre-registered, so it is not claimed here.
- **Icon items** (n = 16) favour exact (87.5 against 31.2). Their alt text repeats the goal's words ("iMac"), and the
  model under-weights literal matches.
- **Budget digests don't separate the arms on shop pages:** B = 300 recall is 99–100% for every arm, because the pages
  are small. The model's advantage only matters for **ordering** a compact digest (top 3), which is the adaptive-digest
  regime of PR #4.

## D: held-out macOS apps (122 items, 17 of 31 tasks kept, 6 apps): **FAIL** (underpowered)

| recall@3 (task-clustered 95% CI) | exact | bm25 | xs | blend[xs] | m | blend[m] |
|---|---|---|---|---|---|---|
| all (122) | 36.1 | 38.5 | 45.9 | **47.8** | 38.0 | 44.8 |
| literal (25) | 52.0 | 64.0 | 50.7 | 72.0 | | |
| goal-only (48) | 18.8 | 18.8 | 33.3 | 22.2 | | |
| icon (54) | 46.3 | 46.3 | 53.1 | 53.7 | | |

- **Gate:** `blend[xs]` − bm25 (the best keyword ranker) = **+9.3 pp [−0.8, +20.7]**. The lower bound misses 0 by 0.8, so
  condition 1 fails. The literal check passes (+20.0 pp). Against exact it is +11.7 pp [+0.8, +23.7].
- **Why it's underpowered:** 14 of 31 tasks were dropped because the control isn't exposed in this macOS version
  (Dictionary search, Grapher's equation editor, Clock presets, Disk Utility close), leaving 26 task clusters. Every
  point estimate favours the model, but the pre-registered rule is applied as written.
- **Secondary (not gated):** desktop windows are larger than shop pages, so the B = 300 digest separates the arms
  here. On the 54 items whose whole window doesn't fit, B = 300 recall is **72.8 for `blend[xs]` against 46.3 for
  keywords**; over all items, 88.0 against 76.2.
- **Deviations, all disclosed:**
  - The oracle locators were rewritten against real dumps (93 locator and 21 path edits, logged in
    `docs/d-macos-locator-fixes.md`). Goals, ids, apps and tags are verified unchanged against the frozen commit.
  - `axdump` now keeps controls that don't advertise an action and labels table rows from their static text. Every
    arm sees the same pages, but this is a small shift from the model's training extraction.

## Decision (per the pre-registered table)

**W pass, D fail → "Web only. The desktop claim waits for the retrain."** The desktop failure reflects low power more
than evidence against the model. The retrain (a pretrained UI base with decision heads, see the strategy review) has
to be judged on a larger desktop test, the external UI-Vision set (83 apps), with the gate pre-registered before
training.

**Change, 2026-11-28 (before any retraining):** all 83 UI-Vision apps are also in GroundCUA (same authors), and the
retrain uses GroundCUA for training. UI-Vision is therefore reported as a **seen-apps** score only. The unseen-app
desktop tests are GUIrilla-Task test (macOS, evaluation-only licence), the six held-out Mac apps of `test_v2`
(Logic Pro, VLC, Keka, Chess, Podcasts, Audio MIDI Setup; VLC is removed from GroundCUA and AgentNet), and the
held-out web sites.

## D-public: no usable dataset

Nine public candidates were checked: AgentNet, OSWorld trajectory sets, Hcompany, WindowsAgentArena, GUI-World,
OS-Genesis, AgentTrek and look-before-you-click. Only `ash256/look-before-you-click` has a goal, an accessibility tree
and a recoverable target per step. After removing training apps it has 3 distinct goals on 2 apps, and 280 of 310
items come from prompt-injection runs, with no success filter. It is rejected as evidence. No public desktop dataset
currently provides goal-level next-step labels with accessibility trees at a useful scale.
