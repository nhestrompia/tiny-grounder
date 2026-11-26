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

## D: held-out macOS apps: pending

The first recording run produced 0 examples. The oracle locators had been written blind, before any dump, as
required. They are being corrected against real `axdump` output with the goals left frozen; the audit trail is
`docs/d-macos-locator-fixes.md`.

## D-public: no usable dataset

Nine public candidates were checked: AgentNet, OSWorld trajectory sets, Hcompany, WindowsAgentArena, GUI-World,
OS-Genesis, AgentTrek and look-before-you-click. Only `ash256/look-before-you-click` has a goal, an accessibility tree
and a recoverable target per step. After removing training apps it has 3 distinct goals on 2 apps, and 280 of 310
items come from prompt-injection runs, with no success filter. It is rejected as evidence. No public desktop dataset
currently provides goal-level next-step labels with accessibility trees at a useful scale.
