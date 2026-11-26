# Decision gates

This document proposes the offline gates for future System-1 decision models. The numbers below come only from the dev split. The test split stayed frozen; `tools/decision_baselines.py` reads only `dev.jsonl` files.

## Dev baselines

| decision | baseline | metric | value |
|---|---|---:|---:|
| SHOW | exact | recall@3 | 0.491 |
| SHOW | exact | top-1 | 0.245 |
| SHOW | bm25 | recall@3 | 0.443 |
| SHOW | bm25 | top-1 | 0.208 |
| NEXT | exact | recall@3 | 0.491 |
| NEXT | exact | top-1 | 0.245 |
| NEXT | bm25 | recall@3 | 0.443 |
| NEXT | bm25 | top-1 | 0.208 |
| NEXT | majority action (click) | accuracy | 0.943 |
| NEXT | textbox → type rule | accuracy | 0.981 |
| ENOUGH | base rate | positive share | 0.380 |
| ENOUGH | majority | precision / recall / AUROC | n/a / 0.000 / n/a |
| ENOUGH | visual_terms_and_empty_text_ratio | precision / recall / AUROC | n/a / 0.000 / 0.451 |
| ENOUGH | vision_arm | precision / recall / AUROC | 1.000 / 0.438 / 0.719 |
| WORKED | base rate | positive share | 0.511 |
| WORKED | majority | precision / recall / AUROC | 0.511 / 1.000 / n/a |
| WORKED | url_changed_after_action | precision / recall / AUROC | 0.534 / 0.937 / 0.540 |
| DONE | base rate | positive share | 0.232 |
| DONE | majority | precision / recall / AUROC | n/a / 0.000 / n/a |
| DONE | goal_line_token_overlap | precision / recall / AUROC | 0.681 / 0.091 / 0.553 |

The SHOW and NEXT rows have the same target elements because both labels come from the same successful next actions. Exact-match ranking is the strongest keyword target baseline. The action-type rule is strong because the dev rows are mostly clicks; it is not evidence that target selection is solved.

The ENOUGH `vision_arm` row is not a usable causal baseline. It measures the benchmark design as much as the decision: the vision arm receives screenshots automatically, while text arms do not offer the same optional screenshot. Keep it as a warning about the arm confound. The non-arm visual heuristic is weak.

## Proposed gates

A model passes a gate only when it beats the best meaningful baseline and meets the cost-safety requirement. Margins are point-estimate margins on dev unless stated otherwise. Final acceptance still needs the frozen test split and an end-to-end cost/success comparison against both the agent alone and a placebo decision model.

| decision | required metric | margin over baseline | calibration requirement | cost meaning |
|---|---|---|---|---|
| SHOW | distinct-element recall@3 | at least +0.10 absolute over exact, with the paired site bootstrap lower bound above zero | if the system emits a target-in-view confidence, ECE ≤ 0.05; precision at confidence ≥ 0.90 must be ≥ 0.95 | fewer elements can be shown without omitting the needed control |
| NEXT | accepted-step joint action+target precision | at least 0.95 precision on at least 30% of steps; target recall@3 must also beat exact by at least +0.10 | ECE ≤ 0.03; empirical precision at the operating confidence threshold must be at least the threshold | a confident step can skip the planner LLM call |
| ENOUGH | precision of predicted “text is enough” | at least 0.97 on at least 70% of states, reported per arm; AUROC at least +0.10 over the best non-arm heuristic | ECE ≤ 0.05; precision at the operating threshold must be at least 0.97 in every evaluated arm | text-only steps can avoid screenshot cost without losing needed information |
| WORKED | precision of predicted “worked” | at least 0.95 on at least 70% of post-action states; AUROC at least +0.10 over the URL-change heuristic | ECE ≤ 0.04; precision at the operating confidence threshold must be at least the threshold | successful actions can skip a re-observation call |
| DONE | precision of predicted “done” | at least 0.98 precision and at least 0.70 recall; AUROC at least +0.10 over the answer-overlap heuristic | ECE ≤ 0.03; precision at the operating confidence threshold must be at least the threshold | finished tasks can end without extra verification steps |

## Reading the gates

Precision matters more than recall for NEXT, WORKED, and DONE because a false positive acts or stops too early. ENOUGH is the reverse decision: the cheap action is to stay in text, so the gate measures precision of the negative class. A model that cannot abstain cannot pass NEXT, WORKED, or DONE.

Every gate must be reported by arm as well as pooled. A pooled pass driven by the vision arm does not count. Every gate also needs a shadow replay estimate before live use: count avoided planner calls or observation calls, then compare estimated cost with the task success rate. The Jev failure mode is a confident but badly calibrated decision model that adds verification rather than removing it.
