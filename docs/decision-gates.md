# Decision gates

This document proposes the offline gates for future System-1 decision models. The numbers below come only from the dev split. The test split stayed frozen; `tools/decision_baselines.py` reads only `dev.jsonl` files.

`enough` and `worked` are the v1 labels. `enough_v2` and `worked_v2` are the counterfactual/progress labels described in `docs/decision-labels.md`. Keep both while comparing rules; new model acceptance should use the v2 labels.

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
| ENOUGH | base rate arm=a11y-cap | positive share (n) | 0.195 (n=574) |
| ENOUGH | base rate arm=find+observe[exact+blend] | positive share (n) | 0.280 (n=325) |
| ENOUGH | base rate arm=find+observe[exact] | positive share (n) | 0.282 (n=323) |
| ENOUGH | base rate arm=find[blend] | positive share (n) | 0.295 (n=305) |
| ENOUGH | base rate arm=find[exact] | positive share (n) | 0.281 (n=317) |
| ENOUGH | base rate arm=vision | positive share (n) | 1.000 (n=368) |
| ENOUGH | majority | precision / recall / AUROC | n/a / 0.000 / n/a |
| ENOUGH | url_changed_after_action | precision / recall / AUROC | 0.373 / 0.618 / 0.490 |
| ENOUGH | empty_text_ratio | precision / recall / AUROC | 0.280 / 0.181 / 0.451 |
| ENOUGH | visual_terms_and_empty_text_ratio | precision / recall / AUROC | n/a / 0.000 / 0.451 |
| ENOUGH | vision_arm | precision / recall / AUROC | 1.000 / 0.438 / 0.719 |
| ENOUGH_V2 | base rate | positive share | 0.188 |
| ENOUGH_V2 | base rate arm=a11y-cap | positive share (n) | 0.072 (n=278) |
| ENOUGH_V2 | base rate arm=find+observe[exact+blend] | positive share (n) | 0.246 (n=69) |
| ENOUGH_V2 | base rate arm=find+observe[exact] | positive share (n) | 0.101 (n=69) |
| ENOUGH_V2 | base rate arm=find[blend] | positive share (n) | 0.261 (n=261) |
| ENOUGH_V2 | base rate arm=find[exact] | positive share (n) | 0.243 (n=280) |
| ENOUGH_V2 | majority | precision / recall / AUROC | n/a / 0.000 / n/a |
| ENOUGH_V2 | url_changed_after_action | precision / recall / AUROC | 0.207 / 0.683 / 0.539 |
| ENOUGH_V2 | empty_text_ratio | precision / recall / AUROC | 0.038 / 0.056 / 0.371 |
| ENOUGH_V2 | visual_terms_and_empty_text_ratio | precision / recall / AUROC | n/a / 0.000 / 0.371 |
| WORKED | base rate | positive share | 0.511 |
| WORKED | base rate arm=a11y-cap | positive share (n) | 0.736 (n=382) |
| WORKED | base rate arm=find+observe[exact+blend] | positive share (n) | 0.633 (n=229) |
| WORKED | base rate arm=find+observe[exact] | positive share (n) | 0.577 (n=227) |
| WORKED | base rate arm=find[blend] | positive share (n) | 0.213 (n=211) |
| WORKED | base rate arm=find[exact] | positive share (n) | 0.161 (n=223) |
| WORKED | base rate arm=vision | positive share (n) | 0.555 (n=281) |
| WORKED | majority | precision / recall / AUROC | 0.511 / 1.000 / n/a |
| WORKED | url_changed_after_action | precision / recall / AUROC | 0.534 / 0.937 / 0.540 |
| WORKED | empty_text_ratio | precision / recall / AUROC | 0.564 / 0.296 / 0.533 |
| WORKED_V2 | base rate | positive share | 0.935 |
| WORKED_V2 | base rate arm=a11y-cap | positive share (n) | 0.963 (n=323) |
| WORKED_V2 | base rate arm=find+observe[exact+blend] | positive share (n) | 0.966 (n=178) |
| WORKED_V2 | base rate arm=find+observe[exact] | positive share (n) | 0.943 (n=176) |
| WORKED_V2 | base rate arm=find[blend] | positive share (n) | 0.895 (n=152) |
| WORKED_V2 | base rate arm=find[exact] | positive share (n) | 0.880 (n=158) |
| WORKED_V2 | base rate arm=vision | positive share (n) | 0.915 (n=82) |
| WORKED_V2 | majority | precision / recall / AUROC | 0.935 / 1.000 / n/a |
| WORKED_V2 | url_changed_after_action | precision / recall / AUROC | 0.988 / 0.978 / 0.903 |
| WORKED_V2 | empty_text_ratio | precision / recall / AUROC | 0.974 / 0.304 / 0.583 |
| DONE | base rate | positive share | 0.232 |
| DONE | majority | precision / recall / AUROC | n/a / 0.000 / n/a |
| DONE | goal_line_token_overlap | precision / recall / AUROC | 0.681 / 0.091 / 0.553 |

## What changed under the v2 labels

The v1 ENOUGH label is still arm-encoded: every dev vision-arm state is positive, while text-arm positive rates are much lower. The v2 pair label removes vision-arm rows entirely. Its dev positive rate is 0.188, with arm rates from 0.072 to 0.261. That remaining variation is task mix and pair coverage, not screenshot-tool availability. The simple empty-text heuristic is worse than the negative majority baseline (AUROC 0.371); URL change is weak as an ENOUGH signal (AUROC 0.539).

The v1 WORKED label varies from 0.161 to 0.736 positive by arm because it treats habitual `find` calls as failure. The v2 progress label compresses dev arm base rates to 0.880–0.966, with a pooled positive rate of 0.935. This is no longer the wild arm artifact seen in v1, but the high rate means majority-positive is a strong baseline.

The important WORKED warning is different: the URL-change heuristic is already very strong under v2 (precision 0.988, recall 0.978, AUROC 0.903). That is partly structural. Most positive labels come from a successful final action or from a next target appearing after navigation. A learned model that merely rediscovers URL change adds no value. The old “+0.10 AUROC over URL change” gate would be impossible and is therefore replaced by an operating-point dominance gate.

## Proposed gates

A model passes a gate only when it beats the best meaningful baseline and meets the cost-safety requirement. Final acceptance still needs the frozen test split and an end-to-end cost/success comparison against both the agent alone and a placebo decision model.

| decision | required metric | margin over baseline | calibration requirement | cost meaning |
|---|---|---|---|---|
| SHOW | distinct-element recall@3 | at least +0.10 absolute over exact, with the paired site bootstrap lower bound above zero | if the system emits a target-in-view confidence, ECE ≤ 0.05; precision at confidence ≥ 0.90 must be ≥ 0.95 | fewer elements can be shown without omitting the needed control |
| NEXT | accepted-step joint action+target precision | at least 0.95 precision on at least 30% of steps; target recall@3 must also beat exact by at least +0.10 | ECE ≤ 0.03; empirical precision at the operating confidence threshold must be at least the threshold | a confident step can skip the planner LLM call |
| ENOUGH v2 | precision of predicted “text is enough” (negative predictive value for needs-screenshot) | at least 0.97 on at least 70% of states in every evaluated arm; AUROC at least +0.10 over the best non-arm dev heuristic, hence at least 0.639 | ECE ≤ 0.05; empirical negative precision at the operating threshold must be at least 0.97 in every arm | text-only steps can avoid screenshot cost without losing needed information |
| WORKED v2 | precision of predicted “worked” at the chosen coverage | dominate the URL-change operating point: either precision ≥ 0.995 at recall ≥ 0.90, or precision > 0.988 at recall ≥ 0.978; report the same operating point per arm | ECE ≤ 0.02; empirical precision at the operating confidence threshold must be at least the threshold | successful actions can skip a re-observation call only if they beat the nearly free progress signal |
| DONE | precision of predicted “done” | at least 0.98 precision and at least 0.70 recall; AUROC at least +0.10 over the answer-overlap heuristic | ECE ≤ 0.03; precision at the operating confidence threshold must be at least the threshold | finished tasks can end without extra verification steps |

## Reading the gates

Precision matters more than recall for NEXT, WORKED, and DONE because a false positive acts or stops too early. ENOUGH is the reverse decision: the cheap action is to stay in text, so the gate measures precision of the negative class. A model that cannot abstain cannot pass NEXT, WORKED, or DONE.

Every gate must be reported by arm as well as pooled. A pooled pass driven by one arm does not count. Every gate also needs a shadow replay estimate before live use: count avoided planner calls or observation calls, then compare estimated cost with the task success rate. The Jev failure mode is a confident but badly calibrated decision model that adds verification rather than removing it.

For WORKED v2 in particular, the first implementation question should not be “can a model reach high accuracy?” Majority positive already does. The question is whether the model can identify the small negative class better than the URL-change heuristic while preserving very high positive precision. If it cannot, the heuristic or a cautious always-verify policy is cheaper and safer.
