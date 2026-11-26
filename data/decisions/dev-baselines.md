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
