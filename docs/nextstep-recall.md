# Next-step recall at a token budget

Full Mind2Web goal/history splits and successful bench-v2 trajectories; null targets excluded. Actual `observe`, no landmarks; hits include equivalent ids. B=ceil(characters/4). Neural results average three seeds; blend α=8 is fixed. Paired `page.site`/seed bootstrap: 2,000 draws, seed 0, imported from `compare_rankers`.

Recall in %. At B=300, Mind2Web mean K is 18.9–19.1 for XS versus 15.0–15.6 for exact. [Full intervals, paired deltas, mean elements shown and recall@5/10/20](../runs/nextstep/report.md). [Tool and exact command](../tools/next_step_recall.py).

| Set (n; sites) | B | Exact | BM25 | XS | M | Blend XS | Blend M |
|---|---:|---:|---:|---:|---:|---:|---:|
| M2W task (1975; 69) | 300 | 38.7 | 33.6 | 71.9 | 70.6 | 68.6 | 69.9 |
| M2W task (1975; 69) | 600 | 49.4 | 46.6 | 80.2 | 78.4 | 79.1 | 78.5 |
| M2W task (1975; 69) | 1200 | 63.2 | 61.9 | 85.3 | 83.2 | 85.1 | 84.1 |
| M2W website (1314; 10) | 300 | 40.2 | 33.6 | 61.6 | 59.8 | 61.8 | 61.3 |
| M2W website (1314; 10) | 600 | 52.9 | 47.2 | 71.1 | 69.2 | 73.1 | 71.7 |
| M2W website (1314; 10) | 1200 | 63.5 | 60.6 | 77.9 | 76.5 | 79.5 | 78.5 |
| M2W domain (5590; 54) | 300 | 42.5 | 37.6 | 67.1 | 63.8 | 64.9 | 64.0 |
| M2W domain (5590; 54) | 600 | 54.5 | 51.6 | 77.3 | 74.7 | 76.7 | 75.3 |
| M2W domain (5590; 54) | 1200 | 66.7 | 65.2 | 83.9 | 81.9 | 83.9 | 82.7 |
| Bench-v2 (248; 19) | 300 | 62.9 | 62.5 | 62.8 | 60.6 | 66.9 | 64.9 |
| Bench-v2 (248; 19) | 600 | 81.0 | 81.5 | 77.7 | 78.8 | 81.0 | 80.8 |
| Bench-v2 (248; 19) | 1200 | 88.3 | 88.3 | 87.4 | 89.2 | 88.6 | 90.6 |

Bench-v2: 248 unique examples from 2,811 recoverable actions; 1,710 dropped, including 1,517 unlabelled/failed, 190 unmatched elements and 3 missing pages. URL replay loses transient state; labels are agent choices. Mind2Web test_domain shares **16/53 domains with training**; see [leakage audit](leakage.md).

- B=300: XS beats both keywords on all Mind2Web splits. Δ vs exact [95% CI], points: task 33.2 [28.8, 37.2]; website 21.4 [14.1, 29.5]; domain 24.6 [21.8, 27.5].
- B=600: XS beats both keywords on all Mind2Web splits. Δ vs exact [95% CI], points: task 30.8 [26.7, 34.6]; website 18.2 [11.3, 25.0]; domain 22.8 [19.6, 26.0].
- B=1200: XS beats both keywords on all Mind2Web splits. Δ vs exact [95% CI], points: task 22.1 [17.5, 26.7]; website 14.4 [9.2, 19.0]; domain 17.2 [14.2, 20.1].
- Bench-v2 shows no neural/blend gain over either keyword ranker with CI excluding zero. XS−exact: B=300 -0.1 [-8.9, 10.0]; B=600 -3.4 [-11.3, 4.9]; B=1200 -0.9 [-10.0, 8.9].
- M never significantly beats XS. XS beats M on domain at all budgets and task at 1,200. Retain XS and advance the goal/history retraining experiment in step 3; use step 5 to decide whether model observe ships.

