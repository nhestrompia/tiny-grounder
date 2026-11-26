# Next-step recall

Recall and paired deltas in percentage points [95% CI]. The primary metric is recall@3: the target or an equivalent id within the first 3 distinct elements after observe-style `equiv_key` deduplication. Task (`Example.source`) clusters and neural seeds are resampled with compare_rankers.paired_bootstrap, 2,000 draws, seed 0. Site/app (`page_site`) intervals are secondary. Alpha fixed at 8.

The real observe formatter is used without landmarks. Tokens = ceil(characters/4). Budget hits use target plus equivalent ids; null examples are excluded. `recall@300`, `recall@600` and `recall@1200` are token-budget digests; raw recall@5/10/20 uses the unbudgeted ranking as before. Neural point estimates average three seeds; family seeds are shared with their blends within each draw.

The structural discriminating-items check runs no ranker: it sizes every distinct element line plus the observe header/footer at B=300. Per-tag rows use the automatic `literal`, `goal-only` and `icon` metadata; tags can overlap, and examples with none of the stored tags are shown as untagged.

Caveats: trajectories contain self-selected actions from successful runs; coordinate-only and failed actions are dropped. Role-only log histories use recovered element tags where possible and inferred HTML tags otherwise. map_rows allows text-only fallback and accepts all matching targets. Mind2Web test_domain shares 16/53 domains with training; see docs/leakage.md.

## d_macos: 122 examples, 26 task clusters (6 site/app clusters)

**Gate: FAIL** — recall@3 blend[xs] − max(exact, bm25) = +9.3 pp [-0.8, +20.7] against bm25; literal recall@3 blend[xs] − exact = +20.0 pp (n=25, threshold −1.0 pp).

### Overall metrics, task-clustered intervals

| Ranker | Metric | Recall [CI] | Mean K | Δ vs exact [CI] | Δ vs BM25 [CI] | Δ XS vs M [CI] |
|---|---|---|---|---|---|---|
| exact | recall@3 distinct | 36.1 [25.2, 47.9] | - | - | -2.5 [-7.6, 0.0] | - |
| exact | recall@300 | 76.2 [61.9, 89.9] | 17.0 | - | 0.0 [0.0, 0.0] | - |
| exact | recall@600 | 90.2 [83.2, 96.0] | 26.5 | - | 0.0 [0.0, 0.0] | - |
| exact | recall@1200 | 100.0 [100.0, 100.0] | 31.7 | - | 0.0 [0.0, 0.0] | - |
| exact | raw recall@5 | 49.2 [38.2, 60.6] | - | - | -2.5 [-7.6, 0.0] | - |
| exact | raw recall@10 | 60.7 [48.9, 72.1] | - | - | 0.0 [0.0, 0.0] | - |
| exact | raw recall@20 | 76.2 [61.9, 89.9] | - | - | 0.0 [0.0, 0.0] | - |
| bm25 | recall@3 distinct | 38.5 [27.3, 50.4] | - | 2.5 [0.0, 7.6] | - | - |
| bm25 | recall@300 | 76.2 [61.9, 89.9] | 17.0 | 0.0 [0.0, 0.0] | - | - |
| bm25 | recall@600 | 90.2 [83.2, 96.0] | 26.5 | 0.0 [0.0, 0.0] | - | - |
| bm25 | recall@1200 | 100.0 [100.0, 100.0] | 31.7 | 0.0 [0.0, 0.0] | - | - |
| bm25 | raw recall@5 | 51.6 [40.9, 62.3] | - | 2.5 [0.0, 7.6] | - | - |
| bm25 | raw recall@10 | 60.7 [48.9, 72.1] | - | 0.0 [0.0, 0.0] | - | - |
| bm25 | raw recall@20 | 76.2 [61.9, 89.9] | - | 0.0 [0.0, 0.0] | - | - |
| xs | recall@3 distinct | 45.9 [29.8, 59.7] | - | 9.8 [-7.3, 26.6] | 7.4 [-9.9, 23.3] | 7.9 [-15.7, 26.4] |
| xs | recall@300 | 87.7 [78.4, 95.6] | 16.5 | 11.5 [-4.1, 26.9] | 11.5 [-4.1, 26.9] | 3.3 [-8.0, 15.9] |
| xs | recall@600 | 95.1 [89.5, 99.2] | 25.9 | 4.9 [-2.5, 12.7] | 4.9 [-2.5, 12.7] | 2.7 [-4.3, 9.6] |
| xs | recall@1200 | 99.2 [96.7, 100.0] | 31.7 | -0.8 [-3.3, 0.0] | -0.8 [-3.3, 0.0] | 0.3 [-2.5, 3.2] |
| xs | raw recall@5 | 56.8 [39.3, 70.5] | - | 7.7 [-11.9, 24.0] | 5.2 [-14.3, 20.7] | 7.4 [-15.2, 25.6] |
| xs | raw recall@10 | 74.3 [58.5, 85.3] | - | 13.7 [-5.4, 28.6] | 13.7 [-5.4, 28.6] | 10.1 [-14.4, 29.7] |
| xs | raw recall@20 | 88.3 [78.5, 97.5] | - | 12.0 [-3.8, 28.9] | 12.0 [-3.8, 28.9] | 5.7 [-8.6, 20.8] |
| blend[xs] | recall@3 distinct | 47.8 [36.5, 57.5] | - | 11.7 [0.8, 23.7] | 9.3 [-0.8, 20.7] | 3.0 [-11.0, 17.4] |
| blend[xs] | recall@300 | 88.0 [79.4, 95.9] | 16.3 | 11.7 [-3.1, 27.3] | 11.7 [-3.1, 27.3] | 3.0 [-8.9, 15.9] |
| blend[xs] | recall@600 | 94.5 [88.5, 99.1] | 26.0 | 4.4 [-3.2, 11.9] | 4.4 [-3.2, 11.9] | 2.7 [-2.9, 9.8] |
| blend[xs] | recall@1200 | 98.9 [95.7, 100.0] | 31.8 | -1.1 [-4.3, 0.0] | -1.1 [-4.3, 0.0] | 0.0 [-3.3, 3.2] |
| blend[xs] | raw recall@5 | 60.9 [49.2, 72.2] | - | 11.7 [-1.9, 26.1] | 9.3 [-3.0, 22.4] | 4.4 [-10.2, 18.3] |
| blend[xs] | raw recall@10 | 75.4 [64.4, 85.1] | - | 14.8 [0.7, 29.2] | 14.8 [0.7, 29.2] | 5.2 [-14.3, 23.0] |
| blend[xs] | raw recall@20 | 87.7 [79.2, 95.9] | - | 11.5 [-3.3, 27.2] | 11.5 [-3.3, 27.2] | 3.0 [-9.6, 15.8] |
| m | recall@3 distinct | 38.0 [24.4, 54.5] | - | 1.9 [-11.5, 22.0] | -0.5 [-14.2, 20.5] | - |
| m | recall@300 | 84.4 [73.2, 94.2] | 16.3 | 8.2 [-7.0, 25.2] | 8.2 [-7.0, 25.2] | - |
| m | recall@600 | 92.3 [85.2, 98.4] | 25.7 | 2.2 [-6.4, 11.2] | 2.2 [-6.4, 11.2] | - |
| m | recall@1200 | 98.9 [96.2, 100.0] | 31.6 | -1.1 [-3.8, 0.0] | -1.1 [-3.8, 0.0] | - |
| m | raw recall@5 | 49.5 [35.1, 65.9] | - | 0.3 [-16.7, 16.8] | -2.2 [-18.5, 14.6] | - |
| m | raw recall@10 | 64.2 [47.5, 83.9] | - | 3.6 [-15.4, 26.9] | 3.6 [-15.4, 26.9] | - |
| m | raw recall@20 | 82.5 [71.1, 93.8] | - | 6.3 [-9.2, 24.2] | 6.3 [-9.2, 24.2] | - |
| blend[m] | recall@3 distinct | 44.8 [30.7, 56.7] | - | 8.7 [-1.6, 21.8] | 6.3 [-4.5, 19.2] | - |
| blend[m] | recall@300 | 85.0 [73.4, 95.1] | 16.1 | 8.7 [-6.0, 26.4] | 8.7 [-6.0, 26.4] | - |
| blend[m] | recall@600 | 91.8 [84.1, 98.2] | 25.8 | 1.6 [-7.1, 10.2] | 1.6 [-7.1, 10.2] | - |
| blend[m] | recall@1200 | 98.9 [96.2, 100.0] | 31.7 | -1.1 [-3.8, 0.0] | -1.1 [-3.8, 0.0] | - |
| blend[m] | raw recall@5 | 56.6 [43.7, 69.0] | - | 7.4 [-8.3, 20.3] | 4.9 [-9.9, 17.4] | - |
| blend[m] | raw recall@10 | 70.2 [54.9, 86.5] | - | 9.6 [-9.0, 30.5] | 9.6 [-9.0, 30.5] | - |
| blend[m] | raw recall@20 | 84.7 [73.4, 95.0] | - | 8.5 [-6.2, 25.8] | 8.5 [-6.2, 25.8] | - |

### Secondary site/app-clustered absolute intervals

| Ranker | Metric | Recall [page_site CI] |
|---|---|---|
| exact | recall@3 distinct | 36.1 [19.0, 56.1] |
| exact | recall@300 | 76.2 [42.1, 99.0] |
| exact | recall@600 | 90.2 [79.9, 100.0] |
| exact | recall@1200 | 100.0 [100.0, 100.0] |
| exact | raw recall@5 | 49.2 [28.1, 73.5] |
| exact | raw recall@10 | 60.7 [33.7, 81.4] |
| exact | raw recall@20 | 76.2 [42.1, 99.0] |
| bm25 | recall@3 distinct | 38.5 [19.5, 58.6] |
| bm25 | recall@300 | 76.2 [42.1, 99.0] |
| bm25 | recall@600 | 90.2 [79.9, 100.0] |
| bm25 | recall@1200 | 100.0 [100.0, 100.0] |
| bm25 | raw recall@5 | 51.6 [29.7, 73.5] |
| bm25 | raw recall@10 | 60.7 [33.7, 81.4] |
| bm25 | raw recall@20 | 76.2 [42.1, 99.0] |
| xs | recall@3 distinct | 45.9 [25.9, 63.4] |
| xs | recall@300 | 87.7 [73.2, 99.0] |
| xs | recall@600 | 95.1 [86.6, 100.0] |
| xs | recall@1200 | 99.2 [96.7, 100.0] |
| xs | raw recall@5 | 56.8 [36.2, 74.7] |
| xs | raw recall@10 | 74.3 [56.3, 90.0] |
| xs | raw recall@20 | 88.3 [73.8, 99.0] |
| blend[xs] | recall@3 distinct | 47.8 [32.3, 59.6] |
| blend[xs] | recall@300 | 88.0 [75.0, 99.0] |
| blend[xs] | recall@600 | 94.5 [84.9, 100.0] |
| blend[xs] | recall@1200 | 98.9 [95.0, 100.0] |
| blend[xs] | raw recall@5 | 60.9 [43.1, 77.2] |
| blend[xs] | raw recall@10 | 75.4 [62.0, 88.3] |
| blend[xs] | raw recall@20 | 87.7 [74.8, 99.0] |
| m | recall@3 distinct | 38.0 [20.2, 63.9] |
| m | recall@300 | 84.4 [68.0, 100.0] |
| m | recall@600 | 92.3 [79.7, 100.0] |
| m | recall@1200 | 98.9 [96.1, 100.0] |
| m | raw recall@5 | 49.5 [30.2, 74.8] |
| m | raw recall@10 | 64.2 [43.0, 91.8] |
| m | raw recall@20 | 82.5 [65.0, 100.0] |
| blend[m] | recall@3 distinct | 44.8 [28.1, 58.7] |
| blend[m] | recall@300 | 85.0 [68.6, 100.0] |
| blend[m] | recall@600 | 91.8 [79.9, 100.0] |
| blend[m] | recall@1200 | 98.9 [96.1, 100.0] |
| blend[m] | raw recall@5 | 56.6 [37.8, 75.5] |
| blend[m] | raw recall@10 | 70.2 [48.8, 89.6] |
| blend[m] | raw recall@20 | 84.7 [68.4, 100.0] |

### Discriminating items at B=300

Whole page fits the B=300 digest for 68 of 122 items; the non-fitting discriminating slice has 54 items.

| Ranker | B=300 recall on non-fitting items [task CI] |
|---|---|---|
| exact | 46.3 [25.0, 70.2] |
| bm25 | 46.3 [25.0, 70.2] |
| xs | 72.2 [54.4, 88.3] |
| blend[xs] | 72.8 [57.1, 89.5] |
| m | 64.8 [48.3, 84.6] |
| blend[m] | 66.0 [47.4, 87.0] |

### Per-tag breakdown, task-clustered intervals

| Tag | N | Ranker | recall@3 distinct [CI] | recall@300 [CI] | recall@600 [CI] | recall@1200 [CI] |
|---|---|---|---|---|---|---|
| literal | 25 | exact | 52.0 [20.8, 86.4] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| literal | 25 | bm25 | 64.0 [32.1, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| literal | 25 | xs | 50.7 [7.4, 100.0] | 96.0 [76.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| literal | 25 | blend[xs] | 72.0 [33.3, 100.0] | 97.3 [80.9, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| literal | 25 | m | 32.0 [9.5, 64.0] | 92.0 [75.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| literal | 25 | blend[m] | 62.7 [25.0, 95.8] | 97.3 [80.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| goal-only | 48 | exact | 18.8 [4.7, 31.9] | 79.2 [63.8, 92.0] | 93.8 [86.4, 100.0] | 100.0 [100.0, 100.0] |
| goal-only | 48 | bm25 | 18.8 [4.7, 31.9] | 79.2 [63.8, 92.0] | 93.8 [86.4, 100.0] | 100.0 [100.0, 100.0] |
| goal-only | 48 | xs | 33.3 [15.2, 47.6] | 84.7 [66.0, 96.4] | 91.0 [79.5, 100.0] | 97.9 [91.3, 100.0] |
| goal-only | 48 | blend[xs] | 22.2 [9.8, 34.8] | 84.0 [66.0, 95.8] | 91.7 [80.9, 100.0] | 97.2 [88.6, 100.0] |
| goal-only | 48 | m | 32.6 [15.2, 51.1] | 82.6 [69.0, 93.3] | 88.2 [76.7, 97.8] | 97.2 [90.2, 100.0] |
| goal-only | 48 | blend[m] | 23.6 [8.7, 38.5] | 81.2 [65.9, 92.7] | 89.6 [79.2, 97.9] | 97.2 [90.2, 100.0] |
| icon | 54 | exact | 46.3 [30.3, 62.7] | 83.3 [67.9, 95.9] | 90.7 [80.8, 98.2] | 100.0 [100.0, 100.0] |
| icon | 54 | bm25 | 46.3 [30.3, 62.7] | 83.3 [67.9, 95.9] | 90.7 [80.8, 98.2] | 100.0 [100.0, 100.0] |
| icon | 54 | xs | 53.1 [39.1, 69.7] | 98.8 [94.2, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| icon | 54 | blend[xs] | 53.7 [40.0, 67.2] | 97.5 [90.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| icon | 54 | m | 48.8 [31.6, 64.2] | 98.1 [88.7, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| icon | 54 | blend[m] | 51.9 [35.8, 67.6] | 98.1 [88.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| untagged | 16 | exact | 6.2 [0.0, 21.4] | 25.0 [5.6, 53.8] | 75.0 [46.2, 94.4] | 100.0 [100.0, 100.0] |
| untagged | 16 | bm25 | 6.2 [0.0, 21.4] | 25.0 [5.6, 53.8] | 75.0 [46.2, 94.4] | 100.0 [100.0, 100.0] |
| untagged | 16 | xs | 35.4 [0.0, 77.8] | 60.4 [28.6, 88.9] | 89.6 [62.5, 100.0] | 100.0 [100.0, 100.0] |
| untagged | 16 | blend[xs] | 31.2 [0.0, 77.8] | 66.7 [31.2, 93.3] | 83.3 [53.3, 100.0] | 100.0 [100.0, 100.0] |
| untagged | 16 | m | 25.0 [0.0, 73.7] | 50.0 [0.0, 89.5] | 77.1 [42.9, 100.0] | 100.0 [100.0, 100.0] |
| untagged | 16 | blend[m] | 22.9 [0.0, 58.8] | 50.0 [0.0, 89.5] | 68.8 [30.8, 94.4] | 100.0 [100.0, 100.0] |

## Data recovery

```json
{
  "d_macos": {
    "raw": 122,
    "null": 0,
    "missing_pages": 0
  }
}
```

Runtime: 1.8 seconds. CPU deterministic inference; all checkpoints retained.
