# Next-step recall

Recall and paired deltas in percentage points [95% CI]. The primary metric is recall@3: the target or an equivalent id within the first 3 distinct elements after observe-style `equiv_key` deduplication. Task (`Example.source`) clusters and neural seeds are resampled with compare_rankers.paired_bootstrap, 2,000 draws, seed 0. Site/app (`page_site`) intervals are secondary. Alpha fixed at 8.

The real observe formatter is used without landmarks. Tokens = ceil(characters/4). Budget hits use target plus equivalent ids; null examples are excluded. `recall@300`, `recall@600` and `recall@1200` are token-budget digests; raw recall@5/10/20 uses the unbudgeted ranking as before. Neural point estimates average three seeds; family seeds are shared with their blends within each draw.

The structural discriminating-items check runs no ranker: it sizes every distinct element line plus the observe header/footer at B=300. Per-tag rows use the automatic `literal`, `goal-only` and `icon` metadata; tags can overlap, and examples with none of the stored tags are shown as untagged.

Caveats: trajectories contain self-selected actions from successful runs; coordinate-only and failed actions are dropped. Role-only log histories use recovered element tags where possible and inferred HTML tags otherwise. map_rows allows text-only fallback and accepts all matching targets. Mind2Web test_domain shares 16/53 domains with training; see docs/leakage.md.

## w_shops: 368 examples, 52 task clusters (5 site/app clusters)

**Gate: PASS** — recall@3 blend[xs] − max(exact, bm25) = +7.3 pp [+1.7, +12.8] against exact; literal recall@3 blend[xs] − exact = +10.8 pp (n=148, threshold −1.0 pp).

### Overall metrics, task-clustered intervals

| Ranker | Metric | Recall [CI] | Mean K | Δ vs exact [CI] | Δ vs BM25 [CI] | Δ XS vs M [CI] |
|---|---|---|---|---|---|---|
| exact | recall@3 distinct | 53.0 [46.2, 59.6] | - | - | 4.1 [0.9, 7.6] | - |
| exact | recall@300 | 99.2 [97.9, 100.0] | 15.6 | - | 0.0 [0.0, 0.0] | - |
| exact | recall@600 | 100.0 [100.0, 100.0] | 18.4 | - | 0.0 [0.0, 0.0] | - |
| exact | recall@1200 | 100.0 [100.0, 100.0] | 18.4 | - | 0.0 [0.0, 0.0] | - |
| exact | raw recall@5 | 64.4 [57.9, 70.4] | - | - | 2.2 [0.0, 4.6] | - |
| exact | raw recall@10 | 83.7 [77.2, 89.3] | - | - | 1.4 [-1.1, 3.6] | - |
| exact | raw recall@20 | 94.6 [90.7, 97.9] | - | - | 0.0 [0.0, 0.0] | - |
| bm25 | recall@3 distinct | 48.9 [42.7, 55.3] | - | -4.1 [-7.6, -0.9] | - | - |
| bm25 | recall@300 | 99.2 [97.9, 100.0] | 15.6 | 0.0 [0.0, 0.0] | - | - |
| bm25 | recall@600 | 100.0 [100.0, 100.0] | 18.4 | 0.0 [0.0, 0.0] | - | - |
| bm25 | recall@1200 | 100.0 [100.0, 100.0] | 18.4 | 0.0 [0.0, 0.0] | - | - |
| bm25 | raw recall@5 | 62.2 [55.2, 68.8] | - | -2.2 [-4.6, 0.0] | - | - |
| bm25 | raw recall@10 | 82.3 [74.7, 88.7] | - | -1.4 [-3.6, 1.1] | - | - |
| bm25 | raw recall@20 | 94.6 [90.7, 97.9] | - | 0.0 [0.0, 0.0] | - | - |
| xs | recall@3 distinct | 63.0 [56.1, 69.6] | - | 10.1 [1.9, 17.5] | 14.1 [6.4, 21.0] | 0.8 [-6.0, 6.6] |
| xs | recall@300 | 98.9 [96.6, 100.0] | 15.1 | -0.3 [-2.9, 1.7] | -0.3 [-2.9, 1.7] | 0.1 [-2.1, 2.0] |
| xs | recall@600 | 100.0 [100.0, 100.0] | 18.4 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| xs | recall@1200 | 100.0 [100.0, 100.0] | 18.4 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| xs | raw recall@5 | 72.2 [65.8, 78.1] | - | 7.8 [1.7, 13.3] | 10.0 [3.9, 15.8] | -0.9 [-5.4, 3.7] |
| xs | raw recall@10 | 82.4 [77.6, 86.9] | - | -1.3 [-5.9, 3.7] | 0.1 [-5.0, 5.8] | -1.3 [-6.0, 2.8] |
| xs | raw recall@20 | 95.0 [91.8, 97.6] | - | 0.5 [-2.4, 3.0] | 0.5 [-2.4, 3.0] | 0.1 [-2.3, 2.6] |
| blend[xs] | recall@3 distinct | 60.3 [54.4, 66.2] | - | 7.3 [1.7, 12.8] | 11.4 [7.0, 16.0] | -0.8 [-5.1, 3.5] |
| blend[xs] | recall@300 | 99.2 [97.1, 100.0] | 15.1 | 0.0 [-2.6, 1.7] | 0.0 [-2.6, 1.7] | 0.2 [-1.6, 2.2] |
| blend[xs] | recall@600 | 100.0 [100.0, 100.0] | 18.4 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| blend[xs] | recall@1200 | 100.0 [100.0, 100.0] | 18.4 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| blend[xs] | raw recall@5 | 72.5 [66.5, 78.3] | - | 8.1 [3.6, 12.4] | 10.2 [5.9, 14.6] | -0.9 [-5.4, 3.6] |
| blend[xs] | raw recall@10 | 86.2 [80.7, 91.3] | - | 2.5 [-1.3, 7.0] | 3.9 [0.0, 8.8] | -0.3 [-3.4, 2.9] |
| blend[xs] | raw recall@20 | 96.2 [93.2, 98.6] | - | 1.6 [-0.8, 4.0] | 1.6 [-0.8, 4.0] | 0.6 [-1.3, 2.7] |
| m | recall@3 distinct | 62.2 [55.0, 69.3] | - | 9.2 [0.6, 17.2] | 13.3 [5.0, 21.2] | - |
| m | recall@300 | 98.8 [97.3, 99.8] | 15.1 | -0.4 [-2.2, 1.3] | -0.4 [-2.2, 1.3] | - |
| m | recall@600 | 100.0 [100.0, 100.0] | 18.4 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | - |
| m | recall@1200 | 100.0 [100.0, 100.0] | 18.4 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | - |
| m | raw recall@5 | 73.1 [65.5, 79.7] | - | 8.7 [1.8, 14.8] | 10.9 [4.0, 17.2] | - |
| m | raw recall@10 | 83.7 [77.6, 89.8] | - | 0.0 [-5.1, 6.2] | 1.4 [-4.0, 8.2] | - |
| m | raw recall@20 | 94.9 [91.4, 97.9] | - | 0.4 [-2.8, 3.3] | 0.4 [-2.8, 3.3] | - |
| blend[m] | recall@3 distinct | 61.1 [54.7, 67.0] | - | 8.2 [1.3, 13.9] | 12.2 [6.5, 17.3] | - |
| blend[m] | recall@300 | 99.0 [97.0, 100.0] | 15.1 | -0.2 [-2.2, 1.5] | -0.2 [-2.2, 1.5] | - |
| blend[m] | recall@600 | 100.0 [100.0, 100.0] | 18.4 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | - |
| blend[m] | recall@1200 | 100.0 [100.0, 100.0] | 18.4 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | - |
| blend[m] | raw recall@5 | 73.4 [66.9, 79.9] | - | 9.0 [3.8, 13.8] | 11.1 [6.1, 16.2] | - |
| blend[m] | raw recall@10 | 86.5 [80.3, 91.9] | - | 2.8 [-1.2, 7.0] | 4.2 [-0.3, 8.8] | - |
| blend[m] | raw recall@20 | 95.6 [91.8, 98.4] | - | 1.0 [-1.4, 3.6] | 1.0 [-1.4, 3.6] | - |

### Secondary site/app-clustered absolute intervals

| Ranker | Metric | Recall [page_site CI] |
|---|---|---|
| exact | recall@3 distinct | 53.0 [37.2, 63.2] |
| exact | recall@300 | 99.2 [96.6, 100.0] |
| exact | recall@600 | 100.0 [100.0, 100.0] |
| exact | recall@1200 | 100.0 [100.0, 100.0] |
| exact | raw recall@5 | 64.4 [49.8, 73.8] |
| exact | raw recall@10 | 83.7 [67.3, 95.6] |
| exact | raw recall@20 | 94.6 [85.8, 100.0] |
| bm25 | recall@3 distinct | 48.9 [36.3, 57.2] |
| bm25 | recall@300 | 99.2 [96.6, 100.0] |
| bm25 | recall@600 | 100.0 [100.0, 100.0] |
| bm25 | recall@1200 | 100.0 [100.0, 100.0] |
| bm25 | raw recall@5 | 62.2 [43.6, 73.1] |
| bm25 | raw recall@10 | 82.3 [63.0, 95.6] |
| bm25 | raw recall@20 | 94.6 [85.8, 100.0] |
| xs | recall@3 distinct | 63.0 [46.3, 75.2] |
| xs | recall@300 | 98.9 [95.1, 100.0] |
| xs | recall@600 | 100.0 [100.0, 100.0] |
| xs | recall@1200 | 100.0 [100.0, 100.0] |
| xs | raw recall@5 | 72.2 [56.3, 83.1] |
| xs | raw recall@10 | 82.4 [70.4, 91.3] |
| xs | raw recall@20 | 95.0 [85.5, 100.0] |
| blend[xs] | recall@3 distinct | 60.3 [46.4, 70.4] |
| blend[xs] | recall@300 | 99.2 [95.8, 100.0] |
| blend[xs] | recall@600 | 100.0 [100.0, 100.0] |
| blend[xs] | recall@1200 | 100.0 [100.0, 100.0] |
| blend[xs] | raw recall@5 | 72.5 [56.7, 82.7] |
| blend[xs] | raw recall@10 | 86.2 [71.9, 96.1] |
| blend[xs] | raw recall@20 | 96.2 [87.8, 100.0] |
| m | recall@3 distinct | 62.2 [42.8, 74.9] |
| m | recall@300 | 98.8 [96.1, 100.0] |
| m | recall@600 | 100.0 [100.0, 100.0] |
| m | recall@1200 | 100.0 [100.0, 100.0] |
| m | raw recall@5 | 73.1 [52.6, 85.0] |
| m | raw recall@10 | 83.7 [67.6, 94.3] |
| m | raw recall@20 | 94.9 [84.3, 100.0] |
| blend[m] | recall@3 distinct | 61.1 [45.0, 72.9] |
| blend[m] | recall@300 | 99.0 [96.2, 100.0] |
| blend[m] | recall@600 | 100.0 [100.0, 100.0] |
| blend[m] | recall@1200 | 100.0 [100.0, 100.0] |
| blend[m] | raw recall@5 | 73.4 [55.9, 85.1] |
| blend[m] | raw recall@10 | 86.5 [68.6, 97.2] |
| blend[m] | raw recall@20 | 95.6 [86.3, 100.0] |

### Discriminating items at B=300

Whole page fits the B=300 digest for 276 of 368 items; the non-fitting discriminating slice has 92 items.

| Ranker | B=300 recall on non-fitting items [task CI] |
|---|---|---|
| exact | 96.7 [90.7, 100.0] |
| bm25 | 96.7 [90.7, 100.0] |
| xs | 95.7 [87.6, 100.0] |
| blend[xs] | 96.7 [89.9, 100.0] |
| m | 95.3 [90.5, 98.9] |
| blend[m] | 96.0 [89.9, 100.0] |

### Per-tag breakdown, task-clustered intervals

| Tag | N | Ranker | recall@3 distinct [CI] | recall@300 [CI] | recall@600 [CI] | recall@1200 [CI] |
|---|---|---|---|---|---|---|
| literal | 148 | exact | 66.2 [56.8, 74.8] | 99.3 [97.8, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| literal | 148 | bm25 | 58.8 [48.9, 68.5] | 99.3 [97.8, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| literal | 148 | xs | 76.1 [69.2, 83.1] | 99.5 [97.2, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| literal | 148 | blend[xs] | 77.0 [70.7, 83.8] | 99.5 [97.1, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| literal | 148 | m | 71.4 [60.8, 82.5] | 99.3 [97.1, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| literal | 148 | blend[m] | 77.7 [70.4, 83.8] | 99.1 [96.7, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| goal-only | 178 | exact | 40.4 [31.4, 48.9] | 98.9 [97.1, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| goal-only | 178 | bm25 | 43.8 [35.4, 51.8] | 98.9 [97.1, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| goal-only | 178 | xs | 60.3 [49.4, 69.8] | 98.1 [94.5, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| goal-only | 178 | blend[xs] | 51.3 [43.1, 59.3] | 98.7 [95.9, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| goal-only | 178 | m | 61.8 [51.2, 70.9] | 98.1 [95.1, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| goal-only | 178 | blend[m] | 52.1 [42.5, 61.6] | 98.7 [96.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| icon | 16 | exact | 87.5 [77.3, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| icon | 16 | bm25 | 18.8 [5.3, 35.7] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| icon | 16 | xs | 39.6 [13.3, 61.1] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| icon | 16 | blend[xs] | 31.2 [7.7, 53.3] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| icon | 16 | m | 29.2 [5.9, 57.1] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| icon | 16 | blend[m] | 25.0 [0.0, 52.6] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| untagged | 26 | exact | 42.3 [24.0, 61.5] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| untagged | 26 | bm25 | 46.2 [26.9, 65.4] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| untagged | 26 | xs | 21.8 [3.7, 45.8] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| untagged | 26 | blend[xs] | 44.9 [25.0, 66.7] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| untagged | 26 | m | 33.3 [13.8, 53.8] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| untagged | 26 | blend[m] | 51.3 [31.0, 73.1] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |

## Data recovery

```json
{
  "w_shops": {
    "raw": 368,
    "null": 0,
    "missing_pages": 0
  }
}
```

Runtime: 2.9 seconds. CPU deterministic inference; all checkpoints retained.
