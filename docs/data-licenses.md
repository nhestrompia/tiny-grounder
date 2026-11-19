# Data and model licences

The code and the released model weights are MIT. Every dataset used to **train** the released weights must allow
that (commercial use allowed, attribution at most). Datasets that do not are used for **evaluation** only: a score
measured on them is published, the data never reaches the weights and is never redistributed.

| Source | Licence | Used for | Credit |
|---|---|---|---|
| Our web crawl (`data/pages`) + Gemma 4 teacher outputs | Gemma 4: Apache-2.0 | training, dev, test | — |
| Mind2Web (`osunlp/Mind2Web`) | CC-BY-4.0 | goal-mode training; test splits for evaluation | Deng et al. 2023, "Mind2Web" (NeurIPS) |
| Screen2AX-Tree (`macpaw-research/Screen2AX-Tree`) + teacher outputs | Apache-2.0 | macOS training | Muryn et al. 2025, arXiv:2507.16704 |
| Our macOS dumps (`data/pages_ax`) + teacher outputs | ours | macOS training | — |
| DeskForge-1M (`docling-project/DeskForge-1M`, a sample: `tools/deskforge.py`) | MIT | Linux desktop training (train, dev); `test_app` (held-out applications) for evaluation | Gurbuz et al. 2026, "DeskForge: Dense Supervision from Desktop Environments for Computer-Use Agents", arXiv:2610.02320 |
| GUIrilla-Task test (`macpaw-research/GUIrilla-Task`) | CC-BY-NC-4.0 | **evaluation only** (macOS benchmark) | Garkot et al. 2025, arXiv:2510.16051 |
| GUIrilla-Task train | CC-BY-NC-4.0 | **not used** for the release; only a separately labelled CC-BY-NC experiment, if ever | as above |

DeskForge screenshots show third-party application interfaces, themes and icons that remain their owners' property; only the accessibility-tree text of a sample is converted, never a screenshot.
Mind2Web asks that the unzipped test files are not redistributed; the release ships none of them.
No dataset is redistributed in this repository; `data/` is gitignored and each converter downloads from the source.
