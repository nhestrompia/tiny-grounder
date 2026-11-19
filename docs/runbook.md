# Runbook: teacher on OpenRouter, training on the Mac

Paid work: the LLM teacher, through OpenRouter, run from the Mac. Training is free and also runs on the Mac (MPS). The RunPod/vLLM path (`gpu/`) still works; see the last section.

Every script is resumable: rerun the same command and finished teacher calls come from `data/teacher_cache/<model>`.

## Before spending money

1. The crawl is finished (`pgrep -f grounder.data.crawl` prints nothing): `uv run python -m grounder.data.crawl --out data/pages --stats`.
2. `uv run pytest -q` passes.
3. `export OPENROUTER_API_KEY=...` (an OpenRouter key; put a credit limit on the key). Credits are loaded on the account; when they run out the teacher stops with "out of credits" and a rerun resumes.
4. Models (checked on 2026-11-19, `curl -s https://openrouter.ai/api/v1/models`):

   | Model | Input / output per 1M tokens | `structured_outputs`, `response_format` |
   |---|---|---|
   | `google/gemma-4-26b-a4b-it` | $0.0765 / $0.255 | both |
   | `google/gemma-4-31b-it` | $0.20 / $0.40 | both |

   Prices change; the pilot prints the real cost. The `:free` variants support `response_format` but not `structured_outputs`, are rate limited, and are not used. The scripts send `provider.require_parameters: true`, so OpenRouter only routes to providers that honour the JSON schema.

## 1. Pilot, two models, identical inputs

```bash
scripts/teacher_api.sh pilot google/gemma-4-26b-a4b-it
scripts/teacher_api.sh pilot google/gemma-4-31b-it
```

Each run teaches 50 seeded English dev pages (`data/pages/dev`) and 20 seeded desktop windows (`data/pages_s2ax/train` and `data/pages_ax/train` together), the same pages for every model, and writes `runs/pilot-<model>/`:

- `teacher.jsonl` (kept examples), `teacher.log` (progress, usage, rejections), `usage-*.json`
- `samples.txt`: 30 random kept examples per kind next to the element they target, then item coverage, token usage and the **projected cost and time of the full run** (21,149 calls).

`MAX_COST_USD=2 scripts/teacher_api.sh pilot ...` stops launching calls above that cost; `TEACHER_CONCURRENCY` (default 32) lowers the load if the log shows many retries.

### Go / no-go (per model)

- [ ] **Blind audit** of 200 outputs (sample from `teacher.jsonl` without looking at the model name, 50 per kind, web and desktop mixed, e.g. `uv run python -m tools.pilot_report ... --per-kind 50`): at least 95% correct (the query really selects that element), actionable (something a person would say) and unambiguous.
- [ ] **Filter retention** (kept / raw, printed as `pass`) at least 75%; no single rejection reason above 10%.
- [ ] **Item coverage** at least 99% (`item coverage` in `samples.txt`), and `failed_batches=0`.
- [ ] Paraphrases sharing no word with the element at least 95%.
- [ ] Projected cost and time of the full run are acceptable.

Take the cheaper model that passes. If both pass, compare audit scores first, cost second. If neither passes, edit `SYSTEM_PROMPT` in `grounder/data/teacher.py` and rerun the pilot (the cache is keyed on the request, so the new prompt costs new calls).

## 2. Full run

```bash
scripts/teacher_api.sh full google/gemma-4-26b-a4b-it
```

Writes `data/teacher/<model>-<source>-<split>.jsonl` for web (`data/pages` train, dev, test) and native macOS (`data/pages_ax`, `data/pages_s2ax`), a `.usage.json` beside each, and `data/teacher/<model>.log`. `MAX_COST_USD` applies to each of those runs separately, not to the total. Stop any time with Ctrl-C; rerun to resume. If a run stops with `HALTED`, nothing is written for that split: fix the cause (credits, key) and rerun.

Check `failed_batches=0` and the `items ...` coverage line of each run; rerun if batches failed (failed ones are retried, finished ones are free).

## 3. Assemble and mix

```bash
scripts/assemble.sh google/gemma-4-26b-a4b-it
```

This builds `data/examples` (web pages + teacher + template examples) and `data/examples-macos` (macOS pages the same way), then `tools/mix.py` writes `data/mix/{train,dev}.jsonl` and `data/mix/dev_<source>.jsonl`.

| Source | Data | Role |
|---|---|---|
| web | `data/pages` + teacher + template examples | train / dev / test |
| macos | `data/pages_ax`, `data/pages_s2ax` + teacher + template examples | train, dev (held-out sites) |
| deskforge | `data/deskforge/{pages,examples}` | train, dev; `test_app` for evaluation |
| mind2web | `data/mind2web` (goal mode) | train, dev (held-out sites); `test_task`, `test_website`, `test_domain` for evaluation |
| guirilla | `data/guirilla` | **evaluation only** (`test_unseen_apps`); never in train, dev or the tokenizer |

Default weights of an epoch: web 0.70, desktop 0.20 (macos and deskforge together, by natural size), mind2web 0.10, as deterministic up- or down-sampling of each source to its share of the natural total. Change them with `--weights web=0.6 desktop=0.3 mind2web=0.1` or `--total N` (`uv run python -m tools.mix --help`). The tool prints the examples per source before and after.

## 4. Train on the Mac

```bash
scripts/train.sh                       # MODELS="xs s m", SEEDS="0", DEVICE=mps on macOS, WORKERS=4, EPOCHS=10
MODELS=m SEEDS="0 1 2" scripts/train.sh   # rerun the winner with three seeds
```

L (8.2M parameters) is over the 5M target, so it is not in the default; add it with `MODELS="xs s m l"`.

The tokenizer trains on the pages and mixed train examples of all training sources (never GUIrilla, never a test split). Checkpoints are selected by the **mean over sources of top-1 on the per-source dev sets**, so the large web source cannot dominate. Each run (`runs/train/<model>-s<seed>/`) is evaluated, together with BM25, on: `dev` (mixed, with a per-source table), `web_test`, `deskforge_test_app`, `mind2web_test_task`, `mind2web_test_website`, `mind2web_test_domain` and `guirilla_test`. `runs/train/results.md` has one column group (top-1, top-5) per eval set, then one table per set with top-1 per kind. ONNX exports go to `dist/<model>-s<seed>`.

Measured on the Mac (Apple M5, `--device mps`, batch 64, data loader included), for the sweep of 300k examples x 10 epochs:

| | XS | S | M | L |
|---|---|---|---|---|
| mps s/1k examples | 2.2 | 2.1 | 3.4 | 6.3 |
| mps hours | 1.8 | 1.8 | 2.8 | 5.3 |

The mix has about 130k to 400k examples per epoch depending on the teacher output, so scale accordingly. The loader costs about 1 s per 1k examples per process; `WORKERS=4` on the Mac.

## Failure recovery

- A script stopped (network, Ctrl-C, sleep): rerun the same command.
- `HALTED: out of credits (HTTP 402)`: add credits and rerun. `API key rejected (HTTP 401)`: check `OPENROUTER_API_KEY`.
- `HALTED: cost limit`: raise `MAX_COST_USD` or unset it, rerun.
- Many `attempt N failed` warnings with HTTP 429: lower `TEACHER_CONCURRENCY` (default 32).
- Repeated `HTTP 404 ... No endpoints found`: no provider of that model supports JSON schema at the moment; try the other model.
- `failed_batches` above zero or coverage under 99%: rerun the same command (failed batches are retried, finished ones are free) and look at `data/teacher/<model>.log`.
- `train.sh` skips runs with `train.done` and eval sets with a `.json` file; delete them to redo. An interrupted training restarts from scratch.

## Optional: RunPod with a self-hosted vLLM

Kept as an alternative to OpenRouter; the scripts in `gpu/` are unchanged apart from sharing `scripts/teacher_lib.sh`. Rent one RTX 4090 (Community Cloud, template RunPod PyTorch, volume 80 GB at `/workspace`; set `HF_TOKEN` for gated weights), then:

```bash
gpu/sync.sh up HOST PORT                    # on the Mac: code + data/pages (+ macOS pages)
ssh root@HOST -p PORT -i ~/.ssh/id_ed25519  # then, in /workspace/tiny-grounder, inside tmux:
gpu/setup_pod.sh                            # uv, project env, vLLM in /workspace/vllm-venv
gpu/serve_teacher.sh gemma                  # log: runs/vllm.log
gpu/pilot.sh gemma                          # ~50 dev pages
gpu/run_teacher.sh gemma                    # web train/dev/test + data/examples (cache: data/teacher_cache/gemma)
gpu/run_teacher_native.sh gemma             # data/teacher/gemma-<source>-<split>.jsonl
gpu/sync.sh down HOST PORT                  # on the Mac: teacher outputs, cache, examples, runs/
```

Then continue on the Mac with `tools/mix.py` and `scripts/train.sh`; the outputs are named `gemma-<source>-<split>.jsonl` (and `gemma-<split>.jsonl` for web), so run `build` by hand as in `scripts/assemble.sh`. Stop and delete the pod when `gpu/sync.sh down` is done (stopping alone keeps billing the volume). Cost at about $0.5/h: roughly $1.5 to $4 for the teacher at 600 to 2000 completion tokens/s.
