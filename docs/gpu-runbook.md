# GPU runbook (RunPod, one RTX 4090)

Paid work: the LLM teacher. Training stays on the Mac by default (see "Where to train"); `gpu/train.sh` can run it on the pod instead. All scripts are in `gpu/`; rerunning any of them resumes where it stopped.

## Before renting (Mac, free)

1. Crawl finished (`pgrep -f grounder.data.crawl` prints nothing). Check it: `uv run python -m grounder.data.crawl --out data/pages --stats` — enough pages in `train`, `dev` and `test`, no site dominating.
2. `uv run pytest -q` passes.
3. SSH key: add `~/.ssh/id_ed25519.pub` in RunPod, Settings, SSH Public Keys.
4. Deploy: Community Cloud, RTX 4090, template **RunPod PyTorch** (CUDA 12.x), container disk 30 GB, **volume 80 GB** mounted at `/workspace`. The volume must hold the gemma weights (~24 GB bf16; vLLM quantises to FP8 while loading), two venvs (~16 GB) and the pages. Add 20 GB of volume if you also compare qwen. Weights that need a licence click: set `HF_TOKEN` on the pod.
5. From the pod's Connect tab, take HOST and PORT of "SSH over exposed TCP".

Billing starts at deploy. Do steps 1-3 first.

## On the pod

Run long commands inside `tmux` (`apt-get install -y tmux` if missing) so a dropped connection does not stop them.

```bash
gpu/sync.sh up HOST PORT                    # on the Mac: code + data/pages
ssh root@HOST -p PORT -i ~/.ssh/id_ed25519  # then, in /workspace/tiny-grounder:
gpu/setup_pod.sh                            # uv, project env, vLLM in /workspace/vllm-venv (~5-10 min)
gpu/serve_teacher.sh gemma                  # first start downloads weights; log: runs/vllm.log
gpu/pilot.sh gemma                          # ~50 dev pages, a few minutes
```

Read `runs/pilot-gemma/samples.txt` (the pilot also prints it). The teacher log line `... calls, N prompt tok/s, M completion tok/s` gives throughput; extrapolate: full-run hours = total completion tokens / (M x 3600), with ~16k calls x ~0.9k completion tokens.

### Decision checklist (gemma)

- [ ] Pass rate (kept / raw) at least 75%, and `failed_batches=0`. No single rejection reason above 10% (`conflicting_query` / `duplicate_query` high means the model repeats itself).
- [ ] Of 100 samples read (30 per kind, `plain`, `paraphrase`, `context`, `goal`), at least 90 are correct (the query really selects that element) and read like something a person would type. Goals must not name the element.
- [ ] "paraphrases sharing no word with the element" at least 95%.
- [ ] Projected full-run time and cost are acceptable (table below).

If it fails, fix the prompt (`SYSTEM_PROMPT` in `grounder/data/teacher.py`, edit on the pod or sync up again) and rerun the pilot. Optional comparison, same 50 pages: `gpu/serve_teacher.sh qwen && gpu/pilot.sh qwen`, then pick the model with the better checklist numbers and throughput. Load the other model only if the first fails; each load costs minutes of paid time. If the two are close, take the faster.

### Full run

```bash
gpu/run_teacher.sh gemma    # train, dev, test, then builds data/examples/*.jsonl; progress + ETA in the log
```

### Where to train

Measured on the Mac (Apple M5, `--device mps`, batch 64, data loader included), seconds per 1k examples and the 300k examples x 10 epochs = 3M-example sweep:

| | XS | S | M | L | all four |
|---|---|---|---|---|---|
| mps s/1k | 2.2 | 2.1 | 3.4 | 6.3 | |
| mps hours | 1.8 | 1.8 | 2.8 | 5.3 | ~12 h |
| cpu s/1k | 2.8 | 4.1 | 5.7 | 11.2 | |
| cpu hours | 2.3 | 3.4 | 4.8 | 9.3 | ~20 h |

The loader is ~1 s per 1k examples per process (token ids are cached per dataset), so on the Mac use `WORKERS=4`, and on a pod `WORKERS=8` or the vCPU count. **Default: train on the Mac overnight** (free): `DEVICE=mps WORKERS=4 gpu/train.sh`. Use the pod only to save wall-clock time: stop vLLM (`kill $(cat runs/vllm.pid)`), run `gpu/train.sh`, and read the first `epoch 0 ... (Ns)` line of M. Gate: M should take under 5 min per 300k-example epoch (the Mac needs ~17 min); if it does not, stop, sync down and train on the Mac.

### Finish

```bash
gpu/sync.sh down HOST PORT   # on the Mac: teacher outputs, cache, examples, runs/, dist/
```

Check `data/examples/*.jsonl` exists locally (and `runs/train/results.md` if you trained on the pod), **then stop and delete the pod** (stopping alone keeps billing the volume).

## Estimated cost

All numbers are estimates, not measurements; replace the throughput with what the pilot prints. Assumptions: 8k pages x ~2 calls = 16k calls, ~0.9k completion tokens each (14M total), price ~$0.5/h (check the current 4090 Community rate), setup + weights download ~0.5 h, pilot ~0.25 h.

| Aggregate decode throughput | Teacher run | Teacher cost | Plus setup/pilot |
|---|---|---|---|
| 600 tok/s | 6.7 h | $3.3 | $3.7 |
| 1200 tok/s | 3.3 h | $1.7 | $2.1 |
| 2000 tok/s | 2.0 h | $1.0 | $1.4 |

Training on the pod would add ~$1-3 (unmeasured; the Mac needs about 12 h for the four sizes). A qwen comparison pilot adds ~0.5 h ($0.25).

## Failure recovery

- Any script stopped (SSH drop, vLLM crash, out of memory): rerun the same command. The teacher cache (`data/teacher_cache/<model>`) returns finished calls instantly; `serve_teacher.sh` restarts vLLM if it is down; `train.sh` skips models whose `test.json` exists (an interrupted model retrains from scratch).
- vLLM out of memory or very slow: `MAX_NUM_SEQS=24 gpu/serve_teacher.sh gemma` (or lower `TEACHER_CONCURRENCY`), check `runs/vllm.log`.
- `failed_batches` above zero: rerun `gpu/run_teacher.sh gemma`; failed batches are retried, finished ones are free.
- Pod lost: the volume survives a stop, not a delete. Sync down regularly during long runs (`gpu/sync.sh down`) so the cache is on the Mac.
