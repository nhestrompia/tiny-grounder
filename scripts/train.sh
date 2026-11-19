#!/usr/bin/env bash
# Tokenizer, then every model in MODELS for every seed in SEEDS on the source mix (data/mix, see tools/mix.py), each
# evaluated (plus BM25) on the mixed dev set and the held-out test sets, and exported.
# Resumable: a finished run is skipped, an interrupted one retrains from scratch; existing eval files are kept.
#
# usage: scripts/train.sh
#   MODELS="xs s m"   configs to train (l is 8.2M parameters, over the 5M target; add it with MODELS="xs s m l")
#   SEEDS="0"         seeds per model; rerun the winner with SEEDS="0 1 2"
#   DEVICE            mps on macOS, else cuda       WORKERS=4   EPOCHS=10   TRAIN_DIR=runs/train   DIST_DIR=dist
# Eval sets: dev (the mix's dev sets, scored per source), web test, DeskForge test_app, Mind2Web test_task / test_website /
# test_domain, GUIrilla test_unseen_apps. GUIrilla is evaluation only: it is in no training, dev or tokenizer input.
set -euo pipefail
export PYTHONUNBUFFERED=1  # progress lines reach the log as they happen
# MPS: hard cap at the recommended working set (~25 GB on 32 GB) and return cached buffers above half of it;
# with the defaults the per-shape buffer cache of the M model passed 40 GB and swapped or ran out of memory.
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-1.0}
export PYTORCH_MPS_LOW_WATERMARK_RATIO=${PYTORCH_MPS_LOW_WATERMARK_RATIO:-0.5}
cd "$(dirname "$0")/.."
DATA=${DATA_DIR:-data}
MIX=$DATA/mix
out=${TRAIN_DIR:-runs/train}
device=${DEVICE:-$([ "$(uname)" = Darwin ] && echo mps || echo cuda)}
[ -f "$MIX/train.jsonl" ] || { echo "no $MIX/train.jsonl: run scripts/assemble.sh MODEL first" >&2; exit 1; }

# Pages of every training source, train splits only (the tokenizer), and with the dev splits (training / dev scoring).
TOKENIZER_PAGES=("$DATA/pages/train" "$DATA/pages_ax/train" "$DATA/pages_s2ax/train" "$DATA/deskforge/pages/train" "$DATA/mind2web/pages/train")
TRAIN_PAGES=("${TOKENIZER_PAGES[@]}" "$DATA/pages/dev" "$DATA/deskforge/pages/dev")

# name|examples|pages (space separated); the set names are the file names of the results.
EVAL_SETS=(
  "dev|$MIX/dev.jsonl|${TRAIN_PAGES[*]}"
  "web_test|$DATA/examples/test.jsonl|$DATA/pages/test"
  "deskforge_test_app|$DATA/deskforge/examples/test_app.jsonl|$DATA/deskforge/pages/test_app"
  "mind2web_test_task|$DATA/mind2web/test_task.jsonl|$DATA/mind2web/pages/test_task"
  "mind2web_test_website|$DATA/mind2web/test_website.jsonl|$DATA/mind2web/pages/test_website"
  "mind2web_test_domain|$DATA/mind2web/test_domain.jsonl|$DATA/mind2web/pages/test_domain"
  "guirilla_test|$DATA/guirilla/examples/test_unseen_apps.jsonl|$DATA/guirilla/pages/test"
)
mkdir -p "$out"

[ -f "$out/tokenizer.json" ] || uv run python -m grounder.tokenizer train \
  --pages "${TOKENIZER_PAGES[@]}" --examples "$MIX/train.jsonl" --out "$out/tokenizer.json"

evaluate() { # evaluate RUN_DIR GRADER_ARGS...: one NAME.json / NAME.md per eval set into RUN_DIR
  local run=$1 spec name examples pages
  shift
  for spec in "${EVAL_SETS[@]}"; do
    IFS='|' read -r name examples pages <<< "$spec"
    [ -f "$run/$name.json" ] && continue
    # shellcheck disable=SC2086  # $pages is a space-separated list of directories
    uv run python -m grounder.eval --pages $pages --examples "$examples" "$@" --out "$run/$name.json" > "$run/$name.md"
  done
}

runs=()
for model in ${MODELS:-xs s m}; do
  for seed in ${SEEDS:-0}; do
    name=$model-s$seed
    runs+=("$name")
    run=$out/$name
    mkdir -p "$run"
    if [ ! -f "$run/train.done" ]; then
      uv run python -m grounder.train --config "configs/$model.yaml" --pages "${TRAIN_PAGES[@]}" --train "$MIX/train.jsonl" \
        --dev "$MIX/dev.jsonl" --tokenizer "$out/tokenizer.json" --out "$run" --seed "$seed" \
        --epochs "${EPOCHS:-10}" --workers "${WORKERS:-4}" --device "$device" 2>&1 | tee "$run/train.log"
      touch "$run/train.done"
    fi
    evaluate "$run" --checkpoint "$run/model.pt" --device "$device"
    [ -f "${DIST_DIR:-dist}/$name/model.onnx" ] || uv run python -m grounder.export --checkpoint "$run/model.pt" \
      --out "${DIST_DIR:-dist}/$name" --pages "$DATA/pages/test" --examples "$DATA/examples/test.jsonl"
  done
done
mkdir -p "$out/bm25"
evaluate "$out/bm25" --baseline bm25

set_names=()
for spec in "${EVAL_SETS[@]}"; do set_names+=("${spec%%|*}"); done
uv run python -m tools.results_table --runs "$out" --models "${runs[@]}" bm25 --splits "${set_names[@]}" | tee "$out/results.md"
