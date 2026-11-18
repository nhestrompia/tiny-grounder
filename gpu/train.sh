#!/usr/bin/env bash
# Tokenizer, then XS/S/M/L with one seed, each evaluated (plus BM25) on dev and test and exported.
# Resumable: a model with test metrics on disk is skipped; an interrupted one retrains from scratch.
# usage: gpu/train.sh   (WORKERS=8 data-loader processes, DEVICE=cuda, EPOCHS=10; on the Mac: DEVICE=mps WORKERS=4)
set -euo pipefail
cd "$(dirname "$0")/.."
source gpu/env.sh
examples=$DATA_DIR/examples
out=${TRAIN_DIR:-runs/train}
mkdir -p "$out"

[ -f "$out/tokenizer.json" ] || uv run python -m grounder.tokenizer train \
  --pages "$PAGES/train" --examples "$examples/train.jsonl" --out "$out/tokenizer.json"

evaluate() { # evaluate NAME ARGS...: dev and test metrics into $out/NAME/{dev,test}.json
  local name=$1
  shift
  for split in dev test; do
    uv run python -m grounder.eval --pages "$PAGES" --examples "$examples/$split.jsonl" "$@" \
      --out "$out/$name/$split.json" > "$out/$name/$split.md"
  done
}

for name in xs s m l; do
  mkdir -p "$out/$name"
  if [ ! -f "$out/$name/test.json" ]; then
    uv run python -m grounder.train --config "configs/$name.yaml" --pages "$PAGES" --train "$examples/train.jsonl" \
      --dev "$examples/dev.jsonl" --tokenizer "$out/tokenizer.json" --out "$out/$name" --seed 0 \
      --epochs "${EPOCHS:-10}" --workers "${WORKERS:-8}" --device "${DEVICE:-cuda}" 2>&1 | tee "$out/$name/train.log"
    evaluate "$name" --checkpoint "$out/$name/model.pt" --device "${DEVICE:-cuda}"
    uv run python -m grounder.export --checkpoint "$out/$name/model.pt" --out "${DIST_DIR:-dist}/$name" \
      --pages "$PAGES" --examples "$examples/test.jsonl"
  fi
done
mkdir -p "$out/bm25"
[ -f "$out/bm25/test.json" ] || evaluate bm25 --baseline bm25

uv run python -m tools.results_table --runs "$out" --models xs s m l bm25 | tee "$out/results.md"
