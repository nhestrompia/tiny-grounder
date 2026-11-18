#!/usr/bin/env bash
# Teacher over train, dev and test pages, then assemble the example JSONL files.
# Resumable: rerun the same command and finished calls come from the cache.
# usage: gpu/run_teacher.sh gemma|qwen
set -euo pipefail
cd "$(dirname "$0")/.."
source gpu/env.sh
[ $# -eq 1 ] || { echo "usage: $0 gemma|qwen" >&2; exit 2; }
pick_model "$1"
out=$DATA_DIR/teacher
mkdir -p "$out"

for split in train dev test; do
  run_teacher_split "$split" "$out/$MODEL-$split.jsonl" --log-every 200 2>&1 | tee -a "$out/$MODEL.log"
done
uv run python -m grounder.data.build --pages "$PAGES" --out "$DATA_DIR/examples" \
  --llm-examples "$out/$MODEL-train.jsonl" "$out/$MODEL-dev.jsonl" "$out/$MODEL-test.jsonl"
