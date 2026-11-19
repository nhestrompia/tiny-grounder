#!/usr/bin/env bash
# Teacher over the macOS accessibility pages (our own dumps and Screen2AX), kept apart from the web outputs.
# Writes data/teacher/<model>-<source>-<split>.jsonl for every split each source has; merged into training on the Mac.
# Resumable like run_teacher.sh. usage: gpu/run_teacher_native.sh gemma|qwen
set -euo pipefail
cd "$(dirname "$0")/.."
source gpu/env.sh
[ $# -eq 1 ] || { echo "usage: $0 gemma|qwen" >&2; exit 2; }
pick_model "$1"
out=$DATA_DIR/teacher
mkdir -p "$out"

for source in pages_ax pages_s2ax; do
  for split in train dev test; do
    [ -d "$DATA_DIR/$source/$split" ] || continue
    PAGES=$DATA_DIR/$source run_teacher_split "$split" "$out/$MODEL-$source-$split.jsonl" --log-every 200 \
      2>&1 | tee -a "$out/$MODEL-native.log"
  done
done
