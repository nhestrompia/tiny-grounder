#!/usr/bin/env bash
# Example files from pages + one teacher's outputs (web and macOS, with template examples), then the source mix.
# usage: scripts/assemble.sh MODEL     (MODEL as given to teacher_api.sh; writes data/examples, data/examples-macos, data/mix)
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/teacher_lib.sh
[ $# -eq 1 ] || { echo "usage: $0 MODEL" >&2; exit 2; }
safe=$(slug_safe "$1")
teacher=$DATA_DIR/teacher
shopt -s nullglob

web=("$teacher/$safe"-web-{train,dev,test}.jsonl)
macos=("$teacher/$safe"-pages_ax-*.jsonl "$teacher/$safe"-pages_s2ax-*.jsonl)
for f in "${web[@]}"; do
  [ -f "$f" ] || { echo "missing $f; run scripts/teacher_api.sh full $1" >&2; exit 1; }
done

uv run python -m grounder.data.build --pages "$DATA_DIR/pages" --out "$DATA_DIR/examples" --llm-examples "${web[@]}" $MIN_ENGLISH_FLAG
uv run python -m grounder.data.build --pages "$DATA_DIR/pages_ax" "$DATA_DIR/pages_s2ax" --out "$DATA_DIR/examples-macos" \
  --llm-examples "${macos[@]}" $MIN_ENGLISH_FLAG
uv run python -m tools.mix --data "$DATA_DIR" --out "$DATA_DIR/mix"
