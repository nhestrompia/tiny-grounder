#!/usr/bin/env bash
# Teacher on ~50 dev pages: rejection counts, throughput and random kept examples to read.
# usage: gpu/pilot.sh [gemma|qwen]   (writes runs/pilot-<model>/; PILOT_PAGES=50 sets the page count)
set -euo pipefail
cd "$(dirname "$0")/.."
source gpu/env.sh
pick_model "${1:-gemma}"
out=runs/pilot-$MODEL
mkdir -p "$out"

run_teacher_split dev "$out/teacher.jsonl" --limit-pages "${PILOT_PAGES:-50}" --log-every 10 2>&1 | tee "$out/teacher.log"
uv run python -m tools.pilot_report --pages "$PAGES/dev" --examples "$out/teacher.jsonl" --per-kind 30 > "$out/samples.txt"

echo
echo "--- throughput (last progress line)"
grep ' calls, ' "$out/teacher.log" | tail -n 1
echo "--- samples: $out/samples.txt"
cat "$out/samples.txt"
