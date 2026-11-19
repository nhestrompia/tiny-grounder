#!/usr/bin/env bash
# The LLM teacher through a hosted OpenAI-compatible API (OpenRouter), run from the Mac. Resumable: finished
# calls come from data/teacher_cache/<model>, one cache per model.
#
#   scripts/teacher_api.sh pilot MODEL   50 English dev pages + 20 desktop windows (seeded, identical for every model)
#                                        -> runs/pilot-<model>/{teacher.jsonl,teacher.log,samples.txt,usage-*.json}
#   scripts/teacher_api.sh full MODEL    web train/dev/test and the macOS pages
#                                        -> data/teacher/<model>-<source>-<split>.jsonl
#
# MODEL is the OpenRouter id, e.g. google/gemma-4-26b-a4b-it; <model> in paths is that id with "/" etc. replaced by "_".
# Needs OPENROUTER_API_KEY. Optional: TEACHER_CONCURRENCY (32), MAX_COST_USD (stop launching calls above it, per
# teacher run), PILOT_WEB_PAGES (50), PILOT_NATIVE_PAGES (20), MIN_ENGLISH.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/teacher_lib.sh
[ $# -eq 2 ] && [[ $1 == pilot || $1 == full ]] || { echo "usage: $0 pilot|full MODEL" >&2; exit 2; }
mode=$1 slug=$2
[ -n "${OPENROUTER_API_KEY:-}" ] || { echo "set OPENROUTER_API_KEY" >&2; exit 2; }
safe=$(slug_safe "$slug")

TEACHER_BASE_URL=${TEACHER_BASE_URL:-https://openrouter.ai/api/v1}
TEACHER_MODEL=$slug
TEACHER_API_KEY_ENV=OPENROUTER_API_KEY
TEACHER_CONCURRENCY=${TEACHER_CONCURRENCY:-32}
# require_parameters: only route to providers that honour response_format json_schema; usage.include: report the cost.
EXTRA_BODY='{"provider": {"require_parameters": true}, "usage": {"include": true}}'
CACHE_DIR=$DATA_DIR/teacher_cache/$safe

# api_run OUT_JSONL TEACHER_ARGS...: more retries than the default, for 429s at this concurrency, and the optional budget.
api_run() {
  local out=$1
  shift
  teacher_run "$out" "$@" --retries 6 ${MAX_COST_USD:+--max-cost-usd "$MAX_COST_USD"}
}

pilot() {
  local out=runs/pilot-$safe web=$DATA_DIR/pages native=("$DATA_DIR/pages_s2ax/train" "$DATA_DIR/pages_ax/train")
  mkdir -p "$out"
  : > "$out/teacher.log"
  api_run "$out/teacher-web.jsonl" --pages "$web" --split dev --limit-pages "${PILOT_WEB_PAGES:-50}" --log-every 10 \
    --summary-out "$out/usage-web.json" 2>&1 | tee -a "$out/teacher.log"
  api_run "$out/teacher-native.jsonl" --pages "${native[@]}" --limit-pages "${PILOT_NATIVE_PAGES:-20}" --log-every 10 \
    --summary-out "$out/usage-native.json" 2>&1 | tee -a "$out/teacher.log"
  cat "$out/teacher-web.jsonl" "$out/teacher-native.jsonl" > "$out/teacher.jsonl"
  uv run python -m tools.pilot_report --pages "$web/dev" "${native[@]}" --examples "$out/teacher.jsonl" --per-kind 30 \
    --summary "$out"/usage-web.json "$out"/usage-native.json > "$out/samples.txt"
  echo "--- samples: $out/samples.txt"
  cat "$out/samples.txt"
}

full() {
  local out=$DATA_DIR/teacher source split
  mkdir -p "$out"
  for split in train dev test; do
    api_run "$out/$safe-web-$split.jsonl" --pages "$DATA_DIR/pages" --split "$split" --log-every 200 \
      --summary-out "$out/$safe-web-$split.usage.json" 2>&1 | tee -a "$out/$safe.log"
  done
  for source in pages_ax pages_s2ax; do
    for split in train dev test; do
      [ -d "$DATA_DIR/$source/$split" ] || continue
      api_run "$out/$safe-$source-$split.jsonl" --pages "$DATA_DIR/$source" --split "$split" --log-every 200 \
        --summary-out "$out/$safe-$source-$split.usage.json" 2>&1 | tee -a "$out/$safe.log"
    done
  done
}

"$mode"
