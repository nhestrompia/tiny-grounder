#!/usr/bin/env bash
# Start the vLLM teacher server in the background and wait until it answers.
# usage: gpu/serve_teacher.sh [gemma|qwen]   (log: runs/vllm.log; stop: kill $(cat runs/vllm.pid))
set -euo pipefail
cd "$(dirname "$0")/.."
source gpu/env.sh
pick_model "${1:-gemma}"
mkdir -p runs

if curl -sf "$TEACHER_BASE_URL/models" >/dev/null && [ "$(cat runs/vllm.model 2>/dev/null)" = "$MODEL" ]; then
  echo "$MODEL teacher already serving"
  exit 0
fi
if [ -f runs/vllm.pid ] && kill -0 "$(cat runs/vllm.pid)" 2>/dev/null; then
  kill "$(cat runs/vllm.pid)"
  while kill -0 "$(cat runs/vllm.pid)" 2>/dev/null; do sleep 1; done
fi

# Prompts are at most ~1.7k tokens and replies ~0.9k, so 4096 leaves headroom and keeps KV cache for more sequences.
# shellcheck disable=SC2086
nohup "$VLLM_VENV/bin/vllm" serve "$MODEL_ID" --served-model-name teacher $VLLM_ARGS \
  --max-model-len 4096 --gpu-memory-utilization 0.92 --max-num-seqs "$MAX_NUM_SEQS" \
  --port "$VLLM_PORT" ${VLLM_EXTRA_ARGS:-} > runs/vllm.log 2>&1 &
echo $! > runs/vllm.pid
echo "$MODEL" > runs/vllm.model

echo "loading $MODEL_ID (first start downloads the weights); log: runs/vllm.log"
until curl -sf "$TEACHER_BASE_URL/models" >/dev/null; do
  if ! kill -0 "$(cat runs/vllm.pid)" 2>/dev/null; then
    tail -n 30 runs/vllm.log
    echo "vllm exited before it was ready" >&2
    exit 1
  fi
  sleep 5
done
echo "$MODEL teacher ready at $TEACHER_BASE_URL"
