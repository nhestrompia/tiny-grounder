# shellcheck shell=bash disable=SC2034
# Settings shared by the gpu/ scripts (sourced, not run). Override any of them from the environment.
VLLM_VENV=${VLLM_VENV:-/workspace/vllm-venv}
VLLM_PORT=${VLLM_PORT:-8000}
TEACHER_BASE_URL=${TEACHER_BASE_URL:-http://localhost:$VLLM_PORT/v1}
TEACHER_CONCURRENCY=${TEACHER_CONCURRENCY:-48}
DATA_DIR=${DATA_DIR:-data}
PAGES=${PAGES:-$DATA_DIR/pages}
# Optional override of the English-page threshold (--min-english) for the teacher and the dataset build;
# unset, both use the default of grounder.data.language.
MIN_ENGLISH_FLAG=${MIN_ENGLISH:+--min-english=$MIN_ENGLISH}
export HF_HOME=${HF_HOME:-/workspace/hf}
export PATH="$HOME/.local/bin:$PATH"

# pick_model gemma|qwen: sets MODEL, MODEL_ID, EXTRA_BODY (merged into every teacher request),
# VLLM_ARGS and MAX_NUM_SEQS.
pick_model() {
  MODEL=${1:-gemma}
  case $MODEL in
    gemma)
      MODEL_ID=google/gemma-4-12B-it
      EXTRA_BODY='{}'
      VLLM_ARGS="--quantization fp8"
      MAX_NUM_SEQS=${MAX_NUM_SEQS:-48}
      ;;
    qwen)
      MODEL_ID=Qwen/Qwen3.5-9B
      EXTRA_BODY='{"chat_template_kwargs": {"enable_thinking": false}}'
      VLLM_ARGS=""
      MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
      ;;
    *)
      echo "model must be gemma or qwen, got '$MODEL'" >&2
      exit 2
      ;;
  esac
}

# Run the teacher over one split of PAGES; the cache makes a rerun resume where it stopped.
# usage: run_teacher_split SPLIT OUT_JSONL [extra teacher args...]
run_teacher_split() {
  local split=$1 out=$2
  shift 2
  uv run python -m grounder.data.teacher --pages "$PAGES" --split "$split" --out "$out" \
    --base-url "$TEACHER_BASE_URL" --model teacher --extra-body "$EXTRA_BODY" \
    --cache-dir "$DATA_DIR/teacher_cache/$MODEL" --concurrency "$TEACHER_CONCURRENCY" --timeout 300 \
    $MIN_ENGLISH_FLAG "$@"
}
