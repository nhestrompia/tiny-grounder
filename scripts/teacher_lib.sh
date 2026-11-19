# shellcheck shell=bash disable=SC2034
# Teacher helpers shared by scripts/teacher_api.sh, scripts/assemble.sh and gpu/env.sh (sourced, not run).
DATA_DIR=${DATA_DIR:-data}
# Optional override of the English-page threshold (--min-english) for the teacher and the dataset build;
# unset, both use the default of grounder.data.language.
MIN_ENGLISH_FLAG=${MIN_ENGLISH:+--min-english=$MIN_ENGLISH}

# slug_safe MODEL_ID: a name usable in file names ("google/gemma-4-26b-a4b-it" -> "google_gemma-4-26b-a4b-it").
slug_safe() { printf %s "$1" | tr -c 'A-Za-z0-9._-' '_'; }

# teacher_run OUT_JSONL TEACHER_ARGS...: the teacher CLI with the endpoint settings below; the cache makes a rerun
# resume where it stopped. Needs TEACHER_BASE_URL, TEACHER_MODEL, TEACHER_CONCURRENCY, EXTRA_BODY and CACHE_DIR;
# TEACHER_API_KEY_ENV (env var holding the key) and TEACHER_TIMEOUT are optional.
teacher_run() {
  local out=$1
  shift
  # shellcheck disable=SC2086
  uv run python -m grounder.data.teacher --out "$out" --base-url "$TEACHER_BASE_URL" --model "$TEACHER_MODEL" \
    ${TEACHER_API_KEY_ENV:+--api-key-env "$TEACHER_API_KEY_ENV"} --extra-body "$EXTRA_BODY" --cache-dir "$CACHE_DIR" \
    --concurrency "$TEACHER_CONCURRENCY" --timeout "${TEACHER_TIMEOUT:-300}" $MIN_ENGLISH_FLAG "$@"
}
