#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-gemma4-google-fp8-dflash}"
PORT="${PORT:-8001}"
BASE_URL="${BASE_URL:-http://127.0.0.1:$PORT}"
ENDPOINT="${ENDPOINT:-/v1/chat/completions}"
TOKENIZER="${TOKENIZER:-google/gemma-4-31B-it}"
DATASET_NAME="${DATASET_NAME:-hf}"
DATASET_PATH="${DATASET_PATH:-philschmid/mt-bench}"
NUM_PROMPTS="${NUM_PROMPTS:-80}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
MODEL="${MODEL:-google/gemma-4-31B-it-vllm-fp8-dflash-16k}"
HF_OUTPUT_LEN="${HF_OUTPUT_LEN:-2048}"
TEMPERATURE="${TEMPERATURE:-0}"
RESULT_NAME="${RESULT_NAME:-mtbench}"
RESULT_DIR_IN_CONTAINER="${RESULT_DIR_IN_CONTAINER:-/workspace/results/$RESULT_NAME}"

docker exec "$CONTAINER_NAME" bash -lc "
set -euo pipefail
python3 - <<'PY'
try:
    import datasets  # noqa: F401
except Exception:
    raise SystemExit(1)
PY
" >/dev/null 2>&1 || docker exec "$CONTAINER_NAME" bash -lc "python3 -m pip install --no-input datasets"

docker exec "$CONTAINER_NAME" bash -lc "
set -euo pipefail
mkdir -p '$RESULT_DIR_IN_CONTAINER'
vllm bench serve \
  --backend openai-chat \
  --base-url '$BASE_URL' \
  --endpoint '$ENDPOINT' \
  --dataset-name '$DATASET_NAME' \
  --tokenizer '$TOKENIZER' \
  --dataset-path '$DATASET_PATH' \
  --num-prompts '$NUM_PROMPTS' \
  --max-concurrency '$MAX_CONCURRENCY' \
  --model '$MODEL' \
  --hf-output-len '$HF_OUTPUT_LEN' \
  --temperature '$TEMPERATURE' \
  --result-dir '$RESULT_DIR_IN_CONTAINER' \
  --save-result \
  --save-detailed
"

echo "benchmark_results=$RESULT_DIR_IN_CONTAINER"
