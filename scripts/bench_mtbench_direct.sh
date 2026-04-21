#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:${PORT:-8001}}"
ENDPOINT="${ENDPOINT:-/v1/chat/completions}"
TOKENIZER="${TOKENIZER:-google/gemma-4-31B-it}"
DATASET_NAME="${DATASET_NAME:-hf}"
DATASET_PATH="${DATASET_PATH:-philschmid/mt-bench}"
NUM_PROMPTS="${NUM_PROMPTS:-80}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
MODEL="${MODEL:-google/gemma-4-31B-it-vllm-fp8-dflash-16k}"
HF_OUTPUT_LEN="${HF_OUTPUT_LEN:-2048}"
TEMPERATURE="${TEMPERATURE:-0}"
RESULT_DIR="${RESULT_DIR:-/workspace/results/mtbench}"

mkdir -p "$RESULT_DIR"

vllm bench serve \
  --backend openai-chat \
  --base-url "$BASE_URL" \
  --endpoint "$ENDPOINT" \
  --dataset-name "$DATASET_NAME" \
  --tokenizer "$TOKENIZER" \
  --dataset-path "$DATASET_PATH" \
  --num-prompts "$NUM_PROMPTS" \
  --max-concurrency "$MAX_CONCURRENCY" \
  --model "$MODEL" \
  --hf-output-len "$HF_OUTPUT_LEN" \
  --temperature "$TEMPERATURE" \
  --result-dir "$RESULT_DIR" \
  --save-result \
  --save-detailed

echo "benchmark_results=$RESULT_DIR"
