#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL="${MODEL:-google/gemma-4-31B-it}"
DRAFT_MODEL="${DRAFT_MODEL:-RedHatAI/gemma-4-31B-it-speculator.dflash}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-google/gemma-4-31B-it-vllm-fp8-dflash-16k}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
QUANTIZATION="${QUANTIZATION:-fp8}"
TEXT_ONLY="${TEXT_ONLY:-1}"

export VLLM_DISABLE_COMPILE_CACHE="${VLLM_DISABLE_COMPILE_CACHE:-1}"

python3 "$WORKSPACE_ROOT/scripts/patch_vllm_gb10_gemma4_dflash_runtime.py"

SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG:-$(python3 - "$DRAFT_MODEL" "$NUM_SPECULATIVE_TOKENS" <<'PY'
import json
import sys

print(
    json.dumps(
        {
            "model": sys.argv[1],
            "num_speculative_tokens": int(sys.argv[2]),
            "method": "dflash",
        },
        separators=(",", ":"),
    )
)
PY
)}"

CMD=(
  vllm serve "$MODEL"
  --host "$HOST"
  --port "$PORT"
  --served-model-name "$SERVED_MODEL_NAME"
  --trust-remote-code
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --quantization "$QUANTIZATION"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --enforce-eager
  --speculative-config "$SPECULATIVE_CONFIG"
)

if [[ "$TEXT_ONLY" == "1" ]]; then
  CMD+=(--limit-mm-per-prompt '{"image":0,"video":0}')
fi

exec "${CMD[@]}"
