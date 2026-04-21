#!/usr/bin/env bash
set -euo pipefail

ROOT="/opt/gemma4-dflash-spark-vllm"
MODE="${1:-dflash}"

if [[ $# -gt 0 ]]; then
  shift
fi

usage() {
  cat <<'EOF'
Usage:
  <image> baseline
  <image> dflash
  <image> bench
  <image> bash

Modes:
  baseline   Run google/gemma-4-31B-it with runtime fp8
  dflash     Run google/gemma-4-31B-it with the Red Hat DFlash draft
  bench      Run the mt-bench helper against a running server
  bash       Open a shell in the image
EOF
}

case "$MODE" in
  baseline)
    exec bash "$ROOT/scripts/serve_gemma4_google_fp8.sh" "$@"
    ;;
  dflash)
    exec bash "$ROOT/scripts/serve_gemma4_google_fp8_dflash.sh" "$@"
    ;;
  bench)
    exec bash "$ROOT/scripts/bench_mtbench_direct.sh" "$@"
    ;;
  bash|shell)
    exec bash "$@"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    exec "$MODE" "$@"
    ;;
esac
