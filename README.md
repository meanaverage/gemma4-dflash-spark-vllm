# Gemma 4 + Red Hat DFlash on DGX Spark GB10

This repository documents a working single-node bring-up of `RedHatAI/gemma-4-31B-it-speculator.dflash` against `google/gemma-4-31B-it` on a DGX Spark / GB10 using `vLLM` nightly.

## Quick Start: `docker run`

If you are on a DGX Spark / GB10 and just want a runnable container, the shortest path is the published image:

```bash
IMAGE=ghcr.io/meanaverage/gemma4-dflash-spark-vllm:latest
HF_CACHE_DIR=${HF_CACHE_DIR:-$HOME/.cache/huggingface}

mkdir -p "$HF_CACHE_DIR"
```

Run the DFlash path directly:

```bash
docker run --rm -d \
  --name gemma4-google-fp8-dflash \
  --gpus all \
  --network host \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$HF_CACHE_DIR:/root/.cache/huggingface" \
  -e PORT=8001 \
  -e SERVED_MODEL_NAME=google/gemma-4-31B-it-vllm-fp8-dflash-16k \
  "$IMAGE" dflash
```

Run the baseline verifier path directly:

```bash
docker run --rm -d \
  --name gemma4-google-fp8 \
  --gpus all \
  --network host \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$HF_CACHE_DIR:/root/.cache/huggingface" \
  -e PORT=8002 \
  -e SERVED_MODEL_NAME=google/gemma-4-31B-it-vllm-fp8-16k \
  "$IMAGE" baseline
```

Check that the servers are up:

```bash
curl http://127.0.0.1:8001/v1/models
curl http://127.0.0.1:8002/v1/models
```

Run the same `mt-bench` comparison from the image itself:

```bash
mkdir -p results

docker run --rm \
  --network host \
  -v "$(pwd)/results:/workspace/results" \
  -e BASE_URL=http://127.0.0.1:8002 \
  -e MODEL=google/gemma-4-31B-it-vllm-fp8-16k \
  -e RESULT_DIR=/workspace/results/baseline \
  "$IMAGE" bench

docker run --rm \
  --network host \
  -v "$(pwd)/results:/workspace/results" \
  -e BASE_URL=http://127.0.0.1:8001 \
  -e MODEL=google/gemma-4-31B-it-vllm-fp8-dflash-16k \
  -e RESULT_DIR=/workspace/results/dflash \
  "$IMAGE" bench
```

## Repository Scope

This repository is aimed at users who want to:

- reproduce the exact serve path that was validated
- understand the runtime patches that made it work
- compare baseline Gemma 4 FP8 serving against DFlash on the same node

## Attribution

This repository, the scripts in it, and the initial documentation were assembled by OpenAI Codex from a live working environment, then reviewed and published by the repository owner.

## What Is Included

- `Dockerfile`
  Thin image on top of `vllm/vllm-openai:nightly` with the patcher and entrypoint baked in.
- `docker/entrypoint.sh`
  Mode-based container entrypoint for `baseline`, `dflash`, and `bench`.
- `scripts/patch_vllm_gb10_gemma4_dflash_runtime.py`
  Runtime patcher for the current `vLLM` nightly path validated here.
- `scripts/serve_gemma4_google_fp8.sh`
  Baseline verifier-only serve script.
- `scripts/serve_gemma4_google_fp8_dflash.sh`
  DFlash serve script using the Red Hat draft model.
- `scripts/bench_mtbench_direct.sh`
  In-container `mt-bench` helper used by the image `bench` mode.
- `scripts/run_vllm_container.sh`
  Generic Docker launcher for the two serve scripts.
- `scripts/bench_mtbench.sh`
  Simple `mt-bench` helper against a running container.

## Verified Environment

The following environment was validated directly:

- Hardware: single NVIDIA DGX Spark / GB10
- Container image: `vllm/vllm-openai:nightly`
- `vLLM`: `0.19.2rc1.dev21+g893611813`
- Verifier: `google/gemma-4-31B-it`
- Draft: `RedHatAI/gemma-4-31B-it-speculator.dflash`
- Quantization: runtime `fp8`
- Mode: text-only
- Context: `16384`
- Tensor parallelism: `1`

## Why The Patch Is Needed

The working path is not a stock `vllm serve` invocation.

The three runtime changes that mattered were:

1. Disable the prebuilt CUTLASS FP8 capability checks for GB10 so `vLLM` falls back to supported kernels.
2. Advertise non-causal support in the Triton backend selector so DFlash can initialize.
3. Force `FlashAttentionBackend` only for the `qwen3_dflash` draft attention path.

The practical result is a split-backend setup:

- Gemma 4 verifier stays on Triton
- DFlash draft attention runs on FlashAttention

## Run From Source Instead

Clone the repository and enter it:

```bash
git clone https://github.com/meanaverage/gemma4-dflash-spark-vllm.git
cd gemma4-dflash-spark-vllm
```

Pull the nightly image:

```bash
docker pull vllm/vllm-openai:nightly
```

Start the baseline verifier on port `8002`:

```bash
CONTAINER_NAME=gemma4-google-fp8 \
PORT=8002 \
SERVED_MODEL_NAME=google/gemma-4-31B-it-vllm-fp8-16k \
ENTRYPOINT_SCRIPT=/workspace/scripts/serve_gemma4_google_fp8.sh \
bash scripts/run_vllm_container.sh
```

Start the DFlash variant on port `8001`:

```bash
CONTAINER_NAME=gemma4-google-fp8-dflash \
PORT=8001 \
SERVED_MODEL_NAME=google/gemma-4-31B-it-vllm-fp8-dflash-16k \
ENTRYPOINT_SCRIPT=/workspace/scripts/serve_gemma4_google_fp8_dflash.sh \
bash scripts/run_vllm_container.sh
```

Check the server:

```bash
curl http://127.0.0.1:8001/v1/models
curl http://127.0.0.1:8002/v1/models
```

Run the same `mt-bench` harness used for the comparison:

```bash
CONTAINER_NAME=gemma4-google-fp8 \
PORT=8002 \
MODEL=google/gemma-4-31B-it-vllm-fp8-16k \
RESULT_NAME=baseline \
bash scripts/bench_mtbench.sh

CONTAINER_NAME=gemma4-google-fp8-dflash \
PORT=8001 \
MODEL=google/gemma-4-31B-it-vllm-fp8-dflash-16k \
RESULT_NAME=dflash \
bash scripts/bench_mtbench.sh
```

The benchmark helper installs `datasets` inside the running container if needed because the stock nightly image may not include the benchmark extras.

## Observed Results

These numbers are from live server metrics during an `mt-bench` run with:

- `num-prompts=80`
- `max-concurrency=1`
- `hf-output-len=2048`
- `temperature=0`

Observed results:

| Variant | Avg generation throughput | Min | Max |
| --- | ---: | ---: | ---: |
| Baseline Gemma 4 FP8 | `5.53 tok/s` | `5.3` | `5.6` |
| Gemma 4 FP8 + DFlash | `15.44 tok/s` | `9.9` | `28.1` |

Observed DFlash acceptance:

- average draft acceptance rate: `28.82%`
- observed range: `15.1%` to `62.2%`

That is roughly a `2.79x` throughput uplift over the baseline in this specific single-node, concurrency-1 test shape.

## Caveats

- This is a runtime patch against a nightly wheel, not an upstreamed fix set.
- This bundle documents the verified `google/gemma-4-31B-it` verifier path only.
- The validated path here is text-only.
- The results above are for single-node `TP=1` and should not be generalized beyond that setup without retesting.

## License

This repository is licensed under `Apache-2.0`. See `LICENSE`.

That license applies to the code and documentation in this repository only.

Model weights, container images, and upstream projects referenced here remain under their own licenses and terms.
