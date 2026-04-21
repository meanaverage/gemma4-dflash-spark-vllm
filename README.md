# Gemma 4 + Red Hat DFlash on DGX Spark GB10

This bundle contains the minimum files we used to bring up `RedHatAI/gemma-4-31B-it-speculator.dflash` against `google/gemma-4-31B-it` on a single DGX Spark / GB10 node with `vLLM` nightly.

It is intentionally small and sanitized:

- no hostnames
- no usernames
- no personal paths
- no local cluster references

## What Is Included

- `scripts/patch_vllm_gb10_gemma4_dflash_runtime.py`
  Runtime patcher for the current `vLLM` nightly path we used.
- `scripts/serve_gemma4_google_fp8.sh`
  Baseline verifier-only serve script.
- `scripts/serve_gemma4_google_fp8_dflash.sh`
  DFlash serve script using the Red Hat draft model.
- `scripts/run_vllm_container.sh`
  Generic Docker launcher for the two serve scripts.
- `scripts/bench_mtbench.sh`
  Simple `mt-bench` helper against a running container.
- `docs/dgx-forums-post.md`
  Forum-ready writeup you can adapt for NVIDIA DGX forums.

## Verified Environment

This is the environment we actually validated:

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

The working path was not a stock `vllm serve` invocation.

The three runtime changes that mattered were:

1. Disable the prebuilt CUTLASS FP8 capability checks for GB10 so `vLLM` falls back to supported kernels.
2. Advertise non-causal support in the Triton backend selector so DFlash can initialize.
3. Force `FlashAttentionBackend` only for the `qwen3_dflash` draft attention path.

The practical result is a split-backend setup:

- Gemma 4 verifier stays on Triton
- DFlash draft attention runs on FlashAttention

## Quick Start

Pull the nightly image first:

```bash
docker pull vllm/vllm-openai:nightly
```

Start the baseline verifier on port `8002`:

```bash
cd publish/gemma4-dflash-spark-vllm

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

Run the same `mt-bench` harness we used:

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
- No license file is included yet. Pick one before publishing the repo publicly.

## Suggested Publish Flow

Initialize a repo in this directory:

```bash
cd publish/gemma4-dflash-spark-vllm
git init
git add .
git commit -m "Initial Gemma 4 DFlash Spark GB10 bring-up bundle"
```

Then create a GitHub repo and add the remote:

```bash
git remote add origin <your-github-repo-url>
git branch -M main
git push -u origin main
```
