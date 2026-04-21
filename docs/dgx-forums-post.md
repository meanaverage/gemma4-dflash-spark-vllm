Posting a working DGX Spark / GB10 path for `RedHatAI/gemma-4-31B-it-speculator.dflash`, since the successful setup here was not the stock path and took a few runtime patches.

The sanitized repro bundle is here:

- https://github.com/meanaverage/gemma4-dflash-spark-vllm

There is also a thin published container path in that repo so DGX users can run the validated setup directly with `docker run` instead of cloning and wiring the scripts manually.

That bundle and its documentation were assembled by OpenAI Codex from the working environment, then reviewed and published by the repository owner.

Test rig

- Single NVIDIA DGX Spark / GB10
- `vllm/vllm-openai:nightly`
- `vLLM 0.19.2rc1.dev21+g893611813`
- Verifier: `google/gemma-4-31B-it`
- Draft: `RedHatAI/gemma-4-31B-it-speculator.dflash`
- Runtime quantization: `fp8`
- Text-only mode
- `max_model_len=16384`
- `max_num_batched_tokens=16384`
- `tensor_parallel_size=1`

The non-obvious part

This did not work for us as a simple “force FlashAttention everywhere” launch.

What actually worked was a split-backend path:

- Gemma 4 verifier stays on Triton
- DFlash draft attention in `qwen3_dflash` is forced onto FlashAttention only for the draft path

The three runtime patches we needed on top of `vLLM` nightly were:

1. Disable the prebuilt CUTLASS FP8 capability checks for GB10 so `vLLM` falls back to supported kernels.
2. Advertise non-causal support in the Triton backend selector so DFlash can initialize.
3. Force `FlashAttentionBackend` only inside `qwen3_dflash`.

Launch shape

We used:

- `VLLM_DISABLE_COMPILE_CACHE=1`
- `--quantization fp8`
- `--max-model-len 16384`
- `--max-num-batched-tokens 16384`
- `--gpu-memory-utilization 0.80`
- `--enforce-eager`
- `--limit-mm-per-prompt '{"image":0,"video":0}'`
- `--speculative-config '{"method":"dflash","model":"RedHatAI/gemma-4-31B-it-speculator.dflash","num_speculative_tokens":8}'`

We set `num_speculative_tokens=8` explicitly.

Benchmark shape

For an apples-to-apples serve comparison, we used:

- `vllm bench serve`
- dataset: `philschmid/mt-bench`
- `num-prompts=80`
- `max-concurrency=1`
- `hf-output-len=2048`
- `temperature=0`

Observed results

Against a plain non-DFlash baseline on the same verifier and same harness:

- baseline generation throughput: `5.53 tok/s` average
- DFlash generation throughput: `15.44 tok/s` average
- uplift: about `2.79x`
- DFlash average draft acceptance rate: `28.82%`
- observed DFlash generation range: `9.9` to `28.1 tok/s`
- observed DFlash acceptance range: `15.1%` to `62.2%`

So the short version is: this draft does run on DGX Spark / GB10 with the stock Google verifier, but today it is not a zero-patch path in `vLLM` nightly. The key was splitting verifier and draft attention backends instead of trying to drive both through one backend.

The exact launchers, runtime patcher, and benchmark helper are in the repo above so others can reproduce the same path without any cluster-specific details.
