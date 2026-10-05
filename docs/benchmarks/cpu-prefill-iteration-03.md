# CPU prefill iteration 03: tiled attention and VNNI experiment

This is an intermediate, three-repetition diagnostic, not a victory claim or a full numerical/quality certification. The latency target is still missed. All trials are retained in the adjacent CSV; medians and ranges are in the JSON.

## Fixed conditions

- Same Qwen3-0.6B Q8_0 model and raw input IDs as iteration 01
- Four pinned virtual CPU cores on the same AMD EPYC 9V74 host
- Context 2048; full-prompt plus eight-step warmup; 128 additional timed decode calls and selections (129 emitted IDs)
- Optimized CPU-native builds; llama.cpp batched prefill remains enabled
- Each of three blocks rotates nn, llama F32 KV and llama standard F16 KV
- All heavy builds and unrelated inference were stopped during measurement
- Virtualized host; physical-host contention, temperature and energy were not controlled/measured

## Observed medians

| Input | Engine | TTFT (s) | Input + generation (s) | Decode (tokens/s) | Peak RSS (MiB) |
|---:|---|---:|---:|---:|---:|
|128|nn|0.716|3.917|39.99|671.3|
|128|llama F32|0.554|3.927|38.90|1114.3|
|128|llama F16|0.532|3.441|44.13|889.9|
|1024|nn|6.531|11.512|26.36|869.0|
|1024|llama F32|4.050|9.395|23.95|1133.1|
|1024|llama F16|4.229|8.922|27.41|908.8|

nn is still slower than standard F16 llama.cpp for both first-token and complete-request latency. The short-input F32 total is effectively tied at this small sample size. No speed superiority is established. Relative to the initial, separately measured sequential nn result (26.78 seconds at input1024), the new 6.53-second median demonstrates a substantial prefill improvement, but the runs are from different times and are not a paired causal estimate.

## Changes and correctness

The candidate adds AVX512-VNNI batched Q8 GEMM and head-major KV panels with four-query attention. Single-token decode math is unchanged. Stage profiles identified attention as a major cost; the tiled attention profile fell from roughly3.5 to1.5 seconds at input1024. Profiles are diagnostic and are not substituted for the balanced measurements above.

The frozen candidate SHA256 is `c98669c85171dfcadfc5c406d8cde3b5a9feaa95e4e985235bfa6bc874ca3139`. It passed 1,370 tiny-model checkpoints at one/four threads and 374 selected real-model checkpoints at four threads, all bit-identical to the immutable original nn. The real-model tier covers20 prompts plus the actual128/1024-token workloads, each with16 teacher-forced continuations. The complete long-context real-model suite at both thread counts remains pending. This establishes checked nn-to-original equivalence only, not llama-to-nn quality equivalence or downstream model accuracy.

These runs predate the build-attestation and runtime-override recording hardening in the portable tools. The native source was frozen before building with Almide0.66.0 `--fast`; accepted numerical runs used default backend settings. New confirmatory runs will use the hardened evidence checks. An earlier profile killed with exit137 while compilation competed for memory is excluded from successful timing evidence and retained separately as a failed attempt.

## Expanded cross-engine numerical diagnostic

The original40-point smoke was expanded to374 checkpoints:20 prompts and the128/1024-token performance inputs, with16 identical fixed LCG teacher-forced continuations each. nn agrees with llama.cpp F32 top1 at360/374 (96.3%). All logits are finite; minimum cosine similarity is0.996810, median KL(llama→nn)0.001421 and maximum0.017658. Some disagreements have meaningful margins (largest llama gap between chosen tokens0.2906), so they cannot all be described as insignificant ties.

These differences already exist in original nn. The optimized iteration03 results are bit-identical to original nn at every checkpoint. The speed work preserves that baseline; this does not establish equal nn/llama model quality. The artificial continuations make this a numerical diagnostic, not language perplexity. All374 checkpoint metrics are in `cpu-cross-engine-expanded-checkpoints.csv`, with the summary in `cpu-cross-engine-expanded.json`.

## Next work

The remaining GEMM cost is the principal target. Experimental backend results are mixed across matrix shapes, so no shape-specific microbenchmark is being presented as a whole-model win. Next candidates must pass the unchanged finite-logit,1e-3 absolute /1e-4 normalized maximum-error and exact-top1 gates before comparative timing. The final comparison must include both llama KV formats and at least ten balanced repetitions per condition.
