# CPU comparison after optimization: fixed-version results

## Result and scope

The optimized nn has lower **input-plus-generation median latency** than both pinned llama.cpp baselines at both tested input lengths. Every one of the five matched blocks favors nn for total latency. Against standard F16 KV, the reduction is **7.19% at input 128 and 26.99% at input 1024**. The short-input first-token difference remains uncertain; it must not be described as a decisive win. The initial aspirational 20% target for every metric/condition was not fully achieved.

These results concern one virtualized CPU host, one model/quantization and the fixed workloads below. They are not a universal engine, GPU, model-quality, energy or cold-start claim.

## Ten trials per engine and input

|Input tokens|Engine|TTFT seconds|Total inference seconds|Decode tokens/s|Peak RSS MiB|
|---:|---|---:|---:|---:|---:|
|128|nn|0.389|3.332|44.00|678.9|
|128|llama-f32|0.477|3.757|38.74|1114.4|
|128|llama-f16|0.418|3.590|40.75|890.0|
|1024|nn|2.996|6.470|36.04|881.3|
|1024|llama-f32|3.843|9.672|21.98|1132.8|
|1024|llama-f16|4.126|8.862|26.45|908.8|

Each column is summarized independently. TTFT includes prompt processing and the initial greedy selection, after model loading/warmup. Total includes TTFT plus 128 additional forward calls and greedy selections, producing 129 emitted IDs. No EOS early exit. RSS is the process high-water mark including load and warmup; lazy page allocation differs between engines. Load/warmup/process timings remain in the raw JSON rather than being mixed into warm-serving TTFT.

Against F32 KV, total latency falls 11.30% at input 128 and 33.10% at input 1024. Long-input TTFT falls 22.05% versus F32 and 27.40% versus F16. Short-input TTFT is 0.389s versus F16's0.418s, but only two of five paired blocks favor nn and the exploratory block interval crosses parity. No short-TTFT superiority claim is made.

The complete 60 trials are in `cpu-final-trials.csv`; medians, paired block ratios and descriptive bootstrap intervals are in `cpu-final-comparison.json`. No trial was dropped or replaced. Five paired blocks are too few to establish broad guarantees. Physical-host interference and temperature were uncontrolled, and no energy measurement was available.

## Fixed protocol

- Model: Qwen3-0.6B Q8_0,640,047,744 bytes; SHA256 `3d1cfc4b845efcda96d63415c986d51781d02b2f55f93fc6add142cd56b4e314`
- Model file is the pinned O6lvl4 GGUF conversion; this is not a claim that the GGUF upload is an official Qwen release
- CPU: virtualized AMD EPYC 9V74; four pinned vCPUs 0–3, explicit four-thread pools
- Same raw token IDs, context 2048, input 128/1024, full prompt plus 8 decode warmup calls
- Each block runs a rotated three-engine order followed by its reverse; five blocks yield ten trials per engine/input
- llama.cpp `8e1642198dcd4e408f8776222d6ae31b74d01187`, CPU-native optimized Release, normal batching 512/ubatch 512 and attention support retained
- Both controlled F32 KV and normal F16 KV are reported
- nn: Almide 0.66.0 (`819bbc74f`), Rust 1.94.0, `--fast`/CPU-native optimized build
- Engine source SHA256 `a7e5d99e95e5f1c0ca06db6cc5eba4368d68452fdf71e249dab35fddd9b4e1f5`; measured binary SHA256 `6025c3aa9c9dffd0c4af52a675eee7889b1e11f22efd6db828e4a5cd10e3b4a3`
- Source implementation commit: `3d9fd2eebe3a7cd7f81ed06ebad5dc92366e650f`

## Correctness and decoder nonregression

The exhaustive real-model suite passed all 1,370 checkpoints: 41 cases at one and four threads, full-context/chunk boundaries, nonzero starts, continued KV state and exact top1. Every logit bit matches immutable original nn; maximum absolute and normalized error are both 0. The unchanged acceptance limits were 1e-3 absolute and 1e-4 normalized, plus finite values and exact top1.

Default and forced AVX2/row-attention tiny suites each passed 1,370 checkpoints and 3,168 sampled IDs with bit-identical continued logits. Ten native unit tests passed, including invalid projection bounds and mixed KV appends. Actual Almide chat build and real-model smoke passed.

A separate contemporaneous decoder-only ABBA/BAAB comparison verified that the prefill work did not regress generation. Ten runs per engine/input, zero permitted slowdown:

|Input|Original nn tokens/s|Optimized nn tokens/s|Median decode-duration reduction|
|---:|---:|---:|---:|
|128|45.574|50.897|10.44%|
|1024|28.906|38.163|24.26%|

All 40×128 measured IDs, warmup IDs and every final 151,936-logit vector were bit-identical. Tiny fresh/poisoned-KV/replay/fresh-reload validation passed. The exploratory one-sided 95% duration-ratio upper bounds were 0.9092 and 0.7678, below 1.0. Shared-process decoder evidence is not an end-to-end or per-engine memory comparison. Details are in `cpu-decode-regression.json` and the delivered regression harness/results.

## Cross-engine quality limitation

The expanded original-nn versus llama.cpp F32 diagnostic matched top1 at 360/374 checkpoints (96.3%), with some nontrivial margins. The optimization preserves original nn results at all tested points; it does not establish equal quality between engines. Fixed artificial teacher-forced continuations are a numerical check, not language perplexity or downstream task accuracy.

## What changed and what was rejected

The adopted implementation combines final-only prompt vocabulary projection, exact compact 16-row Q8 panels, bounded 512-token prefill, checked fast packing, blocked attention and primary head-major KV storage. It preserves the original accumulation/reduction order and sampling behavior. No additional persistent weight cache was introduced.

A faster reassociated GEMM failed 51 tiny logit checkpoints and was rejected without weakening thresholds. A decoder VNNI experiment regressed relevant microbenchmarks and was not adopted. An early profile killed by resource contention is retained as a failed attempt, not a measurement. A bounds issue found in fast-packing review was fixed and regression-tested before use. Earlier diagnostic runs remain separate from this fixed-version series.

## Reproduce and review

Use `tools/cpu-parity/README.md` for the immutable-oracle gates and `tools/cpu-bench/README.md` for pinned builds, workload counts and balanced measurements. The benchmark refuses a wrong model/source, missing build attestation, incomplete quality suite or diagnostic runtime override. Source, configuration, compiler/binary hashes and every final run are retained in the delivered archive. The large full-logit dumps and model weights are not bundled; their checkpoint results and complete regeneration inputs/code are included.

Issues: [#1](https://github.com/almide-graphics/nn/issues/1), [#2](https://github.com/almide-graphics/nn/issues/2), [#3](https://github.com/almide-graphics/nn/issues/3). The implementation is in [PR #4](https://github.com/almide-graphics/nn/pull/4). This report records the pre-merge validation; consult the PR for its current status. Tests described here were run locally. GitHub reported zero check runs for the measured implementation commit; no remote-CI success is claimed.
