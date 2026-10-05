# CPU iteration11: primary head-major KV and checked fast packing

This is a three-repetition diagnostic, not a confirmatory win. Both workloads show lower nn median first-token and total latency than both llama baselines, but several margins are small relative to observed virtual-host variation. The exhaustive real-model gate and a larger fixed-version comparison are in progress.

Same fixed protocol: Qwen3-0.6B Q8_0, four pinned virtual CPU cores, context2048, input128/1024, full-prompt+eight-step warmup,128 additional timed forward/selection calls and129 emitted IDs. Three rotated blocks; optimized llama batching and normal CPU features remain enabled. No other heavy jobs ran during timing.

|Input|Engine|TTFT seconds|Total inference seconds|Decode tokens/s|Peak RSS MiB|
|---:|---|---:|---:|---:|---:|
|128|nn|0.3180|3.2331|43.91|678.9|
|128|llama F32|0.4394|3.5100|43.44|1114.4|
|128|llama F16|0.5662|3.3424|43.70|890.0|
|1024|nn|3.3640|7.2125|33.40|881.3|
|1024|llama F32|3.3837|8.5294|27.73|1133.0|
|1024|llama F16|3.8735|7.9111|31.70|908.9|

The short total advantage over F16 is only3.27%, long total8.83%. Long TTFT versus F32 differs by less than1%. These are not robust superiority claims. All18 trials and ranges are retained, including slow runs.

## Implementation

- Primary KV arrays now use head-major layout, keeping each head's positions contiguous; capacity is unchanged
- Decode, prefill and the row-attention fallback share that layout and preserve exactly the previous arithmetic and causal bounds
- Temporary head-major transpose arrays/copies are removed
- Four-key/four-query QK blocks and64-dimension weighted-V panels improve reuse without changing reduction order
- Projection packing uses checked whole-projection slices, F16C and four-byte copies; fewer-than-four-token tails skip unnecessary packing

An independent static review identified a missing bounds check in an early fast-packing prototype. It was fixed before this candidate's runs. Overflow and truncated-projection regression tests verify panic-before-output-write behavior. A separate decoder VNNI experiment passed unit correctness but regressed key matrix microbenchmarks, so it was not adopted.

## Verification so far

- Ten native unit tests pass, including mixed KV appends and slot2047
- Default path:1,370 tiny logit checkpoints and3,168 sampled IDs, all bit-identical
- Forced AVX2 GEMM plus row attention: the same tiny/sampling coverage, all bit-identical
- Selected real gate:374 checkpoints, all bit-identical to immutable original nn
- Pinned Almide0.66.0 native benchmark and actual qwen_chat builds pass; one-shot chat smoke returned the requested text
- Source and binary attestations are retained; default-path gate provenance is checked before timing
- Exhaustive real t1/t4 gate remains pending at the time of this diagnostic report

The source SHA256 is a7e5d99e95e5f1c0ca06db6cc5eba4368d68452fdf71e249dab35fddd9b4e1f5. The binary SHA256 is6025c3aa9c9dffd0c4af52a675eee7889b1e11f22efd6db828e4a5cd10e3b4a3. Physical-host contention/temperature and energy are uncontrolled or unmeasured. Original nn versus llama output-quality equivalence remains unproven as documented in iteration03.
