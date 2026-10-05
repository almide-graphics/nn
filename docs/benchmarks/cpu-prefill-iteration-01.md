# CPU prefill optimization iteration 1

This is a diagnostic result, not a victory claim. The new prefill path preserves the original nn outputs, but it is still slower than llama.cpp on prompt processing and long-input total latency. Further optimization is in progress.

## Fixed workload

- Qwen3-0.6B Q8_0; SHA256 `3d1cfc4b845efcda96d63415c986d51781d02b2f55f93fc6add142cd56b4e314`
- Virtualized AMD EPYC 9V74; 4 pinned vCPUs; context 2048
- Input 128 or 1024 tokens; 128 additional single-token forward calls and argmax selections, giving 129 emitted IDs
- Full-prompt plus 8-step warmup; model load and tokenization excluded from phase timings
- nn uses F32 KV; both llama.cpp F32 and standard F16 KV are retained
- Three repetitions per engine/input, rotated execution order; diagnostic only
- llama.cpp pinned to `8e1642198dcd4e408f8776222d6ae31b74d01187`, CPU-native Release with batching and ISA features enabled

| Input | Engine | Prefill and first token s | Decode s | Total s | Decode tokens/s | Peak RSS MiB |
|---:|---|---:|---:|---:|---:|---:|
| 128 | nn | 0.770 | 3.123 | 3.893 | 40.99 | 671.0 |
| 128 | llama-f32 | 0.489 | 3.144 | 3.634 | 40.71 | 1114.4 |
| 128 | llama-f16 | 0.542 | 3.039 | 3.581 | 42.12 | 889.8 |
| 1024 | nn | 9.556 | 4.998 | 14.554 | 25.61 | 867.0 |
| 1024 | llama-f32 | 4.107 | 5.271 | 9.377 | 24.29 | 1132.8 |
| 1024 | llama-f16 | 4.012 | 4.659 | 8.671 | 27.48 | 908.8 |

Medians are reported for each metric separately; their displayed components need not sum exactly to the median total.

## Correctness and scope

- Original nn oracle SHA256 `7501f5e7d8eb98691c27678c01a8257fc3a4d8f3ef49033ceb93319b0b15589d`
- Tested native candidate SHA256 `a107d2e4dde646ec70761286bbca828b75c394752cc9e17aeb92152af10bf075`
- Full tiny suite: 1,370/1,370 checkpoints bit-identical across threads 1 and 4
- Real first-tier suite: 374/374 checkpoints bit-identical across 20 prompts and input128/input1024, each with 16 teacher-forced continuation steps
- Strict gate: finite logits, max absolute error <=1e-3, normalized maximum error <=1e-4, exact top1 in both prefill APIs; observed error was zero
- Two bitwise microkernel tests and 11 gate negative-control/guard tests pass
- Exhaustive real long-context/thread-1 coverage is pending; no full quality or universal speed claim

The prior sequential path needed about 2.319 s / 26.781 s for input128 / input1024 on this host. The first batching implementation reduces this substantially, but does not yet meet the latency target.

The remaining work is tracked in #1, #2 and #3.
