# CPU prefill iteration07: exact compact row panels

This is a three-repetition diagnostic. The medians favor nn, but the short-input first-token difference is less than1% and the run ranges overlap. These data do not establish a robust win, much less universal superiority. Confirmatory repeated measurements remain pending.

Conditions are unchanged: same Qwen3-0.6B Q8_0 file, four pinned virtual CPU cores, context2048, full prompt+eight decode warmup,128 additional timed decode calls and greedy selections (129 emitted IDs). Three rotated blocks compare nn with optimized llama.cpp F32 and standard F16 KV. Unrelated CPU jobs were paused. Physical host activity, thermal behavior and energy are not controlled.

|Input|Engine|TTFT seconds|Total inference seconds|Decode tokens/s|Peak RSS MiB|
|---:|---|---:|---:|---:|---:|
|128|nn|0.4352|3.1896|46.99|679.0|
|128|llama F32|0.4980|3.3783|44.68|1114.5|
|128|llama F16|0.4388|3.3944|43.77|889.9|
|1024|nn|3.2624|7.9666|27.27|881.7|
|1024|llama F32|3.7858|9.1067|23.48|1132.8|
|1024|llama F16|3.8816|8.6125|28.56|908.9|

Against standard F16, the observed long-input median TTFT is15.95% lower and total is7.50% lower. The short-input total median is6.04% lower; its TTFT is only0.81% lower. Decode-only long-input throughput remains below F16 llama.cpp in this sample. Do not cherry-pick the F32 baseline to conceal that.

## Exact math, improved layout

The new GEMM maps16 weight rows into ZMM lanes and packs a projection once per512-token chunk. Unsigned weight bytes and a shared per-activation correction preserve each four-product integer partial exactly. Eight independent floating partials, even/odd block updates and the final reduction tree are unchanged from the original nn oracle. Packing is included in prefill timing, not hidden in model load. Packed scratch is bounded to one projection; the existing model bytes and decode algorithm remain unchanged.

A preceding candidate combined integer partials before floating scaling. Its microbenchmarks were fast but the full tiny gate failed51/1,370 logit checkpoints (max error0.01869), despite all top1 choices matching. It was rejected, and the error thresholds were not relaxed. The exact compact candidate passed1,370 tiny-model checkpoints at one/four threads,3,168 sampled IDs with bit-identical continued logits, and374 selected real-model checkpoints at four threads, all bit-identical to the original nn. Full real long-context/thread1 coverage is still pending.

This iteration used the hardened benchmark tool: exact build/source attestation, model/source hashes, numerical-gate provenance, recorded default runtime backend, fixed workloads and emitted-ID counts. The source SHA256 is4d08f6db6b49b208b0277c4878c443de6837064ad6a9d01e255e512fbc0bc94a; the binary SHA256 isedb0b75e550d9e94ceaf702cab65b273fd400cd53532dfbed5993ae557ff6e19. The JSON includes all median/min/max values, and the CSV retains every one of18 trials.

The original nn/llama quality limitation reported in iteration03 remains:360/374 top1 matches across the expanded cross-engine diagnostic. Preserving original nn results is not proof of equal cross-engine model quality.
