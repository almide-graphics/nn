# CPU batched-prefill correctness gate

This is a **correctness-only** test against the original sequential `nn` CPU executor,
not a speed benchmark and not a comparison to llama.cpp. Numerical thresholds are
fixed before measurement:

- Every logit must be finite
- Maximum absolute error <= **1e-3** at each checkpoint
- Maximum absolute error / maximum absolute reference logit <= **1e-4** at each checkpoint
- Top-1 must match exactly, for both `prefill_logits` and the production `prefill_argmax` path
- Bit identity is reported separately and preferred

An optimization passes only when **all** checks pass. There is no average-error,
percentage-agreement, or throughput escape hatch. Raw little-endian float32 logits,
all input IDs, source snapshots, toolchain/source/model hashes, and per-checkpoint
results are retained in the output directory. Dumps are outside inference calls;
this driver does not report timing and must not run during performance measurements.

## Prerequisites

- Rust/Cargo 1.94.0, with the dependencies in `Cargo.lock` available (build is offline)
- Python 3, NumPy, and the official `llama.cpp/gguf-py` package
- The repository's tiny fixture is sufficient; **no model is downloaded**

From the repository root:

```sh
python3 tools/cpu-parity/run.py \
  --gguf-py /path/to/llama.cpp/gguf-py \
  --out /tmp/nn-prefill-parity
```

The default suite runs at 1 and 4 threads. Each requested thread count must fit
both process affinity and Rust `available_parallelism()` (including cgroup limits);
the driver rejects unsafe counts before loading the model. On a small CI runner,
use `--threads 1` and report one-thread coverage honestly. It covers 20 fixed prompts (IDs mapped
modulo the tiny vocabulary for the tiny fixture), lengths 1/3/4/5,
63/64/65, 127/128/129, 511/512/513, 1024, 2032, and 2048; the exact 128/1024-token
throughput inputs; and split prompts exercising nonzero `start_pos`. Each case
has 16 deterministic teacher-forced continuation tokens unless the fixed
2048-token native context leaves fewer positions. Every split boundary and every
continuation token gets full-logit verification. The argmax entry point is exercised
in a separate replay, including its continued KV state. No-model, empty input,
negative/out-of-vocabulary IDs, negative/overflowing positions, and context-overflow
calls are checked; valid continuation after invalid calls verifies rejection does
not corrupt KV state. Cases reuse the loaded
model/cache and overwrite from position zero, which also checks stale-cache handling.
The tiny fixture advertises a 256-token context; longer cases intentionally test the
native executor's fixed 2048-slot KV bounds rather than model quality.

The original source comes from immutable commit
`3a75481915c6faa4c831d72d4a0fec433a08a852`, with only Rust API signatures changed
from `&Vec` / `Deref<Vec>` to slices. Its required SHA256 is
`7501f5e7d8eb98691c27678c01a8257fc3a4d8f3ef49033ceb93319b0b15589d`.
If that commit is not present locally, supply `--reference /path/to/cpu_fast.rs`;
the same immutable hash is still mandatory. Neither inference math nor the reference
is rewritten by this test. The small fp16 conversion shim is copied from `native/gpu.rs`
to avoid building GPU dependencies.

## Real model (explicit opt-in)

```sh
python3 tools/cpu-parity/run.py \
  --gguf-py /path/to/llama.cpp/gguf-py \
  --model /path/to/qwen3-0.6b-q8_0.gguf --real-model \
  --out /tmp/nn-prefill-parity-real
```

The only accepted real-model hash is
`3d1cfc4b845efcda96d63415c986d51781d02b2f55f93fc6add142cd56b4e314`.
Metadata/absolute tensor offsets come from official GGUFReader; Q8_0 weights,
F32 normalization weights, Qwen3 architecture and tied embeddings are checked.
Both engines receive exactly the same IDs, offsets, model bytes, and thread count.

For a quick first diagnostic, select `--threads 4 --case-regex '(prompt-[0-9]+|input-128|input-1024)'`.
Such a run is labeled `full_suite: false`; it is **not** the complete gate.

`--phase prepare` creates frozen source snapshots and builds without inference.
Run `--phase reference`, `--phase candidate`, and `--phase compare` against the
same `--out` to coordinate CPU scheduling. Non-prepare phases use the recorded
model, threads and cases. Do not edit generated manifests or interchange output
directories. Rerun `prepare` after a candidate change and rerun both engines.
A run-identity stamp prevents stale outputs from satisfying a newly prepared gate.
A missing/failed process, stale execution stamp, changed input, missing output, truncated logits, non-finite value,
threshold violation, or top-1 mismatch fails the check. Exit code zero means the
selected suite passed; consult `full_suite` and `threads` before claiming complete
coverage.

The comparator has negative-control tests: `python3 tools/cpu-parity/test_gate.py`.
They independently exercise absolute and normalized thresholds, both top-1 paths,
NaN/Inf rejection, truncation, and stale-run rejection.

## Lightweight CI proposal (not an installed workflow)

Use a one-thread tiny gate on changes to `native/cpu_fast.rs` or this tool. Pin Rust
1.94.0, install NumPy and official `gguf-py`, fetch the pinned Cargo dependencies
once (`cargo fetch --locked --manifest-path tools/cpu-parity/Cargo.toml`), and cache
the Cargo registry. Run:

```sh
python3 tools/cpu-parity/test_gate.py
python3 tools/cpu-parity/run.py --threads 1 --out "$RUNNER_TEMP/cpu-parity"
```

The immutable baseline commit must exist in the checkout (use full checkout history
or explicitly fetch that commit). Upload `summary.json`, `manifest.json`, and logs
on all runs, with raw `.f32` files on failures. This requires no real-model download.
A larger scheduled/opt-in job can run the two-thread-count real-model gate when the
approved pinned weights and at least four available CPUs are supplied.

## Sampled prefill and execution provenance

`--sampled` adds a tiny-only exact sampled-ID gate against the original `decode_sample`:
288 combinations of temperatures 0/0.6/1, top-p -0.1/0/0.5/0.9/1/1.1, seeds
0/42/-1/i64::MAX, and single/multiple/split nonzero-start prefixes. Four identical
teacher-forced continuation steps verify sampled IDs; the final continuation's full
logits also pass the original fixed numerical gate. Invalid sampled-prefill calls
must reject without disturbing continuation. Results live in `sampling-summary.json`
and also gate the main `summary.json`. This never changes the original numeric cases
or thresholds. Example: add `--sampled --threads 1,4` to the normal tiny command.

`--candidate /path/to/frozen/cpu_fast.rs` selects an explicit candidate snapshot.
Otherwise the current native source is copied once before compilation. Keep separate
output directories for different candidates. `NN_PREFILL_KERNEL`,
`NN_PREFILL_ATTENTION` and `NN_PREFILL_PROFILE` are recorded in the manifest/summary
and restored for every phase, preventing a phase from silently testing a different
path. The benchmark's default-path gate requires all three unset. Summaries record
the manifest hash and run identity; older evidence lacking these fields remains
historical and must be rerun for the hardened benchmark protocol.
