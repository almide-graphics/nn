# Fixed CPU inference comparison

This protocol measures the same Qwen3-0.6B Q8_0 file and raw input IDs in nn and
pinned llama.cpp. A numerical gate is required before performance measurements.
It reports every run and makes no automatic victory or model-quality claim.

## Prerequisites

- Linux, `taskset`, Python 3, and sufficient available memory
- nn built with the tested Almide compiler and `--fast`
- llama.cpp at `8e1642198dcd4e408f8776222d6ae31b74d01187`, CPU-native Release
- A local Qwen3-0.6B Q8_0 file whose SHA256 is
  `3d1cfc4b845efcda96d63415c986d51781d02b2f55f93fc6add142cd56b4e314`
- A passing real-model summary from `tools/cpu-parity/run.py` for this exact
  native source, model and thread count

Nothing is downloaded by the benchmark runner. Build llama.cpp separately using
its official build instructions, retaining CPU-native ISA, optimized batching,
OpenMP/llamafile/repack and normal attention support. The measured baseline used
Release/-O3, `GGML_NATIVE=ON`, CPU only, and no BLAS dependency. Build its C API
harness with:

```sh
bash tools/cpu-bench/build-llama-harness.sh /path/to/pinned/llama.cpp
python3 tools/cpu-bench/build-nn-harness.py --almide /path/to/almide-0.66.0 --out /tmp/nn-duel
```

The harness build checks the exact llama.cpp revision. Keep compiler versions,
build options, source hashes and binary hashes with the results.

## Run

```sh
python3 tools/cpu-bench/run.py \
  --model /path/to/qwen3-0.6b-q8_0.gguf \
  --nn-bin /tmp/nn-duel \
  --llama-bin tools/cpu-bench/llama-duel \
  --quality-summary /path/to/full-real-parity/summary.json \
  --threads 4 --repetitions 10 \
  --out /path/to/new-empty-output-directory
```

The nn builder requires official Almide `0.66.0 (release, 819bbc74f)` and Rust/Cargo
1.94.0. It freezes `src/`, `native/`, the benchmark harness and package config,
then builds that frozen tree with `--fast`, native CPU code generation, at most two
Cargo build jobs, and an isolated temporary directory. An externally set
`CARGO_TARGET_DIR` is honored and recorded so a dedicated fingerprinted dependency
cache can be reused between candidates. Generated Cargo.toml/Cargo.lock paths,
snapshots and hashes are also retained. The adjacent `.build.json`
sidecar records all source hashes, compiler/tool hashes and versions, exact command,
environment, log hash and final binary hash. Keep the `.build-files` directory with it.

The runner requires that sidecar (default `<nn-bin>.build.json`, or `--nn-build`) and
verifies the binary and frozen input hashes. `--nn-source` is an optional additional
hash cross-check; it cannot replace the sidecar. Sidecar commands are never executed.

The real numerical gate must include the fixed original oracle, exact predeclared
thresholds, complete finite/exact-top1 checkpoint results, matching aggregate errors,
source snapshots, hashed inputs, current execution stamps and manifest identity.
The three diagnostic `NN_PREFILL_*` overrides must be recorded as unset, matching the
benchmark's default runtime path. Historical summaries without this evidence are
pre-hardening diagnostics and must be rerun for this protocol. Ten or more repetitions
require all 41 fixed cases / 685 checkpoints per tested thread count. Fewer repetitions
may be used for clearly labeled diagnostics after a passing selected real suite.

For even repetition counts, each block runs a rotated engine order followed by
its reverse: each engine appears twice and has the same mean position in a block.
Odd diagnostic counts use rotated orders. Both matched F32 KV and llama.cpp's
standard F16 KV are measured. All engines use the same selected CPU affinity and
thread count. The first available CPUs are selected and recorded.

Stop other heavy compilation, inference and tests before timing. The local lock
prevents concurrent copies of this runner; it cannot stop unrelated work. Ensure
the requested thread count is allowed by both affinity and the CPU quota. The
numerical driver explicitly validates Rust's available parallelism.

## Work and timing definitions

- Context allocation: 2048
- Input lengths: 128 and 1024, from the fixed token fixtures
- Warmup: full prompt plus eight decode calls, followed by logical cache reset
- Timed work: prefill/initial greedy selection, then 128 additional forward calls
  and greedy selections, producing 129 emitted IDs
- EOS never shortens the fixed-work measurement
- Tokenization, model loading, warmup and result-file writes are outside phase timing
- nn F32 KV is compared with both F32 and F16 KV in llama.cpp
- llama.cpp retains batched prefill with batch/ubatch 512

Use nn `decode_ms` against llama.cpp `generation_wall_s`, which both include token
selection. llama.cpp `decode_s` alone excludes argmax and must not be substituted.
First-token comparison includes llama.cpp `first_argmax_s`. The runner checks step
counts, emitted-ID counts, cache/context settings and finite nonnegative latencies.

RSS is the process high-water mark including load and warmup. It reflects actual
allocation/page-in policy; it is not just theoretical KV size. An OS-warm fresh
process is not a cold-storage test. There is no energy measurement or GPU result.

## Evidence

The output directory contains the run design and commands, source/model/binary
hashes, hardware/affinity, every engine JSON, generated IDs, stderr/stdout, process
RSS and CPU time, `trials.csv`, and median/min/max summaries. Preserve failed runs
and do not replace them with favorable retries. Investigate failures and begin a
new, clearly labeled series when the source or environment changes.

Results describe only the tested model, quantization, host and settings. The
numerical suite establishes the checked nn-to-oracle equivalence; it is not a
perplexity or downstream task-accuracy evaluation.

Run the lightweight negative controls with `python3 tools/cpu-bench/test_evidence.py`.
They reject missing/forged tolerances, hidden failed points, wrong oracle/source/model,
unverified runtime overrides, false full-suite claims, wrong binaries/source snapshots,
missing build sidecars and sequential/malformed nn result records.
