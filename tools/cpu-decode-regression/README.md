# Contemporaneous decoder regression

This is the exact Rust driver used for the final original-versus-optimized nn
regression check. It shares model bytes between two independent native states;
it is not an end-to-end or per-engine RSS benchmark.

First prepare the tiny and real numerical gates described in
`../cpu-parity/README.md`. Their frozen source/config/case files provide the
inputs below. Use Rust/Cargo 1.94.0, four available Linux CPUs and a fresh output.
The original source hash must be
`7501f5e7d8eb98691c27678c01a8257fc3a4d8f3ef49033ceb93319b0b15589d`.
The measured candidate hash is
`a7e5d99e95e5f1c0ca06db6cc5eba4368d68452fdf71e249dab35fddd9b4e1f5`.

From the nn repository root:

```sh
set -eu
TINY=/absolute/path/to/prepared-tiny-gate
REAL=/absolute/path/to/prepared-real-gate
MODEL=/absolute/path/to/qwen3-0.6b-q8_0.gguf
TARGET=/absolute/path/to/new-build-target
OUT=/absolute/path/to/new-results
test ! -e "$OUT"
printf '%s  %s\n' 3d1cfc4b845efcda96d63415c986d51781d02b2f55f93fc6add142cd56b4e314 "$MODEL" | sha256sum -c -
CPU_DECODE_ORIGINAL="$REAL/sources/original.rs" \
CPU_DECODE_CANDIDATE="$REAL/sources/candidate.rs" \
CARGO_TARGET_DIR="$TARGET" RUSTFLAGS='-C target-cpu=native' \
cargo build --offline --locked --release --manifest-path tools/cpu-decode-regression/Cargo.toml
ALMIDE_LOCKSTEP_THREADS=4 RAYON_NUM_THREADS=4 CPU_DECODE_EXPECTED_AFFINITY=0-3 \
taskset -c 0-3 "$TARGET/release/nn-decode-regression" \
  testdata/tiny-qwen3-q8_0.gguf "$TINY/config.txt" "$TINY/cases.tsv" \
  "$MODEL" "$REAL/config.txt" "$REAL/cases.tsv" "$OUT" benchmark
```

The offline dependencies must already be cached. Change both affinity arguments
consistently if CPUs 0–3 are unavailable. Preserve compiler/source/binary hashes
and every output record from a new run.

Each prompt is computed once through both sequential decode APIs outside timing.
Every trial replays eight warmup calls, then restarts at the saved initial token
and position N for exactly 128 timed decode calls. Generated KV slots are
replaced, while the prompt prefix remains intact. Tiny validation compares
fresh, deliberately contaminated future-KV, repeated and freshly reloaded states.
All IDs and final logits must agree. The driver repeats those checks outside
real timing, including all 151,936 final logit bits after every measured run.

Five alternating ABBA/BAAB blocks give ten runs per engine/input. The raw TSV
contains every duration, order, timestamp and ID. The timer excludes model
loading, prompt setup, warmup, allocations in the driver, verification and I/O;
inside-decoder allocations and the fixed output-buffer writes remain included.
Use median duration and candidate/original paired-block ratios, retaining every
trial. The final report used a zero-slowdown margin, not a 5% allowance. With only
five blocks, bootstrap intervals are exploratory and n=10 p95 is the maximum.

See `../../docs/benchmarks/cpu-final-results.md`,
`cpu-decode-regression.json` and `cpu-decode-regression-trials.tsv` for the measured
result. The delivered archive also preserves the full original orchestration,
validation outputs and fingerprints.
