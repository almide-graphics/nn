# WASM ternary-path benchmark

Harness: `oplab/bench/gen_bench.sh` generates three self-contained `effect fn main` WASM
programs (`noop` / `float` / `ternary`) that make REPS passes of a row-wise dot over a T×D
matrix, rebuilding the left operand each pass (defeats loop-invariant hoisting). Built with
`almide build … --target wasm --release`, run on `wasmtime`, wall-timed min/median of 7.
`kernel = (mode_total − noop_total) / REPS` isolates the dot from startup + per-pass rebuild.

- **float** = `dot_rows` (plain Σ a·b)
- **ternary** = `ternary_dot_rows` (TWN {−1,0,+1} code dot — list-based, on-the-fly thresholding,
  **not** a bit-packed XNOR/popcount kernel)
- **numpy** float / ternary equivalents timed with `perf_counter` for an absolute reference.

## Results (Opus run, this machine)

| workload | wasm float kernel/dot | wasm ternary kernel/dot | ternary/float | wasm float vs numpy |
|----------|----------------------:|------------------------:|--------------:|--------------------:|
| T=64 D=256  | 2.56 µs | 4.96 µs | **1.94× slower** | ~7× slower/kernel (11× total) |
| T=64 D=1024 | 8.73 µs | 16.05 µs | **1.84× slower** | ~6× slower/kernel |

## Honest conclusions

1. **The current ternary path is ~1.8–1.9× SLOWER than the float dot**, consistently across sizes.
   It does strictly more work per element (compute Δ, threshold, branch, multiply) and gains nothing,
   because the codes are recomputed from floats every call over `List<List<Float>>` — there is no
   bit-packing and no popcount. The "beyond-matmul" speed thesis is **not** realized by this op.
2. **Almide WASM float dot is ~6–7× slower per kernel than numpy** (BLAS/SIMD over contiguous memory).
   Expected: immutable cons-list traversal + per-call `to_lists`/`zip_with`/`fold` allocation vs a
   vectorized contiguous kernel.
3. **What the ternary win actually requires** (a different kernel, not a tweak):
   - quantize once, store **packed** codes (2-bit, or split sign + nonzero-mask bitplanes),
   - dot via **XNOR + popcount** over 64-bit machine words (no per-element multiply/branch),
   - a **contiguous integer tensor** backend instead of `List<List<Float>>`.
   Only then does ternary beat float (≈ memory ÷16–32 and integer-popcount throughput).

So the 362-op library's value today is **verified correctness (numpy 1e-9) + edge/WASM deployability**,
not speed. Speed is a separate, unbuilt kernel-engineering effort.
