#!/bin/bash
# Golden programs: tests/kernels/cross/*.almd and tests/kernels/fail/*.almd, each run on the
# native and the wasm target. A program passes when its stdout, stderr and
# exit code equal <name>.stdout, <name>.stderr and <name>.exit exactly.
#
# These are the stdlib's cross-target fixtures for the 25 functions, ported
# with matrix.X renamed to kernels.X; the expectations are what the stdlib printed
# before epoch 14. `test` blocks cannot assert an abort, so tests/kernels/fail/ (and
# the aborting tails of tests/kernels/cross/) live here rather than in spec/.
#
# usage: scripts/test-golden.sh [native|wasm ...]   (default: both)
set -uo pipefail
cd "$(dirname "$0")/.."
targets=("$@")
[ ${#targets[@]} -eq 0 ] && targets=(native wasm)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
pass=0
fail=0
for prog in tests/kernels/cross/*.almd tests/kernels/fail/*.almd; do
  base="${prog%.almd}"
  for t in "${targets[@]}"; do
    if [ "$t" = wasm ]; then
      almide run "$prog" --target wasm >"$tmp/out" 2>"$tmp/err"
    else
      almide run "$prog" >"$tmp/out" 2>"$tmp/err"
    fi
    ec=$?
    want_ec=$(cat "$base.exit")
    if cmp -s "$tmp/out" "$base.stdout" && cmp -s "$tmp/err" "$base.stderr" && [ "$ec" = "$want_ec" ]; then
      pass=$((pass + 1))
    else
      fail=$((fail + 1))
      echo "FAIL [$t] $prog (exit $ec, want $want_ec)"
      diff "$base.stdout" "$tmp/out" | head -5
      diff "$base.stderr" "$tmp/err" | head -5
    fi
  done
done
echo "golden: $pass passed, $fail failed (${targets[*]})"
[ "$fail" -eq 0 ]
