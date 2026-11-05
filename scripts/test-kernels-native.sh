#!/bin/bash
# Run the spec/kernels/ tests on the NATIVE leg.
#
# `almide test` runs a test file on the wasm leg when it can and falls back
# to native only for a file marked `// wasm:skip`, so plain `almide test`
# exercises wasm. This copies every spec/kernels/*_test.almd into
# .almide/native-spec-kernels/ with that marker on its first line and runs those.
# The wasm leg is `almide test spec/kernels/ --target wasm`.
set -euo pipefail
cd "$(dirname "$0")/.."
out=.almide/native-spec-kernels
rm -rf "$out"
mkdir -p "$out"
for f in spec/kernels/*_test.almd; do
  { echo "// wasm:skip"; cat "$f"; } > "$out/$(basename "$f")"
done
almide test "$out" "$@"
