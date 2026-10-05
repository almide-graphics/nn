#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LLAMA_CPP_DIR="$(cd "${1:?usage: build-llama-harness.sh /path/to/pinned/llama.cpp}" && pwd)"
EXPECTED=8e1642198dcd4e408f8776222d6ae31b74d01187
if [[ "$(git -C "$LLAMA_CPP_DIR" rev-parse HEAD)" != "$EXPECTED" ]]; then
  echo "Expected llama.cpp $EXPECTED" >&2; exit 1
fi
"${CXX:-g++}" -O3 -DNDEBUG -std=c++17 -Wall -Wextra -Wpedantic \
  -I"$LLAMA_CPP_DIR/include" -I"$LLAMA_CPP_DIR/ggml/include" "$HERE/llama-duel.cpp" \
  -L"$LLAMA_CPP_DIR/build/bin" -Wl,-rpath,"$LLAMA_CPP_DIR/build/bin" \
  -lllama -lggml -lggml-base -o "$HERE/llama-duel"
