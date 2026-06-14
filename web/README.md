# nn — Qwen3 chat in the browser, 100% local

The whole inference stack is Almide: the Qwen3-0.6B engine runs on WebGPU
(8 WGSL compute kernels, one command buffer per token) and the byte-level
BPE tokenizer runs as a wasm module. JS only owns the WebGPU device, the
GGUF fetch, and the DOM.

```
index.html ── chat UI (DOM)
worker.js  ── orchestration (WebGPU device, fetch, postMessage)
  chat.js       ── template + decode loop + sampling
  tokenizer.js  ── wasm tokenizer wrapper (8-byte Almide heap ABI)
  m1.js         ── WebGPU engine (port of native/gpu_model.rs)
  wasi.js       ── minimal WASI shim for the tokenizer wasm
native/wgsl/qwen3.wgsl ── the kernels (shared with the native engine)
qwen_tokenizer.wasm    ── examples/qwen_tokenizer_wasm_entry.almd, --target wasm
```

One GGUF feeds both halves: `m1.js` reads the Q8_0 weights, the tokenizer
wasm reads the `tokenizer.ggml.*` section (incl. the precomputed sort
permutation — see `tools/export_gguf.py`).

## Run locally

```bash
# from the repo root (so ../parity and ../native resolve)
python3 -m http.server 8080
# open http://localhost:8080/web/   in a WebGPU browser (Chrome/Edge)
```

The default model is the 640 MB `parity/qwen3-0.6b-q8_0.gguf`. To point at a
hosted copy instead: `http://localhost:8080/web/?model=https://…/model.gguf`.

## Rebuild the tokenizer wasm

```bash
almide build examples/qwen_tokenizer_wasm_entry.almd --target wasm \
  -o web/qwen_tokenizer.wasm
```

## Verify without a browser (Deno WebGPU)

Headless Chrome can't run WebGPU compute on macOS, so the automated checks
use Deno's WebGPU (same wgpu underneath):

```bash
deno run --unstable-webgpu --allow-read web/_m1_test.deno.js      # engine, tiny model, bit-exact vs native
deno run --unstable-webgpu --allow-read web/_real_test.deno.js    # engine, real 0.6B, ids match native GPU
deno run --unstable-webgpu --allow-read web/_chat_test.deno.js    # FULL stack: tokenizer+engine reply == native
node examples/tokenizer_wasm_parity.mjs                            # tokenizer, 20/20 HF parity in wasm
```

## Status

- M0 device + kernel compile ✓
- M1 engine bit-exact vs native (tiny) + real-0.6B ids 32/32 ✓
- M2 tokenizer 20/20 HF parity in wasm, load 1.2 s ✓
- M3 full stack reproduces the native chat reply ✓
- M3+ to do: OPFS caching of the GGUF, mobile limits, public demo URL
