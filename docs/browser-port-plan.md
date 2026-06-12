# L3-3: Browser port — design

Goal: a URL that loads Qwen3-0.6B and chats locally. All inference logic
in Almide (wasm); JS only owns the WebGPU device, fetch/OPFS, and DOM.

## Architecture

```
index.html ─ chat UI (DOM, JS)
   │  postMessage / direct calls
worker.js ─ JS shim: WebGPU device, buffer pool, fetch+OPFS, wasm loader
   │  almide-wasm-bindgen (web/webgpu.almd pattern, like snaidhm)
nn.wasm ─ Almide: qwen_tokenizer + gpu orchestration + chat template
   │  WGSL (same 8 kernels as native/gpu_model.rs)
GPU
```

Decisions:

- **Same WGSL, two hosts.** native/gpu_model.rs's 8 kernels (gemv 2-D
  dispatch, embed, rmsnorm, rmsnorm_inplace, rope, attn GQA, silu_mul,
  add) are host-agnostic by design. Extract to `native/wgsl/*.wgsl` as
  the single source of truth; gpu_model.rs `include_str!`s them, the
  browser fetches them. Zero kernel divergence.
- **Wasm boundary at "ops", not "tokens".** The Almide side builds the
  per-token op list once (mirroring gpu_model.rs's prebuilt Op{pipeline,
  bind, groups}) and hands the shim a flat command description; JS
  replays it into one command buffer per token. Keeps per-token JS↔wasm
  chatter to: token id in, logits-argmax (or top-k block) out.
- **Weights upload**: GGUF fetched once → OPFS (`navigator.storage`),
  ~639 MB. Parse in Almide-wasm (gguf_prims), repack Q8 34B→36B
  u32-aligned blocks (same as native upload_q8), upload as ONE storage
  buffer (640 MB < the 1 GiB maxStorageBufferBindingSize we already
  require on native; needs requestDevice limits bump, Chrome ok).
- **KV cache** stays on GPU (MAX_SEQ 2048 after the cpu_fast bump —
  mirror it).
- **Sampling on wasm side** from a small logits readback: kernel writes
  top-256 (value,index) pairs via the existing argmax-reduction pattern
  extended to k-select, or v1 = full-logits readback (151936×4B = 600 KB
  per token — at 5-10 tok/s on iGPU that is 3-6 MB/s, acceptable for v1).

## Blockers (tracked)

| blocker | issue | state |
|---|---|---|
| wasm rc corruption: 2+ String list-reads + alloc per loop iter | almide#643 | OPEN — kills tokenizer AND gguf string metadata on wasm |
| #433 guard: linked-module record through fold lambda | almide#681 | OPEN — kills qwen_loader importers (stock compiler) |
| process.args wasm returns garbage strings | almide#645 | reopened — avoidable (no argv in browser) |

Not blocked: WGSL kernels (host-agnostic), JS shim, UI, OPFS plumbing,
op-list design — all can land first.

## Milestones

1. **M0 scaffold**: `web/` dir — index.html (chat shell), worker.js
   (device init, limits probe, WGSL fetch+compile, dummy gemv smoke
   test). No wasm yet. Verifiable on any WebGPU browser.
2. **M1 kernels-from-JS**: parse a TINY GGUF (test_weight model from
   qwen.almd's make_test_model, exported by a 20-line python script) in
   JS temporarily, run the full layer stack from worker.js, compare
   logits against native gpu_model for the same tiny model. Proves the
   browser GPU path end-to-end without wasm.
3. **M2 wasm orchestration** (needs #643+#681): nn.wasm owns gguf parse,
   tokenizer, op-list, sampling; worker.js degrades to a dumb executor.
   Acceptance: token_ids parity 20/20 IN THE BROWSER + the ids canary
   on the real 0.6B.
4. **M3 product**: OPFS caching, streaming UI, mobile Safari check
   (storage-buffer limits likely force f16/smaller model there — out of
   scope for v1), publish demo URL (GitHub Pages + Cloudflare R2 or HF
   for the GGUF).

## Risks

- 640 MB single storage buffer on iGPUs: Intel Iris in browsers caps
  maxStorageBufferBindingSize lower than native wgpu (often 128-256 MB).
  Mitigation: split the weight buffer per-layer (28 buffers ≈ 23 MB
  each) — bind group per layer, already how the op list is structured.
  DECISION: do per-layer buffers from M1; native keeps single-buffer.
- wasm memory: GGUF bytes (639 MB) + repack staging would exceed wasm32
  comfort. Mitigation: stream-repack layer by layer during upload
  (fetch ranges → repack → upload → drop), never hold the full file.
  OPFS gives random-access reads (FileSystemSyncAccessHandle in worker).
- wasm String loops (#643) might have siblings even after the fix —
  acceptance = parity ON WASM, never "compiles".
