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

1. **M0 scaffold** ✅ (2026-06-13): `web/` — index.html chat shell,
   worker.js, m0.js (device init + WGSL compile + gemv smoke).
   NB: headless-Chrome automation on macOS cannot run WebGPU compute
   (renderer dies after device creation) — automated verification goes
   through **Deno's WebGPU** (same wgpu as native) instead; browser
   pages remain for human click-testing.
2. **M1 kernels-from-JS** ✅ (2026-06-13): web/m1.js — full GGUF parse,
   Q8 repack, 8 pipelines, 17-op layer list, KV copies.
   `deno run --unstable-webgpu --allow-read web/_m1_test.deno.js`
   → **bit-exact vs native GPU** on the tiny model (max_abs = 0.0), and
   `web/_real_test.deno.js` → real Qwen3-0.6B greedy ids **32/32
   identical** to native, 4.9 tok/s on the 2016 Radeon Pro 560.
   The browser engine is proven; only the wasm side remains.
3. **M2 wasm orchestration** (needs #643 fixed; #681 for loader reuse):
   nn.wasm owns tokenizer, chat template, sampling, op-list; m1.js's
   executor half stays. Acceptance: tokenizer parity 20/20 IN WASM +
   the ids canary on the real 0.6B.
4. **M3 product**: per-layer weight buffers (iGPU limits), OPFS caching,
   streaming UI, publish demo URL (GitHub Pages + R2/HF for the GGUF).

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
