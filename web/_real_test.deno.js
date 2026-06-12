// Full-model browser-path proof via Deno WebGPU: load the real
// Qwen3-0.6B Q8_0 GGUF through web/m1.js, greedy-decode the ids canary
// and compare with the native GPU run (examples/_bench_gpu.almd).
//
//   deno run --unstable-webgpu --allow-read web/_real_test.deno.js
//
// Expected first tokens (canary, "The capital of France is" →):
//   12095,13,576,6722,315,15344,...

const PROMPT = [785, 6722, 315, 9625, 374];
const N_NEW = 32;
// native GPU greedy (bench_gpu, 2026-06-13) — first 32 generated ids
const WANT = [
  12095, 13, 576, 6722, 315, 15344, 374, 21718, 13, 576, 6722, 315,
  17689, 374, 24081, 13, 576, 6722, 315, 5616, 374, 26549, 13, 576,
  6722, 315, 6323, 374, 26194, 13, 576, 6722,
];

import { loadModel } from "./m1.js";

const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
if (!adapter) {
  console.log("REAL_SKIP no adapter");
  Deno.exit(2);
}
const lim = adapter.limits;
console.log(`adapter maxStorageBufferBindingSize=${(lim.maxStorageBufferBindingSize / 1048576) | 0} MiB`);
// per-layer weight buffers: the biggest binding is the embedding
// (~167 MiB), so the iGPU-class 256 MiB profile is enough — request
// exactly that to PROVE portability, not just run.
const device = await adapter.requestDevice({
  requiredLimits: {
    maxStorageBufferBindingSize: Math.min(lim.maxStorageBufferBindingSize, 256 << 20),
    maxBufferSize: Math.min(lim.maxBufferSize, 256 << 20),
  },
});

const wgsl = await Deno.readTextFile("native/wgsl/qwen3.wgsl");
console.log("reading gguf…");
const gguf = (await Deno.readFile("parity/qwen3-0.6b-q8_0.gguf")).buffer;
const t0 = performance.now();
const model = await loadModel(device, wgsl, gguf, (t) => console.log("  " + t));
console.log(`load+upload: ${((performance.now() - t0) / 1000).toFixed(1)}s`);

const argmax = (a) => {
  let b = 0;
  for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i;
  return b;
};

let pos = 0;
let next = 0;
const t1 = performance.now();
for (const t of PROMPT) {
  next = argmax(await model.step(t, pos));
  pos++;
}
const got = [];
for (let k = 0; k < N_NEW; k++) {
  got.push(next);
  next = argmax(await model.step(next, pos));
  pos++;
}
const dt = (performance.now() - t1) / 1000;
console.log(`decode: ${(PROMPT.length + N_NEW)} steps in ${dt.toFixed(1)}s → ${((PROMPT.length + N_NEW) / dt).toFixed(2)} tok/s`);
console.log("got : " + got.join(","));
console.log("want: " + WANT.join(","));
const ok = got.length === WANT.length && got.every((v, i) => v === WANT[i]);
console.log(ok ? "REAL_PASS — browser path reproduces the native GPU ids" : "REAL_FAIL");
Deno.exit(ok ? 0 : 1);
