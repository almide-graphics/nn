// M1 acceptance via Deno's WebGPU (same wgpu underneath, no browser needed):
//   deno run --unstable-webgpu --allow-read web/_m1_test.deno.js
// Headless macOS Chrome can't run compute reliably; this harness exercises
// the exact same web/m1.js the browser uses.
import { loadModel } from "./m1.js";

const adapter = await navigator.gpu?.requestAdapter();
if (!adapter) {
  console.log("M1_SKIP no adapter");
  Deno.exit(2);
}
const device = await adapter.requestDevice();

const wgsl = await Deno.readTextFile("native/wgsl/qwen3.wgsl");
const gguf = (await Deno.readFile("testdata/tiny-qwen3-q8_0.gguf")).buffer;
const refText = await Deno.readTextFile("testdata/tiny_ref_logits.txt");
const ref = refText.trim().split(",").map(Number);

const model = await loadModel(device, wgsl, gguf, (t) => console.log("  " + t));
let pos = 0;
for (const t of [1, 2, 3, 4]) {
  await model.step(t, pos);
  pos++;
}
const logits = await model.step(5, pos);
if (logits.length !== ref.length) {
  console.log(`M1_FAIL vocab mismatch ${logits.length} vs ${ref.length}`);
  Deno.exit(1);
}
let maxRel = 0, maxAbs = 0, argGot = 0, argWant = 0;
for (let i = 0; i < ref.length; i++) {
  const d = Math.abs(logits[i] - ref[i]);
  maxAbs = Math.max(maxAbs, d);
  maxRel = Math.max(maxRel, d / Math.max(1e-6, Math.abs(ref[i])));
  if (logits[i] > logits[argGot]) argGot = i;
  if (ref[i] > ref[argWant]) argWant = i;
}
console.log(`argmax: got ${argGot}, want ${argWant}`);
console.log(`max_abs=${maxAbs.toExponential(3)} max_rel=${maxRel.toExponential(3)}`);
const pass = argGot === argWant && maxAbs < 1e-2;
console.log(pass ? `M1_PASS abs=${maxAbs.toExponential(2)}` : "M1_FAIL");
Deno.exit(pass ? 0 : 1);
