// C (T3 moat), GPU target: the SAME hardened DLGN circuit evaluated in WGSL
// on the GPU must produce the identical truth table to the Almide native/wasm
// runs (research/autograd/circuit_eval.almd). Pure integer/boolean logic, so
// "bit-identical across targets" is provable, not approximate.
//
//   deno run --unstable-webgpu research/autograd/circuit_eval_gpu.mjs
//
// (Deno is the WebGPU harness — headless Chrome can't run compute on this Mac.)

const PICKS = [6, 0, 0, 0]; // XOR, FALSE, FALSE, FALSE — same as the Almide side

const adapter = await navigator.gpu?.requestAdapter();
if (!adapter) { console.log("SKIP no adapter"); Deno.exit(2); }
const device = await adapter.requestDevice();

// WGSL port of gate()/circuit() — identical boolean algebra to circuit_eval.almd
const WGSL = `
@group(0) @binding(0) var<storage, read> picks: array<i32,4>;
@group(0) @binding(1) var<storage, read_write> out: array<i32,4>;

fn gate(g: i32, a: i32, b: i32) -> i32 {
  if (g == 0)  { return 0; }
  if (g == 1)  { return a * b; }
  if (g == 2)  { return a * (1 - b); }
  if (g == 3)  { return a; }
  if (g == 4)  { return (1 - a) * b; }
  if (g == 5)  { return b; }
  if (g == 6)  { return a + b - 2 * a * b; }   // XOR
  if (g == 7)  { return a + b - a * b; }       // OR
  if (g == 8)  { return 1 - (a + b - a * b); }
  if (g == 9)  { return 1 - (a + b - 2 * a * b); }
  if (g == 10) { return 1 - b; }
  if (g == 11) { return 1 - b + a * b; }
  if (g == 12) { return 1 - a; }
  if (g == 13) { return 1 - a + a * b; }
  if (g == 14) { return 1 - a * b; }
  return 1;
}

@compute @workgroup_size(4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = i32(gid.x);
  if (i >= 4) { return; }
  let a = i / 2;
  let b = i - (i / 2) * 2;
  var acc = 0;
  for (var n = 0; n < 4; n = n + 1) {
    let o = gate(picks[n], a, b);
    acc = acc + o - acc * o;   // boolean OR
  }
  out[i] = acc;
}`;

const mod = device.createShaderModule({ code: WGSL });
const pipeline = device.createComputePipeline({ layout: "auto", compute: { module: mod, entryPoint: "main" } });

const picksBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
new Int32Array(picksBuf.getMappedRange()).set(PICKS);
picksBuf.unmap();
const outBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
const stage = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

const bind = device.createBindGroup({
  layout: pipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: picksBuf } },
    { binding: 1, resource: { buffer: outBuf } },
  ],
});
const enc = device.createCommandEncoder();
const pass = enc.beginComputePass();
pass.setPipeline(pipeline);
pass.setBindGroup(0, bind);
pass.dispatchWorkgroups(1);
pass.end();
enc.copyBufferToBuffer(outBuf, 0, stage, 0, 16);
device.queue.submit([enc.finish()]);
await stage.mapAsync(GPUMapMode.READ);
const got = Array.from(new Int32Array(stage.getMappedRange()));
stage.unmap();

const WANT = [0, 1, 1, 0]; // XOR truth table = the Almide native/wasm output
console.log("a b | GPU");
for (let i = 0; i < 4; i++) console.log(`${i >> 1} ${i & 1} |  ${got[i]}`);
const ok = got.length === 4 && got.every((v, i) => v === WANT[i]);
console.log(ok
  ? "C_PASS — GPU(WGSL) truth table is BIT-IDENTICAL to Almide native/wasm"
  : `C_FAIL — got ${JSON.stringify(got)} want ${JSON.stringify(WANT)}`);
Deno.exit(ok ? 0 : 1);
