// nn browser worker — M0: WebGPU device + kernel compile + gemv smoke test.
//
// Serve from the REPO ROOT so the worker can fetch the single-source WGSL:
//   python3 -m http.server 8080      # then open http://localhost:8080/web/
//
// Milestones (docs/browser-port-plan.md):
//   M0 (this file): device init, limits probe, compile native/wgsl/qwen3.wgsl,
//       run one Q8 gemv against CPU reference values.
//   M1: tiny-model full layer stack from JS, logits vs native gpu_model.
//   M2: nn.wasm (Almide) owns gguf/tokenizer/op-list; this file degrades
//       to a dumb executor. Blocked on almide#643/#681.

const post = (m) => self.postMessage(m);

function fp16ToF32(h) {
  const s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  if (e === 31) return f ? NaN : (s ? -Infinity : Infinity);
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

function f32ToFp16(v) {
  // good enough for test scales
  const f32 = new Float32Array(1); const u32 = new Uint32Array(f32.buffer);
  f32[0] = v;
  const x = u32[0];
  const sign = (x >> 16) & 0x8000;
  let exp = ((x >> 23) & 0xff) - 127 + 15;
  let man = (x >> 13) & 0x3ff;
  if (exp <= 0) return sign;
  if (exp >= 31) return sign | 0x7c00;
  return sign | (exp << 10) | man;
}

async function init() {
  try {
    if (!navigator.gpu) throw new Error("WebGPU がこのブラウザにありません");
    post({ type: "status", text: "requesting adapter…" });
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("WebGPU adapter が取れません");
    const lim = adapter.limits;
    post({ type: "status", text: `adapter ok — maxStorageBuffer ${(lim.maxStorageBufferBindingSize / 1048576) | 0} MiB` });

    // per-layer weight buffers need ~23 MiB; ask for a bit of headroom
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: Math.min(lim.maxStorageBufferBindingSize, 256 * 1048576),
        maxBufferSize: Math.min(lim.maxBufferSize, 512 * 1048576),
      },
    });
    device.lost.then((info) => post({ type: "error", text: "device lost: " + info.message }));

    post({ type: "status", text: "compiling kernels…" });
    const wgsl = await (await fetch("../native/wgsl/qwen3.wgsl")).text();
    const module = device.createShaderModule({ code: wgsl });
    const diag = await module.getCompilationInfo();
    const errs = diag.messages.filter((m) => m.type === "error");
    if (errs.length) throw new Error("WGSL: " + errs[0].message);

    await gemvSmokeTest(device, module);
    post({ type: "ready", text: "M0 ok — kernels compile, gemv matches CPU" });
  } catch (e) {
    post({ type: "error", text: String(e.message || e) });
  }
}

// One Q8_0 gemv: 4 rows × 64 cols of known weights against a known x,
// checked against a JS reference. Proves buffer layout (36-byte u32 blocks),
// bind groups and the 2-D dispatch path before any model bytes exist.
async function gemvSmokeTest(device, module) {
  const rows = 4, cols = 64, bpr = cols / 32; // blocks per row
  // weights: w[r][c] = (r + 1) * 0.01 * ((c % 7) - 3); quantize per block
  const wWords = new Uint32Array(rows * bpr * 9);
  const wRef = [];
  for (let r = 0; r < rows; r++) {
    const rowVals = [];
    for (let b = 0; b < bpr; b++) {
      const vals = [];
      for (let i = 0; i < 32; i++) {
        const c = b * 32 + i;
        vals.push((r + 1) * 0.01 * ((c % 7) - 3));
      }
      const amax = Math.max(...vals.map(Math.abs));
      const scale = amax / 127 || 1e-8;
      const q = vals.map((v) => Math.max(-127, Math.min(127, Math.round(v / scale))));
      const base = (r * bpr + b) * 9;
      const sf = new Float32Array([scale]);
      wWords[base] = new Uint32Array(sf.buffer)[0];
      for (let k = 0; k < 8; k++) {
        wWords[base + 1 + k] =
          (q[k * 4] & 0xff) | ((q[k * 4 + 1] & 0xff) << 8) |
          ((q[k * 4 + 2] & 0xff) << 16) | ((q[k * 4 + 3] & 0xff) << 24);
      }
      rowVals.push(...q.map((qq) => qq * scale));
    }
    wRef.push(rowVals);
  }
  const x = new Float32Array(cols);
  for (let i = 0; i < cols; i++) x[i] = Math.sin(i * 0.37);
  const want = wRef.map((row) => row.reduce((acc, w, i) => acc + w * x[i], 0));

  const mk = (data, usage) => {
    const buf = device.createBuffer({ size: data.byteLength, usage, mappedAtCreation: true });
    new (data.constructor)(buf.getMappedRange()).set(data);
    buf.unmap();
    return buf;
  };
  const wBuf = mk(wWords, GPUBufferUsage.STORAGE);
  const xBuf = mk(x, GPUBufferUsage.STORAGE);
  const yBuf = device.createBuffer({ size: rows * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const pBuf = mk(new Uint32Array([rows, cols, bpr, 0]), GPUBufferUsage.UNIFORM);
  const stage = device.createBuffer({ size: rows * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "gemv" },
  });
  const bind = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: wBuf } },
      { binding: 1, resource: { buffer: xBuf } },
      { binding: 2, resource: { buffer: yBuf } },
      { binding: 3, resource: { buffer: pBuf } },
    ],
  });
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bind);
  pass.dispatchWorkgroups(rows, 1, 1);
  pass.end();
  enc.copyBufferToBuffer(yBuf, 0, stage, 0, rows * 4);
  device.queue.submit([enc.finish()]);
  await stage.mapAsync(GPUMapMode.READ);
  const got = Array.from(new Float32Array(stage.getMappedRange()));
  stage.unmap();
  for (let r = 0; r < rows; r++) {
    if (Math.abs(got[r] - want[r]) > 1e-3 * Math.max(1, Math.abs(want[r]))) {
      throw new Error(`gemv smoke mismatch row ${r}: got ${got[r]}, want ${want[r]}`);
    }
  }
}

self.onmessage = (e) => {
  if (e.data.type === "chat") {
    // M2 wires this to nn.wasm. Until then, be honest about it.
    post({ type: "token", text: "（エンジン未接続 — M0スキャフォールドです。" });
    post({ type: "token", text: " カーネルのコンパイルとgemv検証は通っています）" });
    post({ type: "done" });
  }
};

init();
