// nn browser M0 — WebGPU device init + kernel compile + Q8 gemv smoke test.
// Shared by worker.js (product path) and _m0_test.html (headless test page):
// headless Chrome's --virtual-time-budget stalls module workers, so the
// test must be runnable in page context.

export async function initM0(report) {
  if (!navigator.gpu) throw new Error("WebGPU がこのブラウザにありません");
  report("requesting adapter…");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("WebGPU adapter が取れません");
  const lim = adapter.limits;
  report(`adapter ok — maxStorageBuffer ${(lim.maxStorageBufferBindingSize / 1048576) | 0} MiB`);

  // per-layer weight buffers need ~23 MiB; ask for headroom, capped to
  // what the adapter offers
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: Math.min(lim.maxStorageBufferBindingSize, 256 * 1048576),
      maxBufferSize: Math.min(lim.maxBufferSize, 512 * 1048576),
    },
  });

  report("compiling kernels…");
  const wgsl = await (await fetch(new URL("../native/wgsl/qwen3.wgsl", import.meta.url))).text();
  const module = device.createShaderModule({ code: wgsl });
  const diag = await module.getCompilationInfo();
  const errs = diag.messages.filter((m) => m.type === "error");
  if (errs.length) throw new Error("WGSL: " + errs[0].message);

  await gemvSmokeTest(device, module);
  return { device, module };
}

// One Q8_0 gemv: 4 rows × 64 cols of known weights against a known x,
// checked against a JS reference. Proves buffer layout (36-byte u32
// blocks), bind groups and dispatch before any model bytes exist.
async function gemvSmokeTest(device, module) {
  const rows = 4, cols = 64, bpr = cols / 32;
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
    new data.constructor(buf.getMappedRange()).set(data);
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
