// nn browser M1 — full Qwen3 layer stack on WebGPU, faithful port of
// native/gpu_model.rs (same WGSL, same op order, same buffer layout).
//
// Scope: load a Q8_0 GGUF (the tiny seeded test model), build the
// per-token op list, run tokens, read logits back. Verified against
// testdata/tiny_ref_logits.txt produced by the NATIVE GPU path
// (examples/_gpu_tiny_ref.almd) — same GGUF, same kernels, other host.
//
// In M2 the GGUF parse + op-list construction moves into nn.wasm
// (Almide); the executor half of this file stays.

const MAX_SEQ = 512; // mirror native/gpu_model.rs

// ── GGUF parsing (JS mirror of src/gguf_prims.almd) ──

function parseGGUF(buf) {
  const dv = new DataView(buf);
  const u32 = (p) => dv.getUint32(p, true);
  const u64 = (p) => Number(dv.getBigUint64(p, true));
  const dec = new TextDecoder();
  const str = (p) => {
    const len = u64(p);
    return [dec.decode(new Uint8Array(buf, p + 8, len)), p + 8 + len];
  };
  if (u32(0) !== 0x46554747) throw new Error("not a GGUF file");
  const tensorCount = u64(8);
  const metaCount = u64(16);

  const SCALAR = { 0: 1, 1: 1, 7: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 10: 8, 11: 8, 12: 8 };
  const meta = {};
  let p = 24;
  const readVal = (pos, vtype) => {
    if (vtype === 8) { const [s, np] = str(pos); return [s, np]; }
    if (vtype === 9) {
      const et = u32(pos); const cnt = u64(pos + 4);
      let q = pos + 12;
      if (et === 8) { for (let i = 0; i < cnt; i++) q = q + 8 + u64(q); return [null, q]; }
      return [null, q + cnt * SCALAR[et]];
    }
    let v = null;
    if (vtype === 4 || vtype === 5) v = u32(pos);
    else if (vtype === 10 || vtype === 11) v = u64(pos);
    else if (vtype === 6) v = dv.getFloat32(pos, true);
    else if (vtype === 0) v = dv.getUint8(pos);
    return [v, pos + SCALAR[vtype]];
  };
  for (let i = 0; i < metaCount; i++) {
    const [key, p1] = str(p);
    const vtype = u32(p1);
    const [v, p2] = readVal(p1 + 4, vtype);
    if (v !== null) meta[key] = v;
    p = p2;
  }

  const tensors = {};
  for (let i = 0; i < tensorCount; i++) {
    const [name, p1] = str(p);
    const nd = u32(p1);
    let q = p1 + 4;
    const ne = [];
    for (let d = 0; d < nd; d++) { ne.push(u64(q)); q += 8; }
    const dtype = u32(q);
    const off = u64(q + 4);
    tensors[name] = { ne, dtype, off };
    p = q + 12;
  }
  const align = meta["general.alignment"] || 32;
  const dataOff = Math.ceil(p / align) * align;
  return { meta, tensors, dataOff, buf };
}

// ── Q8 repack: 34-byte blocks → 9 u32 words (f32 scale + 8 quant words) ──

function fp16ToF32(h) {
  const s = h & 0x8000 ? -1 : 1, e = (h & 0x7c00) >> 10, f = h & 0x03ff;
  if (e === 0) return s * Math.pow(2, -14) * (f / 1024);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * Math.pow(2, e - 15) * (1 + f / 1024);
}

// Repack a GROUP of Q8 tensors into one Uint32Array (preallocated — the
// real model is ~170M words). One group per layer + one for the
// embedding: per-layer buffers keep every binding under iGPU
// maxStorageBufferBindingSize limits (layers ~23 MiB, embedding
// ~167 MiB), where one monolithic 650 MiB buffer would not bind.
function packGroup(gguf, names) {
  let total = 0;
  for (const name of names) {
    const t = gguf.tensors[name];
    if (!t) throw new Error("missing tensor " + name);
    total += (t.ne[0] / 32) * t.ne[1] * 9;
  }
  const words = new Uint32Array(total);
  const offs = {};
  const bytes = new Uint8Array(gguf.buf);
  const dv = new DataView(gguf.buf);
  const sf = new Float32Array(1);
  const su = new Uint32Array(sf.buffer);
  let w = 0;
  for (const name of names) {
    const t = gguf.tensors[name];
    const [cols, rows] = t.ne; // GGUF ne = [in, out]
    const bpr = cols / 32;
    const base = gguf.dataOff + t.off;
    offs[name] = w;
    for (let r = 0; r < rows; r++) {
      const rb = base + r * bpr * 34;
      for (let b = 0; b < bpr; b++) {
        const bb = rb + b * 34;
        sf[0] = fp16ToF32(bytes[bb] | (bytes[bb + 1] << 8));
        words[w++] = su[0];
        for (let k = 0; k < 8; k++) {
          words[w++] = dv.getUint32(bb + 2 + k * 4, true);
        }
      }
    }
  }
  return { words, offs };
}

class GammaPacker {
  constructor(gguf) {
    this.gguf = gguf;
    this.vals = [];
  }
  pack(name) {
    const t = this.gguf.tensors[name];
    if (!t) throw new Error("missing gamma " + name);
    const n = t.ne[0];
    const dv = new DataView(this.gguf.buf);
    const base = this.gguf.dataOff + t.off;
    const start = this.vals.length;
    for (let i = 0; i < n; i++) this.vals.push(dv.getFloat32(base + i * 4, true));
    return start;
  }
}

// ── Model: pipelines, buffers, op list (port of gpu_model load_model) ──

const ENTRIES = ["gemv", "embed", "rmsnorm", "rope", "attn", "silu_mul", "add", "rmsnorm_inplace"];
const [P_GEMV, P_EMBED, P_RMS, P_ROPE, P_ATTN, P_SILU, P_ADD, P_RMS_IP] = [0, 1, 2, 3, 4, 5, 6, 7];

// binding layouts per pipeline: s=storage-ro, S=storage-rw, u=uniform
const LAYOUTS = [
  "ssSu",   // gemv:  w x y params
  "sSuu",   // embed: w h params step
  "sSsu",   // rms:   in out gammas params
  "Suu",    // rope:  x params step
  "sssSuu", // attn:  q kcache vcache out params step
  "ssSu",   // silu:  gate up out params
  "ssSu",   // add:   a b out params
  "Ssu",    // rms_ip: x gammas params
];

export async function loadModel(device, wgslText, ggufBuf, report = () => {}) {
  const g = parseGGUF(ggufBuf);
  const m = g.meta;
  const nLayers = m["qwen3.block_count"];
  const nHeads = m["qwen3.attention.head_count"];
  const nKv = m["qwen3.attention.head_count_kv"];
  const headDim = m["qwen3.attention.key_length"];
  const hidden = m["qwen3.embedding_length"];
  const ffn = m["qwen3.feed_forward_length"];
  const theta = m["qwen3.rope.freq_base"];
  const vocab = g.tensors["token_embd.weight"].ne[1];
  const qHidden = nHeads * headDim;
  const kvHidden = nKv * headDim;
  report(`model: ${nLayers}L hidden=${hidden} heads=${nHeads}/${nKv} ffn=${ffn} vocab=${vocab}`);

  const mkInit = (data, usage) => {
    const buf = device.createBuffer({ size: data.byteLength, usage, mappedAtCreation: true });
    new data.constructor(buf.getMappedRange()).set(data);
    buf.unmap();
    return buf;
  };

  // repack weights per group (embedding + one group per layer) and
  // upload each group as its own buffer — see packGroup
  const LAYER_TENSORS = ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"];
  const embGroup = packGroup(g, ["token_embd.weight"]);
  const embBuf = mkInit(embGroup.words, GPUBufferUsage.STORAGE);
  const embW = embGroup.offs["token_embd.weight"]; // 0

  const gk = new GammaPacker(g);
  const layerBufs = [];
  const layerW = [];
  const layerG = [];
  for (let l = 0; l < nLayers; l++) {
    const p = `blk.${l}.`;
    const names = LAYER_TENSORS.map((s) => p + s + ".weight");
    const grp = packGroup(g, names);
    layerBufs.push(mkInit(grp.words, GPUBufferUsage.STORAGE));
    layerW.push(names.map((n) => grp.offs[n]));
    layerG.push([
      gk.pack(p + "attn_norm.weight"), gk.pack(p + "attn_q_norm.weight"),
      gk.pack(p + "attn_k_norm.weight"), gk.pack(p + "ffn_norm.weight"),
    ]);
  }
  const outNormG = gk.pack("output_norm.weight");
  const gammasBuf = mkInit(new Float32Array(gk.vals), GPUBufferUsage.STORAGE);

  // pipelines with explicit layouts (mirrors Builder::new)
  const module = device.createShaderModule({ code: wgslText });
  const layouts = LAYOUTS.map((spec) =>
    device.createBindGroupLayout({
      entries: Array.from(spec).map((ch, i) => ({
        binding: i,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: ch === "u" ? "uniform" : ch === "s" ? "read-only-storage" : "storage" },
      })),
    })
  );
  const pipelines = ENTRIES.map((entryPoint, i) =>
    device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [layouts[i]] }),
      compute: { module, entryPoint },
    })
  );

  const storage = (nF32) =>
    device.createBuffer({
      size: nF32 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
  const uniform = (vals) => mkInit(new Uint32Array(vals), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

  // activations
  const h = storage(hidden), xn = storage(hidden);
  const q = storage(qHidden), kCur = storage(kvHidden), vCur = storage(kvHidden);
  const attnOut = storage(qHidden), proj = storage(hidden), h2 = storage(hidden);
  const gate = storage(ffn), up = storage(ffn), gated = storage(ffn);
  const logits = storage(vocab);
  const stepUniform = uniform([0, 0, 1, 0]);
  const kBufs = [], vBufs = [];
  for (let l = 0; l < nLayers; l++) {
    kBufs.push(storage(MAX_SEQ * kvHidden));
    vBufs.push(storage(MAX_SEQ * kvHidden));
  }
  const staging = device.createBuffer({
    size: vocab * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const bind = (layout, bufs) =>
    device.createBindGroup({
      layout: layouts[layout],
      entries: bufs.map((b, i) => ({ binding: i, resource: { buffer: b } })),
    });
  const f32bits = (v) => new Uint32Array(new Float32Array([v]).buffer)[0];

  const gemvOp = (wb, x, y, rows, cols, wOff) => ({
    pipeline: P_GEMV,
    bind: bind(P_GEMV, [wb, x, y, uniform([rows, cols, cols / 32, wOff])]),
    groups: [Math.min(rows, 32768), Math.ceil(rows / 32768), 1],
  });
  const rmsOp = (xi, xo, n, gOff) => ({
    pipeline: P_RMS,
    bind: bind(P_RMS, [xi, xo, gammasBuf, uniform([n, gOff, 0, 0])]),
    groups: [1, 1, 1],
  });
  const rmsIpOp = (x, gOff, heads) => ({
    pipeline: P_RMS_IP,
    bind: bind(P_RMS_IP, [x, gammasBuf, uniform([0, gOff, heads, headDim])]),
    groups: [heads, 1, 1],
  });
  const ropeOp = (x, heads) => ({
    pipeline: P_ROPE,
    bind: bind(P_ROPE, [x, uniform([heads, headDim, f32bits(theta), 0]), stepUniform]),
    groups: [heads, 1, 1],
  });
  const ewOp = (pl, a, c, y, n) => ({
    pipeline: pl,
    bind: bind(pl, [a, c, y, uniform([n, 0, 0, 0])]),
    groups: [Math.ceil(n / 256), 1, 1],
  });

  const preOps = [{
    pipeline: P_EMBED,
    bind: bind(P_EMBED, [embBuf, h, uniform([vocab, hidden, hidden / 32, embW]), stepUniform]),
    groups: [1, 1, 1],
  }];

  const layers = [];
  for (let l = 0; l < nLayers; l++) {
    const wo = layerW[l], go = layerG[l], wb = layerBufs[l];
    const ops = [
      rmsOp(h, xn, hidden, go[0]),
      gemvOp(wb, xn, q, qHidden, hidden, wo[0]),
      gemvOp(wb, xn, kCur, kvHidden, hidden, wo[1]),
      gemvOp(wb, xn, vCur, kvHidden, hidden, wo[2]),
      rmsIpOp(q, go[1], nHeads),
      rmsIpOp(kCur, go[2], nKv),
      ropeOp(q, nHeads),
      ropeOp(kCur, nKv),
      // ← kv append copies happen between split point (8) and the rest
      {
        pipeline: P_ATTN,
        bind: bind(P_ATTN, [q, kBufs[l], vBufs[l], attnOut,
          uniform([nHeads, nKv, headDim, 0]), stepUniform]),
        groups: [nHeads, 1, 1],
      },
      gemvOp(wb, attnOut, proj, hidden, qHidden, wo[3]),
      ewOp(P_ADD, h, proj, h2, hidden),
      rmsOp(h2, xn, hidden, go[3]),
      gemvOp(wb, xn, gate, ffn, hidden, wo[4]),
      gemvOp(wb, xn, up, ffn, hidden, wo[5]),
      ewOp(P_SILU, gate, up, gated, ffn),
      gemvOp(wb, gated, proj, hidden, ffn, wo[6]),
      ewOp(P_ADD, h2, proj, h, hidden),
    ];
    layers.push(ops);
  }
  const postOps = [
    rmsOp(h, xn, hidden, outNormG),
    gemvOp(embBuf, xn, logits, vocab, hidden, embW), // tied embeddings
  ];

  const kvRowBytes = kvHidden * 4;

  async function step(token, pos) {
    device.queue.writeBuffer(stepUniform, 0, new Uint32Array([token, pos, pos + 1, 0]));
    const enc = device.createCommandEncoder();
    const runOps = (ops) => {
      const pass = enc.beginComputePass();
      for (const op of ops) {
        pass.setPipeline(pipelines[op.pipeline]);
        pass.setBindGroup(0, op.bind);
        pass.dispatchWorkgroups(...op.groups);
      }
      pass.end();
    };
    runOps(preOps);
    for (let l = 0; l < nLayers; l++) {
      runOps(layers[l].slice(0, 8));
      const dst = pos * kvRowBytes;
      enc.copyBufferToBuffer(kCur, 0, kBufs[l], dst, kvRowBytes);
      enc.copyBufferToBuffer(vCur, 0, vBufs[l], dst, kvRowBytes);
      runOps(layers[l].slice(8));
    }
    runOps(postOps);
    enc.copyBufferToBuffer(logits, 0, staging, 0, vocab * 4);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(staging.getMappedRange()).slice();
    staging.unmap();
    return out;
  }

  return { step, vocab, nLayers, hidden };
}
