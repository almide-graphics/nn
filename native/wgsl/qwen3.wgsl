struct Step { token: u32, pos: u32, seq: u32, _p: u32 }
struct GemvP { rows: u32, cols: u32, blocks_per_row: u32, w_off: u32 }
struct NormP { n: u32, gamma_off: u32, n_heads: u32, head_dim: u32 }
struct RopeP { n_heads: u32, head_dim: u32, theta: f32, _p: u32 }
struct AttnP { n_q_heads: u32, n_kv_heads: u32, head_dim: u32, _p: u32 }
struct EwP   { n: u32, _a: u32, _b: u32, _c: u32 }

@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> xin: array<f32>;
@group(0) @binding(2) var<storage, read_write> yout: array<f32>;
@group(0) @binding(3) var<uniform> gp: GemvP;

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn gemv(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.y * 32768u + wid.x;
    if (row >= gp.rows) { return; }
    let row_base = gp.w_off + row * gp.blocks_per_row * 9u;
    var acc = 0.0;
    var b = lid.x;
    while (b < gp.blocks_per_row) {
        let base = row_base + b * 9u;
        let scale = bitcast<f32>(w[base]);
        var s = 0.0;
        let xb = b * 32u;
        for (var k = 0u; k < 8u; k = k + 1u) {
            let word = w[base + 1u + k];
            let xk = xb + k * 4u;
            s = s + f32(i32(word << 24u) >> 24u) * xin[xk]
                  + f32(i32(word << 16u) >> 24u) * xin[xk + 1u]
                  + f32(i32(word << 8u) >> 24u) * xin[xk + 2u]
                  + f32(i32(word) >> 24u) * xin[xk + 3u];
        }
        acc = acc + scale * s;
        b = b + 64u;
    }
    partial[lid.x] = acc;
    workgroupBarrier();
    var stride = 32u;
    while (stride > 0u) {
        if (lid.x < stride) { partial[lid.x] = partial[lid.x] + partial[lid.x + stride]; }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    if (lid.x == 0u) { yout[row] = partial[0u]; }
}

// Embedding gather: yout[0..cols] = dequant(emb row = step.token)
@group(0) @binding(0) var<storage, read> ew: array<u32>;
@group(0) @binding(1) var<storage, read_write> eout: array<f32>;
@group(0) @binding(2) var<uniform> egp: GemvP; // rows=vocab, cols=hidden, w_off
@group(0) @binding(3) var<uniform> estep: Step;

@compute @workgroup_size(256)
fn embed(@builtin(local_invocation_id) lid: vec3<u32>) {
    let row_base = egp.w_off + estep.token * egp.blocks_per_row * 9u;
    var i = lid.x;
    while (i < egp.cols) {
        let b = i / 32u;
        let k = i % 32u;
        let base = row_base + b * 9u;
        let scale = bitcast<f32>(ew[base]);
        let word = ew[base + 1u + k / 4u];
        let shift = (k % 4u) * 8u;
        let q = f32(i32(word << (24u - shift)) >> 24u);
        eout[i] = scale * q;
        i = i + 256u;
    }
}

// RMS norm: nout = nin * gamma / rms(nin). One workgroup.
// When n_heads > 0: per-head mode (Qwen3 QK-norm) — one workgroup per head,
// gamma indexed within head_dim.
@group(0) @binding(0) var<storage, read> nin: array<f32>;
@group(0) @binding(1) var<storage, read_write> nout: array<f32>;
@group(0) @binding(2) var<storage, read> gammas: array<f32>;
@group(0) @binding(3) var<uniform> np: NormP;

var<workgroup> nsum: array<f32, 256>;

@compute @workgroup_size(256)
fn rmsnorm(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    var n = np.n;
    var base = 0u;
    var goff = np.gamma_off;
    if (np.n_heads > 0u) {
        n = np.head_dim;
        base = wid.x * np.head_dim;
    }
    var local = 0.0;
    var i = lid.x;
    while (i < n) {
        let v = nin[base + i];
        local = local + v * v;
        i = i + 256u;
    }
    nsum[lid.x] = local;
    workgroupBarrier();
    var stride = 128u;
    while (stride > 0u) {
        if (lid.x < stride) { nsum[lid.x] = nsum[lid.x] + nsum[lid.x + stride]; }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    let inv = inverseSqrt(nsum[0u] / f32(n) + 1e-6);
    i = lid.x;
    while (i < n) {
        nout[base + i] = nin[base + i] * inv * gammas[goff + i];
        i = i + 256u;
    }
}

// In-place per-head RMS norm (QK-norm): one read_write binding — wgpu
// forbids binding one buffer as both read and read_write in a pass. Safe:
// each element is read and written by the same thread, after the barrier.
@group(0) @binding(0) var<storage, read_write> hx: array<f32>;
@group(0) @binding(1) var<storage, read> hgammas: array<f32>;
@group(0) @binding(2) var<uniform> hp: NormP;

var<workgroup> hsum: array<f32, 256>;

@compute @workgroup_size(256)
fn rmsnorm_inplace(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let n = hp.head_dim;
    let base = wid.x * hp.head_dim;
    var local = 0.0;
    var i = lid.x;
    while (i < n) {
        let v = hx[base + i];
        local = local + v * v;
        i = i + 256u;
    }
    hsum[lid.x] = local;
    workgroupBarrier();
    var stride = 128u;
    while (stride > 0u) {
        if (lid.x < stride) { hsum[lid.x] = hsum[lid.x] + hsum[lid.x + stride]; }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    let inv = inverseSqrt(hsum[0u] / f32(n) + 1e-6);
    i = lid.x;
    while (i < n) {
        hx[base + i] = hx[base + i] * inv * hgammas[hp.gamma_off + i];
        i = i + 256u;
    }
}

// NeoX RoPE in place at absolute position step.pos. One workgroup per head.
@group(0) @binding(0) var<storage, read_write> rx: array<f32>;
@group(0) @binding(1) var<uniform> rp: RopeP;
@group(0) @binding(2) var<uniform> rstep: Step;

@compute @workgroup_size(64)
fn rope(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let half = rp.head_dim / 2u;
    let base = wid.x * rp.head_dim;
    var j = lid.x;
    while (j < half) {
        let inv_freq = pow(rp.theta, -2.0 * f32(j) / f32(rp.head_dim));
        let angle = f32(rstep.pos) * inv_freq;
        let c = cos(angle);
        let s = sin(angle);
        let x0 = rx[base + j];
        let x1 = rx[base + half + j];
        rx[base + j] = x0 * c - x1 * s;
        rx[base + half + j] = x0 * s + x1 * c;
        j = j + 64u;
    }
}

// GQA attention, single query row over the KV cache (seq = step.seq rows).
// One workgroup per q-head.
@group(0) @binding(0) var<storage, read> aq: array<f32>;
@group(0) @binding(1) var<storage, read> ak: array<f32>;
@group(0) @binding(2) var<storage, read> av: array<f32>;
@group(0) @binding(3) var<storage, read_write> aout: array<f32>;
@group(0) @binding(4) var<uniform> ap: AttnP;
@group(0) @binding(5) var<uniform> astep: Step;

var<workgroup> scores: array<f32, 512>;
var<workgroup> red: array<f32, 128>;

@compute @workgroup_size(128)
fn attn(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let h = wid.x;
    let dh = ap.head_dim;
    let group = ap.n_q_heads / ap.n_kv_heads;
    let kvh = h / group;
    let kv_row = ap.n_kv_heads * dh;
    let qbase = h * dh;
    let kvbase = kvh * dh;
    let seq = astep.seq;
    let scale = inverseSqrt(f32(dh));

    // scores
    var j = lid.x;
    while (j < seq) {
        var s = 0.0;
        let kb = j * kv_row + kvbase;
        for (var d = 0u; d < dh; d = d + 1u) {
            s = s + aq[qbase + d] * ak[kb + d];
        }
        scores[j] = s * scale;
        j = j + 128u;
    }
    workgroupBarrier();

    // max
    var lmax = -3.0e38;
    j = lid.x;
    while (j < seq) { lmax = max(lmax, scores[j]); j = j + 128u; }
    red[lid.x] = lmax;
    workgroupBarrier();
    var stride = 64u;
    while (stride > 0u) {
        if (lid.x < stride) { red[lid.x] = max(red[lid.x], red[lid.x + stride]); }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    let m = red[0u];
    workgroupBarrier();

    // exp + sum
    var lsum = 0.0;
    j = lid.x;
    while (j < seq) {
        let e = exp(scores[j] - m);
        scores[j] = e;
        lsum = lsum + e;
        j = j + 128u;
    }
    red[lid.x] = lsum;
    workgroupBarrier();
    stride = 64u;
    while (stride > 0u) {
        if (lid.x < stride) { red[lid.x] = red[lid.x] + red[lid.x + stride]; }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    let inv = 1.0 / red[0u];
    workgroupBarrier();

    // weighted V: thread d accumulates over seq
    var d = lid.x;
    while (d < dh) {
        var acc = 0.0;
        for (var jj = 0u; jj < seq; jj = jj + 1u) {
            acc = acc + scores[jj] * av[jj * kv_row + kvbase + d];
        }
        aout[qbase + d] = acc * inv;
        d = d + 128u;
    }
}

// silu(gate) * up
@group(0) @binding(0) var<storage, read> sg: array<f32>;
@group(0) @binding(1) var<storage, read> su: array<f32>;
@group(0) @binding(2) var<storage, read_write> sout: array<f32>;
@group(0) @binding(3) var<uniform> sp: EwP;

@compute @workgroup_size(256)
fn silu_mul(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = wid.x * 256u + lid.x;
    if (i < sp.n) {
        let g = sg[i];
        sout[i] = g / (1.0 + exp(-g)) * su[i];
    }
}

// y = a + b
@group(0) @binding(0) var<storage, read> ada: array<f32>;
@group(0) @binding(1) var<storage, read> adb: array<f32>;
@group(0) @binding(2) var<storage, read_write> ady: array<f32>;
@group(0) @binding(3) var<uniform> adp: EwP;

@compute @workgroup_size(256)
fn add(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = wid.x * 256u + lid.x;
    if (i < adp.n) {
        ady[i] = ada[i] + adb[i];
    }
}
