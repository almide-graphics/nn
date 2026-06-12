// nn native CPU fast path: lockstep token executor (the "win" kernel).
//
// The generic per-op runtime measured 70% of thread-time PARKED: every
// parallel region wakes/drains the pool and all the small glue (rms, rope,
// silu, quantize, attention, residuals) ran serially between regions. Here
// one rayon scope runs the WHOLE token: 8 workers execute every stage in
// lockstep, synchronized by spin barriers (~1µs), with all the glue
// parallelized and fused:
//   - residual adds folded into the o/down GEMV row writes
//   - silu folded into the activation quantization sweep
//   - KV cache lives in a preallocated pool (no per-step append copies)
//   - activations are f32 (the GPU pipeline already validated f32 ids)
//
// Same shape as gpu_model.rs: load once, then decode_argmax per token.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

const MAX_SEQ: usize = 512;
const Q8B: usize = 32; // values per Q8_0 block
const Q8BB: usize = 34; // bytes per Q8_0 block

struct SpinBarrier {
    count: AtomicUsize,
    gen: AtomicUsize,
    n: usize,
}

impl SpinBarrier {
    fn new(n: usize) -> Self {
        SpinBarrier { count: AtomicUsize::new(0), gen: AtomicUsize::new(0), n }
    }
    #[inline]
    fn wait(&self) {
        let g = self.gen.load(Ordering::Acquire);
        if self.count.fetch_add(1, Ordering::AcqRel) + 1 == self.n {
            self.count.store(0, Ordering::Release);
            self.gen.fetch_add(1, Ordering::Release);
        } else {
            while self.gen.load(Ordering::Acquire) == g {
                std::hint::spin_loop();
            }
        }
    }
}

/// Mutable state shared across lockstep workers. Stages write DISJOINT
/// ranges and are separated by barriers, so the raw-pointer sharing is
/// sound; the wrapper just tells the compiler.
#[derive(Clone, Copy)]
struct Shared<T>(*mut T);
unsafe impl<T> Send for Shared<T> {}
unsafe impl<T> Sync for Shared<T> {}

impl<T> Shared<T> {
    // Method receiver forces whole-struct closure capture (Rust 2021
    // disjoint capture would otherwise grab the bare *mut field, which is
    // not Send).
    #[inline]
    fn ptr(&self) -> *mut T {
        self.0
    }
}

struct Cfg {
    n_layers: usize,
    n_heads: usize,
    n_kv: usize,
    head_dim: usize,
    hidden: usize,
    q_hidden: usize,
    kv_hidden: usize,
    ffn: usize,
    vocab: usize,
    rope_theta: f32,
    eps: f32,
    emb_off: usize, // byte offset into wrep
    out_norm_g: usize, // index into gammas
}

struct State {
    cfg: Cfg,
    gammas: Vec<f32>,             // concatenated f32 gammas
    layer_goffs: Vec<[usize; 4]>, // [attn, q, k, ffn] indices into gammas
    layer_woffs: Vec<[usize; 7]>, // [q,k,v,o,gate,up,down] byte offsets into wrep
    // Weights repacked at load: groups of 4 rows interleaved per block —
    // [4×f32 scales][16B pad][4×32 i8 quants] = 160B per group-block,
    // 32-aligned, f16 scales pre-converted. Kills the f16 decode and the
    // 34-byte-stride cache-line straddles in the hot loop.
    wrep: Vec<u32>,
    wrep_base: usize, // element offset aligning the data to 32 bytes
    inv_freq: Vec<f32>,           // head_dim/2 rope frequencies
    // activations (f32)
    h: Vec<f32>,
    xn: Vec<f32>,
    q: Vec<f32>,
    attn_out: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    logits: Vec<f32>,
    // quantized activation scratch: scales + i8 (max over hidden/ffn)
    xq_s: Vec<f32>,
    xq: Vec<i8>,
    // KV cache pool: [layer][pos*kv_hidden + i]
    k_cache: Vec<Vec<f32>>,
    v_cache: Vec<Vec<f32>>,
    // per-thread reduction scratch
    red: Vec<f32>,
    arg_v: Vec<f32>,
    arg_i: Vec<usize>,
}

static STATE: OnceLock<Mutex<Option<State>>> = OnceLock::new();

fn cell() -> &'static Mutex<Option<State>> {
    STATE.get_or_init(|| Mutex::new(None))
}

fn fp16(raw: u16) -> f32 {
    super::gpu::fp16_to_f32_pub(raw)
}

#[allow(clippy::too_many_arguments)]
pub fn load_model(
    raw: &Vec<u8>,
    n_layers: i64,
    n_heads: i64,
    n_kv_heads: i64,
    head_dim: i64,
    ffn_hidden: i64,
    hidden: i64,
    vocab: i64,
    rope_theta: f64,
    eps: f64,
    emb_off: i64,
    out_norm_off: i64,
    gammas_in: &Vec<i64>,
    weights_in: &Vec<i64>,
) -> i64 {
    let n_layers_u = n_layers as usize;
    let hidden_u = hidden as usize;
    let head_dim_u = head_dim as usize;
    let q_hidden = (n_heads * head_dim) as usize;
    let kv_hidden = (n_kv_heads * head_dim) as usize;
    let ffn = ffn_hidden as usize;
    let vocab_u = vocab as usize;

    let mut gammas = Vec::<f32>::new();
    let mut push_g = |g: &mut Vec<f32>, byte_off: usize, n: usize| -> usize {
        let start = g.len();
        for i in 0..n {
            let b = byte_off + i * 4;
            g.push(f32::from_le_bytes([raw[b], raw[b + 1], raw[b + 2], raw[b + 3]]));
        }
        start
    };
    let mut layer_goffs = Vec::with_capacity(n_layers_u);
    for l in 0..n_layers_u {
        layer_goffs.push([
            push_g(&mut gammas, gammas_in[l * 4] as usize, hidden_u),
            push_g(&mut gammas, gammas_in[l * 4 + 1] as usize, head_dim_u),
            push_g(&mut gammas, gammas_in[l * 4 + 2] as usize, head_dim_u),
            push_g(&mut gammas, gammas_in[l * 4 + 3] as usize, hidden_u),
        ]);
    }
    let out_norm_g = push_g(&mut gammas, out_norm_off as usize, hidden_u);

    let mut layer_woffs = Vec::with_capacity(n_layers_u);
    for l in 0..n_layers_u {
        let mut o = [0usize; 7];
        for t in 0..7 {
            o[t] = weights_in[l * 7 + t] as usize;
        }
        layer_woffs.push(o);
    }
    let wrep: Vec<u32> = Vec::new();

    let half = head_dim_u / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|j| 1.0 / (rope_theta as f32).powf(2.0 * j as f32 / head_dim_u as f32))
        .collect();

    let n_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let cfg = Cfg {
        n_layers: n_layers_u,
        n_heads: n_heads as usize,
        n_kv: n_kv_heads as usize,
        head_dim: head_dim_u,
        hidden: hidden_u,
        q_hidden,
        kv_hidden,
        ffn,
        vocab: vocab_u,
        rope_theta: rope_theta as f32,
        eps: eps as f32,
        emb_off: emb_off as usize,
        out_norm_g,
    };
    let st = State {
        cfg,
        gammas,
        layer_goffs,
        layer_woffs,
        wrep,
        wrep_base: 0,
        inv_freq,
        h: vec![0.0; hidden_u],
        xn: vec![0.0; hidden_u.max(ffn)],
        q: vec![0.0; q_hidden + 2 * kv_hidden], // q | k_cur | v_cur contiguous
        attn_out: vec![0.0; q_hidden],
        gate: vec![0.0; ffn],
        up: vec![0.0; ffn],
        logits: vec![0.0; vocab_u],
        xq_s: vec![0.0; hidden_u.max(ffn) / Q8B],
        xq: vec![0; hidden_u.max(ffn)],
        k_cache: (0..n_layers_u).map(|_| vec![0.0; MAX_SEQ * kv_hidden]).collect(),
        v_cache: (0..n_layers_u).map(|_| vec![0.0; MAX_SEQ * kv_hidden]).collect(),
        red: vec![0.0; n_threads],
        arg_v: vec![0.0; n_threads],
        arg_i: vec![0; n_threads],
    };
    *cell().lock().unwrap() = Some(st);
    1
}

/// Spinning workers on hyperthread siblings steal cycles from the sibling
/// doing real work — use physical cores (or ALMIDE_LOCKSTEP_THREADS).
fn lockstep_threads() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        if let Ok(v) = std::env::var("ALMIDE_LOCKSTEP_THREADS") {
            if let Ok(n) = v.parse::<usize>() {
                return n.max(1);
            }
        }
        let logical = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        (logical / 2).max(1)
    })
}

#[inline]
fn split(n: usize, nt: usize, tid: usize) -> (usize, usize) {
    let chunk = n.div_ceil(nt);
    let s = (tid * chunk).min(n);
    (s, (s + chunk).min(n))
}

#[inline(always)]
fn fp16_inline(raw: u16) -> f32 {
    let sign = (raw >> 15) as u32;
    let exp = ((raw >> 10) & 0x1F) as u32;
    let man = (raw & 0x3FF) as u32;
    let bits = if exp == 0 {
        if man == 0 { sign << 31 } else {
            let mut m = man;
            let mut e = 127 - 15 + 1;
            while m & 0x400 == 0 { m <<= 1; e -= 1; }
            (sign << 31) | ((e as u32) << 23) | ((m & 0x3FF) << 13)
        }
    } else if exp == 31 {
        (sign << 31) | (0xFF << 23) | (man << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

/// AVX2 sign+maddubs Q8 dot (the proven runtime recipe), vector f32 FMA
/// accumulators, one horizontal reduction per row.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn q8_row_dot_avx2(xs: &[f32], xq: &[i8], row: &[u8]) -> f32 {
    use std::arch::x86_64::*;
    let n_blocks = row.len() / Q8BB;
    let rp = row.as_ptr();
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut b = 0;
    while b < n_blocks {
        let blk = rp.add(b * Q8BB);
        // F16C scale decode (2 ops) — the scalar bit-twiddle was ~15% of
        // the row cost. Prefetch the next block's cache lines while at it.
        _mm_prefetch::<_MM_HINT_T0>(blk.add(Q8BB) as *const i8);
        _mm_prefetch::<_MM_HINT_T0>(blk.add(Q8BB + 32) as *const i8);
        let raw16 = u16::from_le_bytes([*blk, *blk.add(1)]) as i32;
        let d = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(raw16)));
        let scale = _mm256_set1_ps(d * *xs.get_unchecked(b));
        let w = _mm256_loadu_si256(blk.add(2) as *const __m256i);
        let x = _mm256_loadu_si256(xq.as_ptr().add(b * Q8B) as *const __m256i);
        let ax = _mm256_sign_epi8(x, x);
        let sw = _mm256_sign_epi8(w, x);
        let p32 = _mm256_madd_epi16(_mm256_maddubs_epi16(ax, sw), _mm256_set1_epi16(1));
        if b & 1 == 0 {
            acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), scale, acc0);
        } else {
            acc1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), scale, acc1);
        }
        b += 1;
    }
    let acc = _mm256_add_ps(acc0, acc1);
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let s4 = _mm_add_ps(hi, lo);
    let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    let s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 0b0000_0001));
    _mm_cvtss_f32(s1)
}

/// Two weight rows against one quantized activation: shares the x loads
/// and |x| sign pass across both rows (~15% fewer ops than 2 single dots,
/// and half the x traffic).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn q8_two_row_dot_avx2(xs: &[f32], xq: &[i8], row0: &[u8], row1: &[u8]) -> (f32, f32) {
    use std::arch::x86_64::*;
    let n_blocks = row0.len() / Q8BB;
    let rp0 = row0.as_ptr();
    let rp1 = row1.as_ptr();
    let mut a0 = _mm256_setzero_ps();
    let mut a1 = _mm256_setzero_ps();
    for b in 0..n_blocks {
        let blk0 = rp0.add(b * Q8BB);
        let blk1 = rp1.add(b * Q8BB);
        _mm_prefetch::<_MM_HINT_T0>(blk0.add(Q8BB) as *const i8);
        _mm_prefetch::<_MM_HINT_T0>(blk1.add(Q8BB) as *const i8);
        let x = _mm256_loadu_si256(xq.as_ptr().add(b * Q8B) as *const __m256i);
        let ax = _mm256_sign_epi8(x, x);
        let xsb = *xs.get_unchecked(b);
        let d0 = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(u16::from_le_bytes([*blk0, *blk0.add(1)]) as i32)));
        let d1 = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(u16::from_le_bytes([*blk1, *blk1.add(1)]) as i32)));
        let w0 = _mm256_loadu_si256(blk0.add(2) as *const __m256i);
        let w1 = _mm256_loadu_si256(blk1.add(2) as *const __m256i);
        let p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(ax, _mm256_sign_epi8(w0, x)), _mm256_set1_epi16(1));
        let p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(ax, _mm256_sign_epi8(w1, x)), _mm256_set1_epi16(1));
        a0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p0), _mm256_set1_ps(d0 * xsb), a0);
        a1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p1), _mm256_set1_ps(d1 * xsb), a1);
    }
    #[inline(always)]
    unsafe fn hsum(v: std::arch::x86_64::__m256) -> f32 {
        let hi = _mm256_extractf128_ps(v, 1);
        let lo = _mm256_castps256_ps128(v);
        let s4 = _mm_add_ps(hi, lo);
        let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
        let s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 0b0000_0001));
        _mm_cvtss_f32(s1)
    }
    (hsum(a0), hsum(a1))
}

/// Four interleaved rows (one repacked group) against one quantized
/// activation: x loaded and signed ONCE per block for 4 outputs, scales
/// pre-converted, quants at aligned 160B stride.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn q8_group_dot_avx2(xs: &[f32], xq: &[i8], group: &[u8], n_blocks: usize) -> [f32; 4] {
    use std::arch::x86_64::*;
    let gp = group.as_ptr();
    let mut a0 = _mm256_setzero_ps();
    let mut a1 = _mm256_setzero_ps();
    let mut a2 = _mm256_setzero_ps();
    let mut a3 = _mm256_setzero_ps();
    for b in 0..n_blocks {
        let blk = gp.add(b * 136);
        _mm_prefetch::<_MM_HINT_T0>(blk.add(136) as *const i8);
        _mm_prefetch::<_MM_HINT_T0>(blk.add(136 + 64) as *const i8);
        let x = _mm256_loadu_si256(xq.as_ptr().add(b * Q8B) as *const __m256i);
        let ax = _mm256_sign_epi8(x, x);
        let xsb = _mm256_set1_ps(*xs.get_unchecked(b));
        // 4 f16 scales → 4 f32 lanes in ONE cvtph_ps
        let halves = _mm_loadl_epi64(blk as *const __m128i);
        let sc = _mm_cvtph_ps(halves); // [d0 d1 d2 d3]
        let ones = _mm256_set1_epi16(1);
        let q0 = _mm256_loadu_si256(blk.add(8) as *const __m256i);
        let q1 = _mm256_loadu_si256(blk.add(40) as *const __m256i);
        let q2 = _mm256_loadu_si256(blk.add(72) as *const __m256i);
        let q3 = _mm256_loadu_si256(blk.add(104) as *const __m256i);
        let p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(ax, _mm256_sign_epi8(q0, x)), ones);
        let p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(ax, _mm256_sign_epi8(q1, x)), ones);
        let p2 = _mm256_madd_epi16(_mm256_maddubs_epi16(ax, _mm256_sign_epi8(q2, x)), ones);
        let p3 = _mm256_madd_epi16(_mm256_maddubs_epi16(ax, _mm256_sign_epi8(q3, x)), ones);
        let d0 = _mm256_set1_ps(_mm_cvtss_f32(sc));
        let d1 = _mm256_set1_ps(_mm_cvtss_f32(_mm_shuffle_ps(sc, sc, 0b01)));
        let d2 = _mm256_set1_ps(_mm_cvtss_f32(_mm_shuffle_ps(sc, sc, 0b10)));
        let d3 = _mm256_set1_ps(_mm_cvtss_f32(_mm_shuffle_ps(sc, sc, 0b11)));
        a0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p0), _mm256_mul_ps(d0, xsb), a0);
        a1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p1), _mm256_mul_ps(d1, xsb), a1);
        a2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p2), _mm256_mul_ps(d2, xsb), a2);
        a3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p3), _mm256_mul_ps(d3, xsb), a3);
    }
    #[inline(always)]
    unsafe fn hsum(v: std::arch::x86_64::__m256) -> f32 {
        let hi = _mm256_extractf128_ps(v, 1);
        let lo = _mm256_castps256_ps128(v);
        let s4 = _mm_add_ps(hi, lo);
        let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
        let s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 0b0000_0001));
        _mm_cvtss_f32(s1)
    }
    [hsum(a0), hsum(a1), hsum(a2), hsum(a3)]
}

/// Scalar fallback over the repacked layout.
fn q8_group_dot_scalar(xs: &[f32], xq: &[i8], group: &[u8], n_blocks: usize) -> [f32; 4] {
    let mut acc = [0.0f32; 4];
    for b in 0..n_blocks {
        let blk = &group[b * 136..(b + 1) * 136];
        let xsb = xs[b];
        for lane in 0..4 {
            let d = fp16_inline(u16::from_le_bytes([blk[lane * 2], blk[lane * 2 + 1]]));
            let qs = &blk[8 + lane * 32..8 + (lane + 1) * 32];
            let xb = &xq[b * Q8B..(b + 1) * Q8B];
            let mut s = 0i32;
            for k in 0..Q8B {
                s += (qs[k] as i8) as i32 * xb[k] as i32;
            }
            acc[lane] += d * xsb * s as f32;
        }
    }
    acc
}

#[inline]
fn q8_row_dot(xs: &[f32], xq: &[i8], row: &[u8]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if avx2() {
            return unsafe { q8_row_dot_avx2(xs, xq, row) };
        }
    }
    let mut acc = 0.0f32;
    for (b, blk) in row.chunks_exact(Q8BB).enumerate() {
        let d = fp16_inline(u16::from_le_bytes([blk[0], blk[1]]));
        let qs = &blk[2..2 + Q8B];
        let xb = &xq[b * Q8B..(b + 1) * Q8B];
        let mut s = 0i32;
        for k in 0..Q8B {
            s += (qs[k] as i8) as i32 * xb[k] as i32;
        }
        acc += d * xs[b] * s as f32;
    }
    acc
}

#[cfg(target_arch = "x86_64")]
fn avx2() -> bool {
    static A: OnceLock<bool> = OnceLock::new();
    *A.get_or_init(|| std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") && std::is_x86_feature_detected!("f16c"))
}

#[allow(clippy::too_many_arguments)]
fn token_pass(st: &mut State, raw: &[u8], token: usize, pos: usize, nt: usize) {
    let cfg = &st.cfg;
    let hidden = cfg.hidden;
    let seq = pos + 1;
    let barrier = SpinBarrier::new(nt);

    // raw pointers for lockstep sharing (disjoint writes per stage)
    let h_p = Shared(st.h.as_mut_ptr());
    let xn_p = Shared(st.xn.as_mut_ptr());
    let qkv_p = Shared(st.q.as_mut_ptr());
    let attn_p = Shared(st.attn_out.as_mut_ptr());
    let gate_p = Shared(st.gate.as_mut_ptr());
    let up_p = Shared(st.up.as_mut_ptr());
    let logits_p = Shared(st.logits.as_mut_ptr());
    let xqs_p = Shared(st.xq_s.as_mut_ptr());
    let xq_p = Shared(st.xq.as_mut_ptr());
    let red_p = Shared(st.red.as_mut_ptr());
    let argv_p = Shared(st.arg_v.as_mut_ptr());
    let argi_p = Shared(st.arg_i.as_mut_ptr());
    let kc: Vec<Shared<f32>> = st.k_cache.iter_mut().map(|v| Shared(v.as_mut_ptr())).collect();
    let vc: Vec<Shared<f32>> = st.v_cache.iter_mut().map(|v| Shared(v.as_mut_ptr())).collect();

    let gammas = &st.gammas;
    let goffs = &st.layer_goffs;
    let woffs = &st.layer_woffs;
    let inv_freq = &st.inv_freq;
    let n_heads = cfg.n_heads;
    let n_kv = cfg.n_kv;
    let dh = cfg.head_dim;
    let q_hidden = cfg.q_hidden;
    let kv_hidden = cfg.kv_hidden;
    let ffn = cfg.ffn;
    let vocab = cfg.vocab;
    let eps = cfg.eps;
    let n_layers = cfg.n_layers;
    let emb_off = cfg.emb_off;
    let out_norm_g = cfg.out_norm_g;

    let bar = &barrier;
    rayon::scope(|s| {
        for tid in 0..nt {
            let kc = &kc;
            let vc = &vc;
            s.spawn(move |_| unsafe {
                let h = std::slice::from_raw_parts_mut(h_p.ptr(), hidden);
                let xn = std::slice::from_raw_parts_mut(xn_p.ptr(), hidden.max(ffn));
                let qkv = std::slice::from_raw_parts_mut(qkv_p.ptr(), q_hidden + 2 * kv_hidden);
                let attn_out = std::slice::from_raw_parts_mut(attn_p.ptr(), q_hidden);
                let gate = std::slice::from_raw_parts_mut(gate_p.ptr(), ffn);
                let up = std::slice::from_raw_parts_mut(up_p.ptr(), ffn);
                let logits = std::slice::from_raw_parts_mut(logits_p.ptr(), vocab);
                let xqs = std::slice::from_raw_parts_mut(xqs_p.ptr(), hidden.max(ffn) / Q8B);
                let xq = std::slice::from_raw_parts_mut(xq_p.ptr(), hidden.max(ffn));
                let red = std::slice::from_raw_parts_mut(red_p.ptr(), nt);
                let argv = std::slice::from_raw_parts_mut(argv_p.ptr(), nt);
                let argi = std::slice::from_raw_parts_mut(argi_p.ptr(), nt);

                // ── embed: dequant emb row `token` into h ──
                {
                    let bpr = hidden / Q8B;
                    let row = &raw[emb_off + token * bpr * Q8BB..emb_off + (token + 1) * bpr * Q8BB];
                    let (bs, be_) = split(bpr, nt, tid);
                    for b in bs..be_ {
                        let blk = &row[b * Q8BB..(b + 1) * Q8BB];
                        let d = fp16_inline(u16::from_le_bytes([blk[0], blk[1]]));
                        for k in 0..Q8B {
                            h[b * Q8B + k] = d * (blk[2 + k] as i8) as f32;
                        }
                    }
                }
                bar.wait();

                // helper: rms-norm src → dst (n elems, gamma at g) — two-phase
                let rms = |src: &[f32], dst: &mut [f32], n: usize, g: usize,
                           red: &mut [f32], bar: &SpinBarrier| {
                    let (s0, s1) = split(n, nt, tid);
                    let mut local = 0.0f32;
                    for v in &src[s0..s1] {
                        local += v * v;
                    }
                    red[tid] = local;
                    bar.wait();
                    let mut total = 0.0f32;
                    for r in red.iter().take(nt) {
                        total += r;
                    }
                    let inv = 1.0 / (total / n as f32 + eps).sqrt();
                    for i in s0..s1 {
                        dst[i] = src[i] * inv * gammas[g + i];
                    }
                    bar.wait();
                };

                // helper: quantize xn[0..n] into (xqs, xq), with optional
                // silu(gate)*up fusion as the value source
                let quantize = |vals: &dyn Fn(usize) -> f32, n: usize,
                                xqs: &mut [f32], xq: &mut [i8], bar: &SpinBarrier| {
                    let nb = n / Q8B;
                    let (b0, b1) = split(nb, nt, tid);
                    for b in b0..b1 {
                        let mut amax = 0.0f32;
                        let base = b * Q8B;
                        let mut tmp = [0.0f32; Q8B];
                        for k in 0..Q8B {
                            let v = vals(base + k);
                            tmp[k] = v;
                            amax = amax.max(v.abs());
                        }
                        let d = amax / 127.0;
                        let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
                        xqs[b] = d;
                        for k in 0..Q8B {
                            xq[base + k] = (tmp[k] * inv).round().clamp(-127.0, 127.0) as i8;
                        }
                    }
                    bar.wait();
                };

                // helper: gemv rows [r0,r1) from q8 weights at byte off,
                // n_in cols, into out, plus optional residual source
                let gemv_nb = |w_off: usize, rows: usize, n_in: usize,
                               out: &mut [f32], add_self: bool,
                               xqs: &[f32], xq: &[i8]| {
                    let row_bytes = n_in / Q8B * Q8BB;
                    let (r0, r1) = split(rows, nt, tid);
                    // Single-row dots only: the 2-row variant's summation
                    // order flips a near-tie token vs the HF-exact canary
                    // (and bought <2% anyway). q8_row_dot's stride-4
                    // accumulators are the ids-verified order.
                    for r in r0..r1 {
                        let row = &raw[w_off + r * row_bytes..w_off + (r + 1) * row_bytes];
                        let dot = q8_row_dot(xqs, xq, row);
                        out[r] = if add_self { out[r] + dot } else { dot };
                    }
                };
                let gemv = |w_off: usize, rows: usize, n_in: usize,
                            out: &mut [f32], add_self: bool,
                            xqs: &[f32], xq: &[i8], bar: &SpinBarrier| {
                    gemv_nb(w_off, rows, n_in, out, add_self, xqs, xq);
                    bar.wait();
                };

                // rms + quantize fused: thread splits are 32-multiples
                // (hidden/ffn are), so per-block amax stays thread-local.
                let rms_quant = |src: &[f32], n: usize, g: usize,
                                 red: &mut [f32], xqs: &mut [f32], xq: &mut [i8],
                                 bar: &SpinBarrier| {
                    let (s0, s1) = split(n, nt, tid);
                    let mut local = 0.0f32;
                    for v in &src[s0..s1] {
                        local += v * v;
                    }
                    red[tid] = local;
                    bar.wait();
                    let mut total = 0.0f32;
                    for r in red.iter().take(nt) {
                        total += r;
                    }
                    let inv = 1.0 / (total / n as f32 + eps).sqrt();
                    let b0 = s0 / Q8B;
                    let b1 = s1 / Q8B;
                    for b in b0..b1 {
                        let base = b * Q8B;
                        let mut amax = 0.0f32;
                        let mut tmp = [0.0f32; Q8B];
                        for k in 0..Q8B {
                            let v = src[base + k] * inv * gammas[g + base + k];
                            tmp[k] = v;
                            amax = amax.max(v.abs());
                        }
                        let d = amax / 127.0;
                        let qi = if d > 0.0 { 1.0 / d } else { 0.0 };
                        xqs[b] = d;
                        for k in 0..Q8B {
                            xq[base + k] = (tmp[k] * qi).round().clamp(-127.0, 127.0) as i8;
                        }
                    }
                    bar.wait();
                };

                for l in 0..n_layers {
                    let go = &goffs[l];
                    let wo = &woffs[l];
                    let kcache = std::slice::from_raw_parts_mut(kc[l].ptr(), MAX_SEQ * kv_hidden);
                    let vcache = std::slice::from_raw_parts_mut(vc[l].ptr(), MAX_SEQ * kv_hidden);

                    // rms1 fused with activation quantization
                    rms_quant(h, hidden, go[0], red, xqs, xq, bar);

                    // qkv: three disjoint outputs — ONE barrier for the stage
                    {
                        let (qs, rest) = qkv.split_at_mut(q_hidden);
                        let (ks, vs) = rest.split_at_mut(kv_hidden);
                        gemv_nb(wo[0], q_hidden, hidden, qs, false, xqs, xq);
                        gemv_nb(wo[1], kv_hidden, hidden, ks, false, xqs, xq);
                        gemv_nb(wo[2], kv_hidden, hidden, vs, false, xqs, xq);
                    }
                    bar.wait();

                    // QK-norm + rope, split by head (q heads then k heads)
                    {
                        let total_heads = n_heads + n_kv;
                        let (h0, h1) = split(total_heads, nt, tid);
                        for hh in h0..h1 {
                            let (base, g) = if hh < n_heads {
                                (hh * dh, go[1])
                            } else {
                                (q_hidden + (hh - n_heads) * dh, go[2])
                            };
                            let mut ss = 0.0f32;
                            for k in 0..dh {
                                let v = qkv[base + k];
                                ss += v * v;
                            }
                            let inv = 1.0 / (ss / dh as f32 + eps).sqrt();
                            for k in 0..dh {
                                qkv[base + k] *= inv * gammas[g + k];
                            }
                            let half = dh / 2;
                            for j in 0..half {
                                let angle = pos as f32 * inv_freq[j];
                                let (sn, cs) = angle.sin_cos();
                                let x0 = qkv[base + j];
                                let x1 = qkv[base + half + j];
                                qkv[base + j] = x0 * cs - x1 * sn;
                                qkv[base + half + j] = x0 * sn + x1 * cs;
                            }
                        }
                    }
                    bar.wait();

                    // kv append into the pool (split kv_hidden)
                    {
                        let (i0, i1) = split(kv_hidden, nt, tid);
                        let dst = pos * kv_hidden;
                        kcache[dst + i0..dst + i1].copy_from_slice(&qkv[q_hidden + i0..q_hidden + i1]);
                        vcache[dst + i0..dst + i1].copy_from_slice(&qkv[q_hidden + kv_hidden + i0..q_hidden + kv_hidden + i1]);
                    }
                    bar.wait();

                    // attention: heads split across threads
                    {
                        let group = n_heads / n_kv;
                        let (h0, h1) = split(n_heads, nt, tid);
                        let mut scores = [0.0f32; MAX_SEQ];
                        for hh in h0..h1 {
                            let qb = hh * dh;
                            let kvb = (hh / group) * dh;
                            let scale = 1.0 / (dh as f32).sqrt();
                            let mut m = f32::NEG_INFINITY;
                            for j in 0..seq {
                                let kb = j * kv_hidden + kvb;
                                let mut sdot = 0.0f32;
                                for k in 0..dh {
                                    sdot += qkv[qb + k] * kcache[kb + k];
                                }
                                let sv = sdot * scale;
                                scores[j] = sv;
                                m = m.max(sv);
                            }
                            let mut sum = 0.0f32;
                            for sj in scores.iter_mut().take(seq) {
                                *sj = (*sj - m).exp();
                                sum += *sj;
                            }
                            let invs = 1.0 / sum;
                            for k in 0..dh {
                                attn_out[qb + k] = 0.0;
                            }
                            for j in 0..seq {
                                let w = scores[j] * invs;
                                let vb = j * kv_hidden + kvb;
                                for k in 0..dh {
                                    attn_out[qb + k] += w * vcache[vb + k];
                                }
                            }
                        }
                    }
                    bar.wait();

                    // o-proj (+ residual into h)
                    quantize(&|i| attn_out[i], q_hidden, xqs, xq, bar);
                    gemv(wo[3], hidden, q_hidden, h, true, xqs, xq, bar);

                    // rms2 fused with quantization; gate+up share one barrier
                    rms_quant(h, hidden, go[3], red, xqs, xq, bar);
                    gemv_nb(wo[4], ffn, hidden, gate, false, xqs, xq);
                    gemv_nb(wo[5], ffn, hidden, up, false, xqs, xq);
                    bar.wait();
                    quantize(&|i| {
                        let g = gate[i];
                        g / (1.0 + (-g).exp()) * up[i]
                    }, ffn, xqs, xq, bar);
                    gemv(wo[6], hidden, ffn, h, true, xqs, xq, bar);
                }

                // final norm + logits
                rms(h, xn, hidden, out_norm_g, red, bar);
                quantize(&|i| xn[i], hidden, xqs, xq, bar);
                gemv(emb_off, vocab, hidden, logits, false, xqs, xq, bar);

                // parallel argmax
                {
                    let (r0, r1) = split(vocab, nt, tid);
                    let mut bv = f32::NEG_INFINITY;
                    let mut bi = r0;
                    for (i, &v) in logits[r0..r1].iter().enumerate() {
                        if v > bv {
                            bv = v;
                            bi = r0 + i;
                        }
                    }
                    argv[tid] = bv;
                    argi[tid] = bi;
                }
                bar.wait();
            });
        }
    });
}

/// Feed one token at absolute position `pos`; returns argmax token id.
pub fn decode_argmax(raw: impl std::ops::Deref<Target = Vec<u8>>, token: i64, pos: i64) -> i64 {
    let mut cell = cell().lock().unwrap();
    let Some(st) = cell.as_mut() else { return -1 };
    if pos as usize >= MAX_SEQ { return -2 }
    let nt = lockstep_threads();
    token_pass(st, &raw, token as usize, pos as usize, nt);
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for t in 0..nt {
        if st.arg_v[t] > bv {
            bv = st.arg_v[t];
            best = st.arg_i[t];
        }
    }
    best as i64
}

/// Parity variant: full logits.
pub fn decode_logits(raw: impl std::ops::Deref<Target = Vec<u8>>, token: i64, pos: i64) -> Vec<f64> {
    let mut cell = cell().lock().unwrap();
    let Some(st) = cell.as_mut() else { return vec![] };
    if pos as usize >= MAX_SEQ { return vec![] }
    let nt = lockstep_threads();
    token_pass(st, &raw, token as usize, pos as usize, nt);
    st.logits.iter().map(|&v| v as f64).collect()
}
