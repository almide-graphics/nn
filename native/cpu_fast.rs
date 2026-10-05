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

// 2048 = chat-demo context. KV pool cost: 2048 × kv_hidden(1024) × 2 × 28
// layers × 4B ≈ 470 MB, allocated zeroed at load.
const MAX_SEQ: usize = 2048;
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
    // per-token rope table: head_dim/2 × (sin, cos)
    rope_tab: Vec<f32>,
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
    raw: &[u8],
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
    gammas_in: &[i64],
    weights_in: &[i64],
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
        rope_tab: vec![0.0; head_dim_u],
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

/// f32 dot, 8-wide FMA with 2 accumulators (attention score kernel).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn f32_dot_avx2(a: &[f32], b: &[f32], n: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut i = 0;
    while i + 16 <= n {
        let a0 = _mm256_loadu_ps(a.as_ptr().add(i));
        let b0 = _mm256_loadu_ps(b.as_ptr().add(i));
        let a1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
        let b1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
        acc0 = _mm256_fmadd_ps(a0, b0, acc0);
        acc1 = _mm256_fmadd_ps(a1, b1, acc1);
        i += 16;
    }
    let acc = _mm256_add_ps(acc0, acc1);
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let s4 = _mm_add_ps(hi, lo);
    let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    let s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 0b0000_0001));
    let mut r = _mm_cvtss_f32(s1);
    while i < n {
        r += a[i] * b[i];
        i += 1;
    }
    r
}

/// out += w * v, 8-wide (attention weighted-V accumulate).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn f32_axpy_avx2(out: &mut [f32], v: &[f32], w: f32, n: usize) {
    use std::arch::x86_64::*;
    let wv = _mm256_set1_ps(w);
    let mut i = 0;
    while i + 8 <= n {
        let o = _mm256_loadu_ps(out.as_ptr().add(i));
        let x = _mm256_loadu_ps(v.as_ptr().add(i));
        _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_fmadd_ps(wv, x, o));
        i += 8;
    }
    while i < n {
        out[i] += w * v[i];
        i += 1;
    }
}

/// Vectorized Q8 activation quantization for one 32-value block
/// (llama.cpp's quantize_row_q8_0 recipe): abs-max via AVX, round, pack.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn quant_block_avx2(vals: &[f32], out_q: &mut [i8]) -> f32 {
    use std::arch::x86_64::*;
    let v0 = _mm256_loadu_ps(vals.as_ptr());
    let v1 = _mm256_loadu_ps(vals.as_ptr().add(8));
    let v2 = _mm256_loadu_ps(vals.as_ptr().add(16));
    let v3 = _mm256_loadu_ps(vals.as_ptr().add(24));
    let signbit = _mm256_set1_ps(-0.0);
    let mut maxabs = _mm256_andnot_ps(signbit, v0);
    maxabs = _mm256_max_ps(maxabs, _mm256_andnot_ps(signbit, v1));
    maxabs = _mm256_max_ps(maxabs, _mm256_andnot_ps(signbit, v2));
    maxabs = _mm256_max_ps(maxabs, _mm256_andnot_ps(signbit, v3));
    let m4 = _mm_max_ps(_mm256_extractf128_ps(maxabs, 1), _mm256_castps256_ps128(maxabs));
    let m2 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
    let m1 = _mm_max_ss(m2, _mm_shuffle_ps(m2, m2, 0b0000_0001));
    let amax = _mm_cvtss_f32(m1);
    let d = amax / 127.0;
    let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
    let mul = _mm256_set1_ps(inv);
    let i0 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v0, mul), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    let i1 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v1, mul), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    let i2 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v2, mul), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    let i3 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v3, mul), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    let p01 = _mm256_packs_epi32(i0, i1); // 16 × i16, lane-interleaved
    let p23 = _mm256_packs_epi32(i2, i3);
    let packed = _mm256_packs_epi16(p01, p23); // 32 × i8, lane-interleaved
    // undo the 128-bit lane interleave of packs: order [0,4,1,5,2,6,3,7] dwords
    let fixed = _mm256_permutevar8x32_epi32(packed, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
    _mm256_storeu_si256(out_q.as_mut_ptr() as *mut __m256i, fixed);
    d
}

/// 8-lane exp approximation (Cephes polynomial, ~1 ulp on the silu/softmax
/// range) — scalar libm expf measured ~145k calls/token across silu+softmax.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn exp8_avx2(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let max_in = _mm256_set1_ps(88.376_26);
    let min_in = _mm256_set1_ps(-87.336_54);
    let x = _mm256_min_ps(_mm256_max_ps(x, min_in), max_in);
    let log2e = _mm256_set1_ps(std::f32::consts::LOG2_E);
    let n = _mm256_round_ps(_mm256_mul_ps(x, log2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    let ln2_hi = _mm256_set1_ps(0.693_359_4);
    let ln2_lo = _mm256_set1_ps(-2.121_944_4e-4);
    let mut r = _mm256_fnmadd_ps(n, ln2_hi, x);
    r = _mm256_fnmadd_ps(n, ln2_lo, r);
    // exp(r) on [-ln2/2, ln2/2], degree-5 polynomial
    let c5 = _mm256_set1_ps(1.987_569_1e-4);
    let c4 = _mm256_set1_ps(1.398_199_9e-3);
    let c3 = _mm256_set1_ps(8.333_452e-3);
    let c2 = _mm256_set1_ps(4.166_579_5e-2);
    let c1 = _mm256_set1_ps(1.666_666_6e-1);
    let c0 = _mm256_set1_ps(5.0e-1);
    let mut p = c5;
    p = _mm256_fmadd_ps(p, r, c4);
    p = _mm256_fmadd_ps(p, r, c3);
    p = _mm256_fmadd_ps(p, r, c2);
    p = _mm256_fmadd_ps(p, r, c1);
    p = _mm256_fmadd_ps(p, r, c0);
    let r2 = _mm256_mul_ps(r, r);
    let mut e = _mm256_fmadd_ps(p, r2, r);
    e = _mm256_add_ps(e, _mm256_set1_ps(1.0));
    // scale by 2^n
    let ni = _mm256_cvtps_epi32(n);
    let pow2n = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(ni, _mm256_set1_epi32(127)), 23));
    _mm256_mul_ps(e, pow2n)
}

/// silu(gate)*up over n values, 8-wide.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn silu_mul_avx2(gate: &[f32], up: &[f32], out: &mut [f32], n: usize) {
    use std::arch::x86_64::*;
    let one = _mm256_set1_ps(1.0);
    let mut i = 0;
    while i + 8 <= n {
        let g = _mm256_loadu_ps(gate.as_ptr().add(i));
        let u = _mm256_loadu_ps(up.as_ptr().add(i));
        let e = exp8_avx2(_mm256_sub_ps(_mm256_setzero_ps(), g));
        let s = _mm256_div_ps(g, _mm256_add_ps(one, e));
        _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_mul_ps(s, u));
        i += 8;
    }
    while i < n {
        let g = gate[i];
        out[i] = g / (1.0 + (-g).exp()) * up[i];
        i += 1;
    }
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
    let rope_p = Shared(st.rope_tab.as_mut_ptr());
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
                let rope_tab = std::slice::from_raw_parts_mut(rope_p.ptr(), dh);
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
                // rope angle table: identical for every head and layer at
                // this position — was 24 heads × 64 sincos per LAYER.
                {
                    let half = dh / 2;
                    let (j0, j1) = split(half, nt, tid);
                    for j in j0..j1 {
                        let (sn, cs) = (pos as f32 * inv_freq[j]).sin_cos();
                        rope_tab[j * 2] = sn;
                        rope_tab[j * 2 + 1] = cs;
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
                        let base = b * Q8B;
                        let mut tmp = [0.0f32; Q8B];
                        for k in 0..Q8B {
                            tmp[k] = vals(base + k);
                        }
                        #[cfg(target_arch = "x86_64")]
                        {
                            if avx2() {
                                xqs[b] = unsafe { quant_block_avx2(&tmp, &mut xq[base..base + Q8B]) };
                                continue;
                            }
                        }
                        #[allow(unreachable_code)]
                        {
                            let mut amax = 0.0f32;
                            for k in 0..Q8B {
                                amax = amax.max(tmp[k].abs());
                            }
                            let d = amax / 127.0;
                            let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
                            xqs[b] = d;
                            for k in 0..Q8B {
                                xq[base + k] = (tmp[k] * inv).round().clamp(-127.0, 127.0) as i8;
                            }
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
                        let mut tmp = [0.0f32; Q8B];
                        for k in 0..Q8B {
                            tmp[k] = src[base + k] * inv * gammas[g + base + k];
                        }
                        #[cfg(target_arch = "x86_64")]
                        {
                            if avx2() {
                                xqs[b] = unsafe { quant_block_avx2(&tmp, &mut xq[base..base + Q8B]) };
                                continue;
                            }
                        }
                        #[allow(unreachable_code)]
                        {
                            let mut amax = 0.0f32;
                            for k in 0..Q8B {
                                amax = amax.max(tmp[k].abs());
                            }
                            let d = amax / 127.0;
                            let qi = if d > 0.0 { 1.0 / d } else { 0.0 };
                            xqs[b] = d;
                            for k in 0..Q8B {
                                xq[base + k] = (tmp[k] * qi).round().clamp(-127.0, 127.0) as i8;
                            }
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
                                let sn = rope_tab[j * 2];
                                let cs = rope_tab[j * 2 + 1];
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
                            let qrow = &qkv[qb..qb + dh];
                            for j in 0..seq {
                                let kb = j * kv_hidden + kvb;
                                let sv;
                                #[cfg(target_arch = "x86_64")]
                                {
                                    sv = if avx2() {
                                        unsafe { f32_dot_avx2(qrow, &kcache[kb..kb + dh], dh) * scale }
                                    } else {
                                        let mut sdot = 0.0f32;
                                        for k in 0..dh {
                                            sdot += qrow[k] * kcache[kb + k];
                                        }
                                        sdot * scale
                                    };
                                }
                                #[cfg(not(target_arch = "x86_64"))]
                                {
                                    let mut sdot = 0.0f32;
                                    for k in 0..dh {
                                        sdot += qrow[k] * kcache[kb + k];
                                    }
                                    sv = sdot * scale;
                                }
                                scores[j] = sv;
                                m = m.max(sv);
                            }
                            let mut sum = 0.0f32;
                            #[cfg(target_arch = "x86_64")]
                            {
                                if avx2() {
                                    unsafe {
                                        use std::arch::x86_64::*;
                                        let mv = _mm256_set1_ps(m);
                                        let mut sv = _mm256_setzero_ps();
                                        let mut j = 0;
                                        while j + 8 <= seq {
                                            let s8 = _mm256_loadu_ps(scores.as_ptr().add(j));
                                            let e = exp8_avx2(_mm256_sub_ps(s8, mv));
                                            _mm256_storeu_ps(scores.as_mut_ptr().add(j), e);
                                            sv = _mm256_add_ps(sv, e);
                                            j += 8;
                                        }
                                        let hi = _mm256_extractf128_ps(sv, 1);
                                        let lo = _mm256_castps256_ps128(sv);
                                        let s4 = _mm_add_ps(hi, lo);
                                        let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
                                        sum = _mm_cvtss_f32(_mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 0b0000_0001)));
                                        while j < seq {
                                            let e = (scores[j] - m).exp();
                                            scores[j] = e;
                                            sum += e;
                                            j += 1;
                                        }
                                    }
                                } else {
                                    for sj in scores.iter_mut().take(seq) {
                                        *sj = (*sj - m).exp();
                                        sum += *sj;
                                    }
                                }
                            }
                            #[cfg(not(target_arch = "x86_64"))]
                            for sj in scores.iter_mut().take(seq) {
                                *sj = (*sj - m).exp();
                                sum += *sj;
                            }
                            let invs = 1.0 / sum;
                            let orow = &mut attn_out[qb..qb + dh];
                            for o in orow.iter_mut() {
                                *o = 0.0;
                            }
                            for j in 0..seq {
                                let w = scores[j] * invs;
                                let vb = j * kv_hidden + kvb;
                                #[cfg(target_arch = "x86_64")]
                                {
                                    if avx2() {
                                        unsafe { f32_axpy_avx2(orow, &vcache[vb..vb + dh], w, dh) };
                                        continue;
                                    }
                                }
                                #[allow(unreachable_code)]
                                for k in 0..dh {
                                    orow[k] += w * vcache[vb + k];
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
                    {
                        // silu(gate)*up into `gate` in place (8-wide exp),
                        // then the SIMD block quantizer
                        let (i0, i1) = split(ffn, nt, tid);
                        #[cfg(target_arch = "x86_64")]
                        {
                            if avx2() {
                                let g_in: &[f32] = std::slice::from_raw_parts(gate.as_ptr(), ffn);
                                unsafe {
                                    silu_mul_avx2(&g_in[i0..i1], &up[i0..i1],
                                        &mut gate[i0..i1], i1 - i0)
                                };
                            } else {
                                for i in i0..i1 {
                                    let g = gate[i];
                                    gate[i] = g / (1.0 + (-g).exp()) * up[i];
                                }
                            }
                        }
                        #[cfg(not(target_arch = "x86_64"))]
                        for i in i0..i1 {
                            let g = gate[i];
                            gate[i] = g / (1.0 + (-g).exp()) * up[i];
                        }
                    }
                    bar.wait();
                    quantize(&|i| gate[i], ffn, xqs, xq, bar);
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

// Prefill is a matrix workload, not a sequence of independent GEMVs. Keep
// the decode executor above as the numerical oracle and process a bounded
// number of prompt rows together, layer by layer. No float weight copy or
// repacking is needed: the Q8_0 weights are reused directly from `raw`.
const PREFILL_CHUNK: usize = 64;
const PREFILL_TILE: usize = 4;

/// One Q8_0 weight row against four token rows. Each token retains exactly
/// q8_row_dot_avx2's even/odd block accumulators and horizontal reduction.
/// Weight bytes and F16C scale conversion are shared across the four rows.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn q8_prefill_tile_avx2(
    xs: &[f32], xq: &[i8], row: &[u8], scale_stride: usize, quant_stride: usize,
) -> [f32; PREFILL_TILE] {
    use std::arch::x86_64::*;
    let mut even = [_mm256_setzero_ps(); PREFILL_TILE];
    let mut odd = [_mm256_setzero_ps(); PREFILL_TILE];
    let ones = _mm256_set1_epi16(1);
    for b in 0..row.len() / Q8BB {
        let blk = row.as_ptr().add(b * Q8BB);
        let d = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(
            u16::from_le_bytes([*blk, *blk.add(1)]) as i32,
        )));
        let w = _mm256_loadu_si256(blk.add(2) as *const __m256i);
        for t in 0..PREFILL_TILE {
            let x = _mm256_loadu_si256(xq.as_ptr().add(t * quant_stride + b * Q8B) as *const __m256i);
            let ax = _mm256_sign_epi8(x, x);
            let sw = _mm256_sign_epi8(w, x);
            let p = _mm256_madd_epi16(_mm256_maddubs_epi16(ax, sw), ones);
            let scale = _mm256_set1_ps(d * *xs.get_unchecked(t * scale_stride + b));
            if b & 1 == 0 {
                even[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), scale, even[t]);
            } else {
                odd[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), scale, odd[t]);
            }
        }
    }
    let mut out = [0.0; PREFILL_TILE];
    for t in 0..PREFILL_TILE {
        let acc = _mm256_add_ps(even[t], odd[t]);
        let s4 = _mm_add_ps(_mm256_extractf128_ps(acc, 1), _mm256_castps256_ps128(acc));
        let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
        out[t] = _mm_cvtss_f32(_mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 0b0000_0001)));
    }
    out
}

/// Parallelize output-column tiles. A task owns these columns for every
/// token row; therefore all raw output writes are disjoint. Keeping a
/// small weight-row tile hot in cache amortizes DRAM reads over the batch.
#[allow(clippy::too_many_arguments)]
fn prefill_gemm(
    raw: &[u8], off: usize, rows: usize, n_in: usize,
    xs: &[f32], xq: &[i8], quant_stride: usize, n_tokens: usize,
    out: &mut [f32], out_stride: usize, out_column: usize, add_self: bool,
) {
    use rayon::prelude::*;
    let row_bytes = n_in / Q8B * Q8BB;
    let scale_stride = quant_stride / Q8B;
    let op = Shared(out.as_mut_ptr());
    // Each tile is only 16--64 KiB for the model's projection widths.
    (0..rows.div_ceil(16)).into_par_iter().for_each(|tile| {
        let first = tile * 16;
        let end = (first + 16).min(rows);
        for t in (0..n_tokens).step_by(PREFILL_TILE) {
            for r in first..end {
                let row = &raw[off + r * row_bytes..off + (r + 1) * row_bytes];
                #[cfg(target_arch = "x86_64")]
                if avx2() && t + PREFILL_TILE <= n_tokens {
                    let dots = unsafe { q8_prefill_tile_avx2(
                        &xs[t * scale_stride..], &xq[t * quant_stride..], row,
                        scale_stride, quant_stride,
                    ) };
                    for (j, dot) in dots.into_iter().enumerate() {
                        unsafe {
                            let dst = op.ptr().add((t + j) * out_stride + out_column + r);
                            *dst = if add_self { *dst + dot } else { dot };
                        }
                    }
                    continue;
                }
                for j in t..(t + PREFILL_TILE).min(n_tokens) {
                    let dot = q8_row_dot(&xs[j * scale_stride..], &xq[j * quant_stride..], row);
                    unsafe {
                        let dst = op.ptr().add(j * out_stride + out_column + r);
                        *dst = if add_self { *dst + dot } else { dot };
                    }
                }
            }
        }
    });
}

fn prefill_quantize(vals: &[f32], scales: &mut [f32], quants: &mut [i8]) {
    for (b, block) in vals.chunks_exact(Q8B).enumerate() {
        let dst = &mut quants[b * Q8B..(b + 1) * Q8B];
        #[cfg(target_arch = "x86_64")]
        if avx2() {
            scales[b] = unsafe { quant_block_avx2(block, dst) };
            continue;
        }
        let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
        scales[b] = d;
        for k in 0..Q8B {
            dst[k] = (block[k] * inv).round().clamp(-127.0, 127.0) as i8;
        }
    }
}

// The decode RMS uses one sum per lockstep worker, followed by a serial
// reduction in worker order. Preserve that order even though prefill
// assigns entire token rows to workers rather than splitting every row.
fn prefill_rms_inv(src: &[f32], eps: f32, nt: usize) -> f32 {
    let mut total = 0.0f32;
    for tid in 0..nt {
        let (s, e) = split(src.len(), nt, tid);
        let mut local = 0.0f32;
        for v in &src[s..e] {
            local += v * v;
        }
        total += local;
    }
    1.0 / (total / src.len() as f32 + eps).sqrt()
}

#[allow(clippy::too_many_arguments)]
fn prefill_rms_quant(
    h: &[f32], hidden: usize, gamma: &[f32], eps: f32, nt: usize,
    xs: &mut [f32], xq: &mut [i8], quant_stride: usize,
) {
    use rayon::prelude::*;
    xs.par_chunks_mut(quant_stride / Q8B)
        .zip(xq.par_chunks_mut(quant_stride))
        .zip(h.par_chunks(hidden))
        .for_each(|((scales, quants), row)| {
            let inv = prefill_rms_inv(row, eps, nt);
            for b in 0..hidden / Q8B {
                let mut vals = [0.0; Q8B];
                for k in 0..Q8B {
                    let i = b * Q8B + k;
                    vals[k] = row[i] * inv * gamma[i];
                }
                prefill_quantize(&vals, &mut scales[b..b + 1], &mut quants[b * Q8B..(b + 1) * Q8B]);
            }
        });
}

/// Exactly the decode softmax order, including its SIMD exp and tail.
fn prefill_softmax(scores: &mut [f32], m: f32) -> f32 {
    let seq = scores.len();
    let mut sum = 0.0f32;
    #[cfg(target_arch = "x86_64")]
    if avx2() {
        unsafe {
            use std::arch::x86_64::*;
            let mv = _mm256_set1_ps(m);
            let mut sv = _mm256_setzero_ps();
            let mut j = 0;
            while j + 8 <= seq {
                let s8 = _mm256_loadu_ps(scores.as_ptr().add(j));
                let e = exp8_avx2(_mm256_sub_ps(s8, mv));
                _mm256_storeu_ps(scores.as_mut_ptr().add(j), e);
                sv = _mm256_add_ps(sv, e);
                j += 8;
            }
            let s4 = _mm_add_ps(_mm256_extractf128_ps(sv, 1), _mm256_castps256_ps128(sv));
            let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
            sum = _mm_cvtss_f32(_mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 0b0000_0001)));
            while j < seq {
                let e = (scores[j] - m).exp();
                scores[j] = e;
                sum += e;
                j += 1;
            }
        }
        return 1.0 / sum;
    }
    for score in scores {
        *score = (*score - m).exp();
        sum += *score;
    }
    1.0 / sum
}

fn prefill_pass(st: &mut State, raw: &[u8], tokens: &[i64], start: usize, nt: usize) {
    use rayon::prelude::*;
    let cfg = &st.cfg;
    let hidden = cfg.hidden;
    let q_hidden = cfg.q_hidden;
    let kv_hidden = cfg.kv_hidden;
    let qkv_width = q_hidden + 2 * kv_hidden;
    let ffn = cfg.ffn;
    let dh = cfg.head_dim;
    let stride = hidden.max(ffn).max(q_hidden);
    // Bounded scratch: approximately 4 MiB at 64 tokens for Qwen3-0.6B.
    let capacity = tokens.len().min(PREFILL_CHUNK);
    let mut h = vec![0.0f32; capacity * hidden];
    let mut qkv = vec![0.0f32; capacity * qkv_width];
    let mut attn = vec![0.0f32; capacity * q_hidden];
    let mut gate = vec![0.0f32; capacity * ffn];
    let mut up = vec![0.0f32; capacity * ffn];
    let mut xs = vec![0.0f32; capacity * stride / Q8B];
    let mut xq = vec![0i8; capacity * stride];
    let mut rope = vec![0.0f32; capacity * dh];

    for (chunk_index, chunk) in tokens.chunks(PREFILL_CHUNK).enumerate() {
        let n = chunk.len();
        let chunk_start = start + chunk_index * PREFILL_CHUNK;
        let h = &mut h[..n * hidden];
        let qkv = &mut qkv[..n * qkv_width];
        let attn = &mut attn[..n * q_hidden];
        let gate = &mut gate[..n * ffn];
        let up = &mut up[..n * ffn];
        let xs = &mut xs[..n * stride / Q8B];
        let xq = &mut xq[..n * stride];
        let rope = &mut rope[..n * dh];
        h.par_chunks_mut(hidden).enumerate().for_each(|(t, out)| {
            let rb = hidden / Q8B * Q8BB;
            let row = &raw[cfg.emb_off + chunk[t] as usize * rb..cfg.emb_off + (chunk[t] as usize + 1) * rb];
            for (b, block) in row.chunks_exact(Q8BB).enumerate() {
                let d = fp16_inline(u16::from_le_bytes([block[0], block[1]]));
                for k in 0..Q8B {
                    out[b * Q8B + k] = d * (block[2 + k] as i8) as f32;
                }
            }
        });
        rope.par_chunks_mut(dh).enumerate().for_each(|(t, row)| {
            for j in 0..dh / 2 {
                let (sn, cs) = ((chunk_start + t) as f32 * st.inv_freq[j]).sin_cos();
                row[j * 2] = sn;
                row[j * 2 + 1] = cs;
            }
        });
        for l in 0..cfg.n_layers {
            let go = &st.layer_goffs[l];
            let wo = &st.layer_woffs[l];
            prefill_rms_quant(h, hidden, &st.gammas[go[0]..go[0] + hidden], cfg.eps, nt, xs, xq, stride);
            prefill_gemm(raw, wo[0], q_hidden, hidden, xs, xq, stride, n, qkv, qkv_width, 0, false);
            prefill_gemm(raw, wo[1], kv_hidden, hidden, xs, xq, stride, n, qkv, qkv_width, q_hidden, false);
            prefill_gemm(raw, wo[2], kv_hidden, hidden, xs, xq, stride, n, qkv, qkv_width, q_hidden + kv_hidden, false);
            qkv.par_chunks_mut(qkv_width).enumerate().for_each(|(t, row)| {
                for hh in 0..cfg.n_heads + cfg.n_kv {
                    let (base, g) = if hh < cfg.n_heads {
                        (hh * dh, go[1])
                    } else {
                        (q_hidden + (hh - cfg.n_heads) * dh, go[2])
                    };
                    let mut ss = 0.0f32;
                    for k in 0..dh { ss += row[base + k] * row[base + k]; }
                    let inv = 1.0 / (ss / dh as f32 + cfg.eps).sqrt();
                    for k in 0..dh { row[base + k] *= inv * st.gammas[g + k]; }
                    for j in 0..dh / 2 {
                        let sn = rope[t * dh + j * 2];
                        let cs = rope[t * dh + j * 2 + 1];
                        let x0 = row[base + j];
                        let x1 = row[base + dh / 2 + j];
                        row[base + j] = x0 * cs - x1 * sn;
                        row[base + dh / 2 + j] = x0 * sn + x1 * cs;
                    }
                }
            });
            // Publish all K/V rows for this layer before attention. The
            // query's absolute position still bounds the visible prefix.
            st.k_cache[l][chunk_start * kv_hidden..(chunk_start + n) * kv_hidden]
                .par_chunks_mut(kv_hidden).zip(qkv.par_chunks(qkv_width))
                .for_each(|(dst, row)| dst.copy_from_slice(&row[q_hidden..q_hidden + kv_hidden]));
            st.v_cache[l][chunk_start * kv_hidden..(chunk_start + n) * kv_hidden]
                .par_chunks_mut(kv_hidden).zip(qkv.par_chunks(qkv_width))
                .for_each(|(dst, row)| dst.copy_from_slice(&row[q_hidden + kv_hidden..]));
            let kc = &st.k_cache[l];
            let vc = &st.v_cache[l];
            attn.par_chunks_mut(dh).enumerate().for_each(|(index, out)| {
                let t = index / cfg.n_heads;
                let hh = index % cfg.n_heads;
                let seq = chunk_start + t + 1;
                let kvb = (hh / (cfg.n_heads / cfg.n_kv)) * dh;
                let qr = &qkv[t * qkv_width + hh * dh..t * qkv_width + (hh + 1) * dh];
                let scale = 1.0 / (dh as f32).sqrt();
                let mut scores = [0.0f32; MAX_SEQ];
                let mut m = f32::NEG_INFINITY;
                for j in 0..seq {
                    let kr = &kc[j * kv_hidden + kvb..j * kv_hidden + kvb + dh];
                    let dot;
                    #[cfg(target_arch = "x86_64")]
                    { dot = if avx2() { unsafe { f32_dot_avx2(qr, kr, dh) } } else {
                        qr.iter().zip(kr).fold(0.0f32, |sum, (&q, &k)| sum + q * k)
                    }; }
                    #[cfg(not(target_arch = "x86_64"))]
                    { dot = qr.iter().zip(kr).fold(0.0f32, |sum, (&q, &k)| sum + q * k); }
                    scores[j] = dot * scale;
                    m = m.max(scores[j]);
                }
                let inv = prefill_softmax(&mut scores[..seq], m);
                out.fill(0.0);
                for j in 0..seq {
                    let w = scores[j] * inv;
                    let vr = &vc[j * kv_hidden + kvb..j * kv_hidden + kvb + dh];
                    #[cfg(target_arch = "x86_64")]
                    if avx2() {
                        unsafe { f32_axpy_avx2(out, vr, w, dh) };
                        continue;
                    }
                    for k in 0..dh { out[k] += w * vr[k]; }
                }
            });
            xs.par_chunks_mut(stride / Q8B).zip(xq.par_chunks_mut(stride))
                .zip(attn.par_chunks(q_hidden)).for_each(|((s, q), v)| prefill_quantize(v, s, q));
            prefill_gemm(raw, wo[3], hidden, q_hidden, xs, xq, stride, n, h, hidden, 0, true);
            prefill_rms_quant(h, hidden, &st.gammas[go[3]..go[3] + hidden], cfg.eps, nt, xs, xq, stride);
            prefill_gemm(raw, wo[4], ffn, hidden, xs, xq, stride, n, gate, ffn, 0, false);
            prefill_gemm(raw, wo[5], ffn, hidden, xs, xq, stride, n, up, ffn, 0, false);
            // As in token_pass, keep SIMD silu groups inside each decode
            // worker's range (important for the scalar tail on odd shapes).
            gate.par_chunks_mut(ffn).zip(up.par_chunks(ffn)).for_each(|(g, u)| {
                for tid in 0..nt {
                    let (s, e) = split(ffn, nt, tid);
                    #[cfg(target_arch = "x86_64")]
                    if avx2() {
                        // Copy one SIMD-width input before writing it, so
                        // the source and destination references never alias.
                        let mut i = s;
                        while i + 8 <= e {
                            let mut gi = [0.0f32; 8];
                            gi.copy_from_slice(&g[i..i + 8]);
                            unsafe { silu_mul_avx2(&gi, &u[i..i + 8], &mut g[i..i + 8], 8) };
                            i += 8;
                        }
                        while i < e { g[i] = g[i] / (1.0 + (-g[i]).exp()) * u[i]; i += 1; }
                        continue;
                    }
                    for i in s..e { g[i] = g[i] / (1.0 + (-g[i]).exp()) * u[i]; }
                }
            });
            xs.par_chunks_mut(stride / Q8B).zip(xq.par_chunks_mut(stride))
                .zip(gate.par_chunks(ffn)).for_each(|((s, q), v)| prefill_quantize(v, s, q));
            prefill_gemm(raw, wo[6], hidden, ffn, xs, xq, stride, n, h, hidden, 0, true);
        }
        // Only this last row is needed by the decode state; logits for
        // earlier prompt rows would never be observed by generation.
        st.h.copy_from_slice(&h[(n - 1) * hidden..n * hidden]);
    }
    let inv = prefill_rms_inv(&st.h, cfg.eps, nt);
    for i in 0..hidden {
        st.xn[i] = st.h[i] * inv * st.gammas[cfg.out_norm_g + i];
    }
    prefill_quantize(&st.xn[..hidden], &mut st.xq_s, &mut st.xq);
    let rb = hidden / Q8B * Q8BB;
    st.logits.par_iter_mut().enumerate().for_each(|(r, out)| {
        *out = q8_row_dot(&st.xq_s, &st.xq, &raw[cfg.emb_off + r * rb..cfg.emb_off + (r + 1) * rb]);
    });
}

fn prefill_pool() -> &'static rayon::ThreadPool {
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| rayon::ThreadPoolBuilder::new().num_threads(lockstep_threads()).build().unwrap())
}

/// Feed a nonempty token slice starting at an absolute position. Previous
/// positions must already have valid KV state, as for decode_argmax.
/// Returns -1 without a model, -2 for an invalid context range, and -3 for
/// an empty prompt or invalid token ID. Invalid calls do not modify state.
pub fn prefill_argmax(raw: &[u8], tokens: &[i64], start_pos: i64) -> i64 {
    let mut guard = cell().lock().unwrap();
    let Some(st) = guard.as_mut() else { return -1 };
    if start_pos < 0 || (start_pos as usize).checked_add(tokens.len()).map_or(true, |end| end > MAX_SEQ) { return -2 }
    if tokens.is_empty() || tokens.iter().any(|&t| t < 0 || t as usize >= st.cfg.vocab) { return -3 }
    let nt = lockstep_threads();
    prefill_pool().install(|| prefill_pass(st, raw, tokens, start_pos as usize, nt));
    let mut best = 0usize;
    let mut value = f32::NEG_INFINITY;
    for (i, &v) in st.logits.iter().enumerate() {
        if v > value { value = v; best = i; }
    }
    best as i64
}

/// Prefill parity hook: same execution as prefill_argmax, returning the
/// final prompt row's full logits. Invalid calls return an empty vector.
pub fn prefill_logits(raw: &[u8], tokens: &[i64], start_pos: i64) -> Vec<f64> {
    let mut guard = cell().lock().unwrap();
    let Some(st) = guard.as_mut() else { return vec![] };
    if start_pos < 0 || (start_pos as usize).checked_add(tokens.len()).map_or(true, |end| end > MAX_SEQ)
        || tokens.is_empty() || tokens.iter().any(|&t| t < 0 || t as usize >= st.cfg.vocab) { return vec![] }
    let nt = lockstep_threads();
    prefill_pool().install(|| prefill_pass(st, raw, tokens, start_pos as usize, nt));
    st.logits.iter().map(|&v| v as f64).collect()
}

/// Feed one token at absolute position `pos`; returns argmax token id.
pub fn decode_argmax(raw: &[u8], token: i64, pos: i64) -> i64 {
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

/// Feed one token; sample the next from temperature + top-p (nucleus).
/// temp <= 0 degrades to argmax. Deterministic for a given (seed, pos) —
/// the caller passes one seed per conversation and gets a reproducible
/// stream. Sampling stays native: shipping 152k logits across the FFI and
/// sorting them in Almide per token would dominate the decode step.
pub fn decode_sample(
    raw: &[u8],
    token: i64,
    pos: i64,
    temp: f64,
    top_p: f64,
    seed: i64,
) -> i64 {
    let mut cell = cell().lock().unwrap();
    let Some(st) = cell.as_mut() else { return -1 };
    if pos as usize >= MAX_SEQ {
        return -2;
    }
    let nt = lockstep_threads();
    token_pass(st, &raw, token as usize, pos as usize, nt);
    if temp <= 0.0 {
        let mut best = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for (i, &v) in st.logits.iter().enumerate() {
            if v > bv {
                bv = v;
                best = i;
            }
        }
        return best as i64;
    }
    // top-K preselect (K=256 covers any practical nucleus), then top-p cut
    const K: usize = 256;
    let mut pairs: Vec<(f32, u32)> = st.logits.iter().enumerate().map(|(i, &v)| (v, i as u32)).collect();
    let k = K.min(pairs.len());
    pairs.select_nth_unstable_by(k - 1, |a, b| b.0.partial_cmp(&a.0).unwrap());
    pairs.truncate(k);
    pairs.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let inv_t = 1.0 / temp;
    let m = pairs[0].0 as f64;
    let ps: Vec<f64> = pairs.iter().map(|p| ((p.0 as f64 - m) * inv_t).exp()).collect();
    let sum: f64 = ps.iter().sum();
    let cut = top_p.clamp(0.0, 1.0) * sum;
    let mut n_keep = ps.len();
    let mut acc = 0.0;
    for (i, &p) in ps.iter().enumerate() {
        acc += p;
        if acc >= cut {
            n_keep = i + 1;
            break;
        }
    }
    // xorshift64* seeded by (seed, pos): one user seed → reproducible stream
    let mut s = (seed as u64) ^ 0x9E37_79B9_7F4A_7C15 ^ ((pos as u64).wrapping_mul(0xA24B_AED4_963E_E407));
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    let total: f64 = ps[..n_keep].iter().sum();
    let r = (s >> 11) as f64 / (1u64 << 53) as f64 * total;
    let mut acc2 = 0.0;
    for i in 0..n_keep {
        acc2 += ps[i];
        if r <= acc2 {
            return pairs[i].1 as i64;
        }
    }
    pairs[n_keep - 1].1 as i64
}

/// Parity variant: full logits.
pub fn decode_logits(raw: &[u8], token: i64, pos: i64) -> Vec<f64> {
    let mut cell = cell().lock().unwrap();
    let Some(st) = cell.as_mut() else { return vec![] };
    if pos as usize >= MAX_SEQ { return vec![] }
    let nt = lockstep_threads();
    token_pass(st, &raw, token as usize, pos as usize, nt);
    st.logits.iter().map(|&v| v as f64).collect()
}

#[cfg(test)]
mod prefill_tests {
    use super::*;

    fn weights(rows: usize, blocks: usize) -> Vec<u8> {
        let mut raw = vec![0u8; rows * blocks * Q8BB];
        for r in 0..rows {
            for b in 0..blocks {
                let off = (r * blocks + b) * Q8BB;
                let half = [0x2800u16, 0x3400, 0x3800, 0x3c00][(r + b) % 4].to_le_bytes();
                raw[off..off + 2].copy_from_slice(&half);
                for k in 0..Q8B {
                    raw[off + 2 + k] = (((r * 17 + b * 31 + k * 7) % 255) as i16 - 127) as i8 as u8;
                }
            }
        }
        raw
    }

    fn activations(tokens: usize, stride: usize) -> (Vec<f32>, Vec<i8>) {
        let mut scales = vec![0.0f32; tokens * stride / Q8B];
        let mut quants = vec![0i8; tokens * stride];
        for (i, scale) in scales.iter_mut().enumerate() { *scale = (1 + i % 13) as f32 / 127.0; }
        for (i, quant) in quants.iter_mut().enumerate() { *quant = ((i * 37 % 255) as i16 - 127) as i8; }
        (scales, quants)
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn prefill_tile_preserves_decode_dot_bits() {
        if !avx2() { return; }
        for blocks in [1, 2, 3, 8, 24, 48, 96] {
            let stride = (blocks + 3) * Q8B;
            let raw = weights(1, blocks);
            let (scales, quants) = activations(PREFILL_TILE, stride);
            let got = unsafe { q8_prefill_tile_avx2(&scales, &quants, &raw, stride / Q8B, stride) };
            for t in 0..PREFILL_TILE {
                let expected = q8_row_dot(&scales[t * stride / Q8B..], &quants[t * stride..], &raw);
                assert_eq!(got[t].to_bits(), expected.to_bits(), "blocks={blocks}, token={t}");
            }
        }
    }

    #[test]
    fn prefill_gemm_covers_tiles_tails_strides_and_residuals() {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        pool.install(|| {
            for tokens in [1, 3, 4, 5, 7, 64] {
                let rows = 19;
                let blocks = 8;
                let stride = 12 * Q8B;
                let out_stride = rows + 9;
                let column = 3;
                let raw = weights(rows, blocks);
                let (scales, quants) = activations(tokens, stride);
                for add in [false, true] {
                    let mut out = vec![0.375f32; tokens * out_stride];
                    prefill_gemm(&raw, 0, rows, blocks * Q8B, &scales, &quants, stride, tokens,
                        &mut out, out_stride, column, add);
                    for t in 0..tokens {
                        for r in 0..out_stride {
                            let expected = if r >= column && r < column + rows {
                                let wr = r - column;
                                let dot = q8_row_dot(&scales[t * stride / Q8B..], &quants[t * stride..],
                                    &raw[wr * blocks * Q8BB..(wr + 1) * blocks * Q8BB]);
                                if add { 0.375 + dot } else { dot }
                            } else { 0.375 };
                            assert_eq!(out[t * out_stride + r].to_bits(), expected.to_bits(),
                                "tokens={tokens}, row={r}, token={t}, residual={add}");
                        }
                    }
                }
            }
        });
    }
}
