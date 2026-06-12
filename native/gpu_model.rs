// nn native GPU model runtime (L3-2): the whole Qwen3 decode step on GPU.
//
// One token = one command buffer: write the (token, pos) uniform, encode
// ~480 pre-bound dispatches (28 layers × ~17 ops + embed/final/logits),
// submit, read back ONE i64 (the argmax) — or the full logits for parity.
//
// Residency: all Q8 weights live in a single repacked storage buffer
// (36-byte u32-aligned blocks), norm gammas in one f32 buffer, KV caches
// preallocated to MAX_SEQ, activations in small dedicated buffers. Every
// bind group and per-op uniform is built once at load.

use std::sync::{Mutex, OnceLock};

const MAX_SEQ: usize = 512;
const WG_BLOCKS: u32 = 9; // u32 words per repacked Q8 block

struct Op {
    pipeline: usize, // index into Model.pipelines
    bind: wgpu::BindGroup,
    groups: (u32, u32, u32),
}

struct Layer {
    ops: Vec<Op>,
    k_copy: (u64, u64), // (k src len, v src len) — encoder copies need sizes
}

struct Model {
    device: &'static wgpu::Device,
    queue: &'static wgpu::Queue,
    pipelines: Vec<wgpu::ComputePipeline>,
    step_uniform: wgpu::Buffer, // [token, pos, seq, pad]
    layers: Vec<Layer>,
    pre_ops: Vec<Op>,   // embed gather
    post_ops: Vec<Op>,  // final rms + logits gemv
    k_bufs: Vec<wgpu::Buffer>,
    v_bufs: Vec<wgpu::Buffer>,
    k_cur: wgpu::Buffer,
    v_cur: wgpu::Buffer,
    kv_row_bytes: u64,
    logits: wgpu::Buffer,
    staging: wgpu::Buffer,
    vocab: usize,
}

static MODEL: OnceLock<Mutex<Option<Model>>> = OnceLock::new();

fn model_cell() -> &'static Mutex<Option<Model>> {
    MODEL.get_or_init(|| Mutex::new(None))
}

const SHADERS: &str = include_str!("wgsl/qwen3.wgsl");

fn fp16_to_f32_m(raw: u16) -> f32 {
    super::gpu::fp16_to_f32_pub(raw)
}

struct Builder<'a> {
    device: &'a wgpu::Device,
    layouts: Vec<wgpu::BindGroupLayout>,
    pipelines: Vec<wgpu::ComputePipeline>,
}

fn be(i: u32, buf: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding: i, resource: buf.as_entire_binding() }
}

const P_GEMV: usize = 0;
const P_EMBED: usize = 1;
const P_RMS: usize = 2;
const P_ROPE: usize = 3;
const P_ATTN: usize = 4;
const P_SILU: usize = 5;
const P_ADD: usize = 6;
const P_RMS_IP: usize = 7;

impl<'a> Builder<'a> {
    fn new(device: &'a wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("qwen3-gpu"),
            source: wgpu::ShaderSource::Wgsl(SHADERS.into()),
        });
        let mk_layout = |entries: &[wgpu::BindGroupLayoutEntry]| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries })
        };
        let sto = |b: u32, ro: bool| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: ro },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let uni = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layouts = vec![
            mk_layout(&[sto(0, true), sto(1, true), sto(2, false), uni(3)]),          // gemv
            mk_layout(&[sto(0, true), sto(1, false), uni(2), uni(3)]),                // embed
            mk_layout(&[sto(0, true), sto(1, false), sto(2, true), uni(3)]),          // rms
            mk_layout(&[sto(0, false), uni(1), uni(2)]),                              // rope
            mk_layout(&[sto(0, true), sto(1, true), sto(2, true), sto(3, false), uni(4), uni(5)]), // attn
            mk_layout(&[sto(0, true), sto(1, true), sto(2, false), uni(3)]),          // silu
            mk_layout(&[sto(0, true), sto(1, true), sto(2, false), uni(3)]),          // add
            mk_layout(&[sto(0, false), sto(1, true), uni(2)]),                        // rms in-place
        ];
        let entries = ["gemv", "embed", "rmsnorm", "rope", "attn", "silu_mul", "add", "rmsnorm_inplace"];
        let pipelines = entries
            .iter()
            .zip(&layouts)
            .map(|(entry, layout)| {
                let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[layout],
                    push_constant_ranges: &[],
                });
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&pl),
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
            })
            .collect();
        Builder { device, layouts, pipelines }
    }

    fn uniform(&self, vals: &[u32]) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(vals),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }

    fn storage(&self, n_f32: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (n_f32 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }
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
    _eps: f64,
    emb_off: i64,
    out_norm_off: i64,
    gammas: &Vec<i64>,
    weights: &Vec<i64>,
) -> i64 {
    let Some((device, queue)) = super::gpu::device_queue() else { return 0 };

    let n_layers = n_layers as usize;
    let n_heads_u = n_heads as u32;
    let n_kv_u = n_kv_heads as u32;
    let head_dim_u = head_dim as u32;
    let hidden_u = hidden as usize;
    let q_hidden = (n_heads * head_dim) as usize;
    let kv_hidden = (n_kv_heads * head_dim) as usize;
    let ffn = ffn_hidden as usize;
    let vocab_u = vocab as usize;

    // ── Repack ALL Q8 weights into one buffer; remember word offsets ──
    let mut packed = Vec::<u32>::new();
    let mut woff = |packed: &mut Vec<u32>, byte_off: usize, rows: usize, cols: usize| -> u32 {
        let start = packed.len() as u32;
        let bpr = cols / 32;
        for r in 0..rows {
            let rb = byte_off + r * bpr * 34;
            for b in 0..bpr {
                let bb = rb + b * 34;
                let scale = fp16_to_f32_m(u16::from_le_bytes([raw[bb], raw[bb + 1]]));
                packed.push(scale.to_bits());
                for k in 0..8 {
                    let i = bb + 2 + k * 4;
                    packed.push(u32::from_le_bytes([raw[i], raw[i + 1], raw[i + 2], raw[i + 3]]));
                }
            }
        }
        start
    };

    let emb_w = woff(&mut packed, emb_off as usize, vocab_u, hidden_u);
    // per layer: [q, k, v, o, gate, up, down]
    let dims: [(usize, usize); 7] = [
        (q_hidden, hidden_u),
        (kv_hidden, hidden_u),
        (kv_hidden, hidden_u),
        (hidden_u, q_hidden),
        (ffn, hidden_u),
        (ffn, hidden_u),
        (hidden_u, ffn),
    ];
    let mut layer_woffs = Vec::with_capacity(n_layers);
    for l in 0..n_layers {
        let mut offs = [0u32; 7];
        for (t, &(r, c)) in dims.iter().enumerate() {
            offs[t] = woff(&mut packed, weights[l * 7 + t] as usize, r, c);
        }
        layer_woffs.push(offs);
    }

    // ── Gamma buffer: concatenated f32 gammas; remember word offsets ──
    let mut gbuf = Vec::<f32>::new();
    let mut goff = |gbuf: &mut Vec<f32>, byte_off: usize, n: usize| -> u32 {
        let start = gbuf.len() as u32;
        for i in 0..n {
            let b = byte_off + i * 4;
            gbuf.push(f32::from_le_bytes([raw[b], raw[b + 1], raw[b + 2], raw[b + 3]]));
        }
        start
    };
    let mut layer_goffs = Vec::with_capacity(n_layers);
    for l in 0..n_layers {
        let g = [
            goff(&mut gbuf, gammas[l * 4] as usize, hidden_u),          // attn_norm
            goff(&mut gbuf, gammas[l * 4 + 1] as usize, head_dim as usize), // q_norm
            goff(&mut gbuf, gammas[l * 4 + 2] as usize, head_dim as usize), // k_norm
            goff(&mut gbuf, gammas[l * 4 + 3] as usize, hidden_u),      // ffn_norm
        ];
        layer_goffs.push(g);
    }
    let out_norm_g = goff(&mut gbuf, out_norm_off as usize, hidden_u);

    use wgpu::util::DeviceExt;
    let wbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("weights-q8"),
        contents: bytemuck::cast_slice(&packed),
        usage: wgpu::BufferUsages::STORAGE,
    });
    drop(packed);
    let gammas_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gammas"),
        contents: bytemuck::cast_slice(&gbuf),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let b = Builder::new(device);

    // ── Activation buffers ──
    let h = b.storage(hidden_u);
    let xn = b.storage(hidden_u);
    let q = b.storage(q_hidden);
    let k_cur = b.storage(kv_hidden);
    let v_cur = b.storage(kv_hidden);
    let attn_out = b.storage(q_hidden);
    let proj = b.storage(hidden_u);
    let h2 = b.storage(hidden_u);
    let gate = b.storage(ffn);
    let up = b.storage(ffn);
    let gated = b.storage(ffn);
    let logits = b.storage(vocab_u);

    let step_uniform = b.uniform(&[0, 0, 1, 0]);

    let k_bufs: Vec<_> = (0..n_layers).map(|_| b.storage(MAX_SEQ * kv_hidden)).collect();
    let v_bufs: Vec<_> = (0..n_layers).map(|_| b.storage(MAX_SEQ * kv_hidden)).collect();

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (vocab_u * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // ── Bind groups ──
    let bind = |layout: usize, entries: &[wgpu::BindGroupEntry]| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &b.layouts[layout],
            entries,
        })
    };

    let gemv_groups = |rows: usize| -> (u32, u32, u32) {
        let r = rows as u32;
        (r.min(32768), r.div_ceil(32768), 1)
    };

    let gemv_op = |bld: &Builder, x: &wgpu::Buffer, y: &wgpu::Buffer, rows: usize, cols: usize, w_off: u32| -> Op {
        let u = bld.uniform(&[rows as u32, cols as u32, (cols / 32) as u32, w_off]);
        Op {
            pipeline: P_GEMV,
            bind: bind(P_GEMV, &[be(0, &wbuf), be(1, x), be(2, y), wgpu::BindGroupEntry { binding: 3, resource: u.as_entire_binding() }]),
            groups: gemv_groups(rows),
        }
    };
    let rms_op = |bld: &Builder, xi: &wgpu::Buffer, xo: &wgpu::Buffer, n: usize, g: u32, heads: u32, hd: u32| -> Op {
        let u = bld.uniform(&[n as u32, g, heads, hd]);
        Op {
            pipeline: P_RMS,
            bind: bind(P_RMS, &[be(0, xi), be(1, xo), be(2, &gammas_buf), wgpu::BindGroupEntry { binding: 3, resource: u.as_entire_binding() }]),
            groups: (if heads > 0 { heads } else { 1 }, 1, 1),
        }
    };
    let rms_ip_op = |bld: &Builder, x: &wgpu::Buffer, g: u32, heads: u32| -> Op {
        let u = bld.uniform(&[0, g, heads, head_dim_u]);
        Op {
            pipeline: P_RMS_IP,
            bind: bind(P_RMS_IP, &[be(0, x), be(1, &gammas_buf), wgpu::BindGroupEntry { binding: 2, resource: u.as_entire_binding() }]),
            groups: (heads, 1, 1),
        }
    };
    let rope_op = |bld: &Builder, x: &wgpu::Buffer, heads: u32| -> Op {
        let u = bld.uniform(&[heads, head_dim_u, (rope_theta as f32).to_bits(), 0]);
        Op {
            pipeline: P_ROPE,
            bind: bind(P_ROPE, &[be(0, x), wgpu::BindGroupEntry { binding: 1, resource: u.as_entire_binding() }, be(2, &step_uniform)]),
            groups: (heads, 1, 1),
        }
    };
    let ew_op = |bld: &Builder, pipeline: usize, a: &wgpu::Buffer, c: &wgpu::Buffer, y: &wgpu::Buffer, n: usize| -> Op {
        let u = bld.uniform(&[n as u32, 0, 0, 0]);
        Op {
            pipeline,
            bind: bind(pipeline, &[be(0, a), be(1, c), be(2, y), wgpu::BindGroupEntry { binding: 3, resource: u.as_entire_binding() }]),
            groups: ((n as u32).div_ceil(256), 1, 1),
        }
    };

    // pre: embed gather into h
    let embed_u = b.uniform(&[vocab_u as u32, hidden_u as u32, (hidden_u / 32) as u32, emb_w]);
    let pre_ops = vec![Op {
        pipeline: P_EMBED,
        bind: bind(P_EMBED, &[be(0, &wbuf), be(1, &h),
            wgpu::BindGroupEntry { binding: 2, resource: embed_u.as_entire_binding() },
            be(3, &step_uniform)]),
        groups: (1, 1, 1),
    }];

    let mut layers = Vec::with_capacity(n_layers);
    for l in 0..n_layers {
        let wo = &layer_woffs[l];
        let go = &layer_goffs[l];
        let mut ops = Vec::with_capacity(17);
        ops.push(rms_op(&b, &h, &xn, hidden_u, go[0], 0, 0));
        ops.push(gemv_op(&b, &xn, &q, q_hidden, hidden_u, wo[0]));
        ops.push(gemv_op(&b, &xn, &k_cur, kv_hidden, hidden_u, wo[1]));
        ops.push(gemv_op(&b, &xn, &v_cur, kv_hidden, hidden_u, wo[2]));
        ops.push(rms_ip_op(&b, &q, go[1], n_heads_u));
        ops.push(rms_ip_op(&b, &k_cur, go[2], n_kv_u));
        ops.push(rope_op(&b, &q, n_heads_u));
        ops.push(rope_op(&b, &k_cur, n_kv_u));
        // (kv append happens as encoder copies — see decode)
        ops.push(Op {
            pipeline: P_ATTN,
            bind: bind(P_ATTN, &[be(0, &q), be(1, &k_bufs[l]), be(2, &v_bufs[l]), be(3, &attn_out), wgpu::BindGroupEntry { binding: 4, resource: b.uniform(&[n_heads_u, n_kv_u, head_dim_u, 0]).as_entire_binding() }, be(5, &step_uniform)]),
            groups: (n_heads_u, 1, 1),
        });
        ops.push(gemv_op(&b, &attn_out, &proj, hidden_u, q_hidden, wo[3]));
        ops.push(ew_op(&b, P_ADD, &h, &proj, &h2, hidden_u));
        ops.push(rms_op(&b, &h2, &xn, hidden_u, go[3], 0, 0));
        ops.push(gemv_op(&b, &xn, &gate, ffn, hidden_u, wo[4]));
        ops.push(gemv_op(&b, &xn, &up, ffn, hidden_u, wo[5]));
        ops.push(ew_op(&b, P_SILU, &gate, &up, &gated, ffn));
        ops.push(gemv_op(&b, &gated, &proj, hidden_u, ffn, wo[6]));
        ops.push(ew_op(&b, P_ADD, &h2, &proj, &h, hidden_u));
        layers.push(Layer { ops, k_copy: ((kv_hidden * 4) as u64, (kv_hidden * 4) as u64) });
    }

    // post: final rms + logits gemv
    let post_ops = vec![
        rms_op(&b, &h, &xn, hidden_u, out_norm_g, 0, 0),
        gemv_op(&b, &xn, &logits, vocab_u, hidden_u, emb_w),
    ];

    let m = Model {
        device,
        queue,
        pipelines: b.pipelines,
        step_uniform,
        layers,
        pre_ops,
        post_ops,
        k_bufs,
        v_bufs,
        k_cur,
        v_cur,
        kv_row_bytes: (kv_hidden * 4) as u64,
        logits,
        staging,
        vocab: vocab_u,
    };
    *model_cell().lock().unwrap() = Some(m);
    1
}

fn run_step(m: &Model, token: i64, pos: i64) {
    let step = [token as u32, pos as u32, (pos + 1) as u32, 0u32];
    m.queue.write_buffer(&m.step_uniform, 0, bytemuck::cast_slice(&step));
    let mut enc = m.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        for op in &m.pre_ops {
            pass.set_pipeline(&m.pipelines[op.pipeline]);
            pass.set_bind_group(0, &op.bind, &[]);
            pass.dispatch_workgroups(op.groups.0, op.groups.1, op.groups.2);
        }
        pass.set_pipeline(&m.pipelines[P_GEMV]); // placeholder; reset per op below
    }
    for (l, layer) in m.layers.iter().enumerate() {
        // ops up to and including rope(k) — then kv copies — then the rest
        let split = 8; // first 8 ops end with rope(k_cur)
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            for op in &layer.ops[..split] {
                pass.set_pipeline(&m.pipelines[op.pipeline]);
                pass.set_bind_group(0, &op.bind, &[]);
                pass.dispatch_workgroups(op.groups.0, op.groups.1, op.groups.2);
            }
        }
        let dst_off = pos as u64 * m.kv_row_bytes;
        enc.copy_buffer_to_buffer(&m.k_cur, 0, &m.k_bufs[l], dst_off, layer.k_copy.0);
        enc.copy_buffer_to_buffer(&m.v_cur, 0, &m.v_bufs[l], dst_off, layer.k_copy.1);
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            for op in &layer.ops[split..] {
                pass.set_pipeline(&m.pipelines[op.pipeline]);
                pass.set_bind_group(0, &op.bind, &[]);
                pass.dispatch_workgroups(op.groups.0, op.groups.1, op.groups.2);
            }
        }
    }
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        for op in &m.post_ops {
            pass.set_pipeline(&m.pipelines[op.pipeline]);
            pass.set_bind_group(0, &op.bind, &[]);
            pass.dispatch_workgroups(op.groups.0, op.groups.1, op.groups.2);
        }
    }
    enc.copy_buffer_to_buffer(&m.logits, 0, &m.staging, 0, (m.vocab * 4) as u64);
    m.queue.submit([enc.finish()]);
}

fn read_logits(m: &Model) -> Vec<f32> {
    let slice = m.staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    m.device.poll(wgpu::Maintain::Wait);
    if rx.recv().map(|r| r.is_err()).unwrap_or(true) { return vec![]; }
    let out = bytemuck::cast_slice::<u8, f32>(&slice.get_mapped_range()).to_vec();
    m.staging.unmap();
    out
}

/// Feed one token at absolute position `pos`; return argmax of the logits.
pub fn decode_argmax(token: i64, pos: i64) -> i64 {
    let cell = model_cell().lock().unwrap();
    let Some(m) = cell.as_ref() else { return -1 };
    if pos as usize >= MAX_SEQ { return -2 }
    run_step(m, token, pos);
    let logits = read_logits(m);
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > bv { bv = v; best = i; }
    }
    best as i64
}

/// Parity variant: full logits back to Almide.
pub fn decode_logits(token: i64, pos: i64) -> Vec<f64> {
    let cell = model_cell().lock().unwrap();
    let Some(m) = cell.as_ref() else { return vec![] };
    if pos as usize >= MAX_SEQ { return vec![] }
    run_step(m, token, pos);
    read_logits(m).iter().map(|&v| v as f64).collect()
}
