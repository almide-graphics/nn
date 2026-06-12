// nn native GPU module — wgpu-backed Q8_0 GEMV (L3 spike).
//
// Injected into the generated crate by almide's native-deps mechanism and
// reached from Almide via `@extern(rs, "gpu", …)` (see src/gpu.almd).
//
// Weight layout ON GPU: Q8_0 blocks are repacked at upload from the GGUF
// 34-byte form (f16 scale + 32×i8) to a 36-byte, u32-aligned form
// (f32 scale + 8 words of quants) — storage buffers index by u32, and the
// f16 decode moves to upload time.

use std::sync::{Mutex, OnceLock};

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

static GPU: OnceLock<Option<Gpu>> = OnceLock::new();
static BUFFERS: Mutex<Vec<wgpu::Buffer>> = Mutex::new(Vec::new());

const WGSL: &str = r#"
struct Params { rows: u32, cols: u32, blocks_per_row: u32, _pad: u32 }

@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn gemv(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    // 2-D dispatch: one dimension caps at 65535 workgroups, vocab is 151936.
    let row = wid.y * 32768u + wid.x;
    if (row >= p.rows) { return; }
    let words_per_block = 9u;
    let row_base = row * p.blocks_per_row * words_per_block;
    var acc = 0.0;
    var b = lid.x;
    while (b < p.blocks_per_row) {
        let base = row_base + b * words_per_block;
        let scale = bitcast<f32>(w[base]);
        var s = 0.0;
        let xbase = b * 32u;
        for (var k = 0u; k < 8u; k = k + 1u) {
            let word = w[base + 1u + k];
            let q0 = f32(i32(word << 24u) >> 24u);
            let q1 = f32(i32(word << 16u) >> 24u);
            let q2 = f32(i32(word << 8u) >> 24u);
            let q3 = f32(i32(word) >> 24u);
            let xk = xbase + k * 4u;
            s = s + q0 * x[xk] + q1 * x[xk + 1u] + q2 * x[xk + 2u] + q3 * x[xk + 3u];
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
    if (lid.x == 0u) { y[row] = partial[0u]; }
}
"#;

fn fp16_to_f32(raw: u16) -> f32 {
    let sign = (raw >> 15) as u32;
    let exp = ((raw >> 10) & 0x1F) as u32;
    let man = (raw & 0x3FF) as u32;
    let bits = if exp == 0 {
        if man == 0 { sign << 31 } else {
            // subnormal: normalize
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

fn gpu() -> &'static Option<Gpu> {
    GPU.get_or_init(|| {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))?;
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_limits: wgpu::Limits {
                    max_storage_buffer_binding_size: 1 << 30, // 1 GiB for the weight buffer
                    max_buffer_size: 1 << 30,
                    ..wgpu::Limits::default()
                },
                ..Default::default()
            },
            None,
        )).ok()?;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("q8-gemv"),
            source: wgpu::ShaderSource::Wgsl(WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("q8-gemv"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("gemv"),
            compilation_options: Default::default(),
            cache: None,
        });
        Some(Gpu { device, queue, pipeline, layout })
    })
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub fn fp16_to_f32_pub(raw: u16) -> f32 {
    fp16_to_f32(raw)
}

/// Shared device/queue accessor for sibling native modules (gpu_model).
pub fn device_queue() -> Option<(&'static wgpu::Device, &'static wgpu::Queue)> {
    gpu().as_ref().map(|g| (&g.device, &g.queue))
}

/// 1 if a device is available, 0 otherwise.
pub fn init() -> i64 {
    if gpu().is_some() { 1 } else { 0 }
}

/// Repack a Q8_0 region (rows × cols, llama.cpp layout) from the source
/// buffer into a GPU storage buffer. Returns a buffer handle (or -1).
pub fn upload_q8(data: &Vec<u8>, off: i64, rows: i64, cols: i64) -> i64 {
    let Some(g) = gpu() else { return -1 };
    let rows = rows.max(0) as usize;
    let cols = cols.max(0) as usize;
    let blocks_per_row = cols / 32;
    let src_row = blocks_per_row * 34;
    let off = off.max(0) as usize;
    let mut packed = Vec::<u32>::with_capacity(rows * blocks_per_row * 9);
    for r in 0..rows {
        let rb = off + r * src_row;
        for b in 0..blocks_per_row {
            let bb = rb + b * 34;
            let scale = fp16_to_f32(u16::from_le_bytes([data[bb], data[bb + 1]]));
            packed.push(scale.to_bits());
            for k in 0..8 {
                let i = bb + 2 + k * 4;
                packed.push(u32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]));
            }
        }
    }
    use wgpu::util::DeviceExt;
    let buf = g.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("q8-weights"),
        contents: bytemuck::cast_slice(&packed),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let mut table = BUFFERS.lock().unwrap();
    table.push(buf);
    (table.len() - 1) as i64
}

/// y = W @ x for an uploaded Q8 buffer. x is the activation (len = cols).
pub fn gemv_q8(buf: i64, rows: i64, cols: i64, x: &Vec<f64>) -> Vec<f64> {
    let Some(g) = gpu() else { return vec![] };
    let table = BUFFERS.lock().unwrap();
    let Some(w) = table.get(buf.max(0) as usize) else { return vec![] };
    let rows_u = rows.max(0) as u32;
    let cols_u = cols.max(0) as u32;
    let xf: Vec<f32> = x.iter().map(|&v| v as f32).collect();

    use wgpu::util::DeviceExt;
    let xbuf = g.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&xf),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let ybuf = g.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: rows_u as u64 * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params = [rows_u, cols_u, cols_u / 32, 0u32];
    let pbuf = g.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let staging = g.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: rows_u as u64 * 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &g.layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: w.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: xbuf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: ybuf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: pbuf.as_entire_binding() },
        ],
    });
    let mut enc = g.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&g.pipeline);
        pass.set_bind_group(0, &bind, &[]);
        let x_groups = rows_u.min(32768);
        let y_groups = rows_u.div_ceil(32768);
        pass.dispatch_workgroups(x_groups, y_groups, 1);
    }
    enc.copy_buffer_to_buffer(&ybuf, 0, &staging, 0, rows_u as u64 * 4);
    g.queue.submit([enc.finish()]);

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    g.device.poll(wgpu::Maintain::Wait);
    if rx.recv().map(|r| r.is_err()).unwrap_or(true) { return vec![]; }
    let out: Vec<f64> = bytemuck::cast_slice::<u8, f32>(&slice.get_mapped_range())
        .iter().map(|&v| v as f64).collect();
    staging.unmap();
    out
}
