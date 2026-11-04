//! Wgpu device + kernel runner.
//!
//! Synchronous (pollster-bridged) entry points for the v0.1 CLI/test path.
//! The async scheduler in `rce-vm-scheduler` will use the native async
//! API directly.

use crate::error::{Result, WgpuError};
use crate::kernels;
use bytemuck::{Pod, Zeroable};
use rce_vm_backend_cpu::tensor::DType;
use rce_vm_backend_cpu::{HostTensor, ScalarValue, Value};
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Dims4 {
    a: u32,
    b: u32,
    c: u32,
    d: u32,
}

/// Mirrors the WGSL `BcastDims` struct (uniform buffer layout).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BroadcastDims {
    rank: u32,
    op: u32,
    n: u32,
    _pad: u32,
    out_shape: [u32; 4],
    a_strides: [u32; 4],
    b_strides: [u32; 4],
}

/// Mirrors the WGSL `FusedDims` (matmul+scale+softmax) struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FusedDims {
    m: u32,
    k: u32,
    n: u32,
    _pad0: u32,
    scale: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
}

pub struct WgpuExecutor {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    elementwise_add_pipeline: wgpu::ComputePipeline,
    elementwise_pipeline: wgpu::ComputePipeline,
    elementwise_scalar_pipeline: wgpu::ComputePipeline,
    elementwise_bcast_pipeline: wgpu::ComputePipeline,
    matmul_pipeline: wgpu::ComputePipeline,
    matmul_batched_pipeline: wgpu::ComputePipeline,
    matmul_scale_softmax_pipeline: wgpu::ComputePipeline,
    softmax_last_axis_pipeline: wgpu::ComputePipeline,
    reduce_last_axis_pipeline: wgpu::ComputePipeline,
    transpose_2d_pipeline: wgpu::ComputePipeline,
    activation_pipeline: wgpu::ComputePipeline,
    /// Adapter info string for diagnostics.
    pub adapter_info: String,
}

impl WgpuExecutor {
    /// Initialise wgpu synchronously. Returns `WgpuError::NoAdapter` if
    /// no GPU is available (CI / headless).
    pub fn new() -> Result<Self> {
        pollster::block_on(Self::new_async())
    }

    pub async fn new_async() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|_| WgpuError::NoAdapter)?;
        let info = adapter.get_info();
        let adapter_info = format!(
            "{} {} ({:?}) backend={:?}",
            info.vendor, info.name, info.device_type, info.backend
        );

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("rce-vm-backend-wgpu device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::default(),
            })
            .await
            .map_err(|e| WgpuError::DeviceRequest(format!("{e}")))?;
        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let elementwise_add_pipeline = build_pipeline(
            &device,
            "elementwise_add_f32",
            kernels::ELEMENTWISE_ADD_F32,
            "main",
        )?;
        let elementwise_pipeline =
            build_pipeline(&device, "elementwise_f32", kernels::ELEMENTWISE_F32, "main")?;
        let elementwise_scalar_pipeline = build_pipeline(
            &device,
            "elementwise_scalar_bcast_f32",
            kernels::ELEMENTWISE_SCALAR_BCAST_F32,
            "main",
        )?;
        let matmul_pipeline = build_pipeline(&device, "matmul_f32", kernels::MATMUL_F32, "main")?;
        let matmul_batched_pipeline = build_pipeline(
            &device,
            "matmul_batched_f32",
            kernels::MATMUL_BATCHED_F32,
            "main",
        )?;
        let softmax_last_axis_pipeline = build_pipeline(
            &device,
            "softmax_last_axis_f32",
            kernels::SOFTMAX_LAST_AXIS_F32,
            "main",
        )?;
        let reduce_last_axis_pipeline = build_pipeline(
            &device,
            "reduce_last_axis_f32",
            kernels::REDUCE_LAST_AXIS_F32,
            "main",
        )?;
        let transpose_2d_pipeline = build_pipeline(
            &device,
            "transpose_2d_f32",
            kernels::TRANSPOSE_2D_F32,
            "main",
        )?;
        let activation_pipeline =
            build_pipeline(&device, "activation_f32", kernels::ACTIVATION_F32, "main")?;
        let elementwise_bcast_pipeline = build_pipeline(
            &device,
            "elementwise_bcast_f32",
            kernels::ELEMENTWISE_BCAST_F32,
            "main",
        )?;
        let matmul_scale_softmax_pipeline = build_pipeline(
            &device,
            "matmul_scale_softmax_f32",
            kernels::MATMUL_SCALE_SOFTMAX_F32,
            "main",
        )?;

        Ok(Self {
            device,
            queue,
            elementwise_add_pipeline,
            elementwise_pipeline,
            elementwise_scalar_pipeline,
            elementwise_bcast_pipeline,
            matmul_pipeline,
            matmul_batched_pipeline,
            matmul_scale_softmax_pipeline,
            softmax_last_axis_pipeline,
            reduce_last_axis_pipeline,
            transpose_2d_pipeline,
            activation_pipeline,
            adapter_info,
        })
    }

    /// General N-D broadcast elementwise (rank ≤ 4). `op_code`: 0=add, 1=sub,
    /// 2=mul, 3=div. Outputs a tensor with the broadcast shape; no CPU
    /// pre-broadcast — strides are computed in Rust and the GPU walks them.
    pub fn elementwise_bcast_f32(
        &self,
        a: &HostTensor,
        b: &HostTensor,
        op_code: u32,
    ) -> Result<HostTensor> {
        if a.dtype != DType::F32 || b.dtype != DType::F32 {
            return Err(WgpuError::UnsupportedDtype("bcast: f32 only".into()));
        }
        let out_shape =
            rce_vm_backend_cpu::ml::broadcast::broadcast_shapes(0, &a.shape, &b.shape)
                .map_err(|e| WgpuError::Invalid(format!("{e}")))?;
        if out_shape.len() > 4 {
            return Err(WgpuError::Invalid(format!(
                "bcast: rank > 4 not supported in v0.3 (out shape {out_shape:?})"
            )));
        }
        let n = out_shape.iter().product::<usize>().max(1);
        let mut out_shape_pad = [1u32; 4];
        for (i, d) in out_shape.iter().enumerate() {
            out_shape_pad[i] = *d as u32;
        }
        let a_strides = bcast_strides(&a.shape, &out_shape);
        let b_strides = bcast_strides(&b.shape, &out_shape);
        let dims = BroadcastDims {
            rank: out_shape.len() as u32,
            op: op_code,
            n: n as u32,
            _pad: 0,
            out_shape: out_shape_pad,
            a_strides,
            b_strides,
        };
        let bytes = self.run_bcast_kernel(
            &self.elementwise_bcast_pipeline,
            &a.data,
            &b.data,
            n * 4,
            &dims,
            ((n as u32 + 63) / 64, 1, 1),
        )?;
        HostTensor::new(out_shape, DType::F32, bytes)
            .map_err(|e| WgpuError::Invalid(format!("readback: {e}")))
    }

    /// Fused `softmax((a @ b) * scale, axis=-1)`. f32, 2D inputs, N ≤ 256.
    pub fn matmul_scale_softmax_f32(
        &self,
        a: &HostTensor,
        b: &HostTensor,
        scale: f32,
    ) -> Result<HostTensor> {
        if a.shape.len() != 2 || b.shape.len() != 2 {
            return Err(WgpuError::Invalid(format!(
                "matmul_scale_softmax: 2D only (got {:?} and {:?})",
                a.shape, b.shape
            )));
        }
        if a.dtype != DType::F32 || b.dtype != DType::F32 {
            return Err(WgpuError::UnsupportedDtype("fused: f32 only".into()));
        }
        let m = a.shape[0];
        let k = a.shape[1];
        let k2 = b.shape[0];
        let n = b.shape[1];
        if k != k2 {
            return Err(WgpuError::Invalid(format!(
                "matmul_scale_softmax: inner dim mismatch ({k} vs {k2})"
            )));
        }
        if n > 256 {
            return Err(WgpuError::Invalid(format!(
                "matmul_scale_softmax: N={n} > 256 (workgroup-shared budget)"
            )));
        }
        let dims = FusedDims {
            m: m as u32,
            k: k as u32,
            n: n as u32,
            _pad0: 0,
            scale,
            _pad1: 0.0,
            _pad2: 0.0,
            _pad3: 0.0,
        };
        let bytes = self.run_fused_kernel(
            &self.matmul_scale_softmax_pipeline,
            &a.data,
            &b.data,
            m * n * 4,
            &dims,
            (m as u32, 1, 1),
        )?;
        HostTensor::new(vec![m, n], DType::F32, bytes)
            .map_err(|e| WgpuError::Invalid(format!("readback: {e}")))
    }

    /// Element-wise scalar-broadcast: one operand is a 0-D tensor.
    /// `op_code`: 0=add, 1=sub, 2=mul, 3=div. `scalar_left=true` → `a` is scalar.
    pub fn elementwise_scalar_bcast_f32(
        &self,
        a: &HostTensor,
        b: &HostTensor,
        op_code: u32,
        scalar_left: bool,
    ) -> Result<HostTensor> {
        if a.dtype != DType::F32 || b.dtype != DType::F32 {
            return Err(WgpuError::UnsupportedDtype("scalar broadcast: f32 only".into()));
        }
        let (n_shape, n) = if scalar_left {
            (b.shape.clone(), b.numel())
        } else {
            (a.shape.clone(), a.numel())
        };
        let dims = Dims4 {
            a: n as u32,
            b: op_code,
            c: scalar_left as u32,
            d: 0,
        };
        let bytes = self.run_binary_kernel(
            &self.elementwise_scalar_pipeline,
            &a.data,
            &b.data,
            n * 4,
            dims,
            ((n as u32 + 63) / 64, 1, 1),
        )?;
        HostTensor::new(n_shape, DType::F32, bytes)
            .map_err(|e| WgpuError::Invalid(format!("readback: {e}")))
    }

    /// Batched matmul: `a [B,M,K] × b [B,K,N] → [B,M,N]`. Equal batch dims.
    pub fn matmul_batched_f32(&self, a: &HostTensor, b: &HostTensor) -> Result<HostTensor> {
        if a.shape.len() != 3 || b.shape.len() != 3 {
            return Err(WgpuError::Invalid(format!(
                "matmul_batched: rank-3 only (got {:?} and {:?})",
                a.shape, b.shape
            )));
        }
        if a.dtype != DType::F32 || b.dtype != DType::F32 {
            return Err(WgpuError::UnsupportedDtype("matmul_batched: f32 only".into()));
        }
        if a.shape[0] != b.shape[0] {
            return Err(WgpuError::Invalid(format!(
                "matmul_batched: batch mismatch ({} vs {})",
                a.shape[0], b.shape[0]
            )));
        }
        let batch = a.shape[0];
        let m = a.shape[1];
        let k = a.shape[2];
        let k2 = b.shape[1];
        let n = b.shape[2];
        if k != k2 {
            return Err(WgpuError::Invalid(format!(
                "matmul_batched: inner dim mismatch ({k} vs {k2})"
            )));
        }
        let dims = Dims4 {
            a: batch as u32,
            b: m as u32,
            c: k as u32,
            d: n as u32,
        };
        let bytes = self.run_binary_kernel(
            &self.matmul_batched_pipeline,
            &a.data,
            &b.data,
            batch * m * n * 4,
            dims,
            ((m as u32 + 7) / 8, (n as u32 + 7) / 8, batch as u32),
        )?;
        HostTensor::new(vec![batch, m, n], DType::F32, bytes)
            .map_err(|e| WgpuError::Invalid(format!("readback: {e}")))
    }

    /// Apply an element-wise activation. `op_code`: 0=relu, 1=sigmoid, 2=tanh, 3=gelu.
    pub fn activation_f32(&self, x: &HostTensor, op_code: u32) -> Result<HostTensor> {
        if x.dtype != DType::F32 {
            return Err(WgpuError::UnsupportedDtype("activation: f32 only".into()));
        }
        let n = x.numel();
        let dims = Dims4 {
            a: n as u32,
            b: op_code,
            c: 0,
            d: 0,
        };
        let bytes = self.run_unary_kernel(
            &self.activation_pipeline,
            &x.data,
            x.data.len(),
            dims,
            ((n as u32 + 63) / 64, 1, 1),
        )?;
        HostTensor::new(x.shape.clone(), DType::F32, bytes)
            .map_err(|e| WgpuError::Invalid(format!("readback: {e}")))
    }

    /// Generalised elementwise binary on equal-shape f32 tensors.
    /// `op_code`: 0=add, 1=sub, 2=mul, 3=div.
    pub fn elementwise_f32(
        &self,
        a: &HostTensor,
        b: &HostTensor,
        op_code: u32,
    ) -> Result<HostTensor> {
        if a.shape != b.shape {
            return Err(WgpuError::Invalid(format!(
                "elementwise: shapes differ {:?} vs {:?}",
                a.shape, b.shape
            )));
        }
        if a.dtype != DType::F32 || b.dtype != DType::F32 {
            return Err(WgpuError::UnsupportedDtype(format!(
                "elementwise: f32 only (got {} and {})",
                a.dtype.name(),
                b.dtype.name()
            )));
        }
        let n = a.numel();
        let dims = Dims4 {
            a: n as u32,
            b: op_code,
            c: 0,
            d: 0,
        };
        let bytes = self.run_binary_kernel(
            &self.elementwise_pipeline,
            &a.data,
            &b.data,
            n * 4,
            dims,
            ((n as u32 + 63) / 64, 1, 1),
        )?;
        HostTensor::new(a.shape.clone(), DType::F32, bytes)
            .map_err(|e| WgpuError::Invalid(format!("readback: {e}")))
    }

    /// Reduce along the last axis. `op_code`: 0=sum, 1=mean, 2=max, 3=min.
    /// Only operates when `axes == [last_dim]` and `keepdim == false`;
    /// other reduction patterns return `Invalid` and the interpreter falls
    /// through to the CPU kernel.
    pub fn reduce_last_axis_f32(
        &self,
        x: &HostTensor,
        op_code: u32,
    ) -> Result<HostTensor> {
        if x.dtype != DType::F32 {
            return Err(WgpuError::UnsupportedDtype("reduce: f32 only".into()));
        }
        if x.shape.is_empty() {
            return Err(WgpuError::Invalid("reduce on rank-0 tensor".into()));
        }
        let dim = *x.shape.last().unwrap() as u32;
        let outer: u32 = x.shape[..x.shape.len() - 1]
            .iter()
            .product::<usize>()
            .max(1) as u32;
        let dims = Dims4 {
            a: outer,
            b: dim,
            c: op_code,
            d: 0,
        };
        let out_shape: Vec<usize> = x.shape[..x.shape.len() - 1].to_vec();
        let out_byte_len = (outer as usize) * 4;
        let bytes = self.run_unary_kernel(
            &self.reduce_last_axis_pipeline,
            &x.data,
            out_byte_len,
            dims,
            ((outer + 63) / 64, 1, 1),
        )?;
        HostTensor::new(out_shape, DType::F32, bytes)
            .map_err(|e| WgpuError::Invalid(format!("readback: {e}")))
    }

    /// Softmax along the last axis. f32 only.
    pub fn softmax_last_axis_f32(&self, x: &HostTensor) -> Result<HostTensor> {
        if x.dtype != DType::F32 {
            return Err(WgpuError::UnsupportedDtype(format!(
                "softmax: f32 only (got {})",
                x.dtype.name()
            )));
        }
        if x.shape.is_empty() {
            return Err(WgpuError::Invalid("softmax on rank-0 tensor".into()));
        }
        let dim = *x.shape.last().unwrap() as u32;
        let outer: u32 = x.shape[..x.shape.len() - 1]
            .iter()
            .product::<usize>()
            .max(1) as u32;
        let dims = Dims4 {
            a: outer,
            b: dim,
            c: 0,
            d: 0,
        };
        let bytes = self.run_unary_kernel(
            &self.softmax_last_axis_pipeline,
            &x.data,
            x.data.len(),
            dims,
            ((outer + 63) / 64, 1, 1),
        )?;
        HostTensor::new(x.shape.clone(), DType::F32, bytes)
            .map_err(|e| WgpuError::Invalid(format!("readback: {e}")))
    }

    /// 2-D transpose. Higher-rank transpose returns `Unsupported`.
    pub fn transpose_2d_f32(&self, x: &HostTensor) -> Result<HostTensor> {
        if x.shape.len() != 2 {
            return Err(WgpuError::Invalid(format!(
                "transpose_2d: rank-2 only (got {:?})",
                x.shape
            )));
        }
        if x.dtype != DType::F32 {
            return Err(WgpuError::UnsupportedDtype(format!(
                "transpose_2d: f32 only (got {})",
                x.dtype.name()
            )));
        }
        let rows = x.shape[0] as u32;
        let cols = x.shape[1] as u32;
        let dims = Dims4 {
            a: rows,
            b: cols,
            c: 0,
            d: 0,
        };
        let bytes = self.run_unary_kernel(
            &self.transpose_2d_pipeline,
            &x.data,
            x.data.len(),
            dims,
            ((rows + 7) / 8, (cols + 7) / 8, 1),
        )?;
        HostTensor::new(vec![cols as usize, rows as usize], DType::F32, bytes)
            .map_err(|e| WgpuError::Invalid(format!("readback: {e}")))
    }

    /// Element-wise add over two equal-shape f32 tensors.
    pub fn elementwise_add_f32(&self, a: &HostTensor, b: &HostTensor) -> Result<HostTensor> {
        if a.shape != b.shape {
            return Err(WgpuError::Invalid(format!(
                "elementwise_add: shapes differ {:?} vs {:?}; broadcast not yet on wgpu (v0.2)",
                a.shape, b.shape
            )));
        }
        if a.dtype != DType::F32 || b.dtype != DType::F32 {
            return Err(WgpuError::UnsupportedDtype(format!(
                "elementwise_add: f32 only in v0.1 (got {} and {})",
                a.dtype.name(),
                b.dtype.name()
            )));
        }
        let n = a.numel();
        let dims = Dims4 {
            a: n as u32,
            b: 0,
            c: 0,
            d: 0,
        };
        let out_bytes = self.run_binary_kernel(
            &self.elementwise_add_pipeline,
            &a.data,
            &b.data,
            n * 4,
            dims,
            ((n as u32 + 63) / 64, 1, 1),
        )?;
        HostTensor::new(a.shape.clone(), DType::F32, out_bytes).map_err(|e| {
            WgpuError::Invalid(format!("readback shape mismatch: {e}"))
        })
    }

    /// 2D dense matmul, f32. `a` is `[M,K]`, `b` is `[K,N]`, output is `[M,N]`.
    pub fn matmul_f32(&self, a: &HostTensor, b: &HostTensor) -> Result<HostTensor> {
        if a.shape.len() != 2 || b.shape.len() != 2 {
            return Err(WgpuError::Invalid(format!(
                "matmul: v0.1 wgpu kernel handles 2D only (got {:?} and {:?})",
                a.shape, b.shape
            )));
        }
        if a.dtype != DType::F32 || b.dtype != DType::F32 {
            return Err(WgpuError::UnsupportedDtype(format!(
                "matmul: f32 only in v0.1 (got {} and {})",
                a.dtype.name(),
                b.dtype.name()
            )));
        }
        let m = a.shape[0];
        let k = a.shape[1];
        let k2 = b.shape[0];
        let n = b.shape[1];
        if k != k2 {
            return Err(WgpuError::Invalid(format!(
                "matmul: inner dim mismatch ({k} vs {k2})"
            )));
        }
        let dims = Dims4 {
            a: m as u32,
            b: k as u32,
            c: n as u32,
            d: 0,
        };
        let out_bytes = self.run_binary_kernel(
            &self.matmul_pipeline,
            &a.data,
            &b.data,
            m * n * 4,
            dims,
            ((m as u32 + 7) / 8, (n as u32 + 7) / 8, 1),
        )?;
        HostTensor::new(vec![m, n], DType::F32, out_bytes).map_err(|e| {
            WgpuError::Invalid(format!("readback shape mismatch: {e}"))
        })
    }

    fn run_bcast_kernel(
        &self,
        pipeline: &wgpu::ComputePipeline,
        a_bytes: &[u8],
        b_bytes: &[u8],
        out_byte_len: usize,
        dims: &BroadcastDims,
        workgroups: (u32, u32, u32),
    ) -> Result<Vec<u8>> {
        let device = &self.device;
        let queue = &self.queue;

        let a_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bcast.a"),
            contents: a_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });
        let b_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bcast.b"),
            contents: b_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bcast.out"),
            size: out_byte_len as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let dims_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bcast.dims"),
            contents: bytemuck::bytes_of(dims),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bcast.bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: dims_buf.as_entire_binding() },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("bcast.cmd"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bcast.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
        }
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bcast.staging"),
            size: out_byte_len as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, out_byte_len as u64);
        queue.submit(std::iter::once(encoder.finish()));
        let slice = staging.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = sender.send(r);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let result = pollster::block_on(receiver.receive())
            .ok_or_else(|| WgpuError::Mapping("channel closed".into()))?;
        result.map_err(|e| WgpuError::Mapping(format!("{e}")))?;
        let view = slice.get_mapped_range();
        let bytes = view.to_vec();
        drop(view);
        staging.unmap();
        Ok(bytes)
    }

    fn run_fused_kernel(
        &self,
        pipeline: &wgpu::ComputePipeline,
        a_bytes: &[u8],
        b_bytes: &[u8],
        out_byte_len: usize,
        dims: &FusedDims,
        workgroups: (u32, u32, u32),
    ) -> Result<Vec<u8>> {
        let device = &self.device;
        let queue = &self.queue;

        let a_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fused.a"),
            contents: a_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });
        let b_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fused.b"),
            contents: b_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fused.out"),
            size: out_byte_len as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let dims_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fused.dims"),
            contents: bytemuck::bytes_of(dims),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fused.bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: dims_buf.as_entire_binding() },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fused.cmd"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fused.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
        }
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fused.staging"),
            size: out_byte_len as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, out_byte_len as u64);
        queue.submit(std::iter::once(encoder.finish()));
        let slice = staging.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = sender.send(r);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let result = pollster::block_on(receiver.receive())
            .ok_or_else(|| WgpuError::Mapping("channel closed".into()))?;
        result.map_err(|e| WgpuError::Mapping(format!("{e}")))?;
        let view = slice.get_mapped_range();
        let bytes = view.to_vec();
        drop(view);
        staging.unmap();
        Ok(bytes)
    }

    fn run_unary_kernel(
        &self,
        pipeline: &wgpu::ComputePipeline,
        x_bytes: &[u8],
        out_byte_len: usize,
        dims: Dims4,
        workgroups: (u32, u32, u32),
    ) -> Result<Vec<u8>> {
        let device = &self.device;
        let queue = &self.queue;

        let x_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("x"),
            contents: x_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out"),
            size: out_byte_len as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let dims_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dims"),
            contents: bytemuck::cast_slice(&[dims]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: x_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dims_buf.as_entire_binding() },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cmd"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
        }
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: out_byte_len as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, out_byte_len as u64);
        queue.submit(std::iter::once(encoder.finish()));
        let slice = staging.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = sender.send(r);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let result = pollster::block_on(receiver.receive())
            .ok_or_else(|| WgpuError::Mapping("channel closed".into()))?;
        result.map_err(|e| WgpuError::Mapping(format!("{e}")))?;
        let view = slice.get_mapped_range();
        let bytes = view.to_vec();
        drop(view);
        staging.unmap();
        Ok(bytes)
    }

    fn run_binary_kernel(
        &self,
        pipeline: &wgpu::ComputePipeline,
        a_bytes: &[u8],
        b_bytes: &[u8],
        out_byte_len: usize,
        dims: Dims4,
        workgroups: (u32, u32, u32),
    ) -> Result<Vec<u8>> {
        let device = &self.device;
        let queue = &self.queue;

        let a_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("a"),
            contents: a_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });
        let b_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("b"),
            contents: b_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out"),
            size: out_byte_len as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let dims_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dims"),
            contents: bytemuck::cast_slice(&[dims]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: dims_buf.as_entire_binding() },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cmd"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
        }

        // Stage to a CPU-readable buffer.
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: out_byte_len as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, out_byte_len as u64);
        queue.submit(std::iter::once(encoder.finish()));

        // Read back.
        let slice = staging.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = sender.send(r);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let result = pollster::block_on(receiver.receive())
            .ok_or_else(|| WgpuError::Mapping("channel closed".into()))?;
        result.map_err(|e| WgpuError::Mapping(format!("{e}")))?;

        let view = slice.get_mapped_range();
        let bytes = view.to_vec();
        drop(view);
        staging.unmap();
        Ok(bytes)
    }
}

fn build_pipeline(
    device: &wgpu::Device,
    label: &str,
    src: &str,
    entry: &str,
) -> Result<wgpu::ComputePipeline> {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(src)),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &module,
        entry_point: Some(entry),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    Ok(pipeline)
}

// ── OpDispatch integration ───────────────────────────────────────────

use rce_vm_backend_cpu::ml::broadcast::align_pair;
use rce_vm_backend_cpu::{InterpretError, OpDispatch};
use rce_graph::graph::OpKind;

impl OpDispatch for WgpuExecutor {
    fn handles(&self) -> &[&'static str] {
        &[
            "matmul",
            "softmax",
            "transpose",
            "relu",
            "sigmoid",
            "tanh",
            "gelu",
            "matmul_scale_softmax",
            "reduce_sum",
            "reduce_mean",
            "reduce_max",
            "reduce_min",
        ]
    }

    fn execute(
        &self,
        node_id: usize,
        name: &str,
        args: &[Value],
    ) -> rce_vm_backend_cpu::Result<Value> {
        match name {
            "matmul" => self.dispatch_matmul(node_id, args),
            "softmax" => self.dispatch_softmax(node_id, args),
            "transpose" => self.dispatch_transpose(node_id, args),
            "relu" => self.dispatch_activation(node_id, args, 0),
            "sigmoid" => self.dispatch_activation(node_id, args, 1),
            "tanh" => self.dispatch_activation(node_id, args, 2),
            "gelu" => self.dispatch_activation(node_id, args, 3),
            "matmul_scale_softmax" => self.dispatch_matmul_scale_softmax(node_id, args),
            "reduce_sum" => self.dispatch_reduce(node_id, args, 0),
            "reduce_mean" => self.dispatch_reduce(node_id, args, 1),
            "reduce_max" => self.dispatch_reduce(node_id, args, 2),
            "reduce_min" => self.dispatch_reduce(node_id, args, 3),
            other => Err(InterpretError::Unsupported(format!(
                "wgpu executor: name `{other}` not handled"
            ))),
        }
    }

    fn execute_primitive(
        &self,
        node_id: usize,
        op: &OpKind,
        args: &[Value],
    ) -> rce_vm_backend_cpu::Result<Value> {
        // Only Add/Sub/Mul/Div on f32 tensors are claimed.
        let op_code = match op {
            OpKind::Add => 0u32,
            OpKind::Sub => 1u32,
            OpKind::Mul => 2u32,
            OpKind::Div => 3u32,
            _ => {
                return Err(InterpretError::Unsupported(format!(
                    "wgpu primitive: op {op:?} not handled"
                )));
            }
        };
        if args.len() != 2 {
            return Err(InterpretError::Unsupported(
                "wgpu primitive: expected 2 args".into(),
            ));
        }
        let a = arg_tensor(node_id, &args[0])?;
        let b = arg_tensor(node_id, &args[1])?;
        if a.dtype != DType::F32 || b.dtype != DType::F32 {
            return Err(InterpretError::Unsupported(format!(
                "wgpu primitive: f32 only (got {} and {})",
                a.dtype.name(),
                b.dtype.name()
            )));
        }
        let a_scalar = a.shape.is_empty();
        let b_scalar = b.shape.is_empty();
        let same_shape = a.shape == b.shape;

        let result = if same_shape {
            self.elementwise_f32(&a, &b, op_code)
        } else if a_scalar {
            self.elementwise_scalar_bcast_f32(&a, &b, op_code, true)
        } else if b_scalar {
            self.elementwise_scalar_bcast_f32(&a, &b, op_code, false)
        } else {
            // v0.3: general N-D broadcast on GPU (rank ≤ 4). Falls through
            // to CPU pre-broadcast for rank > 4 via the kernel's own check.
            self.elementwise_bcast_f32(&a, &b, op_code)
        };
        result.map(Value::Tensor).map_err(|e| match e {
            WgpuError::UnsupportedDtype(m) | WgpuError::Invalid(m) => {
                InterpretError::Unsupported(format!("wgpu primitive: {m}"))
            }
            other => InterpretError::Internal(format!("wgpu primitive: {other}")),
        })
    }
}

impl WgpuExecutor {
    fn dispatch_matmul(
        &self,
        node_id: usize,
        args: &[Value],
    ) -> rce_vm_backend_cpu::Result<Value> {
        if args.len() != 2 {
            return Err(InterpretError::ArityMismatch {
                node_id,
                kind: "wgpu:matmul".into(),
                expected: 2,
                got: args.len(),
            });
        }
        let a = arg_tensor(node_id, &args[0])?;
        let b = arg_tensor(node_id, &args[1])?;
        if a.dtype != DType::F32 || b.dtype != DType::F32 {
            return Err(InterpretError::Unsupported(format!(
                "wgpu matmul: f32 only (got {} × {})",
                a.dtype.name(),
                b.dtype.name()
            )));
        }
        if a.shape.len() == 2 && b.shape.len() == 2 {
            self.matmul_f32(&a, &b)
                .map(Value::Tensor)
                .map_err(|e| InterpretError::Internal(format!("wgpu: {e}")))
        } else if a.shape.len() == 3
            && b.shape.len() == 3
            && a.shape[0] == b.shape[0]
        {
            self.matmul_batched_f32(&a, &b)
                .map(Value::Tensor)
                .map_err(|e| InterpretError::Internal(format!("wgpu: {e}")))
        } else {
            // Higher-rank or unequal-batch broadcasts not yet on GPU.
            Err(InterpretError::Unsupported(format!(
                "wgpu matmul: shape pattern not supported in v0.2; got {:?} × {:?}",
                a.shape, b.shape
            )))
        }
    }

    fn dispatch_matmul_scale_softmax(
        &self,
        node_id: usize,
        args: &[Value],
    ) -> rce_vm_backend_cpu::Result<Value> {
        if args.len() != 3 {
            return Err(InterpretError::ArityMismatch {
                node_id,
                kind: "wgpu:matmul_scale_softmax".into(),
                expected: 3,
                got: args.len(),
            });
        }
        let a = arg_tensor(node_id, &args[0])?;
        let b = arg_tensor(node_id, &args[1])?;
        let scale = match &args[2] {
            Value::Scalar(ScalarValue::F32(x)) => *x,
            Value::Scalar(ScalarValue::F64(x)) => *x as f32,
            Value::Scalar(ScalarValue::I32(x)) => *x as f32,
            other => {
                return Err(InterpretError::TypeMismatch {
                    node_id,
                    msg: format!("matmul_scale_softmax scale: expected scalar, got {}", other.type_name()),
                });
            }
        };
        if a.shape.len() != 2 || b.shape.len() != 2 || a.dtype != DType::F32 {
            return Err(InterpretError::Unsupported(format!(
                "wgpu fused: 2D f32 only (got {:?}/{} × {:?}/{})",
                a.shape, a.dtype.name(), b.shape, b.dtype.name()
            )));
        }
        if b.shape[1] > 256 {
            return Err(InterpretError::Unsupported(format!(
                "wgpu fused: N={} > 256 (workgroup-shared budget)",
                b.shape[1]
            )));
        }
        self.matmul_scale_softmax_f32(&a, &b, scale)
            .map(Value::Tensor)
            .map_err(|e| InterpretError::Internal(format!("wgpu fused: {e}")))
    }

    fn dispatch_reduce(
        &self,
        node_id: usize,
        args: &[Value],
        op_code: u32,
    ) -> rce_vm_backend_cpu::Result<Value> {
        // Reduce ops take (x, axes_json, keepdim_bool).
        if args.len() != 3 {
            return Err(InterpretError::ArityMismatch {
                node_id,
                kind: "wgpu:reduce".into(),
                expected: 3,
                got: args.len(),
            });
        }
        let x = arg_tensor(node_id, &args[0])?;
        if x.dtype != DType::F32 {
            return Err(InterpretError::Unsupported(format!(
                "wgpu reduce: f32 only (got {})",
                x.dtype.name()
            )));
        }
        // axes_json arg: must be exactly the last axis, no keepdim.
        let axes_str = match &args[1] {
            Value::Scalar(ScalarValue::String(s)) => s,
            _ => {
                return Err(InterpretError::Unsupported(
                    "wgpu reduce: axes_json must be a string literal".into(),
                ));
            }
        };
        let keepdim = match &args[2] {
            Value::Scalar(ScalarValue::Bool(b)) => *b,
            _ => {
                return Err(InterpretError::Unsupported(
                    "wgpu reduce: keepdim must be a bool literal".into(),
                ));
            }
        };
        if keepdim {
            return Err(InterpretError::Unsupported(
                "wgpu reduce: keepdim=true falls through to CPU in v0.4".into(),
            ));
        }
        // Only handle the last-axis case; other axes fall through.
        let rank = x.shape.len();
        let last_axis = rank.saturating_sub(1) as i64;
        let want = format!("[{last_axis}]");
        let want_neg = "[-1]".to_string();
        let normalized: String = axes_str.chars().filter(|c| !c.is_whitespace()).collect();
        if normalized != want && normalized != want_neg {
            return Err(InterpretError::Unsupported(format!(
                "wgpu reduce: only last-axis supported in v0.4 (got axes={axes_str})"
            )));
        }
        self.reduce_last_axis_f32(&x, op_code)
            .map(Value::Tensor)
            .map_err(|e| InterpretError::Internal(format!("wgpu reduce: {e}")))
    }

    fn dispatch_activation(
        &self,
        node_id: usize,
        args: &[Value],
        op_code: u32,
    ) -> rce_vm_backend_cpu::Result<Value> {
        if args.len() != 1 {
            return Err(InterpretError::ArityMismatch {
                node_id,
                kind: format!("wgpu:activation:{op_code}"),
                expected: 1,
                got: args.len(),
            });
        }
        let x = arg_tensor(node_id, &args[0])?;
        if x.dtype != DType::F32 {
            return Err(InterpretError::Unsupported(format!(
                "wgpu activation: f32 only (got {})",
                x.dtype.name()
            )));
        }
        self.activation_f32(&x, op_code)
            .map(Value::Tensor)
            .map_err(|e| InterpretError::Internal(format!("wgpu: {e}")))
    }

    fn dispatch_softmax(
        &self,
        node_id: usize,
        args: &[Value],
    ) -> rce_vm_backend_cpu::Result<Value> {
        if args.len() != 2 {
            return Err(InterpretError::ArityMismatch {
                node_id,
                kind: "wgpu:softmax".into(),
                expected: 2,
                got: args.len(),
            });
        }
        let x = arg_tensor(node_id, &args[0])?;
        let axis = match &args[1] {
            Value::Scalar(ScalarValue::I32(a)) => *a,
            Value::Scalar(ScalarValue::I64(a)) => *a as i32,
            other => {
                return Err(InterpretError::TypeMismatch {
                    node_id,
                    msg: format!("softmax axis must be int, got {}", other.type_name()),
                });
            }
        };
        // v0.1 wgpu only handles softmax along the *last* axis.
        let rank = x.shape.len() as i32;
        let normalised = if axis < 0 { rank + axis } else { axis };
        if normalised != rank - 1 {
            return Err(InterpretError::Unsupported(format!(
                "wgpu softmax v0.1 supports last-axis only (got axis {axis}, rank {rank})"
            )));
        }
        if x.dtype != DType::F32 {
            return Err(InterpretError::Unsupported(format!(
                "wgpu softmax: f32 only (got {})",
                x.dtype.name()
            )));
        }
        self.softmax_last_axis_f32(&x)
            .map(Value::Tensor)
            .map_err(|e| InterpretError::Internal(format!("wgpu: {e}")))
    }

    fn dispatch_transpose(
        &self,
        node_id: usize,
        args: &[Value],
    ) -> rce_vm_backend_cpu::Result<Value> {
        if args.len() != 2 {
            return Err(InterpretError::ArityMismatch {
                node_id,
                kind: "wgpu:transpose".into(),
                expected: 2,
                got: args.len(),
            });
        }
        let x = arg_tensor(node_id, &args[0])?;
        // axes is a JSON-array string literal in port 1 (matches CPU contract);
        // for v0.1 wgpu we only handle 2D transpose [1,0].
        let axes_str = match &args[1] {
            Value::Scalar(ScalarValue::String(s)) => s.clone(),
            _ => {
                return Err(InterpretError::Unsupported(
                    "wgpu transpose: axes arg must be a string literal".into(),
                ));
            }
        };
        if x.shape.len() != 2 || x.dtype != DType::F32 {
            return Err(InterpretError::Unsupported(format!(
                "wgpu transpose v0.1 supports 2D f32 only; got {:?}/{}",
                x.shape,
                x.dtype.name()
            )));
        }
        // Accept "[1,0]" or "[1, 0]" as the only transposition we kernel.
        let axes_norm: String = axes_str.chars().filter(|c| !c.is_whitespace()).collect();
        if axes_norm != "[1,0]" {
            return Err(InterpretError::Unsupported(format!(
                "wgpu transpose v0.1 only handles axes=[1,0]; got `{axes_str}`"
            )));
        }
        self.transpose_2d_f32(&x)
            .map(Value::Tensor)
            .map_err(|e| InterpretError::Internal(format!("wgpu: {e}")))
    }
}

fn arg_tensor(node_id: usize, v: &Value) -> rce_vm_backend_cpu::Result<HostTensor> {
    match v {
        Value::Tensor(t) => Ok(t.clone()),
        Value::Scalar(ScalarValue::F32(x)) => HostTensor::from_f32(vec![], &[*x]),
        other => Err(InterpretError::TypeMismatch {
            node_id,
            msg: format!("wgpu matmul expected tensor arg, got {}", other.type_name()),
        }),
    }
}

/// Compute broadcast strides for a tensor `t` against the broadcast output
/// `out_shape`. Right-aligned: leading axes (where t has fewer dims) and
/// size-1 axes get stride 0; other axes get the natural row-major stride.
fn bcast_strides(t_shape: &[usize], out_shape: &[usize]) -> [u32; 4] {
    let mut out = [0u32; 4];
    if t_shape.is_empty() {
        // 0-D scalar broadcasts everywhere; all strides 0, single element at index 0.
        return out;
    }
    // Natural strides in t's own coordinate space (row-major).
    let mut natural = vec![1usize; t_shape.len()];
    for i in (0..t_shape.len().saturating_sub(1)).rev() {
        natural[i] = natural[i + 1] * t_shape[i + 1];
    }
    let pad = out_shape.len() - t_shape.len();
    for axis in 0..out_shape.len() {
        if axis < pad {
            out[axis] = 0;
        } else {
            let t_axis = axis - pad;
            if t_shape[t_axis] == 1 && out_shape[axis] != 1 {
                out[axis] = 0;
            } else {
                out[axis] = natural[t_axis] as u32;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test that init succeeds where a GPU is present. CI environments
    /// without a GPU will see this skipped via the `?` early return in the
    /// `match`.
    #[test]
    fn init_or_skip() {
        match WgpuExecutor::new() {
            Ok(e) => println!("wgpu adapter: {}", e.adapter_info),
            Err(WgpuError::NoAdapter) => println!("(skipped: no adapter on this host)"),
            Err(e) => panic!("wgpu init failed: {e}"),
        }
    }

    #[test]
    fn elementwise_add_matches_cpu() {
        let exec = match WgpuExecutor::new() {
            Ok(e) => e,
            Err(WgpuError::NoAdapter) => return,
            Err(e) => panic!("wgpu init failed: {e}"),
        };
        let a = HostTensor::from_f32(vec![5], &[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let b = HostTensor::from_f32(vec![5], &[10.0, 20.0, 30.0, 40.0, 50.0]).unwrap();
        let c = exec.elementwise_add_f32(&a, &b).unwrap();
        assert_eq!(c.as_f32().unwrap(), vec![11.0, 22.0, 33.0, 44.0, 55.0]);
    }

    #[test]
    fn elementwise_mul_matches_cpu() {
        let exec = match WgpuExecutor::new() {
            Ok(e) => e,
            Err(WgpuError::NoAdapter) => return,
            Err(e) => panic!("wgpu init failed: {e}"),
        };
        let a = HostTensor::from_f32(vec![3], &[1.0, 2.0, 3.0]).unwrap();
        let b = HostTensor::from_f32(vec![3], &[10.0, 20.0, 30.0]).unwrap();
        let c = exec.elementwise_f32(&a, &b, 2 /* mul */).unwrap();
        assert_eq!(c.as_f32().unwrap(), vec![10.0, 40.0, 90.0]);
    }

    #[test]
    fn softmax_last_axis_matches_cpu() {
        let exec = match WgpuExecutor::new() {
            Ok(e) => e,
            Err(WgpuError::NoAdapter) => return,
            Err(e) => panic!("wgpu init failed: {e}"),
        };
        // Two rows; each row should sum to 1; rows with same input agree.
        let x = HostTensor::from_f32(vec![2, 3], &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]).unwrap();
        let t = exec.softmax_last_axis_f32(&x).unwrap();
        let v = t.as_f32().unwrap();
        let s0: f32 = v[..3].iter().sum();
        let s1: f32 = v[3..].iter().sum();
        assert!((s0 - 1.0).abs() < 1e-5);
        assert!((s1 - 1.0).abs() < 1e-5);
        for i in 0..3 {
            assert!((v[i] - v[i + 3]).abs() < 1e-6);
        }
    }

    #[test]
    fn transpose_2d_matches_cpu() {
        let exec = match WgpuExecutor::new() {
            Ok(e) => e,
            Err(WgpuError::NoAdapter) => return,
            Err(e) => panic!("wgpu init failed: {e}"),
        };
        let x = HostTensor::from_f32(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let t = exec.transpose_2d_f32(&x).unwrap();
        assert_eq!(t.shape, vec![3, 2]);
        assert_eq!(t.as_f32().unwrap(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn relu_matches_cpu() {
        let exec = match WgpuExecutor::new() {
            Ok(e) => e,
            Err(WgpuError::NoAdapter) => return,
            Err(e) => panic!("wgpu init failed: {e}"),
        };
        let x = HostTensor::from_f32(vec![5], &[-1.0, 0.0, 1.0, -2.0, 3.0]).unwrap();
        let t = exec.activation_f32(&x, 0).unwrap();
        assert_eq!(t.as_f32().unwrap(), vec![0.0, 0.0, 1.0, 0.0, 3.0]);
    }

    #[test]
    fn sigmoid_at_zero_is_half() {
        let exec = match WgpuExecutor::new() {
            Ok(e) => e,
            Err(WgpuError::NoAdapter) => return,
            Err(e) => panic!("wgpu init failed: {e}"),
        };
        let x = HostTensor::from_f32(vec![1], &[0.0]).unwrap();
        let t = exec.activation_f32(&x, 1).unwrap();
        assert!((t.as_f32().unwrap()[0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn gelu_matches_cpu_within_tolerance() {
        let exec = match WgpuExecutor::new() {
            Ok(e) => e,
            Err(WgpuError::NoAdapter) => return,
            Err(e) => panic!("wgpu init failed: {e}"),
        };
        let x = HostTensor::from_f32(vec![3], &[0.5, 1.0, 2.0]).unwrap();
        let t = exec.activation_f32(&x, 3).unwrap();
        let v = t.as_f32().unwrap();
        // Monotone for positive inputs.
        assert!(v[0] < v[1] && v[1] < v[2]);
        // GeLU(0) ≈ 0; GeLU(1) ≈ 0.8413; GeLU(2) ≈ 1.9546 with tanh approximation.
        assert!((v[1] - 0.8413).abs() < 1e-2);
    }

    #[test]
    fn batched_matmul_matches_cpu() {
        let exec = match WgpuExecutor::new() {
            Ok(e) => e,
            Err(WgpuError::NoAdapter) => return,
            Err(e) => panic!("wgpu init failed: {e}"),
        };
        // Batch=2, identity/zero combo (same fixture as the CPU batched test).
        let a = HostTensor::from_f32(
            vec![2, 2, 3],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // batch 0
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, // batch 1
            ],
        )
        .unwrap();
        let b = HostTensor::from_f32(
            vec![2, 3, 2],
            &[
                7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // batch 0
                1.0, 0.0, 0.0, 1.0, 0.0, 0.0, // batch 1
            ],
        )
        .unwrap();
        let c = exec.matmul_batched_f32(&a, &b).unwrap();
        assert_eq!(c.shape, vec![2, 2, 2]);
        let v = c.as_f32().unwrap();
        for (got, exp) in v[..4].iter().zip(&[58.0_f32, 64.0, 139.0, 154.0]) {
            assert!((got - exp).abs() < 1e-3, "got {got}, expected {exp}");
        }
        for (got, exp) in v[4..].iter().zip(&[1.0_f32, 0.0, 0.0, 1.0]) {
            assert!((got - exp).abs() < 1e-3, "got {got}, expected {exp}");
        }
    }

    #[test]
    fn scalar_bcast_mul() {
        let exec = match WgpuExecutor::new() {
            Ok(e) => e,
            Err(WgpuError::NoAdapter) => return,
            Err(e) => panic!("wgpu init failed: {e}"),
        };
        // 0-D scalar × (3,3) tensor — this is the attention "scale" path.
        let s = HostTensor::from_f32(vec![], &[0.5]).unwrap();
        let t = HostTensor::from_f32(vec![3, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
            .unwrap();
        let r = exec
            .elementwise_scalar_bcast_f32(&s, &t, 2 /* mul */, true /* scalar_left */)
            .unwrap();
        assert_eq!(r.shape, vec![3, 3]);
        assert_eq!(
            r.as_f32().unwrap(),
            vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
        );
    }

    #[test]
    fn bcast_row_vector_over_matrix() {
        let exec = match WgpuExecutor::new() {
            Ok(e) => e,
            Err(WgpuError::NoAdapter) => return,
            Err(e) => panic!("wgpu init failed: {e}"),
        };
        // (3,) + (2, 3) → (2, 3) per NumPy broadcasting rules.
        let v = HostTensor::from_f32(vec![3], &[1.0, 2.0, 3.0]).unwrap();
        let m = HostTensor::from_f32(vec![2, 3], &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]).unwrap();
        let r = exec.elementwise_bcast_f32(&v, &m, 0 /* add */).unwrap();
        assert_eq!(r.shape, vec![2, 3]);
        assert_eq!(
            r.as_f32().unwrap(),
            vec![11.0, 22.0, 33.0, 41.0, 52.0, 63.0]
        );
    }

    #[test]
    fn bcast_column_vector_over_matrix() {
        let exec = match WgpuExecutor::new() {
            Ok(e) => e,
            Err(WgpuError::NoAdapter) => return,
            Err(e) => panic!("wgpu init failed: {e}"),
        };
        // (3, 1) + (3, 4) → (3, 4)
        let col = HostTensor::from_f32(vec![3, 1], &[1.0, 2.0, 3.0]).unwrap();
        let m = HostTensor::from_f32(
            vec![3, 4],
            &[
                10.0, 20.0, 30.0, 40.0, 100.0, 200.0, 300.0, 400.0, 1000.0, 2000.0, 3000.0, 4000.0,
            ],
        )
        .unwrap();
        let r = exec.elementwise_bcast_f32(&col, &m, 0 /* add */).unwrap();
        assert_eq!(r.shape, vec![3, 4]);
        assert_eq!(
            r.as_f32().unwrap(),
            vec![
                11.0, 21.0, 31.0, 41.0, 102.0, 202.0, 302.0, 402.0, 1003.0, 2003.0, 3003.0, 4003.0
            ]
        );
    }

    #[test]
    fn fused_matmul_scale_softmax_matches_attention() {
        let exec = match WgpuExecutor::new() {
            Ok(e) => e,
            Err(WgpuError::NoAdapter) => return,
            Err(e) => panic!("wgpu init failed: {e}"),
        };
        // Same fixture as the attention demo: Q = identity-ish, K = identity,
        // dim = 4 → scale = 0.5. Output rows should each be a softmax over
        // [0.5, 0, 0] (and rotations).
        let q = HostTensor::from_f32(
            vec![3, 4],
            &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        )
        .unwrap();
        // K^T (= K transposed): K is identity-rows, so K^T has columns of identity.
        let kt = HostTensor::from_f32(
            vec![4, 3],
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        )
        .unwrap();
        let r = exec.matmul_scale_softmax_f32(&q, &kt, 0.5).unwrap();
        assert_eq!(r.shape, vec![3, 3]);
        let v = r.as_f32().unwrap();
        // Each row is softmax([0.5, 0, 0]) (and rotations) — sums to 1.
        for row in v.chunks(3) {
            let s: f32 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "row sum {s} != 1");
        }
        // Diagonal entries are the max prob (~0.452).
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert!(
                        (v[i * 3 + j] - 0.4519).abs() < 1e-3,
                        "diag[{i}] = {} expected ~0.452",
                        v[i * 3 + j]
                    );
                } else {
                    assert!(
                        (v[i * 3 + j] - 0.2741).abs() < 1e-3,
                        "off[{i},{j}] = {} expected ~0.274",
                        v[i * 3 + j]
                    );
                }
            }
        }
    }

    #[test]
    fn matmul_2d_matches_cpu() {
        let exec = match WgpuExecutor::new() {
            Ok(e) => e,
            Err(WgpuError::NoAdapter) => return,
            Err(e) => panic!("wgpu init failed: {e}"),
        };
        // Same fixture as the CPU test: [[1,2,3],[4,5,6]] × [[7,8],[9,10],[11,12]] = [[58,64],[139,154]]
        let a = HostTensor::from_f32(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = HostTensor::from_f32(vec![3, 2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
        let c = exec.matmul_f32(&a, &b).unwrap();
        assert_eq!(c.shape, vec![2, 2]);
        let v = c.as_f32().unwrap();
        for (got, exp) in v.iter().zip(&[58.0_f32, 64.0, 139.0, 154.0]) {
            assert!((got - exp).abs() < 1e-3, "got {got}, expected {exp}");
        }
    }
}
