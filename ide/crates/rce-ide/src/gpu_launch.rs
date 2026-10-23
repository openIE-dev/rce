//! A real GPU launch source for `ide_energy::gpu_launch` attribution.
//!
//! Best available backend, chosen at open:
//! 1. `wgpu-timestamps`: the adapter supports `TIMESTAMP_QUERY`; every
//!    recorded compute pass writes begin/end GPU timestamps, resolved and
//!    scaled by the queue's timestamp period to nanoseconds. Device clock
//!    values are anchored to the host clock at the first resolve (one
//!    offset), so they line up with energy samples to within submit latency.
//! 2. `host-timed`: no timestamp queries. Each recorded launch is timed on
//!    the CPU from submit to completion (`poll` wait). Coarser: it includes
//!    queue and driver latency, and is labelled as such.
//! 3. No adapter at all: `open_best` returns `Err` with the reason; the
//!    caller reports "no GPU launch source" and attributes nothing.
//!
//! This covers work the IDE itself submits through wgpu (Vulkan, Metal,
//! DX12, GL). Launches by other processes (CUDA kernels, for example)
//! need vendor profilers (CUPTI, rocprofiler, Metal counters), which link C
//! libraries and are out of scope for a pure-Rust build.
use ide_energy::gpu_launch::{Launch, LaunchSource};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode { Timestamps, HostTimed }

pub struct WgpuLaunchSource {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub mode: Mode,
    pub adapter: String,
    qs: Option<wgpu::QuerySet>,
    period_ns: f64,
    anchor: Option<i128>,
    out: Vec<Launch>,
}

const SLOTS: u32 = 2;

fn host_ns() -> u64 {
    ide_energy::meter::now_ns()
}

impl WgpuLaunchSource {
    /// Open the best backend on the highest-performance adapter.
    pub fn open_best() -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            ..Default::default()
        }))
        .map_err(|e| format!("no GPU adapter: {e}"))?;
        let info = adapter.get_info();
        let ts = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("rce-ide gpu launch source"),
            required_features: if ts { wgpu::Features::TIMESTAMP_QUERY } else { wgpu::Features::empty() },
            required_limits: wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits()),
            ..Default::default()
        }))
        .map_err(|e| format!("device: {e}"))?;
        let qs = ts.then(|| device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("launch timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: SLOTS,
        }));
        let period_ns = queue.get_timestamp_period() as f64;
        Ok(WgpuLaunchSource {
            device,
            queue,
            mode: if ts { Mode::Timestamps } else { Mode::HostTimed },
            adapter: format!("{} ({:?}, {:?})", info.name, info.backend, info.device_type),
            qs,
            period_ns,
            anchor: None,
            out: vec![],
        })
    }

    pub fn name(&self) -> &'static str {
        match self.mode { Mode::Timestamps => "wgpu-timestamps", Mode::HostTimed => "host-timed (includes submit latency)" }
    }

    /// Record one launch: `encode` builds the work into the encoder; when
    /// the backend has timestamps it receives the writes to attach to its
    /// compute pass. Blocks until the GPU finishes (attribution needs the
    /// end time).
    pub fn record(
        &mut self,
        kernel: &str,
        caller: &str,
        encode: impl FnOnce(&mut wgpu::CommandEncoder, Option<wgpu::ComputePassTimestampWrites<'_>>),
    ) -> Result<(), String> {
        let mut enc = self.device.create_command_encoder(&Default::default());
        let tw = self.qs.as_ref().map(|q| wgpu::ComputePassTimestampWrites {
            query_set: q,
            beginning_of_pass_write_index: Some(0),
            end_of_pass_write_index: Some(1),
        });
        encode(&mut enc, tw);
        let read = self.qs.as_ref().map(|q| {
            let size = 8 * SLOTS as u64;
            let resolve = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None, size, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false,
            });
            let rb = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None, size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
            });
            enc.resolve_query_set(q, 0..SLOTS, &resolve, 0);
            enc.copy_buffer_to_buffer(&resolve, 0, &rb, 0, size);
            rb
        });
        let t0 = host_ns();
        self.queue.submit([enc.finish()]);
        self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| format!("poll: {e}"))?;
        let t1 = host_ns();
        let (start_ns, end_ns) = match read {
            None => (t0, t1.max(t0 + 1)),
            Some(rb) => {
                rb.slice(..).map_async(wgpu::MapMode::Read, |_| {});
                self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| format!("poll: {e}"))?;
                let v: Vec<u64> = {
                    let view = rb.slice(..).get_mapped_range().map_err(|e| format!("map: {e:?}"))?;
                    view.as_chunks::<8>().0.iter().map(|c| u64::from_le_bytes(*c)).collect()
                };
                rb.unmap();
                let (a, b) = ((v[0] as f64 * self.period_ns) as i128, (v[1] as f64 * self.period_ns) as i128);
                let off = *self.anchor.get_or_insert(t0 as i128 - a);
                let s = (a + off).max(0) as u64;
                (s, ((b + off).max(0) as u64).max(s + 1))
            }
        };
        self.out.push(Launch { kernel: kernel.into(), caller: caller.into(), start_ns, end_ns });
        Ok(())
    }
}

impl LaunchSource for WgpuLaunchSource {
    fn drain(&mut self) -> Result<Vec<Launch>, String> {
        Ok(std::mem::take(&mut self.out))
    }
}

/// Describe the launch source this machine would use, for `telemetry probe`.
pub fn probe() -> String {
    match WgpuLaunchSource::open_best() {
        Ok(s) => format!("gpu launch source: {} on {}", s.name(), s.adapter),
        Err(e) => format!("gpu launch source: none ({e}); GPU energy stays unattributed"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const SHADER: &str = "@group(0) @binding(0) var<storage, read_write> d: array<u32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    var x = id.x; for (var i = 0u; i < 2000u; i++) { x = x * 1664525u + 1013904223u; } d[id.x] = x; }";

    #[test]
    fn records_real_launches_or_says_why_not() {
        let mut src = match WgpuLaunchSource::open_best() {
            Ok(s) => s,
            Err(e) => {
                assert!(e.starts_with("no GPU adapter") || e.starts_with("device"), "{e}");
                eprintln!("skipped: {e}");
                return;
            }
        };
        let dev = src.device.clone();
        let m = dev.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(SHADER.into()) });
        let p = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None, layout: None, module: &m, entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        });
        let buf = dev.create_buffer(&wgpu::BufferDescriptor { label: None, size: 4 * 4096, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
        let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &p.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() }],
        });
        for caller in ["train", "infer"] {
            src.record("lcg", caller, |enc, tw| {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: tw });
                pass.set_pipeline(&p);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(64, 1, 1);
            }).unwrap();
        }
        let l = src.drain().unwrap();
        eprintln!("{}: {:?}", src.name(), l);
        assert_eq!(l.len(), 2);
        assert!(l.iter().all(|x| x.end_ns > x.start_ns));
        assert!(src.drain().unwrap().is_empty());
    }
}
