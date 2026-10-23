//! wgpu renderer: instanced edges (screen-space thick lines) and nodes
//! (sphere impostors with glow, estimate ring, flash and highlight), with a
//! depth buffer. Draws into any colour target over what is there (load),
//! inside a viewport, so the host composes it with its 2D UI and labels.
//! Works on native backends and on browser WebGPU / WebGL2.
use crate::camera::{Orbit, Viewport};
use crate::math;
use crate::scene3d::{EdgeInst, Frame, NodeInst};
use wgpu::util::DeviceExt;

const SHADER: &str = r#"
struct U { vp: mat4x4<f32>, right: vec4<f32>, up: vec4<f32>, size: vec4<f32> };
@group(0) @binding(0) var<uniform> u: U;

fn corner(i: u32) -> vec2<f32> {
  var c = array<vec2<f32>, 6>(vec2(-1.,-1.), vec2(1.,-1.), vec2(1.,1.), vec2(-1.,-1.), vec2(1.,1.), vec2(-1.,1.));
  return c[i];
}

struct EOut { @builtin(position) pos: vec4<f32>, @location(0) color: vec4<f32>, @location(1) side: f32 };
@vertex fn vs_edge(@builtin(vertex_index) vi: u32, @location(0) a: vec3<f32>, @location(1) width: f32, @location(2) b: vec3<f32>, @location(3) color: vec4<f32>) -> EOut {
  let c = corner(vi);
  let pa = u.vp * vec4(a, 1.0);
  let pb = u.vp * vec4(b, 1.0);
  let t = (c.x + 1.0) * 0.5;
  let p = mix(pa, pb, t);
  let sa = pa.xy / max(pa.w, 1e-4) * u.size.xy;
  let sb = pb.xy / max(pb.w, 1e-4) * u.size.xy;
  var d = sb - sa;
  if (length(d) < 1e-4) { d = vec2(1.0, 0.0); }
  let n = normalize(vec2(-d.y, d.x));
  var o: EOut;
  o.pos = vec4(p.xy + n * c.y * width / u.size.xy * p.w, p.z, p.w);
  o.color = color; o.side = c.y;
  return o;
}
@fragment fn fs_edge(i: EOut) -> @location(0) vec4<f32> {
  let a = i.color.a * (1.0 - smoothstep(0.6, 1.0, abs(i.side)));
  return vec4(i.color.rgb * a, a);
}

struct NOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32>, @location(1) color: vec4<f32>, @location(2) flags: vec4<f32> };
@vertex fn vs_node(@builtin(vertex_index) vi: u32, @location(0) p: vec3<f32>, @location(1) r: f32, @location(2) color: vec4<f32>, @location(3) flags: vec4<f32>) -> NOut {
  let c = corner(vi) * 1.6; // room for the glow
  let w = p + (u.right.xyz * c.x + u.up.xyz * c.y) * r;
  var o: NOut;
  o.pos = u.vp * vec4(w, 1.0);
  o.uv = c; o.color = color; o.flags = flags;
  return o;
}
@fragment fn fs_node(i: NOut) -> @location(0) vec4<f32> {
  let d = length(i.uv);
  let pulse = 0.5 + 0.5 * sin(u.size.z * 6.0);
  var col = vec4(0.0);
  if (d <= 1.0) {
    let z = sqrt(max(0.0, 1.0 - d * d));
    let n = vec3(i.uv, z);
    let l = normalize(vec3(-0.4, 0.6, 0.7));
    let lit = 0.35 + 0.65 * max(dot(n, l), 0.0) + 0.25 * pow(max(dot(n, normalize(l + vec3(0., 0., 1.))), 0.0), 24.0);
    var rgb = i.color.rgb * lit;
    // Estimate-only (E_sim): a darker core with a bright rim.
    if (i.flags.y > 0.5 && d < 0.55) { rgb = rgb * 0.45; }
    col = vec4(rgb, 1.0);
  } else {
    // Glow by energy; flash red when |delta| > k sigma; highlight ring.
    let g = i.flags.x * exp(-(d - 1.0) * 5.0) * 0.8;
    col = vec4(i.color.rgb * g, g);
    if (i.flags.z > 0.0 && d < 1.35) { let f = i.flags.z * (0.55 + 0.45 * pulse); col = vec4(1.0 * f, 0.15 * f, 0.15 * f, f); }
    if (i.flags.w > 0.0 && d > 1.12 && d < 1.3) { let h = i.flags.w; col = vec4(h * select(1.0, 0.4, h < 0.9), h, h * select(1.0, 0.5, h < 0.9), h); }
    if (col.a < 0.01) { discard; }
  }
  return col;
}
"#;

pub struct Renderer3d {
    edge: wgpu::RenderPipeline,
    node: wgpu::RenderPipeline,
    bind: wgpu::BindGroup,
    ubuf: wgpu::Buffer,
    depth: Option<(u32, u32, wgpu::TextureView)>,
}

pub const DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth24Plus;

impl Renderer3d {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("graph3d"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let ubuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("graph3d uniforms"),
            size: 112,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: ubuf.as_entire_binding(),
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let blend = wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING;
        let mk = |vs: &str, fs: &str, attrs: &[wgpu::VertexAttribute], stride: u64, write: bool| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(vs),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some(vs),
                    compilation_options: Default::default(),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: stride,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: attrs,
                    })],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: Default::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH,
                    depth_write_enabled: Some(write),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let edge = mk(
            "vs_edge",
            "fs_edge",
            &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32,
                    offset: 12,
                    shader_location: 1,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 16,
                    shader_location: 2,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 32,
                    shader_location: 3,
                },
            ],
            std::mem::size_of::<EdgeInst>() as u64,
            false,
        );
        let node = mk(
            "vs_node",
            "fs_node",
            &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32, 2 => Float32x4, 3 => Float32x4],
            std::mem::size_of::<NodeInst>() as u64,
            true,
        );
        Renderer3d {
            edge,
            node,
            bind,
            ubuf,
            depth: None,
        }
    }

    /// Draw `frame` into `target` (size `tw` x `th`) inside `vp`, over the
    /// existing contents. `t` is seconds, for the anomaly pulse.
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target: &wgpu::TextureView,
        tw: u32,
        th: u32,
        vp: Viewport,
        cam: &Orbit,
        frame: &Frame,
        t: f32,
    ) {
        if self.depth.as_ref().is_none_or(|d| d.0 != tw || d.1 != th) {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("graph3d depth"),
                size: wgpu::Extent3d {
                    width: tw,
                    height: th,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DEPTH,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            self.depth = Some((tw, th, tex.create_view(&Default::default())));
        }
        let aspect = vp.w / vp.h.max(1.0);
        let vpm = cam.view_proj(aspect);
        let v = math::look_at(cam.eye(), cam.target, [0.0, 1.0, 0.0]);
        let mut u: Vec<f32> = vpm.iter().flatten().copied().collect();
        u.extend([
            v[0][0],
            v[1][0],
            v[2][0],
            0.0,
            v[0][1],
            v[1][1],
            v[2][1],
            0.0,
            vp.w / 2.0,
            vp.h / 2.0,
            t,
            0.0,
        ]);
        queue.write_buffer(&self.ubuf, 0, bytemuck::cast_slice(&u));
        let eb = (!frame.edges.is_empty()).then(|| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&frame.edges),
                usage: wgpu::BufferUsages::VERTEX,
            })
        });
        let nb = (!frame.nodes.is_empty()).then(|| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&frame.nodes),
                usage: wgpu::BufferUsages::VERTEX,
            })
        });
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("graph3d"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth.as_ref().unwrap().2,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            let x = vp.x.max(0.0);
            let y = vp.y.max(0.0);
            let w = vp.w.min(tw as f32 - x).max(1.0);
            let h = vp.h.min(th as f32 - y).max(1.0);
            pass.set_viewport(x, y, w, h, 0.0, 1.0);
            pass.set_scissor_rect(x as u32, y as u32, w as u32, h as u32);
            pass.set_bind_group(0, &self.bind, &[]);
            if let Some(eb) = &eb {
                pass.set_pipeline(&self.edge);
                pass.set_vertex_buffer(0, eb.slice(..));
                pass.draw(0..6, 0..frame.edges.len() as u32);
            }
            if let Some(nb) = &nb {
                pass.set_pipeline(&self.node);
                pass.set_vertex_buffer(0, nb.slice(..));
                pass.draw(0..6, 0..frame.nodes.len() as u32);
            }
        }
        queue.submit([enc.finish()]);
    }
}
