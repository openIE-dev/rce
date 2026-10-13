//! wgpu + Parley renderer for a [`Scene`]: one pipeline for solid quads
//! and glyphs (alpha from an R8 atlas), scissor clipping per batch, and an
//! offscreen path that renders to a texture and reads it back.
use crate::scene::{R, Scene};
use parley::{FontContext, FontFamily, LayoutContext, PositionedLayoutItem, StyleProperty};
use std::collections::HashMap;
use swash::scale::{Render, ScaleContext, Source, StrikeWith};
use swash::zeno::Format;
use wgpu::util::DeviceExt;

const SHADER: &str = r#"
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32>, @location(1) color: vec4<f32>, @location(2) mode: f32 };
struct U { size: vec2<f32>, _pad: vec2<f32> };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var atlas: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@vertex fn vs(@location(0) p: vec2<f32>, @location(1) uv: vec2<f32>, @location(2) c: vec4<f32>, @location(3) m: f32) -> VOut {
  var o: VOut;
  o.pos = vec4<f32>(p.x / u.size.x * 2.0 - 1.0, 1.0 - p.y / u.size.y * 2.0, 0.0, 1.0);
  o.uv = uv; o.color = c; o.mode = m;
  return o;
}
@fragment fn fs(i: VOut) -> @location(0) vec4<f32> {
  // Sample in uniform control flow (WGSL requires it; browsers enforce it).
  let t = textureSample(atlas, samp, i.uv).r;
  let a = select(1.0, t, i.mode > 0.5);
  return vec4<f32>(i.color.rgb * i.color.a * a, i.color.a * a);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct V {
    p: [f32; 2],
    uv: [f32; 2],
    c: [f32; 4],
    m: f32,
}

const ATLAS: u32 = 1024;

struct Glyph {
    uv: [f32; 4],
    w: f32,
    h: f32,
    left: f32,
    top: f32,
}

struct Atlas {
    tex: wgpu::Texture,
    x: u32,
    y: u32,
    row: u32,
    cache: HashMap<(u64, u32, u32, u32), Option<Glyph>>,
}

pub struct Renderer {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub format: wgpu::TextureFormat,
    pub adapter_info: wgpu::AdapterInfo,
    pipeline: wgpu::RenderPipeline,
    bind: wgpu::BindGroup,
    ubuf: wgpu::Buffer,
    atlas: Atlas,
    fcx: FontContext,
    lcx: LayoutContext<[u8; 4]>,
    scx: ScaleContext,
    pub family: String,
}

/// Frame statistics.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Stats {
    pub quads: usize,
    pub glyphs: usize,
    pub batches: usize,
}

impl Renderer {
    /// A device with no surface (offscreen), or `Err` with why not.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn headless() -> Result<Self, String> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            force_fallback_adapter: false,
            compatible_surface: None,
            ..Default::default()
        }))
        .map_err(|e| format!("no GPU adapter: {e}"))?;
        // Photon colours are sRGB-encoded values; a non-sRGB target stores them as given.
        Self::with_adapter(&adapter, wgpu::TextureFormat::Rgba8Unorm)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn with_adapter(
        adapter: &wgpu::Adapter,
        format: wgpu::TextureFormat,
    ) -> Result<Self, String> {
        pollster::block_on(Self::new_async(adapter, format))
    }

    pub async fn new_async(
        adapter: &wgpu::Adapter,
        format: wgpu::TextureFormat,
    ) -> Result<Self, String> {
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("rce-ide"),
                required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                    .using_resolution(adapter.limits()),
                ..Default::default()
            })
            .await
            .map_err(|e| format!("device: {e}"))?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let ubuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("glyph atlas"),
            size: wgpu::Extent3d {
                width: ATLAS,
                height: ATLAS,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let view = tex.create_view(&Default::default());
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ubuf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bgl)],
            ..Default::default()
        });
        let attrs =
            wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Float32x4, 3 => Float32];
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("rce-ide"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<V>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &attrs,
                })],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        Ok(Renderer {
            device,
            queue,
            format,
            adapter_info: adapter.get_info(),
            pipeline,
            bind,
            ubuf,
            atlas: Atlas {
                tex,
                x: 1,
                y: 1,
                row: 0,
                cache: HashMap::new(),
            },
            fcx: FontContext::new(),
            lcx: LayoutContext::new(),
            scx: ScaleContext::new(),
            family: "monospace".into(),
        })
    }

    fn glyph(&mut self, font: &parley::FontData, gid: u32, size: f32, sub: u32) -> Option<Glyph> {
        let key = (
            font.data.id() ^ ((font.index as u64) << 56),
            gid,
            size.to_bits(),
            sub,
        );
        if let Some(g) = self.atlas.cache.get(&key) {
            return g.as_ref().map(|g| Glyph {
                uv: g.uv,
                w: g.w,
                h: g.h,
                left: g.left,
                top: g.top,
            });
        }
        let fr = swash::FontRef::from_index(font.data.as_ref(), font.index as usize)?;
        let mut sc = self.scx.builder(fr).size(size).hint(true).build();
        let img = Render::new(&[
            Source::ColorOutline(0),
            Source::Bitmap(StrikeWith::BestFit),
            Source::Outline,
        ])
        .format(Format::Alpha)
        .offset(swash::zeno::Vector::new(sub as f32 / 4.0, 0.0))
        .render(&mut sc, gid as u16);
        let out = img.and_then(|img| {
            let (w, h) = (img.placement.width, img.placement.height);
            if w == 0 || h == 0 || img.data.len() < (w * h) as usize {
                return None;
            }
            let a = &mut self.atlas;
            if a.x + w + 1 > ATLAS {
                a.x = 1;
                a.y += a.row + 1;
                a.row = 0;
            }
            if a.y + h + 1 > ATLAS {
                // Full: start over (the next frame re-rasterizes what it needs).
                a.cache.clear();
                a.x = 1;
                a.y = 1;
                a.row = 0;
            }
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &a.tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: a.x,
                        y: a.y,
                        z: 0,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                &img.data[..(w * h) as usize],
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(w),
                    rows_per_image: Some(h),
                },
                wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
            );
            let f = ATLAS as f32;
            let g = Glyph {
                uv: [
                    a.x as f32 / f,
                    a.y as f32 / f,
                    (a.x + w) as f32 / f,
                    (a.y + h) as f32 / f,
                ],
                w: w as f32,
                h: h as f32,
                left: img.placement.left as f32,
                top: img.placement.top as f32,
            };
            a.x += w + 1;
            a.row = a.row.max(h);
            Some(g)
        });
        let ret = out.as_ref().map(|g| Glyph {
            uv: g.uv,
            w: g.w,
            h: g.h,
            left: g.left,
            top: g.top,
        });
        self.atlas.cache.insert(key, out);
        ret
    }

    fn quad(v: &mut Vec<V>, r: R, uv: [f32; 4], c: [f32; 4], m: f32) {
        let (x0, y0, x1, y1) = (r.x, r.y, r.x + r.w, r.y + r.h);
        let p = |x, y, u, w| V {
            p: [x, y],
            uv: [u, w],
            c,
            m,
        };
        v.extend([
            p(x0, y0, uv[0], uv[1]),
            p(x1, y0, uv[2], uv[1]),
            p(x1, y1, uv[2], uv[3]),
            p(x0, y0, uv[0], uv[1]),
            p(x1, y1, uv[2], uv[3]),
            p(x0, y1, uv[0], uv[3]),
        ]);
    }

    /// Vertices grouped by clip rectangle, in paint order.
    fn tessellate(
        &mut self,
        s: &Scene,
        stats: &mut Stats,
    ) -> (Vec<V>, Vec<(R, std::ops::Range<u32>)>) {
        let mut v = vec![];
        let mut batches: Vec<(R, std::ops::Range<u32>)> = vec![];
        let mut push = |v: &mut Vec<V>, clip: R, start: usize| {
            let end = v.len() as u32;
            match batches.last_mut() {
                Some((c, r)) if *c == clip && r.end == start as u32 => r.end = end,
                _ => batches.push((clip, start as u32..end)),
            }
        };
        // Quads first then text would break paint order; walk both by order of appearance.
        let (mut qi, mut ti) = (0, 0);
        let order = order_of(s);
        for k in order {
            if k {
                let q = &s.quads[qi];
                qi += 1;
                let st = v.len();
                Self::quad(&mut v, q.rect, [0.0; 4], q.color, 0.0);
                stats.quads += 1;
                push(&mut v, q.clip, st);
            } else {
                let t = s.texts[ti].clone();
                ti += 1;
                let st = v.len();
                let fam = self.family.clone();
                let mut b = self.lcx.ranged_builder(&mut self.fcx, &t.text, 1.0, true);
                b.push_default(StyleProperty::FontFamily(FontFamily::Source(fam.into())));
                b.push_default(StyleProperty::FontSize(t.size));
                let mut layout = b.build(&t.text);
                layout.break_all_lines(None);
                let mut glyphs = vec![];
                for line in layout.lines() {
                    for item in line.items() {
                        if let PositionedLayoutItem::GlyphRun(gr) = item {
                            let run = gr.run();
                            let font = run.font().clone();
                            let size = run.font_size();
                            let mut x = gr.offset();
                            let y = gr.baseline();
                            for g in gr.glyphs() {
                                glyphs.push((font.clone(), g.id, size, x + g.x, y - g.y));
                                x += g.advance;
                            }
                        }
                    }
                }
                for (font, id, size, gx, gy) in glyphs {
                    let px = t.rect.x + gx;
                    let sub = ((px.fract() * 4.0) as u32).min(3);
                    if let Some(g) = self.glyph(&font, id, size, sub) {
                        let r = R {
                            x: px.floor() + g.left,
                            y: (t.rect.y + gy).round() - g.top,
                            w: g.w,
                            h: g.h,
                        };
                        Self::quad(&mut v, r, g.uv, t.color, 1.0);
                        stats.glyphs += 1;
                    }
                }
                push(&mut v, t.clip, st);
            }
        }
        stats.batches = batches.len();
        (v, batches)
    }

    /// Draw `scene` into `target` (cleared to `clear`).
    pub fn draw(
        &mut self,
        scene: &Scene,
        target: &wgpu::TextureView,
        w: u32,
        h: u32,
        clear: [f64; 4],
    ) -> Stats {
        self.draw_ops(scene, target, w, h, Some(clear))
    }

    /// Draw `scene` over what `target` already holds (labels over the 3D view).
    pub fn draw_over(&mut self, scene: &Scene, target: &wgpu::TextureView, w: u32, h: u32) -> Stats {
        self.draw_ops(scene, target, w, h, None)
    }

    fn draw_ops(
        &mut self,
        scene: &Scene,
        target: &wgpu::TextureView,
        w: u32,
        h: u32,
        clear: Option<[f64; 4]>,
    ) -> Stats {
        let mut stats = Stats::default();
        let (verts, batches) = self.tessellate(scene, &mut stats);
        self.queue.write_buffer(
            &self.ubuf,
            0,
            bytemuck::cast_slice(&[w as f32, h as f32, 0.0, 0.0]),
        );
        let vb = (!verts.is_empty()).then(|| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&verts),
                    usage: wgpu::BufferUsages::VERTEX,
                })
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: match clear {
                            Some(clear) => wgpu::LoadOp::Clear(wgpu::Color {
                                r: clear[0],
                                g: clear[1],
                                b: clear[2],
                                a: clear[3],
                            }),
                            None => wgpu::LoadOp::Load,
                        },
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            if let Some(vb) = &vb {
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.bind, &[]);
                pass.set_vertex_buffer(0, vb.slice(..));
                for (c, r) in &batches {
                    let c = c.intersect(R {
                        x: 0.0,
                        y: 0.0,
                        w: w as f32,
                        h: h as f32,
                    });
                    if c.is_empty() {
                        continue;
                    }
                    let (x, y) = (c.x.floor() as u32, c.y.floor() as u32);
                    let (cw, ch) = (
                        (c.w.ceil() as u32).min(w - x),
                        (c.h.ceil() as u32).min(h - y),
                    );
                    if cw == 0 || ch == 0 {
                        continue;
                    }
                    pass.set_scissor_rect(x, y, cw, ch);
                    pass.draw(r.clone(), 0..1);
                }
            }
        }
        self.queue.submit([enc.finish()]);
        stats
    }

    /// Width of `text` laid out in the UI font at `size` (for the editor's
    /// caret and selection geometry, and mouse hit-testing).
    pub fn measure(&mut self, text: &str, size: f32) -> f32 {
        let fam = self.family.clone();
        let mut b = self.lcx.ranged_builder(&mut self.fcx, text, 1.0, true);
        b.push_default(StyleProperty::FontFamily(FontFamily::Source(fam.into())));
        b.push_default(StyleProperty::FontSize(size));
        let mut layout = b.build(text);
        layout.break_all_lines(None);
        layout.width()
    }

    /// Editor metrics from the real font (monospace advance).
    pub fn metrics(&mut self, size: f32) -> crate::view::Metrics {
        let n = 64;
        crate::view::Metrics {
            char_w: self.measure(&"0".repeat(n), size) / n as f32,
            line_h: (size * 1.4).ceil(),
            size,
        }
    }

    /// Register font bytes (the browser has no system fonts to query).
    pub fn add_font(&mut self, data: Vec<u8>) -> Option<String> {
        let fams = self.fcx.collection.register_fonts(data.into(), None);
        let id = fams.first()?.0;
        let name = self.fcx.collection.family_name(id)?.to_string();
        self.family = name.clone();
        Some(name)
    }

    /// Draw `scene` into an offscreen texture and start copying it back.
    /// Poll [`Readback::ready`] (the browser resolves the map from its event
    /// loop), then [`Readback::take`].
    pub fn readback_begin(&mut self, scene: &Scene, w: u32, h: u32) -> (Readback, Stats) {
        let tex = self.offscreen_texture(w, h);
        let view = tex.create_view(&Default::default());
        let stats = self.draw(scene, &view, w, h, [0.0, 0.0, 0.0, 1.0]);
        (self.readback_texture(&tex, w, h), stats)
    }

    /// An offscreen colour target in the renderer's format (for hosts that
    /// compose several passes, e.g. the 3D graph view, before reading back).
    pub fn offscreen_texture(&self, w: u32, h: u32) -> wgpu::Texture {
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    }

    /// Start copying `tex` (from [`Self::offscreen_texture`]) back to the CPU.
    pub fn readback_texture(&self, tex: &wgpu::Texture, w: u32, h: u32) -> Readback {
        let row = (w * 4).div_ceil(256) * 256;
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (row * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([enc.finish()]);
        let state = std::sync::Arc::new(std::sync::atomic::AtomicU8::new(0));
        let st = state.clone();
        buf.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            st.store(
                if r.is_ok() { 1 } else { 2 },
                std::sync::atomic::Ordering::SeqCst,
            );
        });
        Readback {
            buf,
            row,
            w,
            h,
            state,
            bgra: matches!(
                self.format,
                wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
            ),
        }
    }

    /// Offscreen: render to an RGBA8 texture and read the pixels back.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn render_rgba(
        &mut self,
        scene: &Scene,
        w: u32,
        h: u32,
    ) -> Result<(Vec<u8>, Stats), String> {
        let (rb, stats) = self.readback_begin(scene, w, h);
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| format!("poll: {e}"))?;
        Ok((rb.take()?, stats))
    }
}

/// An offscreen frame being copied back to the CPU.
pub struct Readback {
    buf: wgpu::Buffer,
    row: u32,
    pub w: u32,
    pub h: u32,
    state: std::sync::Arc<std::sync::atomic::AtomicU8>,
    /// The target was BGRA (browser surfaces): swapped to RGBA on take.
    bgra: bool,
}

impl Readback {
    pub fn ready(&self) -> bool {
        self.state.load(std::sync::atomic::Ordering::SeqCst) != 0
    }

    /// Tightly packed RGBA rows.
    pub fn take(self) -> Result<Vec<u8>, String> {
        match self.state.load(std::sync::atomic::Ordering::SeqCst) {
            1 => {}
            0 => return Err("readback not ready".into()),
            _ => return Err("readback map failed".into()),
        }
        let slice = self.buf.slice(..);
        let data = slice
            .get_mapped_range()
            .map_err(|e| format!("map: {e:?}"))?;
        let mut out = Vec::with_capacity((self.w * self.h * 4) as usize);
        for y in 0..self.h {
            let s = (y * self.row) as usize;
            out.extend_from_slice(&data[s..s + (self.w * 4) as usize]);
        }
        drop(data);
        self.buf.unmap();
        if self.bgra {
            for p in out.as_chunks_mut::<4>().0 {
                p.swap(0, 2);
            }
        }
        Ok(out)
    }
}

/// Paint order of quads (`true`) and texts (`false`). The scene keeps them in
/// separate lists; each carries an index into the original command order.
fn order_of(s: &Scene) -> Vec<bool> {
    let mut v: Vec<(usize, bool)> = s
        .quads
        .iter()
        .map(|q| (q.order, true))
        .chain(s.texts.iter().map(|t| (t.order, false)))
        .collect();
    v.sort_by_key(|x| x.0);
    v.into_iter().map(|x| x.1).collect()
}
