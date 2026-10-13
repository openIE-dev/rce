//! Browser build of the 3D graph mode: the same rce-graph3d view, renderer
//! and Parley text on a WebGPU (or WebGL2) canvas. The native IDE exports
//! the timeline and telemetry as JSON (`examples/graph3d_shots.rs` writes
//! `graph3d-model.json`); the page hands it to [`WebGraph3d::load`].
use crate::gpu::{Readback, Renderer};
use crate::graph_panel::{self, GraphView};
use crate::scene::R;
use rce_graph3d::dash::SourceRow;
use rce_graph3d::render::Renderer3d;
use rce_graph3d::time::Timeline;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct WebGraph3d {
    r: Renderer,
    r3: Renderer3d,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    view: Option<GraphView>,
    rows: Vec<SourceRow>,
    /// `graph`, `hardware` or `telemetry`.
    mode: String,
    tl: Option<Timeline>,
    snap: Option<Readback>,
    backend: String,
    picked: Option<(String, usize)>,
}

#[wasm_bindgen]
impl WebGraph3d {
    pub async fn attach(canvas_id: String) -> Result<WebGraph3d, JsValue> {
        use wasm_bindgen::JsCast;
        let doc = web_sys::window()
            .and_then(|w| w.document())
            .ok_or("no document")?;
        let canvas: web_sys::HtmlCanvasElement = doc
            .get_element_by_id(&canvas_id)
            .ok_or("no canvas")?
            .dyn_into()
            .map_err(|_| "not a canvas")?;
        let (w, h) = (canvas.width().max(1), canvas.height().max(1));
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL;
        let instance = wgpu::Instance::new(desc);
        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
            .map_err(|e| e.to_string())?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .map_err(|e| format!("WebGPU/WebGL2 adapter: {e}"))?;
        let backend = format!("{:?}", adapter.get_info().backend);
        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let r = Renderer::new_async(&adapter, format).await?;
        r.device
            .on_uncaptured_error(std::sync::Arc::new(|e: wgpu::Error| {
                web_sys::console::error_1(&format!("wgpu: {e}").into())
            }));
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: w,
            height: h,
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            color_space: Default::default(),
        };
        surface.configure(&r.device, &config);
        let r3 = Renderer3d::new(&r.device, format);
        Ok(WebGraph3d {
            r,
            r3,
            surface,
            config,
            view: None,
            rows: vec![],
            mode: "graph".into(),
            tl: None,
            snap: None,
            backend,
            picked: None,
        })
    }

    pub fn add_font(&mut self, bytes: Vec<u8>) -> Option<String> {
        self.r.add_font(bytes)
    }

    pub fn backend(&self) -> String {
        self.backend.clone()
    }

    /// `{"timeline": Timeline, "telemetry": [SourceRow]}` from the native side.
    pub fn load(&mut self, json: String) -> Result<(), JsValue> {
        let v: serde_json::Value = serde_json::from_str(&json).map_err(|e| e.to_string())?;
        let tl: Timeline =
            serde_json::from_value(v["timeline"].clone()).map_err(|e| format!("timeline: {e}"))?;
        self.rows = serde_json::from_value(v["telemetry"].clone()).unwrap_or_default();
        self.tl = Some(tl);
        self.set_mode("graph".into());
        Ok(())
    }

    /// `{"divergence":bool,"mapping":{"size":"joules",...},"semantic_zoom":bool,"minimap":bool}`
    pub fn set_style(&mut self, json: String) -> Result<(), JsValue> {
        let v: serde_json::Value = serde_json::from_str(&json).map_err(|e| e.to_string())?;
        if let Some(gv) = &mut self.view {
            if let Some(b) = v["divergence"].as_bool() { gv.style.divergence = b }
            if !v["mapping"].is_null() {
                gv.style.mapping = serde_json::from_value(v["mapping"].clone()).map_err(|e| e.to_string())?;
            }
            if let Some(b) = v["semantic_zoom"].as_bool() { gv.semantic_zoom = b }
            if let Some(b) = v["minimap"].as_bool() { gv.minimap = b }
        }
        Ok(())
    }

    pub fn set_mode(&mut self, mode: String) {
        if let Some(tl) = &self.tl {
            let mut gv = GraphView::new(tl.clone(), mode == "hardware");
            gv.note = "energy: SYNTHETIC demo numbers".into();
            gv.k_sigma = 2.0;
            self.view = Some(gv);
        }
        self.mode = mode;
    }

    fn area(&self) -> R {
        R {
            x: 0.0,
            y: 0.0,
            w: self.config.width as f32,
            h: self.config.height as f32,
        }
    }

    /// A click: returns `"file:line"` for the editor when a code node was picked.
    pub fn click(&mut self, x: f32, y: f32) -> Option<String> {
        let a = self.area();
        let p = self.view.as_mut()?.click(a, x, y);
        self.picked = p.clone();
        p.map(|(f, l)| format!("{f}:{l}"))
    }

    /// Metric picker: `colour`, `size` or `glow` advances to the next metric.
    pub fn cycle_metric(&mut self, channel: String) {
        use crate::graph_panel::Channel;
        let c = match channel.as_str() { "size" => Channel::Size, "glow" => Channel::Glow, _ => Channel::Color };
        if let Some(v) = &mut self.view { v.cycle_metric(c) }
    }

    /// Fold every file expanded by a click under semantic zoom.
    pub fn collapse(&mut self) {
        if let Some(v) = &mut self.view { v.collapse_all() }
    }

    pub fn drag(&mut self, dx: f32, dy: f32) {
        if let Some(v) = &mut self.view {
            v.cam.rotate(-dx * 0.01, dy * 0.01);
        }
    }

    pub fn wheel(&mut self, d: f32) {
        if let Some(v) = &mut self.view {
            v.cam.zoom(if d > 0.0 { 1.1 } else { 1.0 / 1.1 });
        }
    }

    pub fn scrub(&mut self, u: f64) {
        if let Some(v) = &mut self.view {
            v.scrub(u);
        }
    }

    pub fn set_cursor(&mut self, file: String, line: usize) {
        if let Some(v) = &mut self.view {
            v.set_cursor(&file, line);
        }
    }

    /// Screen position of the node labelled `label` (for tests and tours).
    pub fn node_screen(&self, label: String) -> Option<Vec<f32>> {
        let v = self.view.as_ref()?;
        let n = v.snap.graph.nodes.iter().find(|n| n.label == label)?;
        let (x, y, _) = v
            .cam
            .project(GraphView::viewport(self.area()), *v.pos.get(&n.id)?)?;
        Some(vec![x, y])
    }

    pub fn status(&self) -> String {
        let v = self.view.as_ref();
        serde_json::json!({
            "backend": self.backend,
            "mode": self.mode,
            "nodes": v.map(|v| v.snap.graph.nodes.len()),
            "edges": v.map(|v| v.snap.graph.edges.len()),
            "version": v.and_then(|v| v.snap.version),
            "selected": v.and_then(|v| v.selected_info.first().cloned()),
            "cursor_node": v.and_then(|v| v.style.cursor.map(|i| v.snap.graph.nodes[i].label.clone())),
            "picked": self.picked.as_ref().map(|(f, l)| format!("{f}:{l}")),
            "sources": self.rows.len(),
        })
        .to_string()
    }

    fn draw_into(&mut self, target: &wgpu::TextureView, t: f32) {
        let (w, h) = (self.config.width, self.config.height);
        let a = self.area();
        if self.mode == "telemetry" {
            let sc = graph_panel::dashboard_scene(
                &self.rows,
                a,
                w,
                h,
                "rows marked SYNTHETIC are demo series",
            );
            self.r.draw(&sc, target, w, h, [0.0, 0.0, 0.0, 1.0]);
        } else if let Some(v) = &self.view {
            v.draw(
                &mut self.r,
                &mut self.r3,
                target,
                w,
                h,
                a,
                t,
                Some([0.0, 0.0, 0.0, 1.0]),
            );
        }
    }

    pub fn frame(&mut self, t: f32) {
        let tex = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            _ => {
                self.surface.configure(&self.r.device, &self.config);
                return;
            }
        };
        let v = tex.texture.create_view(&Default::default());
        self.draw_into(&v, t);
        self.r.queue.present(tex);
    }

    pub fn snapshot_begin(&mut self) {
        let (w, h) = (self.config.width, self.config.height);
        let tex = self.r.offscreen_texture(w, h);
        let v = tex.create_view(&Default::default());
        self.draw_into(&v, 0.0);
        self.snap = Some(self.r.readback_texture(&tex, w, h));
    }

    pub fn snapshot_ready(&self) -> bool {
        self.snap.as_ref().is_some_and(|s| s.ready())
    }

    pub fn snapshot_take(&mut self) -> Result<Vec<u8>, JsValue> {
        let s = self.snap.take().ok_or("no snapshot started")?;
        Ok(s.take()?)
    }
}
