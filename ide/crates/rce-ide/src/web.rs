//! Browser build: the same photon view and wgpu renderer on a WebGPU canvas.
//!
//! The workspace services (rce-mcp, the sandbox, memory, meters) are native;
//! in the browser the panels show a model handed over by the host page
//! (`set_model` with the JSON of [`crate::model`] types), the editor's
//! diagnostics come from the in-process language server, and taps are
//! queued for the host (`take_actions`). Fonts must be supplied (`add_font`):
//! there are no system fonts to query. The editor is the same rope buffer as
//! on the desktop: the page forwards keys (`key`), text and IME composition
//! (`text`, `preedit`), the mouse (`mouse_down`/`mouse_move`/`mouse_up`) and
//! the wheel (`wheel`).
use crate::buffer::Buffer;
use crate::gpu::Renderer;
use crate::lsp::Lsp;
use crate::model::*;
use crate::{scene, view};
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

struct State {
    editor: EditorVm,
    proposals: Vec<Proposal>,
    receipts: ReceiptsVm,
    memory: MemoryVm,
    net: NetVm,
    lsp: Lsp,
    q: view::Queue,
    last: Option<photon::RenderCtx>,
    buffer: Buffer,
    saved: String,
    focus: Focus,
    dragging: bool,
}

impl State {
    fn edit(&mut self, op: EditOp) {
        if self.buffer.apply(op) {
            let uri = format!("file:///{}", self.editor.path);
            let t = self.buffer.text();
            self.editor.diagnostics = self.lsp.change(&uri, &t);
        }
        self.sync();
    }
    fn sync(&mut self) {
        let e = &mut self.editor;
        e.view = self.buffer.visible();
        e.text = self.buffer.text();
        e.dirty = e.text != self.saved;
        e.can_undo = self.buffer.can_undo();
        e.can_redo = self.buffer.can_redo();
    }
}

#[wasm_bindgen]
pub struct WebIde {
    st: Rc<RefCell<State>>,
    r: Renderer,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    m: view::Metrics,
    snap: Option<crate::gpu::Readback>,
    /// The backend in use (`BrowserWebGpu`, or `Gl` = WebGL2 fallback).
    backend: String,
}

#[wasm_bindgen]
impl WebIde {
    /// Attach to `<canvas id=canvas_id>` with WebGPU.
    pub async fn attach(canvas_id: String) -> Result<WebIde, JsValue> {
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
        // WebGPU first; WebGL2 where the browser has no WebGPU adapter.
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
        let st = State {
            editor: EditorVm::default(),
            proposals: vec![],
            receipts: ReceiptsVm::default(),
            memory: MemoryVm::default(),
            net: NetVm::default(),
            lsp: Lsp::in_process(),
            q: view::Queue::default(),
            last: None,
            buffer: Buffer::default(),
            saved: String::new(),
            focus: Focus::Editor,
            dragging: false,
        };
        Ok(WebIde {
            st: Rc::new(RefCell::new(st)),
            r,
            surface,
            config,
            m: view::Metrics::default(),
            snap: None,
            backend,
        })
    }

    pub fn add_font(&mut self, bytes: Vec<u8>) -> Option<String> {
        let f = self.r.add_font(bytes);
        self.m = self.r.metrics(13.0);
        f
    }

    /// Start rendering the current frame offscreen for a pixel readback
    /// (canvas screenshots of a WebGPU surface are not reliable headless).
    pub fn snapshot_begin(&mut self) {
        let (w, h) = (self.config.width, self.config.height);
        let sc = self.scene(w, h).0;
        self.snap = Some(self.r.readback_begin(&sc, w, h).0);
    }

    pub fn snapshot_ready(&self) -> bool {
        self.snap.as_ref().is_some_and(|s| s.ready())
    }

    /// RGBA8 pixels, `width * height * 4` bytes.
    pub fn snapshot_take(&mut self) -> Result<Vec<u8>, JsValue> {
        let s = self.snap.take().ok_or("no snapshot started")?;
        Ok(s.take()?)
    }

    pub fn backend(&self) -> String {
        self.backend.clone()
    }

    pub fn resize(&mut self, w: u32, h: u32) {
        self.config.width = w.max(1);
        self.config.height = h.max(1);
        self.surface.configure(&self.r.device, &self.config);
    }

    /// Open a file's text in the editor (diagnostics from the in-process server).
    pub fn set_source(&mut self, path: String, text: String) {
        let mut s = self.st.borrow_mut();
        let uri = format!("file:///{path}");
        let diagnostics = s.lsp.open(&uri, &text);
        let server = format!("{} (in-process)", s.lsp.server_name);
        s.buffer = Buffer::new(&text);
        s.saved = text.clone();
        s.editor = EditorVm {
            path,
            text,
            diagnostics,
            server,
            ..Default::default()
        };
        s.sync();
    }

    /// Replace the buffer (keeps the file's saved text for `dirty`); for
    /// demo state handed over from the native IDE.
    pub fn set_buffer(&mut self, text: String) {
        let mut s = self.st.borrow_mut();
        s.buffer.set_text(&text);
        let uri = format!("file:///{}", s.editor.path);
        s.editor.diagnostics = s.lsp.change(&uri, &text);
        s.sync();
    }

    /// An `EditOp` as JSON (e.g. `{"Move":{"to":"WordLeft","select":true}}`).
    pub fn edit_json(&mut self, json: String) -> Result<(), JsValue> {
        let op: EditOp = serde_json::from_str(&json).map_err(|e| e.to_string())?;
        self.st.borrow_mut().edit(op);
        Ok(())
    }

    /// A `KeyboardEvent.key` with modifiers. Returns true when handled (the
    /// page should then `preventDefault`).
    pub fn key(&mut self, key: String, shift: bool, ctrl: bool) -> bool {
        let mut s = self.st.borrow_mut();
        if s.focus == Focus::Memory {
            match key.as_str() {
                "Enter" => {
                    let q = s.memory.query.clone();
                    s.q.borrow_mut().push(Action::Recall(q));
                }
                "Backspace" => {
                    s.memory.query.pop();
                }
                "Escape" => s.focus = Focus::Editor,
                k if !ctrl && k.chars().count() == 1 => s.memory.query.push_str(k),
                _ => return false,
            }
            return true;
        }
        let mv = |to| EditOp::Move { to, select: shift };
        let op = match (key.as_str(), ctrl) {
            ("ArrowLeft", true) => mv(Motion::WordLeft),
            ("ArrowRight", true) => mv(Motion::WordRight),
            ("ArrowLeft", _) => mv(Motion::Left),
            ("ArrowRight", _) => mv(Motion::Right),
            ("ArrowUp", _) => mv(Motion::Up),
            ("ArrowDown", _) => mv(Motion::Down),
            ("Home", true) => mv(Motion::DocStart),
            ("End", true) => mv(Motion::DocEnd),
            ("Home", _) => mv(Motion::LineStart),
            ("End", _) => mv(Motion::LineEnd),
            ("PageUp", _) => mv(Motion::PageUp),
            ("PageDown", _) => mv(Motion::PageDown),
            ("Enter", _) => EditOp::Newline,
            ("Tab", _) => EditOp::Tab,
            ("Backspace", _) => EditOp::Backspace,
            ("Delete", _) => EditOp::Delete,
            ("z", true) | ("Z", true) if shift => EditOp::Redo,
            ("z", true) => EditOp::Undo,
            ("y", true) => EditOp::Redo,
            ("a", true) => EditOp::SelectAll,
            // Left to the browser: its copy/cut/paste events reach the
            // system clipboard (see `copy` and `paste`).
            ("c" | "x" | "v", true) => return false,
            ("s", true) => {
                s.q.borrow_mut().push(Action::Save);
                return true;
            }
            _ => return false,
        };
        s.edit(op);
        true
    }

    /// The browser's `copy`/`cut` event: the selected text for the system
    /// clipboard (`None`: nothing selected, leave it alone).
    pub fn copy(&mut self, cut: bool) -> Option<String> {
        let mut s = self.st.borrow_mut();
        let t = s.buffer.selected_text()?;
        s.edit(if cut { EditOp::Cut } else { EditOp::Copy });
        Some(t)
    }

    /// The browser's `paste` event, with the system clipboard's text.
    pub fn paste(&mut self, text: String) {
        let mut s = self.st.borrow_mut();
        if s.focus == Focus::Editor {
            s.edit(EditOp::Paste(Some(text)));
        } else {
            s.memory.query.push_str(&text);
        }
    }

    /// Typed text or an IME commit (`input` / `compositionend`).
    pub fn text(&mut self, t: String) {
        let mut s = self.st.borrow_mut();
        match s.focus {
            Focus::Editor => s.edit(EditOp::Insert(t)),
            Focus::Memory => s.memory.query.push_str(&t),
        }
    }

    /// IME composition in progress (`compositionupdate`).
    pub fn preedit(&mut self, t: String) {
        self.st.borrow_mut().edit(EditOp::Preedit(t));
    }

    pub fn wheel(&mut self, dy: f32) {
        let lines = (dy / self.m.line_h).round() as i64;
        self.st.borrow_mut().edit(EditOp::Scroll(lines));
    }

    pub fn mouse_down(&mut self, x: f32, y: f32, shift: bool) {
        self.click(x, y);
        let m = self.m;
        let mut s = self.st.borrow_mut();
        let hit = s
            .last
            .as_ref()
            .and_then(|c| view::editor_hit(c, &s.editor, &m, x, y));
        if let Some((line, col)) = hit {
            s.dragging = true;
            s.focus = Focus::Editor;
            s.edit(EditOp::ClickAt {
                line,
                col,
                extend: shift,
            });
        }
    }

    pub fn mouse_move(&mut self, x: f32, y: f32) {
        let m = self.m;
        let mut s = self.st.borrow_mut();
        if !s.dragging {
            return;
        }
        let hit = s
            .last
            .as_ref()
            .and_then(|c| view::editor_hit(c, &s.editor, &m, x, y));
        if let Some((line, col)) = hit {
            s.edit(EditOp::ClickAt {
                line,
                col,
                extend: true,
            });
        }
    }

    pub fn mouse_up(&mut self) {
        self.st.borrow_mut().dragging = false;
    }

    /// Editor state for the page (cursor, selection, dirty), as JSON.
    pub fn editor_json(&self) -> String {
        serde_json::to_string(&self.st.borrow().editor).unwrap_or_default()
    }

    /// Replace the right-hand panels: `{proposals, receipts, memory, net}` JSON.
    pub fn set_model(&mut self, json: String) -> Result<(), JsValue> {
        let v: serde_json::Value = serde_json::from_str(&json).map_err(|e| e.to_string())?;
        let mut s = self.st.borrow_mut();
        if let Ok(p) = serde_json::from_value(v["proposals"].clone()) {
            s.proposals = p;
        }
        if let Ok(r) = serde_json::from_value(v["receipts"].clone()) {
            s.receipts = r;
        }
        if let Ok(m) = serde_json::from_value(v["memory"].clone()) {
            s.memory = m;
        }
        if let Ok(n) = serde_json::from_value(v["net"].clone()) {
            s.net = n;
        }
        Ok(())
    }

    pub fn click(&mut self, x: f32, y: f32) -> bool {
        let hit = {
            let s = self.st.borrow();
            s.last.as_ref().is_some_and(|c| view::click(c, x, y))
        };
        // Focus changes are local; everything else goes to the host.
        let mut s = self.st.borrow_mut();
        let acts: Vec<Action> = s.q.borrow_mut().drain(..).collect();
        for a in acts {
            match a {
                Action::Focus(f) => s.focus = f,
                a => s.q.borrow_mut().push(a),
            }
        }
        hit
    }

    /// Queued actions as JSON (`[{"Accept":1}, ...]`), for the host to perform.
    pub fn take_actions(&mut self) -> String {
        let s = self.st.borrow();
        let a: Vec<Action> = s.q.borrow_mut().drain(..).collect();
        serde_json::to_string(&a).unwrap_or_default()
    }

    fn scene(&mut self, w: u32, h: u32) -> (scene::Scene, photon::RenderCtx) {
        let m = self.m;
        {
            let mut s = self.st.borrow_mut();
            let rows = view::editor_rows(h as f32, &m);
            if s.buffer.rows != rows {
                s.buffer.rows = rows;
                s.buffer.reveal();
                s.sync();
            }
        }
        {
            let s = self.st.borrow();
            let p = Panels {
                editor: &s.editor,
                proposals: &s.proposals,
                receipts: &s.receipts,
                memory: &s.memory,
                net: &s.net,
                focus: s.focus,
            };
            let ctx = view::paint(
                &view::root_with(&p, &s.q, &m, w as f32, h as f32),
                w as f32,
                h as f32,
            );
            (scene::build(&ctx, w as f32, h as f32), ctx)
        }
    }

    pub fn frame(&mut self) {
        let (w, h) = (self.config.width, self.config.height);
        let (sc, ctx) = self.scene(w, h);
        let tex = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            _ => {
                self.surface.configure(&self.r.device, &self.config);
                return;
            }
        };
        let v = tex.texture.create_view(&Default::default());
        self.r.draw(&sc, &v, w, h, [0.0, 0.0, 0.0, 1.0]);
        self.r.queue.present(tex);
        self.st.borrow_mut().last = Some(ctx);
    }
}
