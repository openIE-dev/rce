//! `rce-ide --root DIR --file FILE [--lsp CMD ARGS..] [--no-memory] [--history N]
//!          [--demo QUERY] [--export-model OUT.json] [--offscreen OUT.(ppm|png) --size WxH]`
//! `rce-ide telemetry probe|setup` (energy sources; see docs/TELEMETRY.md)
//!
//! Desktop: a winit window drawn with wgpu. `--offscreen` renders one frame
//! without a window and writes a PPM (no display needed).
#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
mod desktop {
    use rce_ide::gpu::Renderer;
    use rce_ide::view::{self, Metrics, Queue};
    use rce_ide::vm::{Action, EditOp, Focus, Ide, Motion};
    use rce_ide::{app, scene};
    use std::sync::Arc;
    use winit::application::ApplicationHandler;
    use winit::event::{ElementState, Ime, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
    use winit::event_loop::{ActiveEventLoop, EventLoop};
    use winit::keyboard::{Key, ModifiersState, NamedKey};
    use winit::window::{Window, WindowId};

    pub fn frame(
        ide: &mut Ide,
        q: &Queue,
        m: &Metrics,
        w: u32,
        h: u32,
    ) -> (scene::Scene, photon::RenderCtx) {
        ide.set_rows(view::editor_rows(h as f32, m));
        let root = view::root_with(&ide.panels(), q, m, w as f32, h as f32);
        let ctx = view::paint(&root, w as f32, h as f32);
        (scene::build(&ctx, w as f32, h as f32), ctx)
    }

    pub fn offscreen(ide: &mut Ide, out: &str, w: u32, h: u32) -> Result<(), String> {
        let q = Queue::default();
        let mut r = Renderer::headless()?;
        let m = r.metrics(13.0);
        let (s, _) = frame(ide, &q, &m, w, h);
        let t = std::time::Instant::now();
        let (px, st) = r.render_rgba(&s, w, h)?;
        eprintln!(
            "rce-ide: offscreen {w}x{h} on {} ({:?}): {} quads, {} glyphs, {} batches, {:?}",
            r.adapter_info.name,
            r.adapter_info.backend,
            st.quads,
            st.glyphs,
            st.batches,
            t.elapsed()
        );
        let mut ppm = format!("P6\n{w} {h}\n255\n").into_bytes();
        for p in px.chunks(4) {
            ppm.extend_from_slice(&p[..3]);
        }
        if out.ends_with(".png") {
            return rce_ide::png::write(out, w, h, &px);
        }
        std::fs::write(out, ppm).map_err(|e| e.to_string())
    }

    struct Gfx {
        window: Arc<Window>,
        surface: wgpu::Surface<'static>,
        config: wgpu::SurfaceConfiguration,
        r: Renderer,
        m: Metrics,
    }

    /// The 3D graph mode (F3 cycles: graph, graph + hardware, telemetry, off;
    /// F4 cycles the layout). It covers the right of the window; the editor
    /// stays on the left, linked: clicking a node moves the editor there, and
    /// the editor cursor highlights its node.
    struct G3 {
        mode: u8,
        view: rce_ide::graph_panel::GraphView,
        r3: rce_graph3d::render::Renderer3d,
        sampler: rce_graph3d::native::Sampler,
        orbit: Option<(f32, f32)>,
    }

    struct App {
        g3: Option<G3>,
        ide: Ide,
        q: Queue,
        gfx: Option<Gfx>,
        cursor: (f32, f32),
        last: Option<photon::RenderCtx>,
        err: Option<String>,
        mods: ModifiersState,
        /// Left button held in the editor (drag selects).
        dragging: bool,
        /// An IME composition is in progress: key text belongs to it and
        /// arrives as `Ime::Commit`.
        composing: bool,
        /// Earlier Form versions from `--history N` (git), for the scrubber.
        history: Vec<rce_graph3d::history::Program>,
    }

    impl ApplicationHandler for App {
        fn resumed(&mut self, el: &ActiveEventLoop) {
            if self.gfx.is_some() {
                return;
            }
            let window = Arc::new(
                el.create_window(
                    Window::default_attributes()
                        .with_title("RCE IDE")
                        .with_inner_size(winit::dpi::LogicalSize::new(1400, 900)),
                )
                .expect("window"),
            );
            let instance = wgpu::Instance::new(
                wgpu::InstanceDescriptor::new_without_display_handle_from_env(),
            );
            let surface = instance.create_surface(window.clone()).expect("surface");
            let adapter =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    compatible_surface: Some(&surface),
                    ..Default::default()
                }))
                .expect("adapter");
            let caps = surface.get_capabilities(&adapter);
            let format = caps
                .formats
                .iter()
                .copied()
                .find(|f| !f.is_srgb())
                .unwrap_or(caps.formats[0]);
            let r = Renderer::with_adapter(&adapter, format).expect("renderer");
            let size = window.inner_size();
            let config = wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format,
                width: size.width.max(1),
                height: size.height.max(1),
                present_mode: wgpu::PresentMode::AutoVsync,
                desired_maximum_frame_latency: 2,
                alpha_mode: caps.alpha_modes[0],
                view_formats: vec![],
                color_space: Default::default(),
            };
            surface.configure(&r.device, &config);
            window.set_ime_allowed(true);
            let mut r = r;
            let m = r.metrics(13.0);
            self.gfx = Some(Gfx {
                window,
                surface,
                config,
                r,
                m,
            });
        }

        fn window_event(&mut self, el: &ActiveEventLoop, _: WindowId, ev: WindowEvent) {
            match ev {
                WindowEvent::CloseRequested => el.exit(),
                WindowEvent::Resized(s) => {
                    if let Some(g) = &mut self.gfx {
                        g.config.width = s.width.max(1);
                        g.config.height = s.height.max(1);
                        g.surface.configure(&g.r.device, &g.config);
                        g.window.request_redraw();
                    }
                }
                WindowEvent::ModifiersChanged(m) => self.mods = m.state(),
                WindowEvent::CursorMoved { position, .. } => {
                    let sf = self.gfx.as_ref().map_or(1.0, |g| g.window.scale_factor()) as f32;
                    let (ox, oy) = self.cursor;
                    self.cursor = (position.x as f32 / sf, position.y as f32 / sf);
                    if let Some(g3) = &mut self.g3
                        && g3.orbit.is_some()
                    {
                        g3.view.cam.rotate(-(self.cursor.0 - ox) * 0.01, (self.cursor.1 - oy) * 0.01);
                        if let Some(g) = &self.gfx {
                            g.window.request_redraw();
                        }
                        return;
                    }
                    if self.dragging
                        && let Some((line, col)) = self.hit()
                    {
                        self.q.borrow_mut().push(Action::Edit(EditOp::ClickAt {
                            line,
                            col,
                            extend: true,
                        }));
                        self.drain();
                    }
                }
                WindowEvent::MouseInput {
                    state,
                    button: MouseButton::Left,
                    ..
                } => {
                    if state == ElementState::Released {
                        self.dragging = false;
                        let area = self.graph_area();
                        let op = match (&mut self.g3, area) {
                            (Some(g3), Some(a)) => {
                                g3.orbit = None;
                                g3.view.gesture_end(a, self.cursor.0, self.cursor.1)
                            }
                            (Some(g3), None) => { g3.orbit = None; None }
                            _ => None,
                        };
                        if let Some(op) = op {
                            self.graph_edit(op);
                        }
                        return;
                    }
                    if self.graph_click() {
                        self.drain();
                        return;
                    }
                    if let Some(ctx) = &self.last {
                        view::click(ctx, self.cursor.0, self.cursor.1);
                    }
                    if let Some((line, col)) = self.hit() {
                        self.dragging = true;
                        self.q.borrow_mut().push(Action::Edit(EditOp::ClickAt {
                            line,
                            col,
                            extend: self.mods.shift_key(),
                        }));
                    }
                    self.drain();
                }
                WindowEvent::MouseWheel { delta, .. } if self.in_graph() => {
                    let d = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y,
                        MouseScrollDelta::PixelDelta(p) => p.y as f32,
                    };
                    if let Some(g3) = &mut self.g3 {
                        g3.view.cam.zoom(if d > 0.0 { 1.0 / 1.1 } else { 1.1 });
                    }
                    self.drain();
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    let lines = match delta {
                        MouseScrollDelta::LineDelta(_, y) => -(y * 3.0),
                        MouseScrollDelta::PixelDelta(p) => -(p.y as f32 / 18.0),
                    };
                    self.q
                        .borrow_mut()
                        .push(Action::Edit(EditOp::Scroll(lines.round() as i64)));
                    self.drain();
                }
                WindowEvent::Ime(ime) => {
                    match ime {
                        Ime::Preedit(t, _) => {
                            self.composing = !t.is_empty();
                            if self.ide.focus == Focus::Editor {
                                self.q.borrow_mut().push(Action::Edit(EditOp::Preedit(t)))
                            }
                        }
                        Ime::Commit(t) => {
                            self.composing = false;
                            self.text(t)
                        }
                        _ => {}
                    }
                    self.drain();
                }
                WindowEvent::KeyboardInput {
                    event:
                        KeyEvent {
                            logical_key,
                            state: ElementState::Pressed,
                            text,
                            ..
                        },
                    ..
                } => {
                    if let Key::Named(NamedKey::F3) = logical_key {
                        self.cycle_graph();
                        self.drain();
                        return;
                    }
                    if let (Key::Named(NamedKey::F4), Some(g3)) = (&logical_key, &mut self.g3) {
                        use rce_graph3d::layout::Mode;
                        let m = match g3.view.opts.mode {
                            Mode::Force => Mode::ByKind,
                            Mode::ByKind => Mode::ByTime,
                            Mode::ByTime => Mode::Force,
                        };
                        g3.view.set_mode(m);
                        self.drain();
                        return;
                    }
                    if self.in_graph() && self.graph_key(&logical_key) {
                        self.drain();
                        return;
                    }
                    let handled = match self.ide.focus {
                        Focus::Editor => self.editor_key(&logical_key),
                        Focus::Memory => self.memory_key(&logical_key),
                    };
                    if !handled
                        && !self.composing
                        && !self.mods.control_key()
                        && !self.mods.super_key()
                        && let Some(t) = text
                        && !t.chars().any(char::is_control)
                    {
                        self.text(t.to_string());
                    }
                    self.drain();
                }
                WindowEvent::RedrawRequested => self.redraw(),
                _ => {}
            }
        }
    }

    impl App {
        /// The graph panel's area, in logical pixels.
        fn graph_area(&self) -> Option<scene::R> {
            let g = self.gfx.as_ref()?;
            self.g3.as_ref().filter(|x| x.mode > 0)?;
            let sf = g.window.scale_factor() as f32;
            let (w, h) = (g.config.width as f32 / sf, g.config.height as f32 / sf);
            Some(scene::R { x: (w * 0.42).round(), y: 0.0, w: w - (w * 0.42).round(), h })
        }

        fn in_graph(&self) -> bool {
            self.graph_area().is_some_and(|a| self.cursor.0 >= a.x)
        }

        fn cycle_graph(&mut self) {
            let mode = self.g3.as_ref().map_or(0, |g| g.mode);
            let next = (mode + 1) % 4;
            if next == 0 {
                if let Some(g3) = &mut self.g3 {
                    g3.mode = 0;
                }
                return;
            }
            let Some(g) = &self.gfx else { return };
            let mut sampler = match self.g3.take() {
                Some(x) => x.sampler,
                None => rce_graph3d::native::Sampler::open(),
            };
            sampler.tick();
            let tl = rce_ide::graph_panel::timeline_from_ide(&self.ide, &self.history, &sampler.rows, sampler.cpu_util);
            let mut view = rce_ide::graph_panel::GraphView::new(tl, next == 2);
            let receipts = ide_receipt::chain::read_jsonl(&self.ide.ledger_path).unwrap_or_default();
            view.set_reviews(&receipts, &self.ide.server.ws.program().into_iter().collect());
            let r3 = rce_graph3d::render::Renderer3d::new(&g.r.device, g.r.format);
            self.g3 = Some(G3 { mode: next, view, r3, sampler, orbit: None });
        }

        /// A press in the graph panel: pick (and jump the editor) or start orbiting.
        fn graph_click(&mut self) -> bool {
            let Some(a) = self.graph_area().filter(|_| self.in_graph()) else { return false };
            let Some(g3) = &mut self.g3 else { return false };
            if g3.mode == 3 {
                return true;
            }
            // Shift-press on a definition starts a move gesture.
            if self.mods.shift_key() && g3.view.gesture_start(a, self.cursor.0, self.cursor.1) {
                return true;
            }
            match g3.view.click(a, self.cursor.0, self.cursor.1) {
                Some((file, line)) => {
                    if let Err(e) = rce_ide::graph_panel::jump(&mut self.ide, &file, line) {
                        self.err = Some(e);
                    }
                }
                None => g3.orbit = Some(self.cursor),
            }
            true
        }

        /// Graph keys while the pointer is over the graph: F6/F7/F8 cycle the
        /// colour/size/glow metric, Esc folds expanded files, Delete removes
        /// the selected definition, N adds a function to the selected file.
        fn graph_key(&mut self, k: &Key) -> bool {
            use rce_ide::graph_panel::Channel;
            let Some(g3) = self.g3.as_mut().filter(|g| g.mode > 0 && g.mode < 3) else { return false };
            let op = match k {
                Key::Named(NamedKey::F6) => { g3.view.cycle_metric(Channel::Color); return true }
                Key::Named(NamedKey::F7) => { g3.view.cycle_metric(Channel::Size); return true }
                Key::Named(NamedKey::F8) => { g3.view.cycle_metric(Channel::Glow); return true }
                Key::Named(NamedKey::Escape) => { g3.view.collapse_all(); return true }
                Key::Named(NamedKey::Delete) => g3.view.delete_selected(),
                Key::Character(c) if c.as_str() == "n" && !self.mods.control_key() => {
                    let (g, _) = g3.view.shown();
                    let file = g3.view.style.selected.and_then(|i| g.nodes[i].file.clone()).unwrap_or_else(|| self.ide.editor.path.clone());
                    let names: std::collections::BTreeSet<String> = g.nodes.iter().map(|n| n.label.clone()).collect();
                    let name = (1..).map(|i| format!("new_fn_{i}")).find(|n| !names.contains(n)).unwrap_or_default();
                    Some(rce_ide::graph_edit::GraphOp::AddFn { file, name })
                }
                _ => return false,
            };
            if let Some(op) = op {
                self.graph_edit(op);
            }
            true
        }

        /// Apply a graph edit through `rce check` and rebuild the graph.
        fn graph_edit(&mut self, op: rce_ide::graph_edit::GraphOp) {
            match rce_ide::graph_panel::apply_gesture(&mut self.ide, &op) {
                Ok(w) => {
                    eprintln!("rce-ide: graph edit {op:?}: wrote {w:?}");
                    if let Some(g3) = &mut self.g3 {
                        g3.mode -= 1;
                    }
                    self.cycle_graph();
                }
                Err(e) => self.err = Some(format!("graph edit not performed: {e}")),
            }
        }

        fn hit(&self) -> Option<(usize, usize)> {
            let (g, ctx) = (self.gfx.as_ref()?, self.last.as_ref()?);
            view::editor_hit(ctx, &self.ide.editor, &g.m, self.cursor.0, self.cursor.1)
        }

        fn text(&mut self, t: String) {
            match self.ide.focus {
                Focus::Editor => self.q.borrow_mut().push(Action::Edit(EditOp::Insert(t))),
                Focus::Memory => self.ide.memory.query.push_str(&t),
            }
        }

        fn memory_key(&mut self, k: &Key) -> bool {
            match k {
                Key::Named(NamedKey::Enter) => self
                    .q
                    .borrow_mut()
                    .push(Action::Recall(self.ide.memory.query.clone())),
                Key::Named(NamedKey::Backspace) => {
                    self.ide.memory.query.pop();
                }
                Key::Named(NamedKey::Escape) => {
                    self.q.borrow_mut().push(Action::Focus(Focus::Editor))
                }
                _ => return false,
            }
            true
        }

        /// Editor shortcuts; returns false for plain text keys.
        fn editor_key(&mut self, k: &Key) -> bool {
            let ctrl = self.mods.control_key() || self.mods.super_key();
            let select = self.mods.shift_key();
            let mv = |to| EditOp::Move { to, select };
            let op = match k {
                Key::Named(n) => match n {
                    NamedKey::ArrowLeft if ctrl => mv(Motion::WordLeft),
                    NamedKey::ArrowRight if ctrl => mv(Motion::WordRight),
                    NamedKey::ArrowLeft => mv(Motion::Left),
                    NamedKey::ArrowRight => mv(Motion::Right),
                    NamedKey::ArrowUp => mv(Motion::Up),
                    NamedKey::ArrowDown => mv(Motion::Down),
                    NamedKey::Home if ctrl => mv(Motion::DocStart),
                    NamedKey::End if ctrl => mv(Motion::DocEnd),
                    NamedKey::Home => mv(Motion::LineStart),
                    NamedKey::End => mv(Motion::LineEnd),
                    NamedKey::PageUp => mv(Motion::PageUp),
                    NamedKey::PageDown => mv(Motion::PageDown),
                    NamedKey::Enter => EditOp::Newline,
                    NamedKey::Tab => EditOp::Tab,
                    NamedKey::Backspace => EditOp::Backspace,
                    NamedKey::Delete => EditOp::Delete,
                    _ => return false,
                },
                Key::Character(c) if ctrl => match c.to_lowercase().as_str() {
                    "z" if select => EditOp::Redo,
                    "z" => EditOp::Undo,
                    "y" => EditOp::Redo,
                    "a" => EditOp::SelectAll,
                    "c" => EditOp::Copy,
                    "x" => EditOp::Cut,
                    "v" => EditOp::Paste(None),
                    "s" => {
                        self.q.borrow_mut().push(Action::Save);
                        return true;
                    }
                    _ => return true,
                },
                _ => return false,
            };
            self.q.borrow_mut().push(Action::Edit(op));
            true
        }

        fn drain(&mut self) {
            let acts: Vec<Action> = self.q.borrow_mut().drain(..).collect();
            for a in acts {
                if let Err(e) = self.ide.apply(a) {
                    self.err = Some(e);
                }
            }
            if let Some(g) = &self.gfx {
                g.window.request_redraw();
            }
        }

        fn redraw(&mut self) {
            let Some(g) = &mut self.gfx else { return };
            let sf = g.window.scale_factor() as f32;
            let (pw, ph) = (g.config.width, g.config.height);
            let (lw, lh) = ((pw as f32 / sf) as u32, (ph as f32 / sf) as u32);
            let m = g.m;
            let (mut s, ctx) = frame(&mut self.ide, &self.q, &m, lw, lh);
            if let Some(a) = ctx
                .annotations
                .iter()
                .find(|a| a.label.as_deref() == Some("editor-text"))
            {
                // Keep the IME candidate window at the caret.
                let v = &self.ide.editor.view;
                let x = a.rect.x + 5.0 * m.char_w + 8.0 + v.cursor.1 as f32 * m.char_w;
                let y = a.rect.y + (v.cursor.0.saturating_sub(v.first) + 1) as f32 * m.line_h;
                g.window.set_ime_cursor_area(
                    winit::dpi::LogicalPosition::new(x, y),
                    winit::dpi::LogicalSize::new(m.char_w, m.line_h),
                );
            }
            scale(&mut s, sf);
            let tex = match g.surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(t)
                | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
                _ => {
                    g.surface.configure(&g.r.device, &g.config);
                    return;
                }
            };
            let view = tex.texture.create_view(&Default::default());
            g.r.draw(&s, &view, pw, ph, [0.0, 0.0, 0.0, 1.0]);
            if let Some(g3) = self.g3.as_mut().filter(|x| x.mode > 0) {
                let x0 = (pw as f32 * 0.42).round();
                let area = scene::R { x: x0, y: 0.0, w: pw as f32 - x0, h: ph as f32 };
                let v = &self.ide.editor.view;
                g3.view.set_cursor(&self.ide.editor.path, v.cursor.0 + 1);
                if g3.mode == 3 {
                    g3.sampler.tick();
                    let sc = rce_ide::graph_panel::dashboard_scene(&g3.sampler.rows, area, pw, ph, "live; F3: next view");
                    g.r.draw_over(&sc, &view, pw, ph);
                    g.window.request_redraw();
                } else {
                    let t = g3.sampler_t();
                    g3.view.draw(&mut g.r, &mut g3.r3, &view, pw, ph, area, t, None);
                }
            }
            g.r.queue.present(tex);
            self.last = Some(ctx);
        }
    }

    impl G3 {
        fn sampler_t(&self) -> f32 {
            (rce_memory_now() % 100_000) as f32 / 1000.0
        }
    }

    fn rce_memory_now() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }

    fn scale(s: &mut scene::Scene, f: f32) {
        let sc = |r: &mut scene::R| {
            r.x *= f;
            r.y *= f;
            r.w *= f;
            r.h *= f;
        };
        for q in &mut s.quads {
            sc(&mut q.rect);
            sc(&mut q.clip);
        }
        for t in &mut s.texts {
            sc(&mut t.rect);
            sc(&mut t.clip);
            t.size *= f;
        }
    }

    pub fn main() -> Result<(), String> {
        let mut a = std::env::args().skip(1);
        let (mut root, mut file, mut lsp, mut memory, mut off, mut size) =
            (None, None, None, true, None, (1400u32, 900u32));
        let mut demo: Option<String> = None;
        let mut history_n: Option<usize> = None;
        let mut export: Option<String> = None;
        while let Some(x) = a.next() {
            match x.as_str() {
                "--root" => root = a.next(),
                "--file" => file = a.next(),
                "--lsp" => lsp = Some(a.by_ref().collect::<Vec<_>>()),
                "--no-memory" => memory = false,
                "--history" => {
                    history_n = Some(a.next().and_then(|n| n.parse().ok()).ok_or("--history N (commits)")?)
                }
                "--demo" => demo = Some(a.next().ok_or("--demo QUERY")?),
                "--export-model" => export = Some(a.next().ok_or("--export-model OUT.json")?),
                "--offscreen" => off = a.next(),
                "--size" => {
                    let s = a.next().ok_or("--size WxH")?;
                    let (w, h) = s.split_once('x').ok_or("--size WxH")?;
                    size = (
                        w.parse().map_err(|_| "width")?,
                        h.parse().map_err(|_| "height")?,
                    );
                }
                _ => return Err(format!("unknown argument `{x}`")),
            }
        }
        let o = app::Options {
            root: root.ok_or("--root DIR")?.into(),
            file: file.ok_or("--file FILE")?,
            lsp,
            memory,
        };
        let mut ide = app::open(&o)?;
        if let Some(demo) = &demo {
            rce_ide::demo::populate(&mut ide, demo)?;
        }
        if let Some(out) = &export {
            let j = serde_json::to_string_pretty(&rce_ide::demo::export(&ide))
                .map_err(|e| e.to_string())?;
            std::fs::write(out, j).map_err(|e| format!("{out}: {e}"))?;
            if off.is_none() {
                return Ok(());
            }
        }
        if let Some(out) = off {
            return offscreen(&mut ide, &out, size.0, size.1);
        }
        let history = match history_n {
            Some(n) => {
                let h = rce_graph3d::history::programs_from_git(&ide.server.ws.root, n)?;
                eprintln!("rce-ide: history: {} commits that changed .rce files", h.len());
                h
            }
            None => vec![],
        };
        let el = EventLoop::new().map_err(|e| e.to_string())?;
        // After the event loop: the display connection exists by now.
        let clip = rce_ide::clip::System::new();
        eprintln!(
            "rce-ide: clipboard: {}",
            rce_ide::clip::Clipboard::name(&clip)
        );
        ide.clip = Box::new(clip);
        let mut app = App {
            g3: None,
            ide,
            q: Queue::default(),
            gfx: None,
            cursor: (0.0, 0.0),
            last: None,
            err: None,
            mods: ModifiersState::empty(),
            dragging: false,
            composing: false,
            history,
        };
        el.run_app(&mut app).map_err(|e| e.to_string())?;
        if let Some(e) = app.err {
            eprintln!("rce-ide: last error: {e}");
        }
        Ok(())
    }
}

const USAGE: &str = "usage:
  rce-ide --root DIR --file FILE [--lsp CMD...] [--no-memory] [--history N]
          [--demo QUERY] [--export-model OUT.json] [--offscreen OUT.png] [--size WxH]
  rce-ide test ...                 run the IDE checks without a window
  rce-ide telemetry probe|setup|calibrate|export ...
  rce-ide --version | -V           print the release
  rce-ide --help | -h              print this text";

fn main() {
    // `rce-ide telemetry probe|setup`: energy sources, no window needed.
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") && args.first().map(String::as_str) != Some("telemetry") {
        println!("{USAGE}");
        std::process::exit(0);
    }
    if matches!(args.first().map(String::as_str), Some("--version" | "-V")) && args.len() == 1 {
        println!("rce-ide {}", env!("CARGO_PKG_VERSION"));
        std::process::exit(0);
    }
    if args.first().map(String::as_str) == Some("telemetry")
        && args.get(1).map(String::as_str) == Some("calibrate")
    {
        std::process::exit(rce_ide::calibrate::command(&args[2..]));
    }
    if args.first().map(String::as_str) == Some("test") {
        std::process::exit(rce_ide::ci::test(&args[1..], &mut std::io::stdout()));
    }
    if args.first().map(String::as_str) == Some("telemetry")
        && args.get(1).map(String::as_str) == Some("export")
    {
        match rce_ide::ci::export_perfetto(&args[2..]) {
            Ok(o) => { println!("wrote {o} (synthetic E_sim)"); std::process::exit(0) }
            Err(e) => { eprintln!("rce-ide telemetry export: {e}"); std::process::exit(2) }
        }
    }
    if args.first().map(String::as_str) == Some("telemetry") {
        let code = ide_energy::cli::telemetry(&args[1..]);
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        if args.get(1).map(String::as_str) == Some("probe") {
            println!("{}", rce_ide::gpu_launch::probe());
        }
        std::process::exit(code);
    }
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    if let Err(e) = desktop::main() {
        eprintln!("rce-ide: {e}");
        std::process::exit(2);
    }
    #[cfg(any(not(feature = "gpu"), target_arch = "wasm32"))]
    eprintln!("rce-ide was built without the `gpu` feature; only the view models are available");
}
