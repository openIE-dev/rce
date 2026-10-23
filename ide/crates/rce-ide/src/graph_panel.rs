//! The 3D graph mode and the telemetry dashboard (rce-graph3d), composed
//! with the 2D UI: the 2D pass draws the panel background, the 3D pass
//! draws instanced nodes and edges into the panel's viewport, and a last
//! 2D pass draws labels, legend, the time scrubber and the selection card
//! with the same Parley/swash text as the rest of the IDE.
//!
//! Linked views: [`GraphView::click`] picks a node and returns its file and
//! line for the editor; [`GraphView::set_cursor`] highlights the node under
//! the editor cursor.
//!
//! Interaction: under semantic zoom a click on a folded file expands it in
//! place; the metric chips under the header pick the metric for colour,
//! size and glow; review badges mark nodes with a human review that is
//! still valid (green) or expired by a later change (orange). Graph-edit
//! gestures (drag a definition onto another file to move it, Delete to
//! remove it, `N` to add a function) become [`crate::graph_edit::GraphOp`]s
//! that are written only if `rce check` admits the result.
use crate::gpu::Renderer;
use crate::scene::{Quad, R, Scene, TextRun};
use rce_graph3d::camera::{self, Orbit, Viewport};
use rce_graph3d::dash::{self, Prim, SourceRow};
use rce_graph3d::layout::{self, Mode, Options, Positions};
use rce_graph3d::render::Renderer3d;
use rce_graph3d::scene3d::{self, Frame, Style};
use rce_graph3d::time::{Snapshot, Timeline};
use rce_graph3d::lens::Metric;
use rce_graph3d::Graph;

pub struct GraphView {
    pub tl: Timeline,
    pub snap: Snapshot,
    pub pos: Positions,
    pub cam: Orbit,
    pub style: Style,
    pub opts: Options,
    /// Scrubber position, 0..1.
    pub u: f64,
    pub k_sigma: f64,
    pub show_hw: bool,
    pub title: String,
    /// Shown in the header, e.g. "energy: SYNTHETIC demo numbers".
    pub note: String,
    pub selected_info: Vec<String>,
    /// Fold definitions into files when the camera is far (ExplorViz).
    pub semantic_zoom: bool,
    pub minimap: bool,
    /// Files opened in place while the rest stays folded.
    pub expanded: std::collections::BTreeSet<String>,
    /// Review status per node id: true valid, false expired.
    pub reviews: std::collections::BTreeMap<String, bool>,
    /// Node index (in the shown graph) a gesture drag started on.
    pub drag_from: Option<usize>,
}

/// A visual channel the metric picker drives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channel { Color, Size, Glow }

/// Picker order; `None` (default look) comes first.
pub const METRICS: [Option<Metric>; 9] = [
    None, Some(Metric::Joules), Some(Metric::Measured), Some(Metric::Simulated), Some(Metric::Divergence),
    Some(Metric::Actions), Some(Metric::Watts), Some(Metric::Degree), Some(Metric::Rank),
];

pub fn metric_name(m: Option<Metric>) -> &'static str {
    match m {
        None => "default",
        Some(Metric::Rank) => "rank",
        Some(Metric::Joules) => "joules",
        Some(Metric::Measured) => "E_meas",
        Some(Metric::Simulated) => "E_sim",
        Some(Metric::Divergence) => "divergence",
        Some(Metric::Actions) => "actions",
        Some(Metric::Watts) => "watts",
        Some(Metric::Degree) => "degree",
    }
}

const CHIP_W: f32 = 130.0;
const CHIP_H: f32 = 18.0;

const SCRUB_H: f32 = 64.0;

impl GraphView {
    pub fn new(mut tl: Timeline, show_hw: bool) -> GraphView {
        tl.sort();
        let mut g = GraphView {
            snap: tl.at(i64::MIN, 3.0),
            tl,
            pos: Positions::new(),
            cam: Orbit::default(),
            style: Style {
                energy: true,
                max_labels: 40,
                ..Default::default()
            },
            opts: Options::default(),
            u: 1.0,
            k_sigma: 3.0,
            show_hw,
            title: "Form graph".into(),
            note: String::new(),
            selected_info: vec![],
            semantic_zoom: true,
            minimap: true,
            expanded: Default::default(),
            reviews: Default::default(),
            drag_from: None,
        };
        // Lay out every version in order, each warm-started from the last,
        // so scrubbing replays a stable picture.
        let n = g.tl.versions.len();
        for i in 0..n {
            let t = g.tl.versions[i].t_ms;
            g.goto(t);
        }
        g.scrub(1.0);
        g.cam.fit(&g.pos);
        g
    }

    fn goto(&mut self, t: i64) {
        let mut s = self.tl.at(t, self.k_sigma);
        if !self.show_hw {
            let keep: Vec<bool> = s.graph.nodes.iter().map(|n| !n.kind.is_hw()).collect();
            let mut map = vec![usize::MAX; keep.len()];
            let mut nodes = vec![];
            for (i, n) in s.graph.nodes.iter().enumerate() {
                if keep[i] {
                    map[i] = nodes.len();
                    nodes.push(n.clone());
                }
            }
            s.graph.edges = s
                .graph
                .edges
                .iter()
                .filter(|e| keep[e.a] && keep[e.b])
                .map(|e| rce_graph3d::Edge {
                    a: map[e.a],
                    b: map[e.b],
                    ..e.clone()
                })
                .collect();
            s.graph.nodes = nodes;
        }
        let fs = self.tl.first_seen();
        self.pos = layout::layout(&s.graph, &self.opts, &self.pos, &fs);
        self.style.added = s.added.clone();
        self.snap = s;
    }

    /// Move the scrubber (0..1) and replay the graph at that time.
    pub fn scrub(&mut self, u: f64) {
        self.u = u.clamp(0.0, 1.0);
        if let Some(t) = self.tl.scrub(self.u) {
            self.goto(t);
        }
    }

    pub fn set_mode(&mut self, m: Mode) {
        self.opts.mode = m;
        self.pos.clear();
        let u = self.u;
        self.scrub(u);
        self.cam.fit(&self.pos);
    }

    pub fn viewport(area: R) -> Viewport {
        Viewport {
            x: area.x,
            y: area.y + 36.0,
            w: area.w,
            h: (area.h - 36.0 - SCRUB_H).max(1.0),
        }
    }

    /// A click in `area`: the scrubber, or a node. Returns the picked node's
    /// (file, 1-based line) for the editor to jump to.
    pub fn click(&mut self, area: R, x: f32, y: f32) -> Option<(String, usize)> {
        if y >= area.y + area.h - SCRUB_H && x >= area.x + 16.0 {
            self.scrub(((x - area.x - 16.0) / (area.w - 32.0)) as f64);
            return None;
        }
        if let Some(c) = self.picker_hit(area, x, y) {
            self.cycle_metric(c);
            return None;
        }
        let (g, p) = self.shown();
        let hit = camera::pick(&g, &p, &self.cam, Self::viewport(area), x, y);
        if let Some(f) = hit.and_then(|i| rce_graph3d::lens::folded_file(&g.nodes[i])) {
            self.expanded.insert(f.to_string());
            self.style.selected = None;
            self.selected_info = vec![format!("{f}: expanded (Esc folds it again)")];
            return None;
        }
        self.style.selected = hit;
        self.selected_info = hit.map(|i| self.describe_in(&g, i)).unwrap_or_default();
        let n = &g.nodes[hit?];
        Some((n.file.clone()?, n.line?))
    }

    /// The graph as drawn: folded by file when zoomed out (minus the
    /// expanded files), else every definition.
    pub fn shown(&self) -> (Graph, Positions) {
        if self.folded() {
            rce_graph3d::lens::fold_files_except(&self.snap.graph, &self.pos, &self.expanded)
        } else {
            (self.snap.graph.clone(), self.pos.clone())
        }
    }

    /// Fold every expanded file again.
    pub fn collapse_all(&mut self) {
        self.expanded.clear();
    }

    fn chip(area: R, k: usize) -> R {
        R { x: area.x + 12.0 + k as f32 * (CHIP_W + 6.0), y: area.y + 40.0, w: CHIP_W, h: CHIP_H }
    }

    /// Which metric chip (colour, size, glow) is under (x, y).
    pub fn picker_hit(&self, area: R, x: f32, y: f32) -> Option<Channel> {
        [Channel::Color, Channel::Size, Channel::Glow].into_iter().enumerate().find_map(|(k, c)| {
            let r = Self::chip(area, k);
            (x >= r.x && x < r.x + r.w && y >= r.y && y < r.y + r.h).then_some(c)
        })
    }

    pub fn metric(&self, c: Channel) -> Option<Metric> {
        let m = &self.style.mapping;
        match c { Channel::Color => m.color, Channel::Size => m.size, Channel::Glow => m.glow }
    }

    /// Advance a channel to the next metric in [`METRICS`].
    pub fn cycle_metric(&mut self, c: Channel) {
        let cur = self.metric(c);
        let i = METRICS.iter().position(|m| *m == cur).unwrap_or(0);
        let next = METRICS[(i + 1) % METRICS.len()];
        let m = &mut self.style.mapping;
        match c { Channel::Color => m.color = next, Channel::Size => m.size = next, Channel::Glow => m.glow = next }
    }

    /// Start a gesture on the node under (x, y), if any.
    pub fn gesture_start(&mut self, area: R, x: f32, y: f32) -> bool {
        let (g, p) = self.shown();
        self.drag_from = camera::pick(&g, &p, &self.cam, Self::viewport(area), x, y)
            .filter(|&i| g.nodes[i].file.is_some() && rce_graph3d::lens::folded_file(&g.nodes[i]).is_none());
        self.drag_from.is_some()
    }

    /// End a gesture: dropping a definition on a node of another file moves
    /// it into that file.
    pub fn gesture_end(&mut self, area: R, x: f32, y: f32) -> Option<crate::graph_edit::GraphOp> {
        let from = self.drag_from.take()?;
        let (g, p) = self.shown();
        let to = camera::pick(&g, &p, &self.cam, Self::viewport(area), x, y)?;
        drop_op(&g, from, to)
    }

    /// Delete key on the selected definition.
    pub fn delete_selected(&self) -> Option<crate::graph_edit::GraphOp> {
        let (g, _) = self.shown();
        let n = &g.nodes[self.style.selected?];
        (n.file.is_some() && rce_graph3d::lens::folded_file(n).is_none() && !n.kind.is_hw())
            .then(|| crate::graph_edit::GraphOp::Delete { name: n.label.clone() })
    }

    /// Review status per node from review receipts (`ide_receipt::review`)
    /// against the current text of each item.
    pub fn set_reviews(&mut self, receipts: &[ide_receipt::Receipt], files: &crate::graph_edit::Files) {
        self.reviews = review_badges(receipts, files);
    }

    /// The editor cursor moved: highlight the node defined at or above it.
    pub fn set_cursor(&mut self, file: &str, line: usize) {
        self.style.cursor = self.snap.graph.node_at(file, line);
    }

    pub fn select_label(&mut self, label: &str) {
        self.style.selected = self.snap.graph.nodes.iter().position(|n| n.label == label);
        self.selected_info = self
            .style
            .selected
            .map(|i| self.describe(i))
            .unwrap_or_default();
    }

    fn describe(&self, i: usize) -> Vec<String> {
        self.describe_in(&self.snap.graph, i)
    }

    fn describe_in(&self, g: &Graph, i: usize) -> Vec<String> {
        let n = &g.nodes[i];
        let mut v = vec![format!("{}  ({})", n.label, n.kind.as_str())];
        if let (Some(f), Some(l)) = (&n.file, n.line) {
            v.push(format!("{f}:{l}"));
        }
        if let Some((_, note)) = n.detail.split_once(" · ").filter(|_| !n.kind.is_hw()) {
            v.push(format!("concurrency: {note}"));
        }
        if !n.detail.is_empty() && n.kind.is_hw() {
            v.push(rce_graph3d::graph::trunc(&n.detail, 44));
        }
        let e = &n.energy;
        match e.joules {
            Some(j) => v.push(format!(
                "{j:.3} J over {} actions{}",
                e.actions,
                if e.synthetic { " (SYNTHETIC demo)" } else { "" }
            )),
            None if e.actions > 0 => v.push(format!("{} actions, no energy recorded", e.actions)),
            None => {}
        }
        if e.joules.is_some() {
            v.push(format!(
                "E_meas {:.3} J · E_sim-only {:.3} J",
                e.measured_j, e.simulated_j
            ));
        }
        if e.anomaly {
            v.push(format!("|delta| > {} sigma", self.k_sigma));
        }
        if let Some(l) = &n.live {
            v.push(match l.watts {
                Some(w) => format!("live {w:.2} W via {}", l.source.clone().unwrap_or_default()),
                None => format!("live: {}", rce_graph3d::graph::trunc(&l.status, 40)),
            });
            if let Some(u) = l.util {
                v.push(format!("utilisation {:.0} %", u * 100.0));
            }
        }
        match self.reviews.get(&n.id) {
            Some(true) => v.push("reviewed (still valid)".into()),
            Some(false) => v.push("review expired: changed since review".into()),
            None => {}
        }
        let ins = g.edges.iter().filter(|x| x.b == i).count();
        let outs = g.edges.iter().filter(|x| x.a == i).count();
        v.push(format!("{outs} out, {ins} in"));
        for x in g
            .edges
            .iter()
            .filter(|x| x.a == i && x.kind == rce_graph3d::EdgeKind::Ran)
        {
            v.push(format!("ran on {}: {:.2} J", g.nodes[x.b].label, x.weight));
        }
        if let Some(fs) = n.file.as_ref().and(self.snap.facts.get(n.label.as_str())) {
            for f in fs {
                v.push(format!("fact: {f}"));
            }
        }
        v
    }

    fn extent(&self) -> f32 {
        self.pos.values().map(|p| (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt()).fold(1.0, f32::max)
    }

    /// True when the view currently shows files rather than definitions.
    pub fn folded(&self) -> bool {
        self.semantic_zoom
            && rce_graph3d::lens::detail(&self.cam, self.extent()) == rce_graph3d::lens::Detail::Files
    }

    pub fn frame(&self, area: R) -> Frame {
        if self.folded() {
            let (g, p) = self.shown();
            return scene3d::build(&g, &p, &self.cam, Self::viewport(area), &self.style);
        }
        scene3d::build(
            &self.snap.graph,
            &self.pos,
            &self.cam,
            Self::viewport(area),
            &self.style,
        )
    }

    /// 2D under the 3D pass: the panel background.
    pub fn under(&self, area: R) -> Vec<Prim> {
        vec![dash::rect(
            area.x,
            area.y,
            area.w,
            area.h,
            [0.06, 0.065, 0.08, 1.0],
        )]
    }

    /// 2D over the 3D pass: header, labels, legend, scrubber, selection card.
    pub fn over(&self, area: R, f: &Frame) -> Vec<Prim> {
        let mut p = vec![dash::rect(area.x, area.y, area.w, 36.0, dash::BG)];
        let ver = self
            .snap
            .version
            .map(|i| {
                format!(
                    "{} · form {}",
                    self.tl.versions[i].label,
                    self.tl.versions[i].graph.form_hash.get(..10).unwrap_or("")
                )
            })
            .unwrap_or_else(|| "before the first version".into());
        p.push(dash::text(
            area.x + 12.0,
            area.y + 9.0,
            15.0,
            format!("{} · {ver} · layout {:?}", self.title, self.opts.mode),
            dash::FG,
        ));
        if !self.note.is_empty() {
            p.push(dash::text(
                area.x + area.w - 12.0 - self.note.len() as f32 * 6.6,
                area.y + 11.0,
                12.0,
                self.note.clone(),
                dash::WARN,
            ));
        }
        for l in &f.labels {
            p.push(dash::text(l.x, l.y, 11.0, l.text.clone(), l.color));
        }
        // Metric picker chips.
        for (k, c) in [Channel::Color, Channel::Size, Channel::Glow].into_iter().enumerate() {
            let r = Self::chip(area, k);
            let on = self.metric(c).is_some();
            p.push(dash::rect(r.x, r.y, r.w, r.h, if on { [0.18, 0.22, 0.32, 0.95] } else { [0.1, 0.11, 0.14, 0.9] }));
            let name = match c { Channel::Color => "colour", Channel::Size => "size", Channel::Glow => "glow" };
            p.push(dash::text(r.x + 6.0, r.y + 3.0, 11.0, format!("{name}: {}", metric_name(self.metric(c))), if on { dash::FG } else { dash::DIM }));
        }
        // Review badges at each reviewed node's screen position.
        if !self.reviews.is_empty() {
            let vp = Self::viewport(area);
            let (g, pos) = self.shown();
            for n in &g.nodes {
                let (Some(ok), Some(q)) = (self.reviews.get(&n.id), pos.get(&n.id)) else { continue };
                let Some((sx, sy, _)) = self.cam.project(vp, *q) else { continue };
                let col = if *ok { [0.2, 0.75, 0.35, 1.0] } else { [0.95, 0.55, 0.1, 1.0] };
                p.push(dash::rect(sx + 6.0, sy - 12.0, 8.0, 8.0, col));
            }
        }
        // Legend.
        let kinds = [
            rce_graph3d::Kind::State,
            rce_graph3d::Kind::Event,
            rce_graph3d::Kind::Step,
            rce_graph3d::Kind::Fn,
            rce_graph3d::Kind::Type,
            rce_graph3d::Kind::Error,
        ];
        let mut lx = area.x + 12.0;
        let ly = area.y + area.h - SCRUB_H - 22.0;
        for k in kinds {
            p.push(dash::rect(lx, ly + 3.0, 9.0, 9.0, k.color()));
            p.push(dash::text(lx + 13.0, ly, 11.0, k.as_str(), dash::DIM));
            lx += 62.0;
        }
        if self.minimap {
            let (mw, mh) = (150.0, 110.0);
            let (mx, my) = (area.x + area.w - mw - 12.0, area.y + 44.0);
            p.push(dash::rect(mx, my, mw, mh, [0.1, 0.11, 0.14, 0.9]));
            let (pts, here) = rce_graph3d::lens::minimap(&self.pos, &self.cam);
            for q in pts {
                p.push(dash::rect(mx + 4.0 + q[0] * (mw - 8.0), my + 4.0 + q[1] * (mh - 8.0), 2.0, 2.0, dash::DIM));
            }
            p.push(dash::rect(mx + here[0] * (mw - 8.0), my + here[1] * (mh - 8.0), 8.0, 8.0, dash::WARN));
            let z = if self.folded() { "mini-map · zoom: files" } else { "mini-map · zoom: definitions" };
            p.push(dash::text(mx, my + mh + 3.0, 10.0, z, dash::DIM));
        }
        let legend = if self.style.divergence {
            "colour: log2(E_meas/E_sim) per node · blue: model over-predicts · grey: match · red: meter reads more · dark: no paired actions"
        } else if self.style.mapping != rce_graph3d::lens::Mapping::default() {
            "colour/size/glow: user-mapped metrics (see header)"
        } else if self.style.energy {
            "colour: joules (E_meas, else E_sim: dark core) · red pulse: |delta| > k sigma · green ring: new in version · orange: code -> hardware"
        } else {
            "colour: kind"
        };
        p.push(dash::text(lx + 8.0, ly, 11.0, legend, dash::DIM));
        // Scrubber with ticks for versions (tall), receipts (short) and facts (dots).
        let sy = area.y + area.h - SCRUB_H;
        p.push(dash::rect(area.x, sy, area.w, SCRUB_H, dash::BG));
        let (x0, x1) = (area.x + 16.0, area.x + area.w - 16.0);
        p.push(dash::rect(x0, sy + 30.0, x1 - x0, 2.0, dash::DIM));
        if let Some((a, b)) = self.tl.range() {
            let span = (b - a).max(1) as f32;
            let at = |t: i64| x0 + (t - a) as f32 / span * (x1 - x0);
            for v in &self.tl.versions {
                p.push(dash::rect(
                    at(v.t_ms) - 1.0,
                    sy + 18.0,
                    2.0,
                    26.0,
                    [0.35, 0.65, 1.0, 1.0],
                ));
                p.push(dash::text(
                    at(v.t_ms) + 3.0,
                    sy + 6.0,
                    10.0,
                    v.label.clone(),
                    dash::DIM,
                ));
            }
            for x in &self.tl.actions {
                let c = if x.synthetic { dash::WARN } else { dash::OK };
                p.push(dash::rect(at(x.t_ms) - 0.5, sy + 26.0, 1.0, 10.0, c));
            }
            for x in &self.tl.facts {
                p.push(dash::rect(
                    at(x.recorded_at) - 2.0,
                    sy + 40.0,
                    4.0,
                    4.0,
                    [0.75, 0.55, 1.0, 1.0],
                ));
            }
            let hx = x0 + self.u as f32 * (x1 - x0);
            p.push(dash::rect(hx - 2.0, sy + 14.0, 4.0, 34.0, dash::FG));
            p.push(dash::text(
                x0,
                sy + 48.0,
                10.0,
                format!(
                    "t = {} ms · {} actions · {} facts true and known · +{} / -{} items vs previous version · blue: Form versions, green: receipts, amber: synthetic, purple: facts",
                    self.snap.t_ms - a,
                    self.snap.actions,
                    self.snap.facts.values().map(|v| v.len()).sum::<usize>(),
                    self.snap.added.len(),
                    self.snap.removed.len()
                ),
                dash::DIM,
            ));
        }
        if !self.selected_info.is_empty() {
            let (cw, ch) = (330.0, 22.0 + self.selected_info.len() as f32 * 17.0);
            let (cx, cy) = (area.x + area.w - cw - 12.0, area.y + 48.0);
            p.push(dash::rect(cx, cy, cw, ch, [0.13, 0.14, 0.17, 0.94]));
            p.push(dash::rect(cx, cy, 3.0, ch, [0.9, 0.9, 1.0, 1.0]));
            for (i, l) in self.selected_info.iter().enumerate() {
                p.push(dash::text(
                    cx + 12.0,
                    cy + 10.0 + i as f32 * 17.0,
                    if i == 0 { 14.0 } else { 12.0 },
                    l.clone(),
                    if i == 0 { dash::FG } else { dash::DIM },
                ));
            }
        }
        p
    }

    /// Draw the view into `target` over `area` (2D under, 3D, 2D over).
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &self,
        r: &mut Renderer,
        r3: &mut Renderer3d,
        target: &wgpu::TextureView,
        w: u32,
        h: u32,
        area: R,
        t: f32,
        clear: Option<[f64; 4]>,
    ) {
        let under = to_scene(&self.under(area), w, h);
        match clear {
            Some(c) => r.draw(&under, target, w, h, c),
            None => r.draw_over(&under, target, w, h),
        };
        let f = self.frame(area);
        r3.draw(
            &r.device,
            &r.queue,
            target,
            w,
            h,
            Self::viewport(area),
            &self.cam,
            &f,
            t,
        );
        r.draw_over(&to_scene(&self.over(area, &f), w, h), target, w, h);
    }

    /// Offscreen: the whole `w` x `h` image is this view.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn render_rgba(
        &self,
        r: &mut Renderer,
        r3: &mut Renderer3d,
        w: u32,
        h: u32,
    ) -> Result<Vec<u8>, String> {
        let tex = r.offscreen_texture(w, h);
        let view = tex.create_view(&Default::default());
        let area = R {
            x: 0.0,
            y: 0.0,
            w: w as f32,
            h: h as f32,
        };
        self.draw(r, r3, &view, w, h, area, 0.0, Some([0.0, 0.0, 0.0, 1.0]));
        let rb = r.readback_texture(&tex, w, h);
        r.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| format!("poll: {e}"))?;
        rb.take()
    }
}

/// The telemetry dashboard as a scene.
pub fn dashboard_scene(rows: &[SourceRow], area: R, w: u32, h: u32, note: &str) -> Scene {
    to_scene(
        &dash::dashboard(rows, area.x, area.y, area.w, area.h, note),
        w,
        h,
    )
}

/// rce-graph3d 2D primitives as an rce-ide scene (paint order kept).
pub fn to_scene(p: &[Prim], w: u32, h: u32) -> Scene {
    let full = R {
        x: 0.0,
        y: 0.0,
        w: w as f32,
        h: h as f32,
    };
    let mut s = Scene::default();
    for (i, x) in p.iter().enumerate() {
        match x {
            Prim::Rect { x, y, w, h, color } => s.quads.push(Quad {
                rect: R {
                    x: *x,
                    y: *y,
                    w: *w,
                    h: *h,
                },
                color: *color,
                clip: full,
                order: i,
            }),
            Prim::Text {
                x,
                y,
                size,
                text,
                color,
            } => s.texts.push(TextRun {
                rect: R {
                    x: *x,
                    y: *y,
                    w: text.chars().count() as f32 * size * 0.7 + 8.0,
                    h: size * 1.5,
                },
                text: text.clone(),
                size: *size,
                color: *color,
                clip: full,
                order: i,
            }),
        }
    }
    s
}

/// The graph op for dropping node `from` onto node `to`: a move when they
/// live in different files (onto a folded file, into that file).
pub fn drop_op(g: &Graph, from: usize, to: usize) -> Option<crate::graph_edit::GraphOp> {
    let a = &g.nodes[from];
    let b = &g.nodes[to];
    let dest = rce_graph3d::lens::folded_file(b).map(str::to_string).or_else(|| b.file.clone())?;
    (from != to && a.file.as_deref() != Some(dest.as_str()) && !a.kind.is_hw())
        .then(|| crate::graph_edit::GraphOp::Move { name: a.label.clone(), to_file: dest })
}

/// Node id -> review still valid, from `review` receipts in the ledger.
/// The latest mark per node wins; a node whose item is gone is dropped.
pub fn review_badges(receipts: &[ide_receipt::Receipt], files: &crate::graph_edit::Files) -> std::collections::BTreeMap<String, bool> {
    let mut out = std::collections::BTreeMap::new();
    for r in receipts {
        let d = &r.body.draft;
        if d.action != ide_receipt::ActionKind::Other("review".into()) { continue }
        let (Some(node), Some(h)) = (d.detail["node"].as_str(), d.detail["item_hash"].as_str()) else { continue };
        let Some(rest) = node.strip_prefix("code:") else { continue };
        let Some((file, name)) = rest.rsplit_once(':') else { continue };
        let Some(text) = files.get(file).and_then(|src| crate::graph_edit::item_text(src, name)) else {
            out.remove(node);
            continue;
        };
        out.insert(node.to_string(), ide_receipt::review::item_hash(&text) == h);
    }
    out
}

/// Apply a graph gesture to the workspace: refused while the editor has
/// unsaved changes, and refused (nothing written) unless `rce check`
/// admits the whole program afterwards. Returns the files written.
#[cfg(not(target_arch = "wasm32"))]
pub fn apply_gesture(ide: &mut crate::vm::Ide, op: &crate::graph_edit::GraphOp) -> Result<Vec<String>, String> {
    if ide.editor.dirty {
        return Err("save the editor first: graph edits rewrite files on disk".into());
    }
    let before: crate::graph_edit::Files = ide.server.ws.program().into_iter().collect();
    let after = crate::graph_edit::apply(&before, op).map_err(|e| e.join("; "))?;
    let mut wrote = vec![];
    for (k, v) in &after {
        if before.get(k) != Some(v) {
            let p = ide.server.ws.resolve(k, true)?;
            std::fs::write(&p, v).map_err(|e| format!("{k}: {e}"))?;
            wrote.push(k.clone());
        }
    }
    let cur = ide.editor.path.clone();
    ide.open(&cur);
    ide.refresh();
    Ok(wrote)
}

/// Native: the timeline from the IDE's services. `history` holds earlier
/// Form versions `(t_ms, label, program)`; the current workspace program is
/// added as the last version. Receipts come from the IDE's ledger, facts
/// from its memory store, hardware from runtime discovery with live
/// telemetry from `rows`.
#[cfg(not(target_arch = "wasm32"))]
pub fn timeline_from_ide(
    ide: &crate::vm::Ide,
    history: &[rce_graph3d::history::Program],
    rows: &[SourceRow],
    cpu_util: Option<f64>,
) -> Timeline {
    use rce_graph3d::native;
    use rce_graph3d::time::Version;
    let mut versions: Vec<Version> = history
        .iter()
        .map(|(t, l, p)| Version {
            t_ms: *t,
            label: l.clone(),
            graph: native::code_graph_of(p),
        })
        .collect();
    let cur = ide.server.ws.program();
    let fh = rce_memory::form::form_hash(&cur);
    if versions.last().is_none_or(|v| v.graph.form_hash != fh) {
        versions.push(Version {
            t_ms: rce_memory::now_ms(),
            label: "now".into(),
            graph: native::code_graph_of(&cur),
        });
    }
    let receipts = ide_receipt::chain::read_jsonl(&ide.ledger_path).unwrap_or_default();
    let _ = &mut versions;
    let facts = ide
        .server
        .memory
        .as_ref()
        .and_then(|m| rce_memory::engine::MemoryEngine::scan(&m.engine).ok())
        .map(|f| native::facts(&f))
        .unwrap_or_default();
    let mut hardware = rce_graph3d::hw::discover();
    native::attach_live(&mut hardware, rows, cpu_util);
    Timeline {
        versions,
        facts,
        actions: native::actions(&receipts),
        hardware,
    }
}

/// Native: jump the editor to `file:line` (1-based) from a picked node.
#[cfg(not(target_arch = "wasm32"))]
pub fn jump(ide: &mut crate::vm::Ide, file: &str, line: usize) -> Result<(), String> {
    if ide.editor.path != file {
        ide.open(file);
    }
    ide.apply(crate::vm::Action::Edit(crate::buffer::EditOp::ClickAt {
        line: line.saturating_sub(1),
        col: 0,
        extend: false,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_edit::GraphOp;
    use rce_graph3d::time::Version;

    const A: &str = "fn leaf(x: i64) -> i64\n    return x + 1\n\nfn top() -> i64\n    return leaf(2)\n";
    const B: &str = "fn other() -> i64\n    return 3\n";

    fn files() -> crate::graph_edit::Files {
        [("a.rce".to_string(), A.to_string()), ("b.rce".to_string(), B.to_string())].into()
    }

    fn view() -> GraphView {
        let prog: Vec<(String, String)> = files().into_iter().collect();
        let tl = Timeline {
            versions: vec![Version { t_ms: 1, label: "v".into(), graph: rce_graph3d::native::code_graph_of(&prog) }],
            ..Default::default()
        };
        GraphView::new(tl, false)
    }

    fn idx(g: &Graph, label: &str) -> usize {
        g.nodes.iter().position(|n| n.label == label).unwrap()
    }

    #[test]
    fn metric_picker_cycles_and_hits_chips() {
        let mut v = view();
        let area = R { x: 0.0, y: 0.0, w: 800.0, h: 600.0 };
        assert_eq!(v.picker_hit(area, 20.0, 45.0), Some(Channel::Color));
        assert_eq!(v.picker_hit(area, 12.0 + CHIP_W + 6.0 + 1.0, 45.0), Some(Channel::Size));
        assert_eq!(v.picker_hit(area, 20.0, 200.0), None);
        assert_eq!(v.click(area, 20.0, 45.0), None);
        assert_eq!(v.style.mapping.color, Some(Metric::Joules));
        for _ in 0..METRICS.len() - 1 { v.cycle_metric(Channel::Color) }
        assert_eq!(v.style.mapping.color, None);
        let prims = v.over(area, &v.frame(area));
        assert!(prims.iter().any(|p| format!("{p:?}").contains("size: default")));
    }

    #[test]
    fn folded_file_expands_on_click() {
        let mut v = view();
        let area = R { x: 0.0, y: 0.0, w: 800.0, h: 600.0 };
        v.cam.dist = 1e4;
        assert!(v.folded());
        let (g, p) = v.shown();
        assert!(g.nodes.iter().all(|n| n.kind.is_hw() || rce_graph3d::lens::folded_file(n).is_some()));
        let i = g.nodes.iter().position(|n| rce_graph3d::lens::folded_file(n) == Some("a.rce")).unwrap();
        let (sx, sy, _) = v.cam.project(GraphView::viewport(area), p[&g.nodes[i].id]).unwrap();
        // Several folded nodes may overlap at this distance; pick until a.rce opens.
        let _ = v.click(area, sx, sy);
        assert_eq!(v.expanded.len(), 1);
        let opened = v.expanded.iter().next().unwrap().clone();
        let (g2, _) = v.shown();
        assert!(g2.nodes.iter().any(|n| n.file.as_deref() == Some(opened.as_str()) && rce_graph3d::lens::folded_file(n).is_none()));
        v.collapse_all();
        assert!(v.expanded.is_empty());
    }

    #[test]
    fn drop_and_delete_gestures_make_checked_ops() {
        let v = view();
        let (g, _) = v.shown();
        let op = drop_op(&g, idx(&g, "other"), idx(&g, "top")).unwrap();
        assert_eq!(op, GraphOp::Move { name: "other".into(), to_file: "a.rce".into() });
        assert_eq!(drop_op(&g, idx(&g, "top"), idx(&g, "leaf")), None);
        let after = crate::graph_edit::apply(&files(), &op).unwrap();
        assert!(after["a.rce"].contains("fn other()"));
        // Moving `top` away from the `leaf` it calls is refused by rce check.
        let bad = drop_op(&g, idx(&g, "top"), idx(&g, "other")).unwrap();
        assert!(crate::graph_edit::apply(&files(), &bad).is_err());
        let mut v = v;
        v.style.selected = Some(idx(&g, "leaf"));
        assert_eq!(v.delete_selected(), Some(GraphOp::Delete { name: "leaf".into() }));
        // Deleting a called function is refused by the checker.
        assert!(crate::graph_edit::apply(&files(), &v.delete_selected().unwrap()).is_err());
    }

    #[test]
    fn review_badges_valid_then_expired() {
        let mut l = ide_receipt::Ledger::new(ide_receipt::SigningKey::from_bytes(&[3; 32]));
        let text = crate::graph_edit::item_text(A, "leaf").unwrap();
        let (_, r) = ide_receipt::review::mark(&mut l, "code:a.rce:leaf", &text, "david", "p", 1).unwrap();
        let b = review_badges(std::slice::from_ref(&r), &files());
        assert_eq!(b.get("code:a.rce:leaf"), Some(&true));
        let mut f = files();
        f.insert("a.rce".into(), A.replace("x + 1", "x + 2"));
        assert_eq!(review_badges(std::slice::from_ref(&r), &f).get("code:a.rce:leaf"), Some(&false));
        let mut v = view();
        v.set_reviews(&[r], &files());
        let (g, _) = v.shown();
        assert!(v.describe_in(&g, idx(&g, "leaf")).iter().any(|s| s.contains("reviewed")));
        let area = R { x: 0.0, y: 0.0, w: 800.0, h: 600.0 };
        let prims = v.over(area, &v.frame(area));
        assert!(!prims.is_empty());
    }
}
