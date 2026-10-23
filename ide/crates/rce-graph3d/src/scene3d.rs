//! What the GPU draws for one frame: node and edge instances, plus the
//! labels the host draws on top. Pure; tested headless.
use crate::camera::{Orbit, Viewport, radius};
use crate::graph::{EdgeKind, Graph};
use crate::layout::Positions;
use crate::overlay::heat;
use std::collections::BTreeSet;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "gpu", derive(bytemuck::Pod, bytemuck::Zeroable))]
pub struct NodeInst {
    pub pos: [f32; 3],
    pub radius: f32,
    pub color: [f32; 4],
    /// x: glow 0..1, y: estimate-only ring (E_sim), z: flash (|delta| > k sigma), w: highlight ring.
    pub flags: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "gpu", derive(bytemuck::Pod, bytemuck::Zeroable))]
pub struct EdgeInst {
    pub a: [f32; 3],
    pub width: f32,
    pub b: [f32; 3],
    pub _pad: f32,
    pub color: [f32; 4],
}

#[derive(Debug, Clone, PartialEq)]
pub struct Label {
    pub x: f32,
    pub y: f32,
    pub depth: f32,
    pub text: String,
    pub color: [f32; 4],
    pub node: usize,
}

#[derive(Debug, Clone, Default)]
pub struct Style {
    /// Colour by energy (else by kind).
    pub energy: bool,
    /// Colour by E_meas vs E_sim divergence (overrides `energy` colour on
    /// nodes that have paired actions).
    pub divergence: bool,
    /// User-chosen metric per channel (CodeCharta-style); applied last.
    pub mapping: crate::lens::Mapping,
    pub selected: Option<usize>,
    /// Highlighted by the editor cursor.
    pub cursor: Option<usize>,
    /// Ids new in this version (time scrubbing).
    pub added: BTreeSet<String>,
    /// Label at most this many nodes (largest first).
    pub max_labels: usize,
}

#[derive(Debug, Clone, Default)]
pub struct Frame {
    pub nodes: Vec<NodeInst>,
    pub edges: Vec<EdgeInst>,
    pub labels: Vec<Label>,
}

pub fn build(g: &Graph, pos: &Positions, cam: &Orbit, vp: Viewport, st: &Style) -> Frame {
    let mut f = Frame::default();
    let max_j = g
        .nodes
        .iter()
        .filter_map(|n| n.energy.joules)
        .fold(0.0, f64::max);
    let max_w = g
        .nodes
        .iter()
        .filter_map(|n| n.live.as_ref().and_then(|l| l.watts))
        .fold(0.0, f64::max);
    for e in &g.edges {
        let (Some(a), Some(b)) = (pos.get(&g.nodes[e.a].id), pos.get(&g.nodes[e.b].id)) else {
            continue;
        };
        let (color, width) = match e.kind {
            EdgeKind::Ran => (
                [1.0, 0.55, 0.15, 0.9],
                2.0 + (1.0 + e.weight as f32).log2() * 2.0,
            ),
            EdgeKind::Contains => ([0.45, 0.48, 0.55, 0.55], 1.2),
            EdgeKind::Near => ([0.3, 0.95, 0.95, 0.55], 1.0 + 2.0 * e.weight.max(0.0) as f32),
            EdgeKind::Link => ([0.55, 0.75, 0.95, 0.8], 2.5),
            EdgeKind::Writes => (
                [0.45, 0.9, 0.55, 0.8],
                1.5 + e.weight.log2().max(0.0) as f32 * 1.5,
            ),
            EdgeKind::Triggers => (
                [1.0, 0.78, 0.3, 0.8],
                1.5 + e.weight.log2().max(0.0) as f32 * 1.5,
            ),
            EdgeKind::Raises | EdgeKind::Refuses => ([1.0, 0.4, 0.4, 0.8], 1.5),
            EdgeKind::Call => (
                [0.75, 0.55, 1.0, 0.8],
                1.5 + e.weight.log2().max(0.0) as f32 * 1.5,
            ),
            EdgeKind::Ref => (
                [0.6, 0.62, 0.7, 0.55],
                1.0 + e.weight.log2().max(0.0) as f32,
            ),
        };
        f.edges.push(EdgeInst {
            a: *a,
            width: width.min(10.0),
            b: *b,
            _pad: 0.0,
            color,
        });
    }
    let msize = st.mapping.size.map(|m| crate::lens::normalised(g, m));
    let mcolor = st.mapping.color.map(|m| crate::lens::normalised(g, m));
    let mglow = st.mapping.glow.map(|m| crate::lens::normalised(g, m));
    let mut lab = vec![];
    for (i, n) in g.nodes.iter().enumerate() {
        let Some(p) = pos.get(&n.id) else { continue };
        let mut color = n.kind.color();
        let mut flags = [0.0f32; 4];
        if st.energy {
            if let Some(j) = n.energy.joules {
                color = heat(j, max_j);
                flags[0] = 0.4 + 0.6 * (j / max_j.max(1e-12)) as f32;
                flags[1] = (n.energy.measured_j == 0.0) as u8 as f32;
            } else if let Some(w) = n.live.as_ref().and_then(|l| l.watts) {
                color = heat(w, max_w);
                flags[0] = 0.4 + 0.6 * (w / max_w.max(1e-12)) as f32;
            } else if !n.kind.is_hw() {
                color = [color[0] * 0.45, color[1] * 0.45, color[2] * 0.45, 1.0];
            }
            flags[2] = n.energy.anomaly as u8 as f32;
        }
        if st.divergence {
            match n.energy.divergence() {
                Some(d) => {
                    color = crate::overlay::diverge_color(d);
                    flags[0] = (d.abs() / 2.0).min(1.0) as f32;
                }
                None if !n.kind.is_hw() => color = [0.25, 0.25, 0.28, 1.0],
                None => {}
            }
        }
        if n.refused {
            flags[2] = flags[2].max(0.5);
        }
        if st.selected == Some(i) || st.cursor == Some(i) {
            flags[3] = 1.0;
        } else if st.added.contains(&n.id) {
            flags[3] = 0.5;
        }
        let mut r = radius(g, i);
        if let Some(v) = &msize {
            r = 0.15 + 0.85 * v[i].unwrap_or(0.0) * 1.5;
        }
        if let Some(v) = &mcolor {
            color = match v[i] {
                Some(t) => heat(t as f64, 1.0),
                None => [0.25, 0.25, 0.28, 1.0],
            };
        }
        if let Some(v) = &mglow {
            flags[0] = v[i].unwrap_or(0.0);
        }
        f.nodes.push(NodeInst {
            pos: *p,
            radius: r,
            color,
            flags,
        });
        if let Some((x, y, z)) = cam.project(vp, *p)
            && x >= vp.x
            && x <= vp.x + vp.w
            && y >= vp.y
            && y <= vp.y + vp.h
        {
            let mut t = n.label.clone();
            if st.energy {
                if let Some(j) = n.energy.joules {
                    let tag = if n.energy.synthetic { "demo " } else { "" };
                    let src = if n.energy.measured_j > 0.0 {
                        "E_meas"
                    } else {
                        "E_sim"
                    };
                    t = format!("{t}  {j:.2} J {tag}{src}");
                } else if let Some(l) = &n.live {
                    t = match l.watts {
                        Some(w) => {
                            format!("{t}  {w:.1} W {}", l.source.clone().unwrap_or_default())
                        }
                        None => format!("{t}  [{}]", crate::graph::trunc(&l.status, 28)),
                    };
                }
            }
            if st.divergence
                && let Some(d) = n.energy.divergence()
            {
                let tag = if n.energy.synthetic { " SYNTHETIC" } else { "" };
                t = format!("{}  E_meas/E_sim x{:.2}{tag}", n.label, d.exp2());
            }
            let pri = r as f64
                + n.energy.joules.unwrap_or(0.0) / max_j.max(1e-12)
                + (flags[3] as f64) * 10.0;
            let rr = cam.screen_radius(vp, *p, r);
            lab.push((
                pri,
                Label {
                    x: x + rr + 3.0,
                    y: y - 7.0,
                    depth: z,
                    text: t,
                    color: [0.9, 0.92, 0.96, 1.0],
                    node: i,
                },
            ));
        }
    }
    lab.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.node.cmp(&b.1.node)));
    f.labels = lab.into_iter().take(st.max_labels).map(|x| x.1).collect();
    f
}
