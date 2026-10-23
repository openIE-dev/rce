//! Deterministic 3D force-directed layout with a Barnes-Hut octree.
//!
//! Same graph + same options + same warm start => bit-identical positions:
//! seeds come from a hash of each node's id (not its index), iteration order
//! is fixed, and there is no randomness or threading. Warm start keeps nodes
//! that survive a version where they were and places new ones next to their
//! neighbours, so the picture is stable while the code changes.
use crate::graph::Graph;
use crate::math::V3;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    /// Free 3D force layout.
    Force,
    /// Z fixed by node kind (states, events, steps, ... in planes).
    ByKind,
    /// Z fixed by the version a node first appeared in (`first_seen`).
    ByTime,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Options {
    pub mode: Mode,
    pub iterations: u32,
    pub theta: f64,
    pub repulsion: f64,
    pub spring: f64,
    pub rest: f64,
    pub gravity: f64,
    /// Spacing between planes in the layered modes.
    pub layer_gap: f64,
    /// Hardware nodes sit in their own slab this far below the code.
    pub hw_offset: f64,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            mode: Mode::Force,
            iterations: 300,
            theta: 0.8,
            repulsion: 2.0,
            spring: 0.06,
            rest: 1.6,
            gravity: 0.02,
            layer_gap: 2.0,
            hw_offset: 9.0,
        }
    }
}

pub type Positions = BTreeMap<String, V3>;

fn hash(s: &str) -> u64 {
    // FNV-1a, then splitmix to spread bits.
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h = (h ^ (h >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    h = (h ^ (h >> 27)).wrapping_mul(0x94d049bb133111eb);
    h ^ (h >> 31)
}

fn seed(id: &str) -> [f64; 3] {
    let h = hash(id);
    let f = |s: u32| (((h >> s) & 0xFFFFF) as f64 / 0xFFFFF as f64 - 0.5) * 6.0;
    [f(0), f(21), f(42)]
}

struct Cell {
    center: [f64; 3],
    half: f64,
    mass: f64,
    com: [f64; 3],
    body: Option<usize>,
    kids: Option<Box<[Cell; 8]>>,
}

impl Cell {
    fn new(center: [f64; 3], half: f64) -> Cell {
        Cell {
            center,
            half,
            mass: 0.0,
            com: [0.0; 3],
            body: None,
            kids: None,
        }
    }
    fn octant(&self, p: [f64; 3]) -> usize {
        (p[0] >= self.center[0]) as usize
            | (((p[1] >= self.center[1]) as usize) << 1)
            | (((p[2] >= self.center[2]) as usize) << 2)
    }
    fn split(&mut self) {
        let h = self.half / 2.0;
        let c = self.center;
        let mk = |i: usize| {
            Cell::new(
                [
                    c[0] + if i & 1 != 0 { h } else { -h },
                    c[1] + if i & 2 != 0 { h } else { -h },
                    c[2] + if i & 4 != 0 { h } else { -h },
                ],
                h,
            )
        };
        self.kids = Some(Box::new([
            mk(0),
            mk(1),
            mk(2),
            mk(3),
            mk(4),
            mk(5),
            mk(6),
            mk(7),
        ]));
    }
    fn insert(&mut self, i: usize, p: &[[f64; 3]], depth: u32) {
        let q = p[i];
        for (c, v) in self.com.iter_mut().zip(q) {
            *c = (*c * self.mass + v) / (self.mass + 1.0);
        }
        self.mass += 1.0;
        if self.kids.is_none() && self.body.is_none() && self.mass == 1.0 {
            self.body = Some(i);
            return;
        }
        if depth > 40 {
            return; // coincident points: keep as aggregate mass
        }
        if self.kids.is_none() {
            self.split();
            if let Some(b) = self.body.take() {
                let o = self.octant(p[b]);
                self.kids.as_mut().unwrap()[o].insert(b, p, depth + 1);
            }
        }
        let o = self.octant(q);
        self.kids.as_mut().unwrap()[o].insert(i, p, depth + 1);
    }
    fn force(&self, i: usize, p: [f64; 3], theta: f64, k: f64, out: &mut [f64; 3]) {
        if self.mass == 0.0 || self.body == Some(i) && self.kids.is_none() {
            return;
        }
        let d = [p[0] - self.com[0], p[1] - self.com[1], p[2] - self.com[2]];
        let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + 0.01;
        let r = r2.sqrt();
        if self.kids.is_none() || (2.0 * self.half) / r < theta {
            let f = k * self.mass / r2;
            for j in 0..3 {
                out[j] += d[j] / r * f;
            }
            return;
        }
        for c in self.kids.as_ref().unwrap().iter() {
            c.force(i, p, theta, k, out);
        }
    }
}

/// Lay out `g`. `warm` holds positions from the previous version by node
/// id; `first_seen` (by id) feeds [`Mode::ByTime`].
pub fn layout(
    g: &Graph,
    opts: &Options,
    warm: &Positions,
    first_seen: &BTreeMap<String, u32>,
) -> Positions {
    let n = g.nodes.len();
    let mut adj: Vec<Vec<usize>> = vec![vec![]; n];
    for e in &g.edges {
        if e.a < n && e.b < n && e.a != e.b {
            adj[e.a].push(e.b);
            adj[e.b].push(e.a);
        }
    }
    let mut p: Vec<[f64; 3]> = vec![[0.0; 3]; n];
    let mut fixed = vec![false; n];
    for i in 0..n {
        if let Some(w) = warm.get(&g.nodes[i].id) {
            p[i] = [w[0] as f64, w[1] as f64, w[2] as f64];
            fixed[i] = true;
        }
    }
    // New nodes: next to their already-placed neighbours, else seeded.
    for i in 0..n {
        if fixed[i] {
            continue;
        }
        let s = seed(&g.nodes[i].id);
        let placed: Vec<usize> = adj[i].iter().copied().filter(|&j| fixed[j]).collect();
        p[i] = if placed.is_empty() {
            s
        } else {
            let mut c = [0.0; 3];
            for &j in &placed {
                for k in 0..3 {
                    c[k] += p[j][k] / placed.len() as f64;
                }
            }
            [c[0] + s[0] * 0.2, c[1] + s[1] * 0.2, c[2] + s[2] * 0.2]
        };
    }
    let target_z = |i: usize| -> Option<f64> {
        let nd = &g.nodes[i];
        match opts.mode {
            Mode::Force => None,
            Mode::ByKind => Some((nd.kind.layer() as f64 - 3.0) * opts.layer_gap),
            Mode::ByTime => first_seen.get(&nd.id).map(|v| *v as f64 * opts.layer_gap),
        }
    };
    let hw_y = |i: usize| g.nodes[i].kind.is_hw();
    let any_code = g.nodes.iter().any(|n| !n.kind.is_hw());
    let any_hw = g.nodes.iter().any(|n| n.kind.is_hw());
    // Warm-started runs need fewer steps; cooling keeps it convergent.
    let iters = if warm.is_empty() {
        opts.iterations
    } else {
        opts.iterations / 3
    };
    for it in 0..iters {
        let temp = 0.5 * (1.0 - it as f64 / iters.max(1) as f64) + 0.02;
        let mut lo = [f64::MAX; 3];
        let mut hi = [f64::MIN; 3];
        for q in &p {
            for k in 0..3 {
                lo[k] = lo[k].min(q[k]);
                hi[k] = hi[k].max(q[k]);
            }
        }
        let c = [
            (lo[0] + hi[0]) / 2.0,
            (lo[1] + hi[1]) / 2.0,
            (lo[2] + hi[2]) / 2.0,
        ];
        let half = (0..3).map(|k| hi[k] - lo[k]).fold(1.0, f64::max) / 2.0 + 1e-3;
        let mut root = Cell::new(c, half);
        for i in 0..n {
            root.insert(i, &p, 0);
        }
        let mut f = vec![[0.0f64; 3]; n];
        for i in 0..n {
            root.force(i, p[i], opts.theta, opts.repulsion, &mut f[i]);
        }
        for e in &g.edges {
            let (a, b) = (e.a, e.b);
            if a >= n || b >= n || a == b {
                continue;
            }
            let d = [p[b][0] - p[a][0], p[b][1] - p[a][1], p[b][2] - p[a][2]];
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-6);
            let s = opts.spring * (r - opts.rest);
            for k in 0..3 {
                f[a][k] += d[k] / r * s;
                f[b][k] -= d[k] / r * s;
            }
        }
        for i in 0..n {
            let hwn = hw_y(i) && any_code;
            let cy = if hwn { -opts.hw_offset } else { 0.0 };
            f[i][0] -= p[i][0] * opts.gravity;
            f[i][1] -= (p[i][1] - cy) * opts.gravity * if hwn { 6.0 } else { 1.0 };
            f[i][2] -= p[i][2] * opts.gravity;
            let m = (f[i][0] * f[i][0] + f[i][1] * f[i][1] + f[i][2] * f[i][2]).sqrt();
            let lim = if m > temp { temp / m } else { 1.0 };
            let damp = if fixed[i] { 0.3 } else { 1.0 };
            for k in 0..3 {
                p[i][k] += f[i][k] * lim * damp;
            }
            if let Some(z) = target_z(i) {
                p[i][2] = z;
            }
            // Hardware and code keep to their own slabs.
            if any_code {
                if hwn {
                    p[i][1] = p[i][1].min(-opts.hw_offset + 1.5);
                } else if any_hw {
                    p[i][1] = p[i][1].max(-opts.hw_offset * 0.4);
                }
            }
        }
    }
    g.nodes
        .iter()
        .zip(p)
        .map(|(nd, q)| (nd.id.clone(), [q[0] as f32, q[1] as f32, q[2] as f32]))
        .collect()
}
