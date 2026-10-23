//! Orbit camera, projection to the screen, and picking.
use crate::graph::Graph;
use crate::layout::Positions;
use crate::math::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Orbit {
    pub target: V3,
    pub yaw: f32,
    pub pitch: f32,
    pub dist: f32,
    pub fovy: f32,
}

impl Default for Orbit {
    fn default() -> Self {
        Orbit {
            target: [0.0; 3],
            yaw: 0.6,
            pitch: 0.35,
            dist: 20.0,
            fovy: 0.9,
        }
    }
}

/// A rectangle on screen, in pixels.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Viewport {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl Orbit {
    pub fn eye(&self) -> V3 {
        let (cp, sp) = (self.pitch.cos(), self.pitch.sin());
        add(
            self.target,
            [
                self.dist * cp * self.yaw.sin(),
                self.dist * sp,
                self.dist * cp * self.yaw.cos(),
            ],
        )
    }
    pub fn view_proj(&self, aspect: f32) -> M4 {
        let v = look_at(self.eye(), self.target, [0.0, 1.0, 0.0]);
        let p = perspective(self.fovy, aspect, 0.1, 500.0);
        mul(&p, &v)
    }
    pub fn rotate(&mut self, dyaw: f32, dpitch: f32) {
        self.yaw += dyaw;
        self.pitch = (self.pitch + dpitch).clamp(-1.5, 1.5);
    }
    pub fn zoom(&mut self, f: f32) {
        self.dist = (self.dist * f).clamp(1.0, 400.0);
    }
    /// Frame all positions.
    pub fn fit(&mut self, pos: &Positions) {
        if pos.is_empty() {
            return;
        }
        let n = pos.len() as f32;
        let c = pos
            .values()
            .fold([0.0; 3], |a, p| add(a, scale(*p, 1.0 / n)));
        let r = pos.values().map(|p| len(sub(*p, c))).fold(1.0, f32::max);
        self.target = c;
        self.dist = r / (self.fovy / 2.0).tan() * 0.9;
    }
    /// World point to screen pixels and depth (0 near, 1 far); `None` behind the eye.
    pub fn project(&self, vp: Viewport, p: V3) -> Option<(f32, f32, f32)> {
        let c = apply(&self.view_proj(vp.w / vp.h), p);
        if c[3] <= 1e-4 {
            return None;
        }
        let (x, y, z) = (c[0] / c[3], c[1] / c[3], c[2] / c[3]);
        Some((
            vp.x + (x + 1.0) / 2.0 * vp.w,
            vp.y + (1.0 - y) / 2.0 * vp.h,
            z,
        ))
    }
    /// Screen radius in pixels of a sphere of world radius `r` at `p`.
    pub fn screen_radius(&self, vp: Viewport, p: V3, r: f32) -> f32 {
        let d = len(sub(p, self.eye())).max(1e-3);
        r / d / (self.fovy / 2.0).tan() * vp.h / 2.0
    }
}

/// World radius of a node (rank and energy grow it).
pub fn radius(g: &Graph, i: usize) -> f32 {
    let n = &g.nodes[i];
    let base = if n.kind.is_hw() { 0.32 } else { 0.28 };
    base + (n.rank as f32).sqrt() * 0.9
}

/// The node under the pixel `(x, y)`: of all nodes whose projected disc
/// contains the point, the nearest to the eye.
pub fn pick(
    g: &Graph,
    pos: &Positions,
    cam: &Orbit,
    vp: Viewport,
    x: f32,
    y: f32,
) -> Option<usize> {
    let mut best: Option<(f32, usize)> = None;
    for (i, n) in g.nodes.iter().enumerate() {
        let Some(p) = pos.get(&n.id) else { continue };
        let Some((sx, sy, z)) = cam.project(vp, *p) else {
            continue;
        };
        let r = cam.screen_radius(vp, *p, radius(g, i)).max(4.0);
        if (sx - x).powi(2) + (sy - y).powi(2) <= r * r && best.is_none_or(|(bz, _)| z < bz) {
            best = Some((z, i));
        }
    }
    best.map(|b| b.1)
}
